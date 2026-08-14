"""Verify whether CSCR's contrastive loss (InfoNCE-style softmax over
positives/negatives) is structurally less vulnerable to the outlier-drag
effect confirmed in embedllm_outlier_blend_check.py (GRPO-style linear MSE
regression: rho=0.52 between positive-set spread and how far the blended
target lands from any real model).

Theory being tested: MSE regression toward a blended target weights every
positive-advantage model LINEARLY/uniformly regardless of current distance
to the query -- an outlier far from the main cluster contributes just as
much "pull" as a nearby one, dragging the resting point into empty space.
Softmax-based contrastive losses (CSCR's cost_spectrum_info_nce and
friends) instead weight each positive by exp(sim/tau) -- EXPONENTIALLY
decaying with distance -- so once the query starts converging toward one
cluster, the softmax naturally suppresses the pull from a far-away outlier
(self-reinforcing "snap to nearest mode" instead of "average everyone").

Empirical test: for the same queries, compare two resting points --
  1. z_linear: normalize(sum_m t_m * E_m)  -- what plain MSE effectively
     targets (already computed in embedllm_outlier_blend_check.py).
  2. z_softmax: fixed-point iteration z <- normalize(sum_{m in positive}
     softmax(sim(z,E_m)/tau) * E_m), which approximates where a
     softmax/InfoNCE-style loss's positive-pull term would actually
     converge a query to (this IS the multi-positive contrastive gradient's
     fixed point, ignoring the negative-repulsion term which only pushes
     further away from non-positives, not relevant to this specific
     multi-positive-aggregation question).
Compare dist_to_nearest_real for both, and whether each nearest model is
actually IN the positive set (i.e., did it correctly "snap" to a real
positive, or land in the void between two positives).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, "src")

PCA5_DIR = Path("local_descriptors/embedllm-ceiling-pca5")
ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
SPLIT_PATH = ANALYSIS_DIR / "newllm_split.json"

N_SAMPLE = 5000
TAU = 0.1  # softmax temperature for the contrastive-style fixed point
N_ITERS = 20


def softmax_fixed_point(pos_E, tau, n_iters, z_init):
    z = z_init.copy()
    for _ in range(n_iters):
        sims = pos_E @ z  # (k,)
        w = np.exp((sims - sims.max()) / tau)
        w = w / w.sum()
        z_new = (w[:, None] * pos_E).sum(axis=0)
        norm = np.linalg.norm(z_new)
        if norm < 1e-9:
            break
        z_new = z_new / norm
        if np.linalg.norm(z_new - z) < 1e-6:
            z = z_new
            break
        z = z_new
    return z


def main():
    split = json.load(open(SPLIT_PATH, encoding="utf-8"))
    seen_models = split["seen"]
    E = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in seen_models]).astype(np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    n_models = len(seen_models)

    print("Loading EmbedLLM train.csv (cached)...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "label", "prompt"])

    name_to_idx = {n: i for i, n in enumerate(seen_models)}
    rng = np.random.RandomState(0)
    groups = list(df.groupby("prompt_id", sort=False))
    rng.shuffle(groups)

    dist_linear, dist_softmax, spreads = [], [], []
    snap_linear_in_positive, snap_softmax_in_positive = [], []
    used = 0
    for pid, grp in groups:
        if used >= N_SAMPLE:
            break
        labels = np.full(n_models, np.nan, dtype=np.float32)
        for m, v in zip(grp["model_name"], grp["label"]):
            if m in name_to_idx:
                labels[name_to_idx[m]] = float(v)
        mask = ~np.isnan(labels)
        if mask.sum() < 4:
            continue
        vals = labels[mask]
        mean, std = vals.mean(), vals.std()
        if std < 1e-6:
            continue
        t = np.zeros(n_models, dtype=np.float32)
        t[mask] = (vals - mean) / (std + 1e-6)

        pos_idx = np.where(t > 0)[0]
        if len(pos_idx) < 2:
            continue

        # z_linear: plain advantage-weighted blend (what MSE regression targets)
        z_linear = (t[:, None] * E).sum(axis=0)
        z_linear = z_linear / (np.linalg.norm(z_linear) + 1e-12)

        # z_softmax: fixed-point convergence of a softmax-weighted positive pull
        # (the multi-positive-contrastive aggregation mechanism), starting from
        # the same linear blend as the initial guess for a fair comparison.
        pos_E = E[pos_idx]
        z_softmax = softmax_fixed_point(pos_E, TAU, N_ITERS, z_linear)

        d_lin = np.linalg.norm(E - z_linear[None, :], axis=1)
        d_soft = np.linalg.norm(E - z_softmax[None, :], axis=1)

        nearest_lin = np.argmin(d_lin)
        nearest_soft = np.argmin(d_soft)

        pos_E_full = E[pos_idx]
        piu = np.triu_indices(len(pos_idx), k=1)
        pw = np.linalg.norm(pos_E_full[:, None, :] - pos_E_full[None, :, :], axis=-1)
        spread = pw[piu].mean() if len(piu[0]) > 0 else 0.0

        dist_linear.append(d_lin.min())
        dist_softmax.append(d_soft.min())
        spreads.append(spread)
        snap_linear_in_positive.append(nearest_lin in pos_idx)
        snap_softmax_in_positive.append(nearest_soft in pos_idx)
        used += 1

    dist_linear = np.array(dist_linear)
    dist_softmax = np.array(dist_softmax)
    spreads = np.array(spreads)
    snap_linear_in_positive = np.array(snap_linear_in_positive)
    snap_softmax_in_positive = np.array(snap_softmax_in_positive)

    print(f"\nUsable queries: {len(dist_linear)}")
    print(f"dist_to_nearest_real -- LINEAR (MSE-style)     : mean={dist_linear.mean():.4f}  std={dist_linear.std():.4f}")
    print(f"dist_to_nearest_real -- SOFTMAX (contrastive-style): mean={dist_softmax.mean():.4f}  std={dist_softmax.std():.4f}")
    print(f"Ratio (linear/softmax) = {dist_linear.mean()/dist_softmax.mean():.2f}x")

    diff = dist_linear - dist_softmax
    stat, p = wilcoxon(diff)
    print(f"\nPaired Wilcoxon (linear - softmax > 0): stat={stat:.1f}  p={p:.4g}")
    print(f"softmax closer than linear in {(diff > 0).mean():.1%} of queries")

    print(f"\nNearest-point lands ON an actual POSITIVE model:")
    print(f"  linear (MSE-style):      {snap_linear_in_positive.mean():.1%}")
    print(f"  softmax (contrastive-style): {snap_softmax_in_positive.mean():.1%}")

    # does the linear/softmax gap widen with positive-set spread? (mechanism check)
    rho_lin, p_lin = spearmanr(spreads, dist_linear)
    rho_soft, p_soft = spearmanr(spreads, dist_softmax)
    print(f"\nSpearman rho(spread, dist_to_nearest_real):")
    print(f"  linear:  rho={rho_lin:.4f}  p={p_lin:.4g}")
    print(f"  softmax: rho={rho_soft:.4f}  p={p_soft:.4g}")
    print("(if softmax's rho is much smaller, it confirms softmax is specifically "
          "robust to WIDE-spread positive sets, not just uniformly closer)")

    result = {
        "n_queries": len(dist_linear), "tau": TAU, "n_iters": N_ITERS,
        "dist_linear_mean": float(dist_linear.mean()), "dist_softmax_mean": float(dist_softmax.mean()),
        "wilcoxon_stat": float(stat), "wilcoxon_p": float(p),
        "pct_softmax_closer": float((diff > 0).mean()),
        "pct_linear_snaps_to_positive": float(snap_linear_in_positive.mean()),
        "pct_softmax_snaps_to_positive": float(snap_softmax_in_positive.mean()),
        "rho_spread_vs_dist_linear": float(rho_lin), "p_linear": float(p_lin),
        "rho_spread_vs_dist_softmax": float(rho_soft), "p_softmax": float(p_soft),
    }
    out_path = ANALYSIS_DIR / "outlier_drag_loss_comparison_results.json"
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
