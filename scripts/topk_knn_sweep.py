"""Training-free follow-up to routerbench_knn_test.py: instead of letting all
10 other models vote (similarity-clipped, weighted), restrict the vote to only
the top-k FP-nearest neighbors (k=1,2,3,5,7,10) and see how the FP-weighted
proxy's Spearman rho against held-out's true Set B score changes with k.

If a handful of true "beacons" close in FP space are what carry the signal,
rho should already be high (or higher than the full-10-uniform baseline) at
small k. If the signal only emerges once most/all of the pool votes, that
argues against a "few nearby beacons matter most" story and for a more
diffuse, pool-wide capability-alignment signal.
"""
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, ttest_rel

import routerbench_knn_test as rb

NAMES = rb.NAMES
MODELS = rb.MODELS
CEILING_DIR = rb.CEILING_DIR
PERP_DIR = Path("local_descriptors/routerbench-perplexity")
KS = [1, 2, 3, 5, 7, 10]


def fp_proxy_topk(sim_row, other_idx, other_scores, k):
    order = np.argsort(-sim_row[other_idx])[:k]
    topk_idx = [other_idx[j] for j in order]
    w = np.clip(sim_row[topk_idx], 0, None)
    if w.sum() < 1e-9:
        w = np.ones_like(w)
    w = w / w.sum()
    return other_scores[:, [other_idx.index(j) for j in topk_idx]] @ w


def random_k_proxy_mean_rho(true_scores, k, n_draws=30, seed=0):
    """Size-matched control: same-size (k) subset chosen at RANDOM (not by FP
    similarity), uniformly weighted, averaged over n_draws random draws per
    fold. Tests whether FP-topk's advantage is about WHICH k models it picks,
    or just about averaging over k models at all (variance reduction)."""
    rng = np.random.RandomState(seed)
    n = len(NAMES)
    per_fold_rhos = []
    for i in range(n):
        others_idx = [j for j in range(n) if j != i]
        other_scores = true_scores[:, others_idx]
        true_m = true_scores[:, i]
        kk = min(k, len(others_idx))
        draw_rhos = []
        for _ in range(n_draws):
            chosen = rng.choice(len(others_idx), size=kk, replace=False)
            proxy = other_scores[:, chosen].mean(axis=1)
            rho, _ = spearmanr(proxy, true_m)
            draw_rhos.append(rho)
        per_fold_rhos.append(float(np.mean(draw_rhos)))
    return per_fold_rhos


def sweep(set_b, desc_dir, fp_name):
    print(f"\n{'='*70}\n{fp_name} FP -- top-k neighbor sweep (+ random-k size-matched control)\n{'='*70}")
    E = np.stack([np.load(desc_dir / f"{n}.npy") for n in NAMES])
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    sim_full = E @ E.T
    true_scores = np.stack([set_b[m].to_numpy(dtype=float) for m in MODELS], axis=1)

    # full-10 uniform baseline (same for every k -- constant reference line)
    uniform_rhos = []
    for i in range(len(NAMES)):
        others_idx = [j for j in range(len(NAMES)) if j != i]
        other_scores = true_scores[:, others_idx]
        true_m = true_scores[:, i]
        uniform_proxy = other_scores.mean(axis=1)
        rho, _ = spearmanr(uniform_proxy, true_m)
        uniform_rhos.append(rho)
    uniform_mean = float(np.mean(uniform_rhos))

    results = {}
    for k in KS:
        fp_rhos = []
        for i, held_out in enumerate(NAMES):
            others_idx = [j for j in range(len(NAMES)) if j != i]
            other_scores = true_scores[:, others_idx]
            true_m = true_scores[:, i]
            kk = min(k, len(others_idx))
            proxy = fp_proxy_topk(sim_full[i], others_idx, other_scores, kk)
            rho, _ = spearmanr(proxy, true_m)
            fp_rhos.append(rho)
        fp_rhos = np.array(fp_rhos)

        random_k_rhos = np.array(random_k_proxy_mean_rho(true_scores, k))
        t_vs_uniform, p_vs_uniform = ttest_rel(fp_rhos, uniform_rhos)
        t_vs_random, p_vs_random = ttest_rel(fp_rhos, random_k_rhos)
        delta_vs_uniform = fp_rhos - np.array(uniform_rhos)
        delta_vs_random = fp_rhos - random_k_rhos
        results[k] = (float(fp_rhos.mean()), float(random_k_rhos.mean()))
        print(f"  k={k:2d}  FP-topk_rho={fp_rhos.mean():+.4f}  random-k_rho={random_k_rhos.mean():+.4f}  "
              f"delta_vs_random-k={delta_vs_random.mean():+.4f} (p={p_vs_random:.4f})  "
              f"delta_vs_uniform10={delta_vs_uniform.mean():+.4f} (p={p_vs_uniform:.4f})")

    print(f"  [reference] full-10 uniform baseline mean rho = {uniform_mean:+.4f}")
    return results, uniform_mean


def main():
    set_a, set_b = rb.load_data()
    sweep(set_b, CEILING_DIR, "Ceiling")
    sweep(set_b, PERP_DIR, "Perplexity")


if __name__ == "__main__":
    main()
