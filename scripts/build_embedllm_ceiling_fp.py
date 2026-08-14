"""Build and save the EmbedLLM Ceiling FP (112 models x 80 categories,
mean-centered + L2-normalized, same recipe as every other benchmark's
Ceiling FP), plus a PCA-distilled 5-dim variant, then run a GROUP-holdout
unseen-recovery test for both on the held-out Set B (test.csv).

Group holdout (not full 112-fold LOO): a fixed UNSEEN_FRACTION of the 112
models is set aside as "unseen". Every unseen model's kNN-recovery proxy is
computed using ONLY the seen models as neighbors (never other unseen
models, avoiding unseen-unseen leakage) -- this both matches the realistic
"one new model shows up against an established pool" scenario and, more
importantly, keeps the later MLP/query-encoder step to a handful of
training runs (one per seed) instead of 112 full retrains.

Reuses the exact centered-matrix construction from embedllm_scale_analysis.py
(train.csv = Set A).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy import stats
from scipy.stats import spearmanr

FULL_DIR = Path("local_descriptors/embedllm-ceiling")
PCA5_DIR = Path("local_descriptors/embedllm-ceiling-pca5")
OUT_DIR = Path("local_descriptors/embedllm-analysis")
K = 5
UNSEEN_FRACTION = 0.2
GROUP_SEEDS = [0, 1, 2]


def load_data():
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    f_test = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="test.csv")
    set_a = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "category", "label"])
    set_b = pd.read_csv(f_test, usecols=["prompt_id", "model_name", "category", "label"])
    return set_a, set_b


def build_centered_matrix(set_a, models, categories):
    pivot = set_a.pivot_table(index="model_name", columns="category", values="label", aggfunc="mean")
    pivot = pivot.reindex(index=models, columns=categories)
    raw = pivot.to_numpy()
    col_mean = np.nanmean(raw, axis=0, keepdims=True)
    raw = np.where(np.isnan(raw), col_mean, raw)
    pool_mean = raw.mean(axis=0, keepdims=True)
    return raw - pool_mean


def build_setB_matrix(set_b, models, categories):
    pivot = set_b.pivot_table(index="prompt_id", columns="model_name", values="label", aggfunc="mean")
    pivot = pivot.reindex(columns=models).dropna(axis=0, how="any")
    return pivot.to_numpy()


def group_knn_test(true_scores, E, models, seed):
    """Hold out UNSEEN_FRACTION of models as a group. Each unseen model's
    proxy is a similarity-weighted average over the SEEN models only (never
    other unseen models)."""
    rng = np.random.RandomState(seed)
    n = len(models)
    n_unseen = max(1, int(round(n * UNSEEN_FRACTION)))
    perm = rng.permutation(n)
    unseen_idx = sorted(perm[:n_unseen].tolist())
    seen_idx = sorted(perm[n_unseen:].tolist())

    sim_full = E @ E.T
    fp_rhos, uniform_rhos = [], []
    for i in unseen_idx:
        other_scores = true_scores[:, seen_idx]
        true_m = true_scores[:, i]
        sims = sim_full[i, seen_idx]
        w = np.clip(sims, 0, None)
        if w.sum() < 1e-9:
            w = np.ones_like(w)
        w = w / w.sum()
        fp_proxy = other_scores @ w
        uniform_proxy = other_scores.mean(axis=1)
        fp_rho, _ = spearmanr(fp_proxy, true_m)
        uni_rho, _ = spearmanr(uniform_proxy, true_m)
        fp_rhos.append(fp_rho)
        uniform_rhos.append(uni_rho)
    return np.array(fp_rhos), np.array(uniform_rhos), [models[i] for i in unseen_idx], [models[i] for i in seen_idx]


def save_fp(E, models, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(models):
        np.save(out_dir / f"{m}.npy", E[i].astype(np.float32))


def report(name, seed, fp_rhos, uniform_rhos, n_unseen):
    delta = fp_rhos - uniform_rhos
    t, p = stats.ttest_rel(fp_rhos, uniform_rhos)
    n_imp = int((delta > 0).sum())
    print(f"[{name} seed={seed}] mean_fp_rho={fp_rhos.mean():.4f}  mean_uniform_rho={uniform_rhos.mean():.4f}  "
          f"delta={delta.mean():+.4f}  p={p:.4g}  improved={n_imp}/{n_unseen}")
    return {"seed": seed, "mean_fp_rho": float(fp_rhos.mean()), "mean_uniform_rho": float(uniform_rhos.mean()),
            "mean_delta": float(delta.mean()), "p_value": float(p), "n_improved": n_imp, "n_unseen": n_unseen,
            "fp_rho": fp_rhos.tolist(), "uniform_rho": uniform_rhos.tolist()}


def main():
    print("Loading EmbedLLM train/test...", flush=True)
    set_a, set_b = load_data()
    models = sorted(set_a["model_name"].unique())
    categories = sorted(set_a["category"].unique())
    print(f"{len(models)} models, {len(categories)} categories", flush=True)

    C = build_centered_matrix(set_a, models, categories)  # (112, 80)

    # --- full 80-dim Ceiling FP ---
    E_full = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    save_fp(E_full, models, FULL_DIR)
    print(f"Saved full {C.shape[1]}-dim Ceiling FP -> {FULL_DIR}", flush=True)

    # --- PCA k=5 distilled FP ---
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()
    reduced = C @ Vt[:K].T
    E_pca5 = reduced / (np.linalg.norm(reduced, axis=1, keepdims=True) + 1e-12)
    save_fp(E_pca5, models, PCA5_DIR)
    print(f"Saved PCA k={K} Ceiling FP -> {PCA5_DIR}  (cum. explained var={np.cumsum(explained)[K-1]:.4f})", flush=True)

    # --- group-holdout kNN test on held-out Set B, for both FPs, across a few seeds ---
    setB_true = build_setB_matrix(set_b, models, categories)
    print(f"\nSet B usable rows (all {len(models)} models present): {setB_true.shape[0]}", flush=True)

    results = {"unseen_fraction": UNSEEN_FRACTION, "seeds": GROUP_SEEDS, "full_80dim": [], "pca_5dim": []}
    groups_by_seed = {}
    for seed in GROUP_SEEDS:
        fp_rhos_f, uni_rhos_f, unseen_names, seen_names = group_knn_test(setB_true, E_full.astype(np.float32), models, seed)
        results["full_80dim"].append(report("full 80-dim", seed, fp_rhos_f, uni_rhos_f, len(unseen_names)))

        fp_rhos_p, uni_rhos_p, unseen_names_p, seen_names_p = group_knn_test(setB_true, E_pca5.astype(np.float32), models, seed)
        results["pca_5dim"].append(report(f"PCA k={K}", seed, fp_rhos_p, uni_rhos_p, len(unseen_names_p)))

        groups_by_seed[seed] = {"unseen": unseen_names, "seen": seen_names}

    results["pca_explained_variance_at_k5"] = float(np.cumsum(explained)[K-1])
    results["groups_by_seed"] = groups_by_seed

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "embedllm_ceiling_fp_knn_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")
    print("\n(these seen/unseen splits are saved so the later MLP/query-encoder step can reuse the same groups)")


if __name__ == "__main__":
    main()
