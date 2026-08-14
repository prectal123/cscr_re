"""Free, local ablation: does compressing the 22-category Ceiling FP down to
k << 22 PCA dimensions (keeping only the axes that explain real cross-model
variance) preserve -- or improve -- unseen-model kNN recovery performance,
compared to the full 22-dim FP?

Motivation: earlier PCA on domain_FP showed PC1 alone explains ~92% of
cross-model variance (dominated by generic capability), meaning most of the
22 raw category dimensions are highly redundant/noisy on top of that one
axis. If a compact, denoised k-dim FP matches or beats the full 22-dim FP on
unseen-model recovery, that's a concrete argument for scoping the FP down to
"signal-only" dimensions -- and, separately, gives the query encoder an
easier/less noisy target to hit (it currently overfits past epoch 2 on the
full 22-dim target).

No LLM judge calls, no training -- pure re-derivation from existing Set A
scores (same source as build_ceiling_fp_lite20_categoryrate.py) plus SVD.
"""
import pickle
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import spearmanr

import common_lite20 as common

DATA_DIR = Path("local_descriptors/llmrouterbench_lite20")
K_SWEEP = [1, 2, 3, 4, 5, 6, 8, 10, 14, 18, 20]


def load_setA_setB():
    with open(DATA_DIR / "setA_setB_split.pkl", "rb") as f:
        split = pickle.load(f)
    return split["setA"], split["setB"]


def build_setB_eval(setB):
    queries, scores = [], []
    for ds in common.DATASETS:
        d = setB[ds]
        queries.extend(d["queries"])
        scores.append(d["scores"])
    return queries, np.concatenate(scores, axis=0)


def build_centered_category_matrix(setA):
    """(22 categories, 20 models) matrix, pool-mean-centered per category --
    same preprocessing as build_ceiling_fp_lite20_categoryrate.py, stopping
    just before the per-model L2 normalization so PCA operates on the
    un-normalized signal."""
    raw = np.zeros((len(common.DATASETS), len(common.MODELS_20)))
    for i, ds in enumerate(common.DATASETS):
        raw[i, :] = setA[ds]["scores"].mean(axis=0)
    pool_mean = raw.mean(axis=1, keepdims=True)
    centered = raw - pool_mean  # (22, 20)
    return centered


def pca_fit(centered):
    """SVD of the (20 models, 22 categories) matrix -> principal directions
    in category-space ranked by explained cross-model variance."""
    X = centered.T  # (20, 22), rows=models
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()
    return X, Vt, explained


def build_fp_for_k(X, Vt, k):
    """Project each model's centered category-vector onto top-k PCs, then
    L2-normalize -- same final treatment as the original Ceiling FP."""
    reduced = X @ Vt[:k].T  # (20, k)
    E = reduced / (np.linalg.norm(reduced, axis=1, keepdims=True) + 1e-12)
    return E.astype(np.float32)


def knn_test(true_scores, E, pool):
    sim_full = E @ E.T
    fp_rhos, uniform_rhos = [], []
    for i in range(len(pool)):
        others_idx = [j for j in range(len(pool)) if j != i]
        other_scores = true_scores[:, others_idx]
        true_m = true_scores[:, i]
        sims = sim_full[i, others_idx]
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
    return np.array(fp_rhos), np.array(uniform_rhos)


def main():
    setA, setB = load_setA_setB()
    _, true_scores = build_setB_eval(setB)
    pool = common.MODELS_20

    centered = build_centered_category_matrix(setA)
    X, Vt, explained = pca_fit(centered)
    cum_explained = np.cumsum(explained)

    print("PCA of Ceiling FP (category-rate), 20 models x 22 categories:")
    print(f"{'k':>4s} {'explained_var_k':>15s} {'cum_explained':>14s}")
    for k in range(1, len(explained) + 1):
        print(f"{k:>4d} {explained[k-1]:>15.4f} {cum_explained[k-1]:>14.4f}")

    print(f"\n{'k (dims)':>9s} {'cum_var':>8s} {'mean FP rho':>12s} {'mean uniform':>13s} "
          f"{'delta':>9s} {'p-value':>9s} {'improved':>9s}")
    results = []
    for k in K_SWEEP:
        E = build_fp_for_k(X, Vt, k)
        fp_rhos, uniform_rhos = knn_test(true_scores, E, pool)
        delta = fp_rhos - uniform_rhos
        t, p = stats.ttest_rel(fp_rhos, uniform_rhos)
        n_improved = int((delta > 0).sum())
        print(f"{k:>9d} {cum_explained[k-1]:>8.4f} {fp_rhos.mean():>12.4f} {uniform_rhos.mean():>13.4f} "
              f"{delta.mean():>+9.4f} {p:>9.4f} {n_improved:>6d}/20")
        results.append({
            "k": k, "cum_explained_var": float(cum_explained[k-1]),
            "mean_fp_rho": float(fp_rhos.mean()), "mean_uniform_rho": float(uniform_rhos.mean()),
            "mean_delta": float(delta.mean()), "p_value": float(p), "n_improved": n_improved,
        })

    out_path = DATA_DIR / "pca_dim_ablation_results.json"
    import json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"per_pc_explained_variance": explained.tolist(), "sweep": results}, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
