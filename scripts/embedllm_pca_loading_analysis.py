"""Step 1 of the PCA-loading-informed probe allocation idea: compute, for
each of EmbedLLM's 80 categories, how much it contributes to the top-5
principal components of the full-data (V2) Ceiling FP -- using the FULL
Set A data as an oracle/calibration source (this step is offline, one-time,
uses data we already have; it does NOT need to be repeated for future
probe-based FP construction).

Importance score per category = sum over the 5 retained PCs of
  (explained_variance_ratio[k] * loading[k, category]^2)
i.e. a variance-weighted sum of squared loadings -- a category that loads
heavily on PC1 (which explains ~87.8% of variance per prior project
findings) counts far more than one that only loads on PC5.

Output: ranked category importance table + a proposed probe budget curve
(how many probes needed to capture successive % of total importance mass),
to inform tonight's/today's redesigned probe allocation before touching
any GPU training.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
K_PCA = 5


def build_centered_matrix(set_a, models, categories):
    pivot = set_a.pivot_table(index="model_name", columns="category", values="label", aggfunc="mean")
    pivot = pivot.reindex(index=models, columns=categories)
    raw = pivot.to_numpy()
    col_mean = np.nanmean(raw, axis=0, keepdims=True)
    raw = np.where(np.isnan(raw), col_mean, raw)
    pool_mean = raw.mean(axis=0, keepdims=True)
    return raw - pool_mean


def main():
    print("Loading EmbedLLM train.csv (Set A)...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    set_a = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "category", "label"])
    models = sorted(set_a["model_name"].unique())
    categories = sorted(set_a["category"].unique())
    print(f"{len(models)} models, {len(categories)} categories", flush=True)

    C = build_centered_matrix(set_a, models, categories)  # (n_models, 80)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()
    print(f"Explained variance by PC: {[f'{e:.4f}' for e in explained[:K_PCA]]}", flush=True)
    print(f"Cumulative (top-5): {explained[:K_PCA].sum():.4f}", flush=True)

    # Vt shape: (n_components, n_categories) = (min(n_models,80), 80)
    loadings = Vt[:K_PCA]  # (5, 80)
    importance = (explained[:K_PCA, None] * (loadings ** 2)).sum(axis=0)  # (80,)
    importance = importance / importance.sum()  # normalize to sum=1

    order = np.argsort(-importance)
    ranked_categories = [categories[i] for i in order]
    ranked_importance = importance[order]

    print("\n" + "=" * 80)
    print("CATEGORY IMPORTANCE RANKING (top 20)")
    print("=" * 80)
    cum = 0.0
    for rank, (cat, imp) in enumerate(zip(ranked_categories[:20], ranked_importance[:20])):
        cum += imp
        print(f"  {rank+1:>3}. {cat:<40s} importance={imp:.4f}  cumulative={cum:.4f}")

    print("\n" + "=" * 80)
    print("BUDGET CURVE: how many categories needed to capture X% of importance mass")
    print("=" * 80)
    cum_importance = np.cumsum(ranked_importance)
    for target in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        n_needed = int(np.searchsorted(cum_importance, target) + 1)
        print(f"  {target*100:>5.0f}% of importance mass captured by top {n_needed}/{len(categories)} categories")

    out_path = ANALYSIS_DIR / "pca_category_importance.json"
    json.dump({
        "explained_variance_top5": explained[:K_PCA].tolist(),
        "ranked_categories": ranked_categories,
        "ranked_importance": ranked_importance.tolist(),
    }, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
