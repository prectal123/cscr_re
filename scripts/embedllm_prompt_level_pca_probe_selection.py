"""Skips the category-level pivot entirely: instead of (a) computing
category-level PCA importance, then (b) allocating a per-category probe
BUDGET and picking top-variance prompts WITHIN each category (today's
earlier approach, which forces every probe through an 80-category
intermediate representation), this does PCA directly on the (112 models x
29673 prompts) matrix -- every individual prompt gets its own loading on
each of the top-5 PCs, no category grouping involved at all.

For each of PC1-PC5, select the top N_PER_DIM=200 prompts by |loading|
magnitude on that specific dimension, then take the union (deduplicated --
a prompt that loads heavily on multiple PCs only counts once) as the final
probe set. This directly targets "which prompts best reconstruct the 5
retained dimensions" rather than "which prompts are informative within
their category" -- the category structure never enters the selection at
all, only at the very end when building the (model x category) matrix from
the selected probes (still needed because the actual training-time FP
recipe is a category-mean matrix -> PCA, matching V2/V1's construction, so
the probe-sampled result stays comparable).

Density check confirmed: every single one of the 29,673 Set A prompts has
responses from all 112 models (fully dense matrix, no missing-data
handling needed for the prompt-level SVD).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
OUT_DIR = Path("local_descriptors/embedllm-ceiling-promptpca-pca5")
K_PCA = 5
N_PER_DIM = 200


def main():
    print("Loading EmbedLLM train.csv...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "category", "label"])
    models = sorted(df["model_name"].unique())
    categories = sorted(df["category"].unique())
    prompt_ids = sorted(df["prompt_id"].unique())
    print(f"{len(models)} models, {len(categories)} categories, {len(prompt_ids)} prompts", flush=True)

    print("Building (model x prompt) matrix (this is the big one, ~112 x 29673)...", flush=True)
    pivot = df.pivot_table(index="model_name", columns="prompt_id", values="label", aggfunc="mean")
    pivot = pivot.reindex(index=models, columns=prompt_ids)
    raw = pivot.to_numpy()
    n_missing = np.isnan(raw).sum()
    print(f"  missing entries: {n_missing} / {raw.size} ({100*n_missing/raw.size:.2f}%)", flush=True)
    col_mean = np.nanmean(raw, axis=0, keepdims=True)
    raw = np.where(np.isnan(raw), col_mean, raw)
    pool_mean = raw.mean(axis=0, keepdims=True)
    centered = raw - pool_mean  # (112, 29673)

    print("Running SVD directly on the prompt-level matrix...", flush=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()
    print(f"Explained variance by PC (prompt-level): {[f'{e:.4f}' for e in explained[:K_PCA]]}", flush=True)
    print(f"Cumulative (top-5): {explained[:K_PCA].sum():.4f}", flush=True)

    loadings = Vt[:K_PCA]  # (5, 29673)
    selected_ids = set()
    per_dim_selected = []
    for k in range(K_PCA):
        order = np.argsort(-np.abs(loadings[k]))
        top_idx = order[:N_PER_DIM]
        top_prompt_ids = [prompt_ids[i] for i in top_idx]
        per_dim_selected.append(top_prompt_ids)
        n_new = len(set(top_prompt_ids) - selected_ids)
        selected_ids.update(top_prompt_ids)
        print(f"  PC{k+1}: top-{N_PER_DIM} selected, {n_new} new (not already selected by earlier PCs)", flush=True)

    print(f"\nFinal probe set: {len(selected_ids)} unique prompts "
          f"(vs {N_PER_DIM * K_PCA} = {N_PER_DIM}*{K_PCA} before dedup, "
          f"vs uniform V1's 1920, vs Set A's full {len(prompt_ids)})", flush=True)

    # category distribution of the selected probes, just to see how it compares
    # to the category-level approach (informational, not used for selection)
    sub = df[df["prompt_id"].isin(selected_ids)]
    cat_counts = sub.drop_duplicates("prompt_id")["category"].value_counts()
    print(f"\nSelected probes span {cat_counts.shape[0]}/{len(categories)} categories "
          f"(min={cat_counts.min()}, max={cat_counts.max()}, median={cat_counts.median():.0f} per category)", flush=True)

    # Build the actual probe-sampled Ceiling FP: category-mean matrix from
    # ONLY the selected prompts -> PCA -> 5-dim (same recipe as V1/V1.5, for
    # direct comparability)
    print("\nBuilding probe-sampled Ceiling FP from selected prompts...", flush=True)
    pivot2 = sub.pivot_table(index="model_name", columns="category", values="label", aggfunc="mean")
    pivot2 = pivot2.reindex(index=models, columns=categories)
    raw2 = pivot2.to_numpy()
    col_mean2 = np.nanmean(raw2, axis=0, keepdims=True)
    raw2 = np.where(np.isnan(raw2), col_mean2, raw2)
    pool_mean2 = raw2.mean(axis=0, keepdims=True)
    centered2 = raw2 - pool_mean2

    U2, S2, Vt2 = np.linalg.svd(centered2, full_matrices=False)
    explained2 = (S2 ** 2) / (S2 ** 2).sum()
    reduced = centered2 @ Vt2[:K_PCA].T
    E = reduced / (np.linalg.norm(reduced, axis=1, keepdims=True) + 1e-12)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(models):
        np.save(OUT_DIR / f"{m}.npy", E[i].astype(np.float32))
    print(f"  saved -> {OUT_DIR} (top-{K_PCA} cum. explained var of the probe-sampled category matrix"
          f"={np.cumsum(explained2)[K_PCA-1]:.4f})", flush=True)

    out_path = ANALYSIS_DIR / "prompt_level_pca_probe_selection.json"
    json.dump({
        "n_per_dim": N_PER_DIM,
        "total_unique_probes": len(selected_ids),
        "explained_variance_prompt_level": explained[:K_PCA].tolist(),
        "n_categories_covered": int(cat_counts.shape[0]),
    }, open(out_path, "w", encoding="utf-8"), indent=2)
    print(f"\nSaved selection info -> {out_path}")
    print(f"Saved FP -> {OUT_DIR}")


if __name__ == "__main__":
    main()
