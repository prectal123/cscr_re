"""Isolates the floor's effect from the total-budget effect: reproduces the
EXACT scale that was solved for TARGET_TOTAL=1200, MIN_PROBES=6 (the first
attempt that failed on all-seen), then RAISES ONLY THE FLOOR to 15 WITHOUT
re-solving scale for a new target -- so every category that was already
above the old floor (most of the top/mid categories) keeps EXACTLY the same
probe count as the original 1200-probe run, and only the categories that
were sitting at the floor get bumped up. This separates "did raising the
floor help" from "did more probes everywhere help" (the previous 1800-probe
test changed both at once).

The resulting total will land somewhere between 1200 and 1800 depending on
how many categories were floor-bound.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
OUT_DIR = Path("local_descriptors/embedllm-ceiling-pcaweighted-floorablation-pca5")
K_PCA = 5
ORIGINAL_TARGET = 1200
ORIGINAL_MIN = 6
ORIGINAL_MAX = 60
NEW_MIN = 15  # only this changes
UNIFORM_BASELINE_TOTAL = 80 * 24


def load_importance():
    d = json.load(open(ANALYSIS_DIR / "pca_category_importance.json", encoding="utf-8"))
    cats = d["ranked_categories"]
    imp = np.array(d["ranked_importance"])
    return {c: v for c, v in zip(cats, imp)}


def solve_scale(importance_by_cat, categories, target_total, min_n, max_n):
    imp = np.array([importance_by_cat[c] for c in categories])
    sqrt_imp = np.sqrt(imp)

    def total_for_scale(scale):
        n = np.clip(np.round(sqrt_imp * scale), min_n, max_n)
        return n.sum()

    lo, hi = 0.0, 1e6
    for _ in range(60):
        mid = (lo + hi) / 2
        if total_for_scale(mid) < target_total:
            lo = mid
        else:
            hi = mid
    return hi


def apply_scale(importance_by_cat, categories, scale, min_n, max_n):
    imp = np.array([importance_by_cat[c] for c in categories])
    sqrt_imp = np.sqrt(imp)
    n = np.clip(np.round(sqrt_imp * scale), min_n, max_n)
    return {c: int(v) for c, v in zip(categories, n)}


def build_probe_sampled_pca5_weighted(df, allocation, models, categories, out_dir):
    per_prompt = df.groupby("prompt_id").agg(category=("category", "first"), var=("label", "var")).reset_index()
    selected_ids = set()
    for cat, grp in per_prompt.groupby("category"):
        n = allocation.get(cat, NEW_MIN)
        top = grp.nlargest(n, "var")
        selected_ids.update(top["prompt_id"].tolist())
    sub = df[df["prompt_id"].isin(selected_ids)]
    print(f"  {len(selected_ids)} probes selected total ({len(categories)} categories)", flush=True)

    pivot = sub.pivot_table(index="model_name", columns="category", values="label", aggfunc="mean")
    pivot = pivot.reindex(index=models, columns=categories)
    raw = pivot.to_numpy()
    col_mean = np.nanmean(raw, axis=0, keepdims=True)
    raw = np.where(np.isnan(raw), col_mean, raw)
    pool_mean = raw.mean(axis=0, keepdims=True)
    centered = raw - pool_mean

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()
    reduced = centered @ Vt[:K_PCA].T
    E = reduced / (np.linalg.norm(reduced, axis=1, keepdims=True) + 1e-12)

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(models):
        np.save(out_dir / f"{m}.npy", E[i].astype(np.float32))
    print(f"  saved -> {out_dir} (top-{K_PCA} cum. explained var={np.cumsum(explained)[K_PCA-1]:.4f})", flush=True)
    return len(selected_ids)


def main():
    print("Loading EmbedLLM train.csv...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "category", "label", "prompt"])
    models = sorted(df["model_name"].unique())
    categories = sorted(df["category"].unique())

    importance_by_cat = load_importance()
    scale = solve_scale(importance_by_cat, categories, ORIGINAL_TARGET, ORIGINAL_MIN, ORIGINAL_MAX)
    print(f"Recovered scale from original 1200/floor6 run: {scale:.4f}", flush=True)

    old_allocation = apply_scale(importance_by_cat, categories, scale, ORIGINAL_MIN, ORIGINAL_MAX)
    new_allocation = apply_scale(importance_by_cat, categories, scale, NEW_MIN, ORIGINAL_MAX)

    n_changed = sum(1 for c in categories if old_allocation[c] != new_allocation[c])
    old_total = sum(old_allocation.values())
    new_total = sum(new_allocation.values())
    print(f"\nSame scale, floor {ORIGINAL_MIN}->{NEW_MIN}: {n_changed}/80 categories changed "
          f"(all bumped UP from floor, top categories untouched)", flush=True)
    print(f"Total: {old_total} -> {new_total} "
          f"(vs uniform baseline {UNIFORM_BASELINE_TOTAL} -> {100*(1-new_total/UNIFORM_BASELINE_TOTAL):.1f}% reduction)", flush=True)

    sorted_cats = sorted(categories, key=lambda c: -importance_by_cat[c])
    print("\ntop 5 categories (should be UNCHANGED from 1200-run):")
    for c in sorted_cats[:5]:
        print(f"  {c:<40s} old={old_allocation[c]:>3d}  new={new_allocation[c]:>3d}")
    print("bottom 5 categories (should be bumped to new floor):")
    for c in sorted_cats[-5:]:
        print(f"  {c:<40s} old={old_allocation[c]:>3d}  new={new_allocation[c]:>3d}")

    n_probes = build_probe_sampled_pca5_weighted(df, new_allocation, models, categories, OUT_DIR)

    out_path = ANALYSIS_DIR / "pca_weighted_probe_allocation_floorablation.json"
    json.dump({"allocation": new_allocation, "total_probes": n_probes, "scale": scale,
               "old_total_1200run": old_total, "n_categories_changed": n_changed},
              open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved allocation -> {out_path}")
    print(f"Saved FP -> {OUT_DIR}")


if __name__ == "__main__":
    main()
