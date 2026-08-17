"""Step 2: build a probe-sampled Ceiling FP using PCA-loading-weighted,
non-uniform probe allocation per category, instead of the flat N=24/category
baseline (Ceiling V1, 80*24=1920 total probes).

Allocation formula: N_cat = clip(round(sqrt(importance_cat) * scale), MIN, MAX)
scale is found by binary search to hit TARGET_TOTAL probes. sqrt (not raw
proportional) is used because estimation noise for a category's mean
accuracy scales ~1/sqrt(N) -- so probe count should grow with the SQUARE
ROOT of how much precision-per-probe matters, not linearly with importance
alone (a common heuristic in optimal experimental design). MIN_PROBES=6
keeps every category minimally covered (protects against a genuinely novel
model breaking the established category-correlation pattern -- see the
extrapolation-risk caveat from last night's discussion). MAX_PROBES=60 caps
over-investment in the single top category.

Within each category, still uses top-variance PROMPT selection (same as
Ceiling V1) -- only the PER-CATEGORY BUDGET is new, not the within-category
selection rule.

Reuses build_embedllm_ceiling_fp.py-style centered-matrix + PCA construction
(same recipe as V2/V1) so the resulting FP is directly comparable.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
OUT_DIR = Path("local_descriptors/embedllm-ceiling-pcaweighted-pca5")
K_PCA = 5
MIN_PROBES = 15
MAX_PROBES = 60
TARGET_TOTAL = 1800
UNIFORM_BASELINE_TOTAL = 80 * 24  # 1920


def load_importance():
    d = json.load(open(ANALYSIS_DIR / "pca_category_importance.json", encoding="utf-8"))
    cats = d["ranked_categories"]
    imp = np.array(d["ranked_importance"])
    # re-sort back to a category -> importance dict (order doesn't matter for a dict)
    return {c: v for c, v in zip(cats, imp)}


def solve_allocation(importance_by_cat, categories, target_total, min_n, max_n):
    imp = np.array([importance_by_cat[c] for c in categories])
    sqrt_imp = np.sqrt(imp)

    def total_for_scale(scale):
        n = np.clip(np.round(sqrt_imp * scale), min_n, max_n)
        return n.sum(), n

    lo, hi = 0.0, 1e6
    for _ in range(60):
        mid = (lo + hi) / 2
        total, n = total_for_scale(mid)
        if total < target_total:
            lo = mid
        else:
            hi = mid
    total, n = total_for_scale(hi)
    return {c: int(v) for c, v in zip(categories, n)}, int(total)


def build_probe_sampled_pca5_weighted(df, allocation, models, categories, out_dir):
    print("Selecting top-variance probes per category, per-category budget...", flush=True)
    per_prompt = df.groupby("prompt_id").agg(category=("category", "first"), var=("label", "var")).reset_index()
    selected_ids = set()
    for cat, grp in per_prompt.groupby("category"):
        n = allocation.get(cat, MIN_PROBES)
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
    allocation, total = solve_allocation(importance_by_cat, categories, TARGET_TOTAL, MIN_PROBES, MAX_PROBES)

    print("\n" + "=" * 80)
    print(f"PROBE ALLOCATION (target={TARGET_TOTAL}, actual total={total}, "
          f"vs uniform baseline={UNIFORM_BASELINE_TOTAL} -> {100*(1-total/UNIFORM_BASELINE_TOTAL):.1f}% reduction)")
    print("=" * 80)
    sorted_cats = sorted(categories, key=lambda c: -importance_by_cat[c])
    for c in sorted_cats[:10]:
        print(f"  {c:<40s} importance={importance_by_cat[c]:.4f}  probes={allocation[c]}")
    print("  ...")
    for c in sorted_cats[-5:]:
        print(f"  {c:<40s} importance={importance_by_cat[c]:.4f}  probes={allocation[c]}")

    n_probes = build_probe_sampled_pca5_weighted(df, allocation, models, categories, OUT_DIR)

    out_path = ANALYSIS_DIR / "pca_weighted_probe_allocation.json"
    json.dump({"allocation": allocation, "total_probes": n_probes,
               "uniform_baseline": UNIFORM_BASELINE_TOTAL,
               "reduction_pct": 100 * (1 - n_probes / UNIFORM_BASELINE_TOTAL)},
              open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved allocation -> {out_path}")
    print(f"Saved FP -> {OUT_DIR}")


if __name__ == "__main__":
    main()
