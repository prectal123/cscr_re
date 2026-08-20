"""Same PCA-loading-weighted probe allocation as
embedllm_build_pca_weighted_probe_fp.py (V1.5), but squeezed down to a
192-probe budget instead of 1800 -- the literal probe count CSCR's Perplexity
FP originally used, per the user's request to see how far V1.5 degrades when
forced to match that budget on EmbedLLM's much larger, 80-category / ~112-model
pool.

MIN_PROBES dropped from 15 to 1 (unlike the 1800-probe version, a floor of 15
across 80 categories alone needs 1200 probes -- already 6x the entire 192
budget, so the binary-search allocator would have no feasible solution).
Every category still gets at least 1 probe (avoids empty-category NaN issues
in the pivot table), and PCA-importance is left as much room as possible to
concentrate the remaining ~112 probes on the categories it deems important --
this is the one lever that could still help ("PCA를 잘 조절하면 또 몰라").

Expected result (per user, already anticipated): likely a large all-seen
degradation, consistent with 23.3's density-based prediction (~2.4 probes/
category average is close to the density that failed non-significantly on
LLMRouterBench's 22-category test, 19.3) and the floor=6 collapse already
seen on EmbedLLM itself (22.14, 0/3 all-seen at floor=6/~1200 probes -- this
run is a much harsher squeeze than that).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
OUT_DIR = Path("local_descriptors/embedllm-ceiling-pcaweighted-pca5-192")
K_PCA = 5
MIN_PROBES = 1
MAX_PROBES = 60
TARGET_TOTAL = 192
UNIFORM_BASELINE_TOTAL = 80 * 24  # 1920


def load_importance():
    d = json.load(open(ANALYSIS_DIR / "pca_category_importance.json", encoding="utf-8"))
    cats = d["ranked_categories"]
    imp = np.array(d["ranked_importance"])
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
    print(f"{len(models)} models, {len(categories)} categories", flush=True)

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
    for c in sorted_cats[-10:]:
        print(f"  {c:<40s} importance={importance_by_cat[c]:.4f}  probes={allocation[c]}")
    n_at_floor = sum(1 for c in categories if allocation[c] == MIN_PROBES)
    print(f"\n{n_at_floor}/{len(categories)} categories at floor ({MIN_PROBES} probe)", flush=True)

    n_probes = build_probe_sampled_pca5_weighted(df, allocation, models, categories, OUT_DIR)

    out_path = ANALYSIS_DIR / "pca_weighted_probe_allocation_192.json"
    json.dump({"allocation": allocation, "total_probes": n_probes,
               "uniform_baseline": UNIFORM_BASELINE_TOTAL,
               "reduction_pct": 100 * (1 - n_probes / UNIFORM_BASELINE_TOTAL)},
              open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved allocation -> {out_path}")
    print(f"Saved FP -> {OUT_DIR}")


if __name__ == "__main__":
    main()
