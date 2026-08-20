"""RouterBench "Ceiling V1.5" -- same PCA-loading-weighted, non-uniform probe
allocation idea as EmbedLLM's (embedllm_pca_loading_analysis.py +
embedllm_build_pca_weighted_probe_fp.py), ported to RouterBench's 86
eval_name categories, target ~1800 total probes, MIN_PROBES=15 (the winning
EmbedLLM floor), MAX_PROBES=60.

Unlike EmbedLLM's version, this does NOT apply a final PCA-5 compression --
RouterBench's existing Ceiling FP (routerbench_knn_test.py's build_ceiling_fp,
used as rb.CEILING_DIR everywhere in this project) is the RAW 86-dim
mean-centered+L2-normalized vector, uncompressed. Keeping the same 86-dim
output space here isolates "cheaper probe budget" as the only variable
changed vs the existing full-Set-A Ceiling FP, matching the same discipline
used for the Perplexity FP N_PROBES sweep (dims fixed, probes varied).

Output is a drop-in replacement directory usable anywhere rb.CEILING_DIR is
used (e.g. FP_SOURCES in routerbench_perplexity_combined.py-style scripts).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "scripts")
import routerbench_knn_test as rb

OUT_DIR = Path("local_descriptors/routerbench-ceiling-v15")
ANALYSIS_DIR = Path("local_descriptors/routerbench-analysis")
MIN_PROBES = 15
MAX_PROBES = 60
TARGET_TOTAL = 1800
K_PCA_FOR_IMPORTANCE = 5  # only used to weight categories, final FP stays 86-dim raw


def compute_category_importance(set_a, models, cols, eval_names):
    cat_to_idx = {e: i for i, e in enumerate(eval_names)}
    n_models = len(models)
    raw = np.zeros((n_models, len(eval_names)))
    counts = np.zeros((n_models, len(eval_names)))
    for i, col in enumerate(cols):
        for ev, score in zip(set_a["eval_name"], set_a[col]):
            ci = cat_to_idx[ev]
            raw[i, ci] += float(score)
            counts[i, ci] += 1
    raw = raw / np.maximum(counts, 1)
    pool_mean = raw.mean(axis=0, keepdims=True)
    centered = raw - pool_mean  # (n_models, n_eval_names)

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()
    k = min(K_PCA_FOR_IMPORTANCE, Vt.shape[0])
    loadings = Vt[:k]
    importance = (explained[:k, None] * (loadings ** 2)).sum(axis=0)
    importance = importance / importance.sum()
    print(f"Explained variance top-{k}: {[f'{e:.4f}' for e in explained[:k]]} "
          f"(cumulative {explained[:k].sum():.4f})", flush=True)
    return {e: float(v) for e, v in zip(eval_names, importance)}


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


def build_v15_fp(set_a, allocation, models, cols, eval_names, out_dir):
    print("Selecting top-variance probes per category, per-category budget...", flush=True)
    set_a = set_a.copy()
    set_a["_var"] = set_a[cols].astype(float).var(axis=1)

    selected_mask = pd.Series(False, index=set_a.index)
    for ev, grp in set_a.groupby("eval_name"):
        n = allocation.get(ev, MIN_PROBES)
        top_idx = grp.nlargest(n, "_var").index
        selected_mask.loc[top_idx] = True
    sub = set_a[selected_mask]
    print(f"  {len(sub)} probes selected total ({len(eval_names)} categories)", flush=True)

    cat_to_idx = {e: i for i, e in enumerate(eval_names)}
    n_models = len(models)
    raw = np.zeros((n_models, len(eval_names)))
    counts = np.zeros((n_models, len(eval_names)))
    for i, col in enumerate(cols):
        for ev, score in zip(sub["eval_name"], sub[col]):
            ci = cat_to_idx[ev]
            raw[i, ci] += float(score)
            counts[i, ci] += 1
    raw = raw / np.maximum(counts, 1)
    pool_mean = raw.mean(axis=0, keepdims=True)
    centered = raw - pool_mean

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(models):
        vec = centered[i]
        vec = (vec / (np.linalg.norm(vec) + 1e-12)).astype(np.float32)
        np.save(out_dir / f"{m}.npy", vec)
    print(f"  saved -> {out_dir} (86-dim raw, no PCA compression)", flush=True)
    return len(sub)


def main():
    set_a, set_b = rb.load_data()
    models, cols = rb.NAMES, rb.MODELS
    eval_names = sorted(set_a["eval_name"].unique())
    print(f"{len(models)} models, {len(eval_names)} categories", flush=True)

    importance_by_cat = compute_category_importance(set_a, models, cols, eval_names)
    allocation, total = solve_allocation(importance_by_cat, eval_names, TARGET_TOTAL, MIN_PROBES, MAX_PROBES)

    uniform_baseline_total = len(eval_names) * 24  # matches this project's usual flat-N=24/category convention
    print("\n" + "=" * 80)
    print(f"PROBE ALLOCATION (target={TARGET_TOTAL}, actual total={total}, "
          f"vs flat-24/cat baseline={uniform_baseline_total} -> "
          f"{100*(1-total/uniform_baseline_total):.1f}% reduction)")
    print("=" * 80)
    sorted_cats = sorted(eval_names, key=lambda c: -importance_by_cat[c])
    for c in sorted_cats[:10]:
        print(f"  {c:<40s} importance={importance_by_cat[c]:.4f}  probes={allocation[c]}")
    print("  ...")
    for c in sorted_cats[-5:]:
        print(f"  {c:<40s} importance={importance_by_cat[c]:.4f}  probes={allocation[c]}")

    n_probes = build_v15_fp(set_a, allocation, models, cols, eval_names, OUT_DIR)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / "routerbench_ceiling_v15_allocation.json"
    json.dump({"allocation": allocation, "total_probes": n_probes,
               "flat24_baseline": uniform_baseline_total,
               "reduction_pct": 100 * (1 - n_probes / uniform_baseline_total)},
              open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved allocation -> {out_path}")
    print(f"Saved FP -> {OUT_DIR}")


if __name__ == "__main__":
    main()
