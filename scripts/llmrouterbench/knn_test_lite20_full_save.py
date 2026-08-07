"""Re-runs the lightweight-20 kNN unseen-recovery test for all FP types
(Ceiling FP = category-rate, Pseudo Ceiling = 528-probe, Perplexity, plus
uniform baseline) and, unlike the earlier ad-hoc runs, SAVES the results to
JSON -- both the overall (pooled across all 22 datasets) rho per held-out
model, and a per-dataset breakdown (rho computed within each of the 22
datasets' Set B rows separately). Small datasets (e.g. aime, n=12) will have
noisy per-dataset rho -- flagged via n_rows in the output.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "scripts/llmrouterbench")
import common_lite20 as common
import loo_recovery_lite20 as loo

CATRATE_DIR = Path("local_descriptors/llmrouterbench_lite20/ceiling_categoryrate")
OUT_PATH = Path("local_descriptors/llmrouterbench_lite20/knn_test_full_results.json")


def load_fp(desc_dir, pool):
    E = np.stack([np.load(desc_dir / f"{common.NAME_TO_SAFE[m]}.npy") for m in pool])
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def per_dataset_row_ranges(split, pool):
    """Returns {dataset: (start_idx, end_idx)} into the pooled Set B array,
    matching the concatenation order used by build_setB_eval."""
    ranges = {}
    start = 0
    for ds in common.DATASETS:
        n = len(split["setB"][ds]["queries"])
        ranges[ds] = (start, start + n)
        start += n
    return ranges


def test_fp(E, true_scores, pool, ranges):
    sim_full = E @ E.T
    results = {}
    for i, held_out in enumerate(pool):
        others_idx = [j for j in range(len(pool)) if j != i]
        other_scores = true_scores[:, others_idx]
        true_m = true_scores[:, i]

        sims = sim_full[i, others_idx]
        w = np.clip(sims, 0, None)
        if w.sum() < 1e-9:
            w = np.ones_like(w)
        w = w / w.sum()
        fp_proxy = other_scores @ w

        overall_rho, _ = spearmanr(fp_proxy, true_m)
        per_ds = {}
        for ds, (s, e) in ranges.items():
            if e - s < 5:
                per_ds[ds] = {"rho": None, "n_rows": e - s, "note": "too few rows"}
                continue
            rho, _ = spearmanr(fp_proxy[s:e], true_m[s:e])
            per_ds[ds] = {"rho": None if np.isnan(rho) else float(rho), "n_rows": e - s}
        results[held_out] = {"overall_rho": float(overall_rho), "per_dataset": per_ds}
    return results


def test_uniform(true_scores, pool, ranges):
    results = {}
    for i, held_out in enumerate(pool):
        others_idx = [j for j in range(len(pool)) if j != i]
        proxy = true_scores[:, others_idx].mean(axis=1)
        true_m = true_scores[:, i]
        overall_rho, _ = spearmanr(proxy, true_m)
        per_ds = {}
        for ds, (s, e) in ranges.items():
            if e - s < 5:
                per_ds[ds] = {"rho": None, "n_rows": e - s, "note": "too few rows"}
                continue
            rho, _ = spearmanr(proxy[s:e], true_m[s:e])
            per_ds[ds] = {"rho": None if np.isnan(rho) else float(rho), "n_rows": e - s}
        results[held_out] = {"overall_rho": float(overall_rho), "per_dataset": per_ds}
    return results


def main():
    split = loo.load_split()
    setB_queries, true_scores = loo.build_setB_eval(split)
    pool = common.MODELS_20
    ranges = per_dataset_row_ranges(split, pool)
    print(f"Set B total: {true_scores.shape[0]} rows x {len(pool)} models, {len(ranges)} datasets", flush=True)

    all_results = {}

    print("Testing Ceiling FP (category-rate)...", flush=True)
    E = load_fp(CATRATE_DIR, pool)
    all_results["Ceiling_FP"] = test_fp(E, true_scores, pool, ranges)

    print("Testing Pseudo Ceiling (528-probe)...", flush=True)
    E = load_fp(loo.CEILING_DIR, pool)
    all_results["Pseudo_Ceiling"] = test_fp(E, true_scores, pool, ranges)

    print("Testing Perplexity...", flush=True)
    E = load_fp(loo.PERP_DIR, pool)
    all_results["Perplexity"] = test_fp(E, true_scores, pool, ranges)

    print("Testing Uniform baseline...", flush=True)
    all_results["Uniform"] = test_uniform(true_scores, pool, ranges)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved -> {OUT_PATH}", flush=True)

    for fp_name in ["Ceiling_FP", "Pseudo_Ceiling", "Perplexity", "Uniform"]:
        mean_overall = np.mean([v["overall_rho"] for v in all_results[fp_name].values()])
        print(f"{fp_name}: mean overall rho = {mean_overall:.4f}", flush=True)


if __name__ == "__main__":
    main()
