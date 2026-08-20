"""LLMRouterBench Ceiling FP, 900-probe budget, UNIFORM allocation across
the 8 datasets, NO binning (dim=8, matching Pure V2's convention -- one
raw mean-accuracy number per dataset from the selected probe subset only).

This is the genuine LLMRouterBench analogue of EmbedLLM's official
1800-probe headline FP (embedllm-ceiling-scalesweep-uniform-nocompress-
minpctcap3-1800): same probe budget as the CSCR fairness comparison
(900), same uniform-allocation-with-per-category-cap logic, same
"no compression" convention (dim = n_categories, not binned/PCA).

Unlike build_fp_v15_900.py (PCA-importance-weighted allocation, binned
into fixed 192=8*24 dims), this uses:
  - uniform per-dataset quota (clipped to that dataset's own Set A size)
  - top-variance probe SELECTION within each dataset's quota (same
    nlargest(n,"var") rule used everywhere else in this project)
  - dim=8, no binning -- the selected probes' scores are averaged
    directly into one number per dataset, exactly like Pure V2
"""
from pathlib import Path

import numpy as np

import common

OUT_DIR = Path("local_descriptors/llmrouterbench-ceiling-900-uniform-nocompress")
TARGET_TOTAL = 900
MIN_PROBES = 1


def solve_allocation_uniform(sizes_by_ds, target_total, min_n):
    datasets = common.DATASETS
    lo, hi = 0.0, float(max(sizes_by_ds[d] for d in datasets))

    def total_for_quota(q):
        return sum(min(q, sizes_by_ds[d]) for d in datasets)

    for _ in range(60):
        mid = (lo + hi) / 2
        if total_for_quota(mid) < target_total:
            lo = mid
        else:
            hi = mid
    quota = hi
    alloc = {d: max(min_n, int(round(min(quota, sizes_by_ds[d])))) for d in datasets}
    return alloc, sum(alloc.values())


def main():
    setA_by_ds = {}
    for ds in common.DATASETS:
        queries, scores, costs, raw_outputs = common.build_wide_table(ds)
        n = len(queries)
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        n_a = int(n * 0.8)
        idx_a = perm[:n_a]
        setA_by_ds[ds] = {"scores": scores[idx_a]}
        print(f"  {ds:30s} SetA={len(idx_a)}", flush=True)

    sizes_by_ds = {ds: len(setA_by_ds[ds]["scores"]) for ds in common.DATASETS}
    allocation, total = solve_allocation_uniform(sizes_by_ds, TARGET_TOTAL, MIN_PROBES)
    print(f"\nUNIFORM ALLOCATION (target={TARGET_TOTAL}, actual={total}):", flush=True)
    for ds in common.DATASETS:
        print(f"  {ds:30s} probes={allocation[ds]:4d} (cap={sizes_by_ds[ds]})", flush=True)

    n_models = len(common.MODELS_33)
    raw = np.zeros((n_models, len(common.DATASETS)))
    for di, ds in enumerate(common.DATASETS):
        scores = setA_by_ds[ds]["scores"]  # (n_rows, n_models)
        var = scores.var(axis=1)
        order = np.argsort(-var)
        n = allocation[ds]
        selected = order[:n]
        raw[:, di] = scores[selected].mean(axis=0)

    pool_mean = raw.mean(axis=0, keepdims=True)
    centered = raw - pool_mean

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for j, m in enumerate(common.MODELS_33):
        vec = centered[j] / (np.linalg.norm(centered[j]) + 1e-12)
        np.save(OUT_DIR / f"{common.NAME_TO_SAFE[m]}.npy", vec.astype(np.float32))

    E = np.stack([np.load(OUT_DIR / f"{common.NAME_TO_SAFE[m]}.npy") for m in common.MODELS_33])
    sim = E @ E.T
    off = sim[~np.eye(len(common.MODELS_33), dtype=bool)]
    print(f"\n900-uniform-nocompress Ceiling: shape={E.shape} actual_total={total} "
          f"pairwise cos sim mean={off.mean():.4f} std={off.std():.4f}", flush=True)
    print(f"Saved -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
