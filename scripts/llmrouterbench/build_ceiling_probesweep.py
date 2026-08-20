"""LLMRouterBench Ceiling FP, PROBE-COUNT SWEEP, uniform allocation across
the 8 datasets, no binning (dim=8) -- same convention as
build_ceiling_900_uniform_nocompress.py, generalized over multiple probe
budgets so we can test whether "even very few probes suffice for near-
ceiling Unseen AUDC" (found on EmbedLLM, 111 models / 80 categories) also
holds on this much smaller/coarser pool (33 models / 8 categories).

Builds one FP dir per target in TARGETS:
  local_descriptors/llmrouterbench-ceiling-uniform-nocompress-{target}/
The existing llmrouterbench-ceiling-purev2 (full Set A, no cap) is reused
as the top/unlimited anchor -- not rebuilt here.
"""
from pathlib import Path

import numpy as np

import common

OUT_ROOT = Path("local_descriptors")
TARGETS = [64, 160, 320, 480, 640, 900, 1200, 1800]
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
    sizes_by_ds = {ds: len(setA_by_ds[ds]["scores"]) for ds in common.DATASETS}
    print("Set A sizes:", sizes_by_ds, flush=True)

    n_models = len(common.MODELS_33)
    for target in TARGETS:
        allocation, total = solve_allocation_uniform(sizes_by_ds, target, MIN_PROBES)
        raw = np.zeros((n_models, len(common.DATASETS)))
        for di, ds in enumerate(common.DATASETS):
            scores = setA_by_ds[ds]["scores"]
            var = scores.var(axis=1)
            order = np.argsort(-var)
            n = allocation[ds]
            selected = order[:n]
            raw[:, di] = scores[selected].mean(axis=0)

        pool_mean = raw.mean(axis=0, keepdims=True)
        centered = raw - pool_mean

        out_dir = OUT_ROOT / f"llmrouterbench-ceiling-uniform-nocompress-{target}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for j, m in enumerate(common.MODELS_33):
            vec = centered[j] / (np.linalg.norm(centered[j]) + 1e-12)
            np.save(out_dir / f"{common.NAME_TO_SAFE[m]}.npy", vec.astype(np.float32))

        E = np.stack([np.load(out_dir / f"{common.NAME_TO_SAFE[m]}.npy") for m in common.MODELS_33])
        sim = E @ E.T
        off = sim[~np.eye(n_models, dtype=bool)]
        print(f"target={target:5d} actual={total:5d} alloc={allocation} "
              f"cos_sim mean={off.mean():.4f} std={off.std():.4f} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
