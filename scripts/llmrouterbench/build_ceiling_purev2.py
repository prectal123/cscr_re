"""LLMRouterBench "Pure V2" Ceiling FP -- full Set A, no probe subsampling,
no PCA/binning. Mirrors routerbench_knn_test.py's build_ceiling_fp exactly:
per-model mean accuracy per dataset (8 categories = 8-dim, one per
DATASETS entry), mean-centered across models, L2-normalized. This is the
LLMRouterBench analogue of EmbedLLM's `embedllm-ceiling` (V2) and
RouterBench's existing `routerbench-ceiling` -- both of which are
full-data, category-count-dimensional, uncompressed, no probe selection.

Unlike build_fp_v15_900.py (900-probe budget, PCA-weighted allocation
across datasets, binned into a fixed 192=8*24 dims), this uses 100% of
each dataset's Set A rows directly -- no probe selection, no binning,
dim = 8 (one number per dataset, the plain category mean).
"""
import time
from pathlib import Path

import numpy as np

import common

OUT_DIR = Path("local_descriptors/llmrouterbench-ceiling-purev2")


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
        print(f"  {ds:30s} SetA={len(idx_a)} (full, no probe cap)", flush=True)

    n_models = len(common.MODELS_33)
    raw = np.zeros((n_models, len(common.DATASETS)))
    for di, ds in enumerate(common.DATASETS):
        raw[:, di] = setA_by_ds[ds]["scores"].mean(axis=0)  # mean over ALL Set A rows for this dataset

    pool_mean = raw.mean(axis=0, keepdims=True)
    centered = raw - pool_mean

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for j, m in enumerate(common.MODELS_33):
        vec = centered[j] / (np.linalg.norm(centered[j]) + 1e-12)
        np.save(OUT_DIR / f"{common.NAME_TO_SAFE[m]}.npy", vec.astype(np.float32))

    E = np.stack([np.load(OUT_DIR / f"{common.NAME_TO_SAFE[m]}.npy") for m in common.MODELS_33])
    sim = E @ E.T
    off = sim[~np.eye(len(common.MODELS_33), dtype=bool)]
    print(f"\nPure V2 Ceiling: shape={E.shape}  pairwise cos sim mean={off.mean():.4f} std={off.std():.4f}", flush=True)
    print(f"Saved -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
