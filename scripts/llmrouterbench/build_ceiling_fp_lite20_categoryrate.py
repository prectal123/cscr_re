"""Ceiling FP variant using per-CATEGORY accuracy rate (22-dim, one per
dataset) instead of 528 individual noisy probe queries. Each dimension =
a model's mean score across ALL of Set A for that dataset -- much lower
noise per dimension than the fine-probe descriptor, at the cost of losing
within-category resolution. Mean-centered (per-category pool mean removed)
+ L2-normalized, same treatment as the original Ceiling FP.
"""
import pickle
from pathlib import Path

import numpy as np

import common_lite20 as common

DATA_DIR = Path("local_descriptors/llmrouterbench_lite20")
OUT_DIR = DATA_DIR / "ceiling_categoryrate"


def main():
    with open(DATA_DIR / "setA_setB_split.pkl", "rb") as f:
        split = pickle.load(f)
    setA = split["setA"]

    # raw[i, j] = model j's mean score on dataset i (Set A only)
    raw = np.zeros((len(common.DATASETS), len(common.MODELS_20)))
    for i, ds in enumerate(common.DATASETS):
        raw[i, :] = setA[ds]["scores"].mean(axis=0)

    pool_mean = raw.mean(axis=1, keepdims=True)
    centered = raw - pool_mean  # (22, 20)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for j, m in enumerate(common.MODELS_20):
        vec = centered[:, j]
        vec = (vec / (np.linalg.norm(vec) + 1e-12)).astype(np.float32)
        np.save(OUT_DIR / f"{common.NAME_TO_SAFE[m]}.npy", vec)

    print("per-dataset accuracy rate (Set A), rows=dataset, showing min/max across 20 models:")
    for i, ds in enumerate(common.DATASETS):
        print(f"  {ds:28s} min={raw[i].min():.3f} max={raw[i].max():.3f} mean={raw[i].mean():.3f}")

    E = np.stack([np.load(OUT_DIR / f"{common.NAME_TO_SAFE[m]}.npy") for m in common.MODELS_20])
    sim = E @ E.T
    off = sim[~np.eye(len(common.MODELS_20), dtype=bool)]
    print(f"\nCeiling FP (category-rate) lite20: shape={E.shape}  pairwise cosine sim: mean={off.mean():.4f} "
          f"std={off.std():.4f} min={off.min():.4f} max={off.max():.4f}")
    print(f"Saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
