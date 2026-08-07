"""Random FP -- pure noise negative control, same dimensionality as Ceiling V2
(22-dim). Each model gets an independent random Gaussian vector, L2-normalized.
No relationship to actual capability whatsoever. Used to check whether the LOO
pipeline produces above-chance AUC even with a meaningless descriptor (sanity
check on the whole training/eval setup, not just the FP methodology).
"""
from pathlib import Path

import numpy as np

import common_lite20 as common

SEED = 42
DIM = 22
OUT_DIR = Path("local_descriptors/llmrouterbench_lite20/ceiling_random")


def main():
    rng = np.random.RandomState(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for m in common.MODELS_20:
        vec = rng.randn(DIM).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        np.save(OUT_DIR / f"{common.NAME_TO_SAFE[m]}.npy", vec)

    E = np.stack([np.load(OUT_DIR / f"{common.NAME_TO_SAFE[m]}.npy") for m in common.MODELS_20])
    sim = E @ E.T
    off = sim[~np.eye(len(common.MODELS_20), dtype=bool)]
    print(f"Random FP: shape={E.shape}  pairwise cosine sim: mean={off.mean():.4f} std={off.std():.4f} "
          f"min={off.min():.4f} max={off.max():.4f}")
    print(f"Saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
