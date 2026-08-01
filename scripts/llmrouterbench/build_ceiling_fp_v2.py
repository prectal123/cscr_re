"""Ceiling FP (192-dim) rebuilt from the v2 (within-tier-variance) probes."""
import json
import pickle
from pathlib import Path

import numpy as np

import common

DATA_DIR = Path("local_descriptors/llmrouterbench_v2")
OUT_DIR = DATA_DIR / "ceiling"


def main():
    with open(DATA_DIR / "probe_info.json", encoding="utf-8") as f:
        probes = json.load(f)
    with open(DATA_DIR / "setA_setB_split.pkl", "rb") as f:
        split = pickle.load(f)
    setA = split["setA"]

    raw = np.zeros((len(probes), len(common.MODELS_33)))
    for p_i, p in enumerate(probes):
        ds = p["dataset"]
        local_i = p["local_idx_in_setA"]
        raw[p_i, :] = setA[ds]["scores"][local_i]

    pool_mean = raw.mean(axis=1, keepdims=True)
    centered = raw - pool_mean

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for j, m in enumerate(common.MODELS_33):
        vec = centered[:, j]
        vec = (vec / (np.linalg.norm(vec) + 1e-12)).astype(np.float32)
        np.save(OUT_DIR / f"{common.NAME_TO_SAFE[m]}.npy", vec)

    E = np.stack([np.load(OUT_DIR / f"{common.NAME_TO_SAFE[m]}.npy") for m in common.MODELS_33])
    sim = E @ E.T
    off = sim[~np.eye(len(common.MODELS_33), dtype=bool)]
    print(f"Ceiling FP v2: shape={E.shape}  pairwise cosine sim: mean={off.mean():.4f} std={off.std():.4f} "
          f"min={off.min():.4f} max={off.max():.4f}")
    print(f"Saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
