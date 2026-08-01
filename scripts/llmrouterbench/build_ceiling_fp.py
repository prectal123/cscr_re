"""Ceiling FP (192-dim) for the LLMRouterBench 33-model pool: raw score at
each of the 192 variance-selected probes, mean-centered across the 33
models (removes shared/generic difficulty, same fix as RouterBench's
Ceiling FP -- see PROGRESS.md 15.2), then L2-normalized.
"""
import json
import pickle
from pathlib import Path

import numpy as np

import common

DATA_DIR = Path("local_descriptors/llmrouterbench")
OUT_DIR = DATA_DIR / "ceiling"


def main():
    with open(DATA_DIR / "probe_info.json", encoding="utf-8") as f:
        probes = json.load(f)
    with open(DATA_DIR / "setA_setB_split.pkl", "rb") as f:
        split = pickle.load(f)
    setA = split["setA"]

    # build (192, 33) raw score matrix -- probes in the fixed order from probe_info.json
    raw = np.zeros((len(probes), len(common.MODELS_33)))
    for p_i, p in enumerate(probes):
        ds = p["dataset"]
        local_i = p["local_idx_in_setA"]
        raw[p_i, :] = setA[ds]["scores"][local_i]

    pool_mean = raw.mean(axis=1, keepdims=True)  # (192, 1) -- shared difficulty per probe
    centered = raw - pool_mean  # (192, 33)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for j, m in enumerate(common.MODELS_33):
        vec = centered[:, j]
        vec = (vec / (np.linalg.norm(vec) + 1e-12)).astype(np.float32)
        np.save(OUT_DIR / f"{common.NAME_TO_SAFE[m]}.npy", vec)

    E = np.stack([np.load(OUT_DIR / f"{common.NAME_TO_SAFE[m]}.npy") for m in common.MODELS_33])
    sim = E @ E.T
    off = sim[~np.eye(len(common.MODELS_33), dtype=bool)]
    print(f"Ceiling FP: shape={E.shape}  pairwise cosine sim: mean={off.mean():.4f} std={off.std():.4f} "
          f"min={off.min():.4f} max={off.max():.4f}")
    print(f"Saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
