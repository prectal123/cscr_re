"""Ceiling FP variant WITHOUT mean-centering -- tests the user's hypothesis
that subtracting the per-probe pool mean (which removes shared 'how hard is
this query' difficulty) might also be erasing real per-model signal, not
just noise/confound. Raw scores, L2-normalized directly, same 528 probes.
"""
import json
import pickle
from pathlib import Path

import numpy as np

import common_lite20 as common

DATA_DIR = Path("local_descriptors/llmrouterbench_lite20")
OUT_DIR = DATA_DIR / "ceiling_uncentered"


def main():
    with open(DATA_DIR / "probe_info.json", encoding="utf-8") as f:
        probes = json.load(f)
    with open(DATA_DIR / "setA_setB_split.pkl", "rb") as f:
        split = pickle.load(f)
    setA = split["setA"]

    raw = np.zeros((len(probes), len(common.MODELS_20)))
    for p_i, p in enumerate(probes):
        ds = p["dataset"]
        local_i = p["local_idx_in_setA"]
        raw[p_i, :] = setA[ds]["scores"][local_i]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for j, m in enumerate(common.MODELS_20):
        vec = raw[:, j]
        vec = (vec / (np.linalg.norm(vec) + 1e-12)).astype(np.float32)
        np.save(OUT_DIR / f"{common.NAME_TO_SAFE[m]}.npy", vec)

    E = np.stack([np.load(OUT_DIR / f"{common.NAME_TO_SAFE[m]}.npy") for m in common.MODELS_20])
    sim = E @ E.T
    off = sim[~np.eye(len(common.MODELS_20), dtype=bool)]
    print(f"Ceiling FP (uncentered) lite20: shape={E.shape}  pairwise cosine sim: mean={off.mean():.4f} "
          f"std={off.std():.4f} min={off.min():.4f} max={off.max():.4f}")
    print(f"Saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
