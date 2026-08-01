"""Ceiling FP variant that additionally removes the dominant shared axis
(PC1 of the mean-centered probe x model score matrix, ~28.5% of variance,
found to closely match a 'generic strength' cluster largely orthogonal to
per-domain specialization) as a separate scalar, leaving a domain-purified
residual vector for cosine-similarity routing.

raw scores --(subtract per-probe pool mean)--> centered
centered   --(subtract projection onto PC1)--> residual (domain-purified)
model's projection onto PC1                  -> stored separately as a scalar
"""
import json
import pickle
from pathlib import Path

import numpy as np

import common_lite20 as common

DATA_DIR = Path("local_descriptors/llmrouterbench_lite20")
OUT_DIR = DATA_DIR / "ceiling_deflated"
SCALAR_OUT = DATA_DIR / "pc1_loadings.json"


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

    pool_mean = raw.mean(axis=1, keepdims=True)
    centered = raw - pool_mean  # (n_probes, 20)

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pc1_dir = U[:, 0]  # (n_probes,), unit vector
    explained = float((S[0] ** 2) / (S ** 2).sum())
    print(f"PC1 explained variance ratio: {explained:.4f}")

    loadings = centered.T @ pc1_dir  # (20,) -- per-model scalar
    residual = centered - np.outer(pc1_dir, loadings)  # (n_probes, 20), PC1 projected out

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for j, m in enumerate(common.MODELS_20):
        vec = residual[:, j]
        vec = (vec / (np.linalg.norm(vec) + 1e-12)).astype(np.float32)
        np.save(OUT_DIR / f"{common.NAME_TO_SAFE[m]}.npy", vec)

    with open(SCALAR_OUT, "w", encoding="utf-8") as f:
        json.dump({m: float(loadings[j]) for j, m in enumerate(common.MODELS_20)}, f, indent=2)

    E = np.stack([np.load(OUT_DIR / f"{common.NAME_TO_SAFE[m]}.npy") for m in common.MODELS_20])
    sim = E @ E.T
    off = sim[~np.eye(len(common.MODELS_20), dtype=bool)]
    print(f"Ceiling FP (deflated) lite20: shape={E.shape}  pairwise cosine sim: mean={off.mean():.4f} "
          f"std={off.std():.4f} min={off.min():.4f} max={off.max():.4f}")
    print(f"Saved vectors -> {OUT_DIR}")
    print(f"Saved PC1 loadings -> {SCALAR_OUT}")


if __name__ == "__main__":
    main()
