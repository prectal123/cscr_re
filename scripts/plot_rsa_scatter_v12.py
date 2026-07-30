"""Single-panel RSA scatter for v1.2 (JSON->MiniLM) vs Capability.

Unlike plot_rsa_scatter.py (which uses fixed [0,1] axes -- fine for that
script's near-zero-rho comparisons, where zooming in wouldn't change the
"no relationship" conclusion), this one zooms to the actual data range.
Fixed [0,1] axes compress v1.2's real spread (cosine similarities all
sitting in a narrow band, e.g. ~0.67-0.86) into what looks like a tight
diagonal cluster; zoomed in, the same rho=+0.43 correctly reads as a loose,
modest association, not a strong one. Using fixed axes here would visually
overstate how well-aligned v1.2 and capability actually are.
"""
import argparse
import json
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def cosine_sim_matrix(vecs: dict, order: list) -> np.ndarray:
    n = len(order)
    M = np.zeros((n, n))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            va, vb = vecs[a], vecs[b]
            M[i, j] = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
    return M


def upper_tri(M: np.ndarray) -> np.ndarray:
    return M[np.triu_indices(M.shape[0], k=1)]


def exact_mantel_p(A: np.ndarray, B: np.ndarray, observed: float) -> float:
    n = A.shape[0]
    a_flat = upper_tri(A)
    perm_rhos = []
    for perm in permutations(range(n)):
        perm = np.array(perm)
        B_perm = B[np.ix_(perm, perm)]
        rho, _ = spearmanr(a_flat, upper_tri(B_perm))
        perm_rhos.append(rho)
    perm_rhos = np.array(perm_rhos)
    return float(np.mean(np.abs(perm_rhos) >= abs(observed) - 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="experts/pool-mix-instruct-7.json")
    ap.add_argument("--v12_dir", default="local_descriptors/mix-instruct-v12")
    ap.add_argument("--capability_dir", default="local_descriptors/mix-instruct-capability")
    ap.add_argument("--out", default="local_descriptors/analysis/rsa_scatter_v12_vs_capability.png")
    ap.add_argument("--fixed_axes", action="store_true",
                     help="use fixed [0,1] axes instead of zooming to the actual data range "
                          "(see module docstring for why zoomed is the default)")
    args = ap.parse_args()

    pool = json.load(open(args.pool))
    v12_vecs = {m: np.load(f"{args.v12_dir}/{m}.npy") for m in pool}
    cap_vecs = {m: np.load(f"{args.capability_dir}/{m}.npy") for m in pool}

    M_v12 = cosine_sim_matrix(v12_vecs, pool)
    M_cap = cosine_sim_matrix(cap_vecs, pool)

    x = upper_tri(M_v12)
    y = upper_tri(M_cap)
    rho, _ = spearmanr(x, y)
    p = exact_mantel_p(M_v12, M_cap, rho)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    color = "#C4553B"
    ax.scatter(x, y, s=80, color=color, edgecolor="black", linewidth=0.6, zorder=3)

    if args.fixed_axes:
        xlo, xhi, ylo, yhi = 0.0, 1.0, 0.0, 1.0
        axes_desc = "fixed [0,1] axes"
    else:
        pad_x = (x.max() - x.min()) * 0.15
        pad_y = (y.max() - y.min()) * 0.15
        xlo, xhi = x.min() - pad_x, x.max() + pad_x
        ylo, yhi = y.min() - pad_y, y.max() + pad_y
        axes_desc = "axes zoomed to actual data range"
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, zorder=1,
            label="perfect agreement (rho=1)")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    if args.fixed_axes:
        ax.set_aspect("equal")
    ax.set_xlabel("v1.2 (JSON->MiniLM) cosine similarity")
    ax.set_ylabel("Capability (bartscore) cosine similarity")
    sig = "significant" if p < 0.05 else "not significant"
    ax.set_title(f"v1.2 vs Capability ({axes_desc})\n"
                 f"rho={rho:+.3f}, p={p:.3f} ({sig}, exact Mantel)", fontsize=11)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=8, loc="upper left")

    zoom_note = ("fixed to [0,1], matching plot_rsa_scatter.py's convention"
                  if args.fixed_axes else
                  "zoomed -- NOT [0,1] -- to show the real spread")
    fig.suptitle("Do pairwise similarity rankings agree between v1.2 and true capability?\n"
                  f"(each point = one model pair; axes {zoom_note})",
                  fontsize=11, y=1.06)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved -> {out_path}")
    print(f"rho={rho:+.4f} p={p:.4f}")


if __name__ == "__main__":
    main()
