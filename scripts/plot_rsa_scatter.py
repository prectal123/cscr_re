"""Scatter plots for the 3-way RSA comparison, laid out side by side:
left = Logit vs Perplexity, middle = Logit vs Capability,
right = Perplexity vs Capability. Each point is one of the 21 model
pairs; the diagonal line marks where points would fall under perfect
agreement (rho=1) between the two similarity measures being compared.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from itertools import permutations

POOL_PATH = "experts/pool-mix-instruct-7.json"
LOGIT_DIR = Path("local_descriptors/mix-instruct-logit")
PERP_DIR = Path("local_descriptors/mix-instruct-perplexity")
CAP_DIR = Path("local_descriptors/mix-instruct-capability")
OUT_PATH = Path("local_descriptors/analysis/rsa_scatter_3way.png")


def cosine_sim_matrix(vecs: dict, order: list) -> np.ndarray:
    n = len(order)
    M = np.zeros((n, n))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            va, vb = vecs[a], vecs[b]
            M[i, j] = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
    return M


def upper_tri(M: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(M.shape[0], k=1)
    return M[idx]


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
    pool = json.load(open(POOL_PATH))
    order = pool

    logit_vecs = {m: np.load(LOGIT_DIR / f"{m}.npy") for m in pool}
    perp_vecs = {m: np.load(PERP_DIR / f"{m}.npy") for m in pool}
    cap_vecs = {m: np.load(CAP_DIR / f"{m}.npy") for m in pool}

    M_logit = cosine_sim_matrix(logit_vecs, order)
    M_perp = cosine_sim_matrix(perp_vecs, order)
    M_cap = cosine_sim_matrix(cap_vecs, order)

    panels = [
        ("Logit", "Perplexity", M_logit, M_perp, "#3B82C4"),   # blue
        ("Logit", "Capability", M_logit, M_cap, "#E07B39"),    # orange
        ("Perplexity", "Capability", M_perp, M_cap, "#6FAE5C"),  # green
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (name_a, name_b, Ma, Mb, color) in zip(axes, panels):
        x = upper_tri(Ma)
        y = upper_tri(Mb)
        rho, _ = spearmanr(x, y)
        p = exact_mantel_p(Ma, Mb, rho)

        ax.scatter(x, y, s=70, color=color, edgecolor="black", linewidth=0.6, zorder=3)
        lo, hi = 0.0, 1.0
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.2, zorder=1,
                label="perfect agreement (rho=1)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"{name_a} cosine similarity")
        ax.set_ylabel(f"{name_b} cosine similarity")
        sig = "significant" if p < 0.05 else "not significant"
        ax.set_title(f"{name_a} vs {name_b}\nrho={rho:+.3f}, p={p:.3f} ({sig})", fontsize=11)
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_aspect("equal")

    fig.suptitle("Do pairwise similarity rankings agree across representations?\n"
                  "(each point = one of the 21 model pairs; near the diagonal = agreement)",
                  fontsize=12, y=1.04)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
