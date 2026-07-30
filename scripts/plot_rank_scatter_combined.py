"""Rank-scatter, 3 panels side by side (same layout as plot_rsa_scatter.py):
left=Logit vs Capability, middle=Perplexity vs Capability, right=v1.2 vs
Capability. Each point is one of the 21 model pairs; x = rank of that pair
under the panel's representation, y = rank under Capability (both axes
1=most similar). Rank-space version of plot_rsa_scatter.py/plot_rsa_scatter_v12.py
(which plot raw similarity values instead).
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


def ranks_of(sims: np.ndarray) -> np.ndarray:
    order = np.argsort(-sims)  # rank 1 = highest similarity
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(sims) + 1)
    return ranks


def exact_mantel(M_a: np.ndarray, M_b: np.ndarray):
    n = M_a.shape[0]
    a_flat = upper_tri(M_a)
    observed, _ = spearmanr(a_flat, upper_tri(M_b))
    perm_rhos = []
    for perm in permutations(range(n)):
        perm = np.array(perm)
        perm_rhos.append(spearmanr(a_flat, upper_tri(M_b[np.ix_(perm, perm)]))[0])
    p = float(np.mean(np.abs(np.array(perm_rhos)) >= abs(observed) - 1e-12))
    return observed, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="experts/pool-mix-instruct-7.json")
    ap.add_argument("--logit_dir", default="local_descriptors/mix-instruct-logit")
    ap.add_argument("--perplexity_dir", default="local_descriptors/mix-instruct-perplexity")
    ap.add_argument("--v12_dir", default="local_descriptors/mix-instruct-v12")
    ap.add_argument("--capability_dir", default="local_descriptors/mix-instruct-capability")
    ap.add_argument("--out", default="local_descriptors/analysis/rank_scatter_combined.png")
    args = ap.parse_args()

    pool = json.load(open(args.pool))
    n = len(pool)
    n_pairs = n * (n - 1) // 2

    def load(dirpath):
        return {m: np.load(f"{dirpath}/{m}.npy") for m in pool}

    M_logit = cosine_sim_matrix(load(args.logit_dir), pool)
    M_perp = cosine_sim_matrix(load(args.perplexity_dir), pool)
    M_v12 = cosine_sim_matrix(load(args.v12_dir), pool)
    M_cap = cosine_sim_matrix(load(args.capability_dir), pool)

    rank_cap = ranks_of(upper_tri(M_cap))

    panels = [
        ("Logit", M_logit, "#E07B39"),        # orange, same as plot_rsa_scatter.py's Logit/Capability
        ("Perplexity", M_perp, "#6FAE5C"),    # green, same as Perplexity/Capability
        ("v1.2", M_v12, "#C4553B"),           # rust, same as the other v1.2 charts
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (name, M, color) in zip(axes, panels):
        rank_x = ranks_of(upper_tri(M))
        rho, p = exact_mantel(M, M_cap)
        sig = "significant" if p < 0.05 else "not significant"

        ax.plot([1, n_pairs], [1, n_pairs], linestyle="--", color="gray", linewidth=1.2, zorder=1,
                label="perfect rank agreement")
        ax.scatter(rank_x, rank_cap, s=70, color=color, edgecolor="black", linewidth=0.6, zorder=3)

        ax.set_xlim(0.5, n_pairs + 0.5)
        ax.set_ylim(0.5, n_pairs + 0.5)
        ax.set_xlabel(f"rank under {name}")
        ax.set_ylabel("rank under Capability")
        ax.set_title(f"{name} vs Capability\nrho={rho:+.3f}, p={p:.3f} ({sig})", fontsize=11)
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_aspect("equal")

    fig.suptitle(f"Rank agreement with Capability ({n}-model pool, {n_pairs} pairs, exact Mantel test)\n"
                 "(each point = one model pair; near the diagonal = rank preserved)",
                 fontsize=13, y=1.05)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
