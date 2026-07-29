"""3-panel rank bump chart, matching the style of the existing
gap_rank_bump.png (logit vs perplexity), extended to all 3 pairwise
comparisons among {Logit, Perplexity, Capability}, laid out side by
side in one figure: left=Logit/Perplexity, middle=Perplexity/Capability,
right=Logit/Capability.
"""
import json
from itertools import combinations, permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

POOL_PATH = "experts/pool-mix-instruct-7.json"
LOGIT_DIR = Path("local_descriptors/mix-instruct-logit")
PERP_DIR = Path("local_descriptors/mix-instruct-perplexity")
CAP_DIR = Path("local_descriptors/mix-instruct-capability")
OUT_PATH = Path("local_descriptors/analysis/rank_bump_3way.png")

# Matches the abbreviations already used in gap_rank_bump.png
SHORT_NAMES = {
    "eachadea__vicuna-13b-1.1": "vicuna-13b",
    "chavinlo__alpaca-native": "alpaca-native",
    "TheBloke__koala-7B-HF": "koala-7B",
    "stabilityai__stablelm-tuned-alpha-7b": "stablelm",
    "OpenAssistant__oasst-sft-4-pythia-12b-epoch-3.5": "oasst-pythia-12b",
    "google__flan-t5-xxl": "flan-t5-xxl",
    "THUDM__chatglm-6b": "chatglm-6b",
}


def cosine_sim_matrix(vecs: dict, order: list) -> np.ndarray:
    n = len(order)
    M = np.zeros((n, n))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            va, vb = vecs[a], vecs[b]
            M[i, j] = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
    return M


def upper_tri_pairs(M: np.ndarray, order: list):
    """Return list of (label_i, label_j, sim) for i<j."""
    n = M.shape[0]
    out = []
    for i, j in combinations(range(n), 2):
        out.append((SHORT_NAMES[order[i]], SHORT_NAMES[order[j]], M[i, j]))
    return out


def exact_mantel(A: np.ndarray, B: np.ndarray):
    n = A.shape[0]
    idx = np.triu_indices(n, k=1)
    a_flat = A[idx]
    observed, _ = spearmanr(a_flat, B[idx])
    perm_rhos = []
    for perm in permutations(range(n)):
        perm = np.array(perm)
        B_perm = B[np.ix_(perm, perm)]
        rho, _ = spearmanr(a_flat, B_perm[idx])
        perm_rhos.append(rho)
    perm_rhos = np.array(perm_rhos)
    p = float(np.mean(np.abs(perm_rhos) >= abs(observed) - 1e-12))
    return observed, p


def draw_bump_panel(ax, name_a, name_b, Ma, Mb, order, color="#6FA8DC"):
    pairs_a = upper_tri_pairs(Ma, order)
    pairs_b = upper_tri_pairs(Mb, order)

    # each pair identified by frozenset of its two short names
    pair_key = lambda p: frozenset((p[0], p[1]))
    sim_a = {pair_key(p): p[2] for p in pairs_a}
    sim_b = {pair_key(p): p[2] for p in pairs_b}
    labels = {pair_key(p): f"{p[0]}-{p[1]}" for p in pairs_a}

    keys = list(sim_a.keys())
    # rank 1 = highest similarity
    order_a = sorted(keys, key=lambda k: -sim_a[k])
    order_b = sorted(keys, key=lambda k: -sim_b[k])
    rank_a = {k: i for i, k in enumerate(order_a)}
    rank_b = {k: i for i, k in enumerate(order_b)}

    n = len(keys)
    for k in keys:
        ya, yb = rank_a[k], rank_b[k]
        ax.plot([0, 1], [n - 1 - ya, n - 1 - yb], color=color, alpha=0.75, linewidth=1.4, zorder=1)

    for k in keys:
        ax.text(-0.02, n - 1 - rank_a[k], labels[k], ha="right", va="center", fontsize=6.3)
        ax.text(1.02, n - 1 - rank_b[k], labels[k], ha="left", va="center", fontsize=6.3)

    rho, p = exact_mantel(Ma, Mb)
    sig = "significant" if p < 0.05 else "not significant"
    ax.set_title(f"{name_a} vs {name_b}\nrho={rho:+.3f}, p={p:.3f} ({sig})", fontsize=11)
    ax.set_xlim(-0.9, 1.9)
    ax.set_ylim(-1, n)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"rank under\n{name_a}", f"rank under\n{name_b}"], fontsize=9)
    ax.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    return rho, p


def main():
    pool = json.load(open(POOL_PATH))
    order = pool

    logit_vecs = {m: np.load(LOGIT_DIR / f"{m}.npy") for m in pool}
    perp_vecs = {m: np.load(PERP_DIR / f"{m}.npy") for m in pool}
    cap_vecs = {m: np.load(CAP_DIR / f"{m}.npy") for m in pool}

    M_logit = cosine_sim_matrix(logit_vecs, order)
    M_perp = cosine_sim_matrix(perp_vecs, order)
    M_cap = cosine_sim_matrix(cap_vecs, order)

    # Same colors as plot_rsa_scatter.py, keyed to the same comparison
    # (not panel position), so a given pair reads as the same color
    # across both chart types.
    panels = [
        ("LOGIT", "PERPLEXITY", M_logit, M_perp, "#3B82C4"),      # blue
        ("PERPLEXITY", "CAPABILITY", M_perp, M_cap, "#6FAE5C"),   # green
        ("LOGIT", "CAPABILITY", M_logit, M_cap, "#E07B39"),       # orange
    ]

    fig, axes = plt.subplots(1, 3, figsize=(27, 13))
    for ax, (name_a, name_b, Ma, Mb, color) in zip(axes, panels):
        draw_bump_panel(ax, name_a, name_b, Ma, Mb, order, color=color)

    fig.suptitle("Rank of all 21 model pairs across representations "
                  "(flat/parallel lines = ranks preserved; crossing lines = ranks scrambled)",
                  fontsize=14, y=1.02)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
