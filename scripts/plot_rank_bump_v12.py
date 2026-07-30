"""Single-panel rank bump chart for v1.2 (JSON->MiniLM) vs Capability,
matching the exact visual convention of plot_rank_bump_3way.py (same
SHORT_NAMES, same exact-Mantel test, same panel drawing logic).
"""
import argparse
import json
from itertools import combinations, permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

SHORT_NAMES = {
    "eachadea__vicuna-13b-1.1": "vicuna-13b",
    "chavinlo__alpaca-native": "alpaca-native",
    "TheBloke__koala-7B-HF": "koala-7B",
    "stabilityai__stablelm-tuned-alpha-7b": "stablelm",
    "OpenAssistant__oasst-sft-4-pythia-12b-epoch-3.5": "oasst-pythia-12b",
    "google__flan-t5-xxl": "flan-t5-xxl",
    "THUDM__chatglm-6b": "chatglm-6b",
    "mosesjun0h__llama-7b-hf-baize-lora-bf16": "baize-lora",
    "databricks__dolly-v2-12b": "dolly-v2-12b",
    "fnlp__moss-moon-003-sft": "moss-moon-003-sft",
    "mosaicml__mpt-7b-instruct": "mpt-7b-instruct",
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
    n = M.shape[0]
    return [(SHORT_NAMES[order[i]], SHORT_NAMES[order[j]], M[i, j])
            for i, j in combinations(range(n), 2)]


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


def draw_bump_panel(ax, name_a, name_b, Ma, Mb, order, color="#C4553B"):
    pairs_a = upper_tri_pairs(Ma, order)
    pairs_b = upper_tri_pairs(Mb, order)

    pair_key = lambda p: frozenset((p[0], p[1]))
    sim_a = {pair_key(p): p[2] for p in pairs_a}
    sim_b = {pair_key(p): p[2] for p in pairs_b}
    labels = {pair_key(p): f"{p[0]}-{p[1]}" for p in pairs_a}

    keys = list(sim_a.keys())
    order_a = sorted(keys, key=lambda k: -sim_a[k])
    order_b = sorted(keys, key=lambda k: -sim_b[k])
    rank_a = {k: i for i, k in enumerate(order_a)}
    rank_b = {k: i for i, k in enumerate(order_b)}

    n = len(keys)
    for k in keys:
        ya, yb = rank_a[k], rank_b[k]
        ax.plot([0, 1], [n - 1 - ya, n - 1 - yb], color=color, alpha=0.75, linewidth=1.6, zorder=1)

    for k in keys:
        ax.text(-0.02, n - 1 - rank_a[k], labels[k], ha="right", va="center", fontsize=7)
        ax.text(1.02, n - 1 - rank_b[k], labels[k], ha="left", va="center", fontsize=7)

    rho, p = exact_mantel(Ma, Mb)
    sig = "significant" if p < 0.05 else "not significant"
    ax.set_title(f"{name_a} vs {name_b}\nrho={rho:+.3f}, p={p:.3f} ({sig}, exact Mantel)", fontsize=12)
    ax.set_xlim(-1.1, 2.1)
    ax.set_ylim(-1, n)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"rank under\n{name_a}", f"rank under\n{name_b}"], fontsize=10)
    ax.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    return rho, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="experts/pool-mix-instruct-7.json")
    ap.add_argument("--v12_dir", default="local_descriptors/mix-instruct-v12")
    ap.add_argument("--capability_dir", default="local_descriptors/mix-instruct-capability")
    ap.add_argument("--out", default="local_descriptors/analysis/rank_bump_v12_vs_capability.png")
    args = ap.parse_args()

    pool = json.load(open(args.pool))
    v12_vecs = {m: np.load(f"{args.v12_dir}/{m}.npy") for m in pool}
    cap_vecs = {m: np.load(f"{args.capability_dir}/{m}.npy") for m in pool}

    M_v12 = cosine_sim_matrix(v12_vecs, pool)
    M_cap = cosine_sim_matrix(cap_vecs, pool)

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    rho, p = draw_bump_panel(ax, "v1.2 (JSON->MiniLM)", "CAPABILITY", M_v12, M_cap, pool)

    n_pairs = len(pool) * (len(pool) - 1) // 2
    fig.suptitle(f"Rank of all {n_pairs} model pairs: v1.2 vs Capability(bartscore), {len(pool)}-model pool\n"
                  "(flat/parallel lines = ranks preserved; crossing lines = ranks scrambled)",
                  fontsize=12, y=1.03)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"saved -> {out_path}")
    print(f"rho={rho:+.4f} p={p:.4f}")


if __name__ == "__main__":
    main()
