"""Rank-based scalar summaries of FP disagreement (per feedback: rank
order matters more than raw cosine similarity values).

1. Kendall's W (coefficient of concordance): a single 0-1 number
   summarizing how much all 3 rankings (of the 21 model pairs, by
   similarity under Logit / Perplexity / Capability) agree with each
   other simultaneously. 1 = perfect agreement, 0 = no agreement.
   Significance via Monte Carlo permutation (n=21 items is too large
   for exact enumeration, unlike the 7-item Mantel tests elsewhere).

2. Average rank displacement: for each pairwise comparison, how many
   positions (out of 21) does a model-pair's rank shift on average
   when switching representations. Purely intuitive, no stats jargon.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np

POOL_PATH = "experts/pool-mix-instruct-7.json"
LOGIT_DIR = Path("local_descriptors/mix-instruct-logit")
PERP_DIR = Path("local_descriptors/mix-instruct-perplexity")
CAP_DIR = Path("local_descriptors/mix-instruct-capability")
N_PERM = 10000
RNG_SEED = 0


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
    """Rank 1 = highest similarity. Returns rank (1..n) per item."""
    order = np.argsort(-sims)  # descending
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(sims) + 1)
    return ranks


def kendalls_w(rank_matrix: np.ndarray) -> float:
    """rank_matrix: shape (m_rankers, n_items), each row a 1..n ranking."""
    m, n = rank_matrix.shape
    R = rank_matrix.sum(axis=0)  # summed rank per item across rankers
    mean_R = m * (n + 1) / 2
    S = np.sum((R - mean_R) ** 2)
    W = 12 * S / (m ** 2 * (n ** 3 - n))
    return float(W)


def main():
    pool = json.load(open(POOL_PATH))
    order = pool

    logit_vecs = {m: np.load(LOGIT_DIR / f"{m}.npy") for m in pool}
    perp_vecs = {m: np.load(PERP_DIR / f"{m}.npy") for m in pool}
    cap_vecs = {m: np.load(CAP_DIR / f"{m}.npy") for m in pool}

    M_logit = cosine_sim_matrix(logit_vecs, order)
    M_perp = cosine_sim_matrix(perp_vecs, order)
    M_cap = cosine_sim_matrix(cap_vecs, order)

    sim_logit = upper_tri(M_logit)
    sim_perp = upper_tri(M_perp)
    sim_cap = upper_tri(M_cap)
    n_pairs = len(sim_logit)  # 21

    rank_logit = ranks_of(sim_logit)
    rank_perp = ranks_of(sim_perp)
    rank_cap = ranks_of(sim_cap)

    # ---------- 1. Kendall's W across all 3 rankings at once ----------
    rank_matrix = np.stack([rank_logit, rank_perp, rank_cap])  # (3, 21)
    W_obs = kendalls_w(rank_matrix)

    rng = np.random.default_rng(RNG_SEED)
    perm_Ws = []
    for _ in range(N_PERM):
        shuffled = np.stack([rng.permutation(rank_logit),
                              rng.permutation(rank_perp),
                              rng.permutation(rank_cap)])
        perm_Ws.append(kendalls_w(shuffled))
    perm_Ws = np.array(perm_Ws)
    p_W = float(np.mean(perm_Ws >= W_obs))

    print("=== Kendall's W: overall concordance across Logit/Perplexity/Capability rankings ===")
    print(f"n items (model pairs) = {n_pairs}, m rankings = 3")
    print(f"W = {W_obs:.4f}  (0 = no agreement, 1 = perfect agreement)")
    print(f"Monte Carlo p (n={N_PERM} random permutations) = {p_W:.4f}  "
          f"({'significant' if p_W < 0.05 else 'not significant'})")

    # ---------- 2. Average rank displacement, per pairwise comparison ----------
    print("\n=== Average rank displacement (out of 21 possible positions) ===")
    pairwise = [
        ("Logit", "Perplexity", rank_logit, rank_perp),
        ("Logit", "Capability", rank_logit, rank_cap),
        ("Perplexity", "Capability", rank_perp, rank_cap),
    ]
    for name_a, name_b, ra, rb in pairwise:
        disp = np.abs(ra - rb)
        print(f"{name_a:12s} vs {name_b:12s}  avg displacement = {disp.mean():.2f} / {n_pairs} "
              f"(max observed shift: {disp.max()})")

    # reference: expected average displacement under pure random pairing
    rng2 = np.random.default_rng(1)
    rand_disps = []
    base = np.arange(1, n_pairs + 1)
    for _ in range(N_PERM):
        rand_disps.append(np.abs(base - rng2.permutation(base)).mean())
    print(f"\n(reference: average displacement under pure random ranking = "
          f"{np.mean(rand_disps):.2f} / {n_pairs})")


if __name__ == "__main__":
    main()
