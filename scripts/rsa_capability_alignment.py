"""RSA (Representational Similarity Analysis): does the 7x7 pairwise-
similarity structure of Logit/Perplexity descriptors agree with the
structure of actual capability (bartscore-based)?

Computes 3 pairwise RSA comparisons among {Logit, Perplexity, Capability}:
Spearman correlation between the two similarity matrices' upper-triangle
entries, tested with an EXACT Mantel permutation test (all 7! = 5040
relabelings), matching the methodology already used for the earlier
Logit-vs-Perplexity RSA result (rho=-0.079, p=0.762) so results are
directly comparable.
"""
import json
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

POOL_PATH = "experts/pool-mix-instruct-7.json"
LOGIT_DIR = Path("local_descriptors/mix-instruct-logit")
PERP_DIR = Path("local_descriptors/mix-instruct-perplexity")
CAP_DIR = Path("local_descriptors/mix-instruct-capability")


def cosine_sim_matrix(vecs: dict, order: list) -> np.ndarray:
    n = len(order)
    M = np.zeros((n, n))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            va, vb = vecs[a], vecs[b]
            M[i, j] = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
    return M


def upper_tri(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    idx = np.triu_indices(n, k=1)
    return M[idx]


def exact_mantel(A: np.ndarray, B: np.ndarray):
    """Exact permutation test: relabel B's rows/cols with every permutation
    of {0..n-1}, recompute Spearman rho each time, compare to observed."""
    n = A.shape[0]
    a_flat = upper_tri(A)
    observed, _ = spearmanr(a_flat, upper_tri(B))

    perm_rhos = []
    for perm in permutations(range(n)):
        perm = np.array(perm)
        B_perm = B[np.ix_(perm, perm)]
        rho, _ = spearmanr(a_flat, upper_tri(B_perm))
        perm_rhos.append(rho)
    perm_rhos = np.array(perm_rhos)

    p_two_sided = float(np.mean(np.abs(perm_rhos) >= abs(observed) - 1e-12))
    return observed, p_two_sided, len(perm_rhos)


def main():
    pool = json.load(open(POOL_PATH))
    print(f"Pool ({len(pool)}): {pool}\n")

    logit_vecs = {m: np.load(LOGIT_DIR / f"{m}.npy") for m in pool}
    perp_vecs = {m: np.load(PERP_DIR / f"{m}.npy") for m in pool}
    cap_vecs = {m: np.load(CAP_DIR / f"{m}.npy") for m in pool}

    order = pool  # fixed label order shared across all three matrices
    M_logit = cosine_sim_matrix(logit_vecs, order)
    M_perp = cosine_sim_matrix(perp_vecs, order)
    M_cap = cosine_sim_matrix(cap_vecs, order)

    print("=== Similarity matrices built (7x7 cosine similarity) ===")
    for name, M in [("Logit", M_logit), ("Perplexity", M_perp), ("Capability", M_cap)]:
        u = upper_tri(M)
        print(f"{name:12s}: pairwise sim range [{u.min():.4f}, {u.max():.4f}], mean={u.mean():.4f}")
    print()

    comparisons = [
        ("Logit", "Perplexity", M_logit, M_perp),
        ("Logit", "Capability", M_logit, M_cap),
        ("Perplexity", "Capability", M_perp, M_cap),
    ]

    print("=== RSA results (exact Mantel test, 7! = 5040 permutations) ===")
    for name_a, name_b, Ma, Mb in comparisons:
        rho, p, n_perm = exact_mantel(Ma, Mb)
        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"{name_a:12s} vs {name_b:12s}  rho={rho:+.4f}   p={p:.4f}  ({sig}, n_perm={n_perm})")

    print("\n=== Interpretation ===")
    print("rho near 0 / p > 0.05 => the two similarity structures are")
    print("statistically independent -- 'which experts are similar to which'")
    print("disagrees between the two representations being compared.")


if __name__ == "__main__":
    main()
