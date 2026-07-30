"""RSA test: does v1.2 (JSON summary -> MiniLM embedding) track true
capability (bartscore) better than Logit/Perplexity do? Same statistical
convention as rsa_capability_alignment.py -- exact Mantel permutation test
when the pool is small enough to enumerate (<=8!, i.e. pool size <=8),
Monte Carlo otherwise (matches rank_agreement_scalars.py's approach for
larger pools).
"""
import argparse
import json
from itertools import permutations
from math import factorial
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kendalltau


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
    order = np.argsort(-sims)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(sims) + 1)
    return ranks


def mantel_test(M_a, M_b, exact_limit=8, n_perm=20000, seed=0):
    n = M_a.shape[0]
    a_flat = upper_tri(M_a)
    rho_obs, _ = spearmanr(a_flat, upper_tri(M_b))
    if n <= exact_limit:
        count_ge, total = 0, 0
        for perm in permutations(range(n)):
            perm = np.array(perm)
            rho, _ = spearmanr(a_flat, upper_tri(M_b[np.ix_(perm, perm)]))
            if abs(rho) >= abs(rho_obs) - 1e-12:
                count_ge += 1
            total += 1
        p = count_ge / total
        method = f"exact ({factorial(n)} perms)"
    else:
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        count_ge = 0
        for _ in range(n_perm):
            perm = rng.permutation(idx)
            rho, _ = spearmanr(a_flat, upper_tri(M_b[np.ix_(perm, perm)]))
            if abs(rho) >= abs(rho_obs) - 1e-12:
                count_ge += 1
        p = (count_ge + 1) / (n_perm + 1)
        method = f"Monte Carlo ({n_perm} perms)"
    return rho_obs, p, method


def avg_rank_displacement(M_a, M_b):
    disp = np.abs(ranks_of(upper_tri(M_a)) - ranks_of(upper_tri(M_b)))
    return float(disp.mean()), int(disp.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="experts/pool-mix-instruct-7.json")
    ap.add_argument("--v12_dir", default="local_descriptors/mix-instruct-v12")
    ap.add_argument("--capability_dir", default="local_descriptors/mix-instruct-capability")
    ap.add_argument("--logit_dir", default=None,
                     help="optional; only pool members with a .npy here are compared")
    ap.add_argument("--perplexity_dir", default=None)
    args = ap.parse_args()

    pool = json.load(open(args.pool))
    v12_vecs = {m: np.load(f"{args.v12_dir}/{m}.npy") for m in pool}
    cap_vecs = {m: np.load(f"{args.capability_dir}/{m}.npy") for m in pool}

    M_v12 = cosine_sim_matrix(v12_vecs, pool)
    M_cap = cosine_sim_matrix(cap_vecs, pool)

    print(f"Pool ({len(pool)}): {pool}\n")

    rho, p, method = mantel_test(M_v12, M_cap)
    disp, disp_max = avg_rank_displacement(M_v12, M_cap)
    n_pairs = len(pool) * (len(pool) - 1) // 2
    tau, _ = kendalltau(upper_tri(M_v12), upper_tri(M_cap))
    print(f"v1.2 vs Capability: rho={rho:+.4f} p={p:.4f} ({method})")
    print(f"  Kendall tau={tau:+.4f}")
    print(f"  avg rank displacement={disp:.2f}/{n_pairs} (max {disp_max})")

    for name, dirpath in [("Logit", args.logit_dir), ("Perplexity", args.perplexity_dir)]:
        if dirpath is None:
            continue
        try:
            other_vecs = {m: np.load(f"{dirpath}/{m}.npy") for m in pool}
        except FileNotFoundError as e:
            print(f"\n[skip] {name}: {e}")
            continue
        M_other = cosine_sim_matrix(other_vecs, pool)
        rho2, p2, method2 = mantel_test(M_v12, M_other)
        print(f"\nv1.2 vs {name}: rho={rho2:+.4f} p={p2:.4f} ({method2})")
        rho3, p3, method3 = mantel_test(M_other, M_cap)
        print(f"{name} vs Capability: rho={rho3:+.4f} p={p3:.4f} ({method3})")


if __name__ == "__main__":
    main()
