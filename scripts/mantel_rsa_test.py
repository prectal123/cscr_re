"""Mantel/RSA structure test for Ceiling FP on RouterBench -- extends the
MixInstruct RSA test (PROGRESS.md section 11) to check whether Ceiling FP's
FULL 11x11 similarity structure (not just per-model kNN prediction quality)
correlates with the TRUE capability correlation structure between models.

This is training-free (no GPU) and tests something different from the kNN
tests already run: not "can I predict model M's correctness from its
neighbors" but "does the descriptor space's overall shape match the true
capability-similarity shape between all pairs of models".

Method: build both 11x11 matrices (Ceiling FP cosine sim, true capability
Pearson correlation -- same as true_capability_beacon_check.py), flatten the
off-diagonal upper triangle of each, compute Spearman rho between them, and
get an exact-permutation Mantel test p-value (11! is too large, use random
permutations of the label ordering instead, which is the standard way to run
a Mantel test at this N).
"""
import numpy as np
from scipy.stats import spearmanr

import routerbench_knn_test as rb

NAMES = rb.NAMES
MODELS = rb.MODELS
CEILING_DIR = rb.CEILING_DIR
PERP_DIR_NAME = "local_descriptors/routerbench-perplexity"
N_PERM = 100000


def upper_tri(mat):
    n = mat.shape[0]
    idx = np.triu_indices(n, k=1)
    return mat[idx]


def mantel_test(mat_a, mat_b, n_perm=N_PERM, seed=0):
    n = mat_a.shape[0]
    flat_a = upper_tri(mat_a)
    flat_b_orig = upper_tri(mat_b)
    rho_obs, _ = spearmanr(flat_a, flat_b_orig)

    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        mat_b_perm = mat_b[np.ix_(perm, perm)]
        flat_b_perm = upper_tri(mat_b_perm)
        rho_perm, _ = spearmanr(flat_a, flat_b_perm)
        if abs(rho_perm) >= abs(rho_obs):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return rho_obs, p


def build_true_capability_matrix(set_a):
    correctness = np.stack([set_a[m].to_numpy(dtype=float) for m in MODELS], axis=0)
    correctness = (correctness >= 1.0).astype(float)
    n = len(NAMES)
    true_corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            true_corr[i, j] = 1.0 if i == j else np.corrcoef(correctness[i], correctness[j])[0, 1]
    return true_corr


def build_fp_sim_matrix(desc_dir):
    E = np.stack([np.load(f"{desc_dir}/{n}.npy") for n in NAMES])
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    return E @ E.T


def main():
    set_a, set_b = rb.load_data()
    true_corr = build_true_capability_matrix(set_a)

    for fp_name, desc_dir in [("Ceiling", str(CEILING_DIR)), ("Perplexity", PERP_DIR_NAME)]:
        sim = build_fp_sim_matrix(desc_dir)
        rho, p = mantel_test(sim, true_corr)
        print(f"{fp_name:12s} FP-space vs true-capability structure: "
              f"Spearman rho={rho:+.4f}  Mantel p={p:.5f} ({N_PERM} permutations)")


if __name__ == "__main__":
    main()
