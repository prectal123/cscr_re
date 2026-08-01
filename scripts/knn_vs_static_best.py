"""Extends the training-free kNN test (routerbench_knn_test.py): does the
FP-weighted proxy beat not just a uniform-average baseline, but a STATIC-BEST
baseline (always use the single most-accurate-on-Set-A model's actual Set B
score as the predictor, ignoring the query entirely)? This is the
training-free equivalent of "does the router beat pure collapse" -- static-
best behaves exactly like a fully collapsed router (always the same pick).

For held-out model M: fp_proxy = FP-similarity-weighted average of the other
10 models' true Set B scores (as before). static_best_proxy = the single
other model with highest Set A accuracy's actual Set B score, used AS-IS
(not averaged with anything) as the predictor for every prompt.
"""
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr

import routerbench_knn_test as rb

NAMES = rb.NAMES
MODELS = rb.MODELS
CEILING_DIR = rb.CEILING_DIR
PERP_DIR = Path("local_descriptors/routerbench-perplexity")


def test(set_a, set_b, desc_dir, fp_name):
    print(f"\n{'='*70}\n{fp_name} FP -- kNN vs static-best\n{'='*70}")
    E = np.stack([np.load(desc_dir / f"{n}.npy") for n in NAMES])
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    sim_full = E @ E.T

    # NOTE: keep raw continuous (partial-credit) scores here, matching the
    # convention already used in routerbench_knn_test.py's knn_test() -- an
    # earlier version of this script mistakenly binarized via >=1.0, which
    # silently changed the measurement definition and made the "FP vs
    # uniform" numbers incomparable to the already-reported +0.098/p=0.02
    # result. Binarizing also throws away information that Spearman rho
    # (rank-based, doesn't need binary labels) can use directly.
    true_scores_a = np.stack([set_a[m].to_numpy(dtype=float) for m in MODELS], axis=1)
    true_scores_b = np.stack([set_b[m].to_numpy(dtype=float) for m in MODELS], axis=1)

    fp_rhos, uniform_rhos, static_rhos = [], [], []
    for i, held_out in enumerate(NAMES):
        others_idx = [j for j in range(len(NAMES)) if j != i]
        other_scores_b = true_scores_b[:, others_idx]
        true_m = true_scores_b[:, i]

        sims = sim_full[i, others_idx]
        w = np.clip(sims, 0, None)
        if w.sum() < 1e-9:
            w = np.ones_like(w)
        w = w / w.sum()
        fp_proxy = other_scores_b @ w

        uniform_proxy = other_scores_b.mean(axis=1)

        # static_best redefined per user's request: among the 10 candidates,
        # whichever SINGLE model's own raw score achieves the best rho with
        # the held-out model's true score (an oracle-style best-of-10 search
        # over "route to just one fixed model" policies) -- not the model
        # that's merely generically strongest on Set A.
        per_candidate_rhos = [spearmanr(other_scores_b[:, k], true_m)[0] for k in range(other_scores_b.shape[1])]
        static_best_local = int(np.argmax(per_candidate_rhos))
        static_proxy = other_scores_b[:, static_best_local]

        fp_rho, _ = spearmanr(fp_proxy, true_m)
        uni_rho, _ = spearmanr(uniform_proxy, true_m)
        static_rho = per_candidate_rhos[static_best_local]
        fp_rhos.append(fp_rho)
        uniform_rhos.append(uni_rho)
        static_rhos.append(static_rho)
        static_best_name = NAMES[others_idx[static_best_local]]
        print(f"  held out {held_out:35s} FP rho={fp_rho:+.4f}  uniform rho={uni_rho:+.4f}  "
              f"static_best rho={static_rho:+.4f} (best-single={static_best_name})")

    fp_rhos, uniform_rhos, static_rhos = np.array(fp_rhos), np.array(uniform_rhos), np.array(static_rhos)

    d_uni = fp_rhos - uniform_rhos
    t_uni, p_uni = stats.ttest_rel(fp_rhos, uniform_rhos)
    d_static = fp_rhos - static_rhos
    t_static, p_static = stats.ttest_rel(fp_rhos, static_rhos)

    print(f"\n  mean FP rho={fp_rhos.mean():.4f}  mean uniform rho={uniform_rhos.mean():.4f}  "
          f"mean static_best rho={static_rhos.mean():.4f}")
    print(f"  FP vs uniform:     delta={d_uni.mean():+.4f}  p={p_uni:.4f}  ({(d_uni>0).sum()}/11 improved)")
    print(f"  FP vs static_best: delta={d_static.mean():+.4f}  p={p_static:.4f}  ({(d_static>0).sum()}/11 improved)")

    return {
        "mean_fp_rho": float(fp_rhos.mean()), "mean_uniform_rho": float(uniform_rhos.mean()),
        "mean_static_rho": float(static_rhos.mean()),
        "delta_vs_uniform": float(d_uni.mean()), "p_vs_uniform": float(p_uni),
        "delta_vs_static": float(d_static.mean()), "p_vs_static": float(p_static),
    }


def main():
    set_a, set_b = rb.load_data()
    results = {}
    results["Ceiling"] = test(set_a, set_b, CEILING_DIR, "Ceiling")
    results["Perplexity"] = test(set_a, set_b, PERP_DIR, "Perplexity")

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for fp_name, r in results.items():
        print(f"{fp_name:12s} FP vs static_best delta={r['delta_vs_static']:+.4f}  p={r['p_vs_static']:.4f}")


if __name__ == "__main__":
    main()
