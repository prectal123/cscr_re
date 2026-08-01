"""Training-free kNN unseen-recovery test for the lightweight-20 pool, 22
datasets. Same design as knn_test.py (FP-weighted proxy vs uniform, Spearman
rho, paired t-test across the 20 LOO folds), both FPs + direct comparison.
"""
import sys

import numpy as np
from scipy import stats
from scipy.stats import spearmanr

sys.path.insert(0, "scripts/llmrouterbench")
import common_lite20 as common
import loo_recovery_lite20 as loo


def test(true_scores, desc_dir, fp_name, pool):
    E = np.stack([np.load(desc_dir / f"{common.NAME_TO_SAFE[m]}.npy") for m in pool])
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    sim_full = E @ E.T

    fp_rhos, uniform_rhos = [], []
    for i, held_out in enumerate(pool):
        others_idx = [j for j in range(len(pool)) if j != i]
        other_scores = true_scores[:, others_idx]
        true_m = true_scores[:, i]

        sims = sim_full[i, others_idx]
        w = np.clip(sims, 0, None)
        if w.sum() < 1e-9:
            w = np.ones_like(w)
        w = w / w.sum()
        fp_proxy = other_scores @ w
        uniform_proxy = other_scores.mean(axis=1)

        fp_rho, _ = spearmanr(fp_proxy, true_m)
        uni_rho, _ = spearmanr(uniform_proxy, true_m)
        fp_rhos.append(fp_rho)
        uniform_rhos.append(uni_rho)
        print(f"  held out {held_out:35s} FP rho={fp_rho:+.4f}  uniform rho={uni_rho:+.4f}", flush=True)

    fp_rhos, uniform_rhos = np.array(fp_rhos), np.array(uniform_rhos)
    delta = fp_rhos - uniform_rhos
    t, p = stats.ttest_rel(fp_rhos, uniform_rhos)
    print(f"\n  mean FP rho={fp_rhos.mean():.4f}  mean uniform rho={uniform_rhos.mean():.4f}  "
          f"delta={delta.mean():+.4f}  p={p:.4f}  ({(delta>0).sum()}/{len(pool)} improved)", flush=True)
    return fp_rhos, uniform_rhos


def main():
    split = loo.load_split()
    _, true_scores = loo.build_setB_eval(split)
    pool = common.MODELS_20
    print(f"Set B total: {true_scores.shape[0]} rows x {len(pool)} models", flush=True)

    print(f"\n{'='*70}\nCeiling FP\n{'='*70}", flush=True)
    ceiling_rhos, _ = test(true_scores, loo.CEILING_DIR, "Ceiling", pool)

    print(f"\n{'='*70}\nPerplexity FP\n{'='*70}", flush=True)
    perp_rhos, _ = test(true_scores, loo.PERP_DIR, "Perplexity", pool)

    print(f"\n{'='*70}\nCeiling vs Perplexity (direct paired comparison)\n{'='*70}", flush=True)
    delta = ceiling_rhos - perp_rhos
    t, p = stats.ttest_rel(ceiling_rhos, perp_rhos)
    print(f"mean Ceiling rho={ceiling_rhos.mean():.4f}  mean Perplexity rho={perp_rhos.mean():.4f}  "
          f"delta={delta.mean():+.4f}  p={p:.4f}  ({(delta>0).sum()}/{len(pool)} improved)", flush=True)
    for m, d in zip(pool, delta):
        print(f"  {m:38s} delta={d:+.4f}", flush=True)


if __name__ == "__main__":
    main()
