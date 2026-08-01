"""kNN test for the PC1-deflated Ceiling FP (domain-purified residual, PC1
'generic strength' axis removed) vs uniform and vs the original (undeflated)
Ceiling FP, lightweight-20 pool.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import spearmanr

sys.path.insert(0, "scripts/llmrouterbench")
import common_lite20 as common
import loo_recovery_lite20 as loo

DEFLATED_DIR = Path("local_descriptors/llmrouterbench_lite20/ceiling_deflated")


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

    print(f"\n{'='*70}\nCeiling FP (deflated, PC1 removed)\n{'='*70}", flush=True)
    deflated_rhos, uniform_rhos = test(true_scores, DEFLATED_DIR, "Ceiling-deflated", pool)

    print(f"\n{'='*70}\nCeiling FP (original, undeflated)\n{'='*70}", flush=True)
    orig_rhos, _ = test(true_scores, loo.CEILING_DIR, "Ceiling-original", pool)

    print(f"\n{'='*70}\nDeflated vs Original Ceiling (direct paired comparison)\n{'='*70}", flush=True)
    delta = deflated_rhos - orig_rhos
    t, p = stats.ttest_rel(deflated_rhos, orig_rhos)
    print(f"mean deflated rho={deflated_rhos.mean():.4f}  mean original rho={orig_rhos.mean():.4f}  "
          f"delta={delta.mean():+.4f}  p={p:.4f}  ({(delta>0).sum()}/{len(pool)} improved)", flush=True)
    for m, d in zip(pool, delta):
        print(f"  {m:38s} delta={d:+.4f}", flush=True)


if __name__ == "__main__":
    main()
