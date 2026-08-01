"""User's idea (2026-08-01): Ceiling reads domain/expertise, Perplexity
reads fluency/generic quality -- these are DIFFERENT signals, so combining
them (concatenation) might beat either alone. Training-free kNN test,
comparing: Dual (concat) vs Ceiling alone vs Perplexity alone vs uniform.
"""
import sys

import numpy as np
from scipy import stats
from scipy.stats import spearmanr

sys.path.insert(0, "scripts/llmrouterbench")
import common
import loo_recovery as loo


def load_fp(desc_dir, pool):
    E = np.stack([np.load(desc_dir / f"{common.NAME_TO_SAFE[m]}.npy") for m in pool])
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def knn_rhos(E, true_scores, pool):
    sim_full = E @ E.T
    rhos = []
    for i in range(len(pool)):
        others_idx = [j for j in range(len(pool)) if j != i]
        other_scores = true_scores[:, others_idx]
        true_m = true_scores[:, i]
        sims = sim_full[i, others_idx]
        w = np.clip(sims, 0, None)
        if w.sum() < 1e-9:
            w = np.ones_like(w)
        w = w / w.sum()
        fp_proxy = other_scores @ w
        rho, _ = spearmanr(fp_proxy, true_m)
        rhos.append(rho)
    return np.array(rhos)


def main():
    split = loo.load_split()
    _, true_scores = loo.build_setB_eval(split)
    pool = common.MODELS_33

    E_ceiling = load_fp(loo.CEILING_DIR, pool)
    E_perp = load_fp(loo.PERP_DIR, pool)
    E_dual = np.concatenate([E_ceiling, E_perp], axis=1)
    E_dual = E_dual / (np.linalg.norm(E_dual, axis=1, keepdims=True) + 1e-9)

    rho_ceiling = knn_rhos(E_ceiling, true_scores, pool)
    rho_perp = knn_rhos(E_perp, true_scores, pool)
    rho_dual = knn_rhos(E_dual, true_scores, pool)

    print(f"mean rho: Ceiling={rho_ceiling.mean():.4f}  Perplexity={rho_perp.mean():.4f}  "
          f"Dual(concat)={rho_dual.mean():.4f}")

    for name_a, a, name_b, b in [
        ("Dual", rho_dual, "Ceiling", rho_ceiling),
        ("Dual", rho_dual, "Perplexity", rho_perp),
    ]:
        d = a - b
        t, p = stats.ttest_rel(a, b)
        print(f"{name_a} vs {name_b}: delta={d.mean():+.4f}  p={p:.4f}  ({(d>0).sum()}/{len(pool)} improved)")

    print("\nper-model breakdown (Dual vs Ceiling, Dual vs Perplexity):")
    for i, m in enumerate(pool):
        print(f"  {m:38s} Ceiling={rho_ceiling[i]:+.4f}  Perplexity={rho_perp[i]:+.4f}  Dual={rho_dual[i]:+.4f}")


if __name__ == "__main__":
    main()
