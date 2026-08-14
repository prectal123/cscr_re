"""Training-free kNN unseen-recovery test for V1.3 (LLM-judge, category-rate,
22-dim) against Ceiling V2 (category-rate) and Perplexity -- same design as
knn_test_lite20.py (FP-weighted proxy vs uniform, Spearman rho, paired t-test
across the 20 LOO folds). Self-contained (no torch import): loo_recovery_lite20.py
pulls in torch at module level for its training functions, which this
training-free test doesn't need, so load_split/build_setB_eval are reimplemented
here directly instead of importing that module.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import spearmanr

sys.path.insert(0, "scripts/llmrouterbench")
import common_lite20 as common

DATA_DIR = Path("local_descriptors/llmrouterbench_lite20")
V1_3_DIR = DATA_DIR / "v1_3"
CATRATE_DIR = DATA_DIR / "ceiling_categoryrate"
PERP_DIR = DATA_DIR / "perplexity"


def load_split():
    with open(DATA_DIR / "setA_setB_split.pkl", "rb") as f:
        return pickle.load(f)


def build_setB_eval(split):
    queries, scores = [], []
    for ds in common.DATASETS:
        d = split["setB"][ds]
        queries.extend(d["queries"])
        scores.append(d["scores"])
    return queries, np.concatenate(scores, axis=0)


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
    split = load_split()
    _, true_scores = build_setB_eval(split)
    pool = common.MODELS_20
    print(f"Set B total: {true_scores.shape[0]} rows x {len(pool)} models", flush=True)

    print(f"\n{'='*70}\nV1.3 (LLM-judge, category-rate)\n{'='*70}", flush=True)
    v13_rhos, uniform_rhos = test(true_scores, V1_3_DIR, "V1.3", pool)

    print(f"\n{'='*70}\nCeiling V2 (category-rate)\n{'='*70}", flush=True)
    v2_rhos, _ = test(true_scores, CATRATE_DIR, "CeilingV2", pool)

    print(f"\n{'='*70}\nPerplexity FP\n{'='*70}", flush=True)
    perp_rhos, _ = test(true_scores, PERP_DIR, "Perplexity", pool)

    for name_a, rhos_a, name_b, rhos_b in [
        ("V1.3", v13_rhos, "uniform", uniform_rhos),
        ("V1.3", v13_rhos, "CeilingV2", v2_rhos),
        ("V1.3", v13_rhos, "Perplexity", perp_rhos),
        ("CeilingV2", v2_rhos, "Perplexity", perp_rhos),
    ]:
        delta = rhos_a - rhos_b
        t, p = stats.ttest_rel(rhos_a, rhos_b)
        print(f"\n{'='*70}\n{name_a} vs {name_b} (paired)\n{'='*70}", flush=True)
        print(f"mean {name_a} rho={rhos_a.mean():.4f}  mean {name_b} rho={rhos_b.mean():.4f}  "
              f"delta={delta.mean():+.4f}  p={p:.4f}  ({(delta>0).sum()}/{len(pool)} improved)", flush=True)


if __name__ == "__main__":
    main()
