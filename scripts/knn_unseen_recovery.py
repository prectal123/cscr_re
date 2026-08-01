"""Non-parametric (no MLP training, no collapse risk) test of whether FP-space
proximity predicts real capability for an unseen/held-out model.

For each FP type (Perplexity, Ceiling, V1.2) and each held-out model M
(leave-one-out over the 11-model pool):
  1. Compute M's cosine similarity, in FP space, to the other 10 (seen) models.
  2. Build a per-prompt PROXY score for M on Set B (held-out prompts) as the
     similarity-weighted average of the other 10 models' REAL bartscore on
     that prompt: proxy(p) = sum_i max(0, sim(M, i)) * bartscore_i(p) / sum_i max(0, sim(M, i))
     -- i.e. "guess M's score on this prompt using only its FP neighbors,
     never looking at M's own performance."
  3. Compare proxy(p) to M's TRUE bartscore(p) across Set B via Spearman
     correlation. High correlation = FP-space neighbors are informative about
     M's real per-prompt capability, WITHOUT any gradient-descent training
     (so it can't collapse the way the contrastive query encoder does).
  4. Baseline: uniform-weighted proxy (average of the other 10, ignoring FP
     entirely) -- if the FP-weighted proxy doesn't beat this, the FP isn't
     adding information beyond "models are generally correlated with each
     other."

This directly operationalizes "does FP-capability alignment help unseen-model
routing" without touching the collapse-prone contrastive training loop.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from router.utils import load_descriptors
import loo_unseen_recovery as base

OUT_DIR = Path("local_descriptors/analysis")


def main():
    pool_11 = json.load(open(base.POOL_PATH))
    print("Building Set B eval subsample (ground truth, no training involved)...")
    ids, per_prompt = base.load_set_b(pool_11)

    # (n_prompts, n_models) matrix of REAL exp(bartscore), columns in pool_11 order
    true_scores = np.array([[math.exp(per_prompt[pid]["scores"][m]) for m in pool_11] for pid in ids])
    name_to_col = {m: i for i, m in enumerate(pool_11)}

    results = {}
    for fp_name, desc_dir in [("Perplexity", base.PERP_DIR), ("Ceiling", base.CEILING_DIR), ("V1.2", base.V12_DIR)]:
        print(f"\n{'='*60}\nFP type: {fp_name}\n{'='*60}")
        E, desc_names = load_descriptors(str(desc_dir), pool=pool_11)
        E = np.stack(E)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        name_to_row = {n: i for i, n in enumerate(desc_names)}
        sim_full = E @ E.T  # (11, 11) cosine similarity in FP space

        fp_rhos, uniform_rhos = [], []
        for held_out in pool_11:
            others = [m for m in pool_11 if m != held_out]
            m_row = name_to_row[held_out]
            sims = np.array([sim_full[m_row, name_to_row[o]] for o in others])
            w = np.clip(sims, 0, None)
            if w.sum() < 1e-9:
                w = np.ones_like(w)  # fall back to uniform if all similarities are non-positive
            w = w / w.sum()

            other_cols = [name_to_col[o] for o in others]
            held_col = name_to_col[held_out]
            other_scores = true_scores[:, other_cols]        # (n_prompts, 10)
            true_m = true_scores[:, held_col]                 # (n_prompts,)

            fp_proxy = other_scores @ w                        # FP-similarity-weighted proxy
            uniform_proxy = other_scores.mean(axis=1)          # ignores FP entirely

            fp_rho, _ = spearmanr(fp_proxy, true_m)
            uni_rho, _ = spearmanr(uniform_proxy, true_m)
            fp_rhos.append(fp_rho)
            uniform_rhos.append(uni_rho)
            print(f"  held out {held_out:50s} FP-proxy rho={fp_rho:.4f}  uniform-proxy rho={uni_rho:.4f}")

        fp_rhos, uniform_rhos = np.array(fp_rhos), np.array(uniform_rhos)
        print(f"\n--- {fp_name} summary ---")
        print(f"mean FP-weighted proxy rho:      {fp_rhos.mean():.4f} (std {fp_rhos.std():.4f})")
        print(f"mean uniform (no-FP) proxy rho:  {uniform_rhos.mean():.4f} (std {uniform_rhos.std():.4f})")
        print(f"mean improvement over uniform:   {(fp_rhos - uniform_rhos).mean():+.4f}  "
              f"(positive in {int((fp_rhos > uniform_rhos).sum())}/11 folds)")

        results[fp_name] = {
            "held_out_models": pool_11,
            "fp_weighted_rho": fp_rhos.tolist(),
            "uniform_rho": uniform_rhos.tolist(),
            "mean_fp_weighted_rho": float(fp_rhos.mean()),
            "mean_uniform_rho": float(uniform_rhos.mean()),
            "mean_improvement": float((fp_rhos - uniform_rhos).mean()),
            "n_folds_improved": int((fp_rhos > uniform_rhos).sum()),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "knn_unseen_recovery_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}")
    print(f"{'FP':12s} {'mean FP rho':>12s} {'mean uniform rho':>18s} {'mean improvement':>18s} {'folds improved':>15s}")
    for fp_name, r in results.items():
        print(f"{fp_name:12s} {r['mean_fp_weighted_rho']:12.4f} {r['mean_uniform_rho']:18.4f} "
              f"{r['mean_improvement']:+18.4f} {r['n_folds_improved']:>13d}/11")
    print(f"\nSaved -> {OUT_DIR / 'knn_unseen_recovery_results.json'}")


if __name__ == "__main__":
    main()
