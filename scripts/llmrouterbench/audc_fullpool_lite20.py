"""Cost-accuracy tradeoff (AUDC) on the lightweight-20 pool -- full-pool training
(no leave-one-out), Ceiling FP vs Perplexity. Reuses the existing ProjHead +
cost_info_nce training pipeline (loo_recovery_lite20.train_fold) with pool=all
20 models, then sweeps a cost-penalty lambda at INFERENCE time:
    chosen(i, lam) = argmax_m [ sim(q_i, e_m) - lam * cost_norm[m] ]
For each lambda, records (avg cost of chosen models, accuracy on Set B).
Sorting by avg cost and trapezoidal-integrating gives AUDC.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts/llmrouterbench")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import common_lite20 as common
import loo_recovery_lite20 as loo
import loo_unseen_recovery as base
from transformers import AutoModel, AutoTokenizer

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
BETA = 1.0
LAMBDAS = np.concatenate([[0.0], np.geomspace(0.001, 50.0, 40)])
OUT_DIR = Path("local_descriptors/llmrouterbench_lite20")
OUT_PATH = OUT_DIR / f"audc_fullpool_results_seed{SEED}.json"


def audc_curve(sims, cost_norm, true_scores_col, lambdas):
    """sims: (N, M) cosine sims. cost_norm: (M,). true_scores_col: (N, M) binary correct."""
    points = []
    for lam in lambdas:
        scores = sims - lam * cost_norm[None, :]
        chosen = scores.argmax(axis=1)
        acc = float((true_scores_col[np.arange(len(chosen)), chosen] >= 1.0).mean())
        avg_cost = float(cost_norm[chosen].mean())
        points.append((avg_cost, acc, float(lam)))
    return points


def trapz_audc(points):
    pts = sorted(set((round(c, 6), a) for c, a, _ in points))
    costs = np.array([p[0] for p in pts])
    accs = np.array([p[1] for p in pts])
    # normalize cost axis to [0,1] over the observed range so AUDC is comparable across FPs
    if costs.max() > costs.min():
        costs_n = (costs - costs.min()) / (costs.max() - costs.min())
    else:
        costs_n = costs * 0.0
    order = np.argsort(costs_n)
    return float(np.trapz(accs[order], costs_n[order])), costs.min(), costs.max()


def main():
    print(f"DEVICE: {base.DEVICE}  seed={SEED}  lambdas: {len(LAMBDAS)} points", flush=True)
    pool_20 = common.MODELS_20

    base._TOKENIZER = AutoTokenizer.from_pretrained(base.EMBED_MODEL)
    base._BASE_MODEL = AutoModel.from_pretrained(base.EMBED_MODEL).to(base.DEVICE)
    base._BASE_MODEL.eval()

    split = loo.load_split()
    train_rows = loo.build_train_rows(split)
    cost_dict = loo.build_cost_dict(split)
    setB_queries, true_scores = loo.build_setB_eval(split)
    print(f"train_rows={len(train_rows)}  setB={len(setB_queries)}", flush=True)

    setB_embeds = loo.precompute_embeddings(setB_queries, base._TOKENIZER, base._BASE_MODEL)

    cost_arr = np.array([cost_dict[m] for m in pool_20])
    cost_norm_20 = (cost_arr - cost_arr.min()) / (cost_arr.max() - cost_arr.min() + 1e-9)

    results = {}
    t0 = time.time()
    for fp_name, desc_dir in [("Perplexity", loo.PERP_DIR), ("Ceiling", loo.CEILING_DIR)]:
        print(f"\n{'='*70}\nFP: {fp_name} (full pool, no held-out)\n{'='*70}", flush=True)
        head, names20 = loo.train_fold(pool_20, desc_dir, train_rows, cost_dict, seed=SEED, balance_beta=BETA)
        col_idx = [pool_20.index(n) for n in names20]
        scores_reordered = true_scores[:, col_idx]
        cost_reordered = cost_norm_20[col_idx]

        with torch.no_grad():
            q = head(torch.from_numpy(setB_embeds).float().to(base.DEVICE))
            E20, _ = loo.load_descriptors_ordered(desc_dir, names20)
            E20_t = torch.from_numpy(E20).float().to(base.DEVICE)
            E20_t = E20_t / (E20_t.norm(dim=1, keepdim=True) + 1e-9)
            sims = (q @ E20_t.T).cpu().numpy()

        points = audc_curve(sims, cost_reordered, scores_reordered, LAMBDAS)
        audc, cmin, cmax = trapz_audc(points)
        # lambda=0 point is the pure-accuracy (cost-blind) baseline, matches router_overall_accuracy style
        lam0_acc = [p[1] for p in points if p[2] == 0.0][0]
        print(f"[{fp_name}] AUDC(cost-normalized)={audc:.4f}  cost range=[{cmin:.6f},{cmax:.6f}]  "
              f"lambda=0 accuracy={lam0_acc:.4f}  elapsed={time.time()-t0:.0f}s", flush=True)

        results[fp_name] = {
            "points": points,  # list of (avg_cost, accuracy, lambda)
            "audc_cost_normalized": audc,
            "cost_min": cmin,
            "cost_max": cmax,
            "lambda0_accuracy": lam0_acc,
        }

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
