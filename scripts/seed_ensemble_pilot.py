"""Pilot test: does averaging similarity SCORES across 3 independently-seeded
encoders (score-level ensemble) recover a more stable/accurate AUC than any
single seed, or than simply averaging the seeds' final AUC numbers?

4 models chosen to cover the range of patterns seen today (Ceiling FP,
beta=1.0): gpt-4 (consistently strong across seeds), claude-v2 (unstable,
degrading), mistral-7b (consistently weak/null), WizardLM (unstable).

For each model: train 3 heads (seed 0/1/2), keep all 3, compute each head's
sims matrix, average the sims elementwise (score-level ensemble), then run
the same AUC/oracle_match_rate/rank pipeline on the averaged sims. Compare
against: (a) each individual seed's own AUC, (b) the mean of the 3 individual
AUCs (metric-level average, already known from seed_avg table).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import pointbiserialr

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import routerbench_loo_recovery as loo
import loo_unseen_recovery as base
from transformers import AutoModel, AutoTokenizer

HELD_OUTS = [
    "gpt-4-1106-preview",              # consistently strong (0.745/0.739/0.738)
    "claude-v2",                       # unstable, degrading (0.625/0.499/0.473)
    "mistralai__mistral-7b-chat",      # consistently weak/null (0.475/0.497/0.452)
    "WizardLM__WizardLM-13B-V1.2",     # unstable (0.542/0.444/0.441)
]
SEEDS = [0, 1, 2]
BETA = 1.0
OUT_PATH = Path("local_descriptors/routerbench-analysis/seed_ensemble_pilot_results.json")


def compute_sims(head, cls_embeds, E11_t):
    with torch.no_grad():
        q = head(torch.from_numpy(cls_embeds).float().to(base.DEVICE))
        return (q @ E11_t.T).cpu().numpy()


def eval_from_sims(sims, held_out_col, scores_reordered, cost_reordered):
    ranks = []
    n_oracle_is_M = 0
    n_match = 0
    for i in range(scores_reordered.shape[0]):
        row = scores_reordered[i]
        correct_mask = row >= 1.0
        if not correct_mask.any():
            continue
        masked_cost = np.where(correct_mask, cost_reordered, np.inf)
        oracle_idx = int(np.argmin(masked_cost))
        if oracle_idx != held_out_col:
            continue
        n_oracle_is_M += 1
        row_sims = sims[i]
        order = np.argsort(-row_sims)
        rank = int(np.where(order == held_out_col)[0][0]) + 1
        ranks.append(rank)
        if rank == 1:
            n_match += 1

    ho_sims_all = sims[:, held_out_col]
    ho_labels_all = (scores_reordered[:, held_out_col] >= 1.0).astype(int)
    auc = pb_p = float("nan")
    if 0 < ho_labels_all.sum() < len(ho_labels_all):
        auc = float(roc_auc_score(ho_labels_all, ho_sims_all))
        _, pb_p = pointbiserialr(ho_labels_all, ho_sims_all)
        pb_p = float(pb_p)

    return {
        "n_oracle_is_M": n_oracle_is_M,
        "oracle_match_rate": n_match / n_oracle_is_M if n_oracle_is_M else float("nan"),
        "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
        "auc": auc,
        "auc_p": pb_p,
    }


def main():
    print(f"DEVICE: {base.DEVICE}")
    pool_11 = loo.rb.NAMES

    base._TOKENIZER = AutoTokenizer.from_pretrained(base.EMBED_MODEL)
    base._BASE_MODEL = AutoModel.from_pretrained(base.EMBED_MODEL).to(base.DEVICE)
    base._BASE_MODEL.eval()

    set_a, set_b = loo.rb.load_data()
    set_b_texts = set_b["prompt"].tolist()
    cls_embeds = loo.precompute_set_b_embeddings(set_b_texts, base._TOKENIZER, base._BASE_MODEL)
    true_scores = loo.np.stack([set_b[m].to_numpy(dtype=float) for m in loo.rb.MODELS], axis=1)

    train_rows = loo.build_train_rows(set_a)
    cost_dict = loo.build_cost_dict(set_a)
    cost_arr = loo.np.array([cost_dict[n] for n in pool_11])
    cost_norm_11 = (cost_arr - cost_arr.min()) / (cost_arr.max() - cost_arr.min() + 1e-9)

    results = {}
    t0 = time.time()
    for held_out in HELD_OUTS:
        pool_10 = [m for m in pool_11 if m != held_out]

        E10, names10 = loo.load_descriptors_ordered(loo.CEILING_DIR, pool_10)
        held_out_vec = np.load(loo.CEILING_DIR / f"{held_out}.npy")
        all_names = names10 + [held_out]
        E11 = np.stack(list(E10) + [held_out_vec])
        E11 = E11 / (np.linalg.norm(E11, axis=1, keepdims=True) + 1e-9)
        E11_t = torch.from_numpy(E11).float().to(base.DEVICE)
        held_out_col = all_names.index(held_out)
        col_idx = [pool_11.index(n) for n in all_names]
        scores_reordered = true_scores[:, col_idx]
        cost_reordered = cost_norm_11[col_idx]

        per_seed_sims = []
        per_seed_metrics = []
        for seed in SEEDS:
            print(f"\n=== held_out={held_out} seed={seed} (elapsed {time.time()-t0:.0f}s) ===", flush=True)
            head, _ = loo.train_fold(pool_10, loo.CEILING_DIR, train_rows, cost_dict, seed=seed, balance_beta=BETA)
            sims = compute_sims(head, cls_embeds, E11_t)
            per_seed_sims.append(sims)
            m = eval_from_sims(sims, held_out_col, scores_reordered, cost_reordered)
            per_seed_metrics.append(m)
            print(f"  seed={seed}  AUC={m['auc']:.4f}  oracle_match_rate={m['oracle_match_rate']:.4f}", flush=True)

        ensembled_sims = np.mean(per_seed_sims, axis=0)
        ensembled_metrics = eval_from_sims(ensembled_sims, held_out_col, scores_reordered, cost_reordered)
        mean_of_individual_auc = float(np.mean([m["auc"] for m in per_seed_metrics]))
        mean_of_individual_rate = float(np.mean([m["oracle_match_rate"] for m in per_seed_metrics]))

        seed_auc_strs = [f"{m['auc']:.4f}" for m in per_seed_metrics]
        seed_rate_strs = [f"{m['oracle_match_rate']:.4f}" for m in per_seed_metrics]
        print(f"\n  -- {held_out} summary --")
        print(f"  per-seed AUC: {seed_auc_strs}")
        print(f"  mean-of-individual AUC: {mean_of_individual_auc:.4f}")
        print(f"  SCORE-LEVEL ENSEMBLE AUC: {ensembled_metrics['auc']:.4f}")
        print(f"  per-seed oracle_rate: {seed_rate_strs}")
        print(f"  mean-of-individual oracle_rate: {mean_of_individual_rate:.4f}")
        print(f"  SCORE-LEVEL ENSEMBLE oracle_rate: {ensembled_metrics['oracle_match_rate']:.4f}")

        results[held_out] = {
            "per_seed_metrics": per_seed_metrics,
            "mean_of_individual_auc": mean_of_individual_auc,
            "mean_of_individual_rate": mean_of_individual_rate,
            "ensembled_metrics": ensembled_metrics,
        }
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
