"""Multi-seed robustness check for the Ceiling FP LOO pattern (PROGRESS.md 16.4/16.6):
WizardLM/claude-instant-v1 recover well when held out, Yi-34B-Chat/mistral-7b-chat
recover ~0%. Re-run just these 4 folds across 3 seeds (beta=1.0 load-balancing,
same settings as the full run) to see if the pattern is seed-stable or noise.

Also adds a rank/margin diagnostic (user's suggestion): oracle_match_rate is a
strict argmax==held_out metric, so a held-out model that loses a near-tie
cosine-similarity contest to a strong competitor looks identical to one that's
nowhere close. For every prompt where held_out IS the oracle pick, record:
  - rank of held_out's similarity among all 11 candidates (1 = argmax = hit)
  - margin = sim(held_out) - sim(top1 chosen)  (0 if hit, negative otherwise)
This tells us whether "misses" are near-misses (small negative margin, rank 2)
or the held-out model isn't in contention at all (large negative margin).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import routerbench_loo_recovery as loo
import loo_unseen_recovery as base
from transformers import AutoModel, AutoTokenizer

FOLDS = [
    "zero-one-ai__Yi-34B-Chat",       # failed, 0.2%
    "mistralai__mistral-7b-chat",     # failed, 0%
    "WizardLM__WizardLM-13B-V1.2",    # succeeded, 48.4%
    "claude-instant-v1",              # succeeded, 55.6%
]
SEEDS = [0, 1, 2]
OUT_PATH = Path("local_descriptors/routerbench-analysis/ceiling_multiseed_check.json")


def evaluate_fold_with_rank(head, held_out, pool_10, desc_dir, pool_11, cls_embeds, true_scores, cost_norm_11):
    E10, names10 = loo.load_descriptors_ordered(desc_dir, pool_10)
    held_out_vec = np.load(Path(desc_dir) / f"{held_out}.npy")
    all_names = names10 + [held_out]
    E11 = np.stack(list(E10) + [held_out_vec])
    E11 = E11 / (np.linalg.norm(E11, axis=1, keepdims=True) + 1e-9)
    E11_t = torch.from_numpy(E11).float().to(base.DEVICE)

    col_idx = [pool_11.index(n) for n in all_names]
    scores_reordered = true_scores[:, col_idx]
    cost_reordered = cost_norm_11[col_idx]
    held_out_col = all_names.index(held_out)

    with torch.no_grad():
        q = head(torch.from_numpy(cls_embeds).float().to(base.DEVICE))
        sims = (q @ E11_t.T).cpu().numpy()  # (N, 11)

    other_cols = [j for j in range(len(all_names)) if j != held_out_col]

    ranks, margins, gaps = [], [], []
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
        order = np.argsort(-row_sims)  # descending sim -> rank order of candidate indices
        rank = int(np.where(order == held_out_col)[0][0]) + 1  # 1-indexed
        top1_sim = row_sims[order[0]]
        held_out_sim = row_sims[held_out_col]
        margin = float(held_out_sim - top1_sim)  # 0 if rank==1, else negative
        ranks.append(rank)
        margins.append(margin)
        if rank == 1:
            n_match += 1

        # metric 1 (user's idea): among the OTHER 10 experts on this same prompt,
        # does the query sit closer to the co-correct ones or the wrong ones?
        # -- tests whether the query found the right "neighborhood" even when it
        # doesn't specifically peak at held_out.
        correct_others = [j for j in other_cols if row[j] >= 1.0]
        incorrect_others = [j for j in other_cols if row[j] < 1.0]
        if correct_others and incorrect_others:
            gaps.append(float(row_sims[correct_others].mean() - row_sims[incorrect_others].mean()))

    # metric 2: point-biserial corr / AUC between q.E_heldout and held_out's own
    # correctness label, over ALL of Set B (not just the n_oracle_is_M subset) --
    # decouples "does the descriptor carry a real per-prompt signal" from
    # "does held_out win the argmax contest against the other 10".
    ho_sims_all = sims[:, held_out_col]
    ho_labels_all = (scores_reordered[:, held_out_col] >= 1.0).astype(int)
    pb_corr = pb_p = auc = float("nan")
    if 0 < ho_labels_all.sum() < len(ho_labels_all):
        from scipy.stats import pointbiserialr
        pb_corr, pb_p = pointbiserialr(ho_labels_all, ho_sims_all)
        pb_corr, pb_p = float(pb_corr), float(pb_p)
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(ho_labels_all, ho_sims_all))
        except ImportError:
            pass

    return {
        "n_oracle_is_M": n_oracle_is_M,
        "oracle_match_rate": n_match / n_oracle_is_M if n_oracle_is_M else float("nan"),
        "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
        "median_rank": float(np.median(ranks)) if ranks else float("nan"),
        "rank_histogram": {str(r): int(np.sum(np.array(ranks) == r)) for r in range(1, 12)} if ranks else {},
        "mean_margin_when_miss": float(np.mean([m for m, r in zip(margins, ranks) if r != 1])) if any(r != 1 for r in ranks) else float("nan"),
        "mean_margin_all": float(np.mean(margins)) if margins else float("nan"),
        "mean_correct_incorrect_gap": float(np.mean(gaps)) if gaps else float("nan"),
        "n_gap_samples": len(gaps),
        "point_biserial_corr_heldout": pb_corr,
        "point_biserial_p_heldout": pb_p,
        "auc_heldout_correctness": auc,
        "n_setB_total": int(scores_reordered.shape[0]),
        "n_heldout_positive_in_setB": int(ho_labels_all.sum()),
    }


def main():
    print(f"DEVICE: {base.DEVICE}")
    pool_11 = loo.rb.NAMES

    print("Loading frozen MiniLM base (mean pooling)...")
    base._TOKENIZER = AutoTokenizer.from_pretrained(base.EMBED_MODEL)
    base._BASE_MODEL = AutoModel.from_pretrained(base.EMBED_MODEL).to(base.DEVICE)
    base._BASE_MODEL.eval()

    print("Loading RouterBench Set A / Set B...")
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
    for held_out in FOLDS:
        results[held_out] = []
        pool_10 = [m for m in pool_11 if m != held_out]
        for seed in SEEDS:
            print(f"\n=== held_out={held_out} seed={seed} (elapsed {time.time()-t0:.0f}s) ===", flush=True)
            head, names10 = loo.train_fold(pool_10, loo.CEILING_DIR, train_rows, cost_dict,
                                            seed=seed, balance_beta=1.0)
            r = evaluate_fold_with_rank(head, held_out, pool_10, loo.CEILING_DIR, pool_11, cls_embeds,
                                         true_scores, cost_norm_11)
            print(f"  seed={seed}  n_oracle_is_M={r['n_oracle_is_M']}  "
                  f"oracle_match_rate={r['oracle_match_rate']:.4f}  "
                  f"mean_rank={r['mean_rank']:.2f} (chance=6.0)  "
                  f"mean_margin_when_miss={r['mean_margin_when_miss']:.4f}", flush=True)
            print(f"  rank_histogram: {r['rank_histogram']}", flush=True)
            results[held_out].append({"seed": seed, **r})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}\nSUMMARY (Ceiling FP, beta=1.0, 3 seeds)\n{'='*70}")
    for held_out, rows in results.items():
        rates = [r["oracle_match_rate"] for r in rows]
        mranks = [r["mean_rank"] for r in rows]
        print(f"  {held_out:38s} oracle_match_rate={[f'{r:.3f}' for r in rates]}  "
              f"mean_rank={[f'{m:.2f}' for m in mranks]} (chance=6.0)")
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
