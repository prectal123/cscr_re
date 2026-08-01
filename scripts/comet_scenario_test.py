"""User's scenario (2026-08-01): for categories where the in-pool candidates
are "torn" (top-1 vs top-2 mean-centered Ceiling score nearly tied -- no
strong existing preference), check whether the held-out (unseen) model's
Ceiling score "wins" clearly over those torn candidates, and if so, whether
the held-out model's ACTUAL Set B accuracy on those categories genuinely
beats the torn candidates' actual accuracy.

Fully training-free -- uses the raw (pre-L2-norm) mean-centered per-category
Ceiling values directly (comparing individual dimensions across models after
L2-normalizing the whole vector would be confounded by each model's own norm,
so we recompute the raw centered values here instead of loading the saved
.npy files).

Different from the existing kNN test (routerbench_knn_test.py): that one
averages over ALL of Set B unconditionally; this one specifically isolates
prompts where the ambiguous in-pool candidates in an easy/hard sense.
"""
import numpy as np

import routerbench_knn_test as rb

NAMES = rb.NAMES
MODELS = rb.MODELS

TORN_GAP_MAX = 0.02     # top1 - top2 among pool_10 candidates must be <= this to count as "torn"
WIN_MARGIN_MIN = 0.02   # held_out's score must exceed torn top1 by at least this to count as "comet wins"


def build_raw_centered(set_a):
    eval_names = sorted(set_a["eval_name"].unique())
    name_to_idx = {e: i for i, e in enumerate(eval_names)}
    raw = {}
    for name, model_col in zip(NAMES, MODELS):
        vec = np.zeros(len(eval_names))
        counts = np.zeros(len(eval_names))
        for ev, score in zip(set_a["eval_name"], set_a[model_col]):
            vec[name_to_idx[ev]] += float(score)
            counts[name_to_idx[ev]] += 1
        raw[name] = vec / np.maximum(counts, 1)
    pool_matrix = np.stack([raw[n] for n in NAMES])
    pool_mean = pool_matrix.mean(axis=0)
    centered = {n: raw[n] - pool_mean for n in NAMES}
    return centered, eval_names


def main():
    set_a, set_b = rb.load_data()
    centered, eval_names = build_raw_centered(set_a)

    print(f"{'held_out':38s} {'#torn_cats':>10s} {'#comet_wins':>12s} {'#SetB_prompts':>14s} "
          f"{'heldout_acc':>12s} {'torn_best_acc':>14s} {'delta':>8s}")

    pooled_n = 0
    pooled_heldout_correct = 0
    pooled_torn_correct = 0

    for held_out in NAMES:
        pool_10 = [n for n in NAMES if n != held_out]
        n_torn = 0
        n_comet_wins = 0
        winning_cats = []  # (category_idx, torn_leader_name)
        for k in range(len(eval_names)):
            vals = sorted(((centered[n][k], n) for n in pool_10), reverse=True)
            top1_val, top1_name = vals[0]
            top2_val, _ = vals[1]
            gap = top1_val - top2_val
            if gap > TORN_GAP_MAX:
                continue
            n_torn += 1
            ho_val = centered[held_out][k]
            if ho_val - top1_val >= WIN_MARGIN_MIN:
                n_comet_wins += 1
                winning_cats.append((eval_names[k], top1_name))

        if not winning_cats:
            print(f"{held_out:38s} {n_torn:10d} {n_comet_wins:12d} {'--':>14s} {'--':>12s} {'--':>14s} {'--':>8s}")
            continue

        win_eval_names = {ev for ev, _ in winning_cats}
        leader_by_cat = {ev: leader for ev, leader in winning_cats}

        sub_b = set_b[set_b["eval_name"].isin(win_eval_names)]
        n_prompts = len(sub_b)
        if n_prompts == 0:
            print(f"{held_out:38s} {n_torn:10d} {n_comet_wins:12d} {0:14d} {'--':>12s} {'--':>14s} {'--':>8s}")
            continue

        ho_col = dict(zip(NAMES, MODELS))[held_out]
        heldout_correct = (sub_b[ho_col] >= 1.0).sum()
        heldout_acc = heldout_correct / n_prompts

        # torn-leader accuracy: for each prompt, use whichever model was the
        # "torn leader" (top1 among pool_10) in that prompt's specific category
        torn_correct = 0
        for _, row in sub_b.iterrows():
            leader = leader_by_cat[row["eval_name"]]
            leader_col = dict(zip(NAMES, MODELS))[leader]
            if row[leader_col] >= 1.0:
                torn_correct += 1
        torn_acc = torn_correct / n_prompts

        pooled_n += n_prompts
        pooled_heldout_correct += heldout_correct
        pooled_torn_correct += torn_correct

        print(f"{held_out:38s} {n_torn:10d} {n_comet_wins:12d} {n_prompts:14d} "
              f"{heldout_acc:12.4f} {torn_acc:14.4f} {heldout_acc-torn_acc:+8.4f}")

    print(f"\n{'='*90}")
    if pooled_n > 0:
        pooled_ho_acc = pooled_heldout_correct / pooled_n
        pooled_torn_acc = pooled_torn_correct / pooled_n
        print(f"POOLED (n={pooled_n}): held_out_acc={pooled_ho_acc:.4f}  torn_leader_acc={pooled_torn_acc:.4f}  "
              f"delta={pooled_ho_acc-pooled_torn_acc:+.4f}")
    else:
        print("No qualifying prompts found -- thresholds may be too strict.")


if __name__ == "__main__":
    main()
