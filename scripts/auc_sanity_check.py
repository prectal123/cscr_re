"""Sanity check for suspiciously high AUC values: retrain ONE fold (the
highest-AUC case seen so far -- Perplexity FP, code-llama-34b held out,
seed=1, AUC=0.79) and additionally compute AUC with the TRUE labels shuffled
(20 random shuffles) as a negative control.

If the pipeline is correct: shuffled-label AUC should center tightly around
0.5 (chance) regardless of the model/seed, since shuffling destroys any real
relationship between the query-similarity scores and correctness. If shuffled
AUC is ALSO consistently far from 0.5, that reveals a real implementation bug
(e.g. index misalignment) rather than a genuine signal.
"""
import sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import routerbench_loo_recovery as loo
import loo_unseen_recovery as base
from ceiling_multiseed_check import evaluate_fold_with_rank
from transformers import AutoModel, AutoTokenizer

HELD_OUT = "meta__code-llama-instruct-34b-chat"
SEED = 1
BETA = 1.0
N_SHUFFLES = 20


def main():
    print(f"DEVICE: {base.DEVICE}")
    pool_11 = loo.rb.NAMES
    pool_10 = [m for m in pool_11 if m != HELD_OUT]

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

    print(f"\nRetraining [Perplexity] held_out={HELD_OUT} seed={SEED} for sanity check...")
    head, names10 = loo.train_fold(pool_10, loo.PERP_DIR, train_rows, cost_dict, seed=SEED, balance_beta=BETA)
    r = evaluate_fold_with_rank(head, HELD_OUT, pool_10, loo.PERP_DIR, pool_11, cls_embeds,
                                 true_scores, cost_norm_11)
    print(f"\nReal AUC (should match earlier run, ~0.79): {r['auc_heldout_correctness']:.4f}")

    # now recompute AUC manually with the raw sims/labels, plus shuffled-label negative control
    E10, names10b = loo.load_descriptors_ordered(loo.PERP_DIR, pool_10)
    held_out_vec = np.load(loo.PERP_DIR / f"{HELD_OUT}.npy")
    all_names = names10b + [HELD_OUT]
    E11 = np.stack(list(E10) + [held_out_vec])
    E11 = E11 / (np.linalg.norm(E11, axis=1, keepdims=True) + 1e-9)
    E11_t = torch.from_numpy(E11).float().to(base.DEVICE)
    held_out_col = all_names.index(HELD_OUT)
    col_idx = [pool_11.index(n) for n in all_names]
    scores_reordered = true_scores[:, col_idx]

    with torch.no_grad():
        q = head(torch.from_numpy(cls_embeds).float().to(base.DEVICE))
        sims = (q @ E11_t.T).cpu().numpy()

    ho_sims = sims[:, held_out_col]
    ho_labels = (scores_reordered[:, held_out_col] >= 1.0).astype(int)
    real_auc = roc_auc_score(ho_labels, ho_sims)
    print(f"Manually recomputed real AUC (independent of evaluate_fold_with_rank): {real_auc:.4f}")
    print(f"n_positive={ho_labels.sum()}/{len(ho_labels)}  n_prompts={len(ho_labels)}")

    rng = np.random.RandomState(0)
    shuffled_aucs = []
    for i in range(N_SHUFFLES):
        shuffled_labels = ho_labels.copy()
        rng.shuffle(shuffled_labels)
        shuffled_aucs.append(roc_auc_score(shuffled_labels, ho_sims))
    shuffled_aucs = np.array(shuffled_aucs)
    print(f"\nShuffled-label AUC over {N_SHUFFLES} random shuffles:")
    print(f"  mean={shuffled_aucs.mean():.4f}  std={shuffled_aucs.std():.4f}  "
          f"min={shuffled_aucs.min():.4f}  max={shuffled_aucs.max():.4f}")
    print(f"\n{'='*70}")
    if abs(shuffled_aucs.mean() - 0.5) < 0.02:
        print("PASS: shuffled-label AUC centers tightly around 0.5 (chance) -- pipeline looks correct.")
        print(f"The real AUC ({real_auc:.4f}) reflects genuine structure in the trained similarity scores,")
        print("not a computation bug.")
    else:
        print(f"WARNING: shuffled-label AUC mean ({shuffled_aucs.mean():.4f}) deviates meaningfully from 0.5 --")
        print("this suggests a real bug (e.g. index misalignment) independent of any genuine signal.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
