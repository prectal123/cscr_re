"""Pilot test for cost_info_nce_cheapest (single cheapest-positive target,
instead of cost-weighted average over all positives) -- mentor-motivated
fix for the "averaging over positives pulls toward a generalist attractor"
diagnosis (PROGRESS.md 16.2 / this session's mentor advice #1+#5).

Same 4 models as the seed-ensemble pilot (gpt-4, claude-v2, mistral-7b,
WizardLM), Ceiling FP, beta=1.0 load-balancing kept on top, seed=0 only
(quick check before committing to multi-seed). Compares against the known
seed=0 baseline (weighted-average cost_info_nce): gpt-4 AUC=0.745/rate=10.2%,
claude-v2 AUC=0.625/rate=0%, mistral-7b AUC=0.475/rate=0%, WizardLM
AUC=0.542/rate=48.4%.
"""
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import routerbench_loo_recovery as loo
import loo_unseen_recovery as base
from ceiling_multiseed_check import evaluate_fold_with_rank
from transformers import AutoModel, AutoTokenizer

HELD_OUTS = [
    "gpt-4-1106-preview",
    "claude-v2",
    "mistralai__mistral-7b-chat",
    "WizardLM__WizardLM-13B-V1.2",
]
SEED = 0
BETA = 1.0
BASELINE = {
    "gpt-4-1106-preview": (0.745, 0.102),
    "claude-v2": (0.625, 0.0),
    "mistralai__mistral-7b-chat": (0.475, 0.0),
    "WizardLM__WizardLM-13B-V1.2": (0.542, 0.484),
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

    t0 = time.time()
    print(f"\n{'model':38s} {'baseline AUC':>13s} {'NEW AUC':>10s} {'d_AUC':>8s}   "
          f"{'baseline rate':>14s} {'NEW rate':>10s} {'d_rate':>8s}")
    for held_out in HELD_OUTS:
        pool_10 = [m for m in pool_11 if m != held_out]
        print(f"\n=== held_out={held_out} (elapsed {time.time()-t0:.0f}s) ===", flush=True)
        head, names10 = loo.train_fold(pool_10, loo.CEILING_DIR, train_rows, cost_dict,
                                        seed=SEED, loss_name="cost_info_nce_cheapest", balance_beta=BETA)
        r = evaluate_fold_with_rank(head, held_out, pool_10, loo.CEILING_DIR, pool_11, cls_embeds,
                                     true_scores, cost_norm_11)
        base_auc, base_rate = BASELINE[held_out]
        print(f"{held_out:38s} {base_auc:13.4f} {r['auc_heldout_correctness']:10.4f} "
              f"{r['auc_heldout_correctness']-base_auc:+8.4f}   "
              f"{base_rate:14.4f} {r['oracle_match_rate']:10.4f} {r['oracle_match_rate']-base_rate:+8.4f}")


if __name__ == "__main__":
    main()
