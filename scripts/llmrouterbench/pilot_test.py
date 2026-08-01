"""Pilot: single held-out model, Perplexity FP, beta=1.0, seed=0 -- validates
the whole LLMRouterBench LOO pipeline end-to-end before scaling to all 33
folds. Also prints collapse diagnostics to give an early read on whether the
bigger (33-model) pool changes the collapse picture found on RouterBench (11).
"""
import random
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts/llmrouterbench")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import common
import loo_recovery as loo
import loo_unseen_recovery as base
from transformers import AutoModel, AutoTokenizer

HELD_OUT = "gpt-5"
SEED = 0
BETA = 1.0


def main():
    print(f"DEVICE: {base.DEVICE}")
    pool_33 = common.MODELS_33
    pool_32 = [m for m in pool_33 if m != HELD_OUT]

    base._TOKENIZER = AutoTokenizer.from_pretrained(base.EMBED_MODEL)
    base._BASE_MODEL = AutoModel.from_pretrained(base.EMBED_MODEL).to(base.DEVICE)
    base._BASE_MODEL.eval()

    split = loo.load_split()
    train_rows = loo.build_train_rows(split)
    cost_dict = loo.build_cost_dict(split)
    setB_queries, true_scores = loo.build_setB_eval(split)
    print(f"train_rows={len(train_rows)}  setB={len(setB_queries)}")

    setB_embeds = loo.precompute_embeddings(setB_queries, base._TOKENIZER, base._BASE_MODEL)

    cost_arr = np.array([cost_dict[m] for m in pool_33])
    cost_norm_33 = (cost_arr - cost_arr.min()) / (cost_arr.max() - cost_arr.min() + 1e-9)

    rng = random.Random(1)
    probe_idx = rng.sample(range(len(setB_queries)), min(200, len(setB_queries)))
    probe_texts = [setB_queries[i] for i in probe_idx]

    head, names32 = loo.train_fold(pool_32, loo.PERP_DIR, train_rows, cost_dict, seed=SEED, balance_beta=BETA)
    r = loo.evaluate_fold(head, HELD_OUT, pool_32, loo.PERP_DIR, pool_33, setB_embeds, true_scores,
                           cost_norm_33, probe_texts)

    print(f"\nheld_out={HELD_OUT}")
    print(f"n_oracle_is_M={r['n_oracle_is_M']}  oracle_match_rate={r['oracle_match_rate']:.4f}")
    print(f"router_overall_accuracy={r['router_overall_accuracy']:.4f}")
    print(f"AUC={r['auc_heldout_correctness']:.4f} (p={r['auc_p']:.4f}, chance=0.5, "
          f"n_pos={r['n_heldout_positive_in_setB']}/{r['n_setB_total']})")
    n_nonzero = sum(1 for v in r["collapse_nearest_dist"].values() if v > 0)
    top3 = sorted(r["collapse_nearest_dist"].values(), reverse=True)[:3]
    print(f"collapse: n_nonzero={n_nonzero}/33  top3_share={sum(top3)/200:.3f}")
    print(f"nearest_dist: {r['collapse_nearest_dist']}")


if __name__ == "__main__":
    main()
