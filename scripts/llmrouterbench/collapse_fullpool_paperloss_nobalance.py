"""Fast check: train on the FULL 20-model pool (no LOO), paper loss
(cost_spectrum_info_nce), load_balance_loss OFF (beta=0) -- just to read off
the "base" collapse level (top3_share on 200 fixed probes) without the
load-balancing auxiliary term. Trains once per FP instead of 20x.
"""
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "scripts/llmrouterbench")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import common_lite20 as common
import loo_recovery_lite20_paperloss as loo
import loo_unseen_recovery as base
from transformers import AutoModel, AutoTokenizer

SEED = 0
BETA = 0.0


def main():
    print(f"DEVICE: {base.DEVICE}  LOSS: cost_spectrum_info_nce (paper Eq.8, n_bands={loo.N_BANDS})  BETA={BETA}  full pool, no LOO", flush=True)
    pool_20 = common.MODELS_20

    base._TOKENIZER = AutoTokenizer.from_pretrained(base.EMBED_MODEL)
    base._BASE_MODEL = AutoModel.from_pretrained(base.EMBED_MODEL).to(base.DEVICE)
    base._BASE_MODEL.eval()

    split = loo.load_split()
    train_rows = loo.build_train_rows(split)
    cost_dict = loo.build_cost_dict(split)
    setB_queries, true_scores = loo.build_setB_eval(split)

    rng = random.Random(1)
    probe_idx = rng.sample(range(len(setB_queries)), min(200, len(setB_queries)))
    probe_texts = [setB_queries[i] for i in probe_idx]

    t0 = time.time()
    for fp_name, desc_dir in [("Perplexity", loo.PERP_DIR), ("Ceiling", loo.CEILING_DIR)]:
        head, names20 = loo.train_fold(pool_20, desc_dir, train_rows, cost_dict, seed=SEED, balance_beta=BETA)
        E20, _ = loo.load_descriptors_ordered(desc_dir, names20)
        E20_t = torch.from_numpy(E20).float().to(base.DEVICE)
        E20_t = E20_t / (E20_t.norm(dim=1, keepdim=True) + 1e-9)

        nearest_dist = loo.collapse_diagnostic(head, probe_texts, E20_t, names20)
        total = sum(nearest_dist.values())
        top3 = sorted(nearest_dist.values(), reverse=True)[:3]
        top3_share = sum(top3) / total
        order = sorted(nearest_dist.items(), key=lambda x: -x[1])
        print(f"\n[{fp_name}] top3_share = {top3_share:.4f}  (chance=0.15)  elapsed={time.time()-t0:.0f}s", flush=True)
        print(f"  top5: {order[:5]}", flush=True)

    print(f"\nDone. total elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
