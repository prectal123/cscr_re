"""Step 1: train on ALL 33 models (no held-out), beta=0 (no load-balancing --
raw collapse tendency), both FP types, to see the baseline collapse picture
in this bigger pool before any LOO/mitigation. Comparable reference point:
RouterBench (11 models, beta=0): n_nonzero=3/11, top3_share=0.97, router_acc=0.67.
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

SEED = 0
BETA = 0.0


def main():
    print(f"DEVICE: {base.DEVICE}")
    pool_33 = common.MODELS_33

    base._TOKENIZER = AutoTokenizer.from_pretrained(base.EMBED_MODEL)
    base._BASE_MODEL = AutoModel.from_pretrained(base.EMBED_MODEL).to(base.DEVICE)
    base._BASE_MODEL.eval()

    split = loo.load_split()
    train_rows = loo.build_train_rows(split)
    cost_dict = loo.build_cost_dict(split)
    setB_queries, true_scores = loo.build_setB_eval(split)
    print(f"train_rows={len(train_rows)}  setB={len(setB_queries)}", flush=True)

    setB_embeds = loo.precompute_embeddings(setB_queries, base._TOKENIZER, base._BASE_MODEL)

    rng = random.Random(1)
    probe_idx = rng.sample(range(len(setB_queries)), min(200, len(setB_queries)))
    probe_texts = [setB_queries[i] for i in probe_idx]

    for fp_name, desc_dir in [("Perplexity", loo.PERP_DIR), ("Ceiling", loo.CEILING_DIR)]:
        print(f"\n{'='*70}\nFP: {fp_name} (full 33-model pool, beta={BETA})\n{'='*70}", flush=True)
        head, names33 = loo.train_fold(pool_33, desc_dir, train_rows, cost_dict, seed=SEED, balance_beta=BETA)

        E33, _ = loo.load_descriptors_ordered(desc_dir, pool_33)
        E33 = E33 / (np.linalg.norm(E33, axis=1, keepdims=True) + 1e-9)
        E33_t = torch.from_numpy(E33).float().to(base.DEVICE)

        nearest_dist = loo.collapse_diagnostic(head, probe_texts, E33_t, pool_33)
        n_nonzero = sum(1 for v in nearest_dist.values() if v > 0)
        top3 = sorted(nearest_dist.values(), reverse=True)[:3]
        top3_share = sum(top3) / 200

        with torch.no_grad():
            q = head(torch.from_numpy(setB_embeds).float().to(base.DEVICE))
            sims = q @ E33_t.T
            chosen = sims.argmax(dim=1).cpu().numpy()
        router_acc = np.mean([true_scores[i, chosen[i]] >= 1.0 for i in range(len(chosen))])

        print(f"n_nonzero={n_nonzero}/33  top3_share={top3_share:.3f}  router_acc={router_acc:.4f}", flush=True)
        print(f"nearest_dist: {nearest_dist}", flush=True)
        print(f"[reference] RouterBench(11 models, beta=0): n_nonzero=3/11, top3_share=0.970, router_acc=0.670",
              flush=True)
        # relative-to-random concentration factor
        random_top3 = 3 / 33
        print(f"[reference] random-baseline top3_share for 33 models = {random_top3:.3f}  "
              f"(concentration factor = {top3_share/random_top3:.2f}x)", flush=True)


if __name__ == "__main__":
    main()
