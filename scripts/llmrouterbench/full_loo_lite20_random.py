"""Full 20-fold parametric LOO using a pure-noise Random FP (22-dim, no
relationship to actual capability) -- negative control to check whether the
LOO pipeline itself produces above-chance AUC even with a meaningless
descriptor. Expectation: mean AUC should hover near 0.5 (chance) with no
systematic bias, since a held-out model's random vector has no consistent
relationship to which queries it actually answers correctly.
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
RANDOM_DIR = Path("local_descriptors/llmrouterbench_lite20/ceiling_random")
OUT_PATH = Path("local_descriptors/llmrouterbench_lite20/full_loo_random_results.json")


def main():
    print(f"DEVICE: {base.DEVICE}  SEED: {SEED}", flush=True)
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

    import random
    rng = random.Random(1)
    probe_idx = rng.sample(range(len(setB_queries)), min(200, len(setB_queries)))
    probe_texts = [setB_queries[i] for i in probe_idx]

    all_results = {}
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            all_results = json.load(f)

    t0 = time.time()
    fold_results = []
    for held_out in pool_20:
        pool_19 = [m for m in pool_20 if m != held_out]
        print(f"[Random][seed={SEED}] held out: {held_out} ... (elapsed {time.time()-t0:.0f}s)", flush=True)
        head, names19 = loo.train_fold(pool_19, RANDOM_DIR, train_rows, cost_dict, seed=SEED, balance_beta=BETA)
        r = loo.evaluate_fold(head, held_out, pool_19, RANDOM_DIR, pool_20, setB_embeds, true_scores,
                               cost_norm_20, probe_texts)
        print(f"  n_oracle_is_M={r['n_oracle_is_M']}  oracle_match_rate={r['oracle_match_rate']:.4f}  "
              f"router_acc={r['router_overall_accuracy']:.4f}  AUC={r['auc_heldout_correctness']:.4f} "
              f"(p={r['auc_p']:.4f})", flush=True)
        fold_results.append(r)

    all_results[str(SEED)] = fold_results
    mean_auc = np.mean([r["auc_heldout_correctness"] for r in fold_results])
    print(f"\n--- Random FP summary (seed={SEED}) --- mean AUC: {mean_auc:.4f}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
