"""Multi-seed re-run of the lightweight-20 full LOO (seed=0 already done and
saved in full_loo_results.json). Runs seed=1 and seed=2 to check whether the
Ceiling-beats-Perplexity win (delta+0.11, p=0.0038 at seed=0) is a robust
finding or a seed-lucky artifact -- RouterBench's 11-fold multi-seed check
(PROGRESS.md 16.11) already showed some models flip direction across seeds,
so this must be verified before treating the seed=0 result as confirmed.
"""
import json
import random
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

SEEDS = [int(sys.argv[1])] if len(sys.argv) > 1 else [1, 2]
BETA = 1.0
OUT_PATH = Path("local_descriptors/llmrouterbench_lite20/full_loo_multiseed_results.json")


def main():
    print(f"DEVICE: {base.DEVICE}", flush=True)
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

    rng = random.Random(1)
    probe_idx = rng.sample(range(len(setB_queries)), min(200, len(setB_queries)))
    probe_texts = [setB_queries[i] for i in probe_idx]

    all_results = {}
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            all_results = json.load(f)

    t0 = time.time()
    for seed in SEEDS:
        seed_key = str(seed)
        if seed_key in all_results:
            print(f"seed {seed} already present in {OUT_PATH}, skipping", flush=True)
            continue
        all_results[seed_key] = {}
        for fp_name, desc_dir in [("Perplexity", loo.PERP_DIR), ("Ceiling", loo.CEILING_DIR)]:
            print(f"\n{'='*70}\nseed={seed}  FP: {fp_name}\n{'='*70}", flush=True)
            fold_results = []
            for held_out in pool_20:
                pool_19 = [m for m in pool_20 if m != held_out]
                print(f"[seed={seed}][{fp_name}] held out: {held_out} ... (elapsed {time.time()-t0:.0f}s)",
                      flush=True)
                head, names19 = loo.train_fold(pool_19, desc_dir, train_rows, cost_dict, seed=seed,
                                                balance_beta=BETA)
                r = loo.evaluate_fold(head, held_out, pool_19, desc_dir, pool_20, setB_embeds, true_scores,
                                       cost_norm_20, probe_texts)
                print(f"  n_oracle_is_M={r['n_oracle_is_M']}  oracle_match_rate={r['oracle_match_rate']:.4f}  "
                      f"router_acc={r['router_overall_accuracy']:.4f}  AUC={r['auc_heldout_correctness']:.4f} "
                      f"(p={r['auc_p']:.4f})", flush=True)
                fold_results.append(r)
            all_results[seed_key][fp_name] = fold_results

            mean_auc = np.mean([r["auc_heldout_correctness"] for r in fold_results])
            print(f"\n--- seed={seed} {fp_name} summary --- mean AUC: {mean_auc:.4f}", flush=True)

            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(OUT_PATH, "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nSaved -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
