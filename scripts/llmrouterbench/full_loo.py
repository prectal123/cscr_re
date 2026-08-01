"""Step 3: full 33-fold LOO (every model held out once), both Ceiling and
Perplexity FP, beta=1.0, seed=0. Each fold trains fast (~22s, ~2345 rows)
so 66 folds total should finish in well under an hour.
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
import common
import loo_recovery as loo
import loo_unseen_recovery as base
from transformers import AutoModel, AutoTokenizer

SEED = 0
BETA = 1.0
OUT_PATH = Path("local_descriptors/llmrouterbench/full_loo_results.json")


def main():
    print(f"DEVICE: {base.DEVICE}", flush=True)
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

    cost_arr = np.array([cost_dict[m] for m in pool_33])
    cost_norm_33 = (cost_arr - cost_arr.min()) / (cost_arr.max() - cost_arr.min() + 1e-9)

    rng = random.Random(1)
    probe_idx = rng.sample(range(len(setB_queries)), min(200, len(setB_queries)))
    probe_texts = [setB_queries[i] for i in probe_idx]

    all_results = {}
    t0 = time.time()
    for fp_name, desc_dir in [("Perplexity", loo.PERP_DIR), ("Ceiling", loo.CEILING_DIR)]:
        print(f"\n{'='*70}\nFP: {fp_name}\n{'='*70}", flush=True)
        fold_results = []
        for held_out in pool_33:
            pool_32 = [m for m in pool_33 if m != held_out]
            print(f"[{fp_name}] held out: {held_out} ... (elapsed {time.time()-t0:.0f}s)", flush=True)
            head, names32 = loo.train_fold(pool_32, desc_dir, train_rows, cost_dict, seed=SEED, balance_beta=BETA)
            r = loo.evaluate_fold(head, held_out, pool_32, desc_dir, pool_33, setB_embeds, true_scores,
                                   cost_norm_33, probe_texts)
            print(f"  n_oracle_is_M={r['n_oracle_is_M']}  oracle_match_rate={r['oracle_match_rate']:.4f}  "
                  f"router_acc={r['router_overall_accuracy']:.4f}  AUC={r['auc_heldout_correctness']:.4f} "
                  f"(p={r['auc_p']:.4f})", flush=True)
            fold_results.append(r)
        all_results[fp_name] = fold_results

        rates = [r["oracle_match_rate"] for r in fold_results if r["n_oracle_is_M"] > 0]
        n_total = sum(r["n_oracle_is_M"] for r in fold_results)
        pooled_hits = sum(r["n_oracle_is_M"] * r["oracle_match_rate"] for r in fold_results if r["n_oracle_is_M"] > 0)
        pooled_rate = pooled_hits / n_total if n_total else float("nan")
        mean_auc = np.mean([r["auc_heldout_correctness"] for r in fold_results])
        chance = 1 / 33
        print(f"\n--- {fp_name} summary ---", flush=True)
        print(f"simple mean oracle_match_rate: {np.mean(rates):.4f} (chance~{chance:.4f})", flush=True)
        print(f"pooled(n-weighted) oracle_match_rate: {pooled_rate:.4f} (total_n={n_total})", flush=True)
        print(f"mean AUC: {mean_auc:.4f}", flush=True)

        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nSaved -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
