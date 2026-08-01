"""Full 20-fold parametric LOO using Ceiling V2 (category-rate, 22-dim: one
dim per dataset = mean Set A accuracy) instead of Ceiling V1 (528-dim,
individual high-variance probes). User's hypothesis: V1's probe-sampling is
a realistic-but-imperfect approximation of an idealized capability-aligned
oracle descriptor; V2 (full-category averaging, lower per-dim noise) may be
closer to that ideal "ceiling". V2 already won the kNN test (PROGRESS.md
17.8) against uniform/Perplexity/V1 -- this checks if that holds under
actual parametric training too. seed=0, beta=1.0, matching the V1 run for a
fair comparison.
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

SEED = 0
BETA = 1.0
CATRATE_DIR = Path("local_descriptors/llmrouterbench_lite20/ceiling_categoryrate")
OUT_PATH = Path("local_descriptors/llmrouterbench_lite20/full_loo_categoryrate_results.json")


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
    t0 = time.time()
    # Perplexity not re-run here -- reuse the already-confirmed-stable seed=0 Perplexity
    # results from full_loo_results.json (0.4759/0.4718/0.4634 across seeds 0/1/2) for comparison.
    for fp_name, desc_dir in [("CeilingV2_categoryrate", CATRATE_DIR)]:
        print(f"\n{'='*70}\nFP: {fp_name}\n{'='*70}", flush=True)
        fold_results = []
        for held_out in pool_20:
            pool_19 = [m for m in pool_20 if m != held_out]
            print(f"[{fp_name}] held out: {held_out} ... (elapsed {time.time()-t0:.0f}s)", flush=True)
            head, names19 = loo.train_fold(pool_19, desc_dir, train_rows, cost_dict, seed=SEED,
                                            balance_beta=BETA)
            r = loo.evaluate_fold(head, held_out, pool_19, desc_dir, pool_20, setB_embeds, true_scores,
                                   cost_norm_20, probe_texts)
            print(f"  n_oracle_is_M={r['n_oracle_is_M']}  oracle_match_rate={r['oracle_match_rate']:.4f}  "
                  f"router_acc={r['router_overall_accuracy']:.4f}  AUC={r['auc_heldout_correctness']:.4f} "
                  f"(p={r['auc_p']:.4f})", flush=True)
            fold_results.append(r)
        all_results[fp_name] = fold_results

        mean_auc = np.mean([r["auc_heldout_correctness"] for r in fold_results])
        print(f"\n--- {fp_name} summary --- mean AUC: {mean_auc:.4f}", flush=True)

        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nSaved -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
