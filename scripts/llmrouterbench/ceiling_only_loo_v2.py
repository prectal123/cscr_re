"""Ceiling-only 33-fold LOO using the v2 (within-tier-variance) probes/FP --
tests whether fixing the diagnosed probe-selection bug (v1 was dominated by
the lightweight-vs-flagship gap) resolves the significant AUC inversion seen
for several flagship models in the v1 run. Perplexity not touched.
"""
import json
import pickle
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
import loo_recovery as loo  # reuse train_fold, evaluate_fold, precompute_embeddings, etc.
import loo_unseen_recovery as base
from transformers import AutoModel, AutoTokenizer

DATA_DIR = Path("local_descriptors/llmrouterbench_v2")
CEILING_DIR = DATA_DIR / "ceiling"
SEED = 0
BETA = 1.0
OUT_PATH = DATA_DIR / "ceiling_only_loo_results.json"


def load_split_v2():
    with open(DATA_DIR / "setA_setB_split.pkl", "rb") as f:
        return pickle.load(f)


def build_train_rows_v2(split):
    rows = []
    for ds in common.DATASETS:
        d = split["setA"][ds]
        n = len(d["queries"])
        for i in range(n):
            scores = {m: float(d["scores"][i, j]) for j, m in enumerate(common.MODELS_33)}
            rows.append((d["queries"][i], scores))
    return rows


def build_cost_dict_v2(split):
    all_costs = {m: [] for m in common.MODELS_33}
    for ds in common.DATASETS:
        d = split["setA"][ds]
        for j, m in enumerate(common.MODELS_33):
            all_costs[m].extend(d["costs"][:, j].tolist())
    return {m: float(np.mean(v)) for m, v in all_costs.items()}


def build_setB_eval_v2(split):
    queries, scores = [], []
    for ds in common.DATASETS:
        d = split["setB"][ds]
        queries.extend(d["queries"])
        scores.append(d["scores"])
    return queries, np.concatenate(scores, axis=0)


def main():
    print(f"DEVICE: {base.DEVICE}", flush=True)
    pool_33 = common.MODELS_33

    base._TOKENIZER = AutoTokenizer.from_pretrained(base.EMBED_MODEL)
    base._BASE_MODEL = AutoModel.from_pretrained(base.EMBED_MODEL).to(base.DEVICE)
    base._BASE_MODEL.eval()

    split = load_split_v2()
    train_rows = build_train_rows_v2(split)
    cost_dict = build_cost_dict_v2(split)
    setB_queries, true_scores = build_setB_eval_v2(split)
    print(f"train_rows={len(train_rows)}  setB={len(setB_queries)}", flush=True)

    setB_embeds = loo.precompute_embeddings(setB_queries, base._TOKENIZER, base._BASE_MODEL)
    cost_arr = np.array([cost_dict[m] for m in pool_33])
    cost_norm_33 = (cost_arr - cost_arr.min()) / (cost_arr.max() - cost_arr.min() + 1e-9)

    rng = random.Random(1)
    probe_idx = rng.sample(range(len(setB_queries)), min(200, len(setB_queries)))
    probe_texts = [setB_queries[i] for i in probe_idx]

    fold_results = []
    t0 = time.time()
    for held_out in pool_33:
        pool_32 = [m for m in pool_33 if m != held_out]
        print(f"held out: {held_out} ... (elapsed {time.time()-t0:.0f}s)", flush=True)
        head, names32 = loo.train_fold(pool_32, CEILING_DIR, train_rows, cost_dict, seed=SEED, balance_beta=BETA)
        r = loo.evaluate_fold(head, held_out, pool_32, CEILING_DIR, pool_33, setB_embeds, true_scores,
                               cost_norm_33, probe_texts)
        print(f"  n_oracle_is_M={r['n_oracle_is_M']}  oracle_match_rate={r['oracle_match_rate']:.4f}  "
              f"router_acc={r['router_overall_accuracy']:.4f}  AUC={r['auc_heldout_correctness']:.4f} "
              f"(p={r['auc_p']:.4f})", flush=True)
        fold_results.append(r)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(fold_results, f, indent=2)

    aucs = [r["auc_heldout_correctness"] for r in fold_results]
    n_sig_above = sum(1 for r in fold_results if r["auc_heldout_correctness"] > 0.5 and r["auc_p"] < 0.05)
    n_sig_below = sum(1 for r in fold_results if r["auc_heldout_correctness"] < 0.5 and r["auc_p"] < 0.05)
    print(f"\n--- Ceiling v2 summary ---", flush=True)
    print(f"mean AUC: {np.mean(aucs):.4f}", flush=True)
    print(f"significantly ABOVE chance: {n_sig_above}/33, BELOW chance: {n_sig_below}/33", flush=True)
    print(f"Saved -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
