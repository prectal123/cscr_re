"""Set A/B split (80/20, per-dataset, fixed seed -- matches RouterBench
convention) + variance-based stratified probe selection (192 total, split
evenly across the 8 datasets = 24 each, chosen by highest cross-model score
variance on Set A only, per user's design 2026-08-01).

Saves everything needed downstream: split indices, probe indices/queries,
per-model score/cost tables for Set A and Set B.
"""
import json
import pickle
from pathlib import Path

import numpy as np

import common

SPLIT_SEED = 42
SET_A_FRACTION = 0.8
PROBES_PER_DATASET = 24  # 8 datasets x 24 = 192
OUT_DIR = Path("local_descriptors/llmrouterbench")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SPLIT_SEED)

    all_probe_info = []  # list of dicts: dataset, local_idx, query
    setA_data = {}  # ds -> dict(idx_a, scores_a, costs_a, raw_outputs_a, queries_a)
    setB_data = {}

    for ds in common.DATASETS:
        queries, scores, costs, raw_outputs = common.build_wide_table(ds)
        n = len(queries)
        perm = rng.permutation(n)
        n_a = int(n * SET_A_FRACTION)
        idx_a, idx_b = perm[:n_a], perm[n_a:]

        setA_data[ds] = {
            "idx": idx_a.tolist(),
            "scores": scores[idx_a],
            "costs": costs[idx_a],
            "queries": [queries[i] for i in idx_a],
            "raw_outputs": {m: [raw_outputs[m][i] for i in idx_a] for m in common.MODELS_33},
        }
        setB_data[ds] = {
            "idx": idx_b.tolist(),
            "scores": scores[idx_b],
            "costs": costs[idx_b],
            "queries": [queries[i] for i in idx_b],
        }

        # variance-based probe selection on Set A only
        var_a = scores[idx_a].var(axis=1)  # variance across the 33 models, per query
        order = np.argsort(-var_a)
        top_local = order[:PROBES_PER_DATASET]  # indices into idx_a
        for local_i in top_local:
            global_i = int(idx_a[local_i])
            all_probe_info.append({
                "dataset": ds, "local_idx_in_setA": int(local_i), "global_idx": global_i,
                "query": queries[global_i], "variance": float(var_a[local_i]),
            })
        print(f"{ds}: n={n} (SetA={len(idx_a)}, SetB={len(idx_b)})  "
              f"probe variance range selected: [{var_a[top_local].min():.4f}, {var_a[top_local].max():.4f}]  "
              f"(all-Set-A variance range: [{var_a.min():.4f}, {var_a.max():.4f}], mean={var_a.mean():.4f})")

    print(f"\nTotal probes selected: {len(all_probe_info)}")

    with open(OUT_DIR / "probe_info.json", "w", encoding="utf-8") as f:
        json.dump(all_probe_info, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "setA_setB_split.pkl", "wb") as f:
        pickle.dump({"setA": setA_data, "setB": setB_data}, f)

    print(f"Saved -> {OUT_DIR / 'probe_info.json'}, {OUT_DIR / 'setA_setB_split.pkl'}")


if __name__ == "__main__":
    main()
