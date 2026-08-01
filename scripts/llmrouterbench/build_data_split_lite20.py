"""80/20 Set A/B split + variance-based probe selection for the lightweight-20
pool, 22-dataset version. Plain pooled variance is safe here (unlike the
33-model run) since there's no lightweight-vs-flagship tier gap to dominate it.
"""
import json
import pickle
from pathlib import Path

import numpy as np

import common_lite20 as common

SPLIT_SEED = 42
SET_A_FRACTION = 0.8
PROBES_PER_DATASET = 24
OUT_DIR = Path("local_descriptors/llmrouterbench_lite20")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SPLIT_SEED)

    all_probe_info = []
    setA_data = {}
    setB_data = {}

    for ds in common.DATASETS:
        queries, scores, costs, raw_outputs = common.build_wide_table(ds)
        n = len(queries)
        perm = rng.permutation(n)
        n_a = int(n * SET_A_FRACTION)
        idx_a, idx_b = perm[:n_a], perm[n_a:]

        setA_data[ds] = {
            "idx": idx_a.tolist(), "scores": scores[idx_a], "costs": costs[idx_a],
            "queries": [queries[i] for i in idx_a],
            "raw_outputs": {m: [raw_outputs[m][i] for i in idx_a] for m in common.MODELS_20},
        }
        setB_data[ds] = {
            "idx": idx_b.tolist(), "scores": scores[idx_b], "costs": costs[idx_b],
            "queries": [queries[i] for i in idx_b],
        }

        sub_scores_a = scores[idx_a]
        var = sub_scores_a.var(axis=1)
        n_take = min(PROBES_PER_DATASET, len(idx_a))
        order = np.argsort(-var)
        top_local = order[:n_take]
        for local_i in top_local:
            global_i = int(idx_a[local_i])
            all_probe_info.append({
                "dataset": ds, "local_idx_in_setA": int(local_i), "global_idx": global_i,
                "query": queries[global_i], "var": float(var[local_i]),
            })
        print(f"{ds}: n={n} (SetA={len(idx_a)}, SetB={len(idx_b)})  "
              f"selected={n_take}  var range=[{var[top_local].min():.4f},{var[top_local].max():.4f}]")

    print(f"\nTotal probes selected: {len(all_probe_info)}")
    with open(OUT_DIR / "probe_info.json", "w", encoding="utf-8") as f:
        json.dump(all_probe_info, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "setA_setB_split.pkl", "wb") as f:
        pickle.dump({"setA": setA_data, "setB": setB_data}, f)
    print(f"Saved -> {OUT_DIR / 'probe_info.json'}, {OUT_DIR / 'setA_setB_split.pkl'}")


if __name__ == "__main__":
    main()
