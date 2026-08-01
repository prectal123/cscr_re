"""Fixes the diagnosed probe-selection bug: original probe selection ranked
by variance across ALL 33 models pooled together, which got dominated by the
large lightweight-vs-flagship capability gap -- so probes barely captured
genuine within-flagship (or even within-lightweight) discriminative signal,
producing significantly INVERTED AUC for several flagship models.

Fix: score each candidate probe by variance WITHIN the lightweight-20 group
AND variance WITHIN the flagship-13 group separately, require both to be
reasonably high (use their sum) so selected probes are informative for
distinguishing models within EACH tier, not just the tier gap itself.

Saved under a new "_v2" path -- does not touch the original v1 split/FPs.
"""
import json
import pickle
from pathlib import Path

import numpy as np

import common

SPLIT_SEED = 42
SET_A_FRACTION = 0.8
PROBES_PER_DATASET = 24
OUT_DIR = Path("local_descriptors/llmrouterbench_v2")

LIGHTWEIGHT_20 = {
    "DeepHermes-3-Llama-3-8B-Preview", "DeepSeek-R1-0528-Qwen3-8B", "DeepSeek-R1-Distill-Qwen-7B",
    "Fin-R1", "GLM-Z1-9B-0414", "Intern-S1-mini", "Llama-3.1-8B-Instruct", "Llama-3.1-8B-UltraMedical",
    "Llama-3.1-Nemotron-Nano-8B-v1", "MiMo-7B-RL-0530", "MiniCPM4.1-8B", "NVIDIA-Nemotron-Nano-9B-v2",
    "OpenThinker3-7B", "Qwen2.5-Coder-7B-Instruct", "Qwen3-8B", "cogito-v1-preview-llama-8B",
    "gemma-2-9b-it", "glm-4-9b-chat", "granite-3.3-8b-instruct", "internlm3-8b-instruct",
}
LIGHT_IDX = [i for i, m in enumerate(common.MODELS_33) if m in LIGHTWEIGHT_20]
FLAG_IDX = [i for i, m in enumerate(common.MODELS_33) if m not in LIGHTWEIGHT_20]


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
            "raw_outputs": {m: [raw_outputs[m][i] for i in idx_a] for m in common.MODELS_33},
        }
        setB_data[ds] = {
            "idx": idx_b.tolist(), "scores": scores[idx_b], "costs": costs[idx_b],
            "queries": [queries[i] for i in idx_b],
        }

        sub_scores_a = scores[idx_a]
        var_light = sub_scores_a[:, LIGHT_IDX].var(axis=1)
        var_flag = sub_scores_a[:, FLAG_IDX].var(axis=1)
        combined_score = var_light + var_flag  # reward probes discriminative in BOTH tiers

        order = np.argsort(-combined_score)
        top_local = order[:PROBES_PER_DATASET]
        for local_i in top_local:
            global_i = int(idx_a[local_i])
            all_probe_info.append({
                "dataset": ds, "local_idx_in_setA": int(local_i), "global_idx": global_i,
                "query": queries[global_i],
                "var_light": float(var_light[local_i]), "var_flag": float(var_flag[local_i]),
            })
        print(f"{ds}: n={n} (SetA={len(idx_a)}, SetB={len(idx_b)})  "
              f"selected var_light range=[{var_light[top_local].min():.4f},{var_light[top_local].max():.4f}]  "
              f"var_flag range=[{var_flag[top_local].min():.4f},{var_flag[top_local].max():.4f}]")

    print(f"\nTotal probes selected: {len(all_probe_info)}")
    with open(OUT_DIR / "probe_info.json", "w", encoding="utf-8") as f:
        json.dump(all_probe_info, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "setA_setB_split.pkl", "wb") as f:
        pickle.dump({"setA": setA_data, "setB": setB_data}, f)
    print(f"Saved -> {OUT_DIR / 'probe_info.json'}, {OUT_DIR / 'setA_setB_split.pkl'}")


if __name__ == "__main__":
    main()
