"""Shared loading utilities for the LLMRouterBench extension (33-model pool,
8 clean categories, no missing data). Mirrors the RouterBench pipeline
(routerbench_knn_test.py / routerbench_loo_recovery.py) but for this larger,
independently-sourced benchmark, used to test whether a bigger model pool
changes the collapse/unseen-routing picture found on the 11-model RouterBench.
"""
import glob
import json
from pathlib import Path

import numpy as np

BENCH_ROOT = Path("C:/Users/user/git_seminar/LLMRouterBench/results/bench-release")

DS_PATHS = {
    "aime": "aime/hybrid",
    "arenahard": "arenahard",
    "arenahard_coding": "arenahard_coding",
    "arenahard_creative_writing": "arenahard_creative_writing",
    "arenahard_math": "arenahard_math",
    "gpqa": "gpqa/test",
    "livecodebench": "livecodebench/test",
    "livemathbench": "livemathbench/test",
}
DATASETS = list(DS_PATHS.keys())

# verified (2026-08-01) as the 33 models with complete, non-missing coverage
# across all 8 datasets above (mmlupro dropped -- missing deepseek-v3.1-terminus there)
MODELS_33 = [
    "DeepHermes-3-Llama-3-8B-Preview", "DeepSeek-R1-0528-Qwen3-8B", "DeepSeek-R1-Distill-Qwen-7B",
    "Fin-R1", "GLM-Z1-9B-0414", "Intern-S1-mini", "Llama-3.1-8B-Instruct", "Llama-3.1-8B-UltraMedical",
    "Llama-3.1-Nemotron-Nano-8B-v1", "MiMo-7B-RL-0530", "MiniCPM4.1-8B", "NVIDIA-Nemotron-Nano-9B-v2",
    "OpenThinker3-7B", "Qwen2.5-Coder-7B-Instruct", "Qwen3-8B", "cogito-v1-preview-llama-8B",
    "gemma-2-9b-it", "glm-4-9b-chat", "granite-3.3-8b-instruct", "internlm3-8b-instruct",
    # flagship 13
    "claude-sonnet-4", "deepseek-r1-0528", "deepseek-v3-0324", "deepseek-v3.1-terminus",
    "gemini-2.5-flash", "gemini-2.5-pro", "glm-4.6", "gpt-5", "gpt-5-chat", "intern-s1",
    "kimi-k2-0905", "qwen3-235b-a22b-2507", "qwen3-235b-a22b-thinking-2507",
]
assert len(MODELS_33) == 33
NAME_TO_SAFE = {m: m.replace("/", "__") for m in MODELS_33}


def load_dataset_records(ds_name):
    """Returns dict: model -> list of records (aligned by index across models)."""
    path = BENCH_ROOT / DS_PATHS[ds_name]
    out = {}
    for m in MODELS_33:
        jsons = glob.glob(str(path / m / "*.json"))
        if not jsons:
            raise FileNotFoundError(f"no json for {ds_name}/{m}")
        d = json.load(open(jsons[0], encoding="utf-8"))
        out[m] = d["records"]
    # sanity check: same length + same origin_query at a few spot-check indices
    lengths = {m: len(recs) for m, recs in out.items()}
    assert len(set(lengths.values())) == 1, f"{ds_name}: mismatched record counts {lengths}"
    n = list(lengths.values())[0]
    for idx in [0, n // 2, n - 1]:
        qs = {out[m][idx]["origin_query"] for m in MODELS_33}
        assert len(qs) == 1, f"{ds_name} idx={idx}: origin_query mismatch across models"
    return out


def build_wide_table(ds_name):
    """Returns (queries: list[str], scores: (N,33) array, costs: (N,33) array,
    raw_outputs: dict[model] -> list[str] (len N))."""
    recs_by_model = load_dataset_records(ds_name)
    n = len(recs_by_model[MODELS_33[0]])
    queries = [recs_by_model[MODELS_33[0]][i]["origin_query"] for i in range(n)]
    scores = np.zeros((n, len(MODELS_33)))
    costs = np.zeros((n, len(MODELS_33)))
    raw_outputs = {m: [] for m in MODELS_33}
    for j, m in enumerate(MODELS_33):
        for i, r in enumerate(recs_by_model[m]):
            scores[i, j] = float(r.get("score") or 0.0)
            costs[i, j] = float(r.get("cost") or 0.0)
            raw_outputs[m].append(r.get("raw_output") or "")
    return queries, scores, costs, raw_outputs
