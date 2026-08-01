"""Lightweight-20-only pool, 22-category version. Excludes the 13 flagship
models entirely: (a) removes the tier-gap that dominated pooled-variance
probe selection in the 33-model run (only one capability tier now, so plain
variance ranking is safe again), and (b) unlocks 22 dataset categories
instead of 8, since flagship coverage gaps (e.g. missing deepseek-v3.1-terminus
on mmlupro) were the main blocker on most categories.
"""
import glob
import json
from pathlib import Path

import numpy as np

BENCH_ROOT = Path("C:/Users/user/git_seminar/LLMRouterBench/results/bench-release")

DS_PATHS = {
    "aime": "aime/hybrid",
    "arcc": "arcc/test",
    "arenahard": "arenahard",
    "arenahard_coding": "arenahard_coding",
    "arenahard_creative_writing": "arenahard_creative_writing",
    "arenahard_math": "arenahard_math",
    "bbh": "bbh/test",
    "emorynlp": "emorynlp/test",
    "finqa": "finqa/test",
    "gpqa": "gpqa/test",
    "humaneval": "humaneval/test",
    "kandk": "kandk/test",
    "korbench": "korbench/test",
    "livecodebench": "livecodebench/test",
    "livemathbench": "livemathbench/test",
    "math500": "math500/test",
    "mathbench": "mathbench/test",
    "mbpp": "mbpp/test",
    "medqa": "medqa/test",
    "meld": "meld/test",
    "mmlupro": "mmlupro/test_1000",
    "winogrande": "winogrande/valid",
}
DATASETS = list(DS_PATHS.keys())
assert len(DATASETS) == 22

# verified (2026-08-01): full, non-missing coverage across all 22 datasets above
MODELS_20 = [
    "DeepHermes-3-Llama-3-8B-Preview", "DeepSeek-R1-0528-Qwen3-8B", "DeepSeek-R1-Distill-Qwen-7B",
    "Fin-R1", "GLM-Z1-9B-0414", "Intern-S1-mini", "Llama-3.1-8B-Instruct", "Llama-3.1-8B-UltraMedical",
    "Llama-3.1-Nemotron-Nano-8B-v1", "MiMo-7B-RL-0530", "MiniCPM4.1-8B", "NVIDIA-Nemotron-Nano-9B-v2",
    "OpenThinker3-7B", "Qwen2.5-Coder-7B-Instruct", "Qwen3-8B", "cogito-v1-preview-llama-8B",
    "gemma-2-9b-it", "glm-4-9b-chat", "granite-3.3-8b-instruct", "internlm3-8b-instruct",
]
assert len(MODELS_20) == 20
NAME_TO_SAFE = {m: m.replace("/", "__") for m in MODELS_20}


def load_dataset_records(ds_name):
    """Returns dict: model -> list of records (aligned by index across models)."""
    path = BENCH_ROOT / DS_PATHS[ds_name]
    out = {}
    for m in MODELS_20:
        jsons = glob.glob(str(path / m / "*.json"))
        if not jsons:
            raise FileNotFoundError(f"no json for {ds_name}/{m}")
        d = json.load(open(jsons[0], encoding="utf-8"))
        out[m] = d["records"]
    lengths = {m: len(recs) for m, recs in out.items()}
    assert len(set(lengths.values())) == 1, f"{ds_name}: mismatched record counts {lengths}"
    n = list(lengths.values())[0]
    for idx in [0, n // 2, n - 1]:
        qs = {out[m][idx]["origin_query"] for m in MODELS_20}
        assert len(qs) == 1, f"{ds_name} idx={idx}: origin_query mismatch across models"
    return out


def build_wide_table(ds_name):
    """Returns (queries: list[str], scores: (N,20) array, costs: (N,20) array,
    raw_outputs: dict[model] -> list[str] (len N))."""
    recs_by_model = load_dataset_records(ds_name)
    n = len(recs_by_model[MODELS_20[0]])
    queries = [recs_by_model[MODELS_20[0]][i]["origin_query"] for i in range(n)]
    scores = np.zeros((n, len(MODELS_20)))
    costs = np.zeros((n, len(MODELS_20)))
    raw_outputs = {m: [] for m in MODELS_20}
    for j, m in enumerate(MODELS_20):
        for i, r in enumerate(recs_by_model[m]):
            scores[i, j] = float(r.get("score") or 0.0)
            costs[i, j] = float(r.get("cost") or 0.0)
            raw_outputs[m].append(r.get("raw_output") or "")
    return queries, scores, costs, raw_outputs
