"""Build the "ceiling" FP for the unseen-model recovery experiment:
each model's descriptor IS its own bartscore vector -- i.e. a FP that
is capability-aligned by construction, giving an upper-bound reference
for how well unseen-model recovery *could* work if the FP perfectly
tracked capability.

Prompts are split into two DISJOINT sets to avoid any appearance of
circularity between "prompts used to build the ceiling FP" and
"prompts used later to evaluate unseen-model recovery":
  - set A: used to build each model's ceiling FP (this script)
  - set B: reserved for the recovery-rate evaluation (a later script)

Output: local_descriptors/mix-instruct-capability-ceiling/{model}.npy
        (11 models, dense over set A) +
        local_descriptors/mix-instruct-capability-ceiling/split_info.json
        (set A / set B prompt_ids, so the later eval script uses the
        exact same disjoint split)
"""
import json
import math
from pathlib import Path

import numpy as np
from datasets import load_dataset, concatenate_datasets

POOL_PATH = "experts/pool-mix-instruct-11.json"
OUT_DIR = Path("local_descriptors/mix-instruct-capability-ceiling")
SCORE_KEY = "bartscore"
SET_A_FRACTION = 0.8   # build-FP fraction; the rest (set B) is held out for eval
SPLIT_SEED = 42

NAME_TO_HF = {
    "vicuna-13b-1.1": "eachadea__vicuna-13b-1.1",
    "alpaca-native": "chavinlo__alpaca-native",
    "dolly-v2-12b": "databricks__dolly-v2-12b",
    "stablelm-tuned-alpha-7b": "stabilityai__stablelm-tuned-alpha-7b",
    "oasst-sft-4-pythia-12b-epoch-3.5": "OpenAssistant__oasst-sft-4-pythia-12b-epoch-3.5",
    "koala-7B-HF": "TheBloke__koala-7B-HF",
    "llama-7b-hf-baize-lora-bf16": "mosesjun0h__llama-7b-hf-baize-lora-bf16",
    "flan-t5-xxl": "google__flan-t5-xxl",
    "chatglm-6b": "THUDM__chatglm-6b",
    "moss-moon-003-sft": "fnlp__moss-moon-003-sft",
    "mpt-7b-instruct": "mosaicml__mpt-7b-instruct",
    "mpt-7b": "mosaicml__mpt-7b-instruct",
}


def main():
    pool = json.load(open(POOL_PATH))
    pool_set = set(pool)
    print(f"Pool ({len(pool)}): {pool}\n")

    raw = concatenate_datasets([
        load_dataset("llm-blender/mix-instruct", split="train"),
        load_dataset("llm-blender/mix-instruct", split="validation"),
    ])
    print(f"Loaded {len(raw)} mix-instruct rows (train+validation)")

    # dense: keep only prompts where ALL 11 pool models have a bartscore
    per_prompt = {}
    for rec in raw:
        scores = {}
        for cand in rec["candidates"]:
            hf_name = NAME_TO_HF.get(cand["model"])
            if hf_name is None or hf_name not in pool_set:
                continue
            sc = cand["scores"].get(SCORE_KEY)
            if sc is not None:
                scores[hf_name] = sc
        if len(scores) == len(pool):
            per_prompt[rec["id"]] = scores

    all_prompt_ids = sorted(per_prompt.keys())
    n_total = len(all_prompt_ids)
    print(f"Dense prompts (all {len(pool)} models scored): {n_total} / {len(raw)} "
          f"({n_total/len(raw)*100:.1f}%)\n")

    # ---- disjoint split: set A (build FP) / set B (reserved for eval) ----
    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = list(all_prompt_ids)
    rng.shuffle(shuffled)
    n_a = int(n_total * SET_A_FRACTION)
    set_a = sorted(shuffled[:n_a])
    set_b = sorted(shuffled[n_a:])

    overlap = set(set_a) & set(set_b)
    assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} prompts appear in both sets!"
    assert len(set_a) + len(set_b) == n_total

    print("=== Split verification ===")
    print(f"set A (build ceiling FP): {len(set_a)} prompts ({len(set_a)/n_total*100:.1f}%)")
    print(f"set B (reserved for recovery eval): {len(set_b)} prompts ({len(set_b)/n_total*100:.1f}%)")
    print(f"overlap between set A and set B: {len(overlap)} (must be 0) -- {'OK' if len(overlap)==0 else 'FAILED'}")

    # ---- build ceiling FP: each model's own bartscore vector over set A ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dims = set()
    for model in pool:
        raw_scores = np.array(
            [math.exp(per_prompt[pid][model]) for pid in set_a],
            dtype=np.float64,
        )
        norm = np.linalg.norm(raw_scores) + 1e-12
        vec = (raw_scores / norm).astype(np.float32)
        dims.add(vec.shape[0])
        np.save(OUT_DIR / f"{model}.npy", vec)
        print(f"  {model:55s} shape={vec.shape}  raw range=[{raw_scores.min():.4f}, {raw_scores.max():.4f}]")

    print(f"\n=== Dimension-matching check ===")
    print(f"distinct dims across {len(pool)} models: {dims} -- {'OK, all identical' if len(dims)==1 else 'MISMATCH!'}")

    with open(OUT_DIR / "split_info.json", "w") as f:
        json.dump({
            "set_a_build_fp": set_a,
            "set_b_eval_reserved": set_b,
            "seed": SPLIT_SEED,
            "set_a_fraction": SET_A_FRACTION,
        }, f)
    print(f"\nSaved {len(pool)} ceiling FPs (dim={dims.pop()}) + split_info.json -> {OUT_DIR}")


if __name__ == "__main__":
    main()
