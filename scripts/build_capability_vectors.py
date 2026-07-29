"""Build per-model "Capability Vector" ground truth from mix-instruct bartscore.

For each model in the pool, forms one vector indexed by a *unified* set of
prompt_ids (only prompts where ALL pool models have a scored candidate, so
every vector is dense and the same index position = the same prompt across
all models). Each element is that model's bartscore on that prompt
(exp-transformed, matching the convention already used elsewhere in this
codebase -- e.g. MixInstructOracle, compute_descriptors_perplexity.py --
since raw bartscore is a negative log-likelihood). Each vector is then
L2-normalized, same convention as the FP descriptors, so it's directly
comparable/reusable anywhere cosine similarity is used.

Output: local_descriptors/mix-instruct-capability/{model}.npy (one per
pool model, same shape, same prompt order) + a companion
prompt_ids.json recording exactly which prompts were used, for
reproducibility.
"""
import json
import math
from pathlib import Path

import numpy as np
from datasets import load_dataset, concatenate_datasets

POOL_PATH = "experts/pool-mix-instruct-7.json"
OUT_DIR = Path("local_descriptors/mix-instruct-capability")
SCORE_KEY = "bartscore"

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

    # per_prompt[prompt_id] = {model_hf_name: bartscore}
    per_prompt = {}
    for rec in raw:
        scores = {}
        for cand in rec["candidates"]:
            hf_name = NAME_TO_HF.get(cand["model"])
            if hf_name is None or hf_name not in pool_set:
                continue
            sc = cand["scores"].get(SCORE_KEY)
            if sc is None:
                continue
            if hf_name not in scores or sc > scores[hf_name]:
                scores[hf_name] = sc
        if len(scores) == len(pool):  # dense: every pool model scored on this prompt
            per_prompt[rec["id"]] = scores

    prompt_ids = sorted(per_prompt.keys())
    n = len(prompt_ids)
    print(f"Prompts with all {len(pool)} pool models scored (dense): {n} "
          f"out of {len(raw)} total ({n/len(raw)*100:.1f}%)")

    if n == 0:
        raise RuntimeError("No prompts have all pool models scored -- check pool/NAME_TO_HF.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for model in pool:
        raw_scores = np.array(
            [math.exp(per_prompt[pid][model]) for pid in prompt_ids],
            dtype=np.float64,
        )
        norm = np.linalg.norm(raw_scores) + 1e-12
        vec = (raw_scores / norm).astype(np.float32)
        out_path = OUT_DIR / f"{model}.npy"
        np.save(out_path, vec)
        print(f"  {model:55s} shape={vec.shape}  "
              f"raw range=[{raw_scores.min():.4f}, {raw_scores.max():.4f}]  "
              f"saved -> {out_path}")

    with open(OUT_DIR / "prompt_ids.json", "w") as f:
        json.dump(prompt_ids, f)
    print(f"\nSaved {len(pool)} capability vectors (dim={n}) + prompt_ids.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
