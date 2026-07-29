"""Diagnostic: how spread out / cohesive are the 7 capability vectors?

Checks two things:
1. Per-model variance of raw bartscore across prompts (is there any
   prompt-specific signal at all, or is each model's score nearly constant?)
2. Pairwise similarity between the 7 models' capability profiles, using
   BOTH cosine similarity (on the saved, exp-transformed, L2-normalized
   vectors) AND Pearson correlation (on the raw bartscore, pre-exp) --
   because cosine similarity between very-high-dimensional, all-nonnegative
   vectors is known to be structurally inflated (same issue flagged earlier
   for the logit/perplexity descriptors), so comparing both side by side
   shows whether that artifact is actually happening here too.
"""
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from datasets import load_dataset, concatenate_datasets

POOL_PATH = "experts/pool-mix-instruct-7.json"
CAP_DIR = Path("local_descriptors/mix-instruct-capability")
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


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    pool = json.load(open(POOL_PATH))
    prompt_ids = json.load(open(CAP_DIR / "prompt_ids.json"))
    print(f"Pool ({len(pool)} models), {len(prompt_ids)} prompts\n")

    # --- reload raw (pre-exp, pre-normalize) bartscore to check variance
    # and Pearson correlation without the exp/L2-norm transforms in the way
    raw = concatenate_datasets([
        load_dataset("llm-blender/mix-instruct", split="train"),
        load_dataset("llm-blender/mix-instruct", split="validation"),
    ])
    pool_set = set(pool)
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

    raw_bartscore = {
        m: np.array([per_prompt[pid][m] for pid in prompt_ids], dtype=np.float64)
        for m in pool
    }

    # --- 1. per-model variance / spread of raw bartscore
    print("=== Per-model raw bartscore spread ===")
    print(f"{'model':55s} {'mean':>9s} {'std':>9s} {'min':>9s} {'max':>9s}")
    for m in pool:
        v = raw_bartscore[m]
        print(f"{m:55s} {v.mean():9.3f} {v.std():9.3f} {v.min():9.3f} {v.max():9.3f}")

    # --- 2a. cosine similarity on the SAVED (exp + L2-normalized) vectors
    cap_vecs = {m: np.load(CAP_DIR / f"{m}.npy") for m in pool}
    print("\n=== Pairwise similarity: cosine (exp-transformed, L2-normalized vectors) ===")
    cos_vals = []
    for a, b in combinations(pool, 2):
        c = cosine(cap_vecs[a], cap_vecs[b])
        cos_vals.append(c)
        print(f"  {a:45s} <-> {b:45s}  cos={c:.4f}")
    cos_vals = np.array(cos_vals)
    print(f"cosine: mean={cos_vals.mean():.4f} min={cos_vals.min():.4f} max={cos_vals.max():.4f} std={cos_vals.std():.4f}")

    # --- 2b. Pearson correlation on RAW bartscore (mean-centered, so the
    # nonnegative-orthant inflation from 2a can't happen here)
    print("\n=== Pairwise similarity: Pearson correlation (raw bartscore) ===")
    pear_vals = []
    for a, b in combinations(pool, 2):
        r = float(np.corrcoef(raw_bartscore[a], raw_bartscore[b])[0, 1])
        pear_vals.append(r)
        print(f"  {a:45s} <-> {b:45s}  pearson r={r:.4f}")
    pear_vals = np.array(pear_vals)
    print(f"pearson: mean={pear_vals.mean():.4f} min={pear_vals.min():.4f} max={pear_vals.max():.4f} std={pear_vals.std():.4f}")

    print("\n=== Verdict ===")
    print(f"cosine range is {cos_vals.max()-cos_vals.min():.4f} wide (all clustered near "
          f"{cos_vals.mean():.4f}? -> {'YES, likely inflated by nonnegative-orthant effect' if cos_vals.std() < 0.05 else 'spread looks real'})")
    print(f"pearson range is {pear_vals.max()-pear_vals.min():.4f} wide, mean={pear_vals.mean():.4f}")


if __name__ == "__main__":
    main()
