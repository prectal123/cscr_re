"""One-off check: is the mix-instruct dataset's own "best expert" label
already skewed toward one model in our 7-model pool, before any router
was ever trained? If so, a router collapsing to that same model isn't
necessarily failing to learn -- it could be close to Bayes-optimal given
a lopsided label distribution. This replicates MixInstructOracle's label
logic directly (no torch/router import needed) to keep the dependency
footprint small.
"""
import json
import math
from collections import Counter

from datasets import load_dataset

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

MARGIN = 1e-1
SCORE_KEY = "bartscore"

pool = json.load(open("experts/pool-mix-instruct-7.json"))
pool_set = set(pool)
print(f"Pool ({len(pool)}): {pool}\n")

for split in ["train", "validation"]:
    raw = load_dataset("llm-blender/mix-instruct", split=split)

    label_count = Counter()      # how often each model is *among* the winners (label=1)
    sole_winner_count = Counter()  # how often each model is the *only* winner
    n_valid = 0
    n_skipped_no_pool_candidate = 0

    for rec in raw:
        scores = {}
        for cand in rec["candidates"]:
            hf_name = NAME_TO_HF.get(cand["model"])
            if hf_name is None or hf_name not in pool_set:
                continue
            sc = cand["scores"].get(SCORE_KEY)
            if sc is None:
                continue
            sc = math.exp(sc)
            if hf_name not in scores or sc > scores[hf_name]:
                scores[hf_name] = sc

        if not scores:
            n_skipped_no_pool_candidate += 1
            continue

        n_valid += 1
        best_val = max(scores.values())
        winners = [m for m, sc in scores.items() if best_val - sc <= MARGIN]
        for m in winners:
            label_count[m] += 1
        if len(winners) == 1:
            sole_winner_count[winners[0]] += 1

    print(f"=== split: {split} ===")
    print(f"valid samples (>=1 pool model scored): {n_valid}  "
          f"(skipped, no pool candidate: {n_skipped_no_pool_candidate})")
    print(f"{'model':55s} {'label=1 count':>14s} {'label=1 %':>10s} "
          f"{'sole winner count':>18s} {'sole winner %':>14s}")
    for m in pool:
        lc = label_count.get(m, 0)
        sw = sole_winner_count.get(m, 0)
        print(f"{m:55s} {lc:14d} {lc/n_valid*100:9.1f}% {sw:18d} {sw/n_valid*100:13.1f}%")
    print()
