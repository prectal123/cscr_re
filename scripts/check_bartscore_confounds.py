"""Does bartscore track response length / reference length, rather than
(or in addition to) actual answer quality? Empirical check on a sample.
"""
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr

POOL_PATH = "experts/pool-mix-instruct-7.json"
SCORE_KEY = "bartscore"
N_SAMPLE_ROWS = 4000

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
    pool = set(json.load(open(POOL_PATH)))
    raw = load_dataset("llm-blender/mix-instruct", split="train")
    raw = raw.select(range(N_SAMPLE_ROWS))

    cand_len = []      # candidate response length (words)
    bartscores = []
    ref_len_per_pair = []   # reference (output) length, repeated per candidate
    ref_len_by_prompt = []  # one entry per prompt, for prompt-level analysis
    prompt_bart_spread = []  # (max-min) bartscore across pool models, per prompt

    for rec in raw:
        ref_words = len(rec["output"].split())
        pool_scores = []
        for cand in rec["candidates"]:
            hf_name = NAME_TO_HF.get(cand["model"])
            if hf_name is None or hf_name not in pool:
                continue
            sc = cand["scores"].get(SCORE_KEY)
            if sc is None:
                continue
            n_words = len(cand["text"].split())
            cand_len.append(n_words)
            bartscores.append(sc)
            ref_len_per_pair.append(ref_words)
            pool_scores.append(sc)
        if len(pool_scores) >= 2:
            ref_len_by_prompt.append(ref_words)
            prompt_bart_spread.append(max(pool_scores) - min(pool_scores))

    cand_len = np.array(cand_len)
    bartscores = np.array(bartscores)
    ref_len_per_pair = np.array(ref_len_per_pair)
    ref_len_by_prompt = np.array(ref_len_by_prompt)
    prompt_bart_spread = np.array(prompt_bart_spread)

    print(f"n candidate-level (model, prompt) pairs: {len(cand_len)}")
    print(f"n prompts (>=2 pool models scored): {len(ref_len_by_prompt)}\n")

    print("=== Does candidate RESPONSE length correlate with its own bartscore? ===")
    r, p = pearsonr(cand_len, bartscores)
    rho, ps = spearmanr(cand_len, bartscores)
    print(f"Pearson r={r:.4f} (p={p:.2e})   Spearman rho={rho:.4f} (p={ps:.2e})")
    print(f"(response length range: {cand_len.min()}-{cand_len.max()} words, "
          f"median={int(np.median(cand_len))})")

    print("\n=== Does REFERENCE (output) length correlate with bartscore, "
          "for the SAME candidates? ===")
    r2, p2 = pearsonr(ref_len_per_pair, bartscores)
    rho2, ps2 = spearmanr(ref_len_per_pair, bartscores)
    print(f"Pearson r={r2:.4f} (p={p2:.2e})   Spearman rho={rho2:.4f} (p={ps2:.2e})")

    print("\n=== Does REFERENCE length correlate with how much models DISAGREE "
          "(bartscore spread) on that prompt? ===")
    r3, p3 = pearsonr(ref_len_by_prompt, prompt_bart_spread)
    rho3, ps3 = spearmanr(ref_len_by_prompt, prompt_bart_spread)
    print(f"Pearson r={r3:.4f} (p={p3:.2e})   Spearman rho={rho3:.4f} (p={ps3:.2e})")
    print(f"(reference length range: {ref_len_by_prompt.min()}-{ref_len_by_prompt.max()} words)")

    # short-reference subgroup: does bartscore spread blow up for very short references?
    short_mask = ref_len_by_prompt <= 5
    long_mask = ref_len_by_prompt > 5
    print(f"\nPrompts with reference <=5 words: {short_mask.sum()}, "
          f"mean bartscore spread={prompt_bart_spread[short_mask].mean():.3f}")
    print(f"Prompts with reference >5 words: {long_mask.sum()}, "
          f"mean bartscore spread={prompt_bart_spread[long_mask].mean():.3f}")


if __name__ == "__main__":
    main()
