"""Pseudo Ceiling FP for MixInstruct -- fills the "not tested" gap in the
3-benchmark comparison table. Unlike the existing "Ceiling FP" (192 k-means
PSEUDO-CATEGORY averages of bartscore, build_ceiling_fp_centered.py), this
selects individual high-variance PROMPTS directly (no clustering/averaging
step) from Set A, mean-centers, L2-normalizes, and runs the same kNN
unseen-recovery test on the reserved Set B split.

Note: MixInstruct has no real task categories (unlike RouterBench's eval_name
or LLMRouterBench's per-dataset structure), so probe selection here is a
straight top-variance selection across all of Set A, not stratified by
category -- flagged explicitly since it's a methodological simplification
relative to the other two benchmarks' Pseudo Ceiling construction.
"""
import json
import math
from pathlib import Path

import numpy as np
from datasets import load_dataset, concatenate_datasets
from scipy import stats
from scipy.stats import spearmanr

POOL_PATH = "experts/pool-mix-instruct-11.json"
SPLIT_INFO_PATH = Path("local_descriptors/mix-instruct-capability-ceiling/split_info.json")
OUT_DIR = Path("local_descriptors/mix-instruct-analysis")
PSEUDO_CEILING_DIR = Path("local_descriptors/mix-instruct-pseudo-ceiling")
SCORE_KEY = "bartscore"
N_PROBES = 512

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


def load_data():
    pool = json.load(open(POOL_PATH))
    pool_set = set(pool)
    split_info = json.load(open(SPLIT_INFO_PATH))
    set_a_ids = set(split_info["set_a_build_fp"])
    set_b_ids = set(split_info["set_b_eval_reserved"])
    print(f"Pool ({len(pool)}), Set A size: {len(set_a_ids)}, Set B size: {len(set_b_ids)}", flush=True)

    print("Loading mix-instruct dataset...", flush=True)
    raw = concatenate_datasets([
        load_dataset("llm-blender/mix-instruct", split="train"),
        load_dataset("llm-blender/mix-instruct", split="validation"),
    ])

    setA_scores = {}  # pid -> {model: exp(bartscore)}
    setB_scores = {}
    for rec in raw:
        pid = rec["id"]
        if pid not in set_a_ids and pid not in set_b_ids:
            continue
        scores = {}
        for cand in rec["candidates"]:
            hf_name = NAME_TO_HF.get(cand["model"])
            if hf_name is None or hf_name not in pool_set:
                continue
            sc = cand["scores"].get(SCORE_KEY)
            if sc is not None:
                scores[hf_name] = math.exp(sc)
        if len(scores) != len(pool):
            continue
        if pid in set_a_ids:
            setA_scores[pid] = scores
        elif pid in set_b_ids:
            setB_scores[pid] = scores

    print(f"Recovered full-pool scores: Set A {len(setA_scores)}, Set B {len(setB_scores)}", flush=True)
    return pool, setA_scores, setB_scores


def build_pseudo_ceiling_fp(pool, setA_scores):
    print("\nBuilding Pseudo Ceiling FP (global high-variance prompt selection, no clustering)...", flush=True)
    ids = sorted(setA_scores.keys())
    raw = np.array([[setA_scores[pid][m] for m in pool] for pid in ids])  # (n, 11)
    variance = raw.var(axis=1)
    top_idx = np.argsort(-variance)[:N_PROBES]
    print(f"  selected {len(top_idx)} probes out of {len(ids)} Set A prompts", flush=True)

    probe_raw = raw[top_idx]
    pool_mean = probe_raw.mean(axis=1, keepdims=True)
    centered = probe_raw - pool_mean

    PSEUDO_CEILING_DIR.mkdir(parents=True, exist_ok=True)
    for j, m in enumerate(pool):
        vec = centered[:, j]
        vec = (vec / (np.linalg.norm(vec) + 1e-12)).astype(np.float32)
        np.save(PSEUDO_CEILING_DIR / f"{m}.npy", vec)

    E = np.stack([np.load(PSEUDO_CEILING_DIR / f"{m}.npy") for m in pool])
    sim = E @ E.T
    off = sim[~np.eye(len(pool), dtype=bool)]
    print(f"  pairwise cosine sim: mean={off.mean():.4f} std={off.std():.4f} min={off.min():.4f} max={off.max():.4f}",
          flush=True)


def knn_test(pool, setB_scores, desc_dir, fp_name):
    print(f"\n{'='*60}\nkNN unseen-recovery test: {fp_name}\n{'='*60}", flush=True)
    E = np.stack([np.load(desc_dir / f"{m}.npy") for m in pool])
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    sim_full = E @ E.T

    ids = sorted(setB_scores.keys())
    true_scores = np.array([[setB_scores[pid][m] for m in pool] for pid in ids])

    fp_rhos, uniform_rhos = [], []
    for i, held_out in enumerate(pool):
        others_idx = [j for j in range(len(pool)) if j != i]
        sims = sim_full[i, others_idx]
        w = np.clip(sims, 0, None)
        if w.sum() < 1e-9:
            w = np.ones_like(w)
        w = w / w.sum()

        other_scores = true_scores[:, others_idx]
        true_m = true_scores[:, i]
        fp_proxy = other_scores @ w
        uniform_proxy = other_scores.mean(axis=1)

        fp_rho, _ = spearmanr(fp_proxy, true_m)
        uni_rho, _ = spearmanr(uniform_proxy, true_m)
        fp_rhos.append(fp_rho)
        uniform_rhos.append(uni_rho)
        print(f"  held out {held_out:35s} FP-proxy rho={fp_rho:.4f}  uniform-proxy rho={uni_rho:.4f}", flush=True)

    fp_rhos, uniform_rhos = np.array(fp_rhos), np.array(uniform_rhos)
    delta = fp_rhos - uniform_rhos
    t, p = stats.ttest_rel(fp_rhos, uniform_rhos)
    print(f"\nmean FP rho={fp_rhos.mean():.4f}  mean uniform rho={uniform_rhos.mean():.4f}  "
          f"mean delta={delta.mean():+.4f}  paired t-test p={p:.4f}  ({(delta>0).sum()}/{len(pool)} folds improved)",
          flush=True)
    return {
        "fp_rho": fp_rhos.tolist(), "uniform_rho": uniform_rhos.tolist(),
        "mean_fp_rho": float(fp_rhos.mean()), "mean_uniform_rho": float(uniform_rhos.mean()),
        "mean_delta": float(delta.mean()), "paired_t_p": float(p),
        "n_folds_improved": int((delta > 0).sum()),
    }


def main():
    pool, setA_scores, setB_scores = load_data()
    build_pseudo_ceiling_fp(pool, setA_scores)
    result = knn_test(pool, setB_scores, PSEUDO_CEILING_DIR, "Pseudo Ceiling")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "mixinstruct_pseudo_ceiling_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
