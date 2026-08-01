"""Rebuild the ceiling FP at a manageable dimensionality (192, matching
Logit/Perplexity), by clustering Set A's prompts and using per-cluster
average bartscore as each dimension, instead of one dimension per raw
prompt (84,000-dim was technically fine but forced a huge MLP output
layer and made cross-FP-type comparisons awkward).

Steps:
1. Load Set A prompt texts (from the disjoint split saved earlier).
2. Embed each prompt with the same MiniLM backbone used by QueryEncoder
   (CLS-token pooling), so the clustering respects semantic similarity.
3. K-means into 192 clusters.
4. For each model, each cluster's value = mean bartscore (exp-transformed)
   of that model over the prompts in that cluster.
5. L2-normalize each model's resulting 192-dim vector.

Set B (eval-reserved) is untouched -- still in split_info.json, still
disjoint from anything used here.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

POOL_PATH = "experts/pool-mix-instruct-11.json"
OUT_DIR = Path("local_descriptors/mix-instruct-capability-ceiling")
SCORE_KEY = "bartscore"
N_CLUSTERS = 192
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
KMEANS_SEED = 0

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
    split_info = json.load(open(OUT_DIR / "split_info.json"))
    set_a_ids = set(split_info["set_a_build_fp"])
    print(f"Pool ({len(pool)}), Set A size: {len(set_a_ids)}\n")

    print("Reloading mix-instruct to get prompt text + bartscore for Set A...")
    raw = concatenate_datasets([
        load_dataset("llm-blender/mix-instruct", split="train"),
        load_dataset("llm-blender/mix-instruct", split="validation"),
    ])

    prompt_text = {}
    per_prompt_scores = {}
    for rec in raw:
        pid = rec["id"]
        if pid not in set_a_ids:
            continue
        scores = {}
        for cand in rec["candidates"]:
            hf_name = NAME_TO_HF.get(cand["model"])
            if hf_name is None or hf_name not in pool_set:
                continue
            sc = cand["scores"].get(SCORE_KEY)
            if sc is not None:
                scores[hf_name] = sc
        if len(scores) == len(pool):
            prompt_text[pid] = f"{rec['instruction']} {rec['input']}"
            per_prompt_scores[pid] = scores

    ids = sorted(prompt_text.keys())
    assert len(ids) == len(set_a_ids), f"mismatch: {len(ids)} vs {len(set_a_ids)}"
    print(f"Recovered text for all {len(ids)} Set A prompts -- OK\n")

    # ---- embed with MiniLM (CLS pooling), matching QueryEncoder's convention ----
    print(f"Loading {EMBED_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model = AutoModel.from_pretrained(EMBED_MODEL)
    model.eval()

    embeddings = np.zeros((len(ids), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for start in tqdm(range(0, len(ids), BATCH_SIZE), desc="embedding Set A"):
            batch_ids = ids[start:start + BATCH_SIZE]
            batch_text = [prompt_text[i] for i in batch_ids]
            enc = tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
            out = model(**enc).last_hidden_state[:, 0]  # CLS token
            embeddings[start:start + len(batch_ids)] = out.numpy()

    print(f"\nEmbedded {embeddings.shape[0]} prompts, dim={embeddings.shape[1]}")

    # ---- k-means into N_CLUSTERS ----
    print(f"Running k-means (k={N_CLUSTERS})...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=KMEANS_SEED, n_init=10)
    cluster_ids = km.fit_predict(embeddings)
    cluster_sizes = np.bincount(cluster_ids, minlength=N_CLUSTERS)
    print(f"Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
          f"mean={cluster_sizes.mean():.1f}, empty clusters={int((cluster_sizes==0).sum())}")

    # ---- build 192-dim ceiling FP: per-cluster mean bartscore, per model ----
    for model_name in pool:
        cluster_vals = np.zeros(N_CLUSTERS, dtype=np.float64)
        for c in range(N_CLUSTERS):
            members = [ids[i] for i in range(len(ids)) if cluster_ids[i] == c]
            if not members:
                cluster_vals[c] = 0.0
                continue
            vals = [math.exp(per_prompt_scores[pid][model_name]) for pid in members]
            cluster_vals[c] = float(np.mean(vals))
        norm = np.linalg.norm(cluster_vals) + 1e-12
        vec = (cluster_vals / norm).astype(np.float32)
        np.save(OUT_DIR / f"{model_name}.npy", vec)
        print(f"  {model_name:55s} shape={vec.shape}")

    with open(OUT_DIR / "clustering_info.json", "w") as f:
        json.dump({
            "n_clusters": N_CLUSTERS,
            "embed_model": EMBED_MODEL,
            "cluster_sizes": cluster_sizes.tolist(),
            "kmeans_seed": KMEANS_SEED,
        }, f)
    print(f"\nDone. Rebuilt {len(pool)} ceiling FPs at dim={N_CLUSTERS} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
