"""Rebuild the ceiling FP with per-cluster POOL-MEAN CENTERING before L2-normalize.

Diagnosis (2026-07-30 session): the previous clustered ceiling FP
(build_ceiling_fp_clustered.py) had pairwise cosine similarity ~0.98 across
all 11 models -- essentially collinear, meaning no cosine-similarity-based
routing could ever separate them. Root cause: each cluster's raw value is
exp(bartscore), and bartscore magnitude is dominated by a shared "how easy
is this prompt cluster to answer" component that's common across ALL models,
drowning out the actual model-to-model capability differences we want the
FP to capture.

Fix: for each cluster, subtract the pool-average score at that cluster
(across all 11 models) from each model's own score there, BEFORE L2-normalizing.
This removes the shared "difficulty" direction and leaves only each model's
relative deviation from the pool average -- the actual differential signal.

Reuses the same Set-A prompts / embeddings / k-means clustering as
build_ceiling_fp_clustered.py (same KMEANS_SEED=0), just changes step 4-5.
Runs on GPU if available (this session added a CUDA torch build).
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    print(f"DEVICE: {DEVICE}")
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

    print(f"Loading {EMBED_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    model.eval()

    embeddings = np.zeros((len(ids), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for start in tqdm(range(0, len(ids), BATCH_SIZE), desc="embedding Set A"):
            batch_ids = ids[start:start + BATCH_SIZE]
            batch_text = [prompt_text[i] for i in batch_ids]
            enc = tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out = model(**enc).last_hidden_state[:, 0]
            embeddings[start:start + len(batch_ids)] = out.cpu().numpy()

    print(f"\nEmbedded {embeddings.shape[0]} prompts, dim={embeddings.shape[1]}")

    print(f"Running k-means (k={N_CLUSTERS}, seed={KMEANS_SEED}, same as before for a fair A/B)...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=KMEANS_SEED, n_init=10)
    cluster_ids = km.fit_predict(embeddings)
    cluster_sizes = np.bincount(cluster_ids, minlength=N_CLUSTERS)
    print(f"Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
          f"mean={cluster_sizes.mean():.1f}, empty clusters={int((cluster_sizes==0).sum())}")

    # ---- raw per-cluster mean bartscore, per model (BEFORE centering) ----
    raw_vals = {}  # model_name -> (N_CLUSTERS,) array
    for model_name in pool:
        cluster_vals = np.zeros(N_CLUSTERS, dtype=np.float64)
        for c in range(N_CLUSTERS):
            members = [ids[i] for i in range(len(ids)) if cluster_ids[i] == c]
            if not members:
                cluster_vals[c] = 0.0
                continue
            vals = [math.exp(per_prompt_scores[pid][model_name]) for pid in members]
            cluster_vals[c] = float(np.mean(vals))
        raw_vals[model_name] = cluster_vals

    # ---- pool-mean per cluster (the shared "difficulty" component) ----
    pool_matrix = np.stack([raw_vals[m] for m in pool])  # (11, 192)
    pool_mean_per_cluster = pool_matrix.mean(axis=0)     # (192,) shared component

    # ---- center then L2-normalize ----
    for model_name in pool:
        centered = raw_vals[model_name] - pool_mean_per_cluster
        norm = np.linalg.norm(centered) + 1e-12
        vec = (centered / norm).astype(np.float32)
        np.save(OUT_DIR / f"{model_name}.npy", vec)
        print(f"  {model_name:55s} shape={vec.shape}")

    with open(OUT_DIR / "clustering_info.json", "w") as f:
        json.dump({
            "n_clusters": N_CLUSTERS,
            "embed_model": EMBED_MODEL,
            "cluster_sizes": cluster_sizes.tolist(),
            "kmeans_seed": KMEANS_SEED,
            "centering": "pool-mean-per-cluster subtracted before L2-normalize (fixes ~0.98 collinearity)",
        }, f)
    print(f"\nDone. Rebuilt {len(pool)} MEAN-CENTERED ceiling FPs at dim={N_CLUSTERS} -> {OUT_DIR}")

    # quick collinearity check
    E = np.stack([np.load(OUT_DIR / f"{m}.npy") for m in pool])
    sim = E @ E.T
    off = sim[~np.eye(len(pool), dtype=bool)]
    print(f"\nPairwise cosine sim after centering: mean={off.mean():.4f} std={off.std():.4f} "
          f"min={off.min():.4f} max={off.max():.4f}")


if __name__ == "__main__":
    main()
