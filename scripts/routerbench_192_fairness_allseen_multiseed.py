"""Fairness point for RouterBench: COMPAR's own Combined GRPO loss on the
uniform-allocation, uncompressed Ceiling FP, shrunk to CSCR's own original
probe budget (192, matching the paper's "we sample 192 probes from its
training set" for RouterBench -- verified directly from 2508.12491v3).
All-seen only (11-model pool too small for a seen/unseen split, per
routerbench_fair_probe1800_multiseed.py's existing convention).

Reuses the training/eval plumbing from routerbench_fair_probe1800_multiseed
unchanged (generic over FP dir) -- only CEILING_DIR is repointed to the new
192-probe FP built by build_routerbench_ceiling_192.py. This is the
RouterBench-side counterpart to embedllm_uniform_nocompress_192_fairness_*.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
import routerbench_knn_test as rb
from routerbench_fair_probe1800_multiseed import (
    build_grpo_items, precompute_cls, train_combined_fast, eval_proj, CSCR_ROUTERBENCH,
)

SEEDS = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CEILING_DIR = Path("local_descriptors/routerbench-ceiling-uniform-nocompress-192")


def main():
    set_a, set_b = rb.load_data()
    models, cols = rb.NAMES, rb.MODELS
    cost_cols = [f"{c}|total_cost" for c in cols]

    print("Building training rows (GRPO targets + raw labels, shared text set)...", flush=True)
    texts, targets, masks, raw_labels = build_grpo_items(set_a, models, cols)
    print(f"{len(texts)} usable Set A rows", flush=True)

    print(f"Loading frozen MiniLM ONCE (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    hidden_size = base_model.config.hidden_size

    t0 = time.time()
    cls_tr = precompute_cls(texts, tokenizer, base_model)
    print(f"  Set A embeddings cached in {time.time()-t0:.1f}s -> {cls_tr.shape}", flush=True)

    set_b = set_b.dropna(subset=cols + cost_cols)
    b_texts = set_b["prompt"].tolist()
    b_label_maps = set_b[cols].to_numpy(dtype=np.float32).tolist()
    b_costs = set_b[cost_cols].to_numpy(dtype=np.float32).mean(axis=0)
    t0 = time.time()
    cls_setB = precompute_cls(b_texts, tokenizer, base_model)
    print(f"  Set B embeddings cached in {time.time()-t0:.1f}s", flush=True)

    E = np.stack([np.load(CEILING_DIR / f"{n}.npy") for n in models])
    E_t = torch.from_numpy(E).float().to(DEVICE)
    E_t = E_t / (E_t.norm(dim=1, keepdim=True) + 1e-9)
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    print(f"\n{'#'*70}\nceiling-192-uniform-nocompress-combined (FP dim={E.shape[1]}, dir={CEILING_DIR})\n{'#'*70}", flush=True)

    results = []
    for seed in SEEDS:
        proj = train_combined_fast(seed, cls_tr, targets, masks, E_t, hidden_size, f"seed{seed}-ceiling-192")
        knn_metrics, delta, p = eval_proj(proj, cls_setB, E_norm, models, b_label_maps, b_costs, seed)
        results.append({"seed": seed, "knn": knn_metrics, "delta": delta, "p": p})
        print(f"  [seed={seed} ceiling-192] AUDC={knn_metrics['audc']:.4f} Peak={knn_metrics['peak']:.4f} "
              f"QNC={knn_metrics['qnc']:.4f} delta={delta:+.4f} p={p:.4g} "
              f"({'BEATS' if knn_metrics['audc'] > CSCR_ROUTERBENCH else 'below'} CSCR {CSCR_ROUTERBENCH})", flush=True)

    out_path = Path("local_descriptors/routerbench-analysis/fairness_192_uniform_nocompress_results.json")
    json.dump(results, open(out_path, "w"), indent=2)

    audcs = [r["knn"]["audc"] for r in results]
    beats = sum(1 for a in audcs if a > CSCR_ROUTERBENCH)
    print("\n" + "=" * 90)
    print(f"FAIRNESS: RouterBench, 192-probe uniform+uncompressed, all-seen, 3 seeds vs CSCR paper {CSCR_ROUTERBENCH}")
    print("=" * 90)
    print(f"  AUDC: " + " / ".join(f"{a:.4f}" for a in audcs) +
          f"  mean={np.mean(audcs):.4f} (std={np.std(audcs):.4f})  beats CSCR paper: {beats}/{len(audcs)}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
