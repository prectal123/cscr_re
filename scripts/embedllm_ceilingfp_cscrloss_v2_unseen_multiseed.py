"""EmbedLLM analogue of routerbench_ceilingfp_cscrloss_purev2_allseen_multiseed.py:
CSCR's own loss (cost_spectrum_info_nce) applied to the CURRENT (uncompressed,
V2/full-data) Ceiling FP, Unseen protocol.

Sanity-checks a user recollection: an earlier run in this project
(embedllm_newllm_train_encoder_csinfonce.py, historical result mean
AUDC=0.468 over CSCR_UNSEEN=0.4848, i.e. did NOT beat CSCR) used the SAME
loss but a PCA-5 COMPRESSED Ceiling FP, 2 fixed epochs, no holdout early
stopping. Session §25 later found PCA-5 compression alone costs ~0.03 AUDC
under a trainable projection head. This script reruns the same loss on the
current uncompressed FP (embedllm-ceiling, V2/full data) with the current
holdout-based training convention, to see whether removing the compression
tax alone is enough to flip the historical result.

Reuses precompute_cls/train_cscr_fast/eval_proj from
routerbench_fair_probe1800_multiseed.py unchanged (both generic over
model-list + label_map-dict + cost-array shapes, already confirmed
EmbedLLM-compatible via reuse elsewhere this session).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from router.embedllm import load_embedllm
from router.cost_models import compute_cost
from routerbench_fair_probe1800_multiseed import precompute_cls, train_cscr_fast, eval_proj

SEEDS = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CSCR_UNSEEN = 0.4848

ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
CEILING_V2_DIR = Path("local_descriptors/embedllm-ceiling")  # uncompressed, full-data


def build_raw_label_items(df, models):
    """Raw 0/1 label matrix per prompt (no GRPO z-score, no catfilter) --
    matches cost_spectrum_info_nce's own expected input convention (paper's
    Eq.8 loss operates on binary pos/neg labels directly, not a regression
    target). Keeps a prompt only if it has at least one positive AND one
    negative among `models` (degenerate all-same rows carry no loss signal)."""
    name_to_idx = {n: i for i, n in enumerate(models)}
    texts, raw_labels = [], []
    for pid, grp in df.groupby("prompt_id", sort=False):
        labels = np.full(len(models), np.nan, dtype=np.float32)
        for m, v in zip(grp["model_name"], grp["label"]):
            if m in name_to_idx:
                labels[name_to_idx[m]] = float(v)
        mask = ~np.isnan(labels)
        if mask.sum() < 2:
            continue
        vals = labels[mask]
        if vals.std() < 1e-6:
            continue
        text = grp["prompt"].iloc[0]
        texts.append(text)
        raw_labels.append(np.nan_to_num(labels, nan=0.0))
    return texts, np.stack(raw_labels)


def main():
    print("Loading EmbedLLM train.csv...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "label", "prompt"])

    print(f"Loading frozen MiniLM ONCE (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    hidden_size = base_model.config.hidden_size

    results = []
    for seed in SEEDS:
        split_path = ANALYSIS_DIR / ("newllm_split.json" if seed == 0 else f"newllm_split_seed{seed}.json")
        split = json.load(open(split_path, encoding="utf-8"))
        seen_models, unseen_models = split["seen"], split["unseen"]
        print(f"\n{'#'*70}\nseed={seed}: seen={len(seen_models)} unseen={len(unseen_models)}\n{'#'*70}", flush=True)

        texts, raw_labels = build_raw_label_items(df, seen_models)
        print(f"  {len(texts)} usable Set A rows", flush=True)
        t0 = time.time()
        cls_tr = precompute_cls(texts, tokenizer, base_model)
        print(f"  Set A embeddings cached in {time.time()-t0:.1f}s -> {cls_tr.shape}", flush=True)

        cost_raw = np.array([compute_cost(m, 0, cost_type="n_params") for m in seen_models], dtype=np.float32)
        cost_norm = (cost_raw - cost_raw.min()) / (cost_raw.max() - cost_raw.min() + 1e-9)
        cost_norm_t = torch.from_numpy(cost_norm).float().to(DEVICE)
        n_bands = int(round(len(seen_models) ** 0.5))
        print(f"  n_bands={n_bands} (sqrt({len(seen_models)}))", flush=True)

        dataset = load_embedllm("test", candidates=unseen_models)
        eval_texts = [ex["prompt"] for ex in dataset]
        # eval_proj's knn_curve (routerbench_perplexity_combined) indexes
        # label_maps[i][best_j] positionally (RouterBench's own convention:
        # list-of-lists aligned with `models`), not by dict lookup -- convert
        # EmbedLLM's {model_name: label} dicts to that positional form.
        b_label_maps = [[lm.get(m, 0) for m in unseen_models] for lm in (ex["label_map"] for ex in dataset)]
        t0 = time.time()
        cls_setB = precompute_cls(eval_texts, tokenizer, base_model)
        print(f"  Set B unseen-only ({len(eval_texts)} rows) cached in {time.time()-t0:.1f}s", flush=True)
        b_costs = np.array([compute_cost(m, 0, cost_type="n_params") for m in unseen_models], dtype=np.float32)

        E_seen = np.stack([np.load(CEILING_V2_DIR / f"{m}.npy") for m in seen_models]).astype(np.float32)
        E_seen_t = torch.from_numpy(E_seen).float().to(DEVICE)
        E_seen_t = E_seen_t / (E_seen_t.norm(dim=1, keepdim=True) + 1e-9)

        proj = train_cscr_fast(seed, cls_tr, raw_labels, cost_norm_t, E_seen_t, hidden_size, n_bands,
                                f"seed{seed}-ceilingv2-cscrloss-unseen")

        E_unseen = np.stack([np.load(CEILING_V2_DIR / f"{m}.npy") for m in unseen_models]).astype(np.float32)
        E_unseen_norm = E_unseen / (np.linalg.norm(E_unseen, axis=1, keepdims=True) + 1e-12)

        knn_metrics, delta, p = eval_proj(proj, cls_setB, E_unseen_norm, unseen_models, b_label_maps, b_costs, seed)
        results.append({"seed": seed, "n_seen": len(seen_models), "n_unseen": len(unseen_models),
                         "knn": knn_metrics, "delta": delta, "p": p})
        print(f"  [seed={seed}] AUDC={knn_metrics['audc']:.4f} Peak={knn_metrics['peak']:.4f} "
              f"QNC={knn_metrics['qnc']:.4f} delta={delta:+.4f} p={p:.4g} "
              f"({'BEATS' if knn_metrics['audc'] > CSCR_UNSEEN else 'below'} CSCR-unseen {CSCR_UNSEEN})", flush=True)

    audcs = [r["knn"]["audc"] for r in results]
    beats = sum(1 for a in audcs if a > CSCR_UNSEEN)
    print("\n" + "=" * 90)
    print(f"EmbedLLM Unseen: Ceiling FP (V2, uncompressed) x CSCR's own loss, vs CSCR paper {CSCR_UNSEEN}")
    print("=" * 90)
    print(f"  AUDC: " + " / ".join(f"{a:.4f}" for a in audcs) +
          f"  mean={np.mean(audcs):.4f} (std={np.std(audcs):.4f})  beats CSCR paper: {beats}/{len(audcs)}")


if __name__ == "__main__":
    main()
