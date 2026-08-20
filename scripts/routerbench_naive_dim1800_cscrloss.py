"""The "naive fairness" variant the user flagged as possibly closer to what
the professor actually pictures: instead of holding Perplexity FP's
dimensionality fixed at 32 and only growing the probe count averaged into
each dim (routerbench_fair_probe1800_multiseed.py's controlled version),
just give it a literal 1800-dim descriptor -- one probe, one dimension,
exactly matching Ceiling FP's own probe-indexed convention. Uses the
already-built bonus variants from build_routerbench_perplexity_fp_probesweep.py
(routerbench-perplexity-dim1800/ + a same-dim random-vector negative control),
trained with CSCR's actual cost_spectrum_info_nce loss (not Combined).

The random-vector control matters here specifically: at 1800 dims for only
11 models, a win could be pure "more free parameters to fit 11 points"
rather than real signal (the same capacity-vs-reliability confound flagged
in PROGRESS.md 23.3) -- if the random control does about as well as the real
Perplexity-dim1800 FP, that confirms it's not real signal.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from routerbench_perplexity_combined import compute_keep_idx_top50pct
import routerbench_knn_test as rb
from routerbench_fair_probe1800_multiseed import (
    precompute_cls, train_cscr_fast, eval_proj,
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEEDS = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSCR_ROUTERBENCH = 0.711

FP_SOURCES = {
    "perplexity-dim1800-cscrloss": Path("local_descriptors/routerbench-perplexity-dim1800"),
    "perplexity-dim1800-randomcontrol": Path("local_descriptors/routerbench-perplexity-dim1800-randomcontrol"),
}


def build_raw_labels(set_a, models, cols):
    texts, raw_labels = [], []
    for _, row in set_a.iterrows():
        labels = np.array([float(row[c]) for c in cols], dtype=np.float32)
        if labels.std() < 1e-6:
            continue
        texts.append(row["prompt"])
        raw_labels.append(labels)
    return texts, np.stack(raw_labels)


def main():
    set_a, set_b = rb.load_data()
    models, cols = rb.NAMES, rb.MODELS
    cost_cols = [f"{c}|total_cost" for c in cols]

    print("Building raw-label training rows...", flush=True)
    texts, raw_labels = build_raw_labels(set_a, models, cols)
    print(f"{len(texts)} usable Set A rows", flush=True)

    a_costs = set_a.dropna(subset=cost_cols)[cost_cols].to_numpy(dtype=np.float32).mean(axis=0)
    cost_norm = (a_costs - a_costs.min()) / (a_costs.max() - a_costs.min() + 1e-9)
    cost_norm_t = torch.from_numpy(cost_norm).float().to(DEVICE)
    n_bands = int(round(len(models) ** 0.5))
    print(f"n_bands={n_bands} (sqrt({len(models)}))", flush=True)

    print(f"Loading frozen MiniLM ONCE (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    hidden_size = base_model.config.hidden_size

    print("Precomputing frozen CLS embeddings for all training rows...", flush=True)
    t0 = time.time()
    cls_tr = precompute_cls(texts, tokenizer, base_model)
    print(f"  done in {time.time()-t0:.1f}s -> {cls_tr.shape}", flush=True)

    set_b = set_b.dropna(subset=cols + cost_cols)
    b_texts = set_b["prompt"].tolist()
    b_label_maps = set_b[cols].to_numpy(dtype=np.float32).tolist()
    b_costs = set_b[cost_cols].to_numpy(dtype=np.float32).mean(axis=0)
    print(f"Precomputing frozen CLS embeddings for {len(b_texts)} Set B rows...", flush=True)
    t0 = time.time()
    cls_setB = precompute_cls(b_texts, tokenizer, base_model)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    results = {tag: [] for tag in FP_SOURCES}

    for tag, fp_dir in FP_SOURCES.items():
        E = np.stack([np.load(fp_dir / f"{n}.npy") for n in models])
        E_t = torch.from_numpy(E).float().to(DEVICE)
        E_t = E_t / (E_t.norm(dim=1, keepdim=True) + 1e-9)
        E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        print(f"\n{'#'*70}\n{tag} (FP dim={E.shape[1]}, dir={fp_dir})\n{'#'*70}", flush=True)

        for seed in SEEDS:
            proj = train_cscr_fast(seed, cls_tr, raw_labels, cost_norm_t, E_t, hidden_size, n_bands,
                                    f"seed{seed}-{tag}")
            knn_metrics, delta, p = eval_proj(proj, cls_setB, E_norm, models, b_label_maps, b_costs, seed)
            results[tag].append({"seed": seed, "knn": knn_metrics, "delta": delta, "p": p})
            print(f"  [seed={seed} {tag}] AUDC={knn_metrics['audc']:.4f} Peak={knn_metrics['peak']:.4f} "
                  f"QNC={knn_metrics['qnc']:.4f} delta={delta:+.4f} p={p:.4g} "
                  f"({'BEATS' if knn_metrics['audc'] > CSCR_ROUTERBENCH else 'below'} CSCR {CSCR_ROUTERBENCH})",
                  flush=True)

    out_path = Path("local_descriptors/routerbench-analysis/naive_dim1800_cscrloss_results.json")
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n" + "=" * 90)
    print(f"NAIVE PROBE=DIM 1800 COMPARISON on RouterBench (seeds {SEEDS}, all-seen) vs CSCR paper {CSCR_ROUTERBENCH}")
    print("=" * 90)
    for tag in results:
        audcs = [r["knn"]["audc"] for r in results[tag]]
        beats = sum(1 for a in audcs if a > CSCR_ROUTERBENCH)
        print(f"{tag:>32}: " + " / ".join(f"{a:.4f}" for a in audcs) +
              f"  mean={np.mean(audcs):.4f} (std={np.std(audcs):.4f})  beats CSCR paper: {beats}/{len(audcs)}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
