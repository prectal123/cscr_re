"""The missing cell of the FP x Loss 2x2 grid: CSCR's own loss
(cost_spectrum_info_nce, byte-identical port of the paper's Eq.8, confirmed
against the official repo) applied to COMPAR's Ceiling FP (Pure V2, full
Set A, no probe cap -- routerbench-ceiling, same dir used by
routerbench_purev2_ablation3way_allseen_multiseed.py).

Completes the grid:
  CSCR FP   x CSCR loss  = CSCR paper's own number (0.711)
  CSCR FP   x TAR loss   = worse than CSCR's own (routerbench_fair_probe1800_multiseed.py arm 2)
  Ceiling FP x TAR loss  = COMPAR headline (0.7205, Pure V2)
  Ceiling FP x CSCR loss = THIS SCRIPT -- if this also beats 0.711, the gap
                           is attributable to the FP, not the loss.

Reuses build_grpo_items/precompute_cls/train_cscr_fast/eval_proj from
routerbench_fair_probe1800_multiseed.py unchanged -- only the FP dir swaps.
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
    build_grpo_items, precompute_cls, train_cscr_fast, eval_proj, CSCR_ROUTERBENCH,
)

SEEDS = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CEILING_PUREV2_DIR = Path("local_descriptors/routerbench-ceiling")


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

    a_costs = set_a.dropna(subset=cost_cols)[cost_cols].to_numpy(dtype=np.float32).mean(axis=0)
    cost_norm = (a_costs - a_costs.min()) / (a_costs.max() - a_costs.min() + 1e-9)
    cost_norm_t = torch.from_numpy(cost_norm).float().to(DEVICE)
    n_bands = int(round(len(models) ** 0.5))
    print(f"n_bands={n_bands} (sqrt({len(models)}))", flush=True)

    E = np.stack([np.load(CEILING_PUREV2_DIR / f"{n}.npy") for n in models])
    E_t = torch.from_numpy(E).float().to(DEVICE)
    E_t = E_t / (E_t.norm(dim=1, keepdim=True) + 1e-9)
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    print(f"\n{'#'*70}\nceiling-purev2-cscrloss (FP dim={E.shape[1]}, dir={CEILING_PUREV2_DIR})\n{'#'*70}", flush=True)

    results = []
    for seed in SEEDS:
        proj = train_cscr_fast(seed, cls_tr, raw_labels, cost_norm_t, E_t, hidden_size, n_bands,
                                f"seed{seed}-ceiling-purev2-cscrloss")
        knn_metrics, delta, p = eval_proj(proj, cls_setB, E_norm, models, b_label_maps, b_costs, seed)
        results.append({"seed": seed, "knn": knn_metrics, "delta": delta, "p": p})
        print(f"  [seed={seed} ceiling-purev2-cscrloss] AUDC={knn_metrics['audc']:.4f} Peak={knn_metrics['peak']:.4f} "
              f"QNC={knn_metrics['qnc']:.4f} delta={delta:+.4f} p={p:.4g} "
              f"({'BEATS' if knn_metrics['audc'] > CSCR_ROUTERBENCH else 'below'} CSCR {CSCR_ROUTERBENCH})", flush=True)

    out_path = Path("local_descriptors/routerbench-analysis/ceilingfp_cscrloss_purev2_results.json")
    json.dump(results, open(out_path, "w"), indent=2)

    audcs = [r["knn"]["audc"] for r in results]
    beats = sum(1 for a in audcs if a > CSCR_ROUTERBENCH)
    print("\n" + "=" * 90)
    print(f"MISSING CELL: RouterBench, Ceiling FP (Pure V2) x CSCR's own loss, vs CSCR paper {CSCR_ROUTERBENCH}")
    print("=" * 90)
    print(f"  AUDC: " + " / ".join(f"{a:.4f}" for a in audcs) +
          f"  mean={np.mean(audcs):.4f} (std={np.std(audcs):.4f})  beats CSCR paper: {beats}/{len(audcs)}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
