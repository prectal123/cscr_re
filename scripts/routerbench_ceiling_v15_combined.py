"""RouterBench Ceiling V1.5 (PCA-loading-weighted, ~1800-probe, 86-dim raw --
build_routerbench_ceiling_v15.py) through the same Combined (min-pos +
top50pct-catfilter) training/eval pipeline as routerbench_perplexity_combined.py,
same 3 seeds, all-seen only (RouterBench's 11-model pool is too small for a
meaningful unseen split -- established constraint throughout this project).

Companion to routerbench_perplexity_probesweep_combined.py: together the two
scripts answer the mentor's fairness question from both directions --
(a) give CSCR's Perplexity FP the same probe budget as Combined GRPO
(probesweep script), and (b) show Ceiling V2's full-Set-A budget (86-dim,
~2064 probes at flat 24/cat) isn't required either -- V1.5 gets most of the
way to V2's AUDC on a comparable ~1800-probe budget.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from run_audc_eval import paired_bootstrap_audc_cached
import routerbench_knn_test as rb
from routerbench_perplexity_combined import (
    build_items, train, knn_curve, random_curve, audc_qnc_peak,
)

SEEDS = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAM_LIST = np.logspace(-4, 2, 20)
CSCR_ROUTERBENCH = 0.711
CEILING_V2_MEAN = 0.7226  # existing full-budget Ceiling result, for reference in the printed table

V15_DIR = Path("local_descriptors/routerbench-ceiling-v15")
EXISTING_RESULTS_PATH = Path("local_descriptors/routerbench-analysis/perplexity_vs_ceiling_combined_results.json")


def main():
    set_a, set_b = rb.load_data()
    models, cols = rb.NAMES, rb.MODELS
    cost_cols = [f"{c}|total_cost" for c in cols]

    print("Building GRPO targets + top50pct category-filter mask...", flush=True)
    items = build_items(set_a, models, cols)
    print(f"{len(items)} usable Set A rows", flush=True)

    print(f"Loading frozen MiniLM ONCE (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    base_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    set_b = set_b.dropna(subset=cols + cost_cols)
    b_texts = set_b["prompt"].tolist()
    b_label_maps = set_b[cols].to_numpy(dtype=np.float32).tolist()
    b_costs = set_b[cost_cols].to_numpy(dtype=np.float32).mean(axis=0)
    print(f"Set B: {len(b_texts)} rows for final AUDC", flush=True)

    E = np.stack([np.load(V15_DIR / f"{n}.npy") for n in models])
    E_t = torch.from_numpy(E).float().to(DEVICE)
    E_t = E_t / (E_t.norm(dim=1, keepdim=True) + 1e-9)
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    print(f"\ncombined-ceiling-v15 (FP dim={E.shape[1]}, dir={V15_DIR})", flush=True)

    results = []
    for seed in SEEDS:
        enc = train(seed, items, tokenizer, base_model, E_t, f"seed{seed}-ceiling-v15")

        embeds = np.zeros((len(b_texts), E.shape[1]), dtype=np.float32)
        for start in range(0, len(b_texts), 32):
            batch = b_texts[start:start + 32]
            embeds[start:start + len(batch)] = enc.encode(batch)
        sims = embeds @ E_norm.T

        knn_costs, knn_accs, knn_Y = knn_curve(sims, models, b_label_maps, b_costs, LAM_LIST)
        rand_costs, rand_accs, rand_Y = random_curve(models, b_label_maps, b_costs, LAM_LIST, seed=seed)
        knn_metrics = audc_qnc_peak(knn_costs, knn_accs)
        rand_metrics = audc_qnc_peak(rand_costs, rand_accs)

        ko = np.argsort(knn_costs)
        ro = np.argsort(rand_costs)
        mean_delta, (lo, hi), p = paired_bootstrap_audc_cached(
            knn_costs[ko], knn_Y[ko], rand_costs[ro], rand_Y[ro], B=1000, seed=0)

        r = {"seed": seed, "knn": knn_metrics, "random": rand_metrics,
             "delta": float(mean_delta), "p": float(p)}
        results.append(r)
        print(f"  [seed={seed} ceiling-v15] AUDC={knn_metrics['audc']:.4f} Peak={knn_metrics['peak']:.4f} "
              f"QNC={knn_metrics['qnc']:.4f}  delta={mean_delta:+.4f} p={p:.4g}  "
              f"({'BEATS' if knn_metrics['audc'] > CSCR_ROUTERBENCH else 'below'} CSCR {CSCR_ROUTERBENCH})",
              flush=True)

    out_path = Path("local_descriptors/routerbench-analysis/ceiling_v15_combined_results.json")
    json.dump({"combined-ceiling-v15": results}, open(out_path, "w"), indent=2)

    ceiling_v2_audcs = None
    if EXISTING_RESULTS_PATH.exists():
        existing = json.load(open(EXISTING_RESULTS_PATH))
        if "combined-ceiling" in existing:
            ceiling_v2_audcs = [r["knn"]["audc"] for r in existing["combined-ceiling"]]

    audcs = [r["knn"]["audc"] for r in results]
    beats = sum(1 for a in audcs if a > CSCR_ROUTERBENCH)
    print("\n" + "=" * 90)
    print(f"CEILING V1.5 (~1800 probes, 86-dim) on RouterBench (seeds {SEEDS}) vs CSCR {CSCR_ROUTERBENCH}")
    print("=" * 90)
    if ceiling_v2_audcs:
        beats_v2 = sum(1 for a in ceiling_v2_audcs if a > CSCR_ROUTERBENCH)
        print(f"{'combined-ceiling-v2 (existing, full budget)':>46}: " +
              " / ".join(f"{a:.4f}" for a in ceiling_v2_audcs) +
              f"  mean={np.mean(ceiling_v2_audcs):.4f}  beats CSCR: {beats_v2}/{len(ceiling_v2_audcs)}")
    print(f"{'combined-ceiling-v15 (~1800 probes)':>46}: " + " / ".join(f"{a:.4f}" for a in audcs) +
          f"  mean={np.mean(audcs):.4f} (std={np.std(audcs):.4f})  beats CSCR: {beats}/{len(audcs)}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
