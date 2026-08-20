"""Mentor feedback (PROGRESS.md 23.3): is the Ceiling-vs-Perplexity comparison
on RouterBench (22.8/routerbench_perplexity_combined.py) unfair because
Perplexity FP only used N_PROBES=32 (a stale never-updated debug constant)
while Combined GRPO's Ceiling FP uses up to 1800? This reruns the exact same
Combined (min-pos + top50pct-catfilter) methodology with Perplexity FP built
at N_PROBES in {32, 192, 1800} -- SAME 32-dim output for all three (see
build_routerbench_perplexity_fp_probesweep.py for why dimensionality is held
fixed while only the number of probes averaged into each dimension grows;
naively scaling both together would confound "more reliable estimate" with
"more free dimensions fit to only 11 models").

Reuses combined-ceiling's numbers from the existing
perplexity_vs_ceiling_combined_results.json (already run, no need to redo)
and only trains the 3 new Perplexity-FP variants.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from router.query_encoder import QueryEncoder
from router.bandit import BanditStats
from run_audc_eval import interp_to_grid, build_cost_grid, paired_bootstrap_audc_cached
import routerbench_knn_test as rb
from routerbench_perplexity_combined import (
    build_items, minpos_loss, evaluate_holdout, train, knn_curve, random_curve, audc_qnc_peak,
)

EPOCHS = 10
SEEDS = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAM_LIST = np.logspace(-4, 2, 20)
CSCR_ROUTERBENCH = 0.711

FP_SOURCES = {
    "combined-perplexity-32probes": Path("local_descriptors/routerbench-perplexity-nprobes32"),
    "combined-perplexity-192probes": Path("local_descriptors/routerbench-perplexity-nprobes192"),
    "combined-perplexity-1800probes": Path("local_descriptors/routerbench-perplexity-nprobes1800"),
}

EXISTING_RESULTS_PATH = Path("local_descriptors/routerbench-analysis/perplexity_vs_ceiling_combined_results.json")


def main():
    set_a, set_b = rb.load_data()
    models, cols = rb.NAMES, rb.MODELS
    cost_cols = [f"{c}|total_cost" for c in cols]

    print("Building GRPO targets + top50pct category-filter mask (shared across all FP sources)...", flush=True)
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

    results = {tag: [] for tag in FP_SOURCES}

    for tag, fp_dir in FP_SOURCES.items():
        E = np.stack([np.load(fp_dir / f"{n}.npy") for n in models])
        E_t = torch.from_numpy(E).float().to(DEVICE)
        E_t = E_t / (E_t.norm(dim=1, keepdim=True) + 1e-9)
        E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        print(f"\n{'#'*70}\n{tag} (FP dim={E.shape[1]}, dir={fp_dir})\n{'#'*70}", flush=True)

        for seed in SEEDS:
            enc = train(seed, items, tokenizer, base_model, E_t, f"seed{seed}-{tag}")

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
            results[tag].append(r)
            print(f"  [seed={seed} {tag}] AUDC={knn_metrics['audc']:.4f} Peak={knn_metrics['peak']:.4f} "
                  f"QNC={knn_metrics['qnc']:.4f}  delta={mean_delta:+.4f} p={p:.4g}  "
                  f"({'BEATS' if knn_metrics['audc'] > CSCR_ROUTERBENCH else 'below'} CSCR {CSCR_ROUTERBENCH})",
                  flush=True)

    out_path = Path("local_descriptors/routerbench-analysis/perplexity_probesweep_combined_results.json")
    json.dump(results, open(out_path, "w"), indent=2)

    # pull in the existing ceiling numbers for one combined final table
    ceiling_audcs = None
    if EXISTING_RESULTS_PATH.exists():
        existing = json.load(open(EXISTING_RESULTS_PATH))
        if "combined-ceiling" in existing:
            ceiling_audcs = [r["knn"]["audc"] for r in existing["combined-ceiling"]]

    print("\n" + "=" * 90)
    print(f"PERPLEXITY FP PROBE SWEEP (Combined method) on RouterBench (seeds {SEEDS}) vs CSCR {CSCR_ROUTERBENCH}")
    print("=" * 90)
    if ceiling_audcs:
        beats = sum(1 for a in ceiling_audcs if a > CSCR_ROUTERBENCH)
        print(f"{'combined-ceiling (existing)':>32}: " + " / ".join(f"{a:.4f}" for a in ceiling_audcs) +
              f"  mean={np.mean(ceiling_audcs):.4f} (std={np.std(ceiling_audcs):.4f})  beats CSCR: {beats}/{len(ceiling_audcs)}")
    for tag in results:
        audcs = [r["knn"]["audc"] for r in results[tag]]
        beats = sum(1 for a in audcs if a > CSCR_ROUTERBENCH)
        print(f"{tag:>32}: " + " / ".join(f"{a:.4f}" for a in audcs) +
              f"  mean={np.mean(audcs):.4f} (std={np.std(audcs):.4f})  beats CSCR: {beats}/{len(audcs)}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
