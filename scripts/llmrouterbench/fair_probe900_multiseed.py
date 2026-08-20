"""LLMRouterBench version of routerbench_fair_probe1800_multiseed.py: CSCR's
own methodology (Perplexity-900 FP + cost_spectrum_info_nce, the upstream
Eq.8 loss) vs our methodology (Ceiling-900 FP + Combined GRPO / min-pos +
top50pct-catfilter), same 900-probe budget, all-seen, multi-seed (0,1,2).

"Category" for the catfilter is dataset (8 categories, matching this
benchmark's structure -- vs RouterBench's 86 eval_names). Reuses the fast
cached-embedding training/eval helpers from routerbench_fair_probe1800_multiseed.py
unchanged (they're generic over the label/target arrays, not RouterBench-specific).

No external "CSCR paper" reference number exists for this benchmark (it's
not in the CSCR paper -- an independently-sourced pool built for this
project), so this is a direct head-to-head between the two arms, not a
beats-the-paper check.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts/llmrouterbench")
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import common
from routerbench_perplexity_combined import compute_keep_idx_top50pct, knn_curve, random_curve, audc_qnc_peak
from routerbench_fair_probe1800_multiseed import (
    precompute_cls, train_combined_fast, train_cscr_fast, eval_proj,
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEEDS = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAM_LIST = np.logspace(-4, 2, 20)

FP_ROOT = Path("local_descriptors/llmrouterbench_v15_900")
CEILING_DIR = FP_ROOT / "ceiling"
PERPLEXITY_DIR = FP_ROOT / "perplexity"


def build_rows_with_dataset(split):
    """(query, {model: score}, dataset) across ALL Set A, all 8 datasets --
    like loo_recovery.build_train_rows but keeps the dataset tag per row
    (needed for the per-category top50pct catfilter)."""
    rows = []
    for ds in common.DATASETS:
        d = split["setA"][ds]
        n = len(d["queries"])
        for i in range(n):
            scores = {m: float(d["scores"][i, j]) for j, m in enumerate(common.MODELS_33)}
            rows.append((d["queries"][i], scores, ds))
    return rows


def build_cost_dict(split):
    all_costs = {m: [] for m in common.MODELS_33}
    for ds in common.DATASETS:
        d = split["setA"][ds]
        for j, m in enumerate(common.MODELS_33):
            all_costs[m].extend(d["costs"][:, j].tolist())
    return {m: float(np.mean(v)) for m, v in all_costs.items()}


def build_setB_eval(split):
    queries, scores = [], []
    for ds in common.DATASETS:
        d = split["setB"][ds]
        queries.extend(d["queries"])
        scores.append(d["scores"])
    return queries, np.concatenate(scores, axis=0)


def build_items(rows, models):
    """Mirrors routerbench_fair_probe1800_multiseed.build_grpo_items, but
    category = dataset (8 categories) instead of RouterBench's eval_name."""
    n_models = len(models)
    ds_to_idx = {d: i for i, d in enumerate(common.DATASETS)}

    raw_ds_acc = np.zeros((n_models, len(common.DATASETS)))
    counts = np.zeros((n_models, len(common.DATASETS)))
    for _, scores, ds in rows:
        di = ds_to_idx[ds]
        for j, m in enumerate(models):
            raw_ds_acc[j, di] += scores[m]
            counts[j, di] += 1
    raw_ds_acc = raw_ds_acc / np.maximum(counts, 1)

    texts, targets, masks, raw_labels = [], [], [], []
    for text, scores, ds in rows:
        labels = np.array([scores[m] for m in models], dtype=np.float32)
        mean, std = labels.mean(), labels.std()
        if std < 1e-6:
            continue
        target = (labels - mean) / (std + 1e-6)
        mask = np.ones(n_models, dtype=np.float32)
        di = ds_to_idx[ds]
        pos_idx = np.where(labels == 1)[0]
        if len(pos_idx) > 0:
            cat_scores = raw_ds_acc[pos_idx, di]
            keep_pos = compute_keep_idx_top50pct(pos_idx, cat_scores)
            demoted = np.setdiff1d(pos_idx, keep_pos)
            mask[demoted] = 0.0
        texts.append(text)
        targets.append(target)
        masks.append(mask)
        raw_labels.append(labels)
    return texts, np.stack(targets), np.stack(masks), np.stack(raw_labels)


def main():
    import pickle
    with open("local_descriptors/llmrouterbench/setA_setB_split.pkl", "rb") as f:
        split = pickle.load(f)

    models = common.MODELS_33
    rows = build_rows_with_dataset(split)
    print(f"{len(rows)} raw Set A rows across {len(common.DATASETS)} datasets", flush=True)

    print("Building training rows (GRPO targets + raw labels, dataset-as-category)...", flush=True)
    texts, targets, masks, raw_labels = build_items(rows, models)
    print(f"{len(texts)} usable Set A rows", flush=True)

    cost_dict = build_cost_dict(split)
    a_costs = np.array([cost_dict[m] for m in models], dtype=np.float32)
    cost_norm = (a_costs - a_costs.min()) / (a_costs.max() - a_costs.min() + 1e-9)
    cost_norm_t = torch.from_numpy(cost_norm).float().to(DEVICE)
    n_bands = int(round(len(models) ** 0.5))
    print(f"n_bands={n_bands} (sqrt({len(models)}))", flush=True)

    b_texts, b_scores = build_setB_eval(split)
    b_label_maps = b_scores.astype(np.float32).tolist()
    b_costs = a_costs  # same per-model average cost used for training and eval (matches RouterBench convention)
    print(f"Set B: {len(b_texts)} rows for final AUDC", flush=True)

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

    print(f"Precomputing frozen CLS embeddings for {len(b_texts)} Set B rows...", flush=True)
    t0 = time.time()
    cls_setB = precompute_cls(b_texts, tokenizer, base_model)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    results = {"ceiling-900-combined": [], "perplexity-900-cscrloss": []}

    # --- arm 1: Ceiling-900 + Combined GRPO ---
    E = np.stack([np.load(CEILING_DIR / f"{common.NAME_TO_SAFE[m]}.npy") for m in models])
    E_t = torch.from_numpy(E).float().to(DEVICE)
    E_t = E_t / (E_t.norm(dim=1, keepdim=True) + 1e-9)
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    print(f"\n{'#'*70}\nceiling-900-combined (FP dim={E.shape[1]}, dir={CEILING_DIR})\n{'#'*70}", flush=True)
    for seed in SEEDS:
        proj = train_combined_fast(seed, cls_tr, targets, masks, E_t, hidden_size, f"seed{seed}-ceiling-900")
        knn_metrics, delta, p = eval_proj(proj, cls_setB, E_norm, models, b_label_maps, b_costs, seed)
        results["ceiling-900-combined"].append({"seed": seed, "knn": knn_metrics, "delta": delta, "p": p})
        print(f"  [seed={seed} ceiling-900-combined] AUDC={knn_metrics['audc']:.4f} Peak={knn_metrics['peak']:.4f} "
              f"QNC={knn_metrics['qnc']:.4f} delta={delta:+.4f} p={p:.4g}", flush=True)

    # --- arm 2: Perplexity-900 + CSCR's own cost_spectrum_info_nce ---
    E2 = np.stack([np.load(PERPLEXITY_DIR / f"{common.NAME_TO_SAFE[m]}.npy") for m in models])
    E2_t = torch.from_numpy(E2).float().to(DEVICE)
    E2_t = E2_t / (E2_t.norm(dim=1, keepdim=True) + 1e-9)
    E2_norm = E2 / (np.linalg.norm(E2, axis=1, keepdims=True) + 1e-12)
    print(f"\n{'#'*70}\nperplexity-900-cscrloss (FP dim={E2.shape[1]}, dir={PERPLEXITY_DIR})\n{'#'*70}", flush=True)
    for seed in SEEDS:
        proj = train_cscr_fast(seed, cls_tr, raw_labels, cost_norm_t, E2_t, hidden_size, n_bands,
                                f"seed{seed}-perplexity-900-cscrloss")
        knn_metrics, delta, p = eval_proj(proj, cls_setB, E2_norm, models, b_label_maps, b_costs, seed)
        results["perplexity-900-cscrloss"].append({"seed": seed, "knn": knn_metrics, "delta": delta, "p": p})
        print(f"  [seed={seed} perplexity-900-cscrloss] AUDC={knn_metrics['audc']:.4f} Peak={knn_metrics['peak']:.4f} "
              f"QNC={knn_metrics['qnc']:.4f} delta={delta:+.4f} p={p:.4g}", flush=True)

    out_path = FP_ROOT / "fair_probe900_multiseed_results.json"
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n" + "=" * 90)
    print(f"FAIR 900-PROBE COMPARISON on LLMRouterBench (33 models, seeds {SEEDS}, all-seen)")
    print("=" * 90)
    for tag in results:
        audcs = [r["knn"]["audc"] for r in results[tag]]
        print(f"{tag:>28}: " + " / ".join(f"{a:.4f}" for a in audcs) +
              f"  mean={np.mean(audcs):.4f} (std={np.std(audcs):.4f})", flush=True)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
