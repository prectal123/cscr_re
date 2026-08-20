"""LLMRouterBench companion to routerbench_naive_dim1800_cscrloss.py: naive
probe=dim (900-dim, no binning) Perplexity FP + a same-dim random-vector
negative control, both trained with CSCR's actual cost_spectrum_info_nce
loss. Uses build_perplexity_naive_dim900.py's output
(local_descriptors/llmrouterbench_v15_900/perplexity-dim900/ and
perplexity-dim900-randomcontrol/).
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
from fair_probe900_multiseed import build_rows_with_dataset, build_cost_dict, build_setB_eval
from routerbench_fair_probe1800_multiseed import precompute_cls, train_cscr_fast, eval_proj

SEEDS = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

FP_ROOT = Path("local_descriptors/llmrouterbench_v15_900")
FP_SOURCES = {
    "perplexity-dim900-cscrloss": FP_ROOT / "perplexity-dim900",
    "perplexity-dim900-randomcontrol": FP_ROOT / "perplexity-dim900-randomcontrol",
}


def build_raw_labels(rows, models):
    texts, raw_labels = [], []
    for text, scores, ds in rows:
        labels = np.array([scores[m] for m in models], dtype=np.float32)
        if labels.std() < 1e-6:
            continue
        texts.append(text)
        raw_labels.append(labels)
    return texts, np.stack(raw_labels)


def main():
    import pickle
    with open("local_descriptors/llmrouterbench/setA_setB_split.pkl", "rb") as f:
        split = pickle.load(f)

    models = common.MODELS_33
    rows = build_rows_with_dataset(split)
    print(f"{len(rows)} raw Set A rows", flush=True)

    texts, raw_labels = build_raw_labels(rows, models)
    print(f"{len(texts)} usable Set A rows", flush=True)

    cost_dict = build_cost_dict(split)
    a_costs = np.array([cost_dict[m] for m in models], dtype=np.float32)
    cost_norm = (a_costs - a_costs.min()) / (a_costs.max() - a_costs.min() + 1e-9)
    cost_norm_t = torch.from_numpy(cost_norm).float().to(DEVICE)
    n_bands = int(round(len(models) ** 0.5))
    print(f"n_bands={n_bands} (sqrt({len(models)}))", flush=True)

    b_texts, b_scores = build_setB_eval(split)
    b_label_maps = b_scores.astype(np.float32).tolist()
    b_costs = a_costs
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

    results = {tag: [] for tag in FP_SOURCES}

    for tag, fp_dir in FP_SOURCES.items():
        E = np.stack([np.load(fp_dir / f"{common.NAME_TO_SAFE[m]}.npy") for m in models])
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
                  f"QNC={knn_metrics['qnc']:.4f} delta={delta:+.4f} p={p:.4g}", flush=True)

    out_path = FP_ROOT / "naive_dim900_cscrloss_results.json"
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n" + "=" * 90)
    print(f"NAIVE PROBE=DIM 900 COMPARISON on LLMRouterBench (seeds {SEEDS}, all-seen)")
    print("=" * 90)
    for tag in results:
        audcs = [r["knn"]["audc"] for r in results[tag]]
        print(f"{tag:>32}: " + " / ".join(f"{a:.4f}" for a in audcs) +
              f"  mean={np.mean(audcs):.4f} (std={np.std(audcs):.4f})", flush=True)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
