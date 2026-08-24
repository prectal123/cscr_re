"""LLMRouterBench Unseen, filling the last missing cell: CSCR's own loss
(cost_spectrum_info_nce) on CSCR's own FP (Perplexity, dim=192,
llmrouterbench_v15_900/perplexity), 900-probe budget -- the genuine
"CSCR reproduction" baseline for Unseen. All-seen version of this exact
combo already exists (fair_probe900_multiseed.py's "perplexity-900-cscrloss"
arm, mean=0.6712) and is used as the headline table's LLMRouterBench
"vs CSCR" reference; this is its Unseen counterpart, previously missing.

Mirrors cscrfp_comparloss_allseen_unseen_multiseed.py's Unseen branch
exactly (same make_split, same eval plumbing) but swaps COMPAR's
Combined/min(0.3,3) loss for CSCR's own cost_spectrum_info_nce
(train_cscr_fast), matching what
routerbench_ceilingfp_cscrloss_purev2_allseen_multiseed.py did for
RouterBench's missing-cell fill.
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
from routerbench_fair_probe1800_multiseed import precompute_cls, train_cscr_fast, eval_proj
from fair_probe900_multiseed import build_rows_with_dataset, build_cost_dict, build_setB_eval, build_items

SEEDS = [0, 1, 2]
UNSEEN_FRAC = 1 / 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PERPLEXITY_DIR = Path("local_descriptors/llmrouterbench_v15_900/perplexity")


def make_split(seed):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(common.MODELS_33))
    n_unseen = max(1, int(round(len(common.MODELS_33) * UNSEEN_FRAC)))
    unseen_idx = perm[:n_unseen]
    seen_idx = perm[n_unseen:]
    seen = [common.MODELS_33[i] for i in seen_idx]
    unseen = [common.MODELS_33[i] for i in unseen_idx]
    return seen, unseen


def main():
    import pickle
    with open("local_descriptors/llmrouterbench/setA_setB_split.pkl", "rb") as f:
        split = pickle.load(f)

    print(f"Loading frozen MiniLM ONCE (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    hidden_size = base_model.config.hidden_size

    all_rows = build_rows_with_dataset(split)
    cost_dict = build_cost_dict(split)
    b_texts_all, b_scores_all = build_setB_eval(split)
    model_col_idx = {m: i for i, m in enumerate(common.MODELS_33)}
    print(f"{len(all_rows)} raw Set A rows, Set B: {len(b_texts_all)} rows", flush=True)

    t0 = time.time()
    cls_setB = precompute_cls(b_texts_all, tokenizer, base_model)
    print(f"Set B embeddings cached in {time.time()-t0:.1f}s", flush=True)

    results = []
    print(f"\n{'#'*70}\nUNSEEN: Perplexity FP (dim=192) x CSCR's own loss, 900-probe\n{'#'*70}", flush=True)
    for seed in SEEDS:
        seen, unseen = make_split(seed)
        rows_seen = [(text, {m: scores[m] for m in seen}, ds) for text, scores, ds in all_rows]
        texts, targets, masks, raw_labels = build_items(rows_seen, seen)
        cls_tr = precompute_cls(texts, tokenizer, base_model)
        print(f"  seed={seed}: seen={len(seen)} unseen={len(unseen)} usable_rows={len(texts)}", flush=True)

        a_costs = np.array([cost_dict[m] for m in seen], dtype=np.float32)
        cost_norm = (a_costs - a_costs.min()) / (a_costs.max() - a_costs.min() + 1e-9)
        cost_norm_t = torch.from_numpy(cost_norm).float().to(DEVICE)
        n_bands = int(round(len(seen) ** 0.5))

        E_seen = np.stack([np.load(PERPLEXITY_DIR / f"{common.NAME_TO_SAFE[m]}.npy") for m in seen])
        E_seen_t = torch.from_numpy(E_seen).float().to(DEVICE)
        E_seen_t = E_seen_t / (E_seen_t.norm(dim=1, keepdim=True) + 1e-9)

        proj = train_cscr_fast(seed, cls_tr, raw_labels, cost_norm_t, E_seen_t, hidden_size, n_bands,
                                f"unseen-seed{seed}-perplexity-cscrloss")

        unseen_col_idx = [model_col_idx[m] for m in unseen]
        b_scores_unseen = b_scores_all[:, unseen_col_idx]
        b_label_maps = b_scores_unseen.astype(np.float32).tolist()
        b_costs = np.array([cost_dict[m] for m in unseen], dtype=np.float32)
        E_unseen = np.stack([np.load(PERPLEXITY_DIR / f"{common.NAME_TO_SAFE[m]}.npy") for m in unseen])
        E_unseen_norm = E_unseen / (np.linalg.norm(E_unseen, axis=1, keepdims=True) + 1e-12)

        knn_metrics, delta, p = eval_proj(proj, cls_setB, E_unseen_norm, unseen, b_label_maps, b_costs, seed)
        results.append({"seed": seed, "n_seen": len(seen), "n_unseen": len(unseen),
                         "knn": knn_metrics, "delta": delta, "p": p})
        print(f"  [unseen seed={seed}] AUDC={knn_metrics['audc']:.4f} Peak={knn_metrics['peak']:.4f} "
              f"delta={delta:+.4f} p={p:.4g}", flush=True)

    out_path = Path("local_descriptors/llmrouterbench_v15_900") / "cscrfp_cscrloss_unseen_results.json"
    json.dump(results, open(out_path, "w"), indent=2)

    audcs = [r["knn"]["audc"] for r in results]
    print("\n" + "=" * 90)
    print("LLMROUTERBENCH UNSEEN: Perplexity FP x CSCR's own loss (900-probe) -- CSCR reproduction baseline")
    print("=" * 90)
    print(f"  AUDC: " + " / ".join(f"{a:.4f}" for a in audcs) + f"  mean={np.mean(audcs):.4f} (std={np.std(audcs):.4f})")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
