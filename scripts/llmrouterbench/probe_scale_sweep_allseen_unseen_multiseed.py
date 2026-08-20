"""LLMRouterBench PROBE-COUNT SWEEP, All-seen + Unseen, 3 seeds each,
COMPAR's own Combined loss (Catfilter + min(0.3,3)), uniform allocation,
no compression (dim=8).

Tests whether the EmbedLLM finding ("even ~96 probes out of tens of
thousands gets you near-V2 Unseen AUDC") holds on this much smaller /
coarser pool: 33 models (vs 111) and 8 categories (vs 80).

Expensive parts (frozen-MiniLM text encoding) are computed ONCE and
reused across every probe-count target -- only the FP (E) changes
between targets, not the query texts/targets/masks, so re-encoding per
target would be pure waste.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts/llmrouterbench")
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import common
from routerbench_fair_probe1800_multiseed import precompute_cls, make_proj, eval_proj
from fair900_uniform_nocompress_allseen_unseen_multiseed import (
    build_items, minpctcap_loss, train_fast, make_split,
)
from fair_probe900_multiseed import build_rows_with_dataset, build_cost_dict, build_setB_eval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TARGETS = [64, 160, 320, 480, 640, 900, 1200, 1800, "full"]
FP_DIR_ROOT = Path("local_descriptors")


def fp_dir_for(target):
    if target == "full":
        return FP_DIR_ROOT / "llmrouterbench-ceiling-purev2"
    return FP_DIR_ROOT / f"llmrouterbench-ceiling-uniform-nocompress-{target}"


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

    # ---------- precompute ONCE: all-seen training items + embeddings ----------
    print("\nPrecomputing all-seen training items + embeddings (once, reused across all targets)...", flush=True)
    texts_as, targets_as, masks_as = build_items(all_rows, common.MODELS_33)
    print(f"  {len(texts_as)} usable Set A rows (all-seen)", flush=True)
    t0 = time.time()
    cls_tr_as = precompute_cls(texts_as, tokenizer, base_model)
    print(f"  encoded in {time.time()-t0:.1f}s", flush=True)

    # ---------- precompute ONCE per seed: unseen split + training items + embeddings ----------
    print("\nPrecomputing per-seed unseen splits + training items + embeddings (once, reused across all targets)...", flush=True)
    unseen_cache = {}
    for seed in SEEDS:
        seen, unseen = make_split(seed)
        rows_seen = [(text, {m: scores[m] for m in seen}, ds) for text, scores, ds in all_rows]
        texts_u, targets_u, masks_u = build_items(rows_seen, seen)
        t0 = time.time()
        cls_tr_u = precompute_cls(texts_u, tokenizer, base_model)
        print(f"  seed={seed}: seen={len(seen)} unseen={len(unseen)} usable_rows={len(texts_u)} "
              f"encoded in {time.time()-t0:.1f}s", flush=True)

        unseen_col_idx = [model_col_idx[m] for m in unseen]
        b_scores_unseen = b_scores_all[:, unseen_col_idx]
        b_label_maps_u = b_scores_unseen.astype(np.float32).tolist()
        b_costs_u = np.array([cost_dict[m] for m in unseen], dtype=np.float32)

        unseen_cache[seed] = dict(
            seen=seen, unseen=unseen, targets_u=targets_u, masks_u=masks_u,
            cls_tr_u=cls_tr_u, b_label_maps_u=b_label_maps_u, b_costs_u=b_costs_u,
        )

    b_label_maps_as = b_scores_all.astype(np.float32).tolist()
    b_costs_as = np.array([cost_dict[m] for m in common.MODELS_33], dtype=np.float32)

    results = {}

    # ---------- sweep over probe-count targets ----------
    for target in TARGETS:
        fp_dir = fp_dir_for(target)
        print(f"\n{'#'*70}\nTARGET={target} (dir={fp_dir})\n{'#'*70}", flush=True)
        results[str(target)] = {"allseen": [], "unseen": []}

        # all-seen
        E = np.stack([np.load(fp_dir / f"{common.NAME_TO_SAFE[m]}.npy") for m in common.MODELS_33])
        E_t = torch.from_numpy(E).float().to(DEVICE)
        E_t = E_t / (E_t.norm(dim=1, keepdim=True) + 1e-9)
        E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

        for seed in SEEDS:
            proj = train_fast(seed, cls_tr_as, targets_as, masks_as, E_t, hidden_size,
                               f"target{target}-allseen-seed{seed}")
            knn_metrics, delta, p = eval_proj(proj, cls_setB, E_norm, common.MODELS_33,
                                               b_label_maps_as, b_costs_as, seed)
            results[str(target)]["allseen"].append({"seed": seed, "knn": knn_metrics, "delta": delta, "p": p})
        audcs = [r["knn"]["audc"] for r in results[str(target)]["allseen"]]
        print(f"  [target={target} allseen] MEAN AUDC={np.mean(audcs):.4f} (std={np.std(audcs):.4f})", flush=True)

        # unseen
        for seed in SEEDS:
            c = unseen_cache[seed]
            seen, unseen = c["seen"], c["unseen"]
            E_seen = np.stack([np.load(fp_dir / f"{common.NAME_TO_SAFE[m]}.npy") for m in seen])
            E_seen_t = torch.from_numpy(E_seen).float().to(DEVICE)
            E_seen_t = E_seen_t / (E_seen_t.norm(dim=1, keepdim=True) + 1e-9)

            proj = train_fast(seed, c["cls_tr_u"], c["targets_u"], c["masks_u"], E_seen_t, hidden_size,
                               f"target{target}-unseen-seed{seed}")

            E_unseen = np.stack([np.load(fp_dir / f"{common.NAME_TO_SAFE[m]}.npy") for m in unseen])
            E_unseen_norm = E_unseen / (np.linalg.norm(E_unseen, axis=1, keepdims=True) + 1e-12)

            knn_metrics, delta, p = eval_proj(proj, cls_setB, E_unseen_norm, unseen,
                                               c["b_label_maps_u"], c["b_costs_u"], seed)
            results[str(target)]["unseen"].append({
                "seed": seed, "n_seen": len(seen), "n_unseen": len(unseen),
                "knn": knn_metrics, "delta": delta, "p": p,
            })
        audcs = [r["knn"]["audc"] for r in results[str(target)]["unseen"]]
        print(f"  [target={target} unseen] MEAN AUDC={np.mean(audcs):.4f} (std={np.std(audcs):.4f})", flush=True)

        out_path = Path("local_descriptors/llmrouterbench-analysis")
        out_path.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(out_path / "probe_scale_sweep_allseen_unseen_results.json", "w"), indent=2)

    print("\n" + "=" * 90)
    print("LLMROUTERBENCH PROBE-COUNT SWEEP -- uniform, uncompressed, Combined loss")
    print("=" * 90)
    print(f"{'target':>8s} {'allseen AUDC':>14s} {'unseen AUDC':>14s}")
    for target in TARGETS:
        r = results[str(target)]
        as_audcs = [x["knn"]["audc"] for x in r["allseen"]]
        un_audcs = [x["knn"]["audc"] for x in r["unseen"]]
        print(f"{str(target):>8s} {np.mean(as_audcs):>8.4f}±{np.std(as_audcs):<5.4f} "
              f"{np.mean(un_audcs):>8.4f}±{np.std(un_audcs):<5.4f}")

    out_path = Path("local_descriptors/llmrouterbench-analysis/probe_scale_sweep_allseen_unseen_results.json")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
