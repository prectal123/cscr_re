"""LLMRouterBench Ceiling V1.5 + Perplexity(CSCR) FP, both built from the SAME
900-probe budget, PCA-importance-weighted allocation across the 8 datasets
(same recipe as build_routerbench_ceiling_v15.py, ported to the 33-model
pool's 8-dataset category structure). Matches this benchmark's existing
convention (build_perplexity_fp.py) of using identical probes for both FP
types -- unlike RouterBench, where the two FPs historically used separate
probe pools.

Output stays fixed at 192 dims (24 bins/dataset, same as the original
192-probe build_ceiling_fp.py/build_perplexity_fp.py) so results are
directly comparable to the existing full-budget numbers. Each dataset's
PCA-weighted probe allocation is binned into that dataset's fixed 24-dim
slot via group-averaging (same N_DIMS-fixed/N_PROBES-varied trick used for
RouterBench's probe sweep) -- NOT probe=dim, since 900 probes into 192 dims
means most dims average multiple probes.

900 (not 1800) because LLMRouterBench's Set A is only 2345 rows total across
all 8 datasets combined (vs RouterBench's 29197) -- 1800 would have meant
using ~77% of all Set A data as "probes", which isn't a meaningful budget
increase, just using almost everything. 900 (~38% of Set A) keeps a real
gap between probe budget and full-Set-A-training-data-availability while
still being a large increase over the original 192.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import common

OUT_DIR = Path("local_descriptors/llmrouterbench_v15_900")
TARGET_TOTAL = 900
MIN_PROBES = 15
K_PCA_FOR_IMPORTANCE = 5
CE_MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_DIMS_PER_DATASET = 24  # matches original 192 = 8 * 24 convention


def batched_perplexity_fingerprint(texts, tokenizer, model, batch_size=8, max_len=512):
    """Batched GPT-2 cross-entropy scoring -- was previously one text at a
    time (a few hundred ms/call, ~9-10 calls/sec, projected 45+ min for 900
    probes x 33 models). Batching gives the GPU actual parallel work instead
    of 30k sequential tiny forward passes."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # pre-truncate the raw STRING before tokenizing -- some datasets (e.g. aime
    # reasoning CoT) have responses up to ~37k tokens; the tokenizer still has
    # to fully encode the string before token-level truncation kicks in, so
    # long inputs are slow regardless of max_length. ~4 chars/token for GPT-2
    # BPE, so max_len*6 chars is a safe upper bound that's cheap to slice.
    char_cap = max_len * 6
    out = np.zeros(len(texts), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = [(t if t else " ")[:char_cap] for t in texts[start:start + batch_size]]
            enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_len,
                             padding=True)
            input_ids = enc["input_ids"].to(model.device)
            attn = enc["attention_mask"].to(model.device)
            logits = model(input_ids, attention_mask=attn).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attn[:, 1:].contiguous().float()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            tok_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            tok_loss = tok_loss.view(shift_labels.size()) * shift_mask
            n_tok = shift_mask.sum(dim=1).clamp(min=1)
            ce = (tok_loss.sum(dim=1) / n_tok).cpu().numpy()
            vals = np.where(n_tok.cpu().numpy() >= 1, np.exp(ce), np.nan)
            out[start:start + len(batch)] = vals
    return out


def compute_dataset_importance(setA_by_ds):
    """(8,) importance per dataset from PCA loadings of the (33-model x 8-dataset)
    mean-accuracy matrix, mean-centered, same recipe as embedllm_pca_loading_analysis.py."""
    n_models = len(common.MODELS_33)
    raw = np.zeros((n_models, len(common.DATASETS)))
    for di, ds in enumerate(common.DATASETS):
        raw[:, di] = setA_by_ds[ds]["scores"].mean(axis=0)
    pool_mean = raw.mean(axis=0, keepdims=True)
    centered = raw - pool_mean

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()
    k = min(K_PCA_FOR_IMPORTANCE, Vt.shape[0])
    loadings = Vt[:k]
    importance = (explained[:k, None] * (loadings ** 2)).sum(axis=0)
    importance = importance / importance.sum()
    print(f"Explained variance top-{k}: {[f'{e:.4f}' for e in explained[:k]]} "
          f"(cumulative {explained[:k].sum():.4f})", flush=True)
    return {ds: float(v) for ds, v in zip(common.DATASETS, importance)}


def solve_allocation(importance_by_ds, max_by_ds, target_total, min_n):
    datasets = common.DATASETS
    imp = np.array([importance_by_ds[d] for d in datasets])
    caps = np.array([max_by_ds[d] for d in datasets])
    sqrt_imp = np.sqrt(imp)

    def total_for_scale(scale):
        n = np.clip(np.round(sqrt_imp * scale), min_n, caps)
        return n.sum(), n

    lo, hi = 0.0, 1e6
    for _ in range(60):
        mid = (lo + hi) / 2
        total, n = total_for_scale(mid)
        if total < target_total:
            lo = mid
        else:
            hi = mid
    total, n = total_for_scale(hi)
    return {d: int(v) for d, v in zip(datasets, n)}, int(total)


def build_group_bounds(n_probes, n_dims):
    edges = np.linspace(0, n_probes, n_dims + 1).astype(int)
    return list(zip(edges[:-1], edges[1:]))


def main():
    print(f"DEVICE: {DEVICE}", flush=True)
    setA_by_ds = {}
    for ds in common.DATASETS:
        queries, scores, costs, raw_outputs = common.build_wide_table(ds)
        n = len(queries)
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        n_a = int(n * 0.8)
        idx_a = perm[:n_a]
        setA_by_ds[ds] = {
            "queries": [queries[i] for i in idx_a],
            "scores": scores[idx_a],
            "raw_outputs": {m: [raw_outputs[m][i] for i in idx_a] for m in common.MODELS_33},
        }
        print(f"  {ds:30s} SetA={len(idx_a)}", flush=True)

    importance_by_ds = compute_dataset_importance(setA_by_ds)
    max_by_ds = {ds: len(setA_by_ds[ds]["queries"]) for ds in common.DATASETS}
    allocation, total = solve_allocation(importance_by_ds, max_by_ds, TARGET_TOTAL, MIN_PROBES)

    print("\n" + "=" * 80)
    print(f"PROBE ALLOCATION (target={TARGET_TOTAL}, actual total={total})")
    print("=" * 80)
    for ds in sorted(common.DATASETS, key=lambda d: -importance_by_ds[d]):
        print(f"  {ds:30s} importance={importance_by_ds[ds]:.4f}  probes={allocation[ds]:4d}  "
              f"(cap={max_by_ds[ds]})", flush=True)

    # select top-variance probes per dataset, up to allocation
    probe_info = []
    for ds in common.DATASETS:
        scores = setA_by_ds[ds]["scores"]
        var = scores.var(axis=1)
        order = np.argsort(-var)
        n = allocation[ds]
        top_local = order[:n]
        for local_i in top_local:
            probe_info.append({"dataset": ds, "local_idx_in_setA": int(local_i)})

    print(f"\n{len(probe_info)} probes selected total", flush=True)

    # --- Ceiling-900: raw accuracy at selected probes, binned into 24 dims/dataset ---
    ceiling_dir = OUT_DIR / "ceiling"
    ceiling_dir.mkdir(parents=True, exist_ok=True)
    perplexity_dir = OUT_DIR / "perplexity"
    perplexity_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(CE_MODEL_NAME)
    ce_model = AutoModelForCausalLM.from_pretrained(CE_MODEL_NAME).to(DEVICE)
    ce_model.eval()

    ceiling_vecs = {m: [] for m in common.MODELS_33}
    perplexity_vecs = {m: [] for m in common.MODELS_33}

    for ds in common.DATASETS:
        ds_probes = [p for p in probe_info if p["dataset"] == ds]
        n_probes_ds = len(ds_probes)
        groups = build_group_bounds(n_probes_ds, N_DIMS_PER_DATASET)
        scores = setA_by_ds[ds]["scores"]
        local_idxs = [p["local_idx_in_setA"] for p in ds_probes]

        print(f"  [{ds}] {n_probes_ds} probes -> {N_DIMS_PER_DATASET} dims, scoring GPT-2...", flush=True)
        ds_t0 = time.time()
        for j, m in enumerate(common.MODELS_33):
            acc_vals = scores[local_idxs, j]  # (n_probes_ds,)
            acc_dims = np.array([acc_vals[a:b].mean() if b > a else 0.0 for a, b in groups])
            ceiling_vecs[m].append(acc_dims)

            texts = [setA_by_ds[ds]["raw_outputs"][m][i] for i in local_idxs]
            ce_vals = batched_perplexity_fingerprint(texts, tok, ce_model)
            ce_vals = np.nan_to_num(ce_vals, nan=0.0)
            ce_dims = np.array([ce_vals[a:b].mean() if b > a else 0.0 for a, b in groups])
            perplexity_vecs[m].append(ce_dims)
            if (j + 1) % 8 == 0 or j == len(common.MODELS_33) - 1:
                elapsed = time.time() - ds_t0
                print(f"    {j+1}/{len(common.MODELS_33)} models done, {elapsed:.1f}s elapsed", flush=True)

    for m in common.MODELS_33:
        c = np.concatenate(ceiling_vecs[m])  # (192,)
        p = np.concatenate(perplexity_vecs[m])
        ceiling_vecs[m] = c
        perplexity_vecs[m] = p

    # mean-center Ceiling across models (per-dim), then L2-normalize per model
    C = np.stack([ceiling_vecs[m] for m in common.MODELS_33])  # (33, 192)
    C = C - C.mean(axis=0, keepdims=True)
    for j, m in enumerate(common.MODELS_33):
        vec = C[j] / (np.linalg.norm(C[j]) + 1e-12)
        np.save(ceiling_dir / f"{common.NAME_TO_SAFE[m]}.npy", vec.astype(np.float32))

    for m in common.MODELS_33:
        vec = perplexity_vecs[m]
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        np.save(perplexity_dir / f"{common.NAME_TO_SAFE[m]}.npy", vec.astype(np.float32))

    for tag, d in [("Ceiling-900", ceiling_dir), ("Perplexity-900", perplexity_dir)]:
        E = np.stack([np.load(d / f"{common.NAME_TO_SAFE[m]}.npy") for m in common.MODELS_33])
        sim = E @ E.T
        off = sim[~np.eye(len(common.MODELS_33), dtype=bool)]
        print(f"{tag}: shape={E.shape}  pairwise cos sim mean={off.mean():.4f} std={off.std():.4f}", flush=True)

    ANALYSIS_DIR = Path("local_descriptors/llmrouterbench_v15_900")
    json.dump({"allocation": allocation, "total_probes": total, "importance": importance_by_ds},
               open(ANALYSIS_DIR / "allocation.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved FPs -> {ceiling_dir}, {perplexity_dir}")
    print(f"Saved allocation -> {ANALYSIS_DIR / 'allocation.json'}")


if __name__ == "__main__":
    main()
