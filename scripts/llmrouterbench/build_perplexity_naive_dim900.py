"""Naive probe=dim Perplexity FP for LLMRouterBench (companion to
build_routerbench_perplexity_fp_probesweep.py's bonus dim1800 variant) --
same 900 probes as build_fp_v15_900.py (identical deterministic selection,
reproduced here rather than re-reading, so no dependency on that run's
in-memory state), but this time:
  1. Raw per-probe values are SAVED (build_fp_v15_900.py discarded them
     right after binning into the 192-dim controlled FP -- gap noted in
     PROGRESS.md 24.4, fixed here so future variants don't require a full
     GPT-2 rescore).
  2. Output is the literal 900-dim descriptor (one probe = one dimension,
     no group-averaging), matching Ceiling FP's own probe-indexed
     convention -- the "naive fairness" reading the user flagged as maybe
     closer to what the professor pictures than the dims-fixed/probes-varied
     controlled version.
  3. A same-dim random-vector negative control, so a win can be checked
     against "just more free dimensions to fit 33 models" (23.3-style
     capacity confound).

Speed optimizations over build_fp_v15_900.py (which took ~96 min for the
same 900x33 scoring pass):
  - fp16 inference (GTX 1650 Ti has no tensor cores but still gets a solid
    speedup + memory headroom from fp16 on CUDA cores)
  - batch_size 8->16 (fp16 frees enough memory to be safe)
  - max_len 512->256 (GPT-2 CE fingerprint doesn't need the full response;
    first 256 tokens carries most of the fluency signal, and this halves
    attention cost ~4x, linear-layer cost ~2x)
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import common

OUT_ROOT = Path("local_descriptors/llmrouterbench_v15_900")
TARGET_TOTAL = 900
MIN_PROBES = 15
K_PCA_FOR_IMPORTANCE = 5
CE_MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
MAX_LEN = 256


def batched_perplexity_fingerprint(texts, tokenizer, model, batch_size=BATCH_SIZE, max_len=MAX_LEN):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    char_cap = max_len * 6
    out = np.zeros(len(texts), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = [(t if t else " ")[:char_cap] for t in texts[start:start + batch_size]]
            enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_len, padding=True)
            input_ids = enc["input_ids"].to(model.device)
            attn = enc["attention_mask"].to(model.device)
            logits = model(input_ids, attention_mask=attn).logits.float()
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


def main():
    print(f"DEVICE: {DEVICE}  batch_size={BATCH_SIZE} max_len={MAX_LEN}", flush=True)
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

    # reproduce the exact same 900-probe selection as build_fp_v15_900.py (deterministic)
    importance_by_ds = compute_dataset_importance(setA_by_ds)
    max_by_ds = {ds: len(setA_by_ds[ds]["queries"]) for ds in common.DATASETS}
    allocation, total = solve_allocation(importance_by_ds, max_by_ds, TARGET_TOTAL, MIN_PROBES)
    print(f"Reproduced allocation (total={total}): {allocation}", flush=True)

    probe_info = []
    for ds in common.DATASETS:
        scores = setA_by_ds[ds]["scores"]
        var = scores.var(axis=1)
        order = np.argsort(-var)
        n = allocation[ds]
        top_local = order[:n]
        for local_i in top_local:
            probe_info.append({"dataset": ds, "local_idx_in_setA": int(local_i)})
    print(f"{len(probe_info)} probes reproduced", flush=True)

    tok = AutoTokenizer.from_pretrained(CE_MODEL_NAME)
    ce_model = AutoModelForCausalLM.from_pretrained(CE_MODEL_NAME).half().to(DEVICE)
    ce_model.eval()

    raw = {m: np.zeros(len(probe_info), dtype=np.float64) for m in common.MODELS_33}
    t0 = time.time()
    for ds in common.DATASETS:
        idxs = [i for i, p in enumerate(probe_info) if p["dataset"] == ds]
        local_idxs = [probe_info[i]["local_idx_in_setA"] for i in idxs]
        print(f"  [{ds}] {len(idxs)} probes, scoring GPT-2 (fp16, batch={BATCH_SIZE}, max_len={MAX_LEN})...",
              flush=True)
        for j, m in enumerate(common.MODELS_33):
            texts = [setA_by_ds[ds]["raw_outputs"][m][i] for i in local_idxs]
            vals = batched_perplexity_fingerprint(texts, tok, ce_model)
            vals = np.nan_to_num(vals, nan=0.0)
            for k, i in enumerate(idxs):
                raw[m][i] = vals[k]
            if (j + 1) % 8 == 0 or j == len(common.MODELS_33) - 1:
                print(f"    {j+1}/{len(common.MODELS_33)} models done, {time.time()-t0:.1f}s total elapsed",
                      flush=True)

    raw_dir = OUT_ROOT
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez(raw_dir / "raw_perplexity_values_900.npz", **raw)
    json.dump(probe_info, open(raw_dir / "probe_info_900.json", "w", encoding="utf-8"), indent=2)
    print(f"\nSaved raw values -> {raw_dir / 'raw_perplexity_values_900.npz'} "
          f"(+ probe order -> probe_info_900.json)", flush=True)

    # naive probe=dim FP (no binning)
    naive_dir = OUT_ROOT / "perplexity-dim900"
    naive_dir.mkdir(parents=True, exist_ok=True)
    for m in common.MODELS_33:
        vec = raw[m]
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        np.save(naive_dir / f"{common.NAME_TO_SAFE[m]}.npy", vec.astype(np.float32))

    # same-dim random-vector negative control
    rand_dir = OUT_ROOT / "perplexity-dim900-randomcontrol"
    rand_dir.mkdir(parents=True, exist_ok=True)
    rng2 = np.random.RandomState(42)
    for m in common.MODELS_33:
        v = rng2.randn(len(probe_info))
        v = v / (np.linalg.norm(v) + 1e-12)
        np.save(rand_dir / f"{common.NAME_TO_SAFE[m]}.npy", v.astype(np.float32))

    for tag, d in [("perplexity-dim900", naive_dir), ("perplexity-dim900-randomcontrol", rand_dir)]:
        E = np.stack([np.load(d / f"{common.NAME_TO_SAFE[m]}.npy") for m in common.MODELS_33])
        sim = E @ E.T
        off = sim[~np.eye(len(common.MODELS_33), dtype=bool)]
        print(f"{tag}: shape={E.shape}  pairwise cos sim mean={off.mean():.4f} std={off.std():.4f}", flush=True)

    print(f"\nTotal GPT-2 scoring time: {time.time()-t0:.1f}s")
    print(f"Saved -> {naive_dir}, {rand_dir}")


if __name__ == "__main__":
    main()
