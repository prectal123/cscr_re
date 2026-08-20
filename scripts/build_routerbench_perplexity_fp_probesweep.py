"""Fixes the N_PROBES==N_DIMS confound flagged in PROGRESS.md 23.3 before
running the mentor-requested "give CSCR's Perplexity FP the same probe
budget as Combined GRPO" fairness check.

build_routerbench_perplexity_fp.py hard-codes N_PROBES=32 (a stale debug
value never updated to the project's own N=192 standard, per 23.3) AND uses
one probe = one output dimension. Naively bumping N_PROBES to 1800 would
ALSO bump the FP to 1800 dimensions -- for an 11-model pool that swaps "more
reliable per-dimension estimate" (what the mentor's claim is actually about)
for "1800 degrees of freedom fit to 11 points" (a confound in the opposite
direction, likely to look like an improvement for the wrong reason).

This script separates the two:
  - N_DIMS = 32, held FIXED at the original/current dimensionality.
  - N_PROBES in {32, 192, 1800}: total GPT-2-scored probes SAMPLED, then
    partitioned into 32 roughly-equal groups and AVERAGED within each group
    to produce that dimension's value. N_PROBES=32 (1 probe/dim, no
    averaging) reproduces the original script's output exactly, so it also
    serves as the correctness self-check.
  - Nested probe sets (32 subset of 192 subset of 1800, same SPLIT_SEED=42
    shuffle as the original script) so scoring is only ever done once per
    probe -- the three N_PROBES variants are built from one GPT-2 pass over
    the full 1800, not three separate passes.

A bonus true probe=dim 1800-dimensional variant (matching Ceiling FP's own
probe-indexed convention) is also built, together with a same-dimensionality
random-vector negative control, so if that version looks better we can tell
whether it's real signal or just extra free dimensions (17.11-style check).
"""
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = [
    'claude-instant-v1', 'claude-v1', 'claude-v2', 'gpt-3.5-turbo-1106',
    'gpt-4-1106-preview', 'meta/code-llama-instruct-34b-chat',
    'meta/llama-2-70b-chat', 'mistralai/mistral-7b-chat',
    'mistralai/mixtral-8x7b-chat', 'zero-one-ai/Yi-34B-Chat',
    'WizardLM/WizardLM-13B-V1.2',
]
NAMES = [m.replace("/", "__") for m in MODELS]

OUT_ROOT = Path("local_descriptors")
SET_A_FRACTION = 0.8
SPLIT_SEED = 42
N_DIMS = 32
N_PROBES_SWEEP = [32, 192, 1800]
CE_MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cross_entropy_fingerprint(text, tokenizer, model, max_len=1024):
    with torch.no_grad():
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = enc["input_ids"].to(model.device)
        if input_ids.numel() < 2:
            return float("nan")
        logits = model(input_ids).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return token_loss.mean().item()


def perplexity_fingerprint(text, tokenizer, model, max_len=1024):
    ce = cross_entropy_fingerprint(text, tokenizer, model, max_len=max_len)
    return float("inf") if np.isnan(ce) else float(np.exp(ce))


def build_group_bounds(n_probes, n_dims):
    """n_dims roughly-equal contiguous groups covering n_probes items."""
    edges = np.linspace(0, n_probes, n_dims + 1).astype(int)
    return list(zip(edges[:-1], edges[1:]))


def main():
    print(f"DEVICE: {DEVICE}", flush=True)
    print("Loading RouterBench...", flush=True)
    f = hf_hub_download(repo_id="withmartian/routerbench", repo_type="dataset", filename="routerbench_0shot.pkl")
    import pandas as pd
    df = pd.read_pickle(f)
    df = df.dropna(subset=[m for m in MODELS] + ["eval_name", "prompt"])
    rng = np.random.RandomState(SPLIT_SEED)
    idx = rng.permutation(len(df))
    n_a = int(len(df) * SET_A_FRACTION)
    set_a = df.iloc[idx[:n_a]].reset_index(drop=True)
    print(f"Total {len(df)} rows -> Set A {len(set_a)} (Set B untouched)", flush=True)

    max_n = max(N_PROBES_SWEEP)
    import random
    rnd = random.Random(SPLIT_SEED)
    probe_idx = list(range(len(set_a)))
    rnd.shuffle(probe_idx)
    probe_idx = probe_idx[:max_n]  # same shuffle/seed as original script -> nested subsets
    probes = set_a.iloc[probe_idx].reset_index(drop=True)
    print(f"Sampled {len(probes)} probes total (nested subsets for {N_PROBES_SWEEP})", flush=True)

    print(f"Loading {CE_MODEL_NAME} (fixed fingerprinting backbone)...", flush=True)
    tok = AutoTokenizer.from_pretrained(CE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(CE_MODEL_NAME).to(DEVICE)
    model.eval()

    print(f"\nScoring all {len(probes)} probes x {len(NAMES)} models with GPT-2 (done once)...", flush=True)
    raw = {}  # name -> (max_n,) array of raw perplexity values, in probe order
    for name, model_col in zip(NAMES, MODELS):
        resp_col = f"{model_col}|model_response"
        vals = []
        for i, (_, row) in enumerate(probes.iterrows()):
            vals.append(perplexity_fingerprint(row[resp_col], tok, model))
            if (i + 1) % 300 == 0:
                print(f"  [{name}] {i+1}/{len(probes)}", flush=True)
        vals = np.array(vals, dtype=np.float64)
        n_bad = int(np.isinf(vals).sum())
        if n_bad:
            print(f"  {name}: zero-filling {n_bad}/{len(vals)} unscoreable probes", flush=True)
            vals[np.isinf(vals)] = 0.0
        raw[name] = vals
        print(f"  {name:45s} done", flush=True)

    raw_path = OUT_ROOT / "routerbench-perplexity-analysis"
    raw_path.mkdir(parents=True, exist_ok=True)
    np.savez(raw_path / "raw_perplexity_values.npz", **raw)
    print(f"\nSaved raw per-probe values -> {raw_path / 'raw_perplexity_values.npz'}", flush=True)

    # --- N_DIMS=32 fixed, N_PROBES swept (averaged within groups) ---
    for n_probes in N_PROBES_SWEEP:
        groups = build_group_bounds(n_probes, N_DIMS)
        out_dir = OUT_ROOT / f"routerbench-perplexity-nprobes{n_probes}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in NAMES:
            vals_full = raw[name][:n_probes]
            dim_vals = np.array([vals_full[a:b].mean() for a, b in groups], dtype=np.float64)
            vec = (dim_vals / (np.linalg.norm(dim_vals) + 1e-12)).astype(np.float32)
            np.save(out_dir / f"{name}.npy", vec)
        E = np.stack([np.load(out_dir / f"{n}.npy") for n in NAMES])
        sim = E @ E.T
        off = sim[~np.eye(len(NAMES), dtype=bool)]
        print(f"N_PROBES={n_probes:5d} (32-dim, {n_probes//N_DIMS} probes/dim avg): "
              f"pairwise cos sim mean={off.mean():.4f} std={off.std():.4f} -> {out_dir}", flush=True)

    # --- bonus: true probe=dim 1800-dim variant + random negative control ---
    bonus_dir = OUT_ROOT / "routerbench-perplexity-dim1800"
    bonus_dir.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        vals = raw[name]  # already length max_n == 1800
        vec = (vals / (np.linalg.norm(vals) + 1e-12)).astype(np.float32)
        np.save(bonus_dir / f"{name}.npy", vec)
    print(f"\nBonus true-1800-dim (probe=dim) variant -> {bonus_dir}", flush=True)

    rand_dir = OUT_ROOT / "routerbench-perplexity-dim1800-randomcontrol"
    rand_dir.mkdir(parents=True, exist_ok=True)
    rng2 = np.random.RandomState(SPLIT_SEED)
    for name in NAMES:
        v = rng2.randn(max_n)
        v = (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)
        np.save(rand_dir / f"{name}.npy", v)
    print(f"Random-vector negative control (same 1800-dim) -> {rand_dir}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
