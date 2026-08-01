"""Build Perplexity FP for RouterBench, matching the exact split used by
routerbench_knn_test.py (SPLIT_SEED=42, 80/20 Set A/Set B) rather than
compute_descriptors_perplexity.py's own internal split (seed=47, 90/10) --
using that script's split directly here would risk probe/eval leakage since
the two splits don't align.

Methodology is otherwise identical to compute_descriptors_perplexity.py:
GPT-2 cross-entropy fingerprint of each model's real RESPONSE TEXT (not the
target model's own logprobs -- GPT-2 is the fixed fingerprinting backbone
for all models regardless of whether the target model is open-weight or a
closed API). No target-model weight/logprob access needed at all.
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

OUT_DIR = Path("local_descriptors/routerbench-perplexity")
SET_A_FRACTION = 0.8
SPLIT_SEED = 42
N_PROBES = 32
CE_MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cross_entropy_fingerprint(text, tokenizer, model, max_len=1024):
    with torch.no_grad():
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = enc["input_ids"].to(model.device)
        if input_ids.numel() == 0:
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


def main():
    print(f"DEVICE: {DEVICE}")
    print("Loading RouterBench...")
    f = hf_hub_download(repo_id="withmartian/routerbench", repo_type="dataset", filename="routerbench_0shot.pkl")
    import pandas as pd
    df = pd.read_pickle(f)
    df = df.dropna(subset=[m for m in MODELS] + ["eval_name", "prompt"])
    rng = np.random.RandomState(SPLIT_SEED)
    idx = rng.permutation(len(df))
    n_a = int(len(df) * SET_A_FRACTION)
    set_a = df.iloc[idx[:n_a]].reset_index(drop=True)
    print(f"Total {len(df)} rows -> Set A {len(set_a)} (Set B held out, untouched here)")

    import random
    rnd = random.Random(SPLIT_SEED)
    probe_idx = list(range(len(set_a)))
    rnd.shuffle(probe_idx)
    probe_idx = probe_idx[:N_PROBES]
    probes = set_a.iloc[probe_idx]
    print(f"Using {len(probes)} probes from Set A")

    print(f"Loading {CE_MODEL_NAME} (fixed fingerprinting backbone, same for every model)...")
    tok = AutoTokenizer.from_pretrained(CE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(CE_MODEL_NAME).to(DEVICE)
    model.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, model_col in zip(NAMES, MODELS):
        resp_col = f"{model_col}|model_response"
        vals = []
        for _, row in probes.iterrows():
            text = row[resp_col]
            vals.append(perplexity_fingerprint(text, tok, model))
        vals = np.array(vals, dtype=np.float64)
        n_bad = int(np.isinf(vals).sum())
        if n_bad:
            print(f"  {name}: zero-filling {n_bad}/{len(vals)} unscoreable probes")
            vals[np.isinf(vals)] = 0.0
        vec = (vals / (np.linalg.norm(vals) + 1e-12)).astype(np.float32)
        np.save(OUT_DIR / f"{name}.npy", vec)
        print(f"  {name:45s} shape={vec.shape}")

    E = np.stack([np.load(OUT_DIR / f"{n}.npy") for n in NAMES])
    sim = E @ E.T
    off = sim[~np.eye(len(NAMES), dtype=bool)]
    print(f"\nPairwise cosine sim: mean={off.mean():.4f} std={off.std():.4f} min={off.min():.4f} max={off.max():.4f}")
    print(f"Saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
