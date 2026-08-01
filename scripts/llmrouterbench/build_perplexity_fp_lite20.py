"""Perplexity FP (528-dim) for the lightweight-20 pool: GPT-2 cross-entropy
of each model's raw_output text at the SAME 528 probes used for Ceiling FP.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import common_lite20 as common

DATA_DIR = Path("local_descriptors/llmrouterbench_lite20")
OUT_DIR = DATA_DIR / "perplexity"
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


def main():
    print(f"DEVICE: {DEVICE}", flush=True)
    with open(DATA_DIR / "probe_info.json", encoding="utf-8") as f:
        probes = json.load(f)
    with open(DATA_DIR / "setA_setB_split.pkl", "rb") as f:
        split = pickle.load(f)
    setA = split["setA"]

    tok = AutoTokenizer.from_pretrained(CE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(CE_MODEL_NAME).to(DEVICE)
    model.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for m in common.MODELS_20:
        vals = []
        for p in probes:
            ds = p["dataset"]
            local_i = p["local_idx_in_setA"]
            text = setA[ds]["raw_outputs"][m][local_i]
            ce = cross_entropy_fingerprint(text, tok, model)
            vals.append(float(np.exp(ce)) if not np.isnan(ce) else float("nan"))
        vals = np.array(vals, dtype=np.float64)
        n_bad = int(np.isnan(vals).sum())
        if n_bad:
            print(f"  {m}: zero-filling {n_bad}/{len(vals)} unscoreable probes", flush=True)
            vals[np.isnan(vals)] = 0.0
        vec = (vals / (np.linalg.norm(vals) + 1e-12)).astype(np.float32)
        np.save(OUT_DIR / f"{common.NAME_TO_SAFE[m]}.npy", vec)
        print(f"  {m:38s} shape={vec.shape}", flush=True)

    E = np.stack([np.load(OUT_DIR / f"{common.NAME_TO_SAFE[m]}.npy") for m in common.MODELS_20])
    sim = E @ E.T
    off = sim[~np.eye(len(common.MODELS_20), dtype=bool)]
    print(f"\nPerplexity FP lite20: pairwise cosine sim: mean={off.mean():.4f} std={off.std():.4f} "
          f"min={off.min():.4f} max={off.max():.4f}", flush=True)
    print(f"Saved -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
