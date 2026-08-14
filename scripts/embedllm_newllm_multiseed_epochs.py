"""Epoch-scaling variant of the "new LLMs" multi-seed comparison, for Ceiling
V2 (full-category-average, PCA-5) only. Reuses the EXACT SAME seen/unseen
model splits already built for the 2-epoch run (newllm_split.json,
newllm_split_seed{1,2,3,4}.json) so epoch count is the only thing varying --
apples-to-apples against the existing 2-epoch results (mean AUDC 0.468).

EPOCHS=10 matches end_to_end.sh's own reference training invocation
(`--epochs 10`), which our earlier 2-epoch runs did NOT match.

Uses the fast batch-encode+cache eval (embedllm_newllm_fast_eval.py),
validated against the official run_audc_eval.py within ~0.004 AUDC on seed 0.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset as TorchDataset

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from router.query_encoder import QueryEncoder
from router.registry import REGISTRY
from train_query_encoder import cost_spectrum_info_nce
from embedllm_newllm_fast_eval import run_one as fast_eval

PCA5_DIR = Path("local_descriptors/embedllm-ceiling-pca5")
ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 10
BATCH_SIZE = 64
LR = 5e-4
LOG_EVERY = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEEDS = [0]  # pilot: single seed first, check epoch=10 vs epoch=2 (0.4731) before committing to all 5
CSCR_PAPER = {"audc": 0.4848, "peak": 0.565}


def split_path_for(seed):
    return ANALYSIS_DIR / ("newllm_split.json" if seed == 0 else f"newllm_split_seed{seed}.json")


def unseen_dir_for(seed):
    return Path("local_descriptors/embedllm-ceiling-pca5-unseen-only" if seed == 0
                else f"local_descriptors/embedllm-ceiling-pca5-unseen-only-seed{seed}")


class SeenLabelDataset(TorchDataset):
    def __init__(self, df, seen_models):
        name_to_idx = {n: i for i, n in enumerate(seen_models)}
        self.items = []
        for pid, grp in df.groupby("prompt_id", sort=False):
            text = grp["prompt"].iloc[0]
            label = [0.0] * len(seen_models)
            any_pos = False
            for m, v in zip(grp["model_name"], grp["label"]):
                if m in name_to_idx and v == 1:
                    label[name_to_idx[m]] = 1.0
                    any_pos = True
            if any_pos:
                self.items.append((text, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def train_encoder(seed, seen_models, df, tokenizer, base_model, epochs):
    ds = SeenLabelDataset(df, seen_models)
    print(f"  [seed {seed}] train set: {len(ds)} rows, {epochs} epochs", flush=True)

    enc = QueryEncoder.__new__(QueryEncoder)
    torch.nn.Module.__init__(enc)
    enc.tokenizer = tokenizer
    enc.model = base_model
    enc.device = DEVICE
    enc.hidden_size = base_model.config.hidden_size
    enc.proj_dim = 5
    enc.proj = torch.nn.Sequential(
        torch.nn.Linear(enc.hidden_size, enc.hidden_size, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(enc.hidden_size, 5, bias=False),
    ).to(DEVICE)
    enc.model.config.proj_dim = 5
    opt = torch.optim.AdamW(enc.proj.parameters(), lr=LR)

    E_seen = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in seen_models])
    E_seen_t = torch.from_numpy(E_seen).float().to(DEVICE)
    E_seen_t = E_seen_t / (E_seen_t.norm(dim=1, keepdim=True) + 1e-9)

    cost_raw = np.array([REGISTRY[m]["n_params"] for m in seen_models], dtype=np.float32)
    cost_norm = (cost_raw - cost_raw.min()) / (cost_raw.max() - cost_raw.min() + 1e-9)
    cost_norm_t = torch.from_numpy(cost_norm).float().to(DEVICE)

    def collate(batch):
        texts, labels = zip(*batch)
        toks = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt", max_length=256)
        return toks, torch.tensor(labels, dtype=torch.float32)

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    n_batches = len(loader)
    t0 = time.time()
    for ep in range(epochs):
        ep_loss = 0.0
        for bi, (tok, label) in enumerate(loader):
            tok = {k: v.to(DEVICE) for k, v in tok.items()}
            label = label.to(DEVICE)
            with torch.no_grad():
                out = base_model(**tok)
                cls_vec = out.last_hidden_state[:, 0]
            q = enc._project(cls_vec)
            loss = cost_spectrum_info_nce(q, E_seen_t, label.bool(), cost_norm_t)
            loss.backward()
            opt.step()
            opt.zero_grad()
            ep_loss += loss.item()
            if (bi + 1) % LOG_EVERY == 0 or (bi + 1) == n_batches:
                print(f"  [seed {seed}] epoch {ep+1}/{epochs} batch {bi+1}/{n_batches} "
                      f"loss={loss.item():.4f} elapsed={time.time()-t0:.1f}s", flush=True)
        print(f"  [seed {seed}] epoch {ep+1}/{epochs} done, avg_loss={ep_loss/n_batches:.4f}", flush=True)
    return enc


def main():
    print("Loading EmbedLLM train.csv + registry...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "label", "prompt"])

    print(f"Loading frozen MiniLM base ONCE (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    results = []
    for seed in SEEDS:
        print(f"\n{'='*60}\nSEED {seed} (epochs={EPOCHS})\n{'='*60}", flush=True)
        split = json.load(open(split_path_for(seed), encoding="utf-8"))
        seen_models = split["seen"]

        ckpt_dir = Path(f"local_checkpoints/embedllm-newllm-encoder-csinfonce-epoch{EPOCHS}-seed{seed}")
        if (ckpt_dir / "config.json").exists() and (ckpt_dir / "proj.pt").exists():
            print(f"  [seed {seed}] checkpoint already exists, skipping retrain -> {ckpt_dir}", flush=True)
        else:
            enc = train_encoder(seed, seen_models, df, tokenizer, base_model, EPOCHS)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            enc.save(ckpt_dir)
            print(f"  [seed {seed}] saved encoder -> {ckpt_dir}", flush=True)

        r = fast_eval(str(split_path_for(seed)), str(unseen_dir_for(seed)), str(ckpt_dir),
                       label=f"seed{seed}-epoch{EPOCHS}")
        r["seed"] = seed
        results.append(r)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / f"newllm_multiseed_epoch{EPOCHS}_results.json"
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n" + "=" * 70)
    print(f"EPOCH={EPOCHS} MULTI-SEED SUMMARY (Ceiling V2, cost_spectrum_info_nce)")
    print("=" * 70)
    audcs = [r["knn"]["audc"] for r in results]
    peaks = [r["knn"]["peak"] for r in results]
    for r in results:
        beats = "BEATS" if r["knn"]["audc"] > CSCR_PAPER["audc"] else "below"
        print(f"seed={r['seed']}: AUDC={r['knn']['audc']:.4f} ({beats} CSCR 0.4848)  Peak={r['knn']['peak']:.4f}")
    print(f"\nmean AUDC={np.mean(audcs):.4f} (std={np.std(audcs):.4f}, min={min(audcs):.4f}, max={max(audcs):.4f})")
    print(f"mean Peak={np.mean(peaks):.4f} (std={np.std(peaks):.4f})")
    print(f"seeds beating CSCR AUDC: {sum(1 for a in audcs if a > CSCR_PAPER['audc'])}/{len(audcs)}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
