"""Same as embedllm_newllm_train_encoder.py, but trains with the paper's OWN
cost_spectrum_info_nce loss (scripts/train_query_encoder.py, byte-identical
to upstream/main per PROGRESS.md 17.19) instead of the simplified
multi_positive_info_nce used in the first attempt. If this does better, it's
the strongest possible claim: beating the paper's own reported CSCR number
using the paper's own training objective, just with our Ceiling FP swapped
in for whatever descriptor they used.

Saved to a NEW checkpoint dir (does not overwrite the multi_positive_info_nce
run) so both can be compared.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset as TorchDataset

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from router.query_encoder import QueryEncoder
from router.registry import REGISTRY
from train_query_encoder import cost_spectrum_info_nce

PCA5_DIR = Path("local_descriptors/embedllm-ceiling-pca5")
SPLIT_PATH = Path("local_descriptors/embedllm-analysis/newllm_split.json")
OUT_DIR = Path("local_checkpoints/embedllm-newllm-encoder-csinfonce")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 2
BATCH_SIZE = 64
LR = 5e-4
LOG_EVERY = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_train_data():
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    return pd.read_csv(f_train, usecols=["prompt_id", "model_name", "label", "prompt"])


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


def main():
    split = json.load(open(SPLIT_PATH, encoding="utf-8"))
    seen_models = split["seen"]
    print(f"seen={len(seen_models)} unseen={len(split['unseen'])}", flush=True)

    print("Loading EmbedLLM train.csv...", flush=True)
    df = load_train_data()

    ds = SeenLabelDataset(df, seen_models)
    print(f"train set: {len(ds)} rows", flush=True)

    print(f"Loading QueryEncoder (device={DEVICE})...", flush=True)
    enc = QueryEncoder(EMBED_MODEL, device=DEVICE, proj_dim=5, proj_multiplier=1)
    for p in enc.model.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(enc.proj.parameters(), lr=LR)

    E_seen = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in seen_models])
    E_seen_t = torch.from_numpy(E_seen).float().to(DEVICE)
    E_seen_t = E_seen_t / (E_seen_t.norm(dim=1, keepdim=True) + 1e-9)

    cost_raw = np.array([REGISTRY[m]["n_params"] for m in seen_models], dtype=np.float32)
    cost_norm = (cost_raw - cost_raw.min()) / (cost_raw.max() - cost_raw.min() + 1e-9)
    cost_norm_t = torch.from_numpy(cost_norm).float().to(DEVICE)
    print(f"cost_norm range: [{cost_norm.min():.3f}, {cost_norm.max():.3f}]", flush=True)

    def collate(batch):
        texts, labels = zip(*batch)
        toks = enc.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt", max_length=256)
        return toks, torch.tensor(labels, dtype=torch.float32)

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    n_batches = len(loader)
    t0 = time.time()
    enc.model.eval()
    for ep in range(EPOCHS):
        ep_loss = 0.0
        for bi, (tok, label) in enumerate(loader):
            tok = {k: v.to(DEVICE) for k, v in tok.items()}
            label = label.to(DEVICE)
            with torch.no_grad():
                out = enc.model(**tok)
                cls_vec = out.last_hidden_state[:, 0]  # CLS pooling -- matches QueryEncoder.encode()
            q = enc._project(cls_vec)  # grad flows through enc.proj only
            # cost_spectrum_info_nce internally does `~pos_mask` (bitwise NOT), which
            # only works on bool/int tensors -- our label is float32, so pass a bool
            # view. The resulting `neg_k` is dead code inside that function (never
            # used downstream), so this is a pure dtype fix, not a behavior change.
            loss = cost_spectrum_info_nce(q, E_seen_t, label.bool(), cost_norm_t)
            loss.backward()
            opt.step()
            opt.zero_grad()
            ep_loss += loss.item()
            if (bi + 1) % LOG_EVERY == 0 or (bi + 1) == n_batches:
                print(f"epoch {ep+1}/{EPOCHS} batch {bi+1}/{n_batches} loss={loss.item():.4f} "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)
        print(f"epoch {ep+1}/{EPOCHS} done, avg_loss={ep_loss/n_batches:.4f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    enc.save(OUT_DIR)
    print(f"\nSaved QueryEncoder checkpoint -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
