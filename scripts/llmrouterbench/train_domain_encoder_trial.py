"""Trial: train QueryEncoder's projection head (22-dim, matching Ceiling V2)
so that cosine_sim(encoder(query), domain_FP[model]) predicts model's
GRPO-style per-query relative advantage -- (raw_score - query_mean) / (query_std + eps)
across the 20-model group for that query. MiniLM backbone frozen; only the
small projection head trains. domain_FP[model] (Ceiling V2, already built)
is used as a FIXED anchor, not trained.

This is a quick proof-of-concept, not the final rigorous run: small query
subset, few epochs, single seed. Goal is to check the training signal is
sane before committing more time/compute.
"""
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, "scripts/llmrouterbench")
import common_lite20 as common


class QueryEncoder(nn.Module):
    """Inlined copy of src/router/query_encoder.py -- avoids importing the
    `router` package (which pulls in faiss/matplotlib/etc. via __init__.py
    for unrelated modules). Same architecture/save format, so a checkpoint
    saved here can be loaded by the original class later if needed."""

    def __init__(self, model_name: str, device: str | None = None,
                 proj_dim: int = 256, proj_multiplier: int = 1):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.hidden_size = self.model.config.hidden_size
        self.proj_dim = proj_dim
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, proj_multiplier * self.hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(proj_multiplier * self.hidden_size, proj_dim, bias=False),
        ).to(self.device)
        self.model.config.proj_dim = proj_dim

    @torch.no_grad()
    def encode(self, texts, project: bool = True) -> np.ndarray:
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=256).to(self.device)
        outputs = self.model(**batch)
        last_hidden = outputs.last_hidden_state
        cls_vec = last_hidden[:, 0]
        if project:
            z = self._project(cls_vec)
            vec = z.cpu().numpy().astype(np.float32)
        else:
            vec = cls_vec.cpu().numpy().astype(np.float32)
        return vec[0] if single_input else vec

    def _project(self, cls_vec: torch.Tensor) -> torch.Tensor:
        z = self.proj(cls_vec)
        z = F.normalize(z, dim=-1)
        return z

    def save(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        torch.save(self.proj.state_dict(), out_dir / "proj.pt")

DATA_DIR = Path("local_descriptors/llmrouterbench_lite20")
DOMAIN_FP_DIR = DATA_DIR / "ceiling_categoryrate"
OUT_DIR = Path("local_checkpoints/domain-encoder-trial")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TRIAL_N_QUERIES = 12426  # full Set A (22 categories)
N_EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
HOLDOUT_FRAC = 0.15
SEED = 0


def load_query_score_rows():
    with open(DATA_DIR / "setA_setB_split.pkl", "rb") as f:
        split = pickle.load(f)
    setA = split["setA"]
    rows = []  # (query_text, (20,) score array)
    for ds in common.DATASETS:
        d = setA[ds]
        n = len(d["queries"])
        for i in range(n):
            rows.append((d["queries"][i], d["scores"][i, :].astype(np.float32)))
    return rows


def grpo_target(scores: np.ndarray) -> np.ndarray:
    mean = scores.mean()
    std = scores.std()
    return (scores - mean) / (std + 1e-6)


def evaluate_holdout(enc, holdout_texts, holdout_targets, domain_FP):
    from scipy.stats import spearmanr
    enc.model.eval()
    with torch.no_grad():
        holdout_emb = enc.encode(holdout_texts, project=True)  # (N, 22) numpy, L2-normalized
    holdout_cos = holdout_emb @ domain_FP.T  # (N, 20)
    per_query_rho = []
    for i in range(len(holdout_texts)):
        rho, _ = spearmanr(holdout_cos[i], holdout_targets[i])
        if not np.isnan(rho):
            per_query_rho.append(rho)
    return np.array(per_query_rho)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    print("Loading Set A query/score rows...", flush=True)
    rows = load_query_score_rows()
    random.shuffle(rows)
    rows = rows[:TRIAL_N_QUERIES]
    n_holdout = int(len(rows) * HOLDOUT_FRAC)
    holdout_rows, train_rows = rows[:n_holdout], rows[n_holdout:]
    print(f"train={len(train_rows)}  holdout={len(holdout_rows)}", flush=True)

    domain_FP = np.stack([np.load(DOMAIN_FP_DIR / f"{common.NAME_TO_SAFE[m]}.npy") for m in common.MODELS_20])
    domain_FP_t = torch.tensor(domain_FP, dtype=torch.float32, device=device)  # (20, 22)

    enc = QueryEncoder(EMBED_MODEL, device=device, proj_dim=22)
    for p in enc.model.parameters():
        p.requires_grad = False  # freeze MiniLM backbone
    optimizer = torch.optim.Adam(enc.proj.parameters(), lr=LR)

    def embed_batch_trainable(texts):
        """Like enc.encode() but keeps gradient (enc.encode is @torch.no_grad)."""
        batch = enc.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=256).to(device)
        with torch.no_grad():
            outputs = enc.model(**batch)
        cls_vec = outputs.last_hidden_state[:, 0]  # (B, H) -- backbone frozen, no grad needed here either
        z = enc._project(cls_vec)  # (B, 22), grad flows through proj only
        return z

    holdout_texts = [q for q, _ in holdout_rows]
    holdout_targets = np.stack([grpo_target(s) for _, s in holdout_rows])  # (N, 20)

    print("\nTraining projection head (holdout rho checked after every epoch)...", flush=True)
    best_rho, best_epoch = -1.0, -1
    for epoch in range(N_EPOCHS):
        random.shuffle(train_rows)
        total_loss, n_batches = 0.0, 0
        for i in range(0, len(train_rows), BATCH_SIZE):
            batch = train_rows[i:i + BATCH_SIZE]
            texts = [q for q, _ in batch]
            targets = np.stack([grpo_target(s) for _, s in batch])  # (B, 20)
            targets_t = torch.tensor(targets, dtype=torch.float32, device=device)

            q_emb = embed_batch_trainable(texts)  # (B, 22), L2-normalized
            cos_sim = q_emb @ domain_FP_t.T  # (B, 20) -- domain_FP rows already L2-normalized

            loss = F.mse_loss(cos_sim, targets_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        rho_arr = evaluate_holdout(enc, holdout_texts, holdout_targets, domain_FP)
        enc.model.eval()  # encode() sets eval then we go back to training loop next iter (backbone frozen anyway)
        print(f"  epoch {epoch+1}/{N_EPOCHS}  train MSE={total_loss/n_batches:.4f}  "
              f"holdout rho={rho_arr.mean():.4f} (std={rho_arr.std():.4f}, n={len(rho_arr)})", flush=True)
        if rho_arr.mean() > best_rho:
            best_rho, best_epoch = rho_arr.mean(), epoch + 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    enc.save(OUT_DIR)
    print(f"\nSaved final-epoch projection head -> {OUT_DIR}", flush=True)
    print(f"Best holdout rho was {best_rho:.4f} at epoch {best_epoch} "
          f"(saved checkpoint is from the LAST epoch, not necessarily the best -- "
          f"note this if final epoch shows overfitting/decline)", flush=True)


if __name__ == "__main__":
    main()
