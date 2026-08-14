"""Same as embedllm_newllm_grpo_train.py, but with a load-balancing auxiliary
loss added to the GRPO regression objective, to test whether it reduces the
collapse confirmed by embedllm_newllm_grpo_collapse_check.py (top3_share=0.853
vs chance 0.086, only 12/35 unseen models ever selected -- see PROGRESS.md
section 20 follow-up).

load_balance_loss is the same Switch-Transformer/GShard-style term used
earlier in this project (scripts/loo_unseen_recovery.py:324), reused as-is:
penalizes the batch-average softmax routing mass concentrating on a few
experts. Unlike its original contrastive-loss use case, the GRPO primary
loss here is plain MSE on raw cosine similarity (no temperature at all) --
so there's no "existing temperature" to match; BAL_TAU is a new, separately
tuned hyperparameter for this auxiliary term only.

Usage: python scripts/embedllm_newllm_grpo_train_balanced.py [--beta B] [--tau T]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset as TorchDataset

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from router.query_encoder import QueryEncoder
from embedllm_newllm_fast_eval import run_one as fast_eval

PCA5_DIR = Path("local_descriptors/embedllm-ceiling-pca5")
ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
HOLDOUT_FRAC = 0.15
SEED = 0
LOG_EVERY = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPLIT_PATH = ANALYSIS_DIR / "newllm_split.json"
UNSEEN_DIR = Path("local_descriptors/embedllm-ceiling-pca5-unseen-only")
CSCR_PAPER = {"audc": 0.4848, "peak": 0.565}


class QueryTargetDataset(TorchDataset):
    def __init__(self, df, seen_models):
        name_to_idx = {n: i for i, n in enumerate(seen_models)}
        self.items = []
        for pid, grp in df.groupby("prompt_id", sort=False):
            text = grp["prompt"].iloc[0]
            labels = np.full(len(seen_models), np.nan, dtype=np.float32)
            for m, v in zip(grp["model_name"], grp["label"]):
                if m in name_to_idx:
                    labels[name_to_idx[m]] = float(v)
            mask = ~np.isnan(labels)
            if mask.sum() < 2:
                continue
            vals = labels[mask]
            mean, std = vals.mean(), vals.std()
            target = np.zeros(len(seen_models), dtype=np.float32)
            target[mask] = (vals - mean) / (std + 1e-6)
            self.items.append((text, target, mask.astype(np.float32)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def load_balance_loss(q, E, tau):
    """Switch Transformer / GShard-style auxiliary loss (Shazeer et al. 2017).
    Minimized (=1) when batch-average routing mass is uniform across the M
    candidates; grows toward M when concentrated on one. See
    scripts/loo_unseen_recovery.py:324 for the original (contrastive-loss)
    use of this exact formula."""
    M = E.size(0)
    probs = torch.softmax((q @ E.T) / tau, dim=1)
    P = probs.mean(dim=0)
    return M * (P ** 2).sum()


def evaluate_holdout(enc, texts, targets, masks, E_seen_t, batch_size=64):
    enc.model.eval()
    embeds = np.zeros((len(texts), E_seen_t.size(1)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            embeds[start:start + len(batch)] = enc.encode(batch)
    cos = embeds @ E_seen_t.cpu().numpy().T
    rhos = []
    for i in range(len(texts)):
        m = masks[i].astype(bool)
        if m.sum() < 3:
            continue
        rho, _ = spearmanr(cos[i, m], targets[i, m])
        if not np.isnan(rho):
            rhos.append(rho)
    return np.array(rhos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=1.0, help="load-balance loss weight")
    ap.add_argument("--tau", type=float, default=0.05, help="temperature for the load-balance softmax")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    split = json.load(open(SPLIT_PATH, encoding="utf-8"))
    seen_models = split["seen"]
    print(f"seen={len(seen_models)} unseen={len(split['unseen'])}  beta={args.beta} tau={args.tau}", flush=True)

    print("Loading EmbedLLM train.csv...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "label", "prompt"])

    ds_all = QueryTargetDataset(df, seen_models)
    print(f"{len(ds_all)} usable training queries (>=2 seen models answered)", flush=True)

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(ds_all))
    n_holdout = int(len(ds_all) * HOLDOUT_FRAC)
    holdout_idx, train_idx = perm[:n_holdout], perm[n_holdout:]
    train_items = [ds_all.items[i] for i in train_idx]
    holdout_texts = [ds_all.items[i][0] for i in holdout_idx]
    holdout_targets = np.stack([ds_all.items[i][1] for i in holdout_idx])
    holdout_masks = np.stack([ds_all.items[i][2] for i in holdout_idx])
    print(f"train={len(train_items)}  holdout={len(holdout_texts)}", flush=True)

    print(f"Loading QueryEncoder (device={DEVICE})...", flush=True)
    enc = QueryEncoder(EMBED_MODEL, device=DEVICE, proj_dim=5, proj_multiplier=1)
    for p in enc.model.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(enc.proj.parameters(), lr=LR)

    E_seen = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in seen_models])
    E_seen_t = torch.from_numpy(E_seen).float().to(DEVICE)
    E_seen_t = E_seen_t / (E_seen_t.norm(dim=1, keepdim=True) + 1e-9)

    def collate(batch):
        texts, targets, masks = zip(*batch)
        toks = enc.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt", max_length=256)
        return toks, torch.tensor(np.stack(targets)), torch.tensor(np.stack(masks))

    loader = DataLoader(train_items, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    n_batches = len(loader)

    best_rho, best_epoch, best_state = -1.0, -1, None
    t0 = time.time()
    for ep in range(EPOCHS):
        enc.model.eval()
        ep_mse, ep_bal = 0.0, 0.0
        for bi, (tok, target, mask) in enumerate(loader):
            tok = {k: v.to(DEVICE) for k, v in tok.items()}
            target, mask = target.to(DEVICE), mask.to(DEVICE)
            with torch.no_grad():
                out = enc.model(**tok)
                cls_vec = out.last_hidden_state[:, 0]
            q = enc._project(cls_vec)  # (B, 5), L2-normalized
            cos_sim = q @ E_seen_t.T  # (B, n_seen)
            mse = ((cos_sim - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            bal = load_balance_loss(q, E_seen_t, args.tau)
            loss = mse + args.beta * bal
            loss.backward()
            opt.step()
            opt.zero_grad()
            ep_mse += mse.item()
            ep_bal += bal.item()
            if (bi + 1) % LOG_EVERY == 0 or (bi + 1) == n_batches:
                print(f"  epoch {ep+1}/{EPOCHS} batch {bi+1}/{n_batches} mse={mse.item():.4f} "
                      f"bal={bal.item():.4f} elapsed={time.time()-t0:.1f}s", flush=True)

        rho_arr = evaluate_holdout(enc, holdout_texts, holdout_targets, holdout_masks, E_seen_t)
        print(f"epoch {ep+1}/{EPOCHS} done  train_avg_mse={ep_mse/n_batches:.4f}  "
              f"train_avg_bal={ep_bal/n_batches:.4f}  "
              f"holdout_rho={rho_arr.mean():.4f} (std={rho_arr.std():.4f}, n={len(rho_arr)})", flush=True)
        if rho_arr.mean() > best_rho:
            best_rho, best_epoch = rho_arr.mean(), ep + 1
            best_state = {k: v.clone().cpu() for k, v in enc.proj.state_dict().items()}

    print(f"\nBest holdout rho = {best_rho:.4f} at epoch {best_epoch}", flush=True)
    enc.proj.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    ckpt_dir = Path(f"local_checkpoints/embedllm-newllm-encoder-grpo-balanced-seed0-beta{args.beta}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    enc.save(ckpt_dir)
    print(f"Saved best-epoch checkpoint -> {ckpt_dir}", flush=True)

    print("\nRunning fast AUDC eval on unseen-only pool with best checkpoint...", flush=True)
    r = fast_eval(str(SPLIT_PATH), str(UNSEEN_DIR), str(ckpt_dir), label=f"seed0-grpo-balanced-beta{args.beta}")
    r["best_epoch"] = best_epoch
    r["best_holdout_rho"] = float(best_rho)
    r["beta"] = args.beta
    r["tau"] = args.tau
    out_path = ANALYSIS_DIR / f"newllm_grpo_balanced_seed0_beta{args.beta}_results.json"
    json.dump(r, open(out_path, "w"), indent=2)

    print("\n" + "=" * 60)
    print(f"GRPO + LOAD-BALANCE SUMMARY (seed 0, beta={args.beta}, tau={args.tau})")
    print("=" * 60)
    print(f"best_epoch={best_epoch}  best_holdout_rho={best_rho:.4f}")
    print(f"knn: AUDC={r['knn']['audc']:.4f} Peak={r['knn']['peak']:.4f}  "
          f"({'BEATS' if r['knn']['audc'] > CSCR_PAPER['audc'] else 'below'} CSCR {CSCR_PAPER['audc']})")
    print(f"random: AUDC={r['random']['audc']:.4f} Peak={r['random']['peak']:.4f}")
    print("reference (no load-balance): seed0 GRPO AUDC=0.5289 Peak=0.5687 (best_epoch=3)")
    print(f"\nSaved -> {out_path}  checkpoint -> {ckpt_dir}")


if __name__ == "__main__":
    main()
