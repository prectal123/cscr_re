"""GRPO regression variant that fixes outlier-drag by NOT averaging over all
positive-advantage (t_m > 0) models -- only requiring the query to match the
CLOSEST one (min-over-positives), while keeping full mean-MSE over the
negative-advantage models (there's no "multiple valid answer" ambiguity on
the negative side -- q should be far from every wrong model, not just the
nearest one).

    loss_pos = min_{m: t_m>0} (cos_sim(q,E_m) - t_m)^2   (per example, if any positive)
    loss_neg = mean_{m: t_m<=0} (cos_sim(q,E_m) - t_m)^2
    loss = loss_pos + loss_neg

Known risk (flagged before running, not discovered after): "winner take all"
losses like this can be unstable -- which positive is "closest" can flip
between competing candidates across training steps, causing the target to
effectively move around under the model (see Multiple Choice Learning
literature, e.g. Guzman-Rivera et al. 2012, stochastic MCL follow-ups on
exactly this instability). This run's epoch-by-epoch holdout_rho trace is
the direct empirical check for that -- compare its volatility against the
plain-GRPO reference (best_epoch bounced around 3/6/7/8 across seeds there
too, so some volatility is already normal for this setup).
"""
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
CKPT_DIR = Path("local_checkpoints/embedllm-newllm-encoder-grpo-minpos-seed0")
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
            if std < 1e-6:
                continue
            target = np.zeros(len(seen_models), dtype=np.float32)
            target[mask] = (vals - mean) / (std + 1e-6)
            self.items.append((text, target, mask.astype(np.float32)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def minpos_loss(cos_sim, target, mask):
    """cos_sim, target, mask: (B, M). Returns scalar loss."""
    pos_mask = (target > 0) & (mask > 0.5)
    neg_mask = (target <= 0) & (mask > 0.5)

    sq_err = (cos_sim - target) ** 2

    # positive side: min over positive-advantage models per example
    pos_err = sq_err.masked_fill(~pos_mask, float("inf"))
    has_pos = pos_mask.any(dim=1)
    pos_min = pos_err.min(dim=1).values
    pos_min = torch.where(has_pos, pos_min, torch.zeros_like(pos_min))
    loss_pos = (pos_min * has_pos.float()).sum() / has_pos.float().sum().clamp(min=1)

    # negative side: mean over all negative-advantage models, unchanged
    neg_err = (sq_err * neg_mask.float()).sum(dim=1)
    neg_count = neg_mask.float().sum(dim=1).clamp(min=1)
    loss_neg = (neg_err / neg_count).mean()

    return loss_pos + loss_neg


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
    torch.manual_seed(SEED)
    split = json.load(open(SPLIT_PATH, encoding="utf-8"))
    seen_models = split["seen"]
    print(f"seen={len(seen_models)} unseen={len(split['unseen'])}", flush=True)

    print("Loading EmbedLLM train.csv...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "label", "prompt"])

    ds_all = QueryTargetDataset(df, seen_models)
    print(f"{len(ds_all)} usable training queries", flush=True)

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
    rho_trace = []
    t0 = time.time()
    for ep in range(EPOCHS):
        enc.model.eval()
        ep_loss = 0.0
        for bi, (tok, target, mask) in enumerate(loader):
            tok = {k: v.to(DEVICE) for k, v in tok.items()}
            target, mask = target.to(DEVICE), mask.to(DEVICE)
            with torch.no_grad():
                out = enc.model(**tok)
                cls_vec = out.last_hidden_state[:, 0]
            q = enc._project(cls_vec)
            cos_sim = q @ E_seen_t.T
            loss = minpos_loss(cos_sim, target, mask)
            loss.backward()
            opt.step()
            opt.zero_grad()
            ep_loss += loss.item()
            if (bi + 1) % LOG_EVERY == 0 or (bi + 1) == n_batches:
                print(f"  epoch {ep+1}/{EPOCHS} batch {bi+1}/{n_batches} loss={loss.item():.4f} "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)

        rho_arr = evaluate_holdout(enc, holdout_texts, holdout_targets, holdout_masks, E_seen_t)
        rho_trace.append(float(rho_arr.mean()))
        print(f"epoch {ep+1}/{EPOCHS} done  train_avg_loss={ep_loss/n_batches:.4f}  "
              f"holdout_rho={rho_arr.mean():.4f} (std={rho_arr.std():.4f}, n={len(rho_arr)})", flush=True)
        if rho_arr.mean() > best_rho:
            best_rho, best_epoch = rho_arr.mean(), ep + 1
            best_state = {k: v.clone().cpu() for k, v in enc.proj.state_dict().items()}

    print(f"\nBest holdout rho = {best_rho:.4f} at epoch {best_epoch}", flush=True)
    print(f"Full holdout_rho trace by epoch: {[f'{r:.4f}' for r in rho_trace]}", flush=True)
    rho_diffs = np.diff(rho_trace)
    print(f"Epoch-to-epoch |delta rho|: mean={np.abs(rho_diffs).mean():.4f} max={np.abs(rho_diffs).max():.4f} "
          f"(reference plain-GRPO trace, for comparison: 0.015,0.192,-0.006,0.010,... roughly settles then wobbles)",
          flush=True)

    enc.proj.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    enc.save(CKPT_DIR)
    print(f"Saved best-epoch checkpoint -> {CKPT_DIR}", flush=True)

    print("\nRunning fast AUDC eval on unseen-only pool with best checkpoint...", flush=True)
    r = fast_eval(str(SPLIT_PATH), str(UNSEEN_DIR), str(CKPT_DIR), label="seed0-grpo-minpos")
    r["best_epoch"] = best_epoch
    r["best_holdout_rho"] = float(best_rho)
    r["rho_trace"] = rho_trace
    out_path = ANALYSIS_DIR / "newllm_grpo_minpos_seed0_results.json"
    json.dump(r, open(out_path, "w"), indent=2)

    print("\n" + "=" * 60)
    print("GRPO MIN-OVER-POSITIVES SUMMARY (seed 0)")
    print("=" * 60)
    print(f"best_epoch={best_epoch}  best_holdout_rho={best_rho:.4f}")
    print(f"knn: AUDC={r['knn']['audc']:.4f} Peak={r['knn']['peak']:.4f}  "
          f"({'BEATS' if r['knn']['audc'] > CSCR_PAPER['audc'] else 'below'} CSCR {CSCR_PAPER['audc']})")
    print(f"random: AUDC={r['random']['audc']:.4f} Peak={r['random']['peak']:.4f}")
    print("reference (plain GRPO, mean-over-positives): seed0 AUDC=0.5289 Peak=0.5687 (best_epoch=3)")
    print(f"\nSaved -> {out_path}  checkpoint -> {CKPT_DIR}")


if __name__ == "__main__":
    main()
