"""Multi-seed (1-4) version of embedllm_newllm_grpo_train.py -- seed 0 already
done (best_epoch=3, holdout_rho=0.1938, AUDC=0.5269 Peak=0.5673, beats CSCR
0.4848/0.565). Reuses the same seen/unseen splits as every other run this
session, and the same per-epoch holdout-rho best-checkpoint selection (avoids
the epoch=10-contrastive mistake of only checking at the very end).

MiniLM backbone + EmbedLLM train.csv loaded ONCE and reused across seeds.
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
LOG_EVERY = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEEDS = [1, 2, 3, 4]
CSCR_PAPER = {"audc": 0.4848, "peak": 0.565}
SEED0_RESULT = {"seed": 0, "best_epoch": 3, "best_holdout_rho": 0.1938,
                 "knn": {"audc": 0.5269, "qnc": 0.961, "peak": 0.5673},
                 "random": {"audc": 0.3787, "peak": 0.4160}}


def split_path_for(seed):
    return ANALYSIS_DIR / ("newllm_split.json" if seed == 0 else f"newllm_split_seed{seed}.json")


def unseen_dir_for(seed):
    return Path("local_descriptors/embedllm-ceiling-pca5-unseen-only" if seed == 0
                else f"local_descriptors/embedllm-ceiling-pca5-unseen-only-seed{seed}")


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


def train_one_seed(seed, seen_models, df, tokenizer, base_model):
    torch.manual_seed(seed)
    ds_all = QueryTargetDataset(df, seen_models)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ds_all))
    n_holdout = int(len(ds_all) * HOLDOUT_FRAC)
    holdout_idx, train_idx = perm[:n_holdout], perm[n_holdout:]
    train_items = [ds_all.items[i] for i in train_idx]
    holdout_texts = [ds_all.items[i][0] for i in holdout_idx]
    holdout_targets = np.stack([ds_all.items[i][1] for i in holdout_idx])
    holdout_masks = np.stack([ds_all.items[i][2] for i in holdout_idx])
    print(f"  [seed {seed}] train={len(train_items)} holdout={len(holdout_texts)}", flush=True)

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
    opt = torch.optim.Adam(enc.proj.parameters(), lr=LR)

    E_seen = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in seen_models])
    E_seen_t = torch.from_numpy(E_seen).float().to(DEVICE)
    E_seen_t = E_seen_t / (E_seen_t.norm(dim=1, keepdim=True) + 1e-9)

    def collate(batch):
        texts, targets, masks = zip(*batch)
        toks = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt", max_length=256)
        return toks, torch.tensor(np.stack(targets)), torch.tensor(np.stack(masks))

    loader = DataLoader(train_items, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    n_batches = len(loader)

    best_rho, best_epoch, best_state = -1.0, -1, None
    t0 = time.time()
    for ep in range(EPOCHS):
        base_model.eval()
        ep_loss = 0.0
        for bi, (tok, target, mask) in enumerate(loader):
            tok = {k: v.to(DEVICE) for k, v in tok.items()}
            target, mask = target.to(DEVICE), mask.to(DEVICE)
            with torch.no_grad():
                out = base_model(**tok)
                cls_vec = out.last_hidden_state[:, 0]
            q = enc._project(cls_vec)
            cos_sim = q @ E_seen_t.T
            loss = ((cos_sim - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            opt.step()
            opt.zero_grad()
            ep_loss += loss.item()
            if (bi + 1) % LOG_EVERY == 0 or (bi + 1) == n_batches:
                print(f"  [seed {seed}] epoch {ep+1}/{EPOCHS} batch {bi+1}/{n_batches} "
                      f"loss={loss.item():.4f} elapsed={time.time()-t0:.1f}s", flush=True)

        rho_arr = evaluate_holdout(enc, holdout_texts, holdout_targets, holdout_masks, E_seen_t)
        print(f"  [seed {seed}] epoch {ep+1}/{EPOCHS} done  train_avg_loss={ep_loss/n_batches:.4f}  "
              f"holdout_rho={rho_arr.mean():.4f} (n={len(rho_arr)})", flush=True)
        if rho_arr.mean() > best_rho:
            best_rho, best_epoch = rho_arr.mean(), ep + 1
            best_state = {k: v.clone().cpu() for k, v in enc.proj.state_dict().items()}

    enc.proj.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return enc, best_epoch, best_rho


def main():
    print("Loading EmbedLLM train.csv...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "label", "prompt"])

    print(f"Loading frozen MiniLM base ONCE (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    results = [SEED0_RESULT]
    for seed in SEEDS:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)
        split = json.load(open(split_path_for(seed), encoding="utf-8"))
        seen_models = split["seen"]

        ckpt_dir = Path(f"local_checkpoints/embedllm-newllm-encoder-grpo-seed{seed}")
        if (ckpt_dir / "config.json").exists() and (ckpt_dir / "proj.pt").exists():
            print(f"  [seed {seed}] checkpoint already exists, skipping retrain -> {ckpt_dir}", flush=True)
            best_epoch, best_rho = None, None
        else:
            enc, best_epoch, best_rho = train_one_seed(seed, seen_models, df, tokenizer, base_model)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            enc.save(ckpt_dir)
            print(f"  [seed {seed}] best_epoch={best_epoch} best_holdout_rho={best_rho:.4f} saved -> {ckpt_dir}",
                  flush=True)

        r = fast_eval(str(split_path_for(seed)), str(unseen_dir_for(seed)), str(ckpt_dir), label=f"seed{seed}-grpo")
        r["seed"] = seed
        r["best_epoch"] = best_epoch
        r["best_holdout_rho"] = best_rho
        print(f"  [seed {seed}] RESULT: AUDC={r['knn']['audc']:.4f} Peak={r['knn']['peak']:.4f}  "
              f"({'BEATS' if r['knn']['audc'] > CSCR_PAPER['audc'] else 'below'} CSCR)", flush=True)
        results.append(r)

    out_path = ANALYSIS_DIR / "newllm_grpo_multiseed_results.json"
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n" + "=" * 70)
    print("GRPO REGRESSION MULTI-SEED SUMMARY (5 seeds)")
    print("=" * 70)
    audcs = [r["knn"]["audc"] for r in results]
    peaks = [r["knn"]["peak"] for r in results]
    for r in results:
        beats = "BEATS" if r["knn"]["audc"] > CSCR_PAPER["audc"] else "below"
        print(f"seed={r['seed']}: AUDC={r['knn']['audc']:.4f} ({beats} CSCR 0.4848)  Peak={r['knn']['peak']:.4f}  "
              f"best_epoch={r.get('best_epoch')}")
    print(f"\nmean AUDC={np.mean(audcs):.4f} (std={np.std(audcs):.4f}, min={min(audcs):.4f}, max={max(audcs):.4f})")
    print(f"mean Peak={np.mean(peaks):.4f} (std={np.std(peaks):.4f})")
    print(f"seeds beating CSCR AUDC: {sum(1 for a in audcs if a > CSCR_PAPER['audc'])}/{len(audcs)}")
    print("(reference: contrastive 2-epoch mean AUDC=0.468 std=0.029, 1/5 beat CSCR)")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
