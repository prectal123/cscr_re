"""GRPO-style regression training for the EmbedLLM "new LLMs" query encoder,
as an alternative to the contrastive losses tried so far (multi_positive_info_nce,
cost_spectrum_info_nce) -- both of which showed a collapse-toward-generic-
capability failure mode that erodes Ceiling FP's fine domain-alignment edge.

Mirrors scripts/llmrouterbench/train_domain_encoder_trial.py's approach (first
tried on lite20 weeks ago): instead of a positive/negative contrastive target,
regress cosine_sim(query_encoding, domain_FP[model]) toward each SEEN model's
mean-centered relative advantage on that query:
    target[m] = (label[m] - mean_over_seen_models) / (std_over_seen_models + eps)
Mean/std computed using ONLY seen models (never leaks unseen-model info into
the target). Because the target is zero-mean per query, "always point toward
the generically-strongest direction" is not a viable low-loss shortcut the
way it is for a contrastive loss -- this is the structural fix being tested.

Unlike the earlier epoch=10 pilot (which only evaluated once at the end and
turned out to have overfit past epoch 2), this script checks holdout
Spearman rho after EVERY epoch and keeps the best checkpoint, matching the
original lite20 trial's protocol. AUDC/QNC/Peak (the real, expensive-to-
interpret metric) is only computed once, at the end, using that best
checkpoint.
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
LR = 1e-3  # matches train_domain_encoder_trial.py's original lite20 trial
HOLDOUT_FRAC = 0.15
SEED = 0
LOG_EVERY = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPLIT_PATH = ANALYSIS_DIR / "newllm_split.json"  # seed 0's seen/unseen model split (reused throughout)
UNSEEN_DIR = Path("local_descriptors/embedllm-ceiling-pca5-unseen-only")
CKPT_DIR = Path("local_checkpoints/embedllm-newllm-encoder-grpo-seed0")
CSCR_PAPER = {"audc": 0.4848, "peak": 0.565}


class QueryTargetDataset(TorchDataset):
    """One row per training prompt: (text, target_vec[n_seen]) where
    target_vec[m] = (label[m] - mean)/(std+eps) over SEEN models only."""

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
            if mask.sum() < 2:  # need at least 2 seen models to have a meaningful mean/std
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
    cos = embeds @ E_seen_t.cpu().numpy().T  # (N, n_seen)
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
        ep_loss = 0.0
        for bi, (tok, target, mask) in enumerate(loader):
            tok = {k: v.to(DEVICE) for k, v in tok.items()}
            target, mask = target.to(DEVICE), mask.to(DEVICE)
            with torch.no_grad():
                out = enc.model(**tok)
                cls_vec = out.last_hidden_state[:, 0]
            q = enc._project(cls_vec)  # (B, 5), L2-normalized
            cos_sim = q @ E_seen_t.T  # (B, n_seen)
            loss = ((cos_sim - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            opt.step()
            opt.zero_grad()
            ep_loss += loss.item()
            if (bi + 1) % LOG_EVERY == 0 or (bi + 1) == n_batches:
                print(f"  epoch {ep+1}/{EPOCHS} batch {bi+1}/{n_batches} loss={loss.item():.4f} "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)

        rho_arr = evaluate_holdout(enc, holdout_texts, holdout_targets, holdout_masks, E_seen_t)
        print(f"epoch {ep+1}/{EPOCHS} done  train_avg_loss={ep_loss/n_batches:.4f}  "
              f"holdout_rho={rho_arr.mean():.4f} (std={rho_arr.std():.4f}, n={len(rho_arr)})", flush=True)
        if rho_arr.mean() > best_rho:
            best_rho, best_epoch = rho_arr.mean(), ep + 1
            best_state = {k: v.clone().cpu() for k, v in enc.proj.state_dict().items()}

    print(f"\nBest holdout rho = {best_rho:.4f} at epoch {best_epoch}", flush=True)
    enc.proj.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    enc.save(CKPT_DIR)
    print(f"Saved best-epoch checkpoint -> {CKPT_DIR}", flush=True)

    print("\nRunning fast AUDC eval on unseen-only pool with best checkpoint...", flush=True)
    r = fast_eval(str(SPLIT_PATH), str(UNSEEN_DIR), str(CKPT_DIR), label="seed0-grpo")
    r["best_epoch"] = best_epoch
    r["best_holdout_rho"] = float(best_rho)
    out_path = ANALYSIS_DIR / "newllm_grpo_seed0_results.json"
    json.dump(r, open(out_path, "w"), indent=2)

    print("\n" + "=" * 60)
    print("GRPO REGRESSION SUMMARY (seed 0)")
    print("=" * 60)
    print(f"best_epoch={best_epoch}  best_holdout_rho={best_rho:.4f}")
    print(f"knn: AUDC={r['knn']['audc']:.4f} Peak={r['knn']['peak']:.4f}  "
          f"({'BEATS' if r['knn']['audc'] > CSCR_PAPER['audc'] else 'below'} CSCR {CSCR_PAPER['audc']})")
    print(f"random: AUDC={r['random']['audc']:.4f} Peak={r['random']['peak']:.4f}")
    print("reference: epoch=2 contrastive seed0 AUDC=0.4731 Peak=0.536; "
          "epoch=10 contrastive seed0 AUDC=0.4644 Peak=0.497")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
