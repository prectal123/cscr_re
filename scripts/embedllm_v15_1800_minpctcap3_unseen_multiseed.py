"""Unseen-protocol version of embedllm_minpctcap3_allseen_multiseed.py's
min(pct=0.3, K_CAP=3) loss -- the winning generalization of min-pos (K=1)
found on all-seen (0.5867 vs K=1's 0.5652). Same recipe as
embedllm_pct30_unseen_multiseed.py (which established the pct=0.3 unseen
reference, mean=0.5162, 3 seeds) but with minpos_loss swapped for
minpctcap_loss, plus embedding caching (that script recomputed frozen
MiniLM embeddings every epoch -- same fixable slowness pattern as
RouterBench's original script, fixed here).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from router.query_encoder import QueryEncoder
from embedllm_newllm_fast_eval import run_one as fast_eval_unseen

PCA5_DIR = Path("local_descriptors/embedllm-ceiling-pcaweighted-pca5")  # true V1.5: 1800-probe, floor=15
ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
HOLDOUT_FRAC = 0.15
SEEDS = [3, 4, 5, 6, 7, 8, 9]  # 0,1,2 already done -- extending to 10 total seeds
PCT_CATFILTER = 0.3
PCT_MINLOSS = 0.3
K_CAP = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSCR_UNSEEN = 0.4848

REFERENCE = {"combined_top50pct_seed0to2": 0.5163, "combined_pct30_seed0to2": 0.5162}


def compute_keep_idx_pct(pos_idx, scores, pct):
    if len(pos_idx) <= 1:
        return pos_idx
    order = np.argsort(-scores)
    sorted_idx = pos_idx[order]
    n_keep = max(1, int(np.ceil(len(pos_idx) * pct)))
    return sorted_idx[:n_keep]


def build_raw_category_accuracy(df, models, categories):
    pivot = df.pivot_table(index="model_name", columns="category", values="label", aggfunc="mean")
    pivot = pivot.reindex(index=models, columns=categories)
    return pivot.to_numpy()


def build_items(df, models, raw_cat_acc, category_to_idx, pct):
    name_to_idx = {n: i for i, n in enumerate(models)}
    texts, targets, masks = [], [], []
    for pid, grp in df.groupby("prompt_id", sort=False):
        text = grp["prompt"].iloc[0]
        category = grp["category"].iloc[0]
        cat_idx = category_to_idx.get(category)
        labels = np.full(len(models), np.nan, dtype=np.float32)
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
        target = np.zeros(len(models), dtype=np.float32)
        target[mask] = (vals - mean) / (std + 1e-6)

        keep_mask = mask.copy()
        pos_idx = np.where(mask & (labels == 1))[0]
        if cat_idx is not None and len(pos_idx) > 1:
            scores = np.nan_to_num(raw_cat_acc[pos_idx, cat_idx], nan=-1.0)
            keep_pos = compute_keep_idx_pct(pos_idx, scores, pct)
            demoted = np.setdiff1d(pos_idx, keep_pos)
            keep_mask[demoted] = False

        texts.append(text)
        targets.append(target)
        masks.append(keep_mask.astype(np.float32))
    return texts, np.stack(targets), np.stack(masks)


def minpctcap_loss(cos_sim, target, mask, pct=PCT_MINLOSS, k_cap=K_CAP):
    pos_mask = (target > 0) & (mask > 0.5)
    neg_mask = (target <= 0) & (mask > 0.5)
    sq_err = (cos_sim - target) ** 2

    pos_err = sq_err.masked_fill(~pos_mask, float("inf"))
    n_pos = pos_mask.sum(dim=1)
    has_pos = n_pos > 0
    k_eff = torch.clamp(torch.ceil(n_pos.float() * pct).long(), min=1, max=k_cap)
    k_eff = torch.minimum(k_eff, n_pos.clamp(min=1))
    sorted_err, _ = pos_err.sort(dim=1)
    idx = torch.arange(pos_err.size(1), device=pos_err.device).unsqueeze(0)
    take_mask = (idx < k_eff.unsqueeze(1)) & has_pos.unsqueeze(1)
    finite_sorted = torch.where(torch.isfinite(sorted_err), sorted_err, torch.zeros_like(sorted_err))
    pos_sum = (finite_sorted * take_mask.float()).sum(dim=1)
    pos_topk_mean = pos_sum / k_eff.clamp(min=1).float()
    loss_pos = (pos_topk_mean * has_pos.float()).sum() / has_pos.float().sum().clamp(min=1)

    neg_err = (sq_err * neg_mask.float()).sum(dim=1)
    neg_count = neg_mask.float().sum(dim=1).clamp(min=1)
    loss_neg = (neg_err / neg_count).mean()
    return loss_pos + loss_neg


def precompute_cls(texts, tokenizer, base_model, batch_size=64):
    embeds = np.zeros((len(texts), base_model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            toks = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256)
            toks = {k: v.to(DEVICE) for k, v in toks.items()}
            out = base_model(**toks)
            embeds[start:start + len(batch)] = out.last_hidden_state[:, 0].cpu().numpy()
    return embeds


def train_fast(seed, cls_all, targets, masks, E_t, hidden_size):
    n = cls_all.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_ho = int(n * HOLDOUT_FRAC)
    ho_idx, tr_idx = perm[:n_ho], perm[n_ho:]

    cls_tr = torch.from_numpy(cls_all[tr_idx]).float().to(DEVICE)
    tgt_tr = torch.from_numpy(targets[tr_idx]).float().to(DEVICE)
    msk_tr = torch.from_numpy(masks[tr_idx]).float().to(DEVICE)
    cls_ho = torch.from_numpy(cls_all[ho_idx]).float().to(DEVICE)
    tgt_ho = targets[ho_idx]
    msk_ho = masks[ho_idx]

    torch.manual_seed(seed)
    proj = nn.Sequential(
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.ReLU(),
        nn.Linear(hidden_size, E_t.size(1), bias=False),
    ).to(DEVICE)
    opt = torch.optim.Adam(proj.parameters(), lr=LR)
    n_train = cls_tr.size(0)

    best_rho, best_epoch, best_state = -1.0, -1, None
    t0 = time.time()
    for ep in range(EPOCHS):
        order = torch.randperm(n_train, device=DEVICE)
        for start in range(0, n_train, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            q = F.normalize(proj(cls_tr[idx]), dim=-1)
            cos_sim = q @ E_t.T
            loss = minpctcap_loss(cos_sim, tgt_tr[idx], msk_tr[idx])
            loss.backward()
            opt.step()
            opt.zero_grad()

        with torch.no_grad():
            q_ho = F.normalize(proj(cls_ho), dim=-1)
            cos_ho = (q_ho @ E_t.T).cpu().numpy()
        rhos = []
        for i in range(cos_ho.shape[0]):
            m = msk_ho[i].astype(bool)
            if m.sum() < 3:
                continue
            rho, _ = spearmanr(cos_ho[i, m], tgt_ho[i, m])
            if not np.isnan(rho):
                rhos.append(rho)
        rho_mean = np.mean(rhos) if rhos else -1.0
        if rho_mean > best_rho:
            best_rho, best_epoch = rho_mean, ep + 1
            best_state = {k: v.clone() for k, v in proj.state_dict().items()}
    proj.load_state_dict(best_state)
    print(f"  [seed={seed}] best_epoch={best_epoch} best_holdout_rho={best_rho:.4f} time={time.time()-t0:.1f}s",
          flush=True)
    return proj


def main():
    print("Loading EmbedLLM train.csv...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "category", "label", "prompt"])
    categories = sorted(df["category"].unique().tolist())
    category_to_idx = {c: i for i, c in enumerate(categories)}

    print(f"Loading frozen MiniLM base (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    hidden_size = base_model.config.hidden_size

    results = []
    for seed in SEEDS:
        split_path = ANALYSIS_DIR / ("newllm_split.json" if seed == 0 else f"newllm_split_seed{seed}.json")
        split = json.load(open(split_path, encoding="utf-8"))
        seen_models = split["seen"]
        # V1.5 has no separate unseen-only subset dirs -- run_one only reads the "unseen"
        # models' names from split.json out of whatever dir it's given, so pointing it at
        # the full V1.5 dir (all 112 models) works directly, same effect as a pre-filtered subset
        unseen_dir = PCA5_DIR

        E_seen = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in seen_models])
        E_seen_t = torch.from_numpy(E_seen).float().to(DEVICE)
        E_seen_t = E_seen_t / (E_seen_t.norm(dim=1, keepdim=True) + 1e-9)
        raw_cat_acc = build_raw_category_accuracy(df, seen_models, categories)
        texts, targets, masks = build_items(df, seen_models, raw_cat_acc, category_to_idx, PCT_CATFILTER)

        print(f"\n{'='*60}\nSEED {seed}: seen={len(seen_models)}, {len(texts)} usable rows\n{'='*60}", flush=True)
        t0 = time.time()
        cls_all = precompute_cls(texts, tokenizer, base_model)
        print(f"  cached embeddings in {time.time()-t0:.1f}s -> {cls_all.shape}", flush=True)

        proj = train_fast(seed, cls_all, targets, masks, E_seen_t, hidden_size)

        enc = QueryEncoder.__new__(QueryEncoder)
        torch.nn.Module.__init__(enc)
        enc.tokenizer = tokenizer
        enc.model = base_model
        enc.device = DEVICE
        enc.hidden_size = hidden_size
        enc.proj_dim = E_seen_t.size(1)
        enc.proj = proj
        enc.model.config.proj_dim = E_seen_t.size(1)

        ckpt_dir = Path(f"local_checkpoints/embedllm-newllm-encoder-minpctcap3-v15-1800-seed{seed}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        enc.save(ckpt_dir)
        r = fast_eval_unseen(str(split_path), str(unseen_dir), str(ckpt_dir), label=f"minpctcap3-unseen-seed{seed}")
        results.append({"seed": seed, **r["knn"]})
        print(f"  [seed={seed}] AUDC={r['knn']['audc']:.4f} Peak={r['knn']['peak']:.4f} "
              f"({'BEATS' if r['knn']['audc'] > CSCR_UNSEEN else 'below'} CSCR {CSCR_UNSEEN})", flush=True)

    out_path = ANALYSIS_DIR / "unseen_minpctcap3_multiseed_results_v15_1800probes_seeds3to9.json"
    json.dump(results, open(out_path, "w"), indent=2)

    audcs = [r["audc"] for r in results]
    print("\n" + "=" * 90)
    print(f"MIN(PCT=0.3,CAP=3) + CATFILTER pct=0.3, EmbedLLM unseen, {len(SEEDS)} seeds")
    print("=" * 90)
    for r in results:
        print(f"  seed={r['seed']}: AUDC={r['audc']:.4f} Peak={r['peak']:.4f}")
    print(f"\nmean AUDC={np.mean(audcs):.4f} (std={np.std(audcs):.4f})  beats CSCR: {sum(1 for a in audcs if a > CSCR_UNSEEN)}/{len(audcs)}")
    print(f"reference: combined-pct30-K1(seed0-2 mean)={REFERENCE['combined_pct30_seed0to2']}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
