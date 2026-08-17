"""Experimental, exploratory (user's own framing: "just try it once", a
bonus on top of the already-sufficient combined recipe) -- lets catfilter-
surviving positive models' FPs drift a tiny bit toward the queries that
land near them, in addition to the usual query-encoder gradient update.

Safety design (per the risk discussed: seen-model FP drift could break the
unseen protocol's coordinate-frame consistency with untouched unseen-model
FPs, and mutual query<->FP attraction risks the collapse dynamics already
seen with pure contrastive losses in this project):
  - step size is INVERSE to distance: close (already well-matched) queries
    nudge their model boldly, far/uncertain queries nudge only slightly --
    never let an ambiguous match yank a model's FP hard.
  - FP updates are a separate, tiny, non-backprop rule (detached from the
    MLP's gradient optimizer), applied only to catfilter(pct=0.3)-surviving
    positives -- never to negatives, never with a magnitude near the
    query-encoder's own learning rate.
  - Renormalized to unit norm after every update, matching the existing
    L2-normalized FP convention.
  - Tested on the UNSEEN protocol first (highest risk to the project's
    core target metric) with a single seed, before considering all-seen or
    multi-seed.
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

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from router.query_encoder import QueryEncoder
from embedllm_newllm_fast_eval import run_one as fast_eval_unseen

PCA5_DIR = Path("local_descriptors/embedllm-ceiling-pca5")
ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
HOLDOUT_FRAC = 0.15
SEED = 0
PCT = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSCR_UNSEEN = 0.4848

FP_NUDGE_ALPHA = 0.02   # max per-step nudge magnitude (tiny, "깔짝깔짝")
FP_NUDGE_TAU = 1.0      # distance decay scale: step = alpha * exp(-dist/tau)

REFERENCE_COMBINED_PCT30_SEED0 = 0.5264


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
    items = []
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

        items.append((text, target, keep_mask.astype(np.float32)))
    return items


def minpos_loss(cos_sim, target, mask):
    pos_mask = (target > 0) & (mask > 0.5)
    neg_mask = (target <= 0) & (mask > 0.5)
    sq_err = (cos_sim - target) ** 2
    pos_err = sq_err.masked_fill(~pos_mask, float("inf"))
    has_pos = pos_mask.any(dim=1)
    pos_min = pos_err.min(dim=1).values
    pos_min = torch.where(has_pos, pos_min, torch.zeros_like(pos_min))
    loss_pos = (pos_min * has_pos.float()).sum() / has_pos.float().sum().clamp(min=1)
    neg_err = (sq_err * neg_mask.float()).sum(dim=1)
    neg_count = neg_mask.float().sum(dim=1).clamp(min=1)
    loss_neg = (neg_err / neg_count).mean()
    return loss_pos + loss_neg


def nudge_fp(E_t, q, pos_mask):
    """Non-backprop FP refinement: catfilter-surviving positives drift a
    tiny bit toward queries that land close to them, barely at all toward
    queries that land far. Vectorized over the batch."""
    with torch.no_grad():
        cos_sim = q @ E_t.T                      # (B, n_models)
        dist = 1.0 - cos_sim                      # (B, n_models), in ~[0,2]
        step = FP_NUDGE_ALPHA * torch.exp(-dist / FP_NUDGE_TAU)
        weight = step * pos_mask.float()           # (B, n_models), 0 for non-positives
        weight_sum = weight.sum(dim=0)              # (n_models,)
        weighted_q_sum = weight.T @ q                # (n_models, dim)
        delta = weighted_q_sum - weight_sum.unsqueeze(1) * E_t
        E_t_new = E_t + delta
        E_t_new = E_t_new / (E_t_new.norm(dim=1, keepdim=True) + 1e-9)
    return E_t_new


def evaluate_holdout(enc, texts, targets, masks, E_t, batch_size=64):
    enc.model.eval()
    embeds = np.zeros((len(texts), E_t.size(1)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            embeds[start:start + len(batch)] = enc.encode(batch)
    cos = embeds @ E_t.cpu().numpy().T
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
    print("Loading EmbedLLM train.csv...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "category", "label", "prompt"])
    categories = sorted(df["category"].unique().tolist())
    category_to_idx = {c: i for i, c in enumerate(categories)}

    split_path = ANALYSIS_DIR / "newllm_split.json"
    split = json.load(open(split_path, encoding="utf-8"))
    seen_models = split["seen"]
    unseen_dir = Path("local_descriptors/embedllm-ceiling-pca5-unseen-only")

    print(f"Loading frozen MiniLM base (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    E_seen = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in seen_models])
    E_t = torch.from_numpy(E_seen).float().to(DEVICE)
    E_t = E_t / (E_t.norm(dim=1, keepdim=True) + 1e-9)
    E_original = E_t.clone()
    raw_cat_acc = build_raw_category_accuracy(df, seen_models, categories)
    items = build_items(df, seen_models, raw_cat_acc, category_to_idx, PCT)
    print(f"{len(items)} usable training queries, seen models={len(seen_models)}", flush=True)

    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(items))
    n_holdout = int(len(items) * HOLDOUT_FRAC)
    holdout_idx, train_idx = perm[:n_holdout], perm[n_holdout:]
    train_sub = [items[i] for i in train_idx]
    holdout_texts = [items[i][0] for i in holdout_idx]
    holdout_targets = np.stack([items[i][1] for i in holdout_idx])
    holdout_masks = np.stack([items[i][2] for i in holdout_idx])

    enc = QueryEncoder.__new__(QueryEncoder)
    torch.nn.Module.__init__(enc)
    enc.tokenizer = tokenizer
    enc.model = base_model
    enc.device = DEVICE
    enc.hidden_size = base_model.config.hidden_size
    enc.proj_dim = E_t.size(1)
    enc.proj = torch.nn.Sequential(
        torch.nn.Linear(enc.hidden_size, enc.hidden_size, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(enc.hidden_size, E_t.size(1), bias=False),
    ).to(DEVICE)
    enc.model.config.proj_dim = E_t.size(1)
    opt = torch.optim.Adam(enc.proj.parameters(), lr=LR)

    def collate(batch):
        texts, targets, masks = zip(*batch)
        toks = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt", max_length=256)
        return toks, torch.tensor(np.stack(targets)), torch.tensor(np.stack(masks))

    loader = torch.utils.data.DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    best_rho, best_epoch, best_state, best_E = -1.0, -1, None, None
    total_fp_drift = torch.zeros(E_t.size(0), device=DEVICE)
    t0 = time.time()
    for ep in range(EPOCHS):
        base_model.eval()
        for tok, target, mask in loader:
            tok = {k: v.to(DEVICE) for k, v in tok.items()}
            target, mask = target.to(DEVICE), mask.to(DEVICE)
            with torch.no_grad():
                out = base_model(**tok)
                cls_vec = out.last_hidden_state[:, 0]
            q = enc._project(cls_vec)
            cos_sim = q @ E_t.T
            loss = minpos_loss(cos_sim, target, mask)
            loss.backward()
            opt.step()
            opt.zero_grad()

            pos_mask = (target > 0) & (mask > 0.5)
            E_t_new = nudge_fp(E_t, q.detach(), pos_mask)
            total_fp_drift += (E_t_new - E_t).norm(dim=1)
            E_t = E_t_new

        rho_arr = evaluate_holdout(enc, holdout_texts, holdout_targets, holdout_masks, E_t)
        print(f"epoch {ep+1}/{EPOCHS}: holdout_rho={rho_arr.mean():.4f}  "
              f"mean_fp_drift_so_far={total_fp_drift.mean().item():.4f} "
              f"max_fp_drift={total_fp_drift.max().item():.4f}", flush=True)
        if rho_arr.mean() > best_rho:
            best_rho, best_epoch = rho_arr.mean(), ep + 1
            best_state = {k: v.clone().cpu() for k, v in enc.proj.state_dict().items()}
            best_E = E_t.clone()

    net_drift = (best_E - E_original).norm(dim=1)
    print(f"\nbest_epoch={best_epoch} best_holdout_rho={best_rho:.4f} time={time.time()-t0:.1f}s", flush=True)
    print(f"NET drift from original oracle FP position (best_epoch snapshot): "
          f"mean={net_drift.mean().item():.4f} max={net_drift.max().item():.4f} "
          f"(FP vectors are unit-norm, so max possible drift is 2.0)", flush=True)
    enc.proj.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    # save best_E as the unseen-eval FP source: seen models' FPs now drifted,
    # but fast_eval_unseen only needs the unseen-only dir (untouched, oracle
    # positions) -- consistent with the protocol (train encoder against seen
    # FPs, evaluate candidate-selection against unseen FPs that were never
    # touched by any gradient or nudge).
    ckpt_dir = Path("local_checkpoints/embedllm-newllm-encoder-fprefine-seed0")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    enc.save(ckpt_dir)

    r = fast_eval_unseen(str(split_path), str(unseen_dir), str(ckpt_dir), label="fprefine-unseen-seed0")
    print(f"\n[fp-refine unseen seed0] AUDC={r['knn']['audc']:.4f} Peak={r['knn']['peak']:.4f} "
          f"({'BEATS' if r['knn']['audc'] > CSCR_UNSEEN else 'below'} CSCR {CSCR_UNSEEN})", flush=True)
    print(f"reference: combined(pct=0.3), no FP-refine, same seed0 = {REFERENCE_COMBINED_PCT30_SEED0}")

    out_path = ANALYSIS_DIR / "fp_refine_unseen_seed0_results.json"
    json.dump({"knn": r["knn"], "best_epoch": best_epoch, "best_holdout_rho": float(best_rho),
               "cumulative_fp_step_norm_mean": float(total_fp_drift.mean().item()),
               "cumulative_fp_step_norm_max": float(total_fp_drift.max().item()),
               "net_drift_from_oracle_mean": float(net_drift.mean().item()),
               "net_drift_from_oracle_max": float(net_drift.max().item())},
              open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
