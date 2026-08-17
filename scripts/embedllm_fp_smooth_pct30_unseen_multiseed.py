"""Multi-seed validation of FP-smoothing (K=5, beta=0.25, the best point from
the seed0-only sweep) stacked on the combined (min-pos + pct=0.3 catfilter)
recipe, EmbedLLM unseen protocol, seeds 0-2.

seed0-only result was 0.5327 vs the combined-only seed0 baseline of 0.5264
(+0.0063) -- promising but single-seed. Reference multi-seed baseline
(no smoothing, same combined recipe, seeds 0-2) is
unseen_catfilter_pct30_multiseed_results.json: [0.5264, 0.5246, 0.4977],
mean=0.5162. Smoothing must be recomputed per seed since the seen/unseen
split (and therefore the smoothing neighbor pool) differs per seed.
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
SEEDS = [0, 1, 2]
PCT = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSCR_UNSEEN = 0.4848

SMOOTH_K = 5
SMOOTH_BETA = 0.25

REFERENCE_NO_SMOOTH_MULTISEED = [0.5264049090871115, 0.5246252930579587, 0.4976619058233775]


def build_raw_category_accuracy(df, models, categories):
    pivot = df.pivot_table(index="model_name", columns="category", values="label", aggfunc="mean")
    pivot = pivot.reindex(index=models, columns=categories)
    raw = pivot.to_numpy()
    col_mean = np.nanmean(raw, axis=0, keepdims=True)
    return np.where(np.isnan(raw), col_mean, raw)


def smooth_fps(all_models, seen_models, raw_cat_acc_full, E_full, k=SMOOTH_K, beta=SMOOTH_BETA):
    seen_set = set(seen_models)
    seen_idx = [i for i, m in enumerate(all_models) if m in seen_set]
    norm_acc = raw_cat_acc_full / (np.linalg.norm(raw_cat_acc_full, axis=1, keepdims=True) + 1e-12)
    sim_to_seen = norm_acc @ norm_acc[seen_idx].T

    E_new = E_full.copy()
    for i, m in enumerate(all_models):
        sims_row = sim_to_seen[i].copy()
        if m in seen_set:
            self_pos = seen_idx.index(i)
            sims_row[self_pos] = -np.inf
        top_k_local = np.argsort(-sims_row)[:k]
        neighbor_global_idx = [seen_idx[j] for j in top_k_local]
        neighbor_mean = E_full[neighbor_global_idx].mean(axis=0)
        blended = (1 - beta) * E_full[i] + beta * neighbor_mean
        E_new[i] = blended / (np.linalg.norm(blended) + 1e-12)
    drift = np.linalg.norm(E_new - E_full, axis=1)
    print(f"  smoothed {len(all_models)} models (K={k}, beta={beta}): "
          f"drift mean={drift.mean():.4f} max={drift.max():.4f}", flush=True)
    return E_new


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
            order = np.argsort(-scores)
            sorted_idx = pos_idx[order]
            n_keep = max(1, int(np.ceil(len(pos_idx) * pct)))
            keep_pos = sorted_idx[:n_keep]
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


def train(seed, items, tokenizer, base_model, E_t, tag):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
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

    best_rho, best_epoch, best_state = -1.0, -1, None
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
        rho_arr = evaluate_holdout(enc, holdout_texts, holdout_targets, holdout_masks, E_t)
        if rho_arr.mean() > best_rho:
            best_rho, best_epoch = rho_arr.mean(), ep + 1
            best_state = {k: v.clone().cpu() for k, v in enc.proj.state_dict().items()}
    print(f"  [{tag}] best_epoch={best_epoch} best_holdout_rho={best_rho:.4f} time={time.time()-t0:.1f}s", flush=True)
    enc.proj.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return enc


def main():
    print("Loading EmbedLLM train.csv...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "category", "label", "prompt"])
    categories = sorted(df["category"].unique().tolist())
    category_to_idx = {c: i for i, c in enumerate(categories)}

    all_models = sorted(p.stem for p in PCA5_DIR.glob("*.npy"))
    raw_cat_acc_full = build_raw_category_accuracy(df, all_models, categories)
    E_full = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in all_models]).astype(np.float32)

    print(f"Loading frozen MiniLM base (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    results = []
    for seed in SEEDS:
        split_path = ANALYSIS_DIR / ("newllm_split.json" if seed == 0 else f"newllm_split_seed{seed}.json")
        split = json.load(open(split_path, encoding="utf-8"))
        seen_models, unseen_models = split["seen"], split["unseen"]

        print(f"\n{'='*60}\nSEED {seed}: seen={len(seen_models)} unseen={len(unseen_models)}\n{'='*60}", flush=True)
        E_smoothed = smooth_fps(all_models, seen_models, raw_cat_acc_full, E_full)

        model_to_idx = {m: i for i, m in enumerate(all_models)}
        smoothed_dir = Path(f"local_descriptors/embedllm-ceiling-pca5-smoothed-beta{SMOOTH_BETA}-seed{seed}")
        smoothed_unseen_dir = Path(f"local_descriptors/embedllm-ceiling-pca5-smoothed-unseen-only-beta{SMOOTH_BETA}-seed{seed}")
        smoothed_dir.mkdir(parents=True, exist_ok=True)
        smoothed_unseen_dir.mkdir(parents=True, exist_ok=True)
        for m in all_models:
            np.save(smoothed_dir / f"{m}.npy", E_smoothed[model_to_idx[m]])
        for m in unseen_models:
            np.save(smoothed_unseen_dir / f"{m}.npy", E_smoothed[model_to_idx[m]])

        E_seen = np.stack([np.load(smoothed_dir / f"{m}.npy") for m in seen_models])
        E_seen_t = torch.from_numpy(E_seen).float().to(DEVICE)
        E_seen_t = E_seen_t / (E_seen_t.norm(dim=1, keepdim=True) + 1e-9)

        raw_cat_acc_seen = build_raw_category_accuracy(df, seen_models, categories)
        items = build_items(df, seen_models, raw_cat_acc_seen, category_to_idx, PCT)

        enc = train(seed, items, tokenizer, base_model, E_seen_t, f"fpsmooth-beta{SMOOTH_BETA}-unseen-seed{seed}")
        ckpt_dir = Path(f"local_checkpoints/embedllm-newllm-encoder-fpsmooth-beta{SMOOTH_BETA}-seed{seed}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        enc.save(ckpt_dir)
        r = fast_eval_unseen(str(split_path), str(smoothed_unseen_dir), str(ckpt_dir),
                              label=f"fpsmooth-beta{SMOOTH_BETA}-unseen-seed{seed}")
        results.append({"seed": seed, **r["knn"]})
        print(f"  [seed={seed}] AUDC={r['knn']['audc']:.4f} Peak={r['knn']['peak']:.4f} "
              f"({'BEATS' if r['knn']['audc'] > CSCR_UNSEEN else 'below'} CSCR {CSCR_UNSEEN})", flush=True)

    out_path = ANALYSIS_DIR / "fp_smooth_pct30_unseen_multiseed_results.json"
    json.dump(results, open(out_path, "w"), indent=2)

    audcs = [r["audc"] for r in results]
    print("\n" + "=" * 90)
    print(f"FP-SMOOTH (K={SMOOTH_K}, beta={SMOOTH_BETA}) + COMBINED (min-pos + pct0.3 catfilter), EmbedLLM unseen, {len(SEEDS)} seeds")
    print("=" * 90)
    for r in results:
        print(f"  seed={r['seed']}: AUDC={r['audc']:.4f} Peak={r['peak']:.4f}")
    print(f"\nmean AUDC={np.mean(audcs):.4f} (std={np.std(audcs):.4f})  beats CSCR: {sum(1 for a in audcs if a > CSCR_UNSEEN)}/{len(audcs)}")
    ref_mean = float(np.mean(REFERENCE_NO_SMOOTH_MULTISEED))
    print(f"reference (no smoothing, same combined recipe, seed0-2): {REFERENCE_NO_SMOOTH_MULTISEED} mean={ref_mean:.4f}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
