"""Multi-seed (1-3) comparison of the two outlier-drag fixes -- min-over-positives
(embedllm_newllm_grpo_train_minpos.py) and category-track-record filtering
(embedllm_newllm_grpo_train_catfilter.py) -- against the known reference
numbers (seed 0 already run for both, plain GRPO's 4-seed numbers already
known from newllm_grpo_multiseed_seeds0to3_snapshot.json).

Loads EmbedLLM train.csv + frozen MiniLM ONCE and reuses across all 6 runs
(2 variants x 3 seeds) for efficiency, mirroring embedllm_newllm_grpo_beta_sweep.py.

Known seed-0 reference points (already run, not repeated here):
  plain GRPO:    AUDC=0.5289 Peak=0.5687 best_epoch=3
  min-pos:       AUDC=0.5095 Peak=0.5487 best_epoch=2
  cat-filter(k2):AUDC=0.5250 Peak=0.5807 best_epoch=4
Plain GRPO's known seeds 1-3: AUDC=0.4661, 0.4492, 0.4666 (mean of 0-3 = 0.4772)
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
from embedllm_newllm_fast_eval import run_one as fast_eval

PCA5_DIR = Path("local_descriptors/embedllm-ceiling-pca5")
ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
HOLDOUT_FRAC = 0.15
TOP_K = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSCR_PAPER = {"audc": 0.4848, "peak": 0.565}

SEEDS = [1, 2, 3]
SEED0_REF = {
    "grpo": {"audc": 0.5289, "peak": 0.5687},
    "minpos": {"audc": 0.5095, "peak": 0.5487},
    "catfilter": {"audc": 0.5250, "peak": 0.5807},
}


def split_path_for(seed):
    return ANALYSIS_DIR / f"newllm_split_seed{seed}.json"


def unseen_dir_for(seed):
    return Path(f"local_descriptors/embedllm-ceiling-pca5-unseen-only-seed{seed}")


def build_raw_category_accuracy(df, seen_models, categories):
    pivot = df.pivot_table(index="model_name", columns="category", values="label", aggfunc="mean")
    pivot = pivot.reindex(index=seen_models, columns=categories)
    return pivot.to_numpy()


def build_minpos_items(df, seen_models):
    name_to_idx = {n: i for i, n in enumerate(seen_models)}
    items = []
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
        items.append((text, target, mask.astype(np.float32)))
    return items


def build_catfilter_items(df, seen_models, raw_cat_acc, category_to_idx, top_k=TOP_K):
    name_to_idx = {n: i for i, n in enumerate(seen_models)}
    items = []
    for pid, grp in df.groupby("prompt_id", sort=False):
        text = grp["prompt"].iloc[0]
        category = grp["category"].iloc[0]
        cat_idx = category_to_idx.get(category)
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

        keep_mask = mask.copy()
        pos_idx = np.where(mask & (labels == 1))[0]
        if cat_idx is not None and len(pos_idx) > top_k:
            scores = np.nan_to_num(raw_cat_acc[pos_idx, cat_idx], nan=-1.0)
            order = np.argsort(-scores)
            demoted = pos_idx[order[top_k:]]
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


def plain_mse_loss(cos_sim, target, mask):
    return ((cos_sim - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)


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


def train_one(seed, variant, items, tokenizer, base_model, E_seen_t, loss_fn):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(items))
    n_holdout = int(len(items) * HOLDOUT_FRAC)
    holdout_idx, train_idx = perm[:n_holdout], perm[n_holdout:]
    train_items = [items[i] for i in train_idx]
    holdout_texts = [items[i][0] for i in holdout_idx]
    holdout_targets = np.stack([items[i][1] for i in holdout_idx])
    holdout_masks = np.stack([items[i][2] for i in holdout_idx])

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

    def collate(batch):
        texts, targets, masks = zip(*batch)
        toks = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt", max_length=256)
        return toks, torch.tensor(np.stack(targets)), torch.tensor(np.stack(masks))

    loader = torch.utils.data.DataLoader(train_items, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

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
            cos_sim = q @ E_seen_t.T
            loss = loss_fn(cos_sim, target, mask)
            loss.backward()
            opt.step()
            opt.zero_grad()

        rho_arr = evaluate_holdout(enc, holdout_texts, holdout_targets, holdout_masks, E_seen_t)
        if rho_arr.mean() > best_rho:
            best_rho, best_epoch = rho_arr.mean(), ep + 1
            best_state = {k: v.clone().cpu() for k, v in enc.proj.state_dict().items()}
    print(f"  [{variant} seed={seed}] best_epoch={best_epoch} best_holdout_rho={best_rho:.4f} "
          f"time={time.time()-t0:.1f}s", flush=True)

    enc.proj.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return enc, best_epoch, best_rho


def main():
    print("Loading EmbedLLM train.csv...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "category", "label", "prompt"])
    categories = sorted(df["category"].unique().tolist())
    category_to_idx = {c: i for i, c in enumerate(categories)}

    print(f"Loading frozen MiniLM base ONCE (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    results = {"minpos": [], "catfilter": []}
    for seed in SEEDS:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)
        split = json.load(open(split_path_for(seed), encoding="utf-8"))
        seen_models, unseen = split["seen"], split["unseen"]

        E_seen = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in seen_models])
        E_seen_t = torch.from_numpy(E_seen).float().to(DEVICE)
        E_seen_t = E_seen_t / (E_seen_t.norm(dim=1, keepdim=True) + 1e-9)

        raw_cat_acc = build_raw_category_accuracy(df, seen_models, categories)
        minpos_items = build_minpos_items(df, seen_models)
        catfilter_items = build_catfilter_items(df, seen_models, raw_cat_acc, category_to_idx)
        print(f"  seed={seed}: seen={len(seen_models)} minpos_items={len(minpos_items)} "
              f"catfilter_items={len(catfilter_items)}", flush=True)

        for variant, items, loss_fn in [("minpos", minpos_items, minpos_loss),
                                         ("catfilter", catfilter_items, plain_mse_loss)]:
            enc, best_epoch, best_rho = train_one(seed, variant, items, tokenizer, base_model, E_seen_t, loss_fn)
            ckpt_dir = Path(f"local_checkpoints/embedllm-newllm-encoder-grpo-{variant}-seed{seed}")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            enc.save(ckpt_dir)

            r = fast_eval(str(split_path_for(seed)), str(unseen_dir_for(seed)), str(ckpt_dir),
                          label=f"{variant}-seed{seed}")
            r["seed"] = seed
            r["variant"] = variant
            r["best_epoch"] = best_epoch
            r["best_holdout_rho"] = float(best_rho)
            results[variant].append(r)
            print(f"  [{variant} seed={seed}] RESULT: AUDC={r['knn']['audc']:.4f} Peak={r['knn']['peak']:.4f} "
                  f"({'BEATS' if r['knn']['audc'] > CSCR_PAPER['audc'] else 'below'} CSCR)", flush=True)

    out_path = ANALYSIS_DIR / "newllm_grpo_variant_multiseed_results.json"
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n" + "=" * 100)
    print("MULTI-SEED VARIANT COMPARISON (seeds 1-3, plus known seed-0 references)")
    print("=" * 100)

    grpo_seeds = [0.5289, 0.4661, 0.4492, 0.4666]  # known, from newllm_grpo_multiseed_seeds0to3_snapshot.json
    print(f"{'variant':>12} | {'seed0':>7} | {'seed1':>7} | {'seed2':>7} | {'seed3':>7} | {'mean(0-3)':>10} | beats_cscr")
    print("-" * 100)
    print(f"{'grpo(orig)':>12} | {grpo_seeds[0]:>7.4f} | {grpo_seeds[1]:>7.4f} | {grpo_seeds[2]:>7.4f} | {grpo_seeds[3]:>7.4f} | "
          f"{np.mean(grpo_seeds):>10.4f} | {sum(1 for a in grpo_seeds if a > 0.4848)}/4")

    for variant in ["minpos", "catfilter"]:
        seed0 = SEED0_REF[variant]["audc"]
        s123 = [r["knn"]["audc"] for r in results[variant]]
        all4 = [seed0] + s123
        beats = sum(1 for a in all4 if a > CSCR_PAPER["audc"])
        print(f"{variant:>12} | {seed0:>7.4f} | {s123[0]:>7.4f} | {s123[1]:>7.4f} | {s123[2]:>7.4f} | "
              f"{np.mean(all4):>10.4f} | {beats}/4")

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
