"""GRPO regression variant that fixes outlier-drag via WITHIN-CATEGORY
track-record filtering of the positive set (user's idea): when multiple
seen models are simultaneously "correct" (label=1) for a training query,
only the TOP_K of them BY THEIR OWN HISTORICAL ACCURACY IN THAT QUERY'S
CATEGORY (computed from all of Set A, same aggregate-statistics principle
already used to build every Ceiling FP in this project) are kept as
positive training targets. The rest are excluded from the loss entirely for
that example (not flipped to negative -- they DID answer correctly, we just
don't trust that as a category-relevant signal for a model with no track
record there; punishing them as "wrong" would be factually incorrect).

This directly targets the outlier-drag mechanism confirmed in
embedllm_outlier_blend_check.py (rho=0.52 between positive-set FP-space
spread and how far the blended target lands from any real model) from a
different angle than embedllm_newllm_grpo_train_minpos.py's min-loss fix:
instead of changing HOW the positive set is aggregated, this changes WHICH
models are allowed to count as positive in the first place, using
information (category track record) the min-loss/robust-stats fixes never
had access to.

Otherwise identical to the original embedllm_newllm_grpo_train.py (plain
mean-over-kept-positives MSE, same architecture/epochs/seed) so any
difference in outcome is attributable ONLY to this filtering step.
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
TOP_K = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPLIT_PATH = ANALYSIS_DIR / "newllm_split.json"
UNSEEN_DIR = Path("local_descriptors/embedllm-ceiling-pca5-unseen-only")
CKPT_DIR = Path("local_checkpoints/embedllm-newllm-encoder-grpo-catfilter-seed0")
CSCR_PAPER = {"audc": 0.4848, "peak": 0.565}


class QueryTargetDataset(TorchDataset):
    """Same GRPO target construction as before, PLUS a category-track-record
    filter on the positive side: if more than TOP_K seen models are
    positive for this query, only the TOP_K by raw_cat_acc[model, category]
    keep mask=1; the demoted positives get mask=0 (excluded from loss,
    not flipped to negative)."""

    def __init__(self, df, seen_models, raw_cat_acc, category_to_idx):
        name_to_idx = {n: i for i, n in enumerate(seen_models)}
        n_demoted_total, n_queries_with_demotion = 0, 0
        self.items = []
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
            if cat_idx is not None and len(pos_idx) > TOP_K:
                scores = raw_cat_acc[pos_idx, cat_idx]
                # NaN track record (model never seen in this category in Set A) ranks last
                scores = np.nan_to_num(scores, nan=-1.0)
                order = np.argsort(-scores)
                demoted = pos_idx[order[TOP_K:]]
                keep_mask[demoted] = False
                n_demoted_total += len(demoted)
                n_queries_with_demotion += 1

            self.items.append((text, target, keep_mask.astype(np.float32)))
        print(f"  [dataset] {n_queries_with_demotion} queries had >{TOP_K} positives "
              f"({n_demoted_total} total positive-slots demoted/excluded)", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def build_raw_category_accuracy(df, seen_models, categories):
    """Raw (uncentered, unnormalized) per-model per-category mean accuracy
    from Set A -- used ONLY for ranking positive candidates by category
    track record, never as the training target/anchor itself."""
    pivot = df.pivot_table(index="model_name", columns="category", values="label", aggfunc="mean")
    pivot = pivot.reindex(index=seen_models, columns=categories)
    return pivot.to_numpy()  # (n_seen, n_categories), NaN where no data


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
    print(f"seen={len(seen_models)} unseen={len(split['unseen'])}  TOP_K={TOP_K}", flush=True)

    print("Loading EmbedLLM train.csv (with category)...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "category", "label", "prompt"])

    categories = sorted(df["category"].unique().tolist())
    category_to_idx = {c: i for i, c in enumerate(categories)}
    print(f"{len(categories)} categories", flush=True)

    raw_cat_acc = build_raw_category_accuracy(df, seen_models, categories)
    print(f"raw category-accuracy matrix: {raw_cat_acc.shape}, "
          f"{np.isnan(raw_cat_acc).sum()} NaN entries (model/category pairs with no Set A data)", flush=True)

    ds_all = QueryTargetDataset(df, seen_models, raw_cat_acc, category_to_idx)
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
            loss = ((cos_sim - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)
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
    print(f"Full holdout_rho trace: {[f'{r:.4f}' for r in rho_trace]}", flush=True)

    enc.proj.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    enc.save(CKPT_DIR)
    print(f"Saved best-epoch checkpoint -> {CKPT_DIR}", flush=True)

    print("\nRunning fast AUDC eval on unseen-only pool with best checkpoint...", flush=True)
    r = fast_eval(str(SPLIT_PATH), str(UNSEEN_DIR), str(CKPT_DIR), label="seed0-grpo-catfilter")
    r["best_epoch"] = best_epoch
    r["best_holdout_rho"] = float(best_rho)
    r["rho_trace"] = rho_trace
    r["top_k"] = TOP_K
    out_path = ANALYSIS_DIR / "newllm_grpo_catfilter_seed0_results.json"
    json.dump(r, open(out_path, "w"), indent=2)

    print("\n" + "=" * 60)
    print(f"GRPO CATEGORY-TRACK-RECORD FILTER SUMMARY (seed 0, TOP_K={TOP_K})")
    print("=" * 60)
    print(f"best_epoch={best_epoch}  best_holdout_rho={best_rho:.4f}")
    print(f"knn: AUDC={r['knn']['audc']:.4f} Peak={r['knn']['peak']:.4f}  "
          f"({'BEATS' if r['knn']['audc'] > CSCR_PAPER['audc'] else 'below'} CSCR {CSCR_PAPER['audc']})")
    print(f"random: AUDC={r['random']['audc']:.4f} Peak={r['random']['peak']:.4f}")
    print("reference (plain GRPO, no filtering): seed0 AUDC=0.5289 Peak=0.5687 (best_epoch=3)")
    print("reference (min-over-positives):       seed0 AUDC=0.5095 Peak=0.5487 (best_epoch=2)")
    print(f"\nSaved -> {out_path}  checkpoint -> {CKPT_DIR}")


if __name__ == "__main__":
    main()
