"""Does combined (min-pos loss + top50pct catfilter) survive on the more
REALISTIC probe-sampled Ceiling FP ("Ceiling V1", N=24 probes/category)
instead of the oracle-quality full-average Ceiling FP ("Ceiling V2") used
everywhere else this session?

Motivation: vanilla GRPO/cost_spectrum_info_nce on Ceiling V1 never beats
CSCR at N=24 (5-seed AUDC = 0.4161/0.4701/0.4507/0.4575/0.4160, mean=0.4421,
0/5 beat CSCR's 0.4848 -- see newllm_probe_sampling_results.json), and at
N=6 it's sometimes WORSE than random routing. All of this session's "beats CSCR" claims were made on
Ceiling V2 (oracle-informed FP, built from the model's FULL Set A category
accuracy) -- Ceiling V1 is the realistic case (FP built from only a
handful of probe prompts/category, no full benchmark run needed). If
combined can't recover performance here, the outlier-drag fix doesn't
generalize beyond the favorable oracle-FP setting.

3 seeds (0,1,2), unseen protocol only (matches the probe-sampling
infrastructure -- no all-seen probe-FP setup exists yet). Reuses the
already-built local_descriptors/embedllm-ceiling-probeN24-pca5/ (and its
per-seed unseen-only copies) from embedllm_newllm_probe_sampling.py, and
the exact same catfilter/minpos_loss code already verified in
embedllm_combined_multiseed.py -- only the FP source directory changes.
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

PROBE_N = 24
PROBE_FP_DIR = Path(f"local_descriptors/embedllm-ceiling-probeN{PROBE_N}-pca5")
ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
HOLDOUT_FRAC = 0.15
SEEDS = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSCR_UNSEEN = 0.4848
K_PCA = 5

# verified from newllm_probe_sampling_results.json: N=24 AUDC per seed(0-4) =
# 0.4161/0.4701/0.4507/0.4575/0.4160, mean=0.4421, 0/5 beat CSCR (0.4848)
VANILLA_REF_MEAN = 0.4421
VANILLA_REF_BEATS = "0/5"


def compute_keep_idx_top50pct(pos_idx: np.ndarray, scores: np.ndarray) -> np.ndarray:
    if len(pos_idx) <= 1:
        return pos_idx
    order = np.argsort(-scores)
    sorted_idx = pos_idx[order]
    n_keep = max(1, int(np.ceil(len(pos_idx) * 0.5)))
    return sorted_idx[:n_keep]


def build_raw_category_accuracy(df, models, categories):
    pivot = df.pivot_table(index="model_name", columns="category", values="label", aggfunc="mean")
    pivot = pivot.reindex(index=models, columns=categories)
    return pivot.to_numpy()


def build_items(df, models, raw_cat_acc, category_to_idx):
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
        if cat_idx is not None and len(pos_idx) > 0:
            scores = np.nan_to_num(raw_cat_acc[pos_idx, cat_idx], nan=-1.0)
            keep_pos = compute_keep_idx_top50pct(pos_idx, scores)
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


def split_path_for(seed):
    return ANALYSIS_DIR / ("newllm_split.json" if seed == 0 else f"newllm_split_seed{seed}.json")


def main():
    print(f"Loading EmbedLLM train.csv (with category)...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "category", "label", "prompt"])
    categories = sorted(df["category"].unique().tolist())
    category_to_idx = {c: i for i, c in enumerate(categories)}

    probe_fp_models = sorted(p.stem for p in PROBE_FP_DIR.glob("*.npy"))
    print(f"Probe-sampled FP dir (N={PROBE_N}) has {len(probe_fp_models)} models", flush=True)

    print(f"Loading frozen MiniLM base ONCE (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    results = []
    for seed in SEEDS:
        split_path = split_path_for(seed)
        split = json.load(open(split_path, encoding="utf-8"))
        seen_models = [m for m in split["seen"] if m in probe_fp_models]
        unseen_dir = Path(f"local_descriptors/embedllm-ceiling-probeN{PROBE_N}-pca5-unseen-only-seed{seed}")
        print(f"\n{'='*60}\nSEED {seed}: seen={len(seen_models)} (probe-FP covered)\n{'='*60}", flush=True)

        E_seen = np.stack([np.load(PROBE_FP_DIR / f"{m}.npy") for m in seen_models])
        E_seen_t = torch.from_numpy(E_seen).float().to(DEVICE)
        E_seen_t = E_seen_t / (E_seen_t.norm(dim=1, keepdim=True) + 1e-9)
        raw_cat_acc = build_raw_category_accuracy(df, seen_models, categories)
        items = build_items(df, seen_models, raw_cat_acc, category_to_idx)

        enc = train(seed, items, tokenizer, base_model, E_seen_t, f"probeN{PROBE_N}-combined-seed{seed}")
        ckpt_dir = Path(f"local_checkpoints/embedllm-newllm-encoder-probeN{PROBE_N}-combined-seed{seed}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        enc.save(ckpt_dir)
        r = fast_eval_unseen(str(split_path), str(unseen_dir), str(ckpt_dir), label=f"probeN{PROBE_N}-combined-seed{seed}")
        results.append({"seed": seed, **r["knn"]})
        print(f"  [seed={seed} probeN{PROBE_N} combined] AUDC={r['knn']['audc']:.4f} Peak={r['knn']['peak']:.4f} "
              f"({'BEATS' if r['knn']['audc'] > CSCR_UNSEEN else 'below'} CSCR {CSCR_UNSEEN})", flush=True)

    out_path = ANALYSIS_DIR / f"probeN{PROBE_N}_combined_multiseed_results.json"
    json.dump(results, open(out_path, "w"), indent=2)

    audcs = [r["audc"] for r in results]
    print("\n" + "=" * 90)
    print(f"COMBINED on Ceiling V1 (probe-sampled, N={PROBE_N}) -- 3-seed summary")
    print("=" * 90)
    for r in results:
        beats = "BEATS" if r["audc"] > CSCR_UNSEEN else "below"
        print(f"  seed={r['seed']}: AUDC={r['audc']:.4f} ({beats} CSCR {CSCR_UNSEEN})")
    print(f"  mean={np.mean(audcs):.4f} (std={np.std(audcs):.4f})  beats CSCR: {sum(1 for a in audcs if a > CSCR_UNSEEN)}/{len(audcs)}")
    print(f"  reference -- vanilla GRPO on same N={PROBE_N} FP, 5-seed mean={VANILLA_REF_MEAN}, {VANILLA_REF_BEATS} beat CSCR")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
