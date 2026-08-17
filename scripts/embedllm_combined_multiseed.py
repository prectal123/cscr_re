"""Combined fix: category-track-record filtering (top50pct cutoff, just
finalized as the best of the 3 candidate cutoff rules -- see
allseen_catfilter_methodology_multiseed_results.json, mean AUDC 0.5388 vs
top2's 0.5346 and margin's 0.5335, also the most stable of the two viable
options) feeding INTO min-pos loss: catfilter's mask decides which
positive-advantage models are even eligible, then min-pos takes the min
squared-error over whatever positives survive (instead of the mean), while
negatives are still mean-MSE'd as before.

Multi-seed (0-2), BOTH protocols:
  unseen  (vs CSCR paper AUDC=0.4848)
  allseen (vs CSCR paper AUDC=0.541)

Reuses the exact same data-loading / cutoff / training-loop / eval code
paths already verified in embedllm_catfilter_multiseed_costfixed.py (the
just-finished top2/top50pct/margin comparison) and the minpos_loss from
embedllm_newllm_grpo_train_minpos.py -- only the loss function changes
from plain MSE to minpos_loss, and the cutoff is fixed to top50pct only
(no more mode looping).
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
from router.embedllm import load_embedllm
from router.bandit import BanditStats
from router.cost_models import compute_cost
from run_audc_eval import interp_to_grid, build_cost_grid, paired_bootstrap_audc_cached
from embedllm_newllm_fast_eval import run_one as fast_eval_unseen

PCA5_DIR = Path("local_descriptors/embedllm-ceiling-pca5")
ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
HOLDOUT_FRAC = 0.15
SEEDS = [3, 4, 5, 6, 7, 8, 9, 10]
LOG_EVERY = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSCR_UNSEEN = 0.4848
CSCR_ALLSEEN = 0.541
EXCLUDE = {"JaeyeonKang__CCK_Asura_v1"}
LAM_LIST = np.logspace(-4, 2, 20)
K = 20
BANDIT_BETA = 0.000001
COST_GRID_POINTS = 20


def compute_keep_idx_top50pct(pos_idx: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """The finalized catfilter cutoff rule (top50pct), same logic as
    embedllm_catfilter_multiseed_costfixed.py's compute_keep_idx(mode='top50pct')."""
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


def knn_curve(sims, models, label_maps, costs, lam_list, k=K, bandit_beta=BANDIT_BETA):
    n_prompts, n_models = sims.shape
    order = np.argsort(-sims, axis=1)
    topk = order[:, :min(k, n_models)]
    out_costs, out_accs = [], []
    Y = np.zeros((len(lam_list), n_prompts), dtype=np.int32)
    for li, lam in enumerate(lam_list):
        bandit = BanditStats(bandit_lambda=float(lam), beta=bandit_beta)
        tot_cost, tot_acc = 0.0, 0.0
        for i in range(n_prompts):
            best_score, best_j = -np.inf, topk[i, 0]
            for j in topk[i]:
                m = models[j]
                bonus = bandit.get_bonus(m)
                score = bonus + sims[i, j] - lam * costs[j]
                if score > best_score:
                    best_score, best_j = score, j
            chosen = models[best_j]
            acc = 1.0 if label_maps[i].get(chosen, 0) == 1 else 0.0
            cost = float(costs[best_j])
            bandit.update(chosen, accuracy=acc, cost=cost)
            tot_cost += cost
            tot_acc += acc
            Y[li, i] = int(acc)
        out_costs.append(tot_cost / n_prompts)
        out_accs.append(tot_acc / n_prompts)
    return np.array(out_costs), np.array(out_accs), Y


def audc_qnc_peak(costs, accs):
    order = np.argsort(costs)
    c, a = costs[order], accs[order]
    grid = build_cost_grid(c, N_grid=COST_GRID_POINTS)
    a_grid = interp_to_grid(c, a, grid)
    audc = np.trapezoid(a_grid, grid) / (grid[-1] - grid[0])
    peak_idx = np.argmax(a)
    return {"audc": float(audc), "qnc": float(c[peak_idx]), "peak": float(a[peak_idx])}


def eval_allseen(enc, models, texts, label_maps, E_norm, costs):
    embeds = np.zeros((len(texts), E_norm.shape[1]), dtype=np.float32)
    for start in range(0, len(texts), 64):
        batch = texts[start:start + 64]
        embeds[start:start + len(batch)] = enc.encode(batch)
    sims = embeds @ E_norm.T
    knn_costs, knn_accs, knn_Y = knn_curve(sims, models, label_maps, costs, LAM_LIST)
    return audc_qnc_peak(knn_costs, knn_accs)


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

    results = {"unseen": [], "allseen": []}

    # ===================== UNSEEN PROTOCOL =====================
    print("\n" + "#" * 70 + "\nUNSEEN PROTOCOL (combined: minpos-loss + top50pct-catfilter)\n" + "#" * 70, flush=True)
    for seed in SEEDS:
        split_path = ANALYSIS_DIR / ("newllm_split.json" if seed == 0 else f"newllm_split_seed{seed}.json")
        split = json.load(open(split_path, encoding="utf-8"))
        seen_models = split["seen"]
        unseen_dir = Path("local_descriptors/embedllm-ceiling-pca5-unseen-only" if seed == 0
                           else f"local_descriptors/embedllm-ceiling-pca5-unseen-only-seed{seed}")

        E_seen = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in seen_models])
        E_seen_t = torch.from_numpy(E_seen).float().to(DEVICE)
        E_seen_t = E_seen_t / (E_seen_t.norm(dim=1, keepdim=True) + 1e-9)
        raw_cat_acc = build_raw_category_accuracy(df, seen_models, categories)
        items = build_items(df, seen_models, raw_cat_acc, category_to_idx)

        enc = train(seed, items, tokenizer, base_model, E_seen_t, f"unseen-combined-seed{seed}")
        ckpt_dir = Path(f"local_checkpoints/embedllm-newllm-encoder-combined-seed{seed}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        enc.save(ckpt_dir)
        r = fast_eval_unseen(str(split_path), str(unseen_dir), str(ckpt_dir), label=f"unseen-combined-seed{seed}")
        results["unseen"].append({"seed": seed, **r["knn"]})
        print(f"  [unseen seed={seed} combined] AUDC={r['knn']['audc']:.4f} Peak={r['knn']['peak']:.4f} "
              f"({'BEATS' if r['knn']['audc'] > CSCR_UNSEEN else 'below'} CSCR {CSCR_UNSEEN})", flush=True)

    # ===================== ALL-SEEN PROTOCOL =====================
    print("\n" + "#" * 70 + "\nALL-SEEN PROTOCOL (combined)\n" + "#" * 70, flush=True)
    all_models = sorted(p.stem for p in PCA5_DIR.glob("*.npy"))
    models = [m for m in all_models if m not in EXCLUDE]
    costs = np.array([compute_cost(m, 0, cost_type="n_params") for m in models], dtype=np.float32)
    E_all = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in models])
    E_all_t = torch.from_numpy(E_all).float().to(DEVICE)
    E_all_t = E_all_t / (E_all_t.norm(dim=1, keepdim=True) + 1e-9)
    E_all_norm = E_all / (np.linalg.norm(E_all, axis=1, keepdims=True) + 1e-12)
    raw_cat_acc_all = build_raw_category_accuracy(df, models, categories)
    items_all = build_items(df, models, raw_cat_acc_all, category_to_idx)

    dataset = load_embedllm("test", candidates=models)
    texts = [ex["prompt"] for ex in dataset]
    label_maps = [ex["label_map"] for ex in dataset]

    for seed in SEEDS:
        enc = train(seed, items_all, tokenizer, base_model, E_all_t, f"allseen-combined-seed{seed}")
        ckpt_dir = Path(f"local_checkpoints/embedllm-allseen-combined-seed{seed}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        enc.save(ckpt_dir)
        m_result = eval_allseen(enc, models, texts, label_maps, E_all_norm, costs)
        results["allseen"].append({"seed": seed, **m_result})
        print(f"  [allseen seed={seed} combined] AUDC={m_result['audc']:.4f} Peak={m_result['peak']:.4f} "
              f"({'BEATS' if m_result['audc'] > CSCR_ALLSEEN else 'below'} CSCR {CSCR_ALLSEEN})", flush=True)

    out_path = ANALYSIS_DIR / "combined_minpos_top50pct_seeds3to10_results.json"
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n" + "=" * 100)
    print("COMBINED (min-pos loss + top50pct catfilter) MULTI-SEED SUMMARY")
    print("=" * 100)
    for protocol, cscr_ref in [("unseen", CSCR_UNSEEN), ("allseen", CSCR_ALLSEEN)]:
        audcs = [r["audc"] for r in results[protocol]]
        beats = sum(1 for a in audcs if a > cscr_ref)
        print(f"  {protocol:>8} (CSCR={cscr_ref}): " + " / ".join(f"{a:.4f}" for a in audcs) +
              f"  mean={np.mean(audcs):.4f} (std={np.std(audcs):.4f})  beats CSCR: {beats}/{len(audcs)}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
