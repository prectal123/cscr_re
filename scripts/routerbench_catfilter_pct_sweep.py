"""Quick trial: is top50%-catfilter simply too lenient to matter once
stacked with min-pos? RouterBench showed catfilter+mean(0.7400) as the
single best standalone config, but combined(min-pos+top50%-catfilter,
0.7226) undercuts it -- the hypothesis is that top50% rarely removes the
candidate min-pos would have picked anyway (min-pos already implicitly
ignores far-away positives by construction), so catfilter's filtering has
little left to bite on. A stricter cutoff should intervene more often.

Sweeps PCT_SWEEP = [0.1, 0.2, 0.3, 0.5] (0.5 included as a same-seed
reproduction check against the already-recorded 0.7216), all combined with
min-pos loss, on RouterBench Ceiling FP, seed0 only -- fast single-seed
scan before deciding if any level is worth a full multi-seed run.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from router.query_encoder import QueryEncoder
from router.bandit import BanditStats
from run_audc_eval import interp_to_grid, build_cost_grid, paired_bootstrap_audc_cached
import routerbench_knn_test as rb

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
HOLDOUT_FRAC = 0.15
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAM_LIST = np.logspace(-4, 2, 20)
K = 11
BANDIT_BETA = 0.000001
COST_GRID_POINTS = 20
CSCR_ROUTERBENCH = 0.711

PCT_SWEEP = [0.1, 0.2, 0.3, 0.5]
REFERENCE = {"vanilla": 0.7420, "min_pos_alone": 0.7196, "catfilter_top2_mean": 0.7397, "combined_top50pct": 0.7216}


def compute_keep_idx_pct(pos_idx: np.ndarray, scores: np.ndarray, pct: float) -> np.ndarray:
    if len(pos_idx) <= 1:
        return pos_idx
    order = np.argsort(-scores)
    sorted_idx = pos_idx[order]
    n_keep = max(1, int(np.ceil(len(pos_idx) * pct)))
    return sorted_idx[:n_keep]


def build_items(set_a, models, cols, pct):
    eval_names = sorted(set_a["eval_name"].unique())
    cat_to_idx = {e: i for i, e in enumerate(eval_names)}
    n_models = len(models)

    raw_cat_acc = np.zeros((n_models, len(eval_names)))
    counts = np.zeros((n_models, len(eval_names)))
    for i, col in enumerate(cols):
        for ev, score in zip(set_a["eval_name"], set_a[col]):
            ci = cat_to_idx[ev]
            raw_cat_acc[i, ci] += float(score)
            counts[i, ci] += 1
    raw_cat_acc = raw_cat_acc / np.maximum(counts, 1)

    items = []
    n_split, n_total_multi = 0, 0
    for _, row in set_a.iterrows():
        text = row["prompt"]
        labels = np.array([float(row[c]) for c in cols], dtype=np.float32)
        mean, std = labels.mean(), labels.std()
        if std < 1e-6:
            continue
        target = (labels - mean) / (std + 1e-6)
        mask = np.ones(n_models, dtype=np.float32)

        cat_idx = cat_to_idx.get(row["eval_name"])
        pos_idx = np.where(labels == 1)[0]
        if cat_idx is not None and len(pos_idx) > 1:
            n_total_multi += 1
            scores = raw_cat_acc[pos_idx, cat_idx]
            keep_pos = compute_keep_idx_pct(pos_idx, scores, pct)
            if len(keep_pos) < len(pos_idx):
                n_split += 1
            demoted = np.setdiff1d(pos_idx, keep_pos)
            mask[demoted] = 0.0

        items.append((text, target, mask))
    print(f"  pct={pct}: {n_split}/{n_total_multi} multi-positive queries actually filtered "
          f"({n_split/max(n_total_multi,1)*100:.1f}%)", flush=True)
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


def evaluate_holdout(enc, texts, targets, masks, E_t, batch_size=32):
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


def train(items, tokenizer, base_model, E_t, tag):
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
                score = bandit.get_bonus(models[j]) + sims[i, j] - lam * costs[j]
                if score > best_score:
                    best_score, best_j = score, j
            acc = label_maps[i][best_j]
            cost = float(costs[best_j])
            bandit.update(models[best_j], accuracy=acc, cost=cost)
            tot_cost += cost
            tot_acc += acc
            Y[li, i] = int(acc >= 0.5)
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


def main():
    set_a, set_b = rb.load_data()
    models, cols = rb.NAMES, rb.MODELS
    cost_cols = [f"{c}|total_cost" for c in cols]

    E = np.stack([np.load(rb.CEILING_DIR / f"{n}.npy") for n in models])
    E_t = torch.from_numpy(E).float().to(DEVICE)
    E_t = E_t / (E_t.norm(dim=1, keepdim=True) + 1e-9)
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    print(f"Loading frozen MiniLM ONCE (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    set_b = set_b.dropna(subset=cols + cost_cols)
    b_texts = set_b["prompt"].tolist()
    b_label_maps = set_b[cols].to_numpy(dtype=np.float32).tolist()
    b_costs = set_b[cost_cols].to_numpy(dtype=np.float32).mean(axis=0)

    results = {}
    for pct in PCT_SWEEP:
        print(f"\n{'#'*60}\nPCT={pct}\n{'#'*60}", flush=True)
        items = build_items(set_a, models, cols, pct)
        enc = train(items, tokenizer, base_model, E_t, f"pct{pct}")

        embeds = np.zeros((len(b_texts), E.shape[1]), dtype=np.float32)
        for start in range(0, len(b_texts), 32):
            batch = b_texts[start:start + 32]
            embeds[start:start + len(batch)] = enc.encode(batch)
        sims = embeds @ E_norm.T

        knn_costs, knn_accs, knn_Y = knn_curve(sims, models, b_label_maps, b_costs, LAM_LIST)
        knn_metrics = audc_qnc_peak(knn_costs, knn_accs)
        results[pct] = knn_metrics
        print(f"  [pct={pct}] AUDC={knn_metrics['audc']:.4f} Peak={knn_metrics['peak']:.4f} QNC={knn_metrics['qnc']:.4f} "
              f"({'BEATS' if knn_metrics['audc'] > CSCR_ROUTERBENCH else 'below'} CSCR {CSCR_ROUTERBENCH})", flush=True)

    out_path = Path("local_descriptors/routerbench-analysis/catfilter_pct_sweep_seed0_results.json")
    json.dump({str(k): v for k, v in results.items()}, open(out_path, "w"), indent=2)

    print("\n" + "=" * 90)
    print(f"CATFILTER PERCENTILE SWEEP (min-pos + top-pct%-catfilter), RouterBench seed0")
    print("=" * 90)
    for pct in PCT_SWEEP:
        m = results[pct]
        print(f"  pct={pct:>4}: AUDC={m['audc']:.4f} Peak={m['peak']:.4f}")
    print(f"\nreference: vanilla={REFERENCE['vanilla']} min-pos alone={REFERENCE['min_pos_alone']} "
          f"catfilter(top2)+mean={REFERENCE['catfilter_top2_mean']} combined(top50pct)={REFERENCE['combined_top50pct']}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
