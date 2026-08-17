"""Quick generalization check (seed 0 only, late-night speed run): does the
outlier-drag phenomenon and the two EmbedLLM fixes (min-pos, category-track-
record filtering) transfer to RouterBench (11 models, 86 real eval_name
task categories, real $ cost)? Reuses RouterBench's existing Ceiling FP
(local_descriptors/routerbench-ceiling/, 86-dim, mean-centered).

All-seen style (11 models is too small for a meaningful seen/unseen split,
so all 11 both train and are routable at eval, like the EmbedLLM all-seen
protocol) -- Set A (80%) for training/holdout, Set B (20%) for final AUDC.
"""
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
import routerbench_knn_test as rb  # reuse MODELS, NAMES, load_data, CEILING_DIR

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
HOLDOUT_FRAC = 0.15
SEED = 0
TOP_K = 2
LOG_EVERY = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAM_LIST = np.logspace(-4, 2, 20)
K = 11  # all models
BANDIT_BETA = 0.000001
COST_GRID_POINTS = 20


def build_items_and_catacc(set_a, models, cols):
    """Build GRPO targets from RouterBench's wide format (one row per prompt,
    one column per model). eval_name is the category. Also returns the raw
    per-model per-category accuracy matrix for category filtering."""
    eval_names = sorted(set_a["eval_name"].unique())
    cat_to_idx = {e: i for i, e in enumerate(eval_names)}
    n_models = len(models)

    # raw category accuracy (Set A), for category-filter ranking
    raw_cat_acc = np.zeros((n_models, len(eval_names)))
    counts = np.zeros((n_models, len(eval_names)))
    for i, col in enumerate(cols):
        for ev, score in zip(set_a["eval_name"], set_a[col]):
            ci = cat_to_idx[ev]
            raw_cat_acc[i, ci] += float(score)
            counts[i, ci] += 1
    raw_cat_acc = raw_cat_acc / np.maximum(counts, 1)

    items = []
    for _, row in set_a.iterrows():
        text = row["prompt"]
        labels = np.array([float(row[c]) for c in cols], dtype=np.float32)
        mean, std = labels.mean(), labels.std()
        if std < 1e-6:
            continue
        target = (labels - mean) / (std + 1e-6)
        mask = np.ones(n_models, dtype=np.float32)

        cat_idx = cat_to_idx.get(row["eval_name"])
        keep_mask_top2 = mask.copy()
        pos_idx = np.where(labels == 1)[0]
        if cat_idx is not None and len(pos_idx) > TOP_K:
            scores = raw_cat_acc[pos_idx, cat_idx]
            order = np.argsort(-scores)
            demoted = pos_idx[order[TOP_K:]]
            keep_mask_top2[demoted] = 0.0

        items.append((text, target, mask, keep_mask_top2))
    return items, raw_cat_acc, eval_names


def plain_mse_loss(cos_sim, target, mask):
    return ((cos_sim - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)


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


def train(seed, train_items_full, tokenizer, base_model, E_t, mask_idx, loss_fn, tag):
    torch.manual_seed(seed)
    items = [(t, tgt, (m0, m1)[mask_idx]) for (t, tgt, m0, m1) in train_items_full]
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
            loss = loss_fn(cos_sim, target, mask)
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


def random_curve(models, label_maps, costs, lam_list, seed=0):
    import random as pyrandom
    rng = pyrandom.Random(seed)
    n_prompts = len(label_maps)
    out_costs, out_accs = [], []
    Y = np.zeros((len(lam_list), n_prompts), dtype=np.int32)
    for li, lam in enumerate(lam_list):
        weights = np.exp(-float(lam) * costs)
        probs = weights / weights.sum()
        tot_cost, tot_acc = 0.0, 0.0
        for i in range(n_prompts):
            j = rng.choices(range(len(models)), weights=probs, k=1)[0]
            acc = label_maps[i][j]
            cost = float(costs[j])
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
    print(f"Ceiling FP dim: {E.shape[1]}, models: {len(models)}", flush=True)

    print("Building GRPO targets + category track record...", flush=True)
    items, raw_cat_acc, eval_names = build_items_and_catacc(set_a, models, cols)
    print(f"{len(items)} usable Set A rows, {len(eval_names)} categories", flush=True)

    print(f"Loading frozen MiniLM ONCE (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    # Set B for final AUDC eval
    set_b = set_b.dropna(subset=cols + cost_cols)
    b_texts = set_b["prompt"].tolist()
    b_label_maps = set_b[cols].to_numpy(dtype=np.float32).tolist()  # list of length-11 lists
    b_costs = set_b[cost_cols].to_numpy(dtype=np.float32).mean(axis=0)  # avg cost per model across rows (approx, real per-row cost ignored for simplicity)
    print(f"Set B: {len(b_texts)} rows for final AUDC", flush=True)

    results = {}
    configs = [("vanilla-grpo", 0, plain_mse_loss), ("min-pos", 0, minpos_loss), ("catfilter-top2", 1, plain_mse_loss)]
    for tag, mask_idx, loss_fn in configs:
        print(f"\n{'='*50}\n{tag}\n{'='*50}", flush=True)
        enc = train(SEED, items, tokenizer, base_model, E_t, mask_idx, loss_fn, tag)

        embeds = np.zeros((len(b_texts), E.shape[1]), dtype=np.float32)
        for start in range(0, len(b_texts), 32):
            batch = b_texts[start:start + 32]
            embeds[start:start + len(batch)] = enc.encode(batch)
        E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        sims = embeds @ E_norm.T

        knn_costs, knn_accs, knn_Y = knn_curve(sims, models, b_label_maps, b_costs, LAM_LIST)
        rand_costs, rand_accs, rand_Y = random_curve(models, b_label_maps, b_costs, LAM_LIST)
        knn_metrics = audc_qnc_peak(knn_costs, knn_accs)
        rand_metrics = audc_qnc_peak(rand_costs, rand_accs)
        mean_delta, (lo, hi), p = paired_bootstrap_audc_cached(knn_costs, knn_Y, rand_costs, rand_Y, B=1000, seed=0)
        print(f"  [{tag}] AUDC={knn_metrics['audc']:.4f} Peak={knn_metrics['peak']:.4f} "
              f"random_AUDC={rand_metrics['audc']:.4f} delta={mean_delta:+.4f} p={p:.4g}", flush=True)
        results[tag] = {"knn": knn_metrics, "random": rand_metrics, "delta": mean_delta, "p": p}

    print("\n" + "=" * 70)
    print("ROUTERBENCH SEED-0 GENERALIZATION CHECK SUMMARY")
    print("=" * 70)
    for tag in results:
        print(f"{tag:>16}: AUDC={results[tag]['knn']['audc']:.4f}  vs random={results[tag]['random']['audc']:.4f}  "
              f"delta={results[tag]['delta']:+.4f} (p={results[tag]['p']:.3g})")

    import json
    out_path = Path("local_descriptors/routerbench-analysis/grpo_variants_seed0_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
