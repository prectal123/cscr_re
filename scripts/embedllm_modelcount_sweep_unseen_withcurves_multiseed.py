"""Model-pool-size scalability sweep, UNSEEN protocol, with full deferral
curves saved (costs_mean_curve/accs_mean_curve per point) for plotting.

Same fixed-probe design as embedllm_modelcount_sweep_fixedprobes_allseen_multiseed.py
(probe selection done ONCE via full-111-model variance, only which models
get FP vectors varies) -- extended with an Unseen split: for each
(model_count, seed), sample N models, split them into seen (2/3) /
unseen (1/3) via the same UNSEEN_FRAC convention used everywhere else in
this project, train on seen only, evaluate on unseen only.

3 seeds (not 8) to keep runtime reasonable for plotting purposes.
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
from router.embedllm import load_embedllm
from router.bandit import BanditStats
from router.cost_models import compute_cost
from run_audc_eval import interp_to_grid, build_cost_grid

ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
HOLDOUT_FRAC = 0.15
SEEDS = [0, 1, 2]
UNSEEN_FRAC = 1 / 3
PCT_CATFILTER = 0.3
PCT_MINLOSS = 0.3
K_CAP = 3
MIN_PROBES = 1
TARGET_TOTAL = 1800
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSCR_UNSEEN = 0.4848
EXCLUDE = {"JaeyeonKang__CCK_Asura_v1"}
LAM_LIST = np.logspace(-4, 2, 20)
K = 20
BANDIT_BETA = 0.000001
COST_GRID_POINTS = 20

MODEL_COUNTS = [15, 30, 50, 75, 111]  # skip 5/10: too few models left after a 1/3 unseen split


def solve_allocation_uniform(cat_sizes, categories, target_total, min_n):
    lo, hi = 0.0, float(max(cat_sizes[c] for c in categories))

    def total_for_quota(q):
        return sum(min(q, cat_sizes[c]) for c in categories)

    for _ in range(60):
        mid = (lo + hi) / 2
        if total_for_quota(mid) < target_total:
            lo = mid
        else:
            hi = mid
    quota = hi
    alloc = {c: max(min_n, int(round(min(quota, cat_sizes[c])))) for c in categories}
    return alloc, sum(alloc.values())


def select_probes_fixed(df_full, allocation, categories):
    per_prompt = df_full.groupby("prompt_id").agg(category=("category", "first"), var=("label", "var")).reset_index()
    selected_ids = set()
    for cat, grp in per_prompt.groupby("category"):
        n = allocation.get(cat, MIN_PROBES)
        top = grp.nlargest(n, "var")
        selected_ids.update(top["prompt_id"].tolist())
    return selected_ids


def build_fp_from_fixed_probes(df_full, selected_ids, models, categories):
    sub = df_full[df_full["prompt_id"].isin(selected_ids) & df_full["model_name"].isin(models)]
    pivot = sub.pivot_table(index="model_name", columns="category", values="label", aggfunc="mean")
    pivot = pivot.reindex(index=models, columns=categories)
    raw = pivot.to_numpy()
    col_mean = np.nanmean(raw, axis=0, keepdims=True)
    raw = np.where(np.isnan(raw), col_mean, raw)
    pool_mean = raw.mean(axis=0, keepdims=True)
    centered = raw - pool_mean
    E = centered / (np.linalg.norm(centered, axis=1, keepdims=True) + 1e-12)
    return E.astype(np.float32)


def build_raw_category_accuracy(df, models, categories):
    pivot = df.pivot_table(index="model_name", columns="category", values="label", aggfunc="mean")
    pivot = pivot.reindex(index=models, columns=categories)
    return pivot.to_numpy()


def build_items(df, models, raw_cat_acc, category_to_idx, pct):
    name_to_idx = {n: i for i, n in enumerate(models)}
    prompt_ids, targets, masks = [], [], []
    for pid, grp in df.groupby("prompt_id", sort=False):
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

        prompt_ids.append(pid)
        targets.append(target)
        masks.append(keep_mask.astype(np.float32))
    return prompt_ids, np.stack(targets), np.stack(masks)


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


def train_fast(seed, cls_all, targets, masks, E_t, hidden_size, tag):
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
    print(f"  [{tag}] best_epoch={best_epoch} best_holdout_rho={best_rho:.4f} time={time.time()-t0:.1f}s", flush=True)
    return proj


def knn_curve(sims, models, label_maps, costs, lam_list, k=K, bandit_beta=BANDIT_BETA):
    n_prompts, n_models = sims.shape
    order = np.argsort(-sims, axis=1)
    topk = order[:, :min(k, n_models)]
    out_costs, out_accs = [], []
    for lam in lam_list:
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
        out_costs.append(tot_cost / n_prompts)
        out_accs.append(tot_acc / n_prompts)
    return np.array(out_costs), np.array(out_accs)


def audc_qnc_peak(costs, accs):
    order = np.argsort(costs)
    c, a = costs[order], accs[order]
    grid = build_cost_grid(c, N_grid=COST_GRID_POINTS)
    a_grid = interp_to_grid(c, a, grid)
    denom = grid[-1] - grid[0]
    if not np.isfinite(denom) or denom <= 1e-12 or not np.all(np.isfinite(a_grid)):
        return {"audc": float("nan"), "qnc": float(c[np.argmax(a)]), "peak": float(a.max())}
    audc = np.trapezoid(a_grid, grid) / denom
    peak_idx = np.argmax(a)
    return {"audc": float(audc), "qnc": float(c[peak_idx]), "peak": float(a[peak_idx])}


def main():
    print("Loading EmbedLLM train.csv...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "category", "label", "prompt"])
    all_models = sorted([m for m in df["model_name"].unique() if m not in EXCLUDE])
    categories = sorted(df["category"].unique())
    category_to_idx = {c: i for i, c in enumerate(categories)}
    cat_sizes = df.drop_duplicates("prompt_id").groupby("category").size().to_dict()
    print(f"{len(all_models)} candidate models, {len(categories)} categories", flush=True)

    allocation, actual_total = solve_allocation_uniform(cat_sizes, categories, TARGET_TOTAL, MIN_PROBES)
    selected_probe_ids = select_probes_fixed(df, allocation, categories)
    print(f"Fixed probe set selected ONCE (full-111 variance): {len(selected_probe_ids)} probes", flush=True)

    print(f"Loading frozen MiniLM base (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    hidden_size = base_model.config.hidden_size

    print("Precomputing frozen CLS embeddings ONCE for every unique train prompt...", flush=True)
    uniq = df.drop_duplicates("prompt_id")[["prompt_id", "prompt"]].reset_index(drop=True)
    pid_to_row = {pid: i for i, pid in enumerate(uniq["prompt_id"])}
    t0 = time.time()
    cls_cache = precompute_cls(uniq["prompt"].tolist(), tokenizer, base_model)
    print(f"  done in {time.time()-t0:.1f}s -> {cls_cache.shape} ({len(uniq)} unique prompts)", flush=True)

    results = {}
    for n_models in MODEL_COUNTS:
        results[str(n_models)] = {"per_seed": [], "costs_mean_curve": None, "accs_mean_curve": None}
        seed_costs, seed_accs = [], []
        for seed in SEEDS:
            rng = np.random.RandomState(seed * 1000 + n_models)
            if n_models >= len(all_models):
                pool = all_models
            else:
                pool = sorted(rng.choice(all_models, size=n_models, replace=False).tolist())

            perm = rng.permutation(len(pool))
            n_unseen = max(1, int(round(len(pool) * UNSEEN_FRAC)))
            unseen = [pool[i] for i in perm[:n_unseen]]
            seen = [pool[i] for i in perm[n_unseen:]]
            print(f"\n{'#'*70}\nn_models={n_models} seed={seed} (seen={len(seen)} unseen={len(unseen)})\n{'#'*70}", flush=True)

            df_seen = df[df["model_name"].isin(seen)]
            raw_cat_acc = build_raw_category_accuracy(df_seen, seen, categories)
            prompt_ids, targets, masks = build_items(df_seen, seen, raw_cat_acc, category_to_idx, PCT_CATFILTER)
            row_idx = np.array([pid_to_row[pid] for pid in prompt_ids])
            cls_all = cls_cache[row_idx]
            print(f"  {len(prompt_ids)} usable seen rows", flush=True)

            E_seen = build_fp_from_fixed_probes(df, selected_probe_ids, seen, categories)
            E_seen_t = torch.from_numpy(E_seen).float().to(DEVICE)
            E_seen_t = E_seen_t / (E_seen_t.norm(dim=1, keepdim=True) + 1e-9)

            proj = train_fast(seed, cls_all, targets, masks, E_seen_t, hidden_size, f"n{n_models}-seed{seed}")

            dataset = load_embedllm("test", candidates=unseen)
            eval_texts = [ex["prompt"] for ex in dataset]
            label_maps = [ex["label_map"] for ex in dataset]
            cls_setB = precompute_cls(eval_texts, tokenizer, base_model)
            costs = np.array([compute_cost(m, 0, cost_type="n_params") for m in unseen], dtype=np.float32)

            E_unseen = build_fp_from_fixed_probes(df, selected_probe_ids, unseen, categories)
            E_unseen_norm = E_unseen / (np.linalg.norm(E_unseen, axis=1, keepdims=True) + 1e-12)

            with torch.no_grad():
                embeds = F.normalize(proj(torch.from_numpy(cls_setB).float().to(DEVICE)), dim=-1).cpu().numpy()
            sims = embeds @ E_unseen_norm.T
            knn_costs, knn_accs = knn_curve(sims, unseen, label_maps, costs, LAM_LIST)
            m = audc_qnc_peak(knn_costs, knn_accs)
            print(f"  [n_models={n_models} seed={seed}] AUDC={m['audc']:.4f} Peak={m['peak']:.4f} "
                  f"({'BEATS' if m['audc'] > CSCR_UNSEEN else 'below'} CSCR-unseen {CSCR_UNSEEN})", flush=True)
            results[str(n_models)]["per_seed"].append({"seed": seed, "n_seen": len(seen), "n_unseen": len(unseen), **m})
            seed_costs.append(knn_costs)
            seed_accs.append(knn_accs)

        mean_costs = np.mean(seed_costs, axis=0)
        mean_accs = np.mean(seed_accs, axis=0)
        results[str(n_models)]["costs_mean_curve"] = mean_costs.tolist()
        results[str(n_models)]["accs_mean_curve"] = mean_accs.tolist()
        audcs = [r["audc"] for r in results[str(n_models)]["per_seed"] if np.isfinite(r["audc"])]
        print(f"\n[n_models={n_models}] MEAN AUDC={np.mean(audcs):.4f} (std={np.std(audcs):.4f})", flush=True)

        out_path = ANALYSIS_DIR / "modelcount_sweep_unseen_withcurves_results.json"
        json.dump(results, open(out_path, "w"), indent=2)

    print("\n" + "=" * 90)
    print("MODEL-POOL-SIZE SWEEP (UNSEEN, fixed probes, with curves)")
    print("=" * 90)
    for n_models in MODEL_COUNTS:
        audcs = [r["audc"] for r in results[str(n_models)]["per_seed"] if np.isfinite(r["audc"])]
        print(f"  n_models={n_models:>4d}: AUDC={np.mean(audcs):.4f} (std={np.std(audcs):.4f})")
    print(f"\nSaved -> {ANALYSIS_DIR / 'modelcount_sweep_unseen_withcurves_results.json'}")


if __name__ == "__main__":
    main()
