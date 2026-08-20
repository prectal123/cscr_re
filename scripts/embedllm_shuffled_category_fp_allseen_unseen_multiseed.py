"""Negative control: build the 1800-probe FP with the SAME 80-slot
structure, SAME real probe data (real per-model correctness), SAME
per-slot probe counts, and the SAME probe-to-slot assignment shared
across all models -- but the probe-to-slot assignment is a RANDOM
shuffle, not each probe's true category. So slot #5 might now contain a
mix of real gpqa/math/coding probes instead of coherently one topic.

This isolates: does it matter that a "category" dimension is
SEMANTICALLY coherent (one real topic), or is it enough that there are
80 real, internally-consistent (same probes for every model), non-noise
slots? Unlike the earlier pure-random-noise-vector test (zero real data
at all), this keeps 100% real correctness data -- only the CATEGORY
LABELING of that data is scrambled.

Training loop is untouched (loss doesn't need to know what a given FP
dimension "means" -- it only needs consistent per-model target vectors)
-- only the FP-building step's category assignment changes. Same
min(0.3,3)+catfilter recipe, All-seen + Unseen, 3 seeds each.
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
PCT_CATFILTER = 0.3
PCT_MINLOSS = 0.3
K_CAP = 3
MIN_PROBES = 1
TARGET_TOTAL = 1800
SHUFFLE_SEED = 999  # fixed -- one shuffle, shared across all models
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSCR_ALLSEEN = 0.541
CSCR_UNSEEN = 0.4848
EXCLUDE = {"JaeyeonKang__CCK_Asura_v1"}
LAM_LIST = np.logspace(-4, 2, 20)
K = 20
BANDIT_BETA = 0.000001
COST_GRID_POINTS = 20


def solve_allocation_uniform(cat_sizes, categories, target_total, min_n):
    lo, hi = 0.0, float(max(cat_sizes[c] for c in categories))
    def total_for_quota(q):
        return sum(min(q, cat_sizes[c]) for c in categories)
    for _ in range(60):
        mid = (lo + hi) / 2
        if total_for_quota(mid) < target_total: lo = mid
        else: hi = mid
    quota = hi
    alloc = {c: max(min_n, int(round(min(quota, cat_sizes[c])))) for c in categories}
    return alloc, sum(alloc.values())


def build_shuffled_fp(df, allocation, all_models, categories):
    """Select probes per TRUE category (same as official pipeline), but
    then randomly reshuffle which of the 80 slots each selected probe's
    data lands in -- same slot SIZES, random slot CONTENTS, shared
    across all models."""
    per_prompt = df.groupby("prompt_id").agg(category=("category", "first"), var=("label", "var")).reset_index()
    selected_by_cat = {}
    for cat, grp in per_prompt.groupby("category"):
        n = allocation.get(cat, MIN_PROBES)
        top = grp.nlargest(n, "var")
        selected_by_cat[cat] = top["prompt_id"].tolist()

    all_selected_ids = [pid for cat in categories for pid in selected_by_cat.get(cat, [])]
    slot_sizes = [len(selected_by_cat.get(cat, [])) for cat in categories]

    rng = np.random.RandomState(SHUFFLE_SEED)
    shuffled_ids = list(all_selected_ids)
    rng.shuffle(shuffled_ids)
    slot_assignment = {}  # slot_idx -> list of prompt_ids (randomly reassigned)
    ptr = 0
    for slot_idx, size in enumerate(slot_sizes):
        slot_assignment[slot_idx] = shuffled_ids[ptr:ptr + size]
        ptr += size

    sub = df[df["prompt_id"].isin(all_selected_ids)]
    pivot = sub.pivot_table(index="model_name", columns="prompt_id", values="label", aggfunc="mean")
    pivot = pivot.reindex(index=all_models)

    n_models = len(all_models)
    raw = np.zeros((n_models, len(categories)))
    for slot_idx in range(len(categories)):
        pids = slot_assignment[slot_idx]
        if not pids:
            continue
        vals = pivot[pids].to_numpy(dtype=float)  # (n_models, n_pids)
        col_mean = np.nanmean(vals, axis=0, keepdims=True)
        vals = np.where(np.isnan(vals), col_mean, vals)
        raw[:, slot_idx] = vals.mean(axis=1)

    pool_mean = raw.mean(axis=0, keepdims=True)
    centered = raw - pool_mean
    E = centered / (np.linalg.norm(centered, axis=1, keepdims=True) + 1e-12)
    return {m: E[i] for i, m in enumerate(all_models)}, len(all_selected_ids)


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
        if mask.sum() < 2: continue
        vals = labels[mask]
        mean, std = vals.mean(), vals.std()
        if std < 1e-6: continue
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
        texts.append(text); targets.append(target); masks.append(keep_mask.astype(np.float32))
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
    tgt_ho = targets[ho_idx]; msk_ho = masks[ho_idx]
    torch.manual_seed(seed)
    proj = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=False), nn.ReLU(),
                          nn.Linear(hidden_size, E_t.size(1), bias=False)).to(DEVICE)
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
            loss.backward(); opt.step(); opt.zero_grad()
        with torch.no_grad():
            q_ho = F.normalize(proj(cls_ho), dim=-1)
            cos_ho = (q_ho @ E_t.T).cpu().numpy()
        rhos = []
        for i in range(cos_ho.shape[0]):
            m = msk_ho[i].astype(bool)
            if m.sum() < 3: continue
            rho, _ = spearmanr(cos_ho[i, m], tgt_ho[i, m])
            if not np.isnan(rho): rhos.append(rho)
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
                if score > best_score: best_score, best_j = score, j
            chosen = models[best_j]
            acc = 1.0 if label_maps[i].get(chosen, 0) == 1 else 0.0
            cost = float(costs[best_j])
            bandit.update(chosen, accuracy=acc, cost=cost)
            tot_cost += cost; tot_acc += acc
        out_costs.append(tot_cost / n_prompts); out_accs.append(tot_acc / n_prompts)
    return np.array(out_costs), np.array(out_accs)


def audc_qnc_peak(costs, accs):
    order = np.argsort(costs)
    c, a = costs[order], accs[order]
    grid = build_cost_grid(c, N_grid=COST_GRID_POINTS)
    a_grid = interp_to_grid(c, a, grid)
    audc = np.trapezoid(a_grid, grid) / (grid[-1] - grid[0])
    peak_idx = np.argmax(a)
    return {"audc": float(audc), "qnc": float(c[peak_idx]), "peak": float(a[peak_idx])}


def main():
    print("Loading EmbedLLM train.csv...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "category", "label", "prompt"])
    all_models = sorted(df["model_name"].unique())
    models = [m for m in all_models if m not in EXCLUDE]
    categories = sorted(df["category"].unique())
    category_to_idx = {c: i for i, c in enumerate(categories)}
    costs = np.array([compute_cost(m, 0, cost_type="n_params") for m in models], dtype=np.float32)
    cat_sizes = df.drop_duplicates("prompt_id").groupby("category").size().to_dict()

    allocation, actual_total = solve_allocation_uniform(cat_sizes, categories, TARGET_TOTAL, MIN_PROBES)
    E_by_model, n_probes = build_shuffled_fp(df, allocation, all_models, categories)
    print(f"Built SHUFFLED-category FP: {n_probes} real probes, 80 slots, category identity scrambled "
          f"(shuffle_seed={SHUFFLE_SEED})", flush=True)

    raw_cat_acc = build_raw_category_accuracy(df, models, categories)
    print("Building GRPO targets + pct30 catfilter mask...", flush=True)
    texts, targets, masks = build_items(df, models, raw_cat_acc, category_to_idx, PCT_CATFILTER)
    print(f"{len(texts)} usable rows", flush=True)

    print(f"Loading frozen MiniLM base (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters(): p.requires_grad = False
    hidden_size = base_model.config.hidden_size

    print("Precomputing frozen CLS embeddings ONCE...", flush=True)
    t0 = time.time()
    cls_all = precompute_cls(texts, tokenizer, base_model)
    print(f"  done in {time.time()-t0:.1f}s -> {cls_all.shape}", flush=True)

    dataset = load_embedllm("test", candidates=models)
    eval_texts = [ex["prompt"] for ex in dataset]
    label_maps = [ex["label_map"] for ex in dataset]
    t0 = time.time()
    cls_setB = precompute_cls(eval_texts, tokenizer, base_model)
    print(f"Set B (test): {len(eval_texts)} rows cached in {time.time()-t0:.1f}s", flush=True)

    results = {"allseen": [], "unseen": []}

    # ---------- ALL-SEEN ----------
    print(f"\n{'#'*70}\nALL-SEEN (shuffled-category FP)\n{'#'*70}", flush=True)
    E = np.stack([E_by_model[m] for m in models]).astype(np.float32)
    E_t = torch.from_numpy(E).float().to(DEVICE)
    E_t = E_t / (E_t.norm(dim=1, keepdim=True) + 1e-9)
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    for seed in SEEDS:
        proj = train_fast(seed, cls_all, targets, masks, E_t, hidden_size, f"allseen-seed{seed}")
        with torch.no_grad():
            embeds = F.normalize(proj(torch.from_numpy(cls_setB).float().to(DEVICE)), dim=-1).cpu().numpy()
        sims = embeds @ E_norm.T
        knn_costs, knn_accs = knn_curve(sims, models, label_maps, costs, LAM_LIST)
        m = audc_qnc_peak(knn_costs, knn_accs)
        results["allseen"].append({"seed": seed, **m})
        print(f"  [allseen seed={seed}] AUDC={m['audc']:.4f} Peak={m['peak']:.4f} "
              f"({'BEATS' if m['audc'] > CSCR_ALLSEEN else 'below'} CSCR {CSCR_ALLSEEN})", flush=True)
    audcs = [r["audc"] for r in results["allseen"]]
    print(f"  [allseen] MEAN AUDC={np.mean(audcs):.4f} (std={np.std(audcs):.4f})", flush=True)

    # ---------- UNSEEN ----------
    print(f"\n{'#'*70}\nUNSEEN (shuffled-category FP)\n{'#'*70}", flush=True)
    for seed in SEEDS:
        split_path = ANALYSIS_DIR / ("newllm_split.json" if seed == 0 else f"newllm_split_seed{seed}.json")
        split = json.load(open(split_path, encoding="utf-8"))
        seen_models, unseen_models = split["seen"], split["unseen"]

        raw_cat_acc_seen = build_raw_category_accuracy(df, seen_models, categories)
        texts_s, targets_s, masks_s = build_items(df, seen_models, raw_cat_acc_seen, category_to_idx, PCT_CATFILTER)
        cls_all_s = precompute_cls(texts_s, tokenizer, base_model)

        E_seen = np.stack([E_by_model[m] for m in seen_models]).astype(np.float32)
        E_seen_t = torch.from_numpy(E_seen).float().to(DEVICE)
        E_seen_t = E_seen_t / (E_seen_t.norm(dim=1, keepdim=True) + 1e-9)

        proj = train_fast(seed, cls_all_s, targets_s, masks_s, E_seen_t, hidden_size, f"unseen-seed{seed}")

        dataset_u = load_embedllm("test", candidates=unseen_models)
        eval_texts_u = [ex["prompt"] for ex in dataset_u]
        label_maps_u = [ex["label_map"] for ex in dataset_u]
        cls_setB_u = precompute_cls(eval_texts_u, tokenizer, base_model)

        E_unseen = np.stack([E_by_model[m] for m in unseen_models]).astype(np.float32)
        E_unseen_norm = E_unseen / (np.linalg.norm(E_unseen, axis=1, keepdims=True) + 1e-12)
        costs_unseen = np.array([compute_cost(m, 0, cost_type="n_params") for m in unseen_models], dtype=np.float32)

        with torch.no_grad():
            embeds_u = F.normalize(proj(torch.from_numpy(cls_setB_u).float().to(DEVICE)), dim=-1).cpu().numpy()
        sims_u = embeds_u @ E_unseen_norm.T
        knn_costs_u, knn_accs_u = knn_curve(sims_u, unseen_models, label_maps_u, costs_unseen, LAM_LIST)
        m = audc_qnc_peak(knn_costs_u, knn_accs_u)
        results["unseen"].append({"seed": seed, **m})
        print(f"  [unseen seed={seed}] AUDC={m['audc']:.4f} Peak={m['peak']:.4f} "
              f"({'BEATS' if m['audc'] > CSCR_UNSEEN else 'below'} CSCR-unseen {CSCR_UNSEEN})", flush=True)
    audcs_u = [r["audc"] for r in results["unseen"]]
    print(f"  [unseen] MEAN AUDC={np.mean(audcs_u):.4f} (std={np.std(audcs_u):.4f})", flush=True)

    out_path = ANALYSIS_DIR / "shuffled_category_fp_allseen_unseen_results.json"
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n" + "=" * 90)
    print("SHUFFLED-CATEGORY FP (real data, real 80 slots, scrambled category identity)")
    print("=" * 90)
    print(f"  All-seen: mean AUDC={np.mean(audcs):.4f} (std={np.std(audcs):.4f}) -- real FP was 0.5787")
    print(f"  Unseen:   mean AUDC={np.mean(audcs_u):.4f} (std={np.std(audcs_u):.4f}) -- real FP was 0.5232, noise FP was 0.3721")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
