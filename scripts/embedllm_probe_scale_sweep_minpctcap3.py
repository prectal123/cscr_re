"""Probe-count sweep with the finalized method (min(pct=0.3,cap=3) + catfilter
pct=0.3), PCA-weighted allocation (floor=1, matching the 192-probe extreme
test so very low TARGET_TOTAL values are actually reachable), from very few
probes up to "V2-level" (full data, no probe selection, 80-dim uncompressed
anchor). Seed=0 only, per user's request -- this is a scaling-curve/plotting
asset, not a robustness check.

Unlike the old embedllm_probe_scale_sweep.py (floor=6, K=1, only scalar
AUDC saved), this: (a) uses the now-finalized min(pct=0.3,cap=3) loss,
(b) uses floor=1 so the sweep can start much lower than 1000, (c) saves the
FULL (cost, accuracy) deferral curve per point, not just the AUDC/Peak/QNC
scalars, so curves can be plotted together afterward.

Embeddings are cached ONCE for all sweep points (training texts and Set B
eval texts don't depend on which FP is used -- only the projection head's
target E_t changes per point), so the whole sweep is fast despite each
point being retrained.
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
V2_DIR = Path("local_descriptors/embedllm-ceiling")  # full-data, 80-dim, uncompressed anchor
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
HOLDOUT_FRAC = 0.15
SEED = 0
PCT_CATFILTER = 0.3
PCT_MINLOSS = 0.3
K_CAP = 3
K_PCA = 5
MIN_PROBES = 1
MAX_PROBES = 60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSCR_ALLSEEN = 0.541
EXCLUDE = {"JaeyeonKang__CCK_Asura_v1"}
LAM_LIST = np.logspace(-4, 2, 20)
K = 20
BANDIT_BETA = 0.000001
COST_GRID_POINTS = 20

TOTAL_SWEEP = [96, 192, 300, 500, 800, 1200, 1800, 2800, 4200]


def load_importance():
    d = json.load(open(ANALYSIS_DIR / "pca_category_importance.json", encoding="utf-8"))
    cats = d["ranked_categories"]
    imp = np.array(d["ranked_importance"])
    return {c: v for c, v in zip(cats, imp)}


def solve_allocation(importance_by_cat, categories, target_total, min_n, max_n):
    imp = np.array([importance_by_cat[c] for c in categories])
    sqrt_imp = np.sqrt(imp)

    def total_for_scale(scale):
        n = np.clip(np.round(sqrt_imp * scale), min_n, max_n)
        return n.sum(), n

    lo, hi = 0.0, 1e6
    for _ in range(60):
        mid = (lo + hi) / 2
        total, n = total_for_scale(mid)
        if total < target_total:
            lo = mid
        else:
            hi = mid
    total, n = total_for_scale(hi)
    return {c: int(v) for c, v in zip(categories, n)}, int(total)


def build_probe_sampled_fp(df, allocation, models, categories, out_dir):
    per_prompt = df.groupby("prompt_id").agg(category=("category", "first"), var=("label", "var")).reset_index()
    selected_ids = set()
    for cat, grp in per_prompt.groupby("category"):
        n = allocation.get(cat, MIN_PROBES)
        top = grp.nlargest(n, "var")
        selected_ids.update(top["prompt_id"].tolist())
    sub = df[df["prompt_id"].isin(selected_ids)]

    pivot = sub.pivot_table(index="model_name", columns="category", values="label", aggfunc="mean")
    pivot = pivot.reindex(index=models, columns=categories)
    raw = pivot.to_numpy()
    col_mean = np.nanmean(raw, axis=0, keepdims=True)
    raw = np.where(np.isnan(raw), col_mean, raw)
    pool_mean = raw.mean(axis=0, keepdims=True)
    centered = raw - pool_mean

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    reduced = centered @ Vt[:K_PCA].T
    E = reduced / (np.linalg.norm(reduced, axis=1, keepdims=True) + 1e-12)

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(models):
        np.save(out_dir / f"{m}.npy", E[i].astype(np.float32))
    return len(selected_ids)


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

        texts.append(text)
        targets.append(target)
        masks.append(keep_mask.astype(np.float32))
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
    audc = np.trapezoid(a_grid, grid) / (grid[-1] - grid[0])
    peak_idx = np.argmax(a)
    return {"audc": float(audc), "qnc": float(c[peak_idx]), "peak": float(a[peak_idx])}


def main():
    print("Loading EmbedLLM train.csv...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "category", "label", "prompt"])
    all_models = sorted(df["model_name"].unique())  # 112 -- FP construction uses this full list (matches
    # embedllm_build_pca_weighted_probe_fp_192.py's convention: PCA/pool-mean computed over all 112 models,
    # phantom model filtered out only afterward at training/eval time -- excluding it BEFORE the FP build
    # changes the pool-mean and PCA basis for the other 111 models too, which was the bug here)
    models = [m for m in all_models if m not in EXCLUDE]  # 111 -- used for training/eval/costs
    categories = sorted(df["category"].unique())
    category_to_idx = {c: i for i, c in enumerate(categories)}
    costs = np.array([compute_cost(m, 0, cost_type="n_params") for m in models], dtype=np.float32)
    print(f"{len(models)} models, {len(categories)} categories", flush=True)

    importance_by_cat = load_importance()
    raw_cat_acc = build_raw_category_accuracy(df, models, categories)

    print("Building GRPO targets + pct30 catfilter mask (shared across all sweep points)...", flush=True)
    texts, targets, masks = build_items(df, models, raw_cat_acc, category_to_idx, PCT_CATFILTER)
    print(f"{len(texts)} usable rows", flush=True)

    print(f"Loading frozen MiniLM base (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    hidden_size = base_model.config.hidden_size

    print("Precomputing frozen CLS embeddings ONCE (shared across all sweep points)...", flush=True)
    t0 = time.time()
    cls_all = precompute_cls(texts, tokenizer, base_model)
    print(f"  done in {time.time()-t0:.1f}s -> {cls_all.shape}", flush=True)

    dataset = load_embedllm("test", candidates=models)
    eval_texts = [ex["prompt"] for ex in dataset]
    label_maps = [ex["label_map"] for ex in dataset]
    print(f"Set B (test): {len(eval_texts)} rows", flush=True)
    t0 = time.time()
    cls_setB = precompute_cls(eval_texts, tokenizer, base_model)
    print(f"  cached in {time.time()-t0:.1f}s", flush=True)

    results = {}

    for target in TOTAL_SWEEP:
        print(f"\n{'#'*70}\nTARGET_TOTAL={target}\n{'#'*70}", flush=True)
        allocation, actual_total = solve_allocation(importance_by_cat, categories, target, MIN_PROBES, MAX_PROBES)
        fp_dir = Path(f"local_descriptors/embedllm-ceiling-scalesweep-minpctcap3-{target}-pca5")
        n_probes = build_probe_sampled_fp(df, allocation, all_models, categories, fp_dir)
        print(f"  built FP with {n_probes} actual probes -> {fp_dir}", flush=True)

        E = np.stack([np.load(fp_dir / f"{m}.npy") for m in models]).astype(np.float32)
        E_t = torch.from_numpy(E).float().to(DEVICE)
        E_t = E_t / (E_t.norm(dim=1, keepdim=True) + 1e-9)
        E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

        proj = train_fast(SEED, cls_all, targets, masks, E_t, hidden_size, f"scale{target}")

        with torch.no_grad():
            embeds = F.normalize(proj(torch.from_numpy(cls_setB).float().to(DEVICE)), dim=-1).cpu().numpy()
        sims = embeds @ E_norm.T
        knn_costs, knn_accs = knn_curve(sims, models, label_maps, costs, LAM_LIST)
        metrics = audc_qnc_peak(knn_costs, knn_accs)
        results[str(target)] = {
            "actual_probes": n_probes, "dim": int(E.shape[1]),
            "costs": knn_costs.tolist(), "accs": knn_accs.tolist(), **metrics,
        }
        print(f"  [target={target}, actual={n_probes}] AUDC={metrics['audc']:.4f} Peak={metrics['peak']:.4f} "
              f"({'BEATS' if metrics['audc'] > CSCR_ALLSEEN else 'below'} CSCR {CSCR_ALLSEEN})", flush=True)

    # V2 anchor: full data, 80-dim, uncompressed (existing FP, reused)
    print(f"\n{'#'*70}\nV2 anchor (full data, 80-dim uncompressed)\n{'#'*70}", flush=True)
    E2 = np.stack([np.load(V2_DIR / f"{m}.npy") for m in models]).astype(np.float32)
    E2_t = torch.from_numpy(E2).float().to(DEVICE)
    E2_t = E2_t / (E2_t.norm(dim=1, keepdim=True) + 1e-9)
    E2_norm = E2 / (np.linalg.norm(E2, axis=1, keepdims=True) + 1e-12)
    proj2 = train_fast(SEED, cls_all, targets, masks, E2_t, hidden_size, "V2-anchor")
    with torch.no_grad():
        embeds2 = F.normalize(proj2(torch.from_numpy(cls_setB).float().to(DEVICE)), dim=-1).cpu().numpy()
    sims2 = embeds2 @ E2_norm.T
    knn_costs2, knn_accs2 = knn_curve(sims2, models, label_maps, costs, LAM_LIST)
    metrics2 = audc_qnc_peak(knn_costs2, knn_accs2)
    results["V2"] = {"actual_probes": None, "dim": int(E2.shape[1]),
                      "costs": knn_costs2.tolist(), "accs": knn_accs2.tolist(), **metrics2}
    print(f"  [V2] AUDC={metrics2['audc']:.4f} Peak={metrics2['peak']:.4f}", flush=True)

    out_path = ANALYSIS_DIR / "probe_scale_sweep_minpctcap3_results.json"
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n" + "=" * 90)
    print("PROBE SCALE SWEEP (min(0.3,3), floor=1, seed0, all-seen)")
    print("=" * 90)
    for target in TOTAL_SWEEP:
        r = results[str(target)]
        print(f"  target={target:>5} actual={r['actual_probes']:>5}: AUDC={r['audc']:.4f}")
    print(f"  V2 (full data): AUDC={results['V2']['audc']:.4f}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
