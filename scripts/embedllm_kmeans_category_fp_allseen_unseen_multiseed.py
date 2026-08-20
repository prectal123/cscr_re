"""Adaptive-clustering FP: what if the dataset didn't come with a
`category` column at all? Everywhere this project's official pipeline used
EmbedLLM's given 80 categories (probe allocation, probe selection,
catfilter track-record, Ceiling FP construction), replace them with
K-Means clusters (K=80, matching the real category count for a controlled
comparison) computed on frozen-MiniLM query embeddings.

Tests generalizability to the realistic case where category labels aren't
supplied. Prediction going in (based on the category-shuffle negative
control, PROGRESS.md 26.2/Final_Result_Summary2.MD 1.11): fine-grained
category SEMANTICS didn't matter for Unseen (shuffled-but-real-data FP
barely moved AUDC), so self-discovered (imperfect, data-driven) clusters
should work almost as well as the human-labeled categories -- what matters
is having *some* consistent partition to average within, not that the
partition is semantically "correct".

Implementation: overwrite df["category"] with KMeans cluster labels BEFORE
running the exact same functions the official 1800-probe headline pipeline
uses (solve_allocation_uniform / build_probe_sampled_fp / build_items) --
reuses proven code paths unchanged, only the category SOURCE differs.
K-Means itself uses a different random_state per seed (so cluster
assignment varies across seeds too, not just MLP init).
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
from sklearn.cluster import KMeans

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
TARGET_TOTAL = 1800  # matches the official headline probe budget
N_CLUSTERS = 80  # matches the real category count, for a controlled comparison
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
        if total_for_quota(mid) < target_total:
            lo = mid
        else:
            hi = mid
    quota = hi
    alloc = {c: max(min_n, int(round(min(quota, cat_sizes[c])))) for c in categories}
    return alloc, sum(alloc.values())


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
    E = centered / (np.linalg.norm(centered, axis=1, keepdims=True) + 1e-12)
    return E.astype(np.float32), len(selected_ids)


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


def run_point(tag, df, models, categories, category_to_idx, cat_sizes, tokenizer, base_model, hidden_size,
              seed, eval_candidates, eval_dataset_split_name):
    raw_cat_acc = build_raw_category_accuracy(df, models, categories)
    texts, targets, masks = build_items(df, models, raw_cat_acc, category_to_idx, PCT_CATFILTER)
    cls_all = precompute_cls(texts, tokenizer, base_model)

    allocation, actual_total = solve_allocation_uniform(cat_sizes, categories, TARGET_TOTAL, MIN_PROBES)
    E, n_probes = build_probe_sampled_fp(df, allocation, models, categories, None)
    E_t = torch.from_numpy(E).float().to(DEVICE)
    E_t = E_t / (E_t.norm(dim=1, keepdim=True) + 1e-9)
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    proj = train_fast(seed, cls_all, targets, masks, E_t, hidden_size, tag)

    dataset = load_embedllm(eval_dataset_split_name, candidates=eval_candidates)
    eval_texts = [ex["prompt"] for ex in dataset]
    label_maps = [ex["label_map"] for ex in dataset]
    cls_setB = precompute_cls(eval_texts, tokenizer, base_model)
    costs = np.array([compute_cost(m, 0, cost_type="n_params") for m in eval_candidates], dtype=np.float32)

    E_eval = E_norm if eval_candidates == models else None
    if E_eval is None:
        # unseen: E was built over `models` == seen only; caller passes seen models here for FP build,
        # eval_candidates == unseen models needs its own FP slice -- handled by caller instead.
        raise ValueError("eval_candidates must equal FP models in this helper; unseen path builds its own FP")

    with torch.no_grad():
        embeds = F.normalize(proj(torch.from_numpy(cls_setB).float().to(DEVICE)), dim=-1).cpu().numpy()
    sims = embeds @ E_norm.T
    knn_costs, knn_accs = knn_curve(sims, eval_candidates, label_maps, costs, LAM_LIST)
    return audc_qnc_peak(knn_costs, knn_accs), proj, E


def main():
    print("Loading EmbedLLM train.csv...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df_orig = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "category", "label", "prompt"])
    all_models = sorted([m for m in df_orig["model_name"].unique() if m not in EXCLUDE])
    print(f"{len(all_models)} models, {df_orig['category'].nunique()} REAL categories (will be discarded)", flush=True)

    print(f"Loading frozen MiniLM base (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    hidden_size = base_model.config.hidden_size

    uniq = df_orig.drop_duplicates("prompt_id")[["prompt_id", "prompt"]].reset_index(drop=True)
    print(f"Precomputing frozen CLS embeddings ONCE for {len(uniq)} unique prompts (for KMeans + reuse)...", flush=True)
    t0 = time.time()
    cls_uniq = precompute_cls(uniq["prompt"].tolist(), tokenizer, base_model)
    print(f"  done in {time.time()-t0:.1f}s -> {cls_uniq.shape}", flush=True)
    pid_to_emb = {pid: cls_uniq[i] for i, pid in enumerate(uniq["prompt_id"])}

    results = {"allseen": [], "unseen": []}

    for seed in SEEDS:
        print(f"\n{'#'*70}\nseed={seed}: running KMeans(k={N_CLUSTERS})...\n{'#'*70}", flush=True)
        t0 = time.time()
        km = KMeans(n_clusters=N_CLUSTERS, random_state=seed, n_init=10)
        cluster_labels = km.fit_predict(cls_uniq)
        print(f"  KMeans done in {time.time()-t0:.1f}s, inertia={km.inertia_:.1f}", flush=True)
        pid_to_cluster = {pid: f"cluster_{cluster_labels[i]:02d}" for i, pid in enumerate(uniq["prompt_id"])}

        df = df_orig.copy()
        df["category"] = df["prompt_id"].map(pid_to_cluster)
        categories = sorted(df["category"].unique())
        category_to_idx = {c: i for i, c in enumerate(categories)}
        cat_sizes = df.drop_duplicates("prompt_id").groupby("category").size().to_dict()
        print(f"  {len(categories)} pseudo-categories, sizes min={min(cat_sizes.values())} max={max(cat_sizes.values())}",
              flush=True)

        # ---------- ALL-SEEN ----------
        m_as, proj_as, E_as = run_point(f"allseen-seed{seed}", df, all_models, categories, category_to_idx,
                                         cat_sizes, tokenizer, base_model, hidden_size, seed, all_models, "test")
        results["allseen"].append({"seed": seed, **m_as})
        print(f"  [allseen seed={seed}] AUDC={m_as['audc']:.4f} Peak={m_as['peak']:.4f} "
              f"({'BEATS' if m_as['audc'] > CSCR_ALLSEEN else 'below'} CSCR {CSCR_ALLSEEN})", flush=True)

        # ---------- UNSEEN ----------
        split_path = ANALYSIS_DIR / ("newllm_split.json" if seed == 0 else f"newllm_split_seed{seed}.json")
        split = json.load(open(split_path, encoding="utf-8"))
        seen_models, unseen_models = split["seen"], split["unseen"]
        df_seen = df[df["model_name"].isin(seen_models)]
        raw_cat_acc = build_raw_category_accuracy(df_seen, seen_models, categories)
        texts, targets, masks = build_items(df_seen, seen_models, raw_cat_acc, category_to_idx, PCT_CATFILTER)
        cls_all = precompute_cls(texts, tokenizer, base_model)

        allocation, actual_total = solve_allocation_uniform(cat_sizes, categories, TARGET_TOTAL, MIN_PROBES)
        E_seen, n_probes = build_probe_sampled_fp(df_seen, allocation, seen_models, categories, None)
        E_seen_t = torch.from_numpy(E_seen).float().to(DEVICE)
        E_seen_t = E_seen_t / (E_seen_t.norm(dim=1, keepdim=True) + 1e-9)

        proj_u = train_fast(seed, cls_all, targets, masks, E_seen_t, hidden_size, f"unseen-seed{seed}")

        # unseen models' FP -- build on the FULL df (with pseudo-categories) restricted to unseen models,
        # using the SAME fixed probe allocation logic (selection re-run on unseen-only variance, matching
        # how the official headline Unseen pipeline builds unseen FP -- consistent with other scripts)
        E_unseen, _ = build_probe_sampled_fp(df, allocation, unseen_models, categories, None)
        E_unseen_norm = E_unseen / (np.linalg.norm(E_unseen, axis=1, keepdims=True) + 1e-12)

        dataset = load_embedllm("test", candidates=unseen_models)
        eval_texts = [ex["prompt"] for ex in dataset]
        label_maps = [ex["label_map"] for ex in dataset]
        cls_setB = precompute_cls(eval_texts, tokenizer, base_model)
        costs_u = np.array([compute_cost(m, 0, cost_type="n_params") for m in unseen_models], dtype=np.float32)

        with torch.no_grad():
            embeds = F.normalize(proj_u(torch.from_numpy(cls_setB).float().to(DEVICE)), dim=-1).cpu().numpy()
        sims = embeds @ E_unseen_norm.T
        knn_costs, knn_accs = knn_curve(sims, unseen_models, label_maps, costs_u, LAM_LIST)
        m_u = audc_qnc_peak(knn_costs, knn_accs)
        results["unseen"].append({"seed": seed, "n_seen": len(seen_models), "n_unseen": len(unseen_models), **m_u})
        print(f"  [unseen seed={seed}] AUDC={m_u['audc']:.4f} Peak={m_u['peak']:.4f} "
              f"({'BEATS' if m_u['audc'] > CSCR_UNSEEN else 'below'} CSCR-unseen {CSCR_UNSEEN})", flush=True)

        out_path = ANALYSIS_DIR / "kmeans_category_fp_allseen_unseen_results.json"
        json.dump(results, open(out_path, "w"), indent=2)

    print("\n" + "=" * 90)
    print(f"K-MEANS PSEUDO-CATEGORY FP (K={N_CLUSTERS}) vs REAL-CATEGORY HEADLINE vs CSCR")
    print("=" * 90)
    as_audcs = [r["audc"] for r in results["allseen"]]
    un_audcs = [r["audc"] for r in results["unseen"]]
    print(f"  All-seen: {' / '.join(f'{a:.4f}' for a in as_audcs)}  mean={np.mean(as_audcs):.4f} (std={np.std(as_audcs):.4f})  "
          f"[real-category headline=0.5787, CSCR={CSCR_ALLSEEN}]")
    print(f"  Unseen:   {' / '.join(f'{a:.4f}' for a in un_audcs)}  mean={np.mean(un_audcs):.4f} (std={np.std(un_audcs):.4f})  "
          f"[real-category headline=0.5232, CSCR={CSCR_UNSEEN}]")
    print(f"\nSaved -> {ANALYSIS_DIR / 'kmeans_category_fp_allseen_unseen_results.json'}")


if __name__ == "__main__":
    main()
