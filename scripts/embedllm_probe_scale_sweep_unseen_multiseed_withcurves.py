"""UNSEEN-protocol version of the probe-count sweep, requested after the
all-seen sweep looked suspiciously flat/high even at very low probe counts
(80 probes -> AUDC=0.58, barely below V2's full-data 0.5867). User's
concern: with 111 known (seen) models and rich per-model FP vectors, a
projection head could route well on all-seen just by learning a
"category -> which of the 111 KNOWN models is good" lookup, without any
real generalizable text-to-capability mapping -- which wouldn't need much
probe data OR much real signal, since the target models are literally
fixed and already known. That would explain the flatness (a lookup table
needs little data), and predicts UNSEEN accuracy should look very
different: usually lower, and possibly more probe-count-dependent, since
a genuinely novel model has to be judged by proximity to seen models'
patterns rather than looked up directly.

Protocol: standard "new LLM" unseen split (71 seen / 35 unseen models,
`newllm_split*.json`). Projection head is trained using ONLY seen models'
rows and seen models' FP vectors (never sees unseen models' labels/FP
during training). Evaluated on Set B prompts against UNSEEN models' FP
vectors only -- genuine zero-shot generalization test.

No new FPs are built here -- reuses the exact FP directories already on
disk from the all-seen compressed/uncompressed uniform sweeps (same
probe-count points, same allocation), so results are directly comparable
point-for-point against the all-seen numbers already in hand.
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSCR_UNSEEN = 0.4848
LAM_LIST = np.logspace(-4, 2, 20)
K = 20
BANDIT_BETA = 0.000001
COST_GRID_POINTS = 20

# (label, fp_dir, is_compressed) -- reusing FPs already built on disk
POINTS = [
    ("compressed-1800", "embedllm-ceiling-scalesweep-uniform-minpctcap3-1800-pca5"),
    ("compressed-4000", "embedllm-ceiling-scalesweep-uniform-minpctcap3-4000-pca5"),
    ("compressed-8000", "embedllm-ceiling-scalesweep-uniform-minpctcap3-8000-pca5"),
    ("compressed-15000", "embedllm-ceiling-scalesweep-uniform-minpctcap3-15000-pca5"),
    ("compressed-25000", "embedllm-ceiling-scalesweep-uniform-minpctcap3-25000-pca5"),
    ("uncompressed-1800", "embedllm-ceiling-scalesweep-uniform-nocompress-minpctcap3-1800"),
    ("uncompressed-4000", "embedllm-ceiling-scalesweep-uniform-nocompress-minpctcap3-4000"),
    ("uncompressed-8000", "embedllm-ceiling-scalesweep-uniform-nocompress-minpctcap3-8000"),
    ("uncompressed-15000", "embedllm-ceiling-scalesweep-uniform-nocompress-minpctcap3-15000"),
    ("uncompressed-25000", "embedllm-ceiling-scalesweep-uniform-nocompress-minpctcap3-25000"),
    ("uncompressed-96", "embedllm-ceiling-scalesweep-uniform-nocompress-minpctcap3-96"),
    ("uncompressed-300", "embedllm-ceiling-scalesweep-uniform-nocompress-minpctcap3-300"),
    ("V2-full-uncompressed", "embedllm-ceiling"),
    ("full-data-compressed", "embedllm-ceiling-pca5"),
]


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
    categories = sorted(df["category"].unique().tolist())
    category_to_idx = {c: i for i, c in enumerate(categories)}

    print(f"Loading frozen MiniLM base (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    hidden_size = base_model.config.hidden_size

    # per-seed: seen/unseen split, seen-only training data + cached embeddings
    # (shared across ALL points for a given seed -- text/targets don't depend on FP)
    seed_cache = {}
    for seed in SEEDS:
        split_path = ANALYSIS_DIR / ("newllm_split.json" if seed == 0 else f"newllm_split_seed{seed}.json")
        split = json.load(open(split_path, encoding="utf-8"))
        seen_models, unseen_models = split["seen"], split["unseen"]
        raw_cat_acc = build_raw_category_accuracy(df, seen_models, categories)
        texts, targets, masks = build_items(df, seen_models, raw_cat_acc, category_to_idx, PCT_CATFILTER)
        print(f"seed={seed}: seen={len(seen_models)} unseen={len(unseen_models)} rows={len(texts)}", flush=True)
        t0 = time.time()
        cls_all = precompute_cls(texts, tokenizer, base_model)
        print(f"  cached in {time.time()-t0:.1f}s -> {cls_all.shape}", flush=True)

        dataset = load_embedllm("test", candidates=unseen_models)
        eval_texts = [ex["prompt"] for ex in dataset]
        label_maps = [ex["label_map"] for ex in dataset]
        t0 = time.time()
        cls_setB_unseen = precompute_cls(eval_texts, tokenizer, base_model)
        print(f"  Set B unseen-only ({len(eval_texts)} rows) cached in {time.time()-t0:.1f}s", flush=True)

        seed_cache[seed] = dict(seen_models=seen_models, unseen_models=unseen_models,
                                 cls_all=cls_all, targets=targets, masks=masks,
                                 cls_setB_unseen=cls_setB_unseen, label_maps=label_maps)

    results = {}
    for label, fp_dir_name in POINTS:
        fp_dir = Path(f"local_descriptors/{fp_dir_name}")
        print(f"\n{'#'*70}\n{label} ({fp_dir_name})\n{'#'*70}", flush=True)

        seed_metrics = []
        seed_costs, seed_accs = [], []
        for seed in SEEDS:
            c = seed_cache[seed]
            seen_models, unseen_models = c["seen_models"], c["unseen_models"]

            E_seen = np.stack([np.load(fp_dir / f"{m}.npy") for m in seen_models]).astype(np.float32)
            E_seen_t = torch.from_numpy(E_seen).float().to(DEVICE)
            E_seen_t = E_seen_t / (E_seen_t.norm(dim=1, keepdim=True) + 1e-9)

            proj = train_fast(seed, c["cls_all"], c["targets"], c["masks"], E_seen_t, hidden_size,
                               f"{label}-seed{seed}")

            E_unseen = np.stack([np.load(fp_dir / f"{m}.npy") for m in unseen_models]).astype(np.float32)
            E_unseen_norm = E_unseen / (np.linalg.norm(E_unseen, axis=1, keepdims=True) + 1e-12)
            costs_unseen = np.array([compute_cost(m, 0, cost_type="n_params") for m in unseen_models], dtype=np.float32)

            with torch.no_grad():
                embeds = F.normalize(proj(torch.from_numpy(c["cls_setB_unseen"]).float().to(DEVICE)),
                                      dim=-1).cpu().numpy()
            sims = embeds @ E_unseen_norm.T
            knn_costs, knn_accs = knn_curve(sims, unseen_models, c["label_maps"], costs_unseen, LAM_LIST)
            m = audc_qnc_peak(knn_costs, knn_accs)
            seed_metrics.append(m)
            seed_costs.append(knn_costs)
            seed_accs.append(knn_accs)
            print(f"  [{label} seed={seed}] AUDC={m['audc']:.4f} Peak={m['peak']:.4f} QNC={m['qnc']:.4f} "
                  f"({'BEATS' if m['audc'] > CSCR_UNSEEN else 'below'} CSCR-unseen {CSCR_UNSEEN})", flush=True)

        audcs = np.array([m["audc"] for m in seed_metrics])
        peaks = np.array([m["peak"] for m in seed_metrics])
        mean_costs = np.mean(seed_costs, axis=0)
        mean_accs = np.mean(seed_accs, axis=0)
        print(f"  [{label}] MEAN AUDC={audcs.mean():.4f} (std={audcs.std():.4f}) "
              f"Peak={peaks.mean():.4f} (std={peaks.std():.4f})", flush=True)
        results[label] = {"fp_dir": fp_dir_name, "audc_mean": float(audcs.mean()), "audc_std": float(audcs.std()),
                           "peak_mean": float(peaks.mean()), "peak_std": float(peaks.std()),
                           "per_seed": [{"audc": float(m["audc"]), "peak": float(m["peak"]), "qnc": float(m["qnc"])}
                                        for m in seed_metrics],
                           "costs_mean_curve": mean_costs.tolist(), "accs_mean_curve": mean_accs.tolist()}

    out_path = ANALYSIS_DIR / "probe_scale_sweep_unseen_multiseed_withcurves_results.json"
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n" + "=" * 90)
    print("PROBE SCALE SWEEP -- UNSEEN PROTOCOL (min(0.3,3), 3 seeds)")
    print("=" * 90)
    for label, _ in POINTS:
        r = results[label]
        print(f"  {label:<24s}: AUDC={r['audc_mean']:.4f} (std={r['audc_std']:.4f})")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
