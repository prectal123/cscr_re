"""4-way ablation of TAR's two trim mechanisms (catfilter, min-top-k), measured
on COLLAPSE metrics specifically (top3_share, n_models_used, rho(selection,
true_acc), top-true-acc-model's selection share) -- not AUDC. Uncompressed
(80-dim) headline FP throughout, matching PROGRESS.md section 27.1's checkpoint.

Prior ablations in this project (section 22.11/24.12) measured AUDC only, and
did so on the deprecated PCA-5-compressed pipeline. This is the first
collapse-specific 4-way ablation, and the first one on the uncompressed FP.

Variants (catfilter_pct controls build_items' category-track-record demotion;
minloss_pct/minloss_kcap control minpctcap_loss's within-loss top-k trim):
  vanilla:    catfilter_pct=1.0 (off), minloss_pct=1.0, kcap=999 (off) -- mean-over-all-positives
  catfilter:  catfilter_pct=0.3,        minloss_pct=1.0, kcap=999 (off)
  minloss:    catfilter_pct=1.0 (off),  minloss_pct=0.3, kcap=3
  combined:   catfilter_pct=0.3,        minloss_pct=0.3, kcap=3  (= current headline, TAR)

Single seed (0), matching this session's precedent for collapse_check runs.
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
from embedllm_unseen_mixedpool_1800_multiseed import (
    build_raw_category_accuracy, build_items, precompute_cls,
    ANALYSIS_DIR, EMBED_MODEL, DEVICE, EPOCHS, BATCH_SIZE, LR, HOLDOUT_FRAC,
)
from router.embedllm import load_embedllm

SEED = 0
FP_DIR = Path("local_descriptors/embedllm-ceiling-scalesweep-uniform-nocompress-minpctcap3-1800")

VARIANTS = {
    "vanilla":   dict(catfilter_pct=1.0, minloss_pct=1.0, minloss_kcap=999),
    "catfilter": dict(catfilter_pct=0.3, minloss_pct=1.0, minloss_kcap=999),
    "minloss":   dict(catfilter_pct=1.0, minloss_pct=0.3, minloss_kcap=3),
    "combined":  dict(catfilter_pct=0.3, minloss_pct=0.3, minloss_kcap=3),
}


def minpctcap_loss(cos_sim, target, mask, pct, k_cap):
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


def train_variant(seed, cls_all, targets, masks, E_t, hidden_size, minloss_pct, minloss_kcap, tag):
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
            loss = minpctcap_loss(cos_sim, tgt_tr[idx], msk_tr[idx], minloss_pct, minloss_kcap)
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


def collapse_metrics(proj, cls_setB, unseen_models, label_maps, E_unseen):
    with torch.no_grad():
        embeds = F.normalize(proj(torch.from_numpy(cls_setB).float().to(DEVICE)), dim=-1).cpu().numpy()
    sims = embeds @ E_unseen.T
    chosen = np.argmax(sims, axis=1)
    counts = np.bincount(chosen, minlength=len(unseen_models))
    order = np.argsort(-counts)
    n = len(sims)

    top3_share = counts[order[:3]].sum() / n
    n_used = int((counts > 0).sum())
    true_acc = np.array([np.mean([lm.get(m, 0) for lm in label_maps]) for m in unseen_models])
    rho, p = spearmanr(counts, true_acc)
    top_acc_idx = int(np.argmax(true_acc))
    top_selected_idx = order[0]
    return {
        "top3_share": float(top3_share),
        "n_used": n_used,
        "n_total": len(unseen_models),
        "rho": float(rho), "p": float(p),
        "most_selected_model": unseen_models[top_selected_idx],
        "most_selected_share": float(counts[top_selected_idx] / n),
        "most_selected_true_acc": float(true_acc[top_selected_idx]),
        "highest_true_acc_model": unseen_models[top_acc_idx],
        "highest_true_acc": float(true_acc[top_acc_idx]),
        "highest_true_acc_model_share": float(counts[top_acc_idx] / n),
    }


def main():
    print("Loading EmbedLLM train.csv...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "category", "label", "prompt"])
    categories = sorted(df["category"].unique().tolist())
    category_to_idx = {c: i for i, c in enumerate(categories)}

    split = json.load(open(ANALYSIS_DIR / "newllm_split.json", encoding="utf-8"))
    seen_models, unseen_models = split["seen"], split["unseen"]
    print(f"seen={len(seen_models)} unseen={len(unseen_models)}", flush=True)

    print(f"Loading frozen MiniLM base (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    hidden_size = base_model.config.hidden_size

    raw_cat_acc = build_raw_category_accuracy(df, seen_models, categories)

    E_seen = np.stack([np.load(FP_DIR / f"{m}.npy") for m in seen_models]).astype(np.float32)
    E_seen_t = torch.from_numpy(E_seen).float().to(DEVICE)
    E_seen_t = E_seen_t / (E_seen_t.norm(dim=1, keepdim=True) + 1e-9)

    dataset = load_embedllm("test", candidates=unseen_models)
    eval_texts = [ex["prompt"] for ex in dataset]
    label_maps = [ex["label_map"] for ex in dataset]
    print(f"{len(eval_texts)} test prompts (unseen-only candidates)", flush=True)
    cls_setB = precompute_cls(eval_texts, tokenizer, base_model)

    E_unseen = np.stack([np.load(FP_DIR / f"{m}.npy") for m in unseen_models]).astype(np.float32)
    E_unseen = E_unseen / (np.linalg.norm(E_unseen, axis=1, keepdims=True) + 1e-12)

    # cache build_items+cls per distinct catfilter_pct (texts identical across pct, but
    # recompute cls_all per call for safety -- cheap enough, ~80s each)
    cache = {}
    results = {}
    for name, cfg in VARIANTS.items():
        cf_pct = cfg["catfilter_pct"]
        if cf_pct not in cache:
            texts, targets, masks = build_items(df, seen_models, raw_cat_acc, category_to_idx, cf_pct)
            t0 = time.time()
            cls_all = precompute_cls(texts, tokenizer, base_model)
            print(f"  [catfilter_pct={cf_pct}] {len(texts)} rows cached in {time.time()-t0:.1f}s", flush=True)
            cache[cf_pct] = (cls_all, targets, masks)
        cls_all, targets, masks = cache[cf_pct]

        proj = train_variant(SEED, cls_all, targets, masks, E_seen_t, hidden_size,
                              cfg["minloss_pct"], cfg["minloss_kcap"], name)
        metrics = collapse_metrics(proj, cls_setB, unseen_models, label_maps, E_unseen)
        results[name] = metrics
        print(f"  [{name}] top3_share={metrics['top3_share']:.4f} n_used={metrics['n_used']}/{metrics['n_total']} "
              f"rho={metrics['rho']:.4f} (p={metrics['p']:.4f}) "
              f"top_acc_model_share={metrics['highest_true_acc_model_share']:.4f}", flush=True)

    print("\n" + "=" * 90)
    print("4-WAY ABLATION SUMMARY (collapse metrics, uncompressed FP, seed=0)")
    print("=" * 90)
    print(f"{'variant':12s} {'top3_share':>11s} {'n_used':>8s} {'rho':>8s} {'p':>10s} {'top_acc_model_share':>20s}")
    for name in VARIANTS:
        m = results[name]
        print(f"{name:12s} {m['top3_share']:>11.4f} {m['n_used']:>5d}/{m['n_total']:<2d} "
              f"{m['rho']:>8.4f} {m['p']:>10.4f} {m['highest_true_acc_model_share']:>20.4f}")

    out_path = ANALYSIS_DIR / "uncompressed_4way_ablation_collapse_check_seed0.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
