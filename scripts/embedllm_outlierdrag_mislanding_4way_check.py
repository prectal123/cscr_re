"""Direct mechanism checks for catfilter (targets "outlier-drag") and
min(0.3,3) (targets "query mislanding") -- rather than inferring their
effect from aggregate metrics (rho, top3_share) as in
embedllm_uncompressed_ablation_collapse_check.py, measure each mechanism's
OWN stated problem directly, per multi-positive Set B query:

- Outlier-drag: does the trained query embedding land near ANY real
  unseen model, or drift into empty space, as a function of how spread out
  the query's TRUE positive candidates are in FP space? Measured as
  Spearman corr(spread_of_true_positives, distance_to_nearest_real_model)
  -- more positive corr = more drag (more spread -> more empty-space
  landing). Catfilter is hypothesized to reduce this.
- Query mislanding: among a query's TRUE positive candidates, does the
  landed (highest-similarity) one match the BEST one (by overall true
  accuracy), or a worse one? Measured as mislanding_rate = fraction of
  multi-positive queries where landed != best. min(0.3,3) is hypothesized
  to reduce this.

Reuses the exact same 4-way training setup as
embedllm_uncompressed_ablation_collapse_check.py (same variants, same seed,
same uncompressed headline FP) so results are directly comparable -- adds
the two new diagnostics as pure post-hoc analysis of already-computed
similarity matrices (no extra forward passes).
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


def mechanism_diagnostics(proj, cls_setB, unseen_models, label_maps, E_unseen):
    with torch.no_grad():
        embeds = F.normalize(proj(torch.from_numpy(cls_setB).float().to(DEVICE)), dim=-1).cpu().numpy()
    sims = embeds @ E_unseen.T  # (n_queries, n_unseen_models)

    counts = np.bincount(np.argmax(sims, axis=1), minlength=len(unseen_models))
    true_acc = np.array([np.mean([lm.get(m, 0) for lm in label_maps]) for m in unseen_models])

    spreads, near_any_dists = [], []
    mislanding_flags = []
    for i in range(sims.shape[0]):
        lm = label_maps[i]
        pos_idx = [j for j, m in enumerate(unseen_models) if lm.get(m, 0) == 1]
        if len(pos_idx) < 2:
            continue
        pos_vecs = E_unseen[pos_idx]
        c = pos_vecs.mean(axis=0)
        spread = float(np.mean(np.linalg.norm(pos_vecs - c, axis=1)))
        near_any = float(sims[i].max())
        spreads.append(spread)
        near_any_dists.append(1.0 - near_any)  # higher = farther from any real model

        best_local = int(np.argmax(true_acc[pos_idx]))
        best_j = pos_idx[best_local]
        landed_local = int(np.argmax(sims[i, pos_idx]))
        landed_j = pos_idx[landed_local]
        mislanding_flags.append(0 if landed_j == best_j else 1)

    drag_corr, drag_p = spearmanr(spreads, near_any_dists) if len(spreads) > 2 else (float("nan"), float("nan"))
    mislanding_rate = float(np.mean(mislanding_flags)) if mislanding_flags else float("nan")

    return {
        "n_multipos_queries": len(spreads),
        "outlier_drag_corr": float(drag_corr), "outlier_drag_p": float(drag_p),
        "mislanding_rate": mislanding_rate,
        "top3_share": float(counts[np.argsort(-counts)[:3]].sum() / sims.shape[0]),
        "n_used": int((counts > 0).sum()),
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
        metrics = mechanism_diagnostics(proj, cls_setB, unseen_models, label_maps, E_unseen)
        results[name] = metrics
        print(f"  [{name}] outlier_drag_corr={metrics['outlier_drag_corr']:.4f} (p={metrics['outlier_drag_p']:.4f}) "
              f"mislanding_rate={metrics['mislanding_rate']:.4f} (n={metrics['n_multipos_queries']})", flush=True)

    print("\n" + "=" * 90)
    print("MECHANISM-LEVEL CHECK: outlier-drag (catfilter's target) vs query-mislanding (min's target)")
    print("=" * 90)
    print(f"{'variant':12s} {'outlier_drag_corr':>18s} {'p':>8s} {'mislanding_rate':>16s} {'n_multipos':>12s}")
    for name in VARIANTS:
        m = results[name]
        print(f"{name:12s} {m['outlier_drag_corr']:>18.4f} {m['outlier_drag_p']:>8.4f} "
              f"{m['mislanding_rate']:>16.4f} {m['n_multipos_queries']:>12d}")

    out_path = ANALYSIS_DIR / "outlierdrag_mislanding_4way_check_seed0.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
