"""RouterBench Ceiling V1.5 (1761 probes, 86-dim, already built) retrained
with min(pct=0.3, K_CAP=3) instead of min-pos (K=1) -- the winning
generalization found on EmbedLLM all-seen (0.5652->0.5867) and unseen
(0.5162->0.5238). Same top50pct catfilter, same FP, same all-seen protocol
as the K=1 result (mean 0.7282, PROGRESS.md 24.3) for direct comparability.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from routerbench_perplexity_combined import compute_keep_idx_top50pct, knn_curve, random_curve, audc_qnc_peak
from routerbench_fair_probe1800_multiseed import precompute_cls, make_proj, eval_proj
from run_audc_eval import paired_bootstrap_audc_cached
import routerbench_knn_test as rb

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
HOLDOUT_FRAC = 0.15
SEEDS = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSCR_ROUTERBENCH = 0.711
PCT_MINLOSS = 0.3
K_CAP = 3

CEILING_V15_DIR = Path("local_descriptors/routerbench-ceiling-v15")


def build_grpo_items(set_a, models, cols):
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

    texts, targets, masks = [], [], []
    for _, row in set_a.iterrows():
        labels = np.array([float(row[c]) for c in cols], dtype=np.float32)
        mean, std = labels.mean(), labels.std()
        if std < 1e-6:
            continue
        target = (labels - mean) / (std + 1e-6)
        mask = np.ones(n_models, dtype=np.float32)
        cat_idx = cat_to_idx.get(row["eval_name"])
        pos_idx = np.where(labels == 1)[0]
        if cat_idx is not None and len(pos_idx) > 0:
            scores = raw_cat_acc[pos_idx, cat_idx]
            keep_pos = compute_keep_idx_top50pct(pos_idx, scores)
            demoted = np.setdiff1d(pos_idx, keep_pos)
            mask[demoted] = 0.0
        texts.append(row["prompt"])
        targets.append(target)
        masks.append(mask)
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


def train_fast(seed, cls_tr, targets, masks, E_t, hidden_size, tag):
    n = cls_tr.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_ho = int(n * HOLDOUT_FRAC)
    ho_idx, tr_idx = perm[:n_ho], perm[n_ho:]

    from scipy.stats import spearmanr
    cls_tr_t = torch.from_numpy(cls_tr[tr_idx]).float().to(DEVICE)
    tgt_t = torch.from_numpy(targets[tr_idx]).float().to(DEVICE)
    msk_t = torch.from_numpy(masks[tr_idx]).float().to(DEVICE)
    cls_ho_t = torch.from_numpy(cls_tr[ho_idx]).float().to(DEVICE)
    tgt_ho = targets[ho_idx]
    msk_ho = masks[ho_idx]

    proj = make_proj(hidden_size, E_t.size(1), seed)
    opt = torch.optim.Adam(proj.parameters(), lr=LR)
    n_train = cls_tr_t.size(0)

    best_rho, best_epoch, best_state = -1.0, -1, None
    t0 = time.time()
    for ep in range(EPOCHS):
        order = torch.randperm(n_train, device=DEVICE)
        for start in range(0, n_train, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            q = F.normalize(proj(cls_tr_t[idx]), dim=-1)
            cos_sim = q @ E_t.T
            loss = minpctcap_loss(cos_sim, tgt_t[idx], msk_t[idx])
            loss.backward()
            opt.step()
            opt.zero_grad()

        with torch.no_grad():
            q_ho = F.normalize(proj(cls_ho_t), dim=-1)
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


def main():
    set_a, set_b = rb.load_data()
    models, cols = rb.NAMES, rb.MODELS
    cost_cols = [f"{c}|total_cost" for c in cols]

    print("Building GRPO targets + top50pct catfilter mask...", flush=True)
    texts, targets, masks = build_grpo_items(set_a, models, cols)
    print(f"{len(texts)} usable Set A rows", flush=True)

    print(f"Loading frozen MiniLM ONCE (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    hidden_size = base_model.config.hidden_size

    print("Precomputing frozen CLS embeddings for training rows...", flush=True)
    t0 = time.time()
    cls_tr = precompute_cls(texts, tokenizer, base_model)
    print(f"  done in {time.time()-t0:.1f}s -> {cls_tr.shape}", flush=True)

    set_b = set_b.dropna(subset=cols + cost_cols)
    b_texts = set_b["prompt"].tolist()
    b_label_maps = set_b[cols].to_numpy(dtype=np.float32).tolist()
    b_costs = set_b[cost_cols].to_numpy(dtype=np.float32).mean(axis=0)
    print(f"Precomputing frozen CLS embeddings for {len(b_texts)} Set B rows...", flush=True)
    t0 = time.time()
    cls_setB = precompute_cls(b_texts, tokenizer, base_model)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    E = np.stack([np.load(CEILING_V15_DIR / f"{n}.npy") for n in models])
    E_t = torch.from_numpy(E).float().to(DEVICE)
    E_t = E_t / (E_t.norm(dim=1, keepdim=True) + 1e-9)
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    print(f"\nminpctcap3-ceiling-v15 (FP dim={E.shape[1]}, dir={CEILING_V15_DIR})", flush=True)

    results = []
    for seed in SEEDS:
        proj = train_fast(seed, cls_tr, targets, masks, E_t, hidden_size, f"seed{seed}-minpctcap3")
        knn_metrics, delta, p = eval_proj(proj, cls_setB, E_norm, models, b_label_maps, b_costs, seed)
        results.append({"seed": seed, "knn": knn_metrics, "delta": delta, "p": p})
        print(f"  [seed={seed} minpctcap3] AUDC={knn_metrics['audc']:.4f} Peak={knn_metrics['peak']:.4f} "
              f"QNC={knn_metrics['qnc']:.4f} delta={delta:+.4f} p={p:.4g} "
              f"({'BEATS' if knn_metrics['audc'] > CSCR_ROUTERBENCH else 'below'} CSCR {CSCR_ROUTERBENCH})", flush=True)

    out_path = Path("local_descriptors/routerbench-analysis/v15_minpctcap3_multiseed_results.json")
    json.dump({"minpctcap3": results}, open(out_path, "w"), indent=2)

    audcs = [r["knn"]["audc"] for r in results]
    print("\n" + "=" * 90)
    print(f"MIN(PCT=0.3,CAP=3) on RouterBench Ceiling V1.5 (seeds {SEEDS}, all-seen) vs K=1 mean=0.7282 vs CSCR {CSCR_ROUTERBENCH}")
    print("=" * 90)
    print("minpctcap3: " + " / ".join(f"{a:.4f}" for a in audcs) +
          f"  mean={np.mean(audcs):.4f} (std={np.std(audcs):.4f})")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
