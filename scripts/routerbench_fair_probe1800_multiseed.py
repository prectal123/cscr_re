"""The actual mentor-requested fairness test: CSCR's OWN paper methodology
(Perplexity FP + cost_spectrum_info_nce, the Eq.8 Cost-Spectrum InfoNCE
confirmed byte-identical to upstream in PROGRESS.md 9) vs our methodology
(Ceiling V1.5 FP + Combined GRPO / min-pos+top50pct-catfilter), BOTH given
the same ~1800-probe budget, all-seen, multi-seed (RouterBench's 11-model
pool is too small for a seen/unseen split).

Earlier attempt (routerbench_perplexity_probesweep_combined.py) incorrectly
ran the Combined loss on Perplexity FP too -- that answers a different
question (does the loss trick work regardless of FP type, PROGRESS.md 22.4)
and is not what "give CSCR a fair probe budget" means: CSCR's own loss has
to be used for CSCR's own descriptor. Fixed here.

Also ~15x faster than the previous attempt: base MiniLM is FROZEN (only the
small projection head trains), so its CLS embeddings don't change across
epochs or seeds -- they were being needlessly recomputed every single epoch
in every previous script. Precomputed ONCE here for all Set A training rows
and Set B eval rows before the seed loop; each epoch then trains the tiny
2-layer proj head on cached (fixed) vectors only (matrix ops, no transformer
forward pass), matching the caching approach already used in
embedllm_newllm_fast_eval.py.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from run_audc_eval import interp_to_grid, build_cost_grid, paired_bootstrap_audc_cached
from router.bandit import BanditStats
import routerbench_knn_test as rb
from routerbench_perplexity_combined import (
    compute_keep_idx_top50pct, minpos_loss, knn_curve, random_curve, audc_qnc_peak,
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
HOLDOUT_FRAC = 0.15
SEEDS = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAM_LIST = np.logspace(-4, 2, 20)
CSCR_ROUTERBENCH = 0.711

# cost_spectrum_info_nce hyperparams -- exact defaults from train_query_encoder.py
CSCR_TAU = 0.07
CSCR_ALPHA = 0.25
CSCR_TAU_MIN = 0.05
CSCR_GAMMA = 0.2

CEILING_V15_DIR = Path("local_descriptors/routerbench-ceiling-v15")
PERPLEXITY_1800_DIR = Path("local_descriptors/routerbench-perplexity-nprobes1800")


def cost_spectrum_info_nce(z_q, E, label, cost_norm, tau=0.07, n_bands=5, alpha=0.25, tau_min=0.05, gamma=0.2):
    """Byte-identical port of train_query_encoder.py's cost_spectrum_info_nce
    (upstream CSCR paper Eq.8 loss, confirmed in PROGRESS.md 9)."""
    device = z_q.device
    B, M = label.shape
    percentiles = torch.linspace(0, 1, n_bands + 1, device=device)
    cost_bins = torch.quantile(cost_norm.view(-1), percentiles)
    band_idx = torch.bucketize(cost_norm, cost_bins[1:-1])

    sim = z_q @ E.T

    loss_accum, band_cnt = 0.0, 0
    for k in range(n_bands):
        b_mask = (band_idx == k)
        if b_mask.sum() == 0:
            continue
        pos_mask = label.clone()
        pos_mask[:, ~b_mask] = 0
        any_pos = pos_mask.any(1)
        if any_pos.sum() == 0:
            continue
        sim_k = sim[any_pos]
        pos_k = pos_mask[any_pos]

        tau_b = tau_min + alpha * cost_norm[b_mask].mean()
        exp_pos = torch.exp(sim_k / tau_b)
        numer = (exp_pos * pos_k).sum(1)

        cost_pen = gamma * cost_norm.unsqueeze(0)
        logits_k = (sim_k - cost_pen) / tau_b
        denom = torch.exp(logits_k).sum(1)

        loss_accum += -(numer / (denom + 1e-9)).log().mean()
        band_cnt += 1

    if band_cnt == 0:
        return torch.tensor(0., device=device, requires_grad=True)
    return loss_accum / band_cnt


def build_grpo_items(set_a, models, cols):
    """(text, z-scored target, top50pct-catfilter mask) -- for Combined."""
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

    texts, targets, masks, raw_labels = [], [], [], []
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
        raw_labels.append(labels)
    return texts, np.stack(targets), np.stack(masks), np.stack(raw_labels)


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


def make_proj(hidden_size, out_dim, seed):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.ReLU(),
        nn.Linear(hidden_size, out_dim, bias=False),
    ).to(DEVICE)


def train_combined_fast(seed, cls_tr, targets, masks, E_t, hidden_size, tag):
    n = cls_tr.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_ho = int(n * HOLDOUT_FRAC)
    ho_idx, tr_idx = perm[:n_ho], perm[n_ho:]

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
            loss = minpos_loss(cos_sim, tgt_t[idx], msk_t[idx])
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


def train_cscr_fast(seed, cls_tr, raw_labels, cost_norm_t, E_t, hidden_size, n_bands, tag):
    n = cls_tr.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_ho = int(n * HOLDOUT_FRAC)
    ho_idx, tr_idx = perm[:n_ho], perm[n_ho:]

    cls_tr_t = torch.from_numpy(cls_tr[tr_idx]).float().to(DEVICE)
    lbl_t = torch.from_numpy(raw_labels[tr_idx]).float().to(DEVICE)
    cls_ho_t = torch.from_numpy(cls_tr[ho_idx]).float().to(DEVICE)
    lbl_ho_t = torch.from_numpy(raw_labels[ho_idx]).float().to(DEVICE)

    proj = make_proj(hidden_size, E_t.size(1), seed)
    opt = torch.optim.Adam(proj.parameters(), lr=LR)
    n_train = cls_tr_t.size(0)

    best_loss, best_epoch, best_state = float("inf"), -1, None
    t0 = time.time()
    for ep in range(EPOCHS):
        order = torch.randperm(n_train, device=DEVICE)
        for start in range(0, n_train, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            q = F.normalize(proj(cls_tr_t[idx]), dim=-1)
            loss = cost_spectrum_info_nce(q, E_t, lbl_t[idx], cost_norm_t, tau=CSCR_TAU,
                                           n_bands=n_bands, alpha=CSCR_ALPHA, tau_min=CSCR_TAU_MIN,
                                           gamma=CSCR_GAMMA)
            loss.backward()
            opt.step()
            opt.zero_grad()

        with torch.no_grad():
            q_ho = F.normalize(proj(cls_ho_t), dim=-1)
            ho_loss = cost_spectrum_info_nce(q_ho, E_t, lbl_ho_t, cost_norm_t, tau=CSCR_TAU,
                                              n_bands=n_bands, alpha=CSCR_ALPHA, tau_min=CSCR_TAU_MIN,
                                              gamma=CSCR_GAMMA).item()
        if ho_loss < best_loss:
            best_loss, best_epoch = ho_loss, ep + 1
            best_state = {k: v.clone() for k, v in proj.state_dict().items()}
    proj.load_state_dict(best_state)
    print(f"  [{tag}] best_epoch={best_epoch} best_holdout_loss={best_loss:.4f} time={time.time()-t0:.1f}s", flush=True)
    return proj


def eval_proj(proj, cls_setB, E_norm, models, b_label_maps, b_costs, seed):
    with torch.no_grad():
        q = F.normalize(proj(torch.from_numpy(cls_setB).float().to(DEVICE)), dim=-1).cpu().numpy()
    sims = q @ E_norm.T
    knn_costs, knn_accs, knn_Y = knn_curve(sims, models, b_label_maps, b_costs, LAM_LIST)
    rand_costs, rand_accs, rand_Y = random_curve(models, b_label_maps, b_costs, LAM_LIST, seed=seed)
    knn_metrics = audc_qnc_peak(knn_costs, knn_accs)
    ko, ro = np.argsort(knn_costs), np.argsort(rand_costs)
    mean_delta, (lo, hi), p = paired_bootstrap_audc_cached(
        knn_costs[ko], knn_Y[ko], rand_costs[ro], rand_Y[ro], B=1000, seed=0)
    return knn_metrics, float(mean_delta), float(p)


def main():
    set_a, set_b = rb.load_data()
    models, cols = rb.NAMES, rb.MODELS
    cost_cols = [f"{c}|total_cost" for c in cols]

    print("Building training rows (GRPO targets + raw labels, shared text set)...", flush=True)
    texts, targets, masks, raw_labels = build_grpo_items(set_a, models, cols)
    print(f"{len(texts)} usable Set A rows", flush=True)

    print(f"Loading frozen MiniLM ONCE (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    hidden_size = base_model.config.hidden_size

    print("Precomputing frozen CLS embeddings for all training rows (once, shared across seeds+losses)...", flush=True)
    t0 = time.time()
    cls_tr = precompute_cls(texts, tokenizer, base_model)
    print(f"  done in {time.time()-t0:.1f}s -> {cls_tr.shape}", flush=True)

    set_b = set_b.dropna(subset=cols + cost_cols)
    b_texts = set_b["prompt"].tolist()
    b_label_maps = set_b[cols].to_numpy(dtype=np.float32).tolist()
    b_costs = set_b[cost_cols].to_numpy(dtype=np.float32).mean(axis=0)
    print(f"Precomputing frozen CLS embeddings for {len(b_texts)} Set B rows (once)...", flush=True)
    t0 = time.time()
    cls_setB = precompute_cls(b_texts, tokenizer, base_model)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    # cost_norm for CSCR loss: mean Set-A cost per model, min-max normalized (matches train_query_encoder.py)
    a_costs = set_a.dropna(subset=cost_cols)[cost_cols].to_numpy(dtype=np.float32).mean(axis=0)
    cost_norm = (a_costs - a_costs.min()) / (a_costs.max() - a_costs.min() + 1e-9)
    cost_norm_t = torch.from_numpy(cost_norm).float().to(DEVICE)
    n_bands = int(round(len(models) ** 0.5))
    print(f"n_bands={n_bands} (sqrt({len(models)}))", flush=True)

    results = {"ceiling-v15-combined": [], "perplexity-1800-cscrloss": []}

    # --- arm 1: Ceiling V1.5 + Combined GRPO ---
    E = np.stack([np.load(CEILING_V15_DIR / f"{n}.npy") for n in models])
    E_t = torch.from_numpy(E).float().to(DEVICE)
    E_t = E_t / (E_t.norm(dim=1, keepdim=True) + 1e-9)
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    print(f"\n{'#'*70}\nceiling-v15-combined (FP dim={E.shape[1]}, dir={CEILING_V15_DIR})\n{'#'*70}", flush=True)
    for seed in SEEDS:
        proj = train_combined_fast(seed, cls_tr, targets, masks, E_t, hidden_size, f"seed{seed}-ceiling-v15")
        knn_metrics, delta, p = eval_proj(proj, cls_setB, E_norm, models, b_label_maps, b_costs, seed)
        results["ceiling-v15-combined"].append({"seed": seed, "knn": knn_metrics, "delta": delta, "p": p})
        print(f"  [seed={seed} ceiling-v15-combined] AUDC={knn_metrics['audc']:.4f} Peak={knn_metrics['peak']:.4f} "
              f"QNC={knn_metrics['qnc']:.4f} delta={delta:+.4f} p={p:.4g} "
              f"({'BEATS' if knn_metrics['audc'] > CSCR_ROUTERBENCH else 'below'} CSCR {CSCR_ROUTERBENCH})", flush=True)

    # --- arm 2: Perplexity 1800-probe + CSCR's own cost_spectrum_info_nce ---
    E2 = np.stack([np.load(PERPLEXITY_1800_DIR / f"{n}.npy") for n in models])
    E2_t = torch.from_numpy(E2).float().to(DEVICE)
    E2_t = E2_t / (E2_t.norm(dim=1, keepdim=True) + 1e-9)
    E2_norm = E2 / (np.linalg.norm(E2, axis=1, keepdims=True) + 1e-12)
    print(f"\n{'#'*70}\nperplexity-1800-cscrloss (FP dim={E2.shape[1]}, dir={PERPLEXITY_1800_DIR})\n{'#'*70}", flush=True)
    for seed in SEEDS:
        proj = train_cscr_fast(seed, cls_tr, raw_labels, cost_norm_t, E2_t, hidden_size, n_bands,
                                f"seed{seed}-perplexity-1800-cscrloss")
        knn_metrics, delta, p = eval_proj(proj, cls_setB, E2_norm, models, b_label_maps, b_costs, seed)
        results["perplexity-1800-cscrloss"].append({"seed": seed, "knn": knn_metrics, "delta": delta, "p": p})
        print(f"  [seed={seed} perplexity-1800-cscrloss] AUDC={knn_metrics['audc']:.4f} Peak={knn_metrics['peak']:.4f} "
              f"QNC={knn_metrics['qnc']:.4f} delta={delta:+.4f} p={p:.4g} "
              f"({'BEATS' if knn_metrics['audc'] > CSCR_ROUTERBENCH else 'below'} CSCR {CSCR_ROUTERBENCH})", flush=True)

    out_path = Path("local_descriptors/routerbench-analysis/fair_probe1800_multiseed_results.json")
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n" + "=" * 90)
    print(f"FAIR 1800-PROBE COMPARISON on RouterBench (seeds {SEEDS}, all-seen) vs CSCR paper {CSCR_ROUTERBENCH}")
    print("=" * 90)
    for tag in results:
        audcs = [r["knn"]["audc"] for r in results[tag]]
        beats = sum(1 for a in audcs if a > CSCR_ROUTERBENCH)
        print(f"{tag:>28}: " + " / ".join(f"{a:.4f}" for a in audcs) +
              f"  mean={np.mean(audcs):.4f} (std={np.std(audcs):.4f})  beats CSCR paper: {beats}/{len(audcs)}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
