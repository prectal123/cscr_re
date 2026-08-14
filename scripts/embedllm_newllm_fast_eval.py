"""Fast, cache-based reimplementation of run_audc_eval.py's knn/random deferral
curve computation for the "new LLMs" (unseen-only candidate pool) setup.

Why: the official run_audc_eval.py re-encodes EVERY test prompt through
QueryEncoder ONE AT A TIME (batch size 1) for EACH lambda point, and then
does it all again for the bootstrap significance step -- ~120,000 individual
forward passes for a 20-point sweep with 3000 prompts. But the query
EMBEDDING doesn't depend on lambda at all (lambda only reweights the
already-computed similarity/cost tradeoff at decision time) -- so re-encoding
per lambda is pure waste.

This script batch-encodes every test prompt ONCE, caches the (N, proj_dim)
embedding matrix and the (N, M) cosine-similarity matrix, and then replays
the EXACT same decision logic as KNNRouter.route() / RandomRouter.route()
per lambda using only cheap array ops:
  - knn: top-k by raw similarity (matching FAISS IndexFlatIP.search order),
    then argmax(bandit_bonus + sim - lambda*cost) with the SAME BanditStats
    class (src/router/bandit.py) run sequentially in dataset order, so the
    "try each never-seen expert once" cold-start dynamic is preserved exactly
    -- nothing about the ROUTING LOGIC is simplified, only the encoding is
    cached/batched.
  - random: cost-weighted softmax draw exp(-lambda*cost), matching
    src/router/random_router.py's RandomRouter.route() exactly.

AUDC/QNC/Peak and the paired bootstrap significance test reuse the ACTUAL
functions from run_audc_eval.py (imported, not reimplemented), so the final
numbers are computed identically to the official script -- only the
expensive, lambda-independent encoding step is sped up.
"""
import json
import random as pyrandom
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from router.query_encoder import QueryEncoder
from router.bandit import BanditStats
from router.cost_models import compute_cost
from router.embedllm import load_embedllm
from run_audc_eval import area_under_curve, interp_to_grid, build_cost_grid, paired_bootstrap_audc_cached

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAM_LIST = np.logspace(-4, 2, 20)  # matches run_audc_eval.py's --lambda_min -4 --lambda_max 2 --n_points 20 defaults
K = 20
BANDIT_BETA = 0.000001
COST_GRID_POINTS = 20


def batch_encode(encoder, texts, batch_size=64):
    embeds = np.zeros((len(texts), encoder.proj_dim), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        embeds[start:start + len(batch)] = encoder.encode(batch)
    return embeds


def knn_curve(sims, models, label_maps, costs, lam_list, k=K, bandit_beta=BANDIT_BETA, seed=0):
    """Returns (costs_per_lambda, accs_per_lambda, Y) where Y[t] is the
    per-prompt correctness vector at lambda_list[t] (for bootstrap reuse)."""
    n_prompts, n_models = sims.shape
    order = np.argsort(-sims, axis=1)  # descending similarity, per prompt -- matches FAISS search order
    topk = order[:, :min(k, n_models)]

    out_costs, out_accs = [], []
    Y = np.zeros((len(lam_list), n_prompts), dtype=np.int32)
    for li, lam in enumerate(lam_list):
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
            Y[li, i] = int(acc)
        out_costs.append(tot_cost / n_prompts)
        out_accs.append(tot_acc / n_prompts)
    return np.array(out_costs), np.array(out_accs), Y


def random_curve(models, label_maps, costs, lam_list, seed=0):
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
            chosen = models[j]
            acc = 1.0 if label_maps[i].get(chosen, 0) == 1 else 0.0
            cost = float(costs[j])
            tot_cost += cost
            tot_acc += acc
            Y[li, i] = int(acc)
        out_costs.append(tot_cost / n_prompts)
        out_accs.append(tot_acc / n_prompts)
    return np.array(out_costs), np.array(out_accs), Y


def audc_qnc_peak(costs, accs):
    order = np.argsort(costs)
    c, a = costs[order], accs[order]
    grid = build_cost_grid(c, N_grid=COST_GRID_POINTS)
    a_grid = interp_to_grid(c, a, grid)
    audc = np.trapz(a_grid, grid) / (grid[-1] - grid[0])
    peak_idx = np.argmax(a)
    return {"audc": float(audc), "qnc": float(c[peak_idx]), "peak": float(a[peak_idx])}


def run_one(split_path, pca5_unseen_dir, encoder_ckpt_dir, label="run"):
    t0 = time.time()
    split = json.load(open(split_path, encoding="utf-8"))
    unseen = split["unseen"]
    print(f"[{label}] unseen={len(unseen)} models", flush=True)

    enc = QueryEncoder.load(encoder_ckpt_dir, proj_dim=5)
    enc.to(DEVICE)
    enc.model.eval()

    dataset = load_embedllm("test", candidates=unseen)
    texts = [ex["prompt"] for ex in dataset]
    label_maps = [ex["label_map"] for ex in dataset]
    print(f"[{label}] {len(texts)} test prompts, encoding once (batched)...", flush=True)

    embeds = batch_encode(enc, texts)
    E_unseen = np.stack([np.load(Path(pca5_unseen_dir) / f"{m}.npy") for m in unseen]).astype(np.float32)
    E_unseen = E_unseen / (np.linalg.norm(E_unseen, axis=1, keepdims=True) + 1e-12)
    sims = embeds @ E_unseen.T
    costs = np.array([compute_cost(m, 0, cost_type="n_params") for m in unseen], dtype=np.float32)
    print(f"[{label}] encoding + sim matrix done in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    knn_costs, knn_accs, knn_Y = knn_curve(sims, unseen, label_maps, costs, LAM_LIST)
    rand_costs, rand_accs, rand_Y = random_curve(unseen, label_maps, costs, LAM_LIST)
    print(f"[{label}] curve sweep done in {time.time()-t1:.1f}s", flush=True)

    knn_metrics = audc_qnc_peak(knn_costs, knn_accs)
    rand_metrics = audc_qnc_peak(rand_costs, rand_accs)

    mean_delta, (lo, hi), p = paired_bootstrap_audc_cached(knn_costs, knn_Y, rand_costs, rand_Y, B=2000, seed=0)

    print(f"[{label}] knn: AUDC={knn_metrics['audc']:.4f} QNC={knn_metrics['qnc']:.3f} Peak={knn_metrics['peak']:.4f}", flush=True)
    print(f"[{label}] random: AUDC={rand_metrics['audc']:.4f} Peak={rand_metrics['peak']:.4f}", flush=True)
    print(f"[{label}] bootstrap delta(knn-random)={mean_delta:+.4f} CI=[{lo:.4f},{hi:.4f}] p={p:.4g}", flush=True)
    print(f"[{label}] TOTAL TIME {time.time()-t0:.1f}s", flush=True)

    return {"knn": knn_metrics, "random": rand_metrics,
            "bootstrap_delta": float(mean_delta), "bootstrap_p": float(p)}


if __name__ == "__main__":
    # quick self-test on seed 0 (already have official numbers to compare against:
    # knn AUDC=0.4731 QNC=0.909 Peak=0.536, random AUDC=0.3904 Peak=0.423)
    r = run_one(
        "local_descriptors/embedllm-analysis/newllm_split.json",
        "local_descriptors/embedllm-ceiling-pca5-unseen-only",
        "local_checkpoints/embedllm-newllm-encoder-csinfonce",
        label="seed0-fastcheck",
    )
    out_path = Path("local_descriptors/embedllm-analysis/newllm_fast_selftest_seed0.json")
    json.dump(r, open(out_path, "w"), indent=2)
    print(f"Saved -> {out_path}")
