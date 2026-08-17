"""Ensemble eval -- no retraining. Loads N already-trained checkpoints that
share the same frozen MiniLM backbone but differ only in their small
projection head (different seeds), encodes each test query through EVERY
checkpoint's head, and averages the resulting SIMILARITY SCORES (not the
raw query vectors -- averaging vectors first risks the same "empty space"
landing that outlier-drag caused; averaging final scores is the safe
choice) before feeding into the standard KNN+bandit routing eval.

First run: unseen protocol, pct=0.3 combined, seeds 0-2 (checkpoints
already exist from embedllm_pct30_unseen_multiseed.py -- no training here
at all, purely evaluation).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from router.query_encoder import QueryEncoder
from router.embedllm import load_embedllm
from router.bandit import BanditStats
from router.cost_models import compute_cost
from run_audc_eval import interp_to_grid, build_cost_grid

ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSCR_UNSEEN = 0.4848
LAM_LIST = np.logspace(-4, 2, 20)
K = 20
BANDIT_BETA = 0.000001
COST_GRID_POINTS = 20

CKPT_DIRS = [
    Path("local_checkpoints/embedllm-newllm-encoder-pct30-seed0"),
    Path("local_checkpoints/embedllm-newllm-encoder-pct30-seed1"),
    Path("local_checkpoints/embedllm-newllm-encoder-pct30-seed2"),
]
SPLIT_PATH = ANALYSIS_DIR / "newllm_split.json"  # seed0's split (same unseen pool for all 3 heads' eval)
UNSEEN_DIR = Path("local_descriptors/embedllm-ceiling-pca5-unseen-only")

REFERENCE_SEEDS = {"seed0": 0.5264, "seed1": 0.5246, "seed2": 0.4977, "mean": 0.5162}


def knn_curve(sims, models, label_maps, costs, lam_list, k=K, bandit_beta=BANDIT_BETA):
    n_prompts, n_models = sims.shape
    order = np.argsort(-sims, axis=1)
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


def audc_qnc_peak(costs, accs):
    order = np.argsort(costs)
    c, a = costs[order], accs[order]
    grid = build_cost_grid(c, N_grid=COST_GRID_POINTS)
    a_grid = interp_to_grid(c, a, grid)
    audc = np.trapezoid(a_grid, grid) / (grid[-1] - grid[0])
    peak_idx = np.argmax(a)
    return {"audc": float(audc), "qnc": float(c[peak_idx]), "peak": float(a[peak_idx])}


def main():
    split = json.load(open(SPLIT_PATH, encoding="utf-8"))
    unseen_models = split["unseen"]
    print(f"unseen models: {len(unseen_models)}", flush=True)

    E_unseen = np.stack([np.load(UNSEEN_DIR / f"{m}.npy") for m in unseen_models]).astype(np.float32)
    E_unseen = E_unseen / (np.linalg.norm(E_unseen, axis=1, keepdims=True) + 1e-12)
    costs = np.array([compute_cost(m, 0, cost_type="n_params") for m in unseen_models], dtype=np.float32)

    dataset = load_embedllm("test", candidates=unseen_models)
    texts = [ex["prompt"] for ex in dataset]
    label_maps = [ex["label_map"] for ex in dataset]
    print(f"{len(texts)} test prompts", flush=True)

    all_sims = []
    for ckpt_dir in CKPT_DIRS:
        print(f"Encoding with {ckpt_dir}...", flush=True)
        enc = QueryEncoder.load(str(ckpt_dir), proj_dim=5)
        enc.to(DEVICE)
        enc.model.eval()
        embeds = np.zeros((len(texts), 5), dtype=np.float32)
        for start in range(0, len(texts), 64):
            batch = texts[start:start + 64]
            embeds[start:start + len(batch)] = enc.encode(batch)
        sims = embeds @ E_unseen.T
        all_sims.append(sims)

        knn_costs, knn_accs, knn_Y = knn_curve(sims, unseen_models, label_maps, costs, LAM_LIST)
        m = audc_qnc_peak(knn_costs, knn_accs)
        print(f"  [{ckpt_dir.name}] solo AUDC={m['audc']:.4f} Peak={m['peak']:.4f}", flush=True)

    ensembled_sims = np.mean(all_sims, axis=0)
    knn_costs, knn_accs, knn_Y = knn_curve(ensembled_sims, unseen_models, label_maps, costs, LAM_LIST)
    ens_metrics = audc_qnc_peak(knn_costs, knn_accs)

    print("\n" + "=" * 90)
    print(f"ENSEMBLE (mean of {len(CKPT_DIRS)} similarity scores) -- EmbedLLM unseen")
    print("=" * 90)
    print(f"AUDC={ens_metrics['audc']:.4f} Peak={ens_metrics['peak']:.4f} "
          f"({'BEATS' if ens_metrics['audc'] > CSCR_UNSEEN else 'below'} CSCR {CSCR_UNSEEN})")
    print(f"reference (best single seed): {max(REFERENCE_SEEDS['seed0'], REFERENCE_SEEDS['seed1'], REFERENCE_SEEDS['seed2']):.4f}")
    print(f"reference (mean of 3 solo AUDCs, post-hoc average): {REFERENCE_SEEDS['mean']:.4f}")

    out_path = ANALYSIS_DIR / "ensemble_unseen_pct30_results.json"
    json.dump({"ensemble": ens_metrics, "reference_solo_mean": REFERENCE_SEEDS["mean"]}, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
