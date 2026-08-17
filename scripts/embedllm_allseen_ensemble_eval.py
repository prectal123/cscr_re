"""Ensemble eval for the all-seen pct=0.3 combined checkpoints -- no
retraining, pure evaluation. Unlike the unseen protocol (where each seed's
own "unseen" pool differs and can't be fairly shared across checkpoints --
see the leakage found in embedllm_ensemble_eval.py), all-seen uses the
IDENTICAL 111-model pool for every seed, so there's no shared-pool problem
here: all 3 checkpoints can be scored against the same candidates safely.

Averages similarity SCORES (not raw query vectors) across the 3 checkpoints
before the standard KNN+bandit routing eval, same safety rationale as the
unseen version.
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

PCA5_DIR = Path("local_descriptors/embedllm-ceiling-pca5")
ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSCR_ALLSEEN = 0.541
EXCLUDE = {"JaeyeonKang__CCK_Asura_v1"}
LAM_LIST = np.logspace(-4, 2, 20)
K = 20
BANDIT_BETA = 0.000001
COST_GRID_POINTS = 20

CKPT_DIRS = [
    Path("local_checkpoints/embedllm-allseen-encoder-pct30-seed0"),
    Path("local_checkpoints/embedllm-allseen-encoder-pct30-seed1"),
    Path("local_checkpoints/embedllm-allseen-encoder-pct30-seed2"),
]
REFERENCE_MEAN = 0.5652


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
    all_models = sorted(p.stem for p in PCA5_DIR.glob("*.npy"))
    models = [m for m in all_models if m not in EXCLUDE]
    costs = np.array([compute_cost(m, 0, cost_type="n_params") for m in models], dtype=np.float32)

    E = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in models]).astype(np.float32)
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    dataset = load_embedllm("test", candidates=models)
    texts = [ex["prompt"] for ex in dataset]
    label_maps = [ex["label_map"] for ex in dataset]
    print(f"{len(texts)} test prompts, {len(models)} models", flush=True)

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
        sims = embeds @ E_norm.T
        all_sims.append(sims)

        knn_costs, knn_accs, knn_Y = knn_curve(sims, models, label_maps, costs, LAM_LIST)
        m = audc_qnc_peak(knn_costs, knn_accs)
        print(f"  [{ckpt_dir.name}] solo AUDC={m['audc']:.4f} Peak={m['peak']:.4f}", flush=True)

    ensembled_sims = np.mean(all_sims, axis=0)
    knn_costs, knn_accs, knn_Y = knn_curve(ensembled_sims, models, label_maps, costs, LAM_LIST)
    ens_metrics = audc_qnc_peak(knn_costs, knn_accs)

    print("\n" + "=" * 90)
    print(f"ENSEMBLE (mean of {len(CKPT_DIRS)} similarity scores) -- EmbedLLM all-seen, pct=0.3 combined")
    print("=" * 90)
    print(f"AUDC={ens_metrics['audc']:.4f} Peak={ens_metrics['peak']:.4f} "
          f"({'BEATS' if ens_metrics['audc'] > CSCR_ALLSEEN else 'below'} CSCR {CSCR_ALLSEEN})")
    print(f"reference (mean of 3 solo AUDCs): {REFERENCE_MEAN}")

    out_path = ANALYSIS_DIR / "ensemble_allseen_pct30_results.json"
    json.dump({"ensemble": ens_metrics, "reference_solo_mean": REFERENCE_MEAN}, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
