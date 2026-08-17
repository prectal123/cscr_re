"""Plots the actual deferral curves (cost vs accuracy, the curve AUDC is
computed FROM) for V2 (full data), V1 (uniform N=24/category, 1920 probes),
and V1.5 (PCA-weighted, 1800 probes) on EmbedLLM all-seen -- overlaid on
one figure, averaged across each config's available trained-checkpoint
seeds (no retraining, pure re-evaluation).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from router.query_encoder import QueryEncoder
from router.embedllm import load_embedllm
from router.bandit import BanditStats
from router.cost_models import compute_cost
from run_audc_eval import interp_to_grid, build_cost_grid

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXCLUDE = {"JaeyeonKang__CCK_Asura_v1"}
LAM_LIST = np.logspace(-4, 2, 20)
K = 20
BANDIT_BETA = 0.000001
COST_GRID_POINTS = 20
CSCR_ALLSEEN = 0.541

CONFIGS = [
    {"name": "V2 (full Set A, ~29,673 probes)", "color": "#1f77b4",
     "fp_dir": "local_descriptors/embedllm-ceiling-pca5",
     "ckpts": [f"local_checkpoints/embedllm-allseen-encoder-pct30-seed{s}" for s in [0, 1, 2]]},
    {"name": "V1 (uniform, 1920 probes)", "color": "#ff7f0e",
     "fp_dir": "local_descriptors/embedllm-ceiling-probeN24-pca5",
     "ckpts": [f"local_checkpoints/embedllm-allseen-encoder-probeN24-pct30-seed{s}" for s in [0, 1, 2]]},
    {"name": "V1.5 (PCA-weighted, 1800 probes)", "color": "#2ca02c",
     "fp_dir": "local_descriptors/embedllm-ceiling-pcaweighted-pca5",
     "ckpts": [f"local_checkpoints/embedllm-allseen-encoder-pcaweighted-pct30-seed{s}" for s in [0, 1, 2]]
              + [f"local_checkpoints/embedllm-allseen-encoder-pcaweighted-1800-seed{s}" for s in [3, 4]]},
]


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


def eval_one_checkpoint(ckpt_dir, fp_dir, models, costs, texts, label_maps):
    E = np.stack([np.load(Path(fp_dir) / f"{m}.npy") for m in models]).astype(np.float32)
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    enc = QueryEncoder.load(ckpt_dir, proj_dim=5)
    enc.to(DEVICE)
    enc.model.eval()
    embeds = np.zeros((len(texts), 5), dtype=np.float32)
    for start in range(0, len(texts), 64):
        batch = texts[start:start + 64]
        embeds[start:start + len(batch)] = enc.encode(batch)
    sims = embeds @ E_norm.T
    return knn_curve(sims, models, label_maps, costs, LAM_LIST)


def main():
    fig, ax = plt.subplots(figsize=(8, 6))
    grid = None

    for cfg in CONFIGS:
        fp_dir = Path(cfg["fp_dir"])
        all_models = sorted(p.stem for p in fp_dir.glob("*.npy"))
        models = [m for m in all_models if m not in EXCLUDE]
        costs = np.array([compute_cost(m, 0, cost_type="n_params") for m in models], dtype=np.float32)

        dataset = load_embedllm("test", candidates=models)
        texts = [ex["prompt"] for ex in dataset]
        label_maps = [ex["label_map"] for ex in dataset]

        acc_curves = []
        for ckpt in cfg["ckpts"]:
            if not Path(ckpt).exists():
                print(f"  [skip] {ckpt} not found", flush=True)
                continue
            print(f"  evaluating {ckpt} ({len(models)} models)...", flush=True)
            c, a = eval_one_checkpoint(ckpt, fp_dir, models, costs, texts, label_maps)
            if grid is None:
                grid = build_cost_grid(c, N_grid=COST_GRID_POINTS)
            order = np.argsort(c)
            a_grid = interp_to_grid(c[order], a[order], grid)
            acc_curves.append(a_grid)

        mean_acc = np.mean(acc_curves, axis=0)
        std_acc = np.std(acc_curves, axis=0)
        ax.plot(grid, mean_acc, label=f"{cfg['name']} (n={len(acc_curves)} seeds)", color=cfg["color"], linewidth=2)
        ax.fill_between(grid, mean_acc - std_acc, mean_acc + std_acc, color=cfg["color"], alpha=0.15)
        print(f"  {cfg['name']}: AUDC={np.trapezoid(mean_acc, grid)/(grid[-1]-grid[0]):.4f}", flush=True)

    ax.axhline(CSCR_ALLSEEN, color="gray", linestyle="--", linewidth=1, label=f"CSCR reported (AUDC={CSCR_ALLSEEN})")
    ax.set_xlabel("Cost (n_params-based, normalized)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Deferral Curves by Probe Scale (EmbedLLM all-seen)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = "local_descriptors/embedllm-analysis/deferral_curves_by_probe_scale.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
