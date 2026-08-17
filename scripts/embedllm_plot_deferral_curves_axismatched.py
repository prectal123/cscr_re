"""Same deferral-curve plot as embedllm_plot_deferral_curves_by_probe_scale.py,
but with the x-axis converted back to raw parameter count (billions) --
dividing out the internal *0.03 cost-scaling convention -- and axis
limits/style matched to the CSCR paper's Figure (a) EmbedLLM deferral plot
(x: 0-72B params, y: 0.20-0.60 accuracy), so the user can overlay the two
images directly in an image tool for a visual, apples-to-apples-scale
comparison (AUDC itself is already scale-invariant to this conversion --
this is purely for visual alignment, not a new metric).

Caveat carried into the plot as a text note: the CSCR curve in the paper
has no disclosed seed count, while ours is an average over multiple
trained-checkpoint seeds -- a single-run-vs-multi-seed-average comparison,
not a fully matched experimental protocol.
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
K = 200  # effectively no top-K pre-filter (n_models=111 < 200) -- lets lambda alone decide the full cost range
BANDIT_BETA = 0.000001
COST_GRID_POINTS = 20
COST_SCALE_FACTOR = 0.03  # compute_cost(..., cost_type="n_params") = n_params * 0.03

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
    fig, ax = plt.subplots(figsize=(7, 6))
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
        grid_raw_params = grid / COST_SCALE_FACTOR  # convert back to raw param-billions
        ax.plot(grid_raw_params, mean_acc, label=f"{cfg['name']} (n={len(acc_curves)} seeds)",
                 color=cfg["color"], linewidth=2, marker="o", markersize=3)
        ax.fill_between(grid_raw_params, mean_acc - std_acc, mean_acc + std_acc, color=cfg["color"], alpha=0.15)
        print(f"  {cfg['name']}: AUDC={np.trapezoid(mean_acc, grid)/(grid[-1]-grid[0]):.4f} "
              f"(unchanged by axis rescaling)", flush=True)

    # match CSCR paper Figure (a) EmbedLLM axis framing
    ax.set_xlim(0, 72)
    ax.set_ylim(0.20, 0.60)
    ax.set_xlabel("# Params (B)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Combined GRPO -- EmbedLLM all-seen\n(axis-matched to CSCR paper Fig. (a) for overlay)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.02, "Note: CSCR curve's seed count is undisclosed;\nours is averaged over multiple trained seeds.",
             transform=ax.transAxes, fontsize=7, color="gray", va="bottom")
    fig.tight_layout()

    out_path = "local_descriptors/embedllm-analysis/deferral_curves_axismatched_to_cscr_fig.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
