"""Re-evaluate the 3 already-trained all-seen min-pos checkpoints (no
retraining needed -- training uses no cost information, only routing
decisions at eval time do) after fixing two registry bugs found during
audit:
  1. get_param_count() truncated to int(), zeroing the cost of every
     sub-1B model (e.g. Qwen1.5-0.5B-Chat, n_params=0.5 -> cost 0).
  2. Three models were entirely missing from experts/registry.json,
     also defaulting to cost=0: microsoft/phi-1_5 (now added, 1.42B),
     cloudyu/Mixtral_11Bx2_MoE_19B (now added, 19.19B), and
     JaeyeonKang/CCK_Asura_v1 -- whose HF repo no longer exists at all
     (401 Repository Not Found) -- EXCLUDED from the pool entirely,
     same treatment as dolly-v2-12b/mpt-7b-instruct earlier in this
     project.
At lambda=1.0 pre-fix, these 4 zero-cost models absorbed 95.5% of all
routing; at lambda=100, 100%. This script checks how much that inflated
the reported all-seen AUDC=0.579-0.585 vs CSCR's 0.541.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from router.query_encoder import QueryEncoder
from router.embedllm import load_embedllm
from router.bandit import BanditStats
from router.cost_models import compute_cost
from run_audc_eval import interp_to_grid, build_cost_grid, paired_bootstrap_audc_cached

PCA5_DIR = Path("local_descriptors/embedllm-ceiling-pca5")
ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
EXCLUDE = {"JaeyeonKang__CCK_Asura_v1"}  # HF repo no longer exists
LAM_LIST = np.logspace(-4, 2, 20)
K = 20
BANDIT_BETA = 0.000001
COST_GRID_POINTS = 20
CSCR_PAPER_ALLSEEN = 0.541
SEEDS = [0, 1, 2]


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


def random_curve(models, label_maps, costs, lam_list, seed=0):
    import random as pyrandom
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
    audc = np.trapezoid(a_grid, grid) / (grid[-1] - grid[0])
    peak_idx = np.argmax(a)
    return {"audc": float(audc), "qnc": float(c[peak_idx]), "peak": float(a[peak_idx])}


def main():
    all_models = sorted(p.stem for p in PCA5_DIR.glob("*.npy"))
    models = [m for m in all_models if m not in EXCLUDE]
    print(f"Pool: {len(all_models)} -> {len(models)} after excluding {EXCLUDE}", flush=True)

    costs = np.array([compute_cost(m, 0, cost_type="n_params") for m in models], dtype=np.float32)
    zero_cost = [m for m, c in zip(models, costs) if c == 0]
    print(f"Models still at cost=0 after fix: {zero_cost}", flush=True)
    print(f"Cost range: min={costs.min():.4f} max={costs.max():.4f} mean={costs.mean():.4f}", flush=True)

    dataset = load_embedllm("test", candidates=models)
    texts = [ex["prompt"] for ex in dataset]
    label_maps = [ex["label_map"] for ex in dataset]
    print(f"{len(texts)} test prompts, {len(models)} candidate models", flush=True)

    E = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in models]).astype(np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    results = []
    for seed in SEEDS:
        ckpt_dir = Path(f"local_checkpoints/embedllm-allseen-minpos-seed{seed}")
        enc = QueryEncoder.load(str(ckpt_dir), proj_dim=5)
        enc.model.eval()

        embeds = np.zeros((len(texts), 5), dtype=np.float32)
        for start in range(0, len(texts), 64):
            batch = texts[start:start + 64]
            embeds[start:start + len(batch)] = enc.encode(batch)
        sims = embeds @ E.T

        t0 = time.time()
        knn_costs, knn_accs, knn_Y = knn_curve(sims, models, label_maps, costs, LAM_LIST)
        rand_costs, rand_accs, rand_Y = random_curve(models, label_maps, costs, LAM_LIST, seed=seed)
        knn_metrics = audc_qnc_peak(knn_costs, knn_accs)
        rand_metrics = audc_qnc_peak(rand_costs, rand_accs)
        mean_delta, (lo, hi), p = paired_bootstrap_audc_cached(knn_costs, knn_Y, rand_costs, rand_Y, B=2000, seed=0)

        r = {"seed": seed, "knn": knn_metrics, "random": rand_metrics,
             "bootstrap_delta": float(mean_delta), "bootstrap_p": float(p)}
        results.append(r)
        print(f"[seed={seed}] AUDC={knn_metrics['audc']:.4f} QNC={knn_metrics['qnc']:.3f} "
              f"Peak={knn_metrics['peak']:.4f}  ({'BEATS' if knn_metrics['audc'] > CSCR_PAPER_ALLSEEN else 'below'} "
              f"CSCR {CSCR_PAPER_ALLSEEN})  [{time.time()-t0:.1f}s]", flush=True)

    out_path = ANALYSIS_DIR / "allseen_minpos_multiseed_costfixed_results.json"
    json.dump(results, open(out_path, "w"), indent=2)

    audcs = [r["knn"]["audc"] for r in results]
    print("\n" + "=" * 80)
    print(f"COST-FIXED ALL-SEEN SUMMARY vs CSCR {CSCR_PAPER_ALLSEEN}")
    print("=" * 80)
    print(f"mean AUDC={np.mean(audcs):.4f} (std={np.std(audcs):.4f})")
    print(f"beats CSCR: {sum(1 for a in audcs if a > CSCR_PAPER_ALLSEEN)}/{len(audcs)}")
    print("\nPre-fix reference (buggy costs): 0.5792 / 0.5847 / 0.5834, mean=0.5824")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
