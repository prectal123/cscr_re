"""Compare Perplexity / Ceiling / V1.2 FPs via Area-Under-Deferral-Curve (AUDC).

Unlike loo_unseen_recovery.py (which trains on 10-of-11 to test unseen-model
generalization), this trains ONE query encoder per FP type on the FULL 11-model
pool -- the standard "how good is routing with this descriptor" question -- then
sweeps a cost-weight lambda over Set B (held-out prompts) to trace an
(cost, accuracy) deferral curve, same idea as PROGRESS.md section 13's
run_audc_eval.py, but self-contained here so it's consistent with all the fixes
from this session (mean pooling, MARGIN recalibration, GPU).

For each FP type, for each lambda in a sweep:
  chosen_model(prompt) = argmax_i [ cos_sim(q, e_i) - lambda * cost_norm_i ]
  accuracy(lambda) = mean bartscore(chosen_model, prompt) over Set B
  cost(lambda)     = mean cost_norm(chosen_model) over Set B
Oracle does the same argmax but using the TRUE bartscore instead of the router's
similarity -- the best achievable frontier. Random is a flat line (expected
value of picking uniformly at random, independent of lambda).

AUDC = trapezoidal area under the (cost, accuracy) curve over a shared cost range,
higher is better.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import loo_unseen_recovery as base
from router.utils import load_descriptors, load_cost_dict

OUT_DIR = Path("local_descriptors/analysis")
N_LAMBDAS = 41
LAMBDA_MAX = 3.0


def compute_deferral_curve(sims, true_scores, cost_norm, lambdas):
    """sims: (N, M) router similarity scores. true_scores: (N, M) real bartscore
    (exp-transformed). cost_norm: (M,). Returns (costs, accs) arrays over lambdas."""
    costs, accs = [], []
    for lam in lambdas:
        chosen = (sims - lam * cost_norm[None, :]).argmax(axis=1)
        accs.append(true_scores[np.arange(len(chosen)), chosen].mean())
        costs.append(cost_norm[chosen].mean())
    return np.array(costs), np.array(accs)


def audc(costs, accs, cost_lo, cost_hi):
    """Trapezoidal area under accs(costs) restricted to [cost_lo, cost_hi],
    after sorting by cost and averaging duplicate accs at the same cost."""
    order = np.argsort(costs)
    c, a = costs[order], accs[order]
    mask = (c >= cost_lo) & (c <= cost_hi)
    if mask.sum() < 2:
        return float("nan")
    return float(np.trapz(a[mask], c[mask]))


def main():
    print(f"DEVICE: {base.DEVICE}")
    pool_11 = json.load(open(base.POOL_PATH))

    base._TOKENIZER = AutoTokenizer.from_pretrained(base.EMBED_MODEL)
    base._BASE_MODEL = AutoModel.from_pretrained(base.EMBED_MODEL).to(base.DEVICE)
    base._BASE_MODEL.eval()

    print("Building Set B eval subsample...")
    ids, per_prompt = base.load_set_b(pool_11)
    print("Precomputing mean-pooled embeddings for Set B...")
    cls_embeds = base.precompute_embeddings(ids, per_prompt, base._TOKENIZER, base._BASE_MODEL)

    print("Parsing MixInstruct 'train' split once...")
    train_rows = base.build_train_index(pool_11)

    lambdas = np.linspace(0, LAMBDA_MAX, N_LAMBDAS)

    results = {}
    for fp_name, desc_dir in [("Perplexity", base.PERP_DIR), ("Ceiling", base.CEILING_DIR), ("V1.2", base.V12_DIR)]:
        print(f"\n{'='*60}\nFP type: {fp_name}\n{'='*60}")
        # train_fold's desc_names comes from load_descriptors' os.walk order, NOT
        # necessarily pool_11's order -- use desc_names as the authoritative column
        # order for everything downstream (E, cost, true_scores) instead of assuming
        # it matches pool_11.
        head, desc_names = base.train_fold(pool_11, desc_dir, train_rows)
        assert set(desc_names) == set(pool_11), f"pool mismatch: {desc_names} vs {pool_11}"

        E, _ = load_descriptors(str(desc_dir), pool=desc_names)
        E = np.stack(E)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

        cost = load_cost_dict(desc_names, cost_type="n_params")
        cost_arr = np.array([cost[n] for n in desc_names], dtype=np.float32)
        cost_norm = (cost_arr - cost_arr.min()) / (cost_arr.max() - cost_arr.min() + 1e-9)

        head.eval()
        with torch.no_grad():
            q = head(torch.from_numpy(cls_embeds).float().to(base.DEVICE))
            E_t = torch.from_numpy(E).float().to(base.DEVICE)
            sims = (q @ E_t.T).cpu().numpy()  # (N, 11)

        true_scores = np.array([[math.exp(per_prompt[pid]["scores"][m]) for m in desc_names] for pid in ids])

        router_costs, router_accs = compute_deferral_curve(sims, true_scores, cost_norm, lambdas)
        oracle_costs, oracle_accs = compute_deferral_curve(true_scores, true_scores, cost_norm, lambdas)
        random_acc = true_scores.mean()
        random_cost = cost_norm.mean()

        cost_lo, cost_hi = float(cost_norm.min()), float(cost_norm.max())
        router_audc = audc(router_costs, router_accs, cost_lo, cost_hi)
        oracle_audc = audc(oracle_costs, oracle_accs, cost_lo, cost_hi)
        random_audc = random_acc * (cost_hi - cost_lo)  # flat line reference area

        print(f"router AUDC={router_audc:.5f}  oracle AUDC={oracle_audc:.5f}  "
              f"random(flat) AUDC={random_audc:.5f}")
        print(f"router avg acc range: [{router_accs.min():.4f}, {router_accs.max():.4f}]  "
              f"vs random flat acc={random_acc:.4f}")

        results[fp_name] = {
            "lambdas": lambdas.tolist(),
            "router_costs": router_costs.tolist(),
            "router_accs": router_accs.tolist(),
            "oracle_costs": oracle_costs.tolist(),
            "oracle_accs": oracle_accs.tolist(),
            "random_acc": float(random_acc),
            "random_cost": float(random_cost),
            "router_audc": router_audc,
            "oracle_audc": oracle_audc,
            "random_audc": random_audc,
            "cost_range": [cost_lo, cost_hi],
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "audc_compare_fps.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"{'FP':12s} {'router AUDC':>12s} {'oracle AUDC':>12s} {'random AUDC':>12s} {'router/random':>14s}")
    for fp_name, r in results.items():
        ratio = r["router_audc"] / r["random_audc"] if r["random_audc"] else float("nan")
        print(f"{fp_name:12s} {r['router_audc']:12.5f} {r['oracle_audc']:12.5f} "
              f"{r['random_audc']:12.5f} {ratio:14.3f}")

    print(f"\nSaved -> {OUT_DIR / 'audc_compare_fps.json'}")


if __name__ == "__main__":
    main()
