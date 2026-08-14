"""Free, local ablation: how many top-variance probes/category does a
Ceiling-V1-style FP (GT scores, mean-centered + L2-normalized) need before
kNN unseen-recovery performance degrades? Uses the EXISTING probe_info.json
(528 probes, 24/category, already variance-sorted descending within each
category) -- for each N in the sweep, keep only the top-N probes/category,
rebuild the FP, and run the same kNN test as knn_test_lite20.py. No LLM
judge calls, no cost -- pure resampling of already-computed Set A scores.

Motivation: today's V1.3 (LLM-judge, 3 probes/category) lost significantly to
Ceiling V2 (full-category average) in kNN. Before spending money scaling up
the LLM-judge pipeline, find the probe-count floor using free GT-based data
as a stand-in for "a perfect judge" -- this pins down both (a) whether 3/cat
was simply too few, and (b) the minimum viable count worth paying for later.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import spearmanr

sys.path.insert(0, "scripts/llmrouterbench")
import common_lite20 as common

DATA_DIR = Path("local_descriptors/llmrouterbench_lite20")
N_SWEEP = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24]


def load_setA_setB():
    with open(DATA_DIR / "setA_setB_split.pkl", "rb") as f:
        split = pickle.load(f)
    return split["setA"], split["setB"]


def build_setB_eval(setB):
    queries, scores = [], []
    for ds in common.DATASETS:
        d = setB[ds]
        queries.extend(d["queries"])
        scores.append(d["scores"])
    return queries, np.concatenate(scores, axis=0)


def build_fp_for_n(probe_info, setA, n_per_category):
    """Top-n_per_category probes/category (already variance-sorted desc) -> mean-centered, L2-normalized FP."""
    by_ds = {}
    for p in probe_info:
        by_ds.setdefault(p["dataset"], []).append(p)

    rows = []
    for ds in common.DATASETS:
        entries = by_ds[ds][:n_per_category]  # already descending by var
        for e in entries:
            rows.append(setA[ds]["scores"][e["local_idx_in_setA"], :])  # (20,)
    raw = np.stack(rows)  # (n_probes_total, 20)

    pool_mean = raw.mean(axis=1, keepdims=True)
    centered = raw - pool_mean
    E = centered.T  # (20, n_probes_total)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    return E.astype(np.float32), raw.shape[0]


def knn_test(true_scores, E, pool):
    sim_full = E @ E.T
    fp_rhos, uniform_rhos = [], []
    for i in range(len(pool)):
        others_idx = [j for j in range(len(pool)) if j != i]
        other_scores = true_scores[:, others_idx]
        true_m = true_scores[:, i]
        sims = sim_full[i, others_idx]
        w = np.clip(sims, 0, None)
        if w.sum() < 1e-9:
            w = np.ones_like(w)
        w = w / w.sum()
        fp_proxy = other_scores @ w
        uniform_proxy = other_scores.mean(axis=1)
        fp_rho, _ = spearmanr(fp_proxy, true_m)
        uni_rho, _ = spearmanr(uniform_proxy, true_m)
        fp_rhos.append(fp_rho)
        uniform_rhos.append(uni_rho)
    return np.array(fp_rhos), np.array(uniform_rhos)


def main():
    with open(DATA_DIR / "probe_info.json", encoding="utf-8") as f:
        probe_info = json.load(f)
    setA, setB = load_setA_setB()
    _, true_scores = build_setB_eval(setB)
    pool = common.MODELS_20

    print(f"{'N/category':>10s} {'total probes':>13s} {'mean FP rho':>12s} {'mean uniform':>13s} "
          f"{'delta':>9s} {'p-value':>9s} {'improved':>9s}")
    results = []
    for n in N_SWEEP:
        E, n_total = build_fp_for_n(probe_info, setA, n)
        fp_rhos, uniform_rhos = knn_test(true_scores, E, pool)
        delta = fp_rhos - uniform_rhos
        t, p = stats.ttest_rel(fp_rhos, uniform_rhos)
        n_improved = int((delta > 0).sum())
        print(f"{n:>10d} {n_total:>13d} {fp_rhos.mean():>12.4f} {uniform_rhos.mean():>13.4f} "
              f"{delta.mean():>+9.4f} {p:>9.4f} {n_improved:>6d}/20")
        results.append({
            "n_per_category": n, "n_total_probes": n_total,
            "mean_fp_rho": float(fp_rhos.mean()), "mean_uniform_rho": float(uniform_rhos.mean()),
            "mean_delta": float(delta.mean()), "p_value": float(p), "n_improved": n_improved,
        })

    out_path = DATA_DIR / "probe_count_ablation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
