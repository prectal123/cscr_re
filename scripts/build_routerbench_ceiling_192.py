"""RouterBench Ceiling FP at CSCR's own original probe budget (192 target,
uniform per-category allocation, NO compression -- dim=86, one raw
mean-accuracy number per eval_name category, matching the current-official
convention already used for EmbedLLM's 192-probe fairness point and
LLMRouterBench's probe-scale sweep).

routerbench-ceiling (the existing default dir) uses ALL of Set A per
category (unlimited, effectively "Pure V2"). routerbench-ceiling-v15 is the
older PCA-weighted/binned probe-limited FP used for the current 1800-probe
headline. This fills the gap: COMPAR's own probe-limited FP under the
uniform-allocation+uncompressed convention, at the SAME 192-probe budget
CSCR's paper itself reports using for RouterBench.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
import routerbench_knn_test as rb

OUT_DIR = Path("local_descriptors/routerbench-ceiling-uniform-nocompress-192")
TARGET_TOTAL = 192
MIN_PROBES = 1


def solve_allocation_uniform(sizes_by_cat, categories, target_total, min_n):
    lo, hi = 0.0, float(max(sizes_by_cat[c] for c in categories))

    def total_for_quota(q):
        return sum(min(q, sizes_by_cat[c]) for c in categories)

    for _ in range(60):
        mid = (lo + hi) / 2
        if total_for_quota(mid) < target_total:
            lo = mid
        else:
            hi = mid
    quota = hi
    alloc = {c: max(min_n, int(round(min(quota, sizes_by_cat[c])))) for c in categories}
    return alloc, sum(alloc.values())


def main():
    set_a, set_b = rb.load_data()
    models, cols = rb.NAMES, rb.MODELS
    categories = sorted(set_a["eval_name"].unique())

    scores = set_a[cols].to_numpy(dtype=np.float64)  # (n_rows, 11)
    eval_names = set_a["eval_name"].to_numpy()
    var = scores.var(axis=1)

    sizes_by_cat = {c: int((eval_names == c).sum()) for c in categories}
    allocation, total = solve_allocation_uniform(sizes_by_cat, categories, TARGET_TOTAL, MIN_PROBES)
    print(f"UNIFORM ALLOCATION (target={TARGET_TOTAL}, actual={total}), {len(categories)} categories", flush=True)

    n_models = len(models)
    raw = np.zeros((n_models, len(categories)))
    for ci, cat in enumerate(categories):
        cat_mask = eval_names == cat
        cat_idx = np.where(cat_mask)[0]
        cat_var = var[cat_idx]
        order = np.argsort(-cat_var)
        n = allocation[cat]
        selected = cat_idx[order[:n]]
        raw[:, ci] = scores[selected].mean(axis=0)

    pool_mean = raw.mean(axis=0, keepdims=True)
    centered = raw - pool_mean

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for j, m in enumerate(models):
        vec = centered[j] / (np.linalg.norm(centered[j]) + 1e-12)
        np.save(OUT_DIR / f"{m}.npy", vec.astype(np.float32))

    E = np.stack([np.load(OUT_DIR / f"{m}.npy") for m in models])
    sim = E @ E.T
    off = sim[~np.eye(n_models, dtype=bool)]
    print(f"192-uniform-nocompress Ceiling: shape={E.shape} actual_total={total} "
          f"cos_sim mean={off.mean():.4f} std={off.std():.4f} -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
