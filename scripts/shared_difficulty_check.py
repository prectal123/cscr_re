"""Check how much of each model's raw per-eval_name accuracy pattern is just
the pool-shared "task difficulty" component that Ceiling FP's mean-centering
step explicitly removes (see build_ceiling_fp_centered.py / routerbench_knn_test.
build_ceiling_fp). If a model's raw accuracy vector correlates very strongly
with the pool-mean vector, most of its variance IS shared difficulty -- and
after mean-centering, little idiosyncratic (model-specific) signal survives
for the query encoder to latch onto, even if it has close neighbors in FP space.
"""
import numpy as np
from scipy.stats import pearsonr

import routerbench_knn_test as rb

NAMES = rb.NAMES
MODELS = rb.MODELS


def main():
    set_a, set_b = rb.load_data()
    eval_names = sorted(set_a["eval_name"].unique())
    name_to_idx = {e: i for i, e in enumerate(eval_names)}

    raw = {}
    for name, model_col in zip(NAMES, MODELS):
        vec = np.zeros(len(eval_names))
        counts = np.zeros(len(eval_names))
        for ev, score in zip(set_a["eval_name"], set_a[model_col]):
            vec[name_to_idx[ev]] += float(score)
            counts[name_to_idx[ev]] += 1
        raw[name] = vec / np.maximum(counts, 1)

    pool_matrix = np.stack([raw[n] for n in NAMES])
    pool_mean = pool_matrix.mean(axis=0)

    print(f"{'='*70}\nHow much of each model's raw accuracy pattern is shared difficulty?\n{'='*70}")
    rows = []
    for name in NAMES:
        r, p = pearsonr(raw[name], pool_mean)
        centered = raw[name] - pool_mean
        residual_std = centered.std()
        rows.append((name, r, residual_std))

    rows.sort(key=lambda x: -x[1])  # highest correlation with shared difficulty first
    for name, r, resid_std in rows:
        flag = "  <-- weak/unstable AUC" if name in ("mistralai__mistral-7b-chat", "claude-v2") else ""
        flag2 = "  <-- strong/stable AUC" if name in ("gpt-4-1106-preview", "meta__code-llama-instruct-34b-chat") else ""
        print(f"  {name:38s} corr_with_pool_mean(R)={r:+.4f}  residual_std_after_centering={resid_std:.4f}{flag}{flag2}")


if __name__ == "__main__":
    main()
