"""User's refined hypothesis: what matters for unseen-model routing isn't
descriptor-space isolation per se, but whether the held-out model has enough
TRUE-capability-similar neighbors in the training pool -- FP-space proximity
is only useful as a "beacon" if it actually reflects real capability
similarity. A descriptor could place a model near others that are close in
cosine-sim but have nothing to do with its actual per-prompt correctness
pattern, in which case that "closeness" is not a real beacon at all.

This computes, for each pair of RouterBench models, the TRUE capability
correlation (Pearson/phi coefficient of their binary correctness vectors
across all of Set A) -- independent of any descriptor -- then compares each
model's true-capability neighbors against its Ceiling-FP descriptor
neighbors (from descriptor_isolation_check.py) to see whether FP-space
closeness actually tracks real capability closeness for the failing models.
"""
import numpy as np

import routerbench_knn_test as rb

NAMES = rb.NAMES
MODELS = rb.MODELS


def main():
    set_a, set_b = rb.load_data()

    # true capability correctness matrix: (11 models) x (n_setA rows), binary
    correctness = np.stack([set_a[m].to_numpy(dtype=float) for m in MODELS], axis=0)
    correctness = (correctness >= 1.0).astype(float)

    n = len(NAMES)
    true_corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                true_corr[i, j] = -np.inf
                continue
            true_corr[i, j] = np.corrcoef(correctness[i], correctness[j])[0, 1]

    print(f"{'='*70}\nTRUE capability correlation (Pearson/phi on Set A binary correctness)\n{'='*70}")
    rows = []
    for i, name in enumerate(NAMES):
        order = np.argsort(-true_corr[i])[:3]
        top3_names = [NAMES[j] for j in order]
        top3_vals = true_corr[i, order]
        true_beacon_density = float(top3_vals.mean())
        rows.append((name, true_beacon_density, top3_names, top3_vals))

    rows.sort(key=lambda r: r[1])
    for name, tbd, top3_names, top3_vals in rows:
        flag = "  <-- LOO FAILED (0%, consistently)" if name in ("zero-one-ai__Yi-34B-Chat", "mistralai__mistral-7b-chat") else ""
        flag2 = "  <-- LOO UNSTABLE (great at seed0, ~0 at seed1/2)" if name in ("WizardLM__WizardLM-13B-V1.2", "claude-instant-v1") else ""
        top3_str = ", ".join(f"{nm}={v:+.3f}" for nm, v in zip(top3_names, top3_vals))
        print(f"  {name:38s} true_beacon_density(top3 avg corr)={tbd:+.4f}{flag}{flag2}")
        print(f"      top3 true-capability neighbors: {top3_str}")

    # base rate: overall mean pairwise correlation (excluding diagonal)
    off = true_corr[true_corr != -np.inf]
    print(f"\noverall mean pairwise true-capability correlation (all 110 pairs): {off.mean():.4f}")


if __name__ == "__main__":
    main()
