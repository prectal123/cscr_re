"""Check descriptor-space isolation for Ceiling and Perplexity FPs on RouterBench.

Motivation: LOO results (PROGRESS.md section 16.4/16.5) show Yi-34B-Chat and
mistral-7b-chat recover ~0% when held out, despite being heavily favored
during joint (all-11) training. Hypothesis: their descriptor sits far from
every other model's descriptor, so a query encoder trained only on the other
10 has nothing nearby to generalize toward, regardless of how "strong" the
model's collapse-driven in-pool share was.
"""
import numpy as np
from pathlib import Path

import routerbench_knn_test as rb

CEILING_DIR = rb.CEILING_DIR
PERP_DIR = Path("local_descriptors/routerbench-perplexity")
NAMES = rb.NAMES


def analyze(desc_dir, fp_name):
    E = np.stack([np.load(desc_dir / f"{n}.npy") for n in NAMES])
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    sim = E @ E.T
    np.fill_diagonal(sim, -np.inf)

    print(f"\n{'='*70}\n{fp_name} FP -- isolation / beacon-density per model (top-3 neighbors)\n{'='*70}")
    rows = []
    for i, name in enumerate(NAMES):
        others = sim[i][sim[i] != -np.inf]
        mean_sim = others.mean()
        order = np.argsort(-sim[i])[:3]  # top-3 nearest neighbor indices
        top3_sims = sim[i, order]
        top3_names = [NAMES[j] for j in order]
        beacon_density_top3 = float(top3_sims.mean())  # "how many strong friends nearby", not just 1
        rows.append((name, mean_sim, beacon_density_top3, top3_names, top3_sims))

    rows.sort(key=lambda r: r[2])  # most isolated by beacon-density (top-3 avg) first
    for name, mean_sim, beacon_density_top3, top3_names, top3_sims in rows:
        flag = "  <-- LOO FAILED (0%)" if name in ("zero-one-ai__Yi-34B-Chat", "mistralai__mistral-7b-chat") else ""
        top3_str = ", ".join(f"{n}={s:+.3f}" for n, s in zip(top3_names, top3_sims))
        print(f"  {name:38s} beacon_density(top3 avg)={beacon_density_top3:+.4f}  mean_all={mean_sim:+.4f}{flag}")
        print(f"      top3: {top3_str}")

    return rows


def main():
    analyze(CEILING_DIR, "Ceiling")
    analyze(PERP_DIR, "Perplexity")


if __name__ == "__main__":
    main()
