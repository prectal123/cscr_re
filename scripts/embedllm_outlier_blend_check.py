"""Check the "outlier drags the blended target into empty space" hypothesis,
using only existing FP embeddings + labels -- no GPU, no training needed.

For each training query:
  1. Build the same GRPO target t_m = (label_m - mean)/(std+eps) over seen
     models (embedllm_newllm_grpo_train.py's exact construction).
  2. Build the advantage-weighted blended target z* = normalize(sum_m t_m * E_m)
     (only positive-advantage models contribute meaningfully once weighted).
  3. Measure how far z* lands from the NEAREST actual seen-model FP
     (dist_to_nearest_real) -- if the blend lands in genuinely empty space,
     this will be large relative to typical inter-model distances.
  4. Measure the "spread" of the positive-advantage subset (mean pairwise
     distance among models with t_m > 0) -- this is the hypothesized cause.

Then correlate (3) vs (4) across queries: if the hypothesis holds, wider
positive-set spread should predict the blend landing farther from any real
model.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr

sys.path.insert(0, "src")

PCA5_DIR = Path("local_descriptors/embedllm-ceiling-pca5")
ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
SPLIT_PATH = ANALYSIS_DIR / "newllm_split.json"

N_SAMPLE = 5000  # subsample queries for speed; full set is ~30k


def main():
    split = json.load(open(SPLIT_PATH, encoding="utf-8"))
    seen_models = split["seen"]
    print(f"seen models: {len(seen_models)}")

    E = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in seen_models]).astype(np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    n_models = len(seen_models)

    # typical inter-model distance, for scale reference
    pairwise = np.linalg.norm(E[:, None, :] - E[None, :, :], axis=-1)
    iu = np.triu_indices(n_models, k=1)
    typical_dist = pairwise[iu].mean()
    print(f"typical pairwise distance among seen models: {typical_dist:.4f} "
          f"(min={pairwise[iu].min():.4f}, max={pairwise[iu].max():.4f})")

    print("Loading EmbedLLM train.csv (should be cached already)...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "label", "prompt"])

    name_to_idx = {n: i for i, n in enumerate(seen_models)}
    rng = np.random.RandomState(0)

    spreads, dists, n_positive_list, n_total_list = [], [], [], []
    outlier_examples = []

    groups = list(df.groupby("prompt_id", sort=False))
    rng.shuffle(groups)
    used = 0
    for pid, grp in groups:
        if used >= N_SAMPLE:
            break
        labels = np.full(n_models, np.nan, dtype=np.float32)
        for m, v in zip(grp["model_name"], grp["label"]):
            if m in name_to_idx:
                labels[name_to_idx[m]] = float(v)
        mask = ~np.isnan(labels)
        if mask.sum() < 4:  # need a few positive+negative for a meaningful check
            continue
        vals = labels[mask]
        mean, std = vals.mean(), vals.std()
        if std < 1e-6:
            continue  # all-same-label query: no positive/negative split, skip (matches Dr.GRPO's "std->0 degenerate" case too)
        t = np.zeros(n_models, dtype=np.float32)
        t[mask] = (vals - mean) / (std + 1e-6)

        pos_idx = np.where(t > 0)[0]
        if len(pos_idx) < 2:
            continue  # need >=2 positive models to talk about "spread"

        # blended target: advantage-weighted combination (matches the
        # gradient-implied direction of the current per-model MSE loss)
        z = (t[:, None] * E).sum(axis=0)
        z = z / (np.linalg.norm(z) + 1e-12)

        dist_to_nearest_real = np.linalg.norm(E - z[None, :], axis=1).min()

        pos_E = E[pos_idx]
        pos_pairwise = np.linalg.norm(pos_E[:, None, :] - pos_E[None, :, :], axis=-1)
        piu = np.triu_indices(len(pos_idx), k=1)
        spread = pos_pairwise[piu].mean() if len(piu[0]) > 0 else 0.0

        spreads.append(spread)
        dists.append(dist_to_nearest_real)
        n_positive_list.append(len(pos_idx))
        n_total_list.append(int(mask.sum()))
        if spread > typical_dist * 1.3 and len(outlier_examples) < 8:
            outlier_examples.append({
                "prompt_id": str(pid), "spread": float(spread),
                "dist_to_nearest_real": float(dist_to_nearest_real),
                "n_positive": int(len(pos_idx)), "n_total": int(mask.sum()),
                "positive_models": [seen_models[i] for i in pos_idx],
            })
        used += 1

    spreads, dists = np.array(spreads), np.array(dists)
    print(f"\nUsable queries: {len(spreads)}")
    print(f"positive-set spread: mean={spreads.mean():.4f} std={spreads.std():.4f} "
          f"(typical inter-model dist={typical_dist:.4f})")
    print(f"dist_to_nearest_real: mean={dists.mean():.4f} std={dists.std():.4f} "
          f"min={dists.min():.4f} max={dists.max():.4f}")

    rho, p = spearmanr(spreads, dists)
    print(f"\nSpearman rho(positive_set_spread, dist_to_nearest_real) = {rho:.4f}  p={p:.4g}")
    print("(positive rho = hypothesis SUPPORTED: wider-spread 'correct' models drag the "
          "blended target farther from any real model)")

    # split into wide-spread vs tight-spread halves, compare dist_to_nearest_real
    median_spread = np.median(spreads)
    wide = dists[spreads >= median_spread]
    tight = dists[spreads < median_spread]
    print(f"\nWide-spread half (n={len(wide)}): mean dist_to_nearest_real = {wide.mean():.4f}")
    print(f"Tight-spread half (n={len(tight)}): mean dist_to_nearest_real = {tight.mean():.4f}")
    print(f"Ratio (wide/tight) = {wide.mean()/tight.mean():.2f}x")

    print(f"\n{len(outlier_examples)} example queries with unusually wide positive-set spread:")
    for ex in outlier_examples:
        print(f"  spread={ex['spread']:.3f} dist_to_nearest={ex['dist_to_nearest_real']:.3f} "
              f"n_pos={ex['n_positive']}/{ex['n_total']}  models={ex['positive_models'][:5]}"
              f"{'...' if len(ex['positive_models']) > 5 else ''}")

    result = {
        "typical_inter_model_dist": float(typical_dist),
        "n_queries": len(spreads),
        "spread_mean": float(spreads.mean()), "spread_std": float(spreads.std()),
        "dist_mean": float(dists.mean()), "dist_std": float(dists.std()),
        "spearman_rho": float(rho), "spearman_p": float(p),
        "wide_half_mean_dist": float(wide.mean()), "tight_half_mean_dist": float(tight.mean()),
        "outlier_examples": outlier_examples,
    }
    out_path = ANALYSIS_DIR / "outlier_blend_check_results.json"
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
