"""Free, local test: does model "weight class" (flagship/large vs small)
produce a clean, statistically distinguishable gap signal on RouterBench's
11-model pool -- and if so, is that gap ORTHOGONAL to domain-specific
(within-tier) signal, or does it just collapse into the same axis?

Motivation: mentor feedback suggested flagship models should participate in
FP-building (currently LLMRouterBench-lite20 is all 7-9B open models, no
flagship). Before paying for new flagship evals on that benchmark, check the
underlying hypothesis cheaply on RouterBench, which already mixes flagship
(gpt-4-1106-preview, claude-v2) with small open models (mistral-7b-chat,
WizardLM-13B-V1.2) on the same eval_name-tagged tasks with real GT scores.

Method:
  1. Rebuild the per-eval_name mean-centered category matrix (same as
     build_ceiling_fp in routerbench_knn_test.py, stopping BEFORE the
     per-model L2 normalization so magnitude is preserved).
  2. group_axis = mean(large-tier centered vectors) - mean(small-tier
     centered vectors) -- the raw direction/magnitude of the tier gap.
  3. Permutation test: is ||group_axis|| for the REAL large/small split
     bigger than for random 7-vs-4 splits of the same 11 models? (n=11 is
     small -- this checks the split isn't just an artifact of splitting.)
  4. within-group residuals: each model's vector minus its OWN tier's mean
     (isolates domain-specific deviation, controlling for tier). PCA on
     residuals -> "domain axes" ranked by variance explained.
  5. Orthogonality: cosine sim of group_axis against each domain axis, plus
     what fraction of group_axis's squared norm the domain-axis subspace
     captures (low = orthogonal, high = entangled).
  6. Complementary view: full-pool PCA (all 11 models, group-agnostic, same
     style as pca_dim_ablation.py) -- does PC1 already align with the tier
     label? Report each model's PC1/PC2 score next to its group.

Caveat: only 11 models, split 7 (large) / 4 (small) -- likely underpowered
for hard significance claims (see [[project_routerbench_small_class_count]]).
This is a directional/exploratory check, not a confirmatory one.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

MODELS = [
    'claude-instant-v1', 'claude-v1', 'claude-v2', 'gpt-3.5-turbo-1106',
    'gpt-4-1106-preview', 'meta/code-llama-instruct-34b-chat',
    'meta/llama-2-70b-chat', 'mistralai/mistral-7b-chat',
    'mistralai/mixtral-8x7b-chat', 'zero-one-ai/Yi-34B-Chat',
    'WizardLM/WizardLM-13B-V1.2',
]
NAMES = [m.replace("/", "__") for m in MODELS]

# Weight-class split. Closed-API models have no official param counts;
# classification here is by well-established generation/tier (GPT-4 >>
# GPT-3.5, Claude-v2 > claude-instant), not a claimed exact size.
LARGE = {
    'claude-v2', 'claude-v1', 'gpt-4-1106-preview',
}
SMALL = {
    'claude-instant-v1', 'gpt-3.5-turbo-1106',
    'mistralai/mistral-7b-chat', 'mistralai/mixtral-8x7b-chat',
    'zero-one-ai/Yi-34B-Chat', 'meta/code-llama-instruct-34b-chat',
    'meta/llama-2-70b-chat', 'WizardLM/WizardLM-13B-V1.2',
}
assert LARGE | SMALL == set(MODELS)
assert not (LARGE & SMALL)

SET_A_FRACTION = 0.8
SPLIT_SEED = 42
N_PERM = 20000
OUT_DIR = Path("local_descriptors/routerbench-analysis")


def load_data():
    print("Loading RouterBench...", flush=True)
    f = hf_hub_download(repo_id="withmartian/routerbench", repo_type="dataset", filename="routerbench_0shot.pkl")
    df = pd.read_pickle(f)
    df = df.dropna(subset=[m for m in MODELS] + ["eval_name", "prompt"])
    rng = np.random.RandomState(SPLIT_SEED)
    idx = rng.permutation(len(df))
    n_a = int(len(df) * SET_A_FRACTION)
    set_a = df.iloc[idx[:n_a]].reset_index(drop=True)
    print(f"Total {len(df)} rows -> Set A {len(set_a)}", flush=True)
    return set_a


def build_centered_matrix(set_a):
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

    pool_matrix = np.stack([raw[n] for n in NAMES])  # (11, n_eval)
    pool_mean = pool_matrix.mean(axis=0)
    centered = pool_matrix - pool_mean  # (11, n_eval)
    return centered, eval_names


def main():
    set_a = load_data()
    C, eval_names = build_centered_matrix(set_a)  # (11, n_eval)
    print(f"  {len(eval_names)} distinct eval tasks, centered matrix shape={C.shape}\n", flush=True)

    large_idx = [i for i, m in enumerate(MODELS) if m in LARGE]
    small_idx = [i for i, m in enumerate(MODELS) if m in SMALL]
    print(f"Large tier ({len(large_idx)}): {[MODELS[i] for i in large_idx]}")
    print(f"Small tier ({len(small_idx)}): {[MODELS[i] for i in small_idx]}\n", flush=True)

    # --- 1. Group-gap vector + permutation test ---
    group_axis = C[large_idx].mean(axis=0) - C[small_idx].mean(axis=0)
    real_norm = np.linalg.norm(group_axis)

    rng = np.random.RandomState(0)
    perm_norms = np.zeros(N_PERM)
    n_large = len(large_idx)
    for p in range(N_PERM):
        perm = rng.permutation(11)
        pl, ps = perm[:n_large], perm[n_large:]
        perm_axis = C[pl].mean(axis=0) - C[ps].mean(axis=0)
        perm_norms[p] = np.linalg.norm(perm_axis)
    percentile = (perm_norms < real_norm).mean()
    print(f"Group-gap vector norm (real large/small split): {real_norm:.4f}")
    print(f"Random 7v4-split gap norm: mean={perm_norms.mean():.4f} std={perm_norms.std():.4f}")
    print(f"Real split percentile among {N_PERM} random splits: {percentile*100:.1f}%\n", flush=True)

    # --- 2. Within-tier residuals -> domain axes (PCA) ---
    residuals = C.copy()
    residuals[large_idx] -= C[large_idx].mean(axis=0)
    residuals[small_idx] -= C[small_idx].mean(axis=0)

    U, S, Vt = np.linalg.svd(residuals, full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()
    print("Within-tier residual PCA (domain axes, tier-mean removed):")
    for k in range(min(5, len(explained))):
        cos = abs(np.dot(group_axis, Vt[k]) / (real_norm * np.linalg.norm(Vt[k]) + 1e-12))
        print(f"  domain-PC{k+1}: explained_var={explained[k]:.4f}  cos_sim(group_axis, this PC)={cos:.4f}")

    # fraction of group_axis's squared norm captured by top-k domain PCs
    print()
    for k in [1, 2, 3, 5, len(explained)]:
        proj = Vt[:k] @ group_axis  # (k,)
        captured = (proj ** 2).sum() / (real_norm ** 2 + 1e-12)
        print(f"  top-{k} domain axes capture {captured*100:.1f}% of group_axis's squared norm")

    # --- 3. Full-pool PCA (group-agnostic), PC1 vs tier label ---
    print("\nFull-pool PCA (all 11 models, ignoring tier label):")
    Uf, Sf, Vtf = np.linalg.svd(C, full_matrices=False)
    scores = C @ Vtf.T  # (11, rank) -- each model's coordinate on each overall PC
    explained_f = (Sf ** 2) / (Sf ** 2).sum()
    print(f"  PC1 explains {explained_f[0]*100:.1f}% of variance, PC2 explains {explained_f[1]*100:.1f}%")
    print(f"  {'model':38s} {'tier':6s} {'PC1':>8s} {'PC2':>8s}")
    for i, m in enumerate(MODELS):
        tier = "large" if m in LARGE else "small"
        print(f"  {m:38s} {tier:6s} {scores[i,0]:>8.4f} {scores[i,1]:>8.4f}")

    pc1_large = scores[large_idx, 0]
    pc1_small = scores[small_idx, 0]
    from scipy import stats
    t, p = stats.ttest_ind(pc1_large, pc1_small)
    print(f"\n  PC1 large vs small: mean_large={pc1_large.mean():.4f} mean_small={pc1_small.mean():.4f} "
          f"t-test p={p:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "large_models": sorted(LARGE), "small_models": sorted(SMALL),
        "group_axis_norm": float(real_norm),
        "perm_test_percentile": float(percentile),
        "perm_norms_mean": float(perm_norms.mean()), "perm_norms_std": float(perm_norms.std()),
        "domain_pc_explained_var": explained.tolist(),
        "domain_pc_cos_sim_with_group_axis": [
            float(abs(np.dot(group_axis, Vt[k]) / (real_norm * np.linalg.norm(Vt[k]) + 1e-12)))
            for k in range(len(explained))
        ],
        "full_pool_pc1_explained": float(explained_f[0]),
        "full_pool_pc2_explained": float(explained_f[1]),
        "full_pool_pc1_scores": {MODELS[i]: float(scores[i, 0]) for i in range(11)},
        "full_pool_pc1_ttest_p": float(p),
    }
    out_path = OUT_DIR / "routerbench_flagship_orthogonality_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
