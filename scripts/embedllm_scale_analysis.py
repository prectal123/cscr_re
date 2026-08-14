"""EmbedLLM (112 open-weight models, 80 categories, GT labels only -- no
response text, so no Perplexity/V1.2 FP here, Ceiling-FP-style only) reruns
of two things done on smaller pools:

  1. PCA dimensionality ablation (see pca_dim_ablation.py, lite20/22-cat) --
     now with 80 categories and, critically, 112 models instead of 20/11, so
     the "how many real dims does the FP need" question gets answered with
     real statistical power for the first time.
  2. Scale (large vs small open-weight) group-gap + orthogonality test (see
     routerbench_flagship_orthogonality.py) -- same method, n=112 instead of
     n=11, so the RouterBench result (small-n, unstable) gets a much
     better-powered re-check. Note: EmbedLLM has NO proprietary flagship
     models (no GPT-4/Claude) -- this tests "does raw open-weight SCALE
     produce an orthogonal-to-domain gap", not "flagship brand vs not".

Also runs the kNN unseen-model recovery test (Set A -> FP, Set B = native
test.csv held-out prompts) at this much larger pool size, directly
addressing the recurring "n=11 is too small, don't over-read fold noise"
concern from RouterBench.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy import stats
from scipy.stats import spearmanr

OUT_DIR = Path("local_descriptors/embedllm-analysis")
CEILING_DIR = Path("local_descriptors/embedllm-ceiling")
N_PERM = 20000

# Manual overrides for names the size regex can't parse (or gets wrong).
SIZE_OVERRIDES_B = {
    "microsoft__phi-1_5": 1.3,
    "microsoft__phi-2": 2.7,
    "google__gemma-2b-it": 2.0,
    "google__gemma-7b-it": 7.0,
    "bigscience__bloom-7b1": 7.1,
    "zhengr__MixTAO-7Bx2-MoE-v8.1": 12.9,   # 2x7B MoE, ~12.9B active/total per HF card
    "cloudyu__Mixtral_11Bx2_MoE_19B": 19.0,
    "Plaban81__Moe-4x7b-math-reason-code": 24.0,  # 4x7B MoE
}


def load_data():
    print("Loading EmbedLLM train/test...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    f_test = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="test.csv")
    set_a = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "category", "label"])
    set_b = pd.read_csv(f_test, usecols=["prompt_id", "model_name", "category", "label"])
    print(f"  Set A: {len(set_a)} rows, {set_a['prompt_id'].nunique()} prompts", flush=True)
    print(f"  Set B: {len(set_b)} rows, {set_b['prompt_id'].nunique()} prompts", flush=True)
    return set_a, set_b


def build_centered_matrix(set_a, models, categories):
    """(n_models, n_categories) pool-mean-centered mean-accuracy matrix."""
    pivot = set_a.pivot_table(index="model_name", columns="category", values="label", aggfunc="mean")
    pivot = pivot.reindex(index=models, columns=categories)
    raw = pivot.to_numpy()  # (n_models, n_categories), may contain NaN for missing cells
    col_mean = np.nanmean(raw, axis=0, keepdims=True)
    raw = np.where(np.isnan(raw), col_mean, raw)  # impute missing model/category cells with pool mean
    pool_mean = raw.mean(axis=0, keepdims=True)
    centered = raw - pool_mean
    return centered


def build_setB_matrix(set_b, models, categories):
    """(n_prompts_kept, n_models) matrix for prompts where ALL models have a label."""
    pivot = set_b.pivot_table(index="prompt_id", columns="model_name", values="label", aggfunc="mean")
    pivot = pivot.reindex(columns=models)
    pivot = pivot.dropna(axis=0, how="any")
    return pivot.to_numpy()  # (n_prompts, n_models)


def extract_size_b(model_name):
    if model_name in SIZE_OVERRIDES_B:
        return SIZE_OVERRIDES_B[model_name]
    m = re.search(r'(\d+(?:\.\d+)?)[Bb](?:[^a-zA-Z]|$)', model_name)
    if m:
        return float(m.group(1))
    return None


def knn_test(true_scores, E, models):
    sim_full = E @ E.T
    fp_rhos, uniform_rhos = [], []
    for i in range(len(models)):
        others_idx = [j for j in range(len(models)) if j != i]
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
    set_a, set_b = load_data()
    models = sorted(set_a["model_name"].unique())
    categories = sorted(set_a["category"].unique())
    print(f"\n{len(models)} models, {len(categories)} categories\n", flush=True)

    C = build_centered_matrix(set_a, models, categories)  # (112, 80)

    # ---------- 1. PCA dimensionality ablation ----------
    print("=" * 70)
    print("PART 1: PCA dimensionality ablation (n=112 models, 80 categories)")
    print("=" * 70)
    X = C  # (112, 80), rows=models already centered per-category
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()
    cum = np.cumsum(explained)
    print(f"{'k':>4s} {'explained':>10s} {'cumulative':>11s}")
    for k in range(min(15, len(explained))):
        print(f"{k+1:>4d} {explained[k]:>10.4f} {cum[k]:>11.4f}")

    setB_true = build_setB_matrix(set_b, models, categories)  # (n_prompts, 112) -- may be small if coverage is sparse
    print(f"\nSet B usable rows (all 112 models present): {setB_true.shape[0]}")

    K_SWEEP = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, min(80, len(explained))]
    K_SWEEP = sorted(set(k for k in K_SWEEP if k <= len(explained)))
    print(f"\n{'k':>4s} {'cum_var':>8s} {'mean FP rho':>12s} {'mean uniform':>13s} {'delta':>9s} {'p-value':>9s} {'improved':>10s}")
    pca_results = []
    if setB_true.shape[0] >= 20:  # only run kNN test if enough fully-covered Set B rows
        for k in K_SWEEP:
            reduced = X @ Vt[:k].T
            E = reduced / (np.linalg.norm(reduced, axis=1, keepdims=True) + 1e-12)
            fp_rhos, uniform_rhos = knn_test(setB_true, E.astype(np.float32), models)
            delta = fp_rhos - uniform_rhos
            t, p = stats.ttest_rel(fp_rhos, uniform_rhos)
            n_imp = int((delta > 0).sum())
            print(f"{k:>4d} {cum[k-1]:>8.4f} {fp_rhos.mean():>12.4f} {uniform_rhos.mean():>13.4f} "
                  f"{delta.mean():>+9.4f} {p:>9.4f} {n_imp:>7d}/{len(models)}")
            pca_results.append({"k": k, "cum_var": float(cum[k-1]), "mean_fp_rho": float(fp_rhos.mean()),
                                 "mean_uniform_rho": float(uniform_rhos.mean()), "mean_delta": float(delta.mean()),
                                 "p_value": float(p), "n_improved": n_imp})
    else:
        print("  (skipped -- too few Set B rows with full 112-model coverage)")

    # ---------- 2. Scale group-gap + orthogonality ----------
    print("\n" + "=" * 70)
    print("PART 2: scale (large vs small open-weight) group-gap + orthogonality")
    print("=" * 70)
    sizes = {m: extract_size_b(m) for m in models}
    parsed = {m: s for m, s in sizes.items() if s is not None}
    unparsed = [m for m, s in sizes.items() if s is None]
    print(f"Parsed size for {len(parsed)}/{len(models)} models. Unparsed (excluded from group test): {unparsed}\n")

    sorted_by_size = sorted(parsed.items(), key=lambda kv: kv[1])
    median_size = np.median([s for _, s in sorted_by_size])
    large_models = [m for m, s in sorted_by_size if s >= median_size]
    small_models = [m for m, s in sorted_by_size if s < median_size]
    print(f"Median size split: {median_size:.1f}B -- large={len(large_models)} small={len(small_models)}")
    print(f"  large size range: {min(s for _,s in sorted_by_size if s>=median_size):.1f}B - "
          f"{max(s for _,s in sorted_by_size if s>=median_size):.1f}B")
    print(f"  small size range: {min(s for _,s in sorted_by_size if s<median_size):.1f}B - "
          f"{max(s for _,s in sorted_by_size if s<median_size):.1f}B\n")

    model_to_row = {m: i for i, m in enumerate(models)}
    large_idx = [model_to_row[m] for m in large_models]
    small_idx = [model_to_row[m] for m in small_models]

    group_axis = C[large_idx].mean(axis=0) - C[small_idx].mean(axis=0)
    real_norm = np.linalg.norm(group_axis)

    rng = np.random.RandomState(0)
    all_idx = large_idx + small_idx
    n_large = len(large_idx)
    perm_norms = np.zeros(N_PERM)
    for p in range(N_PERM):
        perm = rng.permutation(all_idx)
        pl, ps = perm[:n_large], perm[n_large:]
        perm_norms[p] = np.linalg.norm(C[pl].mean(axis=0) - C[ps].mean(axis=0))
    percentile = (perm_norms < real_norm).mean()
    print(f"Group-gap norm (real large/small split): {real_norm:.4f}")
    print(f"Random split gap norm: mean={perm_norms.mean():.4f} std={perm_norms.std():.4f}")
    print(f"Real split percentile: {percentile*100:.1f}%\n")

    residuals = C.copy()
    residuals[large_idx] -= C[large_idx].mean(axis=0)
    residuals[small_idx] -= C[small_idx].mean(axis=0)
    Ur, Sr, Vtr = np.linalg.svd(residuals, full_matrices=False)
    explained_r = (Sr ** 2) / (Sr ** 2).sum()
    print("Within-tier residual PCA (domain axes):")
    for k in range(min(5, len(explained_r))):
        cos = abs(np.dot(group_axis, Vtr[k]) / (real_norm * np.linalg.norm(Vtr[k]) + 1e-12))
        print(f"  domain-PC{k+1}: explained_var={explained_r[k]:.4f} cos_sim(group_axis)={cos:.4f}")
    print()
    for k in [1, 2, 3, 5, 10, len(explained_r)]:
        proj = Vtr[:k] @ group_axis
        captured = (proj ** 2).sum() / (real_norm ** 2 + 1e-12)
        print(f"  top-{k} domain axes capture {captured*100:.1f}% of group_axis squared norm")

    # Continuous check: correlation of log-size with full-pool PC1
    print("\nContinuous check: log(size) vs full-pool PC1 (all 112 models)")
    scores_full = C @ Vt.T
    log_sizes = np.array([np.log(parsed[m]) if m in parsed else np.nan for m in models])
    valid = ~np.isnan(log_sizes)
    rho_size_pc1, p_size_pc1 = spearmanr(log_sizes[valid], scores_full[valid, 0])
    print(f"  spearman(log_size, PC1) = {rho_size_pc1:.4f}  p={p_size_pc1:.4g}  (n={valid.sum()})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "n_models": len(models), "n_categories": len(categories),
        "setB_rows_full_coverage": int(setB_true.shape[0]),
        "pca_dim_ablation": pca_results,
        "pca_full_explained_variance": explained[:20].tolist(),
        "size_parsed_count": len(parsed), "size_unparsed": unparsed,
        "median_size_split_B": float(median_size),
        "large_models": large_models, "small_models": small_models,
        "group_axis_norm": float(real_norm),
        "perm_test_percentile": float(percentile),
        "perm_norms_mean": float(perm_norms.mean()), "perm_norms_std": float(perm_norms.std()),
        "domain_pc_cos_sim_with_group_axis": [
            float(abs(np.dot(group_axis, Vtr[k]) / (real_norm * np.linalg.norm(Vtr[k]) + 1e-12)))
            for k in range(len(explained_r))
        ],
        "spearman_logsize_vs_pc1": float(rho_size_pc1), "spearman_logsize_vs_pc1_p": float(p_size_pc1),
    }
    out_path = OUT_DIR / "embedllm_scale_analysis_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
