"""Pseudo Ceiling FP for RouterBench -- fills the "not tested" gap in the
3-benchmark comparison table. Unlike the existing "Ceiling FP" (eval_name
CATEGORY-averaged, routerbench_knn_test.py), this selects individual
high-variance PROMPTS stratified per eval_name category (same methodology
as LLMRouterBench's probe selection, scripts/llmrouterbench/build_data_split_lite20.py),
mean-centers, L2-normalizes, and runs the same kNN unseen-recovery test.
"""
import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from scipy import stats
from scipy.stats import spearmanr

MODELS = [
    'claude-instant-v1', 'claude-v1', 'claude-v2', 'gpt-3.5-turbo-1106',
    'gpt-4-1106-preview', 'meta/code-llama-instruct-34b-chat',
    'meta/llama-2-70b-chat', 'mistralai/mistral-7b-chat',
    'mistralai/mixtral-8x7b-chat', 'zero-one-ai/Yi-34B-Chat',
    'WizardLM/WizardLM-13B-V1.2',
]
NAMES = [m.replace("/", "__") for m in MODELS]

OUT_DIR = Path("local_descriptors/routerbench-analysis")
PSEUDO_CEILING_DIR = Path("local_descriptors/routerbench-pseudo-ceiling")
SET_A_FRACTION = 0.8
SPLIT_SEED = 42
N_PROBES_PER_CATEGORY = 6  # 86 categories x 6 ~= 516, comparable to LLMRouterBench's 528


def load_data():
    print("Loading RouterBench...", flush=True)
    f = hf_hub_download(repo_id="withmartian/routerbench", repo_type="dataset", filename="routerbench_0shot.pkl")
    import pandas as pd
    df = pd.read_pickle(f)
    df = df.dropna(subset=[m for m in MODELS] + ["eval_name", "prompt"])
    rng = np.random.RandomState(SPLIT_SEED)
    idx = rng.permutation(len(df))
    n_a = int(len(df) * SET_A_FRACTION)
    set_a = df.iloc[idx[:n_a]].reset_index(drop=True)
    set_b = df.iloc[idx[n_a:]].reset_index(drop=True)
    print(f"Total {len(df)} rows -> Set A {len(set_a)}, Set B {len(set_b)}", flush=True)
    return set_a, set_b


def build_pseudo_ceiling_fp(set_a):
    print("\nBuilding Pseudo Ceiling FP (per-eval_name high-variance probe selection)...", flush=True)
    scores = set_a[MODELS].to_numpy(dtype=float)  # (n, 11)
    variance = scores.var(axis=1)

    selected_idx = []
    eval_names = sorted(set_a["eval_name"].unique())
    for ev in eval_names:
        cat_idx = np.where(set_a["eval_name"].to_numpy() == ev)[0]
        cat_var = variance[cat_idx]
        n_take = min(N_PROBES_PER_CATEGORY, len(cat_idx))
        top_local = np.argsort(-cat_var)[:n_take]
        selected_idx.extend(cat_idx[top_local].tolist())

    print(f"  {len(eval_names)} eval_name categories, {len(selected_idx)} probes selected total", flush=True)

    probe_scores = scores[selected_idx]  # (n_probes, 11)
    pool_mean = probe_scores.mean(axis=1, keepdims=True)
    centered = probe_scores - pool_mean

    PSEUDO_CEILING_DIR.mkdir(parents=True, exist_ok=True)
    for j, name in enumerate(NAMES):
        vec = centered[:, j]
        vec = (vec / (np.linalg.norm(vec) + 1e-12)).astype(np.float32)
        np.save(PSEUDO_CEILING_DIR / f"{name}.npy", vec)

    with open(PSEUDO_CEILING_DIR / "probe_info.json", "w") as f:
        json.dump({"selected_idx": selected_idx, "n_probes": len(selected_idx),
                    "n_probes_per_category": N_PROBES_PER_CATEGORY}, f, indent=2)

    E = np.stack([np.load(PSEUDO_CEILING_DIR / f"{n}.npy") for n in NAMES])
    sim = E @ E.T
    off = sim[~np.eye(len(NAMES), dtype=bool)]
    print(f"  pairwise cosine sim: mean={off.mean():.4f} std={off.std():.4f} min={off.min():.4f} max={off.max():.4f}",
          flush=True)


def knn_test(set_b, desc_dir, fp_name):
    print(f"\n{'='*60}\nkNN unseen-recovery test: {fp_name}\n{'='*60}", flush=True)
    E = np.stack([np.load(desc_dir / f"{n}.npy") for n in NAMES])
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    sim_full = E @ E.T

    true_scores = np.stack([set_b[m].to_numpy(dtype=float) for m in MODELS], axis=1)

    fp_rhos, uniform_rhos = [], []
    for i, held_out in enumerate(NAMES):
        others_idx = [j for j in range(len(NAMES)) if j != i]
        sims = sim_full[i, others_idx]
        w = np.clip(sims, 0, None)
        if w.sum() < 1e-9:
            w = np.ones_like(w)
        w = w / w.sum()

        other_scores = true_scores[:, others_idx]
        true_m = true_scores[:, i]
        fp_proxy = other_scores @ w
        uniform_proxy = other_scores.mean(axis=1)

        fp_rho, _ = spearmanr(fp_proxy, true_m)
        uni_rho, _ = spearmanr(uniform_proxy, true_m)
        fp_rhos.append(fp_rho)
        uniform_rhos.append(uni_rho)
        print(f"  held out {held_out:35s} FP-proxy rho={fp_rho:.4f}  uniform-proxy rho={uni_rho:.4f}", flush=True)

    fp_rhos, uniform_rhos = np.array(fp_rhos), np.array(uniform_rhos)
    delta = fp_rhos - uniform_rhos
    t, p = stats.ttest_rel(fp_rhos, uniform_rhos)
    print(f"\nmean FP rho={fp_rhos.mean():.4f}  mean uniform rho={uniform_rhos.mean():.4f}  "
          f"mean delta={delta.mean():+.4f}  paired t-test p={p:.4f}  ({(delta>0).sum()}/11 folds improved)", flush=True)
    return {
        "fp_rho": fp_rhos.tolist(), "uniform_rho": uniform_rhos.tolist(),
        "mean_fp_rho": float(fp_rhos.mean()), "mean_uniform_rho": float(uniform_rhos.mean()),
        "mean_delta": float(delta.mean()), "paired_t_p": float(p),
        "n_folds_improved": int((delta > 0).sum()),
    }


def main():
    set_a, set_b = load_data()
    build_pseudo_ceiling_fp(set_a)
    result = knn_test(set_b, PSEUDO_CEILING_DIR, "Pseudo Ceiling")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "routerbench_pseudo_ceiling_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
