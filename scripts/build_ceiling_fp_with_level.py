"""Ceiling FP variant that adds back ONE extra dimension for each model's
overall (pool-mean-centered) accuracy -- the "shared difficulty / absolute
level" component that the original Ceiling FP's per-category mean-centering
strips out entirely (PROGRESS.md 16.10). Everything else (86 per-eval_name
mean-centered dims) is unchanged; this just appends dim 87 = model's own
overall Set A accuracy minus the pool's average overall accuracy, before the
final L2-normalize.

Saved to a NEW directory (not overwriting the original Ceiling FP), per
standing instruction to never clobber prior results.
"""
import numpy as np
from pathlib import Path

import routerbench_knn_test as rb

NAMES = rb.NAMES
MODELS = rb.MODELS
OUT_DIR = Path("local_descriptors/routerbench-ceiling-with-level")


def main():
    set_a, set_b = rb.load_data()
    eval_names = sorted(set_a["eval_name"].unique())
    name_to_idx = {e: i for i, e in enumerate(eval_names)}

    raw = {}
    overall_acc = {}
    for name, model_col in zip(NAMES, MODELS):
        vec = np.zeros(len(eval_names))
        counts = np.zeros(len(eval_names))
        for ev, score in zip(set_a["eval_name"], set_a[model_col]):
            vec[name_to_idx[ev]] += float(score)
            counts[name_to_idx[ev]] += 1
        raw[name] = vec / np.maximum(counts, 1)
        overall_acc[name] = float((set_a[model_col] >= 1.0).mean())

    pool_matrix = np.stack([raw[n] for n in NAMES])
    pool_mean = pool_matrix.mean(axis=0)
    pool_mean_overall = np.mean(list(overall_acc.values()))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"pool mean overall accuracy: {pool_mean_overall:.4f}")
    for name in NAMES:
        centered_86 = raw[name] - pool_mean
        level_dim = overall_acc[name] - pool_mean_overall
        full = np.concatenate([centered_86, [level_dim]])
        vec = (full / (np.linalg.norm(full) + 1e-12)).astype(np.float32)
        np.save(OUT_DIR / f"{name}.npy", vec)
        print(f"  {name:38s} overall_acc={overall_acc[name]:.4f}  level_dim={level_dim:+.4f}  shape={vec.shape}")

    E = np.stack([np.load(OUT_DIR / f"{n}.npy") for n in NAMES])
    sim = E @ E.T
    off = sim[~np.eye(len(NAMES), dtype=bool)]
    print(f"\npairwise cosine sim: mean={off.mean():.4f} std={off.std():.4f} min={off.min():.4f} max={off.max():.4f}")
    print(f"Saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
