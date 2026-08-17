"""Generate newllm_split_seed{N}.json + the corresponding unseen-only PCA5
FP directory for seeds 5-10, using the EXACT same build_split() logic
(random.Random(seed).shuffle) and the same registry-covered model universe
as the original embedllm_newllm_multiseed.py used for seeds 1-4, so seeds
5-10 are methodologically consistent with the existing seeds.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "src")
from router.registry import REGISTRY

PCA5_DIR = Path("local_descriptors/embedllm-ceiling-pca5")
ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
NEW_SEEDS = [5, 6, 7, 8, 9, 10]


def build_split(models, seed):
    import random
    rng = random.Random(seed)
    perm = models[:]
    rng.shuffle(perm)
    n_seen = round(len(perm) * 2 / 3)
    return sorted(perm[:n_seen]), sorted(perm[n_seen:])


def main():
    all_fp_models = sorted(p.stem for p in PCA5_DIR.glob("*.npy"))
    reg_models = [m for m in all_fp_models if m in REGISTRY]
    print(f"{len(reg_models)} registry-covered models (of {len(all_fp_models)} total FP files)")

    for seed in NEW_SEEDS:
        split_path = ANALYSIS_DIR / f"newllm_split_seed{seed}.json"
        if split_path.exists():
            print(f"seed {seed}: split already exists, skipping")
            continue
        seen, unseen = build_split(reg_models, seed)
        json.dump({"seen": seen, "unseen": unseen, "seed": seed}, open(split_path, "w"), indent=2)

        unseen_dir = Path(f"local_descriptors/embedllm-ceiling-pca5-unseen-only-seed{seed}")
        unseen_dir.mkdir(parents=True, exist_ok=True)
        for m in unseen:
            shutil.copy(PCA5_DIR / f"{m}.npy", unseen_dir / f"{m}.npy")
        print(f"seed {seed}: seen={len(seen)} unseen={len(unseen)} -> {split_path}")

    print("Done.")


if __name__ == "__main__":
    main()
