"""Collapse diagnostic for the CURRENT best checkpoint (TAR/COMPAR: min(pct=0.3,cap=3)
+ top50pct catfilter loss, trained on the true V1.5 FP -- 1800-probe, PCA-weighted
floor=15, PCA-5 compressed) -- adapted from embedllm_newllm_grpo_collapse_check.py
(which diagnosed the ORIGINAL vanilla-GRPO seed0 checkpoint: top3_share=0.853,
rho=0.42, see PROGRESS.md section 21+).

This has been on the "next step" list three times over (PROGRESS.md 24.17-3,
25.6-5, 26.8-3) without ever actually being run against a post-outlier-drag-fix
checkpoint. Same two checks, same UNSEEN-only candidate pool protocol, just
pointed at the current method's checkpoint/FP instead of the original one.

Note: no checkpoint exists yet for the very latest "uncompressed + uniform
allocation" pipeline (PROGRESS.md section 25-26, headline All-seen=0.5787
Unseen=0.5232) since those sweep scripts never call enc.save(). This runs
against the V1.5 (1800-probe, PCA-5) checkpoint instead, which IS saved
(embedllm-newllm-encoder-minpctcap3-v15-1800-seed0) and was used for the
probe-budget fairness claim (section 24.15, 10-seed unseen mean=0.5128).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from router.query_encoder import QueryEncoder
from router.embedllm import load_embedllm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
FP_DIR = Path("local_descriptors/embedllm-ceiling-pcaweighted-pca5")  # true V1.5, same as used for training/eval
SPLIT_PATH = ANALYSIS_DIR / "newllm_split.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="local_checkpoints/embedllm-newllm-encoder-minpctcap3-v15-1800-seed0")
    ap.add_argument("--out_suffix", default="v15_minpctcap3_seed0")
    args = ap.parse_args()
    CKPT_DIR = Path(args.ckpt_dir)

    split = json.load(open(SPLIT_PATH, encoding="utf-8"))
    unseen = split["unseen"]
    print(f"unseen models: {len(unseen)}")

    enc = QueryEncoder.load(str(CKPT_DIR), proj_dim=5)
    enc.to(DEVICE)
    enc.model.eval()

    dataset = load_embedllm("test", candidates=unseen)
    texts = [ex["prompt"] for ex in dataset]
    label_maps = [ex["label_map"] for ex in dataset]
    print(f"{len(texts)} test prompts")

    embeds = np.zeros((len(texts), 5), dtype=np.float32)
    for start in range(0, len(texts), 64):
        batch = texts[start:start + 64]
        embeds[start:start + len(batch)] = enc.encode(batch)

    E_unseen = np.stack([np.load(FP_DIR / f"{m}.npy") for m in unseen]).astype(np.float32)
    E_unseen = E_unseen / (np.linalg.norm(E_unseen, axis=1, keepdims=True) + 1e-12)
    sims = embeds @ E_unseen.T  # (N, n_unseen)

    # --- Check 1: routing distribution, pure argmax(sim), no cost pressure ---
    chosen = np.argmax(sims, axis=1)
    counts = np.bincount(chosen, minlength=len(unseen))
    order = np.argsort(-counts)
    print("\n=== Routing distribution (argmax cos_sim, no cost pressure) ===")
    for i in order:
        print(f"  {unseen[i]:45s} {counts[i]:5d}  ({counts[i]/len(texts):.1%})")

    top3_share = counts[order[:3]].sum() / len(texts)
    n_used = int((counts > 0).sum())
    print(f"\ntop3_share = {top3_share:.4f}  (chance = {3/len(unseen):.4f})")
    print(f"models ever chosen: {n_used}/{len(unseen)}")

    # --- Check 2: is the collapse target actually the best true-accuracy model? ---
    true_acc = np.array([np.mean([lm.get(m, 0) for lm in label_maps]) for m in unseen])
    print("\n=== True accuracy vs selection count, per unseen model (top 10 by selection) ===")
    for i in order[:10]:
        print(f"  {unseen[i]:45s} true_acc={true_acc[i]:.4f}  selected={counts[i]:5d} ({counts[i]/len(texts):.1%})")

    rho, p = spearmanr(counts, true_acc)
    print(f"\nSpearman rho(selection_count, true_accuracy) = {rho:.4f}  p={p:.4f}")

    top_selected_idx = order[0]
    top_acc_idx = int(np.argmax(true_acc))
    print(f"\nMost-selected model: {unseen[top_selected_idx]} (true_acc={true_acc[top_selected_idx]:.4f}, "
          f"selected {counts[top_selected_idx]/len(texts):.1%})")
    print(f"Highest-true-accuracy model: {unseen[top_acc_idx]} (true_acc={true_acc[top_acc_idx]:.4f}, "
          f"selected {counts[top_acc_idx]/len(texts):.1%})")

    result = {
        "ckpt_dir": str(CKPT_DIR),
        "fp_dir": str(FP_DIR),
        "top3_share": float(top3_share),
        "chance_top3_share": 3 / len(unseen),
        "n_models_used": n_used,
        "n_models_total": len(unseen),
        "spearman_rho_selection_vs_true_acc": float(rho),
        "spearman_p": float(p),
        "most_selected_model": unseen[top_selected_idx],
        "most_selected_share": float(counts[top_selected_idx] / len(texts)),
        "most_selected_true_acc": float(true_acc[top_selected_idx]),
        "highest_true_acc_model": unseen[top_acc_idx],
        "highest_true_acc": float(true_acc[top_acc_idx]),
        "highest_true_acc_model_share": float(counts[top_acc_idx] / len(texts)),
        "per_model": [
            {"model": unseen[i], "true_acc": float(true_acc[i]), "selected_count": int(counts[i]),
             "selected_share": float(counts[i] / len(texts))}
            for i in order
        ],
    }
    out_path = ANALYSIS_DIR / f"newllm_grpo_collapse_check_{args.out_suffix}.json"
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
