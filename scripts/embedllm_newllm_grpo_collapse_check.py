"""Diagnose whether the GRPO-regression query encoder (embedllm_newllm_grpo_train.py,
seed 0 checkpoint) exhibits the same expert-collapse failure mode found
throughout this project (PROGRESS.md sections 13-17, 20) despite its
zero-mean-per-query target being specifically designed to remove the
"always point at the generically strongest model" shortcut.

Two checks, both on the UNSEEN-only candidate pool (the actual eval setup):
  1. Routing distribution at near-zero cost pressure (lambda ~= 1e-4, i.e.
     pure argmax(cos_sim), no cost penalty) over all ~3000 EmbedLLM test
     prompts -- top3_share, number of distinct models ever chosen.
  2. Whether the collapse target(s) are actually the models with the
     highest TRUE mean accuracy on this test set (the "true oracle" check
     from section 17.20/20) -- Spearman rho between (selection count) and
     (true accuracy) across the 22 unseen models.
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
UNSEEN_DIR = Path("local_descriptors/embedllm-ceiling-pca5-unseen-only")
SPLIT_PATH = ANALYSIS_DIR / "newllm_split.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="local_checkpoints/embedllm-newllm-encoder-grpo-seed0")
    ap.add_argument("--out_suffix", default="seed0")
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

    E_unseen = np.stack([np.load(UNSEEN_DIR / f"{m}.npy") for m in unseen]).astype(np.float32)
    E_unseen = E_unseen / (np.linalg.norm(E_unseen, axis=1, keepdims=True) + 1e-12)
    sims = embeds @ E_unseen.T  # (N, 22)

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
    print("\n=== True accuracy vs selection count, per unseen model ===")
    for i in order:
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
