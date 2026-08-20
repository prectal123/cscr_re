"""Collapse diagnostic against the TRUE current headline checkpoint: uncompressed
(80-dim, no PCA) + uniform-allocation 1800-probe Ceiling FP, TAR loss
(min(30%,cap3) + top50pct catfilter), standard unseen-only-candidates protocol
(the exact recipe behind PROGRESS.md's headline Unseen AUDC=0.5232).

Supersedes embedllm_v15_minpctcap3_collapse_check.py's result (that one ran
against the deprecated V1.5/PCA-5-compressed checkpoint -- per PROGRESS.md
section 25, PCA-5 compression alone costs ~0.03 AUDC and is no longer used
anywhere in the project as of 2026-08-19).

Reuses the exact training code (build_raw_category_accuracy, build_items,
minpctcap_loss, precompute_cls, train_fast) from
embedllm_unseen_mixedpool_1800_multiseed.py (the script that produced the
0.5787/0.5232 headline numbers) rather than reimplementing it, so this is
trained identically -- only the eval-time candidate pool changes (unseen-only
here, vs mixed seen+unseen there). Also saves the trained checkpoint (the
headline pipeline's own scripts never call torch.save) so it doesn't need to
be retrained if this diagnostic needs to be re-run later.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from embedllm_unseen_mixedpool_1800_multiseed import (
    build_raw_category_accuracy, build_items, train_fast, precompute_cls,
    ANALYSIS_DIR, EMBED_MODEL, PCT_CATFILTER, DEVICE,
)
from router.embedllm import load_embedllm

SEED = 0
FP_DIR = Path("local_descriptors/embedllm-ceiling-scalesweep-uniform-nocompress-minpctcap3-1800")
CKPT_PATH = Path("local_checkpoints/embedllm-newllm-encoder-uncompressed-headline-seed0/proj.pt")


def main():
    print("Loading EmbedLLM train.csv...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "category", "label", "prompt"])
    categories = sorted(df["category"].unique().tolist())
    category_to_idx = {c: i for i, c in enumerate(categories)}

    split = json.load(open(ANALYSIS_DIR / "newllm_split.json", encoding="utf-8"))
    seen_models, unseen_models = split["seen"], split["unseen"]
    print(f"seen={len(seen_models)} unseen={len(unseen_models)}", flush=True)

    print(f"Loading frozen MiniLM base (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    hidden_size = base_model.config.hidden_size

    raw_cat_acc = build_raw_category_accuracy(df, seen_models, categories)
    texts, targets, masks = build_items(df, seen_models, raw_cat_acc, category_to_idx, PCT_CATFILTER)
    print(f"seen-only training rows: {len(texts)}", flush=True)
    t0 = time.time()
    cls_all = precompute_cls(texts, tokenizer, base_model)
    print(f"  cached in {time.time()-t0:.1f}s -> {cls_all.shape}", flush=True)

    E_seen = np.stack([np.load(FP_DIR / f"{m}.npy") for m in seen_models]).astype(np.float32)
    E_seen_t = torch.from_numpy(E_seen).float().to(DEVICE)
    E_seen_t = E_seen_t / (E_seen_t.norm(dim=1, keepdim=True) + 1e-9)
    print(f"FP dim = {E_seen_t.size(1)}", flush=True)

    proj = train_fast(SEED, cls_all, targets, masks, E_seen_t, hidden_size, "uncompressed-headline-seed0")

    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(proj.state_dict(), CKPT_PATH)
    print(f"Saved checkpoint -> {CKPT_PATH}", flush=True)

    # --- Standard UNSEEN-ONLY-candidates eval (matches headline 0.5232 protocol) ---
    dataset = load_embedllm("test", candidates=unseen_models)
    eval_texts = [ex["prompt"] for ex in dataset]
    label_maps = [ex["label_map"] for ex in dataset]
    print(f"{len(eval_texts)} test prompts (unseen-only candidates)", flush=True)

    cls_setB = precompute_cls(eval_texts, tokenizer, base_model)
    with torch.no_grad():
        embeds = F.normalize(proj(torch.from_numpy(cls_setB).float().to(DEVICE)), dim=-1).cpu().numpy()

    E_unseen = np.stack([np.load(FP_DIR / f"{m}.npy") for m in unseen_models]).astype(np.float32)
    E_unseen = E_unseen / (np.linalg.norm(E_unseen, axis=1, keepdims=True) + 1e-12)
    sims = embeds @ E_unseen.T

    chosen = np.argmax(sims, axis=1)
    counts = np.bincount(chosen, minlength=len(unseen_models))
    order = np.argsort(-counts)
    print("\n=== Routing distribution (argmax cos_sim, no cost pressure) ===")
    for i in order:
        print(f"  {unseen_models[i]:45s} {counts[i]:5d}  ({counts[i]/len(eval_texts):.1%})")

    top3_share = counts[order[:3]].sum() / len(eval_texts)
    n_used = int((counts > 0).sum())
    print(f"\ntop3_share = {top3_share:.4f}  (chance = {3/len(unseen_models):.4f})")
    print(f"models ever chosen: {n_used}/{len(unseen_models)}")

    true_acc = np.array([np.mean([lm.get(m, 0) for lm in label_maps]) for m in unseen_models])
    rho, p = spearmanr(counts, true_acc)
    print(f"\nSpearman rho(selection_count, true_accuracy) = {rho:.4f}  p={p:.4f}")

    top_selected_idx = order[0]
    top_acc_idx = int(np.argmax(true_acc))
    print(f"\nMost-selected model: {unseen_models[top_selected_idx]} (true_acc={true_acc[top_selected_idx]:.4f}, "
          f"selected {counts[top_selected_idx]/len(eval_texts):.1%})")
    print(f"Highest-true-accuracy model: {unseen_models[top_acc_idx]} (true_acc={true_acc[top_acc_idx]:.4f}, "
          f"selected {counts[top_acc_idx]/len(eval_texts):.1%})")

    result = {
        "ckpt_path": str(CKPT_PATH),
        "fp_dir": str(FP_DIR),
        "top3_share": float(top3_share),
        "chance_top3_share": 3 / len(unseen_models),
        "n_models_used": n_used,
        "n_models_total": len(unseen_models),
        "spearman_rho_selection_vs_true_acc": float(rho),
        "spearman_p": float(p),
        "most_selected_model": unseen_models[top_selected_idx],
        "most_selected_share": float(counts[top_selected_idx] / len(eval_texts)),
        "most_selected_true_acc": float(true_acc[top_selected_idx]),
        "highest_true_acc_model": unseen_models[top_acc_idx],
        "highest_true_acc": float(true_acc[top_acc_idx]),
        "highest_true_acc_model_share": float(counts[top_acc_idx] / len(eval_texts)),
        "per_model": [
            {"model": unseen_models[i], "true_acc": float(true_acc[i]), "selected_count": int(counts[i]),
             "selected_share": float(counts[i] / len(eval_texts))}
            for i in order
        ],
    }
    out_path = ANALYSIS_DIR / "newllm_grpo_collapse_check_uncompressed_headline_seed0.json"
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
