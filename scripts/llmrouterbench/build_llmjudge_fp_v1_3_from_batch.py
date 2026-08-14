"""Collect the V1.3 LLM-judge batch (see submit_llmjudge_v1_3_batch.py),
aggregate the 66 per-probe judge scores into a (22 categories, 20 models)
category-mean matrix, mean-center per category and L2-normalize per model
(same recipe as build_ceiling_fp_lite20.py / build_ceiling_fp_lite20_categoryrate.py),
and save local_descriptors/llmrouterbench_lite20/v1_3/{model}.npy.

Usage: python build_llmjudge_fp_v1_3_from_batch.py
Safe to re-run while the batch is still processing -- it will just report
the current status and exit without writing anything.
"""
import json
import os
import pickle
from pathlib import Path

import anthropic
import numpy as np

import common_lite20 as common

DATA_DIR = Path("local_descriptors/llmrouterbench_lite20")
OUT_DIR = DATA_DIR / "v1_3"
API_KEY_FILE = Path(r"C:\Users\user\anthropic_key.txt")

# Execution-graded code categories: don't trust a text-reading LLM judge over
# actual test-execution results. The batch was submitted before this exclusion
# was wired in (see chat), so judge results for these DO exist in the batch --
# we deliberately discard them and substitute the original 0/1 execution score
# instead, rescaled to the judge's 1-10 range so per-category magnitudes stay
# comparable after mean-centering + L2-normalize.
CODE_CATEGORIES = {"humaneval", "mbpp", "livecodebench"}


def _load_api_key_env():
    """Read the API key from a local file at runtime (never printed/logged) if
    ANTHROPIC_API_KEY isn't already set in the environment."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    if API_KEY_FILE.exists():
        os.environ["ANTHROPIC_API_KEY"] = API_KEY_FILE.read_text(encoding="utf-8").strip()


def load_probe_order():
    """Returns list of (dataset, probe_rank) in the same order used at submit time."""
    with open(DATA_DIR / "v1_3_probe_selection.json", encoding="utf-8") as f:
        selected = json.load(f)["selected"]
    order = []
    for cat in selected:
        ds = cat["dataset"]
        for p_rank in range(len(cat["probes"])):
            order.append((ds, p_rank))
    return order


def main():
    with open(DATA_DIR / "v1_3_batch_id.json", encoding="utf-8") as f:
        batch_info = json.load(f)
    batch_id = batch_info["batch_id"]

    _load_api_key_env()
    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    print(f"Batch {batch_id}: status={batch.processing_status}  counts={batch.request_counts}")
    if batch.processing_status != "ended":
        print("Not finished yet -- re-run this script later.")
        return

    probe_order = load_probe_order()  # [(dataset, probe_rank), ...] len 66
    probe_index = {key: i for i, key in enumerate(probe_order)}
    model_index = {m: j for j, m in enumerate(common.MODELS_20)}

    raw = np.full((len(probe_order), len(common.MODELS_20)), np.nan)
    n_ok, n_err = 0, 0
    for result in client.messages.batches.results(batch_id):
        ds, p_rank_str, safe_model = result.custom_id.split("__", 2)
        p_rank = int(p_rank_str)
        model = next(m for m, s in common.NAME_TO_SAFE.items() if s.replace(".", "-") == safe_model)

        if result.result.type != "succeeded":
            print(f"  [skip] {result.custom_id}: {result.result.type}")
            n_err += 1
            continue

        text = next(
            (b.text for b in result.result.message.content if b.type == "text"), None
        )
        if text is None:
            print(f"  [skip] {result.custom_id}: no text block")
            n_err += 1
            continue
        try:
            parsed = json.loads(text)
            score = float(parsed["score"])
        except Exception as e:
            print(f"  [skip] {result.custom_id}: parse error {e}")
            n_err += 1
            continue

        i = probe_index[(ds, p_rank)]
        j = model_index[model]
        raw[i, j] = score
        n_ok += 1

    print(f"Parsed {n_ok} scores, {n_err} errors/skips (expected total {len(probe_order) * len(common.MODELS_20)}).")
    n_missing = int(np.isnan(raw).sum())
    if n_missing:
        print(f"WARNING: {n_missing} (probe, model) cells missing -- filling with the probe's row mean.")
        row_mean = np.nanmean(raw, axis=1, keepdims=True)
        nan_mask = np.isnan(raw)
        raw = np.where(nan_mask, np.broadcast_to(row_mean, raw.shape), raw)

    # aggregate 66 probes (3/category) -> 22-dim category-mean matrix
    datasets = [c for c in dict.fromkeys(ds for ds, _ in probe_order)]  # unique, first-seen order
    cat_scores = np.zeros((len(datasets), len(common.MODELS_20)))
    for d_i, ds in enumerate(datasets):
        rows = [probe_index[(ds, r)] for r in range(sum(1 for dd, _ in probe_order if dd == ds))]
        cat_scores[d_i, :] = raw[rows, :].mean(axis=0)

    # Override code categories with the original execution-graded score (see CODE_CATEGORIES note above)
    code_ds_present = [ds for ds in datasets if ds in CODE_CATEGORIES]
    if code_ds_present:
        with open(DATA_DIR / "setA_setB_split.pkl", "rb") as f:
            setA = pickle.load(f)["setA"]
        for ds in code_ds_present:
            d_i = datasets.index(ds)
            exec_scores = setA[ds]["scores"].mean(axis=0)  # (20,) in [0,1], model order == common.MODELS_20
            cat_scores[d_i, :] = exec_scores * 9.0 + 1.0  # rescale to 1-10 to match judge-scale magnitude
        print(f"Substituted execution-graded scores for: {code_ds_present} (discarded their judge results)")

    np.save(DATA_DIR / "v1_3_category_scores_raw.npy", cat_scores)
    with open(DATA_DIR / "v1_3_category_order.json", "w", encoding="utf-8") as f:
        json.dump(datasets, f)

    # mean-center per category across the pool, L2-normalize per model (matches build_ceiling_fp_lite20.py)
    pool_mean = cat_scores.mean(axis=1, keepdims=True)
    centered = cat_scores - pool_mean

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for j, m in enumerate(common.MODELS_20):
        vec = centered[:, j]
        vec = (vec / (np.linalg.norm(vec) + 1e-12)).astype(np.float32)
        np.save(OUT_DIR / f"{common.NAME_TO_SAFE[m]}.npy", vec)

    E = np.stack([np.load(OUT_DIR / f"{common.NAME_TO_SAFE[m]}.npy") for m in common.MODELS_20])
    sim = E @ E.T
    off = sim[~np.eye(len(common.MODELS_20), dtype=bool)]
    print(f"V1.3 FP: shape={E.shape}  pairwise cosine sim: mean={off.mean():.4f} std={off.std():.4f} "
          f"min={off.min():.4f} max={off.max():.4f}")
    print(f"Saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
