"""Probe-sampled ("Ceiling V1"-style) Ceiling FP for EmbedLLM's "new LLMs"
comparison, at N in {6, 12, 24} probes/category, EPOCHS=2 (unchanged from
the original run -- only the FP construction varies here, not training
budget). Reuses the SAME 5 seen/unseen model splits as the earlier 2-epoch
full-average (Ceiling V2) run for a like-for-like comparison.

Probe selection: within each of EmbedLLM's 80 categories, rank prompts by
cross-model label VARIANCE (computed over all registry-covered models --
this is a design-time property of the pool, independent of which specific
model later gets held out as "unseen", so it doesn't leak seen/unseen
information) and take the top N. This mirrors probe_count_ablation.py's
methodology (top-variance selection) applied fresh to EmbedLLM, which has no
precomputed probe_info.json of its own.

For each N: build the probe-sampled category matrix -> PCA-5 (own PCA per N,
not reusing the full-average PCA-5) -> for each seed, build unseen-only FP
dir + FAISS index, train encoder (cost_spectrum_info_nce, 2 epochs) on seen
labels against this FP, fast-eval on unseen candidates.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset as TorchDataset

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from router.query_encoder import QueryEncoder
from router.registry import REGISTRY
from train_query_encoder import cost_spectrum_info_nce
from embedllm_newllm_fast_eval import run_one as fast_eval

ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 2
BATCH_SIZE = 64
LR = 5e-4
LOG_EVERY = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_PCA = 5

N_SWEEP = [6, 12, 24]
SEEDS = [0, 1, 2, 3, 4]
CSCR_PAPER = {"audc": 0.4848, "peak": 0.565}


def split_path_for(seed):
    return ANALYSIS_DIR / ("newllm_split.json" if seed == 0 else f"newllm_split_seed{seed}.json")


class SeenLabelDataset(TorchDataset):
    def __init__(self, df, seen_models):
        name_to_idx = {n: i for i, n in enumerate(seen_models)}
        self.items = []
        for pid, grp in df.groupby("prompt_id", sort=False):
            text = grp["prompt"].iloc[0]
            label = [0.0] * len(seen_models)
            any_pos = False
            for m, v in zip(grp["model_name"], grp["label"]):
                if m in name_to_idx and v == 1:
                    label[name_to_idx[m]] = 1.0
                    any_pos = True
            if any_pos:
                self.items.append((text, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def build_probe_sampled_pca5(df, n_per_category, models, categories, out_dir):
    print(f"  Selecting top-{n_per_category} variance probes/category...", flush=True)
    per_prompt = df.groupby("prompt_id").agg(category=("category", "first"), var=("label", "var")).reset_index()
    selected_ids = set()
    for cat, grp in per_prompt.groupby("category"):
        top = grp.nlargest(n_per_category, "var")
        selected_ids.update(top["prompt_id"].tolist())
    sub = df[df["prompt_id"].isin(selected_ids)]
    print(f"  {len(selected_ids)} probes selected total ({len(categories)} categories)", flush=True)

    pivot = sub.pivot_table(index="model_name", columns="category", values="label", aggfunc="mean")
    pivot = pivot.reindex(index=models, columns=categories)
    raw = pivot.to_numpy()
    col_mean = np.nanmean(raw, axis=0, keepdims=True)
    raw = np.where(np.isnan(raw), col_mean, raw)
    pool_mean = raw.mean(axis=0, keepdims=True)
    centered = raw - pool_mean

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()
    reduced = centered @ Vt[:K_PCA].T
    E = reduced / (np.linalg.norm(reduced, axis=1, keepdims=True) + 1e-12)

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(models):
        np.save(out_dir / f"{m}.npy", E[i].astype(np.float32))
    print(f"  saved -> {out_dir} (top-{K_PCA} cum. explained var={np.cumsum(explained)[K_PCA-1]:.4f})", flush=True)
    return out_dir


def train_encoder(seed, seen_models, df_labels, tokenizer, base_model, fp_dir):
    ds = SeenLabelDataset(df_labels, seen_models)
    print(f"  [seed {seed}] train set: {len(ds)} rows", flush=True)

    enc = QueryEncoder.__new__(QueryEncoder)
    torch.nn.Module.__init__(enc)
    enc.tokenizer = tokenizer
    enc.model = base_model
    enc.device = DEVICE
    enc.hidden_size = base_model.config.hidden_size
    enc.proj_dim = K_PCA
    enc.proj = torch.nn.Sequential(
        torch.nn.Linear(enc.hidden_size, enc.hidden_size, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(enc.hidden_size, K_PCA, bias=False),
    ).to(DEVICE)
    enc.model.config.proj_dim = K_PCA
    opt = torch.optim.AdamW(enc.proj.parameters(), lr=LR)

    E_seen = np.stack([np.load(fp_dir / f"{m}.npy") for m in seen_models])
    E_seen_t = torch.from_numpy(E_seen).float().to(DEVICE)
    E_seen_t = E_seen_t / (E_seen_t.norm(dim=1, keepdim=True) + 1e-9)

    cost_raw = np.array([REGISTRY[m]["n_params"] for m in seen_models], dtype=np.float32)
    cost_norm = (cost_raw - cost_raw.min()) / (cost_raw.max() - cost_raw.min() + 1e-9)
    cost_norm_t = torch.from_numpy(cost_norm).float().to(DEVICE)

    def collate(batch):
        texts, labels = zip(*batch)
        toks = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt", max_length=256)
        return toks, torch.tensor(labels, dtype=torch.float32)

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    n_batches = len(loader)
    t0 = time.time()
    for ep in range(EPOCHS):
        ep_loss = 0.0
        for bi, (tok, label) in enumerate(loader):
            tok = {k: v.to(DEVICE) for k, v in tok.items()}
            label = label.to(DEVICE)
            with torch.no_grad():
                out = base_model(**tok)
                cls_vec = out.last_hidden_state[:, 0]
            q = enc._project(cls_vec)
            loss = cost_spectrum_info_nce(q, E_seen_t, label.bool(), cost_norm_t)
            loss.backward()
            opt.step()
            opt.zero_grad()
            ep_loss += loss.item()
            if (bi + 1) % LOG_EVERY == 0 or (bi + 1) == n_batches:
                print(f"  [seed {seed}] epoch {ep+1}/{EPOCHS} batch {bi+1}/{n_batches} "
                      f"loss={loss.item():.4f} elapsed={time.time()-t0:.1f}s", flush=True)
        print(f"  [seed {seed}] epoch {ep+1}/{EPOCHS} done, avg_loss={ep_loss/n_batches:.4f}", flush=True)
    return enc


def main():
    print("Loading EmbedLLM train.csv (with category) + registry...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df_full = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "label", "prompt", "category"])
    df_labels = df_full[["prompt_id", "model_name", "label", "prompt"]]

    all_models = sorted(df_full["model_name"].unique())
    models = [m for m in all_models if m in REGISTRY]
    categories = sorted(df_full["category"].unique())
    print(f"{len(models)} registry-covered models, {len(categories)} categories", flush=True)

    print(f"Loading frozen MiniLM base ONCE (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    all_results = {}
    for n in N_SWEEP:
        print(f"\n{'#'*70}\nPROBE COUNT N={n}\n{'#'*70}", flush=True)
        fp_dir = Path(f"local_descriptors/embedllm-ceiling-probeN{n}-pca5")
        if not any(fp_dir.glob("*.npy")):
            build_probe_sampled_pca5(df_full, n, models, categories, fp_dir)
        else:
            print(f"  FP dir already built -> {fp_dir}", flush=True)

        n_results = []
        for seed in SEEDS:
            print(f"\n{'='*60}\nN={n} SEED {seed}\n{'='*60}", flush=True)
            split = json.load(open(split_path_for(seed), encoding="utf-8"))
            seen_models, unseen_models = split["seen"], split["unseen"]

            unseen_dir = Path(f"local_descriptors/embedllm-ceiling-probeN{n}-pca5-unseen-only-seed{seed}")
            unseen_dir.mkdir(parents=True, exist_ok=True)
            for m in unseen_models:
                shutil.copy(fp_dir / f"{m}.npy", unseen_dir / f"{m}.npy")

            ckpt_dir = Path(f"local_checkpoints/embedllm-newllm-encoder-csinfonce-probeN{n}-seed{seed}")
            if (ckpt_dir / "config.json").exists() and (ckpt_dir / "proj.pt").exists():
                print(f"  [seed {seed}] checkpoint already exists, skipping retrain -> {ckpt_dir}", flush=True)
            else:
                enc = train_encoder(seed, seen_models, df_labels, tokenizer, base_model, fp_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                enc.save(ckpt_dir)
                print(f"  [seed {seed}] saved encoder -> {ckpt_dir}", flush=True)

            r = fast_eval(str(split_path_for(seed)), str(unseen_dir), str(ckpt_dir), label=f"N{n}-seed{seed}")
            r["seed"] = seed
            n_results.append(r)

        all_results[n] = n_results
        audcs = [r["knn"]["audc"] for r in n_results]
        peaks = [r["knn"]["peak"] for r in n_results]
        print(f"\n--- N={n} summary ---")
        for r in n_results:
            beats = "BEATS" if r["knn"]["audc"] > CSCR_PAPER["audc"] else "below"
            print(f"  seed={r['seed']}: AUDC={r['knn']['audc']:.4f} ({beats} CSCR)  Peak={r['knn']['peak']:.4f}")
        print(f"  mean AUDC={np.mean(audcs):.4f} (std={np.std(audcs):.4f})  mean Peak={np.mean(peaks):.4f}")
        print(f"  seeds beating CSCR: {sum(1 for a in audcs if a > CSCR_PAPER['audc'])}/{len(audcs)}")

    out_path = ANALYSIS_DIR / "newllm_probe_sampling_results.json"
    json.dump(all_results, open(out_path, "w"), indent=2)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY: probe-sampled Ceiling FP, EPOCHS=2, 5 seeds each")
    print("=" * 70)
    print(f"{'N':>5s} {'mean AUDC':>10s} {'std':>7s} {'mean Peak':>10s} {'beat CSCR':>10s}")
    for n in N_SWEEP:
        audcs = [r["knn"]["audc"] for r in all_results[n]]
        peaks = [r["knn"]["peak"] for r in all_results[n]]
        print(f"{n:>5d} {np.mean(audcs):>10.4f} {np.std(audcs):>7.4f} {np.mean(peaks):>10.4f} "
              f"{sum(1 for a in audcs if a > CSCR_PAPER['audc']):>7d}/5")
    print("(reference: full-average Ceiling V2, 2 epoch: mean AUDC=0.468, std=0.029, 1/5 beat CSCR)")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
