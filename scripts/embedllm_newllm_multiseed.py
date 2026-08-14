"""Multi-seed version of the "new LLMs" (2/3 seen / 1/3 unseen) AUDC/QNC/Peak
comparison. Seed 0 was already run manually (cost_spectrum_info_nce ->
AUDC=0.4731, Peak=0.536); this script runs SEEDS (additional seeds) end to
end and appends to a combined results file, so we can see the spread across
splits rather than judging from a single draw.

For efficiency, the frozen MiniLM backbone + EmbedLLM train/test data are
loaded ONCE and reused across seeds (only the small proj head + FAISS index
+ split differ per seed). The final AUDC/QNC/Peak number for each seed is
still computed via the actual run_audc_eval.py (subprocess) for methodological
fidelity -- no shortcuts on the metric itself, only on setup/training reuse.
"""
import json
import re
import shutil
import subprocess
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

SEEDS = [1, 2, 3, 4]
PCA5_DIR = Path("local_descriptors/embedllm-ceiling-pca5")
ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 2
BATCH_SIZE = 64
LR = 5e-4
LOG_EVERY = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Already-known seed 0 result (cost_spectrum_info_nce), for the final combined table.
SEED0_RESULT = {"seed": 0, "audc": 0.4731, "qnc": 0.909, "peak": 0.536,
                 "random_audc": 0.3904, "random_peak": 0.423}
CSCR_PAPER = {"audc": 0.4848, "peak": 0.565}


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


def build_split(models, seed):
    import random
    rng = random.Random(seed)
    perm = models[:]
    rng.shuffle(perm)
    n_seen = round(len(perm) * 2 / 3)
    return sorted(perm[:n_seen]), sorted(perm[n_seen:])


def train_encoder_for_seed(seed, seen_models, df, tokenizer, base_model):
    ds = SeenLabelDataset(df, seen_models)
    print(f"  [seed {seed}] train set: {len(ds)} rows", flush=True)

    enc = QueryEncoder.__new__(QueryEncoder)  # reuse loaded backbone, skip re-download
    torch.nn.Module.__init__(enc)
    enc.tokenizer = tokenizer
    enc.model = base_model
    enc.device = DEVICE
    enc.hidden_size = base_model.config.hidden_size
    enc.proj_dim = 5
    enc.proj = torch.nn.Sequential(
        torch.nn.Linear(enc.hidden_size, enc.hidden_size, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(enc.hidden_size, 5, bias=False),
    ).to(DEVICE)
    enc.model.config.proj_dim = 5

    opt = torch.optim.AdamW(enc.proj.parameters(), lr=LR)

    E_seen = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in seen_models])
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


def parse_audc_table(stdout):
    m = re.search(r"knn\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)", stdout)
    knn = {"audc": float(m.group(1)), "qnc": float(m.group(2)), "peak": float(m.group(3))} if m else None
    m = re.search(r"random\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)", stdout)
    rand = {"audc": float(m.group(1)), "qnc": float(m.group(2)), "peak": float(m.group(3))} if m else None
    m = re.search(r"mean delta = ([\-\d.]+).*?p = ([\d.e\-]+)", stdout)
    delta = {"mean_delta": float(m.group(1)), "p": float(m.group(2))} if m else None
    return knn, rand, delta


def main():
    print("Loading EmbedLLM train/test + registry...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "label", "prompt"])

    all_fp_models = sorted(p.stem for p in PCA5_DIR.glob("*.npy"))
    reg_models = [m for m in all_fp_models if m in REGISTRY]
    print(f"{len(reg_models)} registry-covered models", flush=True)

    print(f"Loading frozen MiniLM base ONCE (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    results = [SEED0_RESULT]
    for seed in SEEDS:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)
        seen, unseen = build_split(reg_models, seed)
        print(f"  seen={len(seen)} unseen={len(unseen)}", flush=True)

        split_path = ANALYSIS_DIR / f"newllm_split_seed{seed}.json"
        json.dump({"seen": seen, "unseen": unseen, "seed": seed}, open(split_path, "w"), indent=2)

        unseen_dir = Path(f"local_descriptors/embedllm-ceiling-pca5-unseen-only-seed{seed}")
        unseen_dir.mkdir(parents=True, exist_ok=True)
        for m in unseen:
            shutil.copy(PCA5_DIR / f"{m}.npy", unseen_dir / f"{m}.npy")

        index_dir = Path(f"local_indexes/embedllm-newllm-seed{seed}")
        index_dir.mkdir(parents=True, exist_ok=True)
        import faiss
        E_unseen = np.stack([np.load(unseen_dir / f"{m}.npy") for m in unseen]).astype(np.float32)
        E_unseen = E_unseen / (np.linalg.norm(E_unseen, axis=1, keepdims=True) + 1e-12)
        index = faiss.IndexFlatIP(E_unseen.shape[1])
        index.add(E_unseen)
        index_path = index_dir / "faiss_index.ivf"
        faiss.write_index(index, str(index_path))
        json.dump(unseen, open(str(index_path) + ".labels.json", "w"))

        ckpt_dir = Path(f"local_checkpoints/embedllm-newllm-encoder-csinfonce-seed{seed}")
        if (ckpt_dir / "config.json").exists() and (ckpt_dir / "proj.pt").exists():
            print(f"  [seed {seed}] encoder checkpoint already exists, skipping retrain -> {ckpt_dir}", flush=True)
        else:
            enc = train_encoder_for_seed(seed, seen, df, tokenizer, base_model)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            enc.save(ckpt_dir)
            print(f"  [seed {seed}] saved encoder -> {ckpt_dir}", flush=True)

        print(f"  [seed {seed}] running run_audc_eval.py...", flush=True)
        cmd = [
            sys.executable, "scripts/run_audc_eval.py",
            "--routers", "knn", "random", "oracle",
            "--dataset", "embedllm", "--cost_type", "n_params",
            "--knn_index", str(index_path),
            "--knn_labels", str(index_path) + ".labels.json",
            "--knn_encoder_ckpt", str(ckpt_dir),
            "--knn_bandit_beta", "0.000001", "--k", "20",
            "--parametric_embedding_dir", str(unseen_dir),
            "--n_points", "20", "--cost_grid_points", "20",
            "--sig_pair", "knn,random",
            "--no_show",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=str(Path.cwd()))
        knn, rand, delta = parse_audc_table(proc.stdout)
        if knn is None:
            print(f"  [seed {seed}] FAILED to parse output. stderr tail:\n{proc.stderr[-2000:]}", flush=True)
            results.append({"seed": seed, "audc": None, "qnc": None, "peak": None, "error": True})
            continue
        print(f"  [seed {seed}] RESULT: knn AUDC={knn['audc']:.4f} QNC={knn['qnc']:.3f} Peak={knn['peak']:.4f}  "
              f"random AUDC={rand['audc']:.4f} Peak={rand['peak']:.4f}  "
              f"delta={delta['mean_delta']:+.4f} p={delta['p']:.4g}", flush=True)
        results.append({"seed": seed, "audc": knn["audc"], "qnc": knn["qnc"], "peak": knn["peak"],
                         "random_audc": rand["audc"], "random_peak": rand["peak"],
                         "delta_vs_random": delta["mean_delta"], "p_vs_random": delta["p"]})

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / "newllm_multiseed_results.json"
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n" + "=" * 70)
    print("MULTI-SEED SUMMARY (cost_spectrum_info_nce, 2/3 seen / 1/3 unseen)")
    print("=" * 70)
    audcs, peaks = [], []
    for r in results:
        if r.get("audc") is None:
            continue
        audcs.append(r["audc"]); peaks.append(r["peak"])
        beats_cscr_audc = r["audc"] > CSCR_PAPER["audc"]
        beats_cscr_peak = r["peak"] > CSCR_PAPER["peak"]
        print(f"seed={r['seed']}: AUDC={r['audc']:.4f} ({'BEATS' if beats_cscr_audc else 'below'} CSCR 0.4848)  "
              f"Peak={r['peak']:.4f} ({'BEATS' if beats_cscr_peak else 'below'} CSCR 0.565)")
    if audcs:
        print(f"\nmean AUDC={np.mean(audcs):.4f} (std={np.std(audcs):.4f}, min={min(audcs):.4f}, max={max(audcs):.4f})")
        print(f"mean Peak={np.mean(peaks):.4f} (std={np.std(peaks):.4f}, min={min(peaks):.4f}, max={max(peaks):.4f})")
        n_beat_audc = sum(1 for a in audcs if a > CSCR_PAPER["audc"])
        n_beat_peak = sum(1 for p in peaks if p > CSCR_PAPER["peak"])
        print(f"seeds beating CSCR AUDC: {n_beat_audc}/{len(audcs)}")
        print(f"seeds beating CSCR Peak: {n_beat_peak}/{len(peaks)}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
