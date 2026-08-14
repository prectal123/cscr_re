"""Beta sweep for the load-balanced GRPO loss (embedllm_newllm_grpo_train_balanced.py),
seed 0 only, loading EmbedLLM train.csv + frozen MiniLM ONCE and reusing across
betas (mirrors embedllm_newllm_grpo_multiseed.py's efficient-reuse pattern).

beta=0 (no balance) and beta=1.0 already run separately:
  beta=0:   AUDC=0.5289 Peak=0.5687  top3_share=0.853  rho(sel,true_acc)=0.42(p=0.012)
  beta=1.0: AUDC=0.4487 Peak=0.4693  top3_share=0.376  rho(sel,true_acc)=0.07(ns)
Sweeping the values in between to find a point that reduces collapse without
destroying the accuracy-signal correlation.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from router.query_encoder import QueryEncoder
from router.embedllm import load_embedllm
from embedllm_newllm_fast_eval import run_one as fast_eval

PCA5_DIR = Path("local_descriptors/embedllm-ceiling-pca5")
ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
UNSEEN_DIR = Path("local_descriptors/embedllm-ceiling-pca5-unseen-only")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
HOLDOUT_FRAC = 0.15
SEED = 0
TAU = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT_PATH = ANALYSIS_DIR / "newllm_split.json"
CSCR_PAPER = {"audc": 0.4848, "peak": 0.565}

BETAS = [0.05, 0.1, 0.2, 0.5]


class QueryTargetDataset(torch.utils.data.Dataset):
    def __init__(self, df, seen_models):
        name_to_idx = {n: i for i, n in enumerate(seen_models)}
        self.items = []
        for pid, grp in df.groupby("prompt_id", sort=False):
            text = grp["prompt"].iloc[0]
            labels = np.full(len(seen_models), np.nan, dtype=np.float32)
            for m, v in zip(grp["model_name"], grp["label"]):
                if m in name_to_idx:
                    labels[name_to_idx[m]] = float(v)
            mask = ~np.isnan(labels)
            if mask.sum() < 2:
                continue
            vals = labels[mask]
            mean, std = vals.mean(), vals.std()
            target = np.zeros(len(seen_models), dtype=np.float32)
            target[mask] = (vals - mean) / (std + 1e-6)
            self.items.append((text, target, mask.astype(np.float32)))

    def __len__(self):
        return len(self.items)


def load_balance_loss(q, E, tau):
    M = E.size(0)
    probs = torch.softmax((q @ E.T) / tau, dim=1)
    P = probs.mean(dim=0)
    return M * (P ** 2).sum()


def evaluate_holdout(enc, texts, targets, masks, E_seen_t, batch_size=64):
    enc.model.eval()
    embeds = np.zeros((len(texts), E_seen_t.size(1)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            embeds[start:start + len(batch)] = enc.encode(batch)
    cos = embeds @ E_seen_t.cpu().numpy().T
    rhos = []
    for i in range(len(texts)):
        m = masks[i].astype(bool)
        if m.sum() < 3:
            continue
        rho, _ = spearmanr(cos[i, m], targets[i, m])
        if not np.isnan(rho):
            rhos.append(rho)
    return np.array(rhos)


def collapse_stats(enc, unseen, texts, label_maps):
    embeds = np.zeros((len(texts), 5), dtype=np.float32)
    for start in range(0, len(texts), 64):
        batch = texts[start:start + 64]
        embeds[start:start + len(batch)] = enc.encode(batch)
    E_unseen = np.stack([np.load(UNSEEN_DIR / f"{m}.npy") for m in unseen]).astype(np.float32)
    E_unseen = E_unseen / (np.linalg.norm(E_unseen, axis=1, keepdims=True) + 1e-12)
    sims = embeds @ E_unseen.T
    chosen = np.argmax(sims, axis=1)
    counts = np.bincount(chosen, minlength=len(unseen))
    order = np.argsort(-counts)
    top3_share = counts[order[:3]].sum() / len(texts)
    n_used = int((counts > 0).sum())
    true_acc = np.array([np.mean([lm.get(m, 0) for lm in label_maps]) for m in unseen])
    rho, p = spearmanr(counts, true_acc)
    return {"top3_share": float(top3_share), "n_used": n_used, "n_total": len(unseen),
            "rho_selection_vs_true_acc": float(rho), "p": float(p)}


def train_one_beta(beta, seen_models, train_items, holdout_texts, holdout_targets, holdout_masks,
                    tokenizer, base_model, E_seen_t):
    torch.manual_seed(SEED)
    enc = QueryEncoder.__new__(QueryEncoder)
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
    opt = torch.optim.Adam(enc.proj.parameters(), lr=LR)

    def collate(batch):
        texts, targets, masks = zip(*batch)
        toks = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt", max_length=256)
        return toks, torch.tensor(np.stack(targets)), torch.tensor(np.stack(masks))

    loader = torch.utils.data.DataLoader(train_items, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    best_rho, best_epoch, best_state = -1.0, -1, None
    t0 = time.time()
    for ep in range(EPOCHS):
        base_model.eval()
        for tok, target, mask in loader:
            tok = {k: v.to(DEVICE) for k, v in tok.items()}
            target, mask = target.to(DEVICE), mask.to(DEVICE)
            with torch.no_grad():
                out = base_model(**tok)
                cls_vec = out.last_hidden_state[:, 0]
            q = enc._project(cls_vec)
            cos_sim = q @ E_seen_t.T
            mse = ((cos_sim - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            bal = load_balance_loss(q, E_seen_t, TAU)
            loss = mse + beta * bal
            loss.backward()
            opt.step()
            opt.zero_grad()

        rho_arr = evaluate_holdout(enc, holdout_texts, holdout_targets, holdout_masks, E_seen_t)
        if rho_arr.mean() > best_rho:
            best_rho, best_epoch = rho_arr.mean(), ep + 1
            best_state = {k: v.clone().cpu() for k, v in enc.proj.state_dict().items()}
    print(f"  [beta={beta}] best_epoch={best_epoch} best_holdout_rho={best_rho:.4f} time={time.time()-t0:.1f}s", flush=True)

    enc.proj.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return enc, best_epoch, best_rho


def main():
    print("Loading EmbedLLM train.csv...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "label", "prompt"])

    split = json.load(open(SPLIT_PATH, encoding="utf-8"))
    seen_models, unseen = split["seen"], split["unseen"]

    ds_all = QueryTargetDataset(df, seen_models)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(ds_all))
    n_holdout = int(len(ds_all) * HOLDOUT_FRAC)
    holdout_idx, train_idx = perm[:n_holdout], perm[n_holdout:]
    train_items = [ds_all.items[i] for i in train_idx]
    holdout_texts = [ds_all.items[i][0] for i in holdout_idx]
    holdout_targets = np.stack([ds_all.items[i][1] for i in holdout_idx])
    holdout_masks = np.stack([ds_all.items[i][2] for i in holdout_idx])
    print(f"train={len(train_items)} holdout={len(holdout_texts)}", flush=True)

    print(f"Loading frozen MiniLM base ONCE (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    E_seen = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in seen_models])
    E_seen_t = torch.from_numpy(E_seen).float().to(DEVICE)
    E_seen_t = E_seen_t / (E_seen_t.norm(dim=1, keepdim=True) + 1e-9)

    print("Loading EmbedLLM test set (for both fast_eval AUDC and collapse stats)...", flush=True)
    test_dataset = load_embedllm("test", candidates=unseen)
    test_texts = [ex["prompt"] for ex in test_dataset]
    test_label_maps = [ex["label_map"] for ex in test_dataset]

    results = []
    for beta in BETAS:
        print(f"\n{'='*60}\nBETA={beta}\n{'='*60}", flush=True)
        enc, best_epoch, best_rho = train_one_beta(
            beta, seen_models, train_items, holdout_texts, holdout_targets, holdout_masks,
            tokenizer, base_model, E_seen_t)

        ckpt_dir = Path(f"local_checkpoints/embedllm-newllm-encoder-grpo-balanced-seed0-beta{beta}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        enc.save(ckpt_dir)

        r = fast_eval(str(SPLIT_PATH), str(UNSEEN_DIR), str(ckpt_dir), label=f"beta{beta}")
        cstats = collapse_stats(enc, unseen, test_texts, test_label_maps)
        r["beta"] = beta
        r["best_epoch"] = best_epoch
        r["best_holdout_rho"] = float(best_rho)
        r["collapse"] = cstats
        results.append(r)
        print(f"  [beta={beta}] AUDC={r['knn']['audc']:.4f} Peak={r['knn']['peak']:.4f}  "
              f"top3_share={cstats['top3_share']:.3f}  n_used={cstats['n_used']}/{cstats['n_total']}  "
              f"rho(sel,true_acc)={cstats['rho_selection_vs_true_acc']:.3f}(p={cstats['p']:.3f})", flush=True)

    out_path = ANALYSIS_DIR / "newllm_grpo_beta_sweep_results.json"
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n" + "=" * 100)
    print("BETA SWEEP SUMMARY (seed 0)")
    print("=" * 100)
    print(f"{'beta':>6} | {'AUDC':>7} | {'Peak':>7} | {'top3_share':>10} | {'n_used':>7} | {'rho(sel,acc)':>13} | {'p':>7}")
    print("-" * 100)
    print(f"{'0(ref)':>6} | {0.5289:>7.4f} | {0.5687:>7.4f} | {0.853:>10.3f} | {'12/35':>7} | {0.42:>13.3f} | {0.012:>7.3f}")
    for r in results:
        c = r["collapse"]
        print(f"{r['beta']:>6} | {r['knn']['audc']:>7.4f} | {r['knn']['peak']:>7.4f} | {c['top3_share']:>10.3f} | "
              f"{c['n_used']}/{c['n_total']:>4} | {c['rho_selection_vs_true_acc']:>13.3f} | {c['p']:>7.3f}")
    print(f"{'1.0(ref)':>6} | {0.4487:>7.4f} | {0.4693:>7.4f} | {0.376:>10.3f} | {'31/35':>7} | {0.07:>13.3f} | {0.708:>7.3f}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
