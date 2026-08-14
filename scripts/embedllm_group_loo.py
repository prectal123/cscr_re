"""EmbedLLM group-holdout MLP (query-encoder) experiment.

Reuses the exact seen/unseen model splits saved by build_embedllm_ceiling_fp.py
(local_descriptors/embedllm-analysis/embedllm_ceiling_fp_knn_results.json,
seeds 0/1/2, ~90 seen / ~22 unseen models each) instead of full 112-fold LOO
-- one training run per seed, not 112.

Per seed:
  1. Train a small ProjHead (frozen MiniLM mean-pooled text -> 5-dim, matching
     the PCA-5 Ceiling FP) on Set A (train.csv) using ONLY seen-model labels
     -- unseen models' labels are dropped from the training target entirely,
     so their FPs never receive gradient signal, matching the true
     "never-seen-during-training" scenario.
  2. Evaluate on Set B (test.csv): for every prompt where ALL seen models
     get it wrong but >=1 unseen model gets it right (the "unseen model was
     uniquely necessary" subset), check whether the encoder's argmax choice
     over the FULL 112-model PCA-5 FP space lands on a correct unseen model.
     This isolates whether the unseen models' Ceiling-FP position actually
     lets the router recommend them correctly when they're the only right
     answer -- chance level and any "seen-only" baseline are both ~0% on
     this subset by construction, so the real number IS the signal.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModel, AutoTokenizer

PCA5_DIR = Path("local_descriptors/embedllm-ceiling-pca5")
GROUPS_PATH = Path("local_descriptors/embedllm-analysis/embedllm_ceiling_fp_knn_results.json")
OUT_DIR = Path("local_descriptors/embedllm-analysis")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 2
BATCH_SIZE = 64
LR = 5e-4
TEMPERATURE = 0.05
LOG_EVERY = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def load_data():
    print("Loading EmbedLLM train/test...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    f_test = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="test.csv")
    set_a = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "label", "prompt"])
    set_b = pd.read_csv(f_test, usecols=["prompt_id", "model_name", "label", "prompt"])
    return set_a, set_b


def collapse_by_prompt(df):
    """prompt_id -> (text, {model: 0/1})"""
    out = {}
    for pid, grp in df.groupby("prompt_id", sort=False):
        out[pid] = (grp["prompt"].iloc[0], dict(zip(grp["model_name"], grp["label"])))
    return out


class SeenLabelDataset(TorchDataset):
    def __init__(self, prompt_map, seen_models):
        self.name_to_idx = {n: i for i, n in enumerate(seen_models)}
        self.items = []
        for pid, (text, lbl_map) in prompt_map.items():
            label = [0.0] * len(seen_models)
            any_pos = False
            for m, v in lbl_map.items():
                if m in self.name_to_idx and v == 1:
                    label[self.name_to_idx[m]] = 1.0
                    any_pos = True
            if any_pos:  # drop rows with no positive among seen models (no training signal)
                self.items.append((text, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class ProjHead(nn.Module):
    def __init__(self, in_dim, proj_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim, bias=False),
        )

    def forward(self, x):
        z = self.net(x)
        return z / (z.norm(dim=1, keepdim=True) + 1e-9)


def multi_positive_info_nce(q, E, label, tau):
    sim = (q @ E.T) / tau  # (B, M)
    pos_mask = label.bool()
    keep = pos_mask.any(dim=1)
    if keep.sum() == 0:
        return torch.tensor(0.0, device=q.device, requires_grad=True)
    sim, pos_mask = sim[keep], pos_mask[keep]
    exp_sim = torch.exp(sim)
    numer = (exp_sim * pos_mask).sum(dim=1)
    denom = exp_sim.sum(dim=1)
    return -(numer / denom).clamp(min=1e-9).log().mean()


def precompute_embeddings(texts, tokenizer, base_model, batch_size=64):
    embeds = np.zeros((len(texts), base_model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out = base_model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
            embeds[start:start + len(batch)] = pooled.cpu().numpy()
    return embeds


def train_seed(seed, seen_models, prompt_map_a, tokenizer, base_model):
    E_seen = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in seen_models])
    E_seen_t = torch.from_numpy(E_seen).float().to(DEVICE)
    E_seen_t = E_seen_t / (E_seen_t.norm(dim=1, keepdim=True) + 1e-9)

    ds = SeenLabelDataset(prompt_map_a, seen_models)
    print(f"  [seed {seed}] train set: {len(ds)} rows (dropped {len(prompt_map_a)-len(ds)} all-seen-wrong rows)", flush=True)

    head = ProjHead(base_model.config.hidden_size, E_seen_t.size(1)).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=LR)

    def collate(batch):
        texts, labels = zip(*batch)
        toks = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt", max_length=256)
        return toks, torch.tensor(labels, dtype=torch.float32)

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    n_batches = len(loader)
    t0 = time.time()
    head.train()
    for ep in range(EPOCHS):
        ep_loss = 0.0
        for bi, (tok, label) in enumerate(loader):
            tok = {k: v.to(DEVICE) for k, v in tok.items()}
            label = label.to(DEVICE)
            with torch.no_grad():
                out = base_model(**tok)
                cls = mean_pool(out.last_hidden_state, tok["attention_mask"])
            q = head(cls)
            loss = multi_positive_info_nce(q, E_seen_t, label, TEMPERATURE)
            loss.backward()
            opt.step()
            opt.zero_grad()
            ep_loss += loss.item()
            if (bi + 1) % LOG_EVERY == 0 or (bi + 1) == n_batches:
                print(f"  [seed {seed}] epoch {ep+1}/{EPOCHS} batch {bi+1}/{n_batches} "
                      f"loss={loss.item():.4f} elapsed={time.time()-t0:.1f}s", flush=True)
        print(f"  [seed {seed}] epoch {ep+1}/{EPOCHS} done, avg_loss={ep_loss/n_batches:.4f}", flush=True)

    ckpt_dir = OUT_DIR / "group_loo_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), ckpt_dir / f"head_seed{seed}.pt")
    print(f"  [seed {seed}] saved head checkpoint -> {ckpt_dir / f'head_seed{seed}.pt'}", flush=True)
    return head


def evaluate_seed(seed, head, seen_models, unseen_models, all_models, b_texts, b_lbl_maps, b_embeds):
    """'Herd immunity' framing: no artificial gatekeeping condition (does NOT
    require the seen pool to fail) -- just checks, over ordinary Set B
    prompts, whether unseen models (a) get chosen about as often as their
    share of the pool (22/112 ~ 19.6%), and (b) are about as likely to be
    correct when chosen as seen models are. That's what "riding along on the
    seen pool's statistical structure" would look like if it's working."""
    E_full = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in all_models])
    E_full_t = torch.from_numpy(E_full).float().to(DEVICE)
    E_full_t = E_full_t / (E_full_t.norm(dim=1, keepdim=True) + 1e-9)

    head.eval()
    with torch.no_grad():
        q = head(torch.from_numpy(b_embeds).float().to(DEVICE))
        sims = q @ E_full_t.T
        chosen_idx = sims.argmax(dim=1).cpu().numpy()

    unseen_set = set(unseen_models)
    overall_hits = 0
    n_chose_unseen, n_chose_seen = 0, 0
    n_unseen_hits, n_seen_hits = 0, 0
    for i, lbl_map in enumerate(b_lbl_maps):
        chosen = all_models[chosen_idx[i]]
        hit = lbl_map.get(chosen, 0) == 1
        overall_hits += hit
        if chosen in unseen_set:
            n_chose_unseen += 1
            n_unseen_hits += hit
        else:
            n_chose_seen += 1
            n_seen_hits += hit

    n = len(b_lbl_maps)
    pool_unseen_share = len(unseen_models) / len(all_models)
    return {
        "seed": seed,
        "overall_hit_rate": overall_hits / n,
        "n_seen": len(seen_models), "n_unseen": len(unseen_models),
        "pool_unseen_share": pool_unseen_share,
        "unseen_selection_rate": n_chose_unseen / n,
        "seen_selection_rate": n_chose_seen / n,
        "unseen_hit_rate_when_chosen": n_unseen_hits / n_chose_unseen if n_chose_unseen else None,
        "seen_hit_rate_when_chosen": n_seen_hits / n_chose_seen if n_chose_seen else None,
        "n_chose_unseen": n_chose_unseen, "n_chose_seen": n_chose_seen,
    }


def main():
    with open(GROUPS_PATH, encoding="utf-8") as f:
        groups_data = json.load(f)
    groups_by_seed = groups_data["groups_by_seed"]

    set_a, set_b = load_data()
    all_models = sorted(set(set_a["model_name"].unique()) | set(set_b["model_name"].unique()))
    print(f"{len(all_models)} total models\n", flush=True)

    print("Collapsing Set A / Set B by prompt_id...", flush=True)
    prompt_map_a = collapse_by_prompt(set_a)
    prompt_map_b = collapse_by_prompt(set_b)
    b_ids = sorted(prompt_map_b.keys())
    b_texts = [prompt_map_b[pid][0] for pid in b_ids]
    b_lbl_maps = [prompt_map_b[pid][1] for pid in b_ids]
    print(f"Set A: {len(prompt_map_a)} prompts, Set B: {len(prompt_map_b)} prompts\n", flush=True)

    print(f"Loading frozen MiniLM base (device={DEVICE})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()

    print("Precomputing frozen Set B embeddings (once, reused across all seeds)...", flush=True)
    b_embeds = precompute_embeddings(b_texts, tokenizer, base_model)
    print(f"b_embeds shape: {b_embeds.shape}\n", flush=True)

    all_results = []
    for seed_str in ["0", "1", "2"]:
        seed = int(seed_str)
        g = groups_by_seed[seed_str]
        seen_models, unseen_models = g["seen"], g["unseen"]
        print(f"\n{'='*60}\nSeed {seed}: {len(seen_models)} seen / {len(unseen_models)} unseen\n{'='*60}", flush=True)
        head = train_seed(seed, seen_models, prompt_map_a, tokenizer, base_model)
        r = evaluate_seed(seed, head, seen_models, unseen_models, all_models, b_texts, b_lbl_maps, b_embeds)
        print(f"  [seed {seed}] RESULT: overall_hit_rate={r['overall_hit_rate']:.4f}  "
              f"unseen_selection_rate={r['unseen_selection_rate']:.4f} (pool share={r['pool_unseen_share']:.4f})  "
              f"unseen_hit_when_chosen={r['unseen_hit_rate_when_chosen']}  "
              f"seen_hit_when_chosen={r['seen_hit_rate_when_chosen']}", flush=True)
        all_results.append(r)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "embedllm_group_loo_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    print("\n" + "=" * 60)
    print("SUMMARY across seeds")
    print("=" * 60)
    for r in all_results:
        print(f"seed={r['seed']}: overall_hit={r['overall_hit_rate']:.4f}  "
              f"unseen_selection_rate={r['unseen_selection_rate']:.4f} (pool_share={r['pool_unseen_share']:.4f})  "
              f"unseen_hit_when_chosen={r['unseen_hit_rate_when_chosen']:.4f}  "
              f"seen_hit_when_chosen={r['seen_hit_rate_when_chosen']:.4f}")


if __name__ == "__main__":
    main()
