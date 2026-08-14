"""No retraining -- reuses the 3 saved head checkpoints from
embedllm_group_loo.py (local_descriptors/embedllm-analysis/group_loo_checkpoints)
to check whether the query encoder collapsed to a small set of "generalist"
models (picking near-independent of query content) rather than doing real
per-query routing.

Two checks:
  1. Overall selection concentration (all 112 candidates, all 3 seeds): what
     share of the 3000 Set B choices does the single most-picked model
     absorb? A healthy, query-sensitive router should spread choices across
     many models; a collapsed one concentrates on a handful regardless of
     query.
  2. Per unseen-model (model, seed) pair: selection_count vs
     hit_rate_when_chosen. The user's hypothesis: models that dominate
     selection are doing so BLINDLY (collapse), not because they're
     genuinely good fits -- so selection_count and hit_rate_when_chosen
     should be NEGATIVELY correlated across unseen (model, seed) instances.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer

PCA5_DIR = Path("local_descriptors/embedllm-ceiling-pca5")
GROUPS_PATH = Path("local_descriptors/embedllm-analysis/embedllm_ceiling_fp_knn_results.json")
CKPT_DIR = Path("local_descriptors/embedllm-analysis/group_loo_checkpoints")
OUT_DIR = Path("local_descriptors/embedllm-analysis")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


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


def load_test_data():
    f_test = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="test.csv")
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    set_b = pd.read_csv(f_test, usecols=["prompt_id", "model_name", "label", "prompt"])
    set_a_models = pd.read_csv(f_train, usecols=["model_name"])
    all_models = sorted(set(set_a_models["model_name"].unique()) | set(set_b["model_name"].unique()))
    prompt_map = {}
    for pid, grp in set_b.groupby("prompt_id", sort=False):
        prompt_map[pid] = (grp["prompt"].iloc[0], dict(zip(grp["model_name"], grp["label"])))
    ids = sorted(prompt_map.keys())
    texts = [prompt_map[pid][0] for pid in ids]
    lbl_maps = [prompt_map[pid][1] for pid in ids]
    return all_models, texts, lbl_maps


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


def main():
    with open(GROUPS_PATH, encoding="utf-8") as f:
        groups_data = json.load(f)
    groups_by_seed = groups_data["groups_by_seed"]

    print("Loading Set B + model list...", flush=True)
    all_models, b_texts, b_lbl_maps = load_test_data()
    print(f"{len(all_models)} models, {len(b_texts)} Set B prompts\n", flush=True)

    print(f"Loading frozen MiniLM base (device={DEVICE})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    b_embeds = precompute_embeddings(b_texts, tokenizer, base_model)

    E_full = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in all_models])
    E_full_t = torch.from_numpy(E_full).float().to(DEVICE)
    E_full_t = E_full_t / (E_full_t.norm(dim=1, keepdim=True) + 1e-9)

    unseen_hit_rows = []  # (seed, model, selection_count, hit_count)
    for seed_str in ["0", "1", "2"]:
        seed = int(seed_str)
        g = groups_by_seed[seed_str]
        unseen_models = set(g["unseen"])

        head = ProjHead(base_model.config.hidden_size, E_full_t.size(1)).to(DEVICE)
        head.load_state_dict(torch.load(CKPT_DIR / f"head_seed{seed}.pt", map_location=DEVICE))
        head.eval()
        with torch.no_grad():
            q = head(torch.from_numpy(b_embeds).float().to(DEVICE))
            sims = q @ E_full_t.T
            chosen_idx = sims.argmax(dim=1).cpu().numpy()

        chosen_models = [all_models[i] for i in chosen_idx]
        sel_counts = pd.Series(chosen_models).value_counts()

        # --- overall concentration (all 112 candidates) ---
        top1_share = sel_counts.iloc[0] / len(chosen_models)
        top5_share = sel_counts.iloc[:5].sum() / len(chosen_models)
        n_models_ever_chosen = len(sel_counts)
        probs = sel_counts.to_numpy() / sel_counts.sum()
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(all_models))
        print(f"[seed {seed}] top1_share={top1_share:.4f} ({sel_counts.index[0]})  "
              f"top5_share={top5_share:.4f}  n_models_ever_chosen={n_models_ever_chosen}/{len(all_models)}  "
              f"entropy={entropy:.3f}/{max_entropy:.3f} ({entropy/max_entropy*100:.1f}% of max)", flush=True)

        # --- per-unseen-model selection_count vs hit_rate_when_chosen ---
        for m in unseen_models:
            sel_count = int(sel_counts.get(m, 0))
            if sel_count == 0:
                continue
            hit_count = sum(1 for i, cm in enumerate(chosen_models)
                             if cm == m and b_lbl_maps[i].get(m, 0) == 1)
            unseen_hit_rows.append({"seed": seed, "model": m, "selection_count": sel_count,
                                     "hit_rate_when_chosen": hit_count / sel_count})

    print(f"\n{len(unseen_hit_rows)} (model, seed) unseen instances with >=1 selection", flush=True)
    df = pd.DataFrame(unseen_hit_rows)
    print(df.sort_values("selection_count", ascending=False).to_string(index=False))

    rho, p = spearmanr(df["selection_count"], df["hit_rate_when_chosen"])
    print(f"\nSpearman(selection_count, hit_rate_when_chosen) across unseen (model,seed) instances: "
          f"rho={rho:.4f}  p={p:.4g}  n={len(df)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "embedllm_selection_diagnostic_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"unseen_instances": unseen_hit_rows, "spearman_rho": float(rho), "spearman_p": float(p)},
                   f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
