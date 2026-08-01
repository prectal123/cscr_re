"""LOO unseen-model routing experiment on RouterBench (the real MLP version,
following up on the training-free kNN test in routerbench_knn_test.py).

For each FP type (Ceiling, Perplexity -- V1.2 deprioritized) and each of the
11 RouterBench models M:
  1. Train a query encoder (frozen MiniLM, MEAN-POOLED -- see PROGRESS.md
     section 15.1 for why CLS pooling was wrong -- + small trainable MLP,
     cost_info_nce loss) on the OTHER 10 models only.
  2. Add M's descriptor post-hoc (not used in training).
  3. On Set B (held-out prompts):
     a. Collapse diagnostic: pairwise cosine similarity among 200 diverse
        prompts' query vectors, before/after training, + nearest-descriptor
        distribution -- same method used to diagnose collapse on MixInstruct.
     b. Oracle-match rate: oracle = argmin-cost among models that scored 1.0
        on that prompt (matches get_best_expert's "cheapest correct" convention
        already used elsewhere in this codebase). For prompts where M IS the
        oracle pick, does the trained router's argmax choice equal M?

RouterBench prompts span 86 real, distinct task categories (vs MixInstruct's
largely homogeneous instruction-following style) -- the open question this
script answers is whether that extra diversity is enough for the query
encoder to avoid the prompt-agnostic collapse found on MixInstruct.
"""
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModel, AutoTokenizer

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import loo_unseen_recovery as base  # reuse mean_pool, ProjHead, cost_info_nce, DEVICE, EMBED_MODEL, hyperparams
import routerbench_knn_test as rb   # reuse load_data (same Set A/B split), MODELS, NAMES

CEILING_DIR = rb.CEILING_DIR
PERP_DIR = Path("local_descriptors/routerbench-perplexity")
OUT_DIR = Path("local_descriptors/routerbench-analysis")
N_COLLAPSE_PROBES = 200


class RBFoldDataset(TorchDataset):
    """Binary-correctness version of FoldDataset -- no MARGIN needed since
    RouterBench scores are already 0/1, not a continuous metric."""

    def __init__(self, train_rows, pool_10):
        self.name_to_idx = {n: i for i, n in enumerate(pool_10)}
        self.num_experts = len(pool_10)
        self.items = []
        for prompt, scores in train_rows:
            sub = {m: s for m, s in scores.items() if m in self.name_to_idx}
            if not sub or max(sub.values()) < 1.0:
                continue  # no pool_10 model got this one right -> no positive label
            label = [0.0] * self.num_experts
            for m, s in sub.items():
                if s >= 1.0:
                    label[self.name_to_idx[m]] = 1.0
            self.items.append((prompt, label, None))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class RBSplitDataset(TorchDataset):
    """User's alternative to cost_info_nce_cheapest: instead of picking one
    fixed cheapest-positive target per row (which discards info about other,
    pricier-but-correct experts), duplicate each multi-positive row into
    several ONE-HOT rows, one per positive label. Every gradient step still
    only ever sees a single, unambiguous target (no within-step averaging
    over positives), but no positive-label information is thrown away --
    it's just spread across separate steps/epochs instead of blended within
    one loss call. Works directly with cost_info_nce_cheapest (or plain
    cost_info_nce) since a one-hot label's "cheapest positive" is trivially
    itself."""

    def __init__(self, train_rows, pool_10):
        self.name_to_idx = {n: i for i, n in enumerate(pool_10)}
        self.num_experts = len(pool_10)
        self.items = []
        for prompt, scores in train_rows:
            sub = {m: s for m, s in scores.items() if m in self.name_to_idx}
            positives = [m for m, s in sub.items() if s >= 1.0]
            for m in positives:
                label = [0.0] * self.num_experts
                label[self.name_to_idx[m]] = 1.0
                self.items.append((prompt, label, None))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def build_train_rows(set_a):
    rows = []
    for _, row in set_a.iterrows():
        scores = {name: float(row[col]) for name, col in zip(rb.NAMES, rb.MODELS)}
        rows.append((row["prompt"], scores))
    return rows


def build_cost_dict(set_a):
    cost = {}
    for name, model_col in zip(rb.NAMES, rb.MODELS):
        cost[name] = float(set_a[f"{model_col}|total_cost"].mean())
    return cost


def precompute_set_b_embeddings(texts, tokenizer, base_model, batch_size=64):
    embeds = np.zeros((len(texts), base_model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
            enc = {k: v.to(base.DEVICE) for k, v in enc.items()}
            out = base_model(**enc)
            pooled = base.mean_pool(out.last_hidden_state, enc["attention_mask"])
            embeds[start:start + len(batch)] = pooled.cpu().numpy()
    return embeds


def load_descriptors_ordered(desc_dir, pool):
    from router.utils import load_descriptors
    E, names = load_descriptors(str(desc_dir), pool=pool)
    return np.stack(E), names


def train_fold(pool_10, desc_dir, train_rows, cost_dict, hidden_in_dim=384, seed=0, loss_name="cost_info_nce",
               balance_beta=0.0, dataset_cls=RBFoldDataset, epochs=None):
    torch.manual_seed(seed)
    E, desc_names = load_descriptors_ordered(desc_dir, pool_10)
    E = torch.from_numpy(E).float().to(base.DEVICE)
    E = E / (E.norm(dim=1, keepdim=True) + 1e-9)
    proj_dim = E.size(1)

    ds = dataset_cls(train_rows, desc_names)
    cost_tensor = torch.tensor([cost_dict[n] for n in desc_names], dtype=torch.float32).to(base.DEVICE)
    cost_tensor = (cost_tensor - cost_tensor.min()) / (cost_tensor.max() - cost_tensor.min() + 1e-9)

    head = base.ProjHead(hidden_in_dim, proj_dim).to(base.DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=base.LR)

    def collate(batch):
        texts, idxs, _ = zip(*batch)
        toks = base._TOKENIZER(list(texts), padding=True, truncation=True, return_tensors="pt", max_length=256)
        return toks, torch.tensor(idxs, dtype=torch.long)

    loader = DataLoader(ds, batch_size=base.BATCH_SIZE, shuffle=True, collate_fn=collate)
    n_batches = len(loader)
    n_epochs = epochs if epochs is not None else base.EPOCHS
    print(f"    train set: {len(ds)} rows, {n_batches} batches/epoch x {n_epochs} epochs", flush=True)

    head.train()
    t0 = time.time()
    for ep in range(n_epochs):
        ep_loss = 0.0
        for bi, (tok, label) in enumerate(loader):
            tok = {k: v.to(base.DEVICE) for k, v in tok.items()}
            label = label.to(base.DEVICE)
            with torch.no_grad():
                out = base._BASE_MODEL(**tok)
                cls = base.mean_pool(out.last_hidden_state, tok["attention_mask"])
            q = head(cls)
            if loss_name == "cost_spectrum_info_nce":
                n_bands = int(len(desc_names) ** 0.5)
                loss = base.cost_spectrum_info_nce(q, E, label, cost_tensor, n_bands=n_bands,
                                                    alpha=0.25, tau_min=0.05, gamma=0.2)
            elif loss_name == "cost_info_nce_cheapest":
                loss = base.cost_info_nce_cheapest(q, E, label, cost_tensor, tau=base.TEMPERATURE)
            else:
                loss = base.cost_info_nce(q, E, label, cost_tensor, tau=base.TEMPERATURE)
            bal_loss = base.load_balance_loss(q, E) if balance_beta > 0 else None
            total_loss = loss + balance_beta * bal_loss if bal_loss is not None else loss
            total_loss.backward()
            opt.step()
            opt.zero_grad()
            ep_loss += loss.item()
            if (bi + 1) % 200 == 0 or (bi + 1) == n_batches:
                bal_str = f" bal_loss={bal_loss.item():.4f}" if bal_loss is not None else ""
                print(f"    epoch {ep+1}/{n_epochs} batch {bi+1}/{n_batches} "
                      f"loss={loss.item():.4f}{bal_str} elapsed={time.time()-t0:.1f}s", flush=True)
        print(f"    epoch {ep+1}/{n_epochs} done, avg_loss={ep_loss/n_batches:.4f}", flush=True)

    return head, desc_names


def collapse_diagnostic(head, probe_texts, E11, all_names):
    with torch.no_grad():
        enc = base._TOKENIZER(probe_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        enc = {k: v.to(base.DEVICE) for k, v in enc.items()}
        out = base._BASE_MODEL(**enc)
        cls = base.mean_pool(out.last_hidden_state, enc["attention_mask"])
        q = head(cls)
        sim = q @ q.T
        off = sim[~torch.eye(len(probe_texts), dtype=torch.bool, device=base.DEVICE)]
        sims_to_E = q @ E11.T
        nearest = sims_to_E.argmax(dim=1)
        counts = torch.bincount(nearest, minlength=len(all_names))
    return off.mean().item(), off.std().item(), dict(zip(all_names, counts.cpu().tolist()))


def evaluate_fold(head, held_out, pool_10, desc_dir, pool_11, cls_embeds, true_scores, cost_norm_11, probe_idx, probe_texts):
    E10, names10 = load_descriptors_ordered(desc_dir, pool_10)
    held_out_vec = np.load(Path(desc_dir) / f"{held_out}.npy")
    all_names = names10 + [held_out]
    E11 = np.stack(list(E10) + [held_out_vec])
    E11 = E11 / (np.linalg.norm(E11, axis=1, keepdims=True) + 1e-9)
    E11_t = torch.from_numpy(E11).float().to(base.DEVICE)

    col_idx = [pool_11.index(n) for n in all_names]  # reorder true_scores/cost columns to match all_names
    scores_reordered = true_scores[:, col_idx]
    cost_reordered = cost_norm_11[col_idx]

    with torch.no_grad():
        q = head(torch.from_numpy(cls_embeds).float().to(base.DEVICE))
        sims = q @ E11_t.T
        chosen_idx = sims.argmax(dim=1).cpu().numpy()

    n_oracle_is_M = 0
    n_match = 0
    n_router_correct = 0
    held_out_col = all_names.index(held_out)
    for i in range(scores_reordered.shape[0]):
        row = scores_reordered[i]
        correct_mask = row >= 1.0
        if correct_mask.any():
            masked_cost = np.where(correct_mask, cost_reordered, np.inf)
            oracle_idx = int(np.argmin(masked_cost))
        else:
            oracle_idx = None
        if row[chosen_idx[i]] >= 1.0:
            n_router_correct += 1
        if oracle_idx == held_out_col:
            n_oracle_is_M += 1
            if chosen_idx[i] == held_out_col:
                n_match += 1

    oracle_match_rate = n_match / n_oracle_is_M if n_oracle_is_M else float("nan")
    router_acc = n_router_correct / scores_reordered.shape[0]

    off_mean, off_std, nearest_dist = collapse_diagnostic(
        head, probe_texts, E11_t, all_names
    )

    return {
        "held_out": held_out,
        "n_oracle_is_M": n_oracle_is_M,
        "oracle_match_rate": oracle_match_rate,
        "router_overall_accuracy": router_acc,
        "collapse_qq_sim_mean": off_mean,
        "collapse_qq_sim_std": off_std,
        "collapse_nearest_dist": nearest_dist,
    }


def main():
    print(f"DEVICE: {base.DEVICE}")
    pool_11 = rb.NAMES

    print("Loading frozen MiniLM base (mean pooling)...")
    base._TOKENIZER = AutoTokenizer.from_pretrained(base.EMBED_MODEL)
    base._BASE_MODEL = AutoModel.from_pretrained(base.EMBED_MODEL).to(base.DEVICE)
    base._BASE_MODEL.eval()

    print("Loading RouterBench Set A / Set B...")
    set_a, set_b = rb.load_data()

    print("Precomputing mean-pooled embeddings for Set B...")
    set_b_texts = set_b["prompt"].tolist()
    cls_embeds = precompute_set_b_embeddings(set_b_texts, base._TOKENIZER, base._BASE_MODEL)
    true_scores = np.stack([set_b[m].to_numpy(dtype=float) for m in rb.MODELS], axis=1)
    print(f"cls_embeds shape: {cls_embeds.shape}, true_scores shape: {true_scores.shape}\n")

    rng = random.Random(1)
    probe_idx = rng.sample(range(len(set_b_texts)), N_COLLAPSE_PROBES)
    probe_texts = [set_b_texts[i] for i in probe_idx]

    print("Building train rows from Set A...")
    train_rows = build_train_rows(set_a)
    cost_dict = build_cost_dict(set_a)
    cost_arr = np.array([cost_dict[n] for n in pool_11])
    cost_norm_11 = (cost_arr - cost_arr.min()) / (cost_arr.max() - cost_arr.min() + 1e-9)
    print(f"Train rows: {len(train_rows)}\n")

    all_results = {}
    for fp_name, desc_dir in [("Ceiling", CEILING_DIR), ("Perplexity", PERP_DIR)]:
        print(f"\n{'='*60}\nFP type: {fp_name}\n{'='*60}")
        fold_results = []
        for held_out in pool_11:
            pool_10 = [m for m in pool_11 if m != held_out]
            print(f"[{fp_name}] held out: {held_out} ...")
            head, names10 = train_fold(pool_10, desc_dir, train_rows, cost_dict)
            r = evaluate_fold(head, held_out, pool_10, desc_dir, pool_11, cls_embeds, true_scores,
                               cost_norm_11, probe_idx, probe_texts)
            print(f"  n_oracle_is_M={r['n_oracle_is_M']}  oracle_match_rate={r['oracle_match_rate']:.4f}  "
                  f"router_overall_acc={r['router_overall_accuracy']:.4f}  "
                  f"collapse_qq_sim={r['collapse_qq_sim_mean']:.4f}")
            print(f"  nearest_dist: {r['collapse_nearest_dist']}")
            fold_results.append(r)
        all_results[fp_name] = fold_results

        rates = [r["oracle_match_rate"] for r in fold_results if not math.isnan(r["oracle_match_rate"])]
        chance = 1.0 / len(pool_11)
        print(f"\n--- {fp_name} summary ---")
        print(f"mean oracle_match_rate across {len(rates)} folds: {np.mean(rates):.4f} (chance ~ {chance:.4f})")
        print(f"mean router_overall_accuracy: {np.mean([r['router_overall_accuracy'] for r in fold_results]):.4f}")
        print(f"mean collapse q-q sim: {np.mean([r['collapse_qq_sim_mean'] for r in fold_results]):.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "routerbench_loo_recovery_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {OUT_DIR / 'routerbench_loo_recovery_results.json'}")


if __name__ == "__main__":
    main()
