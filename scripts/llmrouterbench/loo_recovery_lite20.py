"""LOO unseen-model routing for the lightweight-20 pool, 22-dataset version.
Mirrors loo_recovery.py exactly but points at common_lite20 / llmrouterbench_lite20.
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset

sys.path.insert(0, "scripts/llmrouterbench")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import common_lite20 as common
import loo_unseen_recovery as base

DATA_DIR = Path("local_descriptors/llmrouterbench_lite20")
CEILING_DIR = DATA_DIR / "ceiling"
PERP_DIR = DATA_DIR / "perplexity"
N_COLLAPSE_PROBES = 200


def load_split():
    with open(DATA_DIR / "setA_setB_split.pkl", "rb") as f:
        return pickle.load(f)


def build_train_rows(split):
    rows = []
    for ds in common.DATASETS:
        d = split["setA"][ds]
        n = len(d["queries"])
        for i in range(n):
            scores = {m: float(d["scores"][i, j]) for j, m in enumerate(common.MODELS_20)}
            rows.append((d["queries"][i], scores))
    return rows


def build_cost_dict(split):
    all_costs = {m: [] for m in common.MODELS_20}
    for ds in common.DATASETS:
        d = split["setA"][ds]
        for j, m in enumerate(common.MODELS_20):
            all_costs[m].extend(d["costs"][:, j].tolist())
    return {m: float(np.mean(v)) for m, v in all_costs.items()}


def build_setB_eval(split):
    queries, scores = [], []
    for ds in common.DATASETS:
        d = split["setB"][ds]
        queries.extend(d["queries"])
        scores.append(d["scores"])
    return queries, np.concatenate(scores, axis=0)


class LiteFoldDataset(TorchDataset):
    def __init__(self, train_rows, pool_names):
        self.name_to_idx = {n: i for i, n in enumerate(pool_names)}
        self.num_experts = len(pool_names)
        self.items = []
        for prompt, scores in train_rows:
            sub = {m: s for m, s in scores.items() if m in self.name_to_idx}
            if not sub or max(sub.values()) < 1.0:
                continue
            label = [0.0] * self.num_experts
            for m, s in sub.items():
                if s >= 1.0:
                    label[self.name_to_idx[m]] = 1.0
            self.items.append((prompt, label, None))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def load_descriptors_ordered(desc_dir, pool):
    return np.stack([np.load(desc_dir / f"{common.NAME_TO_SAFE[m]}.npy") for m in pool]), pool


def precompute_embeddings(texts, tokenizer, base_model, batch_size=64):
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


def train_fold(pool_19, desc_dir, train_rows, cost_dict, hidden_in_dim=384, seed=0, balance_beta=0.0,
               epochs=None):
    torch.manual_seed(seed)
    E, desc_names = load_descriptors_ordered(desc_dir, pool_19)
    E = torch.from_numpy(E).float().to(base.DEVICE)
    E = E / (E.norm(dim=1, keepdim=True) + 1e-9)
    proj_dim = E.size(1)

    ds = LiteFoldDataset(train_rows, desc_names)
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
            loss = base.cost_info_nce(q, E, label, cost_tensor, tau=base.TEMPERATURE)
            bal_loss = base.load_balance_loss(q, E) if balance_beta > 0 else None
            total_loss = loss + balance_beta * bal_loss if bal_loss is not None else loss
            total_loss.backward()
            opt.step()
            opt.zero_grad()
            ep_loss += loss.item()
        print(f"    epoch {ep+1}/{n_epochs} done, avg_loss={ep_loss/n_batches:.4f} elapsed={time.time()-t0:.1f}s",
              flush=True)

    return head, desc_names


def collapse_diagnostic(head, probe_texts, E20, all_names):
    with torch.no_grad():
        enc = base._TOKENIZER(probe_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        enc = {k: v.to(base.DEVICE) for k, v in enc.items()}
        out = base._BASE_MODEL(**enc)
        cls = base.mean_pool(out.last_hidden_state, enc["attention_mask"])
        q = head(cls)
        sims_to_E = q @ E20.T
        nearest = sims_to_E.argmax(dim=1)
        counts = torch.bincount(nearest, minlength=len(all_names))
    return dict(zip(all_names, counts.cpu().tolist()))


def evaluate_fold(head, held_out, pool_19, desc_dir, pool_20, setB_embeds, true_scores, cost_norm_20,
                   probe_texts):
    E19, names19 = load_descriptors_ordered(desc_dir, pool_19)
    held_out_vec = np.load(desc_dir / f"{common.NAME_TO_SAFE[held_out]}.npy")
    all_names = names19 + [held_out]
    E20 = np.stack(list(E19) + [held_out_vec])
    E20 = E20 / (np.linalg.norm(E20, axis=1, keepdims=True) + 1e-9)
    E20_t = torch.from_numpy(E20).float().to(base.DEVICE)

    col_idx = [pool_20.index(n) for n in all_names]
    scores_reordered = true_scores[:, col_idx]
    cost_reordered = cost_norm_20[col_idx]
    held_out_col = all_names.index(held_out)

    with torch.no_grad():
        q = head(torch.from_numpy(setB_embeds).float().to(base.DEVICE))
        sims = (q @ E20_t.T).cpu().numpy()

    n_oracle_is_M = 0
    n_match = 0
    n_router_correct = 0
    for i in range(scores_reordered.shape[0]):
        row = scores_reordered[i]
        correct_mask = row >= 1.0
        chosen_idx = int(np.argmax(sims[i]))
        if row[chosen_idx] >= 1.0:
            n_router_correct += 1
        if correct_mask.any():
            masked_cost = np.where(correct_mask, cost_reordered, np.inf)
            oracle_idx = int(np.argmin(masked_cost))
        else:
            oracle_idx = None
        if oracle_idx == held_out_col:
            n_oracle_is_M += 1
            if chosen_idx == held_out_col:
                n_match += 1

    from scipy.stats import pointbiserialr
    from sklearn.metrics import roc_auc_score
    ho_sims_all = sims[:, held_out_col]
    ho_labels_all = (scores_reordered[:, held_out_col] >= 1.0).astype(int)
    auc = pb_p = float("nan")
    if 0 < ho_labels_all.sum() < len(ho_labels_all):
        auc = float(roc_auc_score(ho_labels_all, ho_sims_all))
        _, pb_p = pointbiserialr(ho_labels_all, ho_sims_all)

    nearest_dist = collapse_diagnostic(head, probe_texts, E20_t, all_names)

    return {
        "held_out": held_out,
        "n_oracle_is_M": n_oracle_is_M,
        "oracle_match_rate": n_match / n_oracle_is_M if n_oracle_is_M else float("nan"),
        "router_overall_accuracy": n_router_correct / scores_reordered.shape[0],
        "auc_heldout_correctness": auc,
        "auc_p": float(pb_p),
        "n_setB_total": int(scores_reordered.shape[0]),
        "n_heldout_positive_in_setB": int(ho_labels_all.sum()),
        "collapse_nearest_dist": nearest_dist,
    }
