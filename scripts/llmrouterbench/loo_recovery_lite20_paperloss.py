"""Same as loo_recovery_lite20.py, EXCEPT train_fold uses the literal CSCR
paper loss (cost_spectrum_info_nce, Eq.8, band-based, n_bands=5) instead of
the simplified cost_info_nce (no bands, with an added cost-weighted positive
averaging term that isn't actually in the paper's own code -- see PROGRESS.md
17.19). Everything else (data loading, evaluate_fold, collapse_diagnostic,
architecture) is unchanged and re-exported from loo_recovery_lite20 so this
stays a controlled, single-variable swap.
"""
import time

import torch
from torch.utils.data import DataLoader

import common_lite20 as common
import loo_unseen_recovery as base
from loo_recovery_lite20 import (  # noqa: F401 -- re-exported for callers
    DATA_DIR, CEILING_DIR, PERP_DIR, N_COLLAPSE_PROBES,
    load_split, build_train_rows, build_cost_dict, build_setB_eval,
    LiteFoldDataset, load_descriptors_ordered, precompute_embeddings,
    collapse_diagnostic, evaluate_fold,
)

N_BANDS = 5


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
    print(f"    [paper-loss, n_bands={N_BANDS}] train set: {len(ds)} rows, {n_batches} batches/epoch x "
          f"{n_epochs} epochs", flush=True)

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
            loss = base.cost_spectrum_info_nce(q, E, label, cost_tensor, n_bands=N_BANDS)
            bal_loss = base.load_balance_loss(q, E) if balance_beta > 0 else None
            total_loss = loss + balance_beta * bal_loss if bal_loss is not None else loss
            total_loss.backward()
            opt.step()
            opt.zero_grad()
            ep_loss += loss.item()
        print(f"    epoch {ep+1}/{n_epochs} done, avg_loss={ep_loss/n_batches:.4f} elapsed={time.time()-t0:.1f}s",
              flush=True)

    return head, desc_names
