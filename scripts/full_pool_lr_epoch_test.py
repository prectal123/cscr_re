"""Mentor idea #2 (2026-08-01): does a much lower learning rate + more
epochs (slower, more careful convergence) reduce collapse, compared to the
default LR=5e-4/2-epoch schedule that may lock in a collapse attractor early?

Full 11-model pool (NOT leave-one-out), Perplexity FP, beta=0 (no
load-balancing -- isolating the LR/epoch effect on its own, not mixed with
the balancing auxiliary loss). Single seed. Compares against the already-
known beta=0 baseline from the earlier beta sweep (PROGRESS.md 16.3):
n_nonzero=3/11, top3_share=97%, router_acc(lambda=0)=67%.
"""
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import routerbench_loo_recovery as loo
import loo_unseen_recovery as base
from transformers import AutoModel, AutoTokenizer

SEED = 0
LR = 5e-5      # 10x lower than default 5e-4
EPOCHS = 10    # 5x more than default 2
N_COLLAPSE_PROBES = 200


def main():
    print(f"DEVICE: {base.DEVICE}  LR={LR}  EPOCHS={EPOCHS} (default: LR={base.LR} EPOCHS={base.EPOCHS})", flush=True)
    pool_11 = loo.rb.NAMES

    base._TOKENIZER = AutoTokenizer.from_pretrained(base.EMBED_MODEL)
    base._BASE_MODEL = AutoModel.from_pretrained(base.EMBED_MODEL).to(base.DEVICE)
    base._BASE_MODEL.eval()

    set_a, set_b = loo.rb.load_data()
    set_b_texts = set_b["prompt"].tolist()
    cls_embeds = loo.precompute_set_b_embeddings(set_b_texts, base._TOKENIZER, base._BASE_MODEL)
    true_scores = loo.np.stack([set_b[m].to_numpy(dtype=float) for m in loo.rb.MODELS], axis=1)

    train_rows = loo.build_train_rows(set_a)
    cost_dict = loo.build_cost_dict(set_a)

    torch.manual_seed(SEED)
    E, desc_names = loo.load_descriptors_ordered(loo.PERP_DIR, pool_11)
    E = torch.from_numpy(E).float().to(base.DEVICE)
    E = E / (E.norm(dim=1, keepdim=True) + 1e-9)
    proj_dim = E.size(1)

    ds = loo.RBFoldDataset(train_rows, desc_names)
    cost_tensor = torch.tensor([cost_dict[n] for n in desc_names], dtype=torch.float32).to(base.DEVICE)
    cost_tensor = (cost_tensor - cost_tensor.min()) / (cost_tensor.max() - cost_tensor.min() + 1e-9)

    head = base.ProjHead(384, proj_dim).to(base.DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=LR)

    def collate(batch):
        texts, idxs, _ = zip(*batch)
        toks = base._TOKENIZER(list(texts), padding=True, truncation=True, return_tensors="pt", max_length=256)
        return toks, torch.tensor(idxs, dtype=torch.long)

    loader = DataLoader(ds, batch_size=base.BATCH_SIZE, shuffle=True, collate_fn=collate)
    n_batches = len(loader)
    print(f"train set: {len(ds)} rows, {n_batches} batches/epoch x {EPOCHS} epochs", flush=True)

    rng = random.Random(1)
    probe_idx = rng.sample(range(len(set_b_texts)), N_COLLAPSE_PROBES)
    probe_texts = [set_b_texts[i] for i in probe_idx]

    head.train()
    t0 = time.time()
    for ep in range(EPOCHS):
        ep_loss = 0.0
        for bi, (tok, label) in enumerate(loader):
            tok = {k: v.to(base.DEVICE) for k, v in tok.items()}
            label = label.to(base.DEVICE)
            with torch.no_grad():
                out = base._BASE_MODEL(**tok)
                cls = base.mean_pool(out.last_hidden_state, tok["attention_mask"])
            q = head(cls)
            loss = base.cost_info_nce(q, E, label, cost_tensor, tau=base.TEMPERATURE)
            loss.backward()
            opt.step()
            opt.zero_grad()
            ep_loss += loss.item()
            if (bi + 1) % 200 == 0 or (bi + 1) == n_batches:
                print(f"    epoch {ep+1}/{EPOCHS} batch {bi+1}/{n_batches} loss={loss.item():.4f} "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)
        # collapse + accuracy check after every epoch, so we can see the trajectory
        off_mean, off_std, nearest_dist = loo.collapse_diagnostic(head, probe_texts, E, desc_names)
        n_nonzero = sum(1 for v in nearest_dist.values() if v > 0)
        top3 = sorted(nearest_dist.values(), reverse=True)[:3]
        top3_share = sum(top3) / N_COLLAPSE_PROBES
        with torch.no_grad():
            q_b = head(torch.from_numpy(cls_embeds).float().to(base.DEVICE))
            sims_b = q_b @ E.T
            chosen = sims_b.argmax(dim=1).cpu().numpy()
        router_acc = np.mean([true_scores[i, chosen[i]] >= 1.0 for i in range(len(chosen))])
        print(f"  epoch {ep+1}/{EPOCHS} avg_loss={ep_loss/n_batches:.4f} elapsed={time.time()-t0:.1f}s  "
              f"n_nonzero={n_nonzero}/11  top3_share={top3_share:.3f}  router_acc={router_acc:.4f}", flush=True)
        print(f"    nearest_dist: {nearest_dist}", flush=True)

    print("\nDone. Compare to known beta=0/default-LR baseline: n_nonzero=3/11, top3_share=0.97, router_acc=0.67",
          flush=True)


if __name__ == "__main__":
    main()
