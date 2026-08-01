"""Leave-one-out unseen-model routing experiment.

For each FP type (Perplexity, Ceiling/BartFP) and each of the 11
MixInstruct models M:
  1. Train a query encoder (frozen MiniLM + small trainable MLP,
     cost_spectrum_info_nce loss, matching the paper's baseline) on the
     OTHER 10 models only -- M's descriptor and M's labels never appear
     during training (MixInstructOracle's pool filtering handles this).
  2. Add M's descriptor (post-hoc, not used in training) to the 11-way
     candidate set.
  3. On Set B (held-out eval prompts, disjoint from anything used to
     build the Ceiling FP), compare the trained system's routing choice
     against the Oracle (true best bartscore among the 11 candidates)
     for every prompt where M IS the Oracle's true pick.
     -> primary metric: Oracle-match rate specifically on M's prompts
     -> secondary metric: average bartscore regret (chosen vs true best)

MiniLM's frozen CLS embeddings for Set B are computed ONCE and reused
across all folds/FP-types; only the small trained projection differs
per fold, which is cheap to reapply.
"""
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, "src")
from router.mix_instruct import MixInstructOracle
from router.utils import load_descriptors, load_cost_dict
from torch.utils.data import Dataset as TorchDataset

POOL_PATH = "experts/pool-mix-instruct-11.json"
CEILING_DIR = Path("local_descriptors/mix-instruct-capability-ceiling")
PERP_DIR = Path("local_descriptors/mix-instruct-perplexity")
V12_DIR = Path("local_descriptors/mix-instruct-v1.2")
OUT_DIR = Path("local_descriptors/analysis")
SCORE_KEY = "bartscore"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
N_EVAL_PROMPTS = 5000
# scripts/train_query_encoder.py argparse defaults (paper's own reference training script,
# byte-identical to upstream/main): epochs=2, batch_size=64, lr=5e-4, temperature=0.05.
# No train-set subsampling there either -- it trains on the full ~100k-row split each epoch.
# We match all of this now that local GPU makes the full run fast.
EPOCHS = 2
BATCH_SIZE = 64
LR = 5e-4
TEMPERATURE = 0.05
SEED = 0
LOG_EVERY = 200  # print a heartbeat every N batches so slow-vs-stuck is visible
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MixInstructOracle.MARGIN (0.1, from the paper authors' own repo, byte-identical --
# verified via `git diff upstream/main -- src/router/mix_instruct.py`) makes 82% of rows
# label ALL 10 candidates as positive (avg 9.04/10), leaving almost no contrastive signal.
# The paper's text does not specify this threshold -- it's an implementation detail of their
# reference code, not a stated hyperparameter -- so we recalibrate it here for OUR training
# rather than deviating from anything the paper actually claims. Chosen empirically from the
# score-gap distribution (best - 2nd best) on this pool: median gap 0.0075, p75 0.019.
# MARGIN=0.01 -> avg 2.92/10 positives/row, all-positive rows down from 82% to 5.6%.
TRAIN_MARGIN = 0.01

NAME_TO_HF = {
    "vicuna-13b-1.1": "eachadea__vicuna-13b-1.1",
    "alpaca-native": "chavinlo__alpaca-native",
    "dolly-v2-12b": "databricks__dolly-v2-12b",
    "stablelm-tuned-alpha-7b": "stabilityai__stablelm-tuned-alpha-7b",
    "oasst-sft-4-pythia-12b-epoch-3.5": "OpenAssistant__oasst-sft-4-pythia-12b-epoch-3.5",
    "koala-7B-HF": "TheBloke__koala-7B-HF",
    "llama-7b-hf-baize-lora-bf16": "mosesjun0h__llama-7b-hf-baize-lora-bf16",
    "flan-t5-xxl": "google__flan-t5-xxl",
    "chatglm-6b": "THUDM__chatglm-6b",
    "moss-moon-003-sft": "fnlp__moss-moon-003-sft",
    "mpt-7b-instruct": "mosaicml__mpt-7b-instruct",
    "mpt-7b": "mosaicml__mpt-7b-instruct",
}


# ---------- Step 1: build Set B eval data (text + per-model bartscore) ----------
def load_set_b(pool):
    split_info = json.load(open(CEILING_DIR / "split_info.json"))
    set_b_ids = set(split_info["set_b_eval_reserved"])
    pool_set = set(pool)

    raw = concatenate_datasets([
        load_dataset("llm-blender/mix-instruct", split="train"),
        load_dataset("llm-blender/mix-instruct", split="validation"),
    ])

    per_prompt = {}
    for rec in raw:
        pid = rec["id"]
        if pid not in set_b_ids:
            continue
        scores = {}
        for cand in rec["candidates"]:
            hf_name = NAME_TO_HF.get(cand["model"])
            if hf_name is None or hf_name not in pool_set:
                continue
            sc = cand["scores"].get(SCORE_KEY)
            if sc is not None:
                scores[hf_name] = sc
        if len(scores) == len(pool):
            per_prompt[pid] = {
                "text": f"{rec['instruction']} {rec['input']}",
                "scores": scores,
            }

    ids = sorted(per_prompt.keys())
    rng = random.Random(SEED)
    rng.shuffle(ids)
    ids = ids[:N_EVAL_PROMPTS]
    print(f"Set B eval subsample: {len(ids)} prompts (dense over {len(pool)} models)")
    return ids, per_prompt


# ---------- Step 1b: parse the MixInstruct 'train' split ONCE, reused by every fold ----------
def build_train_index(pool_11):
    """Parse the raw HF 'train' split exactly once into per-row {model: exp(bartscore)}
    dicts, matching MixInstructOracle's own name-canon + max-per-model-dedup + exp-transform
    logic. Re-instantiating MixInstructOracle (which calls load_dataset internally) once per
    fold was the source of a hang -- a second load_dataset call on the same cached Arrow file
    while load_set_b()'s dataset object is still alive appears to block on Windows. Loading the
    split once and re-filtering in memory per fold sidesteps this entirely and is much faster.
    """
    name_to_hf = MixInstructOracle.NAME_TO_HF
    pool_set = set(pool_11)
    raw = load_dataset("llm-blender/mix-instruct", split="train")
    rows = []
    skipped = 0
    for rec in raw:
        prompt = MixInstructOracle._build_prompt(rec["instruction"], rec["input"])
        scores = {}
        for cand in rec["candidates"]:
            hf_name = name_to_hf.get(cand["model"])
            if hf_name is None or hf_name not in pool_set:
                continue
            sc = cand["scores"].get(SCORE_KEY)
            if sc is None:
                continue
            sc = math.exp(sc) if SCORE_KEY == "bartscore" else sc
            if hf_name not in scores or sc > scores[hf_name]:
                scores[hf_name] = sc
        if not scores:
            skipped += 1
            continue
        rows.append((prompt, scores))
    print(f"Train index: {len(rows)} rows ({skipped} skipped, no matching experts)")
    return rows


class FoldDataset(TorchDataset):
    """In-memory, per-fold relabeling of the precomputed train index -- no I/O, no
    load_dataset call. Mirrors MixInstructOracle's multi-hot labeling scheme, but with
    TRAIN_MARGIN (calibrated for this pool) instead of the reference repo's MARGIN=0.1,
    which produces near-degenerate (82% all-positive) labels here."""

    MARGIN = TRAIN_MARGIN

    def __init__(self, train_rows, pool_10):
        self.name_to_idx = {n: i for i, n in enumerate(pool_10)}
        self.num_experts = len(pool_10)
        self.items = []
        for prompt, scores in train_rows:
            sub = {m: sc for m, sc in scores.items() if m in self.name_to_idx}
            if not sub:
                continue
            best_val = max(sub.values())
            label = [0.0] * self.num_experts
            for m, sc in sub.items():
                if best_val - sc <= self.MARGIN:
                    label[self.name_to_idx[m]] = 1.0
            self.items.append((prompt, label, None))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# all-MiniLM-L6-v2 is trained/intended for MEAN pooling over tokens (attention-mask
# weighted), not CLS-token pooling. Using [:, 0] (what both our earlier version and the
# paper's own QueryEncoder.encode() do) gives severely anisotropic embeddings -- pairwise
# cosine sim among 200 distinct prompts averaged 0.62 with CLS pooling vs 0.05 with mean
# pooling. That anisotropy was the root cause of the query encoder collapsing to a
# prompt-agnostic constant output regardless of loss/descriptor/label fixes.
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


# ---------- Step 2: precompute frozen MiniLM mean-pooled embeddings, once ----------
def precompute_embeddings(ids, per_prompt, tokenizer, base_model, batch_size=64):
    embeds = np.zeros((len(ids), base_model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size]
            texts = [per_prompt[i]["text"] for i in batch_ids]
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out = base_model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
            embeds[start:start + len(batch_ids)] = pooled.cpu().numpy()
    return embeds


# ---------- Step 3: per-fold training (small MLP head only) ----------
class ProjHead(nn.Module):
    """Mirrors QueryEncoder.proj (src/router/query_encoder.py:26-30) exactly:
    Linear(bias=False) -> ReLU -> Linear(bias=False), proj_multiplier=1."""

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


def cost_info_nce(q, E, label, cost_norm, tau, lam=0.5):
    sim = (q @ E.T) / tau
    pos_mask = label.bool()
    keep = pos_mask.any(dim=1)
    if keep.sum() == 0:
        return torch.tensor(0.0, device=q.device, requires_grad=True)
    sim, pos_mask = sim[keep], pos_mask[keep]
    w_pos = (1.0 - cost_norm).unsqueeze(0) * pos_mask
    w_pos = w_pos / (w_pos.sum(dim=1, keepdim=True) + 1e-9)
    numer = (torch.exp(sim) * w_pos).sum(dim=1)
    denom = torch.exp(sim - lam * cost_norm).sum(dim=1)
    return -(numer / denom).clamp(min=1e-9).log().mean()


def cost_info_nce_cheapest(q, E, label, cost_norm, tau, lam=0.5):
    """Variant of cost_info_nce that targets only the SINGLE cheapest positive
    per row (matching this codebase's "cheapest-correct" oracle convention)
    instead of a cost-weighted AVERAGE over all positives. Motivated by a
    mentor's diagnosis (2026-07-31): weighted-averaging over multiple positive
    labels rewards the query for being simultaneously close to several
    experts at once, which pulls toward whichever direction jointly satisfies
    many prompts' positive sets -- i.e. a "generalist" attractor, favoring
    collapse (PROGRESS.md section 16.2). Picking one fixed (cost-determined,
    not similarity-determined) target per row removes that multi-positive
    averaging pressure without introducing a self-reinforcing feedback loop
    (unlike picking the positive the model ALREADY prefers, which would make
    collapse worse, not better)."""
    sim = (q @ E.T) / tau
    pos_mask = label.bool()
    keep = pos_mask.any(dim=1)
    if keep.sum() == 0:
        return torch.tensor(0.0, device=q.device, requires_grad=True)
    sim, pos_mask = sim[keep], pos_mask[keep]
    cost_row = cost_norm.unsqueeze(0).expand_as(pos_mask)
    masked_cost = torch.where(pos_mask, cost_row, torch.full_like(cost_row, float("inf")))
    target_idx = masked_cost.argmin(dim=1)
    numer = torch.exp(sim.gather(1, target_idx.unsqueeze(1)).squeeze(1))
    denom = torch.exp(sim - lam * cost_norm).sum(dim=1)
    return -(numer / denom).clamp(min=1e-9).log().mean()


def cost_spectrum_info_nce(z_q, E, label, cost_norm, tau=0.07, n_bands=5, alpha=0.25, tau_min=0.05, gamma=0.2):
    """Ported verbatim from scripts/train_query_encoder.py (paper's own reference
    code, byte-identical to upstream/main) -- this is Eq. 8 (l^CS) in the paper,
    the band-based loss, as opposed to cost_info_nce (no bands) which is what
    the rest of this file otherwise uses. `tau` is accepted for call-site
    consistency with cost_info_nce but is unused in the paper's own code too --
    band-specific temperature tau_b (from tau_min/alpha) is what's actually used."""
    device = z_q.device
    B, M = label.shape

    percentiles = torch.linspace(0, 1, n_bands + 1, device=device)
    cost_bins = torch.quantile(cost_norm.view(-1), percentiles)
    band_idx = torch.bucketize(cost_norm, cost_bins[1:-1])

    sim = (z_q @ E.T)

    loss_accum, band_cnt = 0.0, 0
    for k in range(n_bands):
        b_mask = (band_idx == k)
        if b_mask.sum() == 0:
            continue

        pos_mask = label.clone()
        pos_mask[:, ~b_mask] = 0

        any_pos = pos_mask.any(1)
        if any_pos.sum() == 0:
            continue
        sim_k = sim[any_pos]
        pos_k = pos_mask[any_pos]

        tau_b = tau_min + alpha * cost_norm[b_mask].mean()

        exp_pos = torch.exp(sim_k / tau_b)
        numer = (exp_pos * pos_k).sum(1)

        cost_pen = gamma * cost_norm.unsqueeze(0)
        logits_k = (sim_k - cost_pen) / tau_b
        denom = torch.exp(logits_k).sum(1)

        loss_accum += -(numer / (denom + 1e-9)).log().mean()
        band_cnt += 1

    if band_cnt == 0:
        return torch.tensor(0., device=device, requires_grad=True)

    return loss_accum / band_cnt


def load_balance_loss(q, E, tau=None):
    """Switch Transformer / GShard-style load-balancing auxiliary loss (Shazeer
    et al. 2017's original fix for MoE "expert collapse"), adapted to this
    setting: penalizes the batch-average routing probability mass deviating
    from uniform across the M candidate experts. Minimized (=1) when every
    expert gets equal average probability (1/M each); grows toward M when
    routing concentrates on a single expert -- exactly the failure mode this
    whole investigation found (PROGRESS.md section 15). No cost/label info
    needed -- purely a diversity-of-usage regularizer, added on top of
    whichever primary loss (cost_info_nce etc.) is already in use.

    IMPORTANT: must use the SAME temperature as the primary loss/inference-time
    argmax (tau=0.05 by default here, matching TEMPERATURE). Without dividing
    by tau, raw cosine similarities (~[-1,1]) produce an almost-uniform softmax
    regardless of how sharply argmax actually concentrates -- this was tried
    first and gave a near-zero, ineffective gradient (bal_loss stuck at ~1.0)
    even while nearest_dist stayed fully collapsed.
    """
    if tau is None:
        tau = TEMPERATURE
    M = E.size(0)
    probs = torch.softmax((q @ E.T) / tau, dim=1)  # (B, M)
    P = probs.mean(dim=0)                           # (M,) batch-average routing mass per expert
    return M * (P ** 2).sum()


def train_fold(pool_10, desc_dir, train_rows, hidden_in_dim=384, seed=0):
    torch.manual_seed(seed)
    E, desc_names = load_descriptors(str(desc_dir), pool=pool_10)
    E = torch.from_numpy(np.stack(E)).float().to(DEVICE)
    E = E / (E.norm(dim=1, keepdim=True) + 1e-9)
    proj_dim = E.size(1)

    ds = FoldDataset(train_rows, desc_names)
    cost = load_cost_dict(desc_names, cost_type="n_params")
    cost_tensor = torch.tensor([cost[n] for n in desc_names], dtype=torch.float32).to(DEVICE)
    cost_tensor = (cost_tensor - cost_tensor.min()) / (cost_tensor.max() - cost_tensor.min() + 1e-9)

    global _TOKENIZER, _BASE_MODEL
    head = ProjHead(hidden_in_dim, proj_dim).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=LR)

    def collate(batch):
        texts, idxs, _ = zip(*batch)
        toks = _TOKENIZER(list(texts), padding=True, truncation=True, return_tensors="pt", max_length=256)
        return toks, torch.tensor(idxs, dtype=torch.long)

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    n_batches = len(loader)
    print(f"    train set: {len(ds)} rows, {n_batches} batches/epoch x {EPOCHS} epochs", flush=True)

    head.train()
    t0 = time.time()
    for ep in range(EPOCHS):
        ep_loss = 0.0
        for bi, (tok, label) in enumerate(loader):
            tok = {k: v.to(DEVICE) for k, v in tok.items()}
            label = label.to(DEVICE)
            with torch.no_grad():
                out = _BASE_MODEL(**tok)
                cls = mean_pool(out.last_hidden_state, tok["attention_mask"])
            q = head(cls)
            loss = cost_info_nce(q, E, label, cost_tensor, tau=TEMPERATURE)
            loss.backward()
            opt.step()
            opt.zero_grad()
            ep_loss += loss.item()
            if (bi + 1) % LOG_EVERY == 0 or (bi + 1) == n_batches:
                elapsed = time.time() - t0
                print(f"    epoch {ep+1}/{EPOCHS} batch {bi+1}/{n_batches} "
                      f"loss={loss.item():.4f} elapsed={elapsed:.1f}s", flush=True)
        print(f"    epoch {ep+1}/{EPOCHS} done, avg_loss={ep_loss / n_batches:.4f}", flush=True)

    return head, desc_names


# ---------- Step 4: evaluate a fold against Oracle on Set B ----------
def evaluate_fold(head, held_out, pool_10, desc_dir, pool_11, cls_embeds, ids, per_prompt):
    E10, names10 = load_descriptors(str(desc_dir), pool=pool_10)
    held_out_vec = np.load(desc_dir / f"{held_out}.npy")
    all_names = names10 + [held_out]
    E11 = np.stack(list(E10) + [held_out_vec])
    E11 = E11 / (np.linalg.norm(E11, axis=1, keepdims=True) + 1e-9)
    E11_t = torch.from_numpy(E11).float().to(DEVICE)

    with torch.no_grad():
        q = head(torch.from_numpy(cls_embeds).float().to(DEVICE))  # (n_eval, proj_dim)
        sims = q @ E11_t.T  # (n_eval, 11)
        chosen_idx = sims.argmax(dim=1).cpu().numpy()

    n_oracle_is_M = 0
    n_match = 0
    regrets = []
    for i, pid in enumerate(ids):
        scores = per_prompt[pid]["scores"]  # dict over pool_11
        oracle_model = max(pool_11, key=lambda m: scores[m])
        chosen_model = all_names[chosen_idx[i]]
        best_score = math.exp(scores[oracle_model])
        chosen_score = math.exp(scores[chosen_model])
        regrets.append(best_score - chosen_score)
        if oracle_model == held_out:
            n_oracle_is_M += 1
            if chosen_model == held_out:
                n_match += 1

    oracle_match_rate = n_match / n_oracle_is_M if n_oracle_is_M else float("nan")
    return {
        "held_out": held_out,
        "n_oracle_is_M": n_oracle_is_M,
        "oracle_match_rate": oracle_match_rate,
        "avg_regret": float(np.mean(regrets)),
    }


def main():
    global _TOKENIZER, _BASE_MODEL
    pool_11 = json.load(open(POOL_PATH))

    print(f"Loading frozen MiniLM base (device={DEVICE})...")
    _TOKENIZER = AutoTokenizer.from_pretrained(EMBED_MODEL)
    _BASE_MODEL = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    _BASE_MODEL.eval()

    print("Building Set B eval subsample...")
    ids, per_prompt = load_set_b(pool_11)

    print("Precomputing frozen CLS embeddings for Set B (once, reused across all folds)...")
    cls_embeds = precompute_embeddings(ids, per_prompt, _TOKENIZER, _BASE_MODEL)
    print(f"cls_embeds shape: {cls_embeds.shape}\n")

    print("Parsing MixInstruct 'train' split once (reused, relabeled per fold in memory)...")
    train_rows = build_train_index(pool_11)

    all_results = {}
    for fp_name, desc_dir in [("Perplexity", PERP_DIR), ("Ceiling", CEILING_DIR), ("V1.2", V12_DIR)]:
        print(f"\n{'='*60}\nFP type: {fp_name}\n{'='*60}")
        fold_results = []
        for held_out in pool_11:
            pool_10 = [m for m in pool_11 if m != held_out]
            print(f"[{fp_name}] held out: {held_out} ...")
            head, names10 = train_fold(pool_10, desc_dir, train_rows)
            r = evaluate_fold(head, held_out, pool_10, desc_dir, pool_11, cls_embeds, ids, per_prompt)
            print(f"  n_oracle_is_M={r['n_oracle_is_M']}  "
                  f"oracle_match_rate={r['oracle_match_rate']:.4f}  "
                  f"avg_regret={r['avg_regret']:.4f}")
            fold_results.append(r)
        all_results[fp_name] = fold_results

        rates = [r["oracle_match_rate"] for r in fold_results if not math.isnan(r["oracle_match_rate"])]
        chance = 1.0 / len(pool_11)
        print(f"\n--- {fp_name} summary ---")
        print(f"mean oracle_match_rate across {len(rates)} folds: {np.mean(rates):.4f} "
              f"(chance level ~ {chance:.4f})")
        print(f"mean avg_regret across folds: {np.mean([r['avg_regret'] for r in fold_results]):.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "loo_unseen_recovery_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {OUT_DIR / 'loo_unseen_recovery_results.json'}")


if __name__ == "__main__":
    main()
