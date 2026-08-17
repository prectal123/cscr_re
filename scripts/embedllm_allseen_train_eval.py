"""Basic ALL-SEEN routing AUDC on EmbedLLM (no held-out/unseen models at
all) -- every one of the 112 models is both trained on (Set A labels) and
routable at eval time (Set B / test.csv). This is the standard "how good is
this as a router" measurement, as opposed to the "new LLMs" unseen-model
recovery protocol used throughout section 20/21 (which specifically holds
out 1/3 of the pool and never gives it gradient signal).

CSCR's paper reports 0.541 AUDC on this standard EmbedLLM routing setup
(distinct from the "new LLMs" protocol's 0.4848) -- this script's number is
the one to compare against that.

Uses the min-over-positives loss (embedllm_newllm_grpo_train_minpos.py),
the best-performing outlier-drag fix from section 21 (mean AUDC=0.513,
4/4 seeds beat CSCR on the unseen protocol), applied here to the full pool.
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
from router.bandit import BanditStats
from router.cost_models import compute_cost
from run_audc_eval import area_under_curve, interp_to_grid, build_cost_grid, paired_bootstrap_audc_cached

PCA5_DIR = Path("local_descriptors/embedllm-ceiling-pca5")
ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
HOLDOUT_FRAC = 0.15
SEEDS = [0, 1, 2]
LOG_EVERY = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSCR_PAPER_ALLSEEN = 0.541
LAM_LIST = np.logspace(-4, 2, 20)
K = 20
BANDIT_BETA = 0.000001
COST_GRID_POINTS = 20


class QueryTargetDataset(torch.utils.data.Dataset):
    def __init__(self, df, models):
        name_to_idx = {n: i for i, n in enumerate(models)}
        self.items = []
        for pid, grp in df.groupby("prompt_id", sort=False):
            text = grp["prompt"].iloc[0]
            labels = np.full(len(models), np.nan, dtype=np.float32)
            for m, v in zip(grp["model_name"], grp["label"]):
                if m in name_to_idx:
                    labels[name_to_idx[m]] = float(v)
            mask = ~np.isnan(labels)
            if mask.sum() < 2:
                continue
            vals = labels[mask]
            mean, std = vals.mean(), vals.std()
            if std < 1e-6:
                continue
            target = np.zeros(len(models), dtype=np.float32)
            target[mask] = (vals - mean) / (std + 1e-6)
            self.items.append((text, target, mask.astype(np.float32)))

    def __len__(self):
        return len(self.items)


def minpos_loss(cos_sim, target, mask):
    pos_mask = (target > 0) & (mask > 0.5)
    neg_mask = (target <= 0) & (mask > 0.5)
    sq_err = (cos_sim - target) ** 2
    pos_err = sq_err.masked_fill(~pos_mask, float("inf"))
    has_pos = pos_mask.any(dim=1)
    pos_min = pos_err.min(dim=1).values
    pos_min = torch.where(has_pos, pos_min, torch.zeros_like(pos_min))
    loss_pos = (pos_min * has_pos.float()).sum() / has_pos.float().sum().clamp(min=1)
    neg_err = (sq_err * neg_mask.float()).sum(dim=1)
    neg_count = neg_mask.float().sum(dim=1).clamp(min=1)
    loss_neg = (neg_err / neg_count).mean()
    return loss_pos + loss_neg


def evaluate_holdout(enc, texts, targets, masks, E_t, batch_size=64):
    enc.model.eval()
    embeds = np.zeros((len(texts), E_t.size(1)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            embeds[start:start + len(batch)] = enc.encode(batch)
    cos = embeds @ E_t.cpu().numpy().T
    rhos = []
    for i in range(len(texts)):
        m = masks[i].astype(bool)
        if m.sum() < 3:
            continue
        rho, _ = spearmanr(cos[i, m], targets[i, m])
        if not np.isnan(rho):
            rhos.append(rho)
    return np.array(rhos)


def knn_curve(sims, models, label_maps, costs, lam_list, k=K, bandit_beta=BANDIT_BETA):
    n_prompts, n_models = sims.shape
    order = np.argsort(-sims, axis=1)
    topk = order[:, :min(k, n_models)]
    out_costs, out_accs = [], []
    Y = np.zeros((len(lam_list), n_prompts), dtype=np.int32)
    for li, lam in enumerate(lam_list):
        bandit = BanditStats(bandit_lambda=float(lam), beta=bandit_beta)
        tot_cost, tot_acc = 0.0, 0.0
        for i in range(n_prompts):
            best_score, best_j = -np.inf, topk[i, 0]
            for j in topk[i]:
                m = models[j]
                bonus = bandit.get_bonus(m)
                score = bonus + sims[i, j] - lam * costs[j]
                if score > best_score:
                    best_score, best_j = score, j
            chosen = models[best_j]
            acc = 1.0 if label_maps[i].get(chosen, 0) == 1 else 0.0
            cost = float(costs[best_j])
            bandit.update(chosen, accuracy=acc, cost=cost)
            tot_cost += cost
            tot_acc += acc
            Y[li, i] = int(acc)
        out_costs.append(tot_cost / n_prompts)
        out_accs.append(tot_acc / n_prompts)
    return np.array(out_costs), np.array(out_accs), Y


def random_curve(models, label_maps, costs, lam_list, seed=0):
    import random as pyrandom
    rng = pyrandom.Random(seed)
    n_prompts = len(label_maps)
    out_costs, out_accs = [], []
    Y = np.zeros((len(lam_list), n_prompts), dtype=np.int32)
    for li, lam in enumerate(lam_list):
        weights = np.exp(-float(lam) * costs)
        probs = weights / weights.sum()
        tot_cost, tot_acc = 0.0, 0.0
        for i in range(n_prompts):
            j = rng.choices(range(len(models)), weights=probs, k=1)[0]
            chosen = models[j]
            acc = 1.0 if label_maps[i].get(chosen, 0) == 1 else 0.0
            cost = float(costs[j])
            tot_cost += cost
            tot_acc += acc
            Y[li, i] = int(acc)
        out_costs.append(tot_cost / n_prompts)
        out_accs.append(tot_acc / n_prompts)
    return np.array(out_costs), np.array(out_accs), Y


def audc_qnc_peak(costs, accs):
    order = np.argsort(costs)
    c, a = costs[order], accs[order]
    grid = build_cost_grid(c, N_grid=COST_GRID_POINTS)
    a_grid = interp_to_grid(c, a, grid)
    audc = np.trapz(a_grid, grid) / (grid[-1] - grid[0])
    peak_idx = np.argmax(a)
    return {"audc": float(audc), "qnc": float(c[peak_idx]), "peak": float(a[peak_idx])}


def run_all_seen_eval(models, ckpt_dir, label):
    enc = QueryEncoder.load(str(ckpt_dir), proj_dim=5)
    enc.to(DEVICE)
    enc.model.eval()

    dataset = load_embedllm("test", candidates=models)
    texts = [ex["prompt"] for ex in dataset]
    label_maps = [ex["label_map"] for ex in dataset]
    print(f"[{label}] {len(texts)} test prompts, {len(models)} candidate models", flush=True)

    embeds = np.zeros((len(texts), 5), dtype=np.float32)
    for start in range(0, len(texts), 64):
        batch = texts[start:start + 64]
        embeds[start:start + len(batch)] = enc.encode(batch)
    E = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in models]).astype(np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    sims = embeds @ E.T
    costs = np.array([compute_cost(m, 0, cost_type="n_params") for m in models], dtype=np.float32)

    t0 = time.time()
    knn_costs, knn_accs, knn_Y = knn_curve(sims, models, label_maps, costs, LAM_LIST)
    rand_costs, rand_accs, rand_Y = random_curve(models, label_maps, costs, LAM_LIST)
    print(f"[{label}] curve sweep done in {time.time()-t0:.1f}s", flush=True)

    knn_metrics = audc_qnc_peak(knn_costs, knn_accs)
    rand_metrics = audc_qnc_peak(rand_costs, rand_accs)
    mean_delta, (lo, hi), p = paired_bootstrap_audc_cached(knn_costs, knn_Y, rand_costs, rand_Y, B=2000, seed=0)
    print(f"[{label}] knn: AUDC={knn_metrics['audc']:.4f} QNC={knn_metrics['qnc']:.3f} Peak={knn_metrics['peak']:.4f}", flush=True)
    print(f"[{label}] random: AUDC={rand_metrics['audc']:.4f} Peak={rand_metrics['peak']:.4f}", flush=True)
    print(f"[{label}] bootstrap delta(knn-random)={mean_delta:+.4f} p={p:.4g}", flush=True)
    return {"knn": knn_metrics, "random": rand_metrics, "bootstrap_delta": float(mean_delta), "bootstrap_p": float(p)}


def train_one_seed(seed, models, df, tokenizer, base_model, E_t):
    torch.manual_seed(seed)
    ds_all = QueryTargetDataset(df, models)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ds_all))
    n_holdout = int(len(ds_all) * HOLDOUT_FRAC)
    holdout_idx, train_idx = perm[:n_holdout], perm[n_holdout:]
    train_items = [ds_all.items[i] for i in train_idx]
    holdout_texts = [ds_all.items[i][0] for i in holdout_idx]
    holdout_targets = np.stack([ds_all.items[i][1] for i in holdout_idx])
    holdout_masks = np.stack([ds_all.items[i][2] for i in holdout_idx])
    print(f"  [seed {seed}] {len(ds_all)} queries, train={len(train_items)} holdout={len(holdout_texts)}", flush=True)

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
        for bi, (tok, target, mask) in enumerate(loader):
            tok = {k: v.to(DEVICE) for k, v in tok.items()}
            target, mask = target.to(DEVICE), mask.to(DEVICE)
            with torch.no_grad():
                out = base_model(**tok)
                cls_vec = out.last_hidden_state[:, 0]
            q = enc._project(cls_vec)
            cos_sim = q @ E_t.T
            loss = minpos_loss(cos_sim, target, mask)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if (bi + 1) % LOG_EVERY == 0:
                print(f"    [seed {seed}] epoch {ep+1} batch {bi+1} loss={loss.item():.4f} "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)

        rho_arr = evaluate_holdout(enc, holdout_texts, holdout_targets, holdout_masks, E_t)
        if rho_arr.mean() > best_rho:
            best_rho, best_epoch = rho_arr.mean(), ep + 1
            best_state = {k: v.clone().cpu() for k, v in enc.proj.state_dict().items()}
    print(f"  [seed {seed}] best_epoch={best_epoch} best_holdout_rho={best_rho:.4f} time={time.time()-t0:.1f}s", flush=True)

    enc.proj.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return enc, best_epoch, best_rho


def main():
    models = sorted(p.stem for p in PCA5_DIR.glob("*.npy"))
    print(f"All-seen pool: {len(models)} models", flush=True)

    print("Loading EmbedLLM train.csv...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "label", "prompt"])

    print(f"Loading frozen MiniLM base ONCE (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    E = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in models])
    E_t = torch.from_numpy(E).float().to(DEVICE)
    E_t = E_t / (E_t.norm(dim=1, keepdim=True) + 1e-9)

    results = []
    for seed in SEEDS:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)
        enc, best_epoch, best_rho = train_one_seed(seed, models, df, tokenizer, base_model, E_t)
        ckpt_dir = Path(f"local_checkpoints/embedllm-allseen-minpos-seed{seed}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        enc.save(ckpt_dir)

        r = run_all_seen_eval(models, ckpt_dir, label=f"allseen-minpos-seed{seed}")
        r["seed"] = seed
        r["best_epoch"] = best_epoch
        r["best_holdout_rho"] = float(best_rho)
        results.append(r)
        print(f"  [seed={seed}] RESULT: AUDC={r['knn']['audc']:.4f} Peak={r['knn']['peak']:.4f} "
              f"({'BEATS' if r['knn']['audc'] > CSCR_PAPER_ALLSEEN else 'below'} CSCR all-seen {CSCR_PAPER_ALLSEEN})",
              flush=True)

    out_path = ANALYSIS_DIR / "allseen_minpos_multiseed_results.json"
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n" + "=" * 80)
    print(f"ALL-SEEN ROUTING AUDC SUMMARY (min-pos, seeds {SEEDS}) vs CSCR {CSCR_PAPER_ALLSEEN}")
    print("=" * 80)
    audcs = [r["knn"]["audc"] for r in results]
    for r in results:
        print(f"seed={r['seed']}: AUDC={r['knn']['audc']:.4f} Peak={r['knn']['peak']:.4f} "
              f"({'BEATS' if r['knn']['audc'] > CSCR_PAPER_ALLSEEN else 'below'})")
    print(f"\nmean AUDC={np.mean(audcs):.4f} (std={np.std(audcs):.4f})")
    print(f"beats CSCR all-seen ({CSCR_PAPER_ALLSEEN}): {sum(1 for a in audcs if a > CSCR_PAPER_ALLSEEN)}/{len(audcs)}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
