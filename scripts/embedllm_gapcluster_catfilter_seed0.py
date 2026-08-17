"""Small trial: a 4th catfilter cutoff rule -- "largest-gap clustering with
a significance floor" -- combining the two ideas from this session's design
discussion:

  1. Instead of a fixed count (top2) or fixed fraction (top50pct) or fixed
     absolute margin (0.15), find the LARGEST CONSECUTIVE GAP in the sorted
     category-track-record scores of a query's positive candidates, and cut
     there. This is a 1D version of Jenks natural-breaks / 2-cluster
     1D k-means -- the data itself decides where the "real" break is,
     instead of a hand-picked count/fraction.
  2. Guard against over-splitting on trivial gaps (e.g. [100,99] shouldn't
     be split just because there's *a* gap): only split if the largest gap
     is at least `K_SIGMA` standard deviations of that CATEGORY's overall
     accuracy spread (computed once per category, over ALL models -- a
     more stable noise-scale estimate than using only the 2-5 positives of
     a single query). Below that floor, treat the whole positive set as one
     trustworthy cluster (keep everyone).

Single seed (0), all-seen only, standalone (not combined with min-pos) --
a quick trial to see where this lands relative to the 3 already-tested
cutoff rules (top2=0.5430, top50pct=0.5432, margin=0.5381 at seed0, see
allseen_catfilter_methodology_multiseed_results.json) before deciding
whether it's worth a full multi-seed run.
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
from run_audc_eval import interp_to_grid, build_cost_grid, paired_bootstrap_audc_cached

PCA5_DIR = Path("local_descriptors/embedllm-ceiling-pca5")
ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
HOLDOUT_FRAC = 0.15
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSCR_ALLSEEN = 0.541
EXCLUDE = {"JaeyeonKang__CCK_Asura_v1"}
LAM_LIST = np.logspace(-4, 2, 20)
K = 20
BANDIT_BETA = 0.000001
COST_GRID_POINTS = 20
K_SIGMA = 1.0  # significance floor: largest gap must be >= K_SIGMA * category's own accuracy std


def compute_keep_idx_gapcluster(pos_idx: np.ndarray, scores: np.ndarray, cat_std: float, k_sigma: float = K_SIGMA) -> np.ndarray:
    """Largest-gap 1D clustering with a noise floor. Returns the subset of
    pos_idx to KEEP (the top cluster, or everyone if no gap clears the floor)."""
    if len(pos_idx) <= 1:
        return pos_idx
    order = np.argsort(-scores)
    sorted_idx = pos_idx[order]
    sorted_scores = scores[order]
    gaps = sorted_scores[:-1] - sorted_scores[1:]
    gap_pos = int(np.argmax(gaps))
    gap_max = gaps[gap_pos]
    threshold = k_sigma * cat_std
    if gap_max >= threshold:
        return sorted_idx[:gap_pos + 1]
    return sorted_idx


def build_raw_category_accuracy(df, models, categories):
    pivot = df.pivot_table(index="model_name", columns="category", values="label", aggfunc="mean")
    pivot = pivot.reindex(index=models, columns=categories)
    return pivot.to_numpy()


def build_items(df, models, raw_cat_acc, category_to_idx, cat_std_by_idx):
    name_to_idx = {n: i for i, n in enumerate(models)}
    items = []
    n_split, n_nosplit = 0, 0
    for pid, grp in df.groupby("prompt_id", sort=False):
        text = grp["prompt"].iloc[0]
        category = grp["category"].iloc[0]
        cat_idx = category_to_idx.get(category)
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

        keep_mask = mask.copy()
        pos_idx = np.where(mask & (labels == 1))[0]
        if cat_idx is not None and len(pos_idx) > 1:
            scores = np.nan_to_num(raw_cat_acc[pos_idx, cat_idx], nan=-1.0)
            cat_std = cat_std_by_idx[cat_idx]
            keep_pos = compute_keep_idx_gapcluster(pos_idx, scores, cat_std)
            if len(keep_pos) < len(pos_idx):
                n_split += 1
            else:
                n_nosplit += 1
            demoted = np.setdiff1d(pos_idx, keep_pos)
            keep_mask[demoted] = False

        items.append((text, target, keep_mask.astype(np.float32)))
    print(f"  gap-cluster stats: {n_split} queries split, {n_nosplit} queries kept whole "
          f"({n_split/(max(n_split+n_nosplit,1))*100:.1f}% split rate)", flush=True)
    return items


def plain_mse_loss(cos_sim, target, mask):
    return ((cos_sim - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)


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


def audc_qnc_peak(costs, accs):
    order = np.argsort(costs)
    c, a = costs[order], accs[order]
    grid = build_cost_grid(c, N_grid=COST_GRID_POINTS)
    a_grid = interp_to_grid(c, a, grid)
    audc = np.trapezoid(a_grid, grid) / (grid[-1] - grid[0])
    peak_idx = np.argmax(a)
    return {"audc": float(audc), "qnc": float(c[peak_idx]), "peak": float(a[peak_idx])}


def main():
    all_models = sorted(p.stem for p in PCA5_DIR.glob("*.npy"))
    models = [m for m in all_models if m not in EXCLUDE]
    print(f"All-seen pool: {len(models)} models", flush=True)

    costs = np.array([compute_cost(m, 0, cost_type="n_params") for m in models], dtype=np.float32)

    print("Loading EmbedLLM train.csv...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "category", "label", "prompt"])
    categories = sorted(df["category"].unique().tolist())
    category_to_idx = {c: i for i, c in enumerate(categories)}
    raw_cat_acc = build_raw_category_accuracy(df, models, categories)
    cat_std_by_idx = np.nanstd(raw_cat_acc, axis=0)
    print(f"  per-category accuracy std across all models: mean={np.nanmean(cat_std_by_idx):.4f} "
          f"min={np.nanmin(cat_std_by_idx):.4f} max={np.nanmax(cat_std_by_idx):.4f}", flush=True)

    print(f"Loading frozen MiniLM base (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    E = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in models])
    E_t = torch.from_numpy(E).float().to(DEVICE)
    E_t = E_t / (E_t.norm(dim=1, keepdim=True) + 1e-9)
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    items = build_items(df, models, raw_cat_acc, category_to_idx, cat_std_by_idx)
    print(f"{len(items)} usable training queries", flush=True)

    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(items))
    n_holdout = int(len(items) * HOLDOUT_FRAC)
    holdout_idx, train_idx = perm[:n_holdout], perm[n_holdout:]
    train_items = [items[i] for i in train_idx]
    holdout_texts = [items[i][0] for i in holdout_idx]
    holdout_targets = np.stack([items[i][1] for i in holdout_idx])
    holdout_masks = np.stack([items[i][2] for i in holdout_idx])

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
            cos_sim = q @ E_t.T
            loss = plain_mse_loss(cos_sim, target, mask)
            loss.backward()
            opt.step()
            opt.zero_grad()
        rho_arr = evaluate_holdout(enc, holdout_texts, holdout_targets, holdout_masks, E_t)
        if rho_arr.mean() > best_rho:
            best_rho, best_epoch = rho_arr.mean(), ep + 1
            best_state = {k: v.clone().cpu() for k, v in enc.proj.state_dict().items()}
    print(f"best_epoch={best_epoch} best_holdout_rho={best_rho:.4f} time={time.time()-t0:.1f}s", flush=True)
    enc.proj.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    ckpt_dir = Path("local_checkpoints/embedllm-allseen-gapcluster-seed0")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    enc.save(ckpt_dir)

    dataset = load_embedllm("test", candidates=models)
    texts = [ex["prompt"] for ex in dataset]
    label_maps = [ex["label_map"] for ex in dataset]

    embeds = np.zeros((len(texts), 5), dtype=np.float32)
    for start in range(0, len(texts), 64):
        batch = texts[start:start + 64]
        embeds[start:start + len(batch)] = enc.encode(batch)
    sims = embeds @ E_norm.T

    knn_costs, knn_accs, knn_Y = knn_curve(sims, models, label_maps, costs, LAM_LIST)
    metrics = audc_qnc_peak(knn_costs, knn_accs)

    out_path = ANALYSIS_DIR / "allseen_gapcluster_seed0_results.json"
    json.dump({"seed": SEED, "k_sigma": K_SIGMA, **metrics}, open(out_path, "w"), indent=2)

    print("\n" + "=" * 80)
    print(f"GAP-CLUSTER CATFILTER (K_SIGMA={K_SIGMA}) -- seed0 all-seen trial")
    print("=" * 80)
    print(f"AUDC={metrics['audc']:.4f} Peak={metrics['peak']:.4f}  "
          f"({'BEATS' if metrics['audc'] > CSCR_ALLSEEN else 'below'} CSCR {CSCR_ALLSEEN})")
    print("reference (same seed0, all-seen): top2=0.5430  top50pct=0.5432  margin=0.5381")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
