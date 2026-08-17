"""Decide the category-filter cutoff methodology (top2 / top50pct / margin)
via multi-seed comparison on EmbedLLM, both protocols (unseen vs CSCR
0.4848, all-seen vs CSCR 0.541). All three cutoff rules share ONE function
(compute_keep_idx) -- they differ only in which subset of the positive-
advantage candidates survives, not in any other code path -- so any
performance difference is attributable purely to the cutoff rule itself,
not to accidental implementation drift between three separately-written
variants.

Uses the already-fixed cost registry (experts/registry.json) and excludes
the phantom model (JaeyeonKang__CCK_Asura_v1, HF repo no longer exists) --
see PROGRESS.md #21's cost-bug section.

Missing baseline filled in here too: EmbedLLM all-seen vanilla-GRPO was
never actually run (only min-pos and catfilter-top2 were) -- added as a
4th config for completeness on the all-seen protocol.
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
from embedllm_newllm_fast_eval import run_one as fast_eval_unseen

PCA5_DIR = Path("local_descriptors/embedllm-ceiling-pca5")
ANALYSIS_DIR = Path("local_descriptors/embedllm-analysis")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
HOLDOUT_FRAC = 0.15
SEEDS = [0, 1, 2]
MARGIN = 0.15
LOG_EVERY = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSCR_UNSEEN = 0.4848
CSCR_ALLSEEN = 0.541
EXCLUDE = {"JaeyeonKang__CCK_Asura_v1"}
LAM_LIST = np.logspace(-4, 2, 20)
K = 20
BANDIT_BETA = 0.000001
COST_GRID_POINTS = 20

CUTOFF_MODES = ["top2", "top50pct", "margin"]


def compute_keep_idx(pos_idx: np.ndarray, scores: np.ndarray, mode: str) -> np.ndarray:
    """THE single shared cutoff rule -- all three catfilter variants call
    this and ONLY this to decide which positive-advantage candidates survive.
    pos_idx: original indices of positive-advantage models for this query.
    scores: their category track-record accuracy (same order as pos_idx).
    Returns the subset of pos_idx to KEEP.
    """
    if len(pos_idx) <= 1:
        return pos_idx
    order = np.argsort(-scores)
    sorted_idx = pos_idx[order]
    sorted_scores = scores[order]
    if mode == "top2":
        return sorted_idx[:2]
    elif mode == "top50pct":
        n_keep = max(1, int(np.ceil(len(pos_idx) * 0.5)))
        return sorted_idx[:n_keep]
    elif mode == "margin":
        best = sorted_scores[0]
        return sorted_idx[sorted_scores >= best - MARGIN]
    else:
        raise ValueError(f"unknown cutoff mode: {mode}")


def build_raw_category_accuracy(df, models, categories):
    pivot = df.pivot_table(index="model_name", columns="category", values="label", aggfunc="mean")
    pivot = pivot.reindex(index=models, columns=categories)
    return pivot.to_numpy()


def build_items(df, models, raw_cat_acc, category_to_idx):
    """Build items ONCE with all 3 masks precomputed (one per cutoff mode)
    plus the unfiltered mask, so training loops just pick a column."""
    name_to_idx = {n: i for i, n in enumerate(models)}
    items = []
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

        pos_idx = np.where(mask & (labels == 1))[0]
        masks_by_mode = {"unfiltered": mask.astype(np.float32)}
        if cat_idx is not None and len(pos_idx) > 0:
            scores = np.nan_to_num(raw_cat_acc[pos_idx, cat_idx], nan=-1.0)
            for mode in CUTOFF_MODES:
                keep_pos = compute_keep_idx(pos_idx, scores, mode)
                demoted = np.setdiff1d(pos_idx, keep_pos)
                km = mask.copy()
                km[demoted] = False
                masks_by_mode[mode] = km.astype(np.float32)
        else:
            for mode in CUTOFF_MODES:
                masks_by_mode[mode] = mask.astype(np.float32)

        items.append((text, target, masks_by_mode))
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


def train(seed, items, mode, tokenizer, base_model, E_t, tag):
    torch.manual_seed(seed)
    sub_items = [(t, tgt, masks[mode]) for (t, tgt, masks) in items]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(sub_items))
    n_holdout = int(len(sub_items) * HOLDOUT_FRAC)
    holdout_idx, train_idx = perm[:n_holdout], perm[n_holdout:]
    train_sub = [sub_items[i] for i in train_idx]
    holdout_texts = [sub_items[i][0] for i in holdout_idx]
    holdout_targets = np.stack([sub_items[i][1] for i in holdout_idx])
    holdout_masks = np.stack([sub_items[i][2] for i in holdout_idx])

    enc = QueryEncoder.__new__(QueryEncoder)
    torch.nn.Module.__init__(enc)
    enc.tokenizer = tokenizer
    enc.model = base_model
    enc.device = DEVICE
    enc.hidden_size = base_model.config.hidden_size
    enc.proj_dim = E_t.size(1)
    enc.proj = torch.nn.Sequential(
        torch.nn.Linear(enc.hidden_size, enc.hidden_size, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(enc.hidden_size, E_t.size(1), bias=False),
    ).to(DEVICE)
    enc.model.config.proj_dim = E_t.size(1)
    opt = torch.optim.Adam(enc.proj.parameters(), lr=LR)

    def collate(batch):
        texts, targets, masks = zip(*batch)
        toks = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt", max_length=256)
        return toks, torch.tensor(np.stack(targets)), torch.tensor(np.stack(masks))

    loader = torch.utils.data.DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

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
    print(f"  [{tag}] best_epoch={best_epoch} best_holdout_rho={best_rho:.4f} time={time.time()-t0:.1f}s", flush=True)
    enc.proj.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return enc


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
    audc = np.trapezoid(a_grid, grid) / (grid[-1] - grid[0])
    peak_idx = np.argmax(a)
    return {"audc": float(audc), "qnc": float(c[peak_idx]), "peak": float(a[peak_idx])}


def eval_allseen(enc, models, texts, label_maps, E_norm, costs):
    embeds = np.zeros((len(texts), E_norm.shape[1]), dtype=np.float32)
    for start in range(0, len(texts), 64):
        batch = texts[start:start + 64]
        embeds[start:start + len(batch)] = enc.encode(batch)
    sims = embeds @ E_norm.T
    knn_costs, knn_accs, knn_Y = knn_curve(sims, models, label_maps, costs, LAM_LIST)
    return audc_qnc_peak(knn_costs, knn_accs)


def main():
    print("Loading EmbedLLM train.csv...", flush=True)
    f_train = hf_hub_download(repo_id="RZ412/EmbedLLM", repo_type="dataset", filename="train.csv")
    df = pd.read_csv(f_train, usecols=["prompt_id", "model_name", "category", "label", "prompt"])
    categories = sorted(df["category"].unique().tolist())
    category_to_idx = {c: i for i, c in enumerate(categories)}

    print(f"Loading frozen MiniLM base ONCE (device={DEVICE})...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    base_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    results = {"allseen": {m: [] for m in CUTOFF_MODES}}

    # ===================== ALL-SEEN PROTOCOL =====================
    # (unseen protocol skipped this round -- user wants to finalize the
    # catfilter cutoff methodology on all-seen first, 3 modes x 3 seeds = 9
    # runs, before spending time on the unseen protocol too)
    print("\n" + "#" * 70 + "\nALL-SEEN PROTOCOL\n" + "#" * 70, flush=True)
    all_models = sorted(p.stem for p in PCA5_DIR.glob("*.npy"))
    models = [m for m in all_models if m not in EXCLUDE]
    costs = np.array([compute_cost(m, 0, cost_type="n_params") for m in models], dtype=np.float32)
    E_all = np.stack([np.load(PCA5_DIR / f"{m}.npy") for m in models])
    E_all_t = torch.from_numpy(E_all).float().to(DEVICE)
    E_all_t = E_all_t / (E_all_t.norm(dim=1, keepdim=True) + 1e-9)
    E_all_norm = E_all / (np.linalg.norm(E_all, axis=1, keepdims=True) + 1e-12)
    raw_cat_acc_all = build_raw_category_accuracy(df, models, categories)
    items_all = build_items(df, models, raw_cat_acc_all, category_to_idx)

    dataset = load_embedllm("test", candidates=models)
    texts = [ex["prompt"] for ex in dataset]
    label_maps = [ex["label_map"] for ex in dataset]

    for seed in SEEDS:
        for mode in CUTOFF_MODES:
            enc = train(seed, items_all, mode, tokenizer, base_model, E_all_t, f"allseen-seed{seed}-{mode}")
            ckpt_dir = Path(f"local_checkpoints/embedllm-allseen-catcompare-{mode}-seed{seed}")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            enc.save(ckpt_dir)
            m_result = eval_allseen(enc, models, texts, label_maps, E_all_norm, costs)
            results["allseen"][mode].append({"seed": seed, **m_result})
            print(f"  [allseen seed={seed} {mode}] AUDC={m_result['audc']:.4f} Peak={m_result['peak']:.4f} "
                  f"({'BEATS' if m_result['audc'] > CSCR_ALLSEEN else 'below'} CSCR)", flush=True)

    out_path = ANALYSIS_DIR / "allseen_catfilter_methodology_multiseed_results.json"
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n" + "=" * 100)
    print("CATFILTER CUTOFF METHODOLOGY COMPARISON -- ALL-SEEN ONLY (top2 / top50pct / margin)")
    print("=" * 100)
    for mode in CUTOFF_MODES:
        audcs = [r["audc"] for r in results["allseen"][mode]]
        beats = sum(1 for a in audcs if a > CSCR_ALLSEEN)
        print(f"  {mode:>12}: " + " / ".join(f"{a:.4f}" for a in audcs) +
              f"  mean={np.mean(audcs):.4f} (std={np.std(audcs):.4f})  beats CSCR {CSCR_ALLSEEN}: {beats}/{len(audcs)}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
