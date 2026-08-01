"""Full multi-seed control: BOTH Ceiling AND Perplexity FP, all 11 folds each,
seeds 0/1/2, beta=1.0 load-balancing -- same settings as full_loo_with_rank.py
(seed=0) but repeated across seeds so every result (oracle_match_rate, rank,
AUC, point-biserial) can be checked for seed-stability with a real control
group (Perplexity) alongside Ceiling, not just Ceiling in isolation.

66 fold-trainings total (11 folds x 2 FP types x 3 seeds) x ~220s ~= 4 hours.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import routerbench_loo_recovery as loo
import loo_unseen_recovery as base
from ceiling_multiseed_check import evaluate_fold_with_rank
from transformers import AutoModel, AutoTokenizer

SEEDS = [1, 2]  # seed=0 already computed in full_loo_with_rank_results.json -- don't redo it
BETA = 1.0
OUT_DIR = Path("local_descriptors/routerbench-analysis")
OUT_PATH = OUT_DIR / "full_loo_multiseed_results.json"
SEED0_PATH = OUT_DIR / "full_loo_with_rank_results.json"  # seed=0 source (task #9)

# reuse already-computed seed 1/2 folds from a previous partial run of this
# script, if present, instead of retraining them again
_existing = {}
if OUT_PATH.exists():
    with open(OUT_PATH) as f:
        _existing = json.load(f)


def main():
    print(f"DEVICE: {base.DEVICE}")
    pool_11 = loo.rb.NAMES

    print("Loading frozen MiniLM base (mean pooling)...")
    base._TOKENIZER = AutoTokenizer.from_pretrained(base.EMBED_MODEL)
    base._BASE_MODEL = AutoModel.from_pretrained(base.EMBED_MODEL).to(base.DEVICE)
    base._BASE_MODEL.eval()

    print("Loading RouterBench Set A / Set B...")
    set_a, set_b = loo.rb.load_data()
    set_b_texts = set_b["prompt"].tolist()
    cls_embeds = loo.precompute_set_b_embeddings(set_b_texts, base._TOKENIZER, base._BASE_MODEL)
    true_scores = loo.np.stack([set_b[m].to_numpy(dtype=float) for m in loo.rb.MODELS], axis=1)

    train_rows = loo.build_train_rows(set_a)
    cost_dict = loo.build_cost_dict(set_a)
    cost_arr = loo.np.array([cost_dict[n] for n in pool_11])
    cost_norm_11 = (cost_arr - cost_arr.min()) / (cost_arr.max() - cost_arr.min() + 1e-9)

    all_results = {"Ceiling": {}, "Perplexity": {}}
    t0 = time.time()
    for fp_name, desc_dir in [("Ceiling", loo.CEILING_DIR), ("Perplexity", loo.PERP_DIR)]:
        for held_out in pool_11:
            all_results[fp_name][held_out] = []
            pool_10 = [m for m in pool_11 if m != held_out]
            existing_rows = {r["seed"]: r for r in _existing.get(fp_name, {}).get(held_out, [])}
            for seed in SEEDS:
                if seed in existing_rows:
                    print(f"\n=== [{fp_name}] held_out={held_out} seed={seed} -- reusing existing result, "
                          f"skipping retrain ===", flush=True)
                    r = existing_rows[seed]
                    all_results[fp_name][held_out].append(r)
                    continue
                print(f"\n=== [{fp_name}] held_out={held_out} seed={seed} "
                      f"(elapsed {time.time()-t0:.0f}s) ===", flush=True)
                head, names10 = loo.train_fold(pool_10, desc_dir, train_rows, cost_dict,
                                                seed=seed, balance_beta=BETA)
                r = evaluate_fold_with_rank(head, held_out, pool_10, desc_dir, pool_11, cls_embeds,
                                             true_scores, cost_norm_11)
                r["seed"] = seed
                print(f"  n_oracle_is_M={r['n_oracle_is_M']}  oracle_match_rate={r['oracle_match_rate']:.4f}  "
                      f"mean_rank={r['mean_rank']:.2f}  AUC={r['auc_heldout_correctness']:.4f} "
                      f"(p={r['point_biserial_p_heldout']:.4f})", flush=True)
                all_results[fp_name][held_out].append(r)

            # incremental save after each held_out completes, so partial progress survives
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(OUT_PATH, "w") as f:
                json.dump(all_results, f, indent=2)

    # merge in seed=0 (already computed by task #9 in full_loo_with_rank.py) so the
    # final saved file has all 3 seeds without ever having retrained seed=0
    print(f"\nMerging in seed=0 results from {SEED0_PATH} ...")
    with open(SEED0_PATH) as f:
        seed0_data = json.load(f)
    for fp_name in ["Ceiling", "Perplexity"]:
        for r in seed0_data[fp_name]:
            r = dict(r)
            r["seed"] = 0
            all_results[fp_name][r["held_out"]].insert(0, r)
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}\nSUMMARY (3 seeds each)\n{'='*70}")
    for fp_name in ["Ceiling", "Perplexity"]:
        print(f"\n--- {fp_name} ---")
        for held_out in pool_11:
            rows = all_results[fp_name][held_out]
            rates = [r["oracle_match_rate"] for r in rows]
            aucs = [r["auc_heldout_correctness"] for r in rows]
            print(f"  {held_out:38s} oracle_match_rate={[f'{r:.3f}' for r in rates]}  "
                  f"AUC={[f'{a:.3f}' for a in aucs]}")

    # Ceiling vs Perplexity delta -- per seed, per model, plus pooled/mean across seeds
    print(f"\n{'='*70}\nCEILING vs PERPLEXITY -- delta per seed\n{'='*70}")
    for seed in [0] + SEEDS:
        print(f"\n-- seed={seed} --")
        c_rows = {ho: [r for r in all_results["Ceiling"][ho] if r["seed"] == seed][0] for ho in pool_11}
        p_rows = {ho: [r for r in all_results["Perplexity"][ho] if r["seed"] == seed][0] for ho in pool_11}

        c_num = sum(c_rows[ho]["n_oracle_is_M"] * c_rows[ho]["oracle_match_rate"] for ho in pool_11)
        c_den = sum(c_rows[ho]["n_oracle_is_M"] for ho in pool_11)
        p_num = sum(p_rows[ho]["n_oracle_is_M"] * p_rows[ho]["oracle_match_rate"] for ho in pool_11)
        p_den = sum(p_rows[ho]["n_oracle_is_M"] for ho in pool_11)
        c_pooled = c_num / c_den if c_den else float("nan")
        p_pooled = p_num / p_den if p_den else float("nan")

        c_auc_mean = float(loo.np.mean([c_rows[ho]["auc_heldout_correctness"] for ho in pool_11]))
        p_auc_mean = float(loo.np.mean([p_rows[ho]["auc_heldout_correctness"] for ho in pool_11]))

        print(f"  pooled oracle_match_rate: Ceiling={c_pooled:.4f}  Perplexity={p_pooled:.4f}  "
              f"delta={c_pooled - p_pooled:+.4f}")
        print(f"  mean AUC:                 Ceiling={c_auc_mean:.4f}  Perplexity={p_auc_mean:.4f}  "
              f"delta={c_auc_mean - p_auc_mean:+.4f}")
        for ho in pool_11:
            d_auc = c_rows[ho]["auc_heldout_correctness"] - p_rows[ho]["auc_heldout_correctness"]
            d_rate = c_rows[ho]["oracle_match_rate"] - p_rows[ho]["oracle_match_rate"]
            print(f"    {ho:38s} AUC delta={d_auc:+.4f}  oracle_rate delta={d_rate:+.4f}")

    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
