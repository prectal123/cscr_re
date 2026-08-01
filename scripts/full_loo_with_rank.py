"""Re-run the full 11-fold LOO experiments (Ceiling + Perplexity FP, beta=1.0
load-balancing, seed=0 -- same settings as the original ceiling_loo_beta1.0
and perplexity_loo_beta1.0 runs) but with the rank/margin contention diagnostic
added (see ceiling_multiseed_check.py's evaluate_fold_with_rank): oracle_match_rate
alone doesn't distinguish "held-out lost a near-tie to a strong competitor"
from "held-out was never in contention" -- this adds mean_rank (1=hit, chance=6.0)
and mean_margin_when_miss (cosine-sim gap to the top choice when not a hit).
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

SEED = 0
BETA = 1.0
OUT_DIR = Path("local_descriptors/routerbench-analysis")


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

    all_results = {}
    t0 = time.time()
    for fp_name, desc_dir in [("Ceiling", loo.CEILING_DIR), ("Perplexity", loo.PERP_DIR)]:
        print(f"\n{'='*70}\nFP type: {fp_name}  (seed={SEED}, beta={BETA})\n{'='*70}")
        fold_results = []
        for held_out in pool_11:
            pool_10 = [m for m in pool_11 if m != held_out]
            print(f"\n[{fp_name}] held out: {held_out} ... (elapsed {time.time()-t0:.0f}s)", flush=True)
            head, names10 = loo.train_fold(pool_10, desc_dir, train_rows, cost_dict,
                                            seed=SEED, balance_beta=BETA)
            r = evaluate_fold_with_rank(head, held_out, pool_10, desc_dir, pool_11, cls_embeds,
                                         true_scores, cost_norm_11)
            r["held_out"] = held_out
            print(f"  n_oracle_is_M={r['n_oracle_is_M']}  oracle_match_rate={r['oracle_match_rate']:.4f}  "
                  f"mean_rank={r['mean_rank']:.2f} (chance=6.0)  median_rank={r['median_rank']:.1f}  "
                  f"mean_margin_when_miss={r['mean_margin_when_miss']:.4f}", flush=True)
            print(f"  rank_histogram: {r['rank_histogram']}", flush=True)
            print(f"  mean_correct_incorrect_gap={r['mean_correct_incorrect_gap']:.4f} (n={r['n_gap_samples']})  "
                  f"point_biserial_corr={r['point_biserial_corr_heldout']:.4f} (p={r['point_biserial_p_heldout']:.4f})  "
                  f"AUC={r['auc_heldout_correctness']:.4f} (chance=0.5, n_pos={r['n_heldout_positive_in_setB']}/{r['n_setB_total']})",
                  flush=True)
            fold_results.append(r)
        all_results[fp_name] = fold_results

        n_total = sum(r["n_oracle_is_M"] for r in fold_results)
        pooled_hits = sum(r["n_oracle_is_M"] * r["oracle_match_rate"] for r in fold_results
                           if r["n_oracle_is_M"] > 0)
        pooled_rate = pooled_hits / n_total if n_total else float("nan")
        rates = [r["oracle_match_rate"] for r in fold_results if r["n_oracle_is_M"] > 0]
        print(f"\n--- {fp_name} summary ---")
        print(f"simple mean oracle_match_rate: {sum(rates)/len(rates):.4f} (chance ~0.0909)")
        print(f"pooled(n-weighted) oracle_match_rate: {pooled_rate:.4f}  (total_n={n_total})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "full_loo_with_rank_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
