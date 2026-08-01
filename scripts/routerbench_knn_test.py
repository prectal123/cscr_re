"""Repeat the MLP-free kNN unseen-model-recovery test (see knn_unseen_recovery.py)
on RouterBench instead of MixInstruct.

Why RouterBench: much larger (36,497 rows vs MixInstruct's ~100k but spread over
only ~1 task style), genuinely diverse (86 distinct eval tasks -- hellaswag,
grade-school-math, mmlu-professional-law, arc-challenge, winogrande, etc, not
just instruction-following), clean binary correct/incorrect scores (no bartscore
length confound), real per-response dollar cost, and actual response text per
model (so both Ceiling and V1.2 FPs are buildable). No logprobs available, so
Perplexity FP is skipped here.

Ceiling FP: uses eval_name as the natural "cluster" (much cleaner than
MiniLM-embedding k-means) -- for each model, mean accuracy per eval_name on
Set A, mean-centered across the pool (removes shared task-difficulty), then
L2-normalized. Same fix as build_ceiling_fp_centered.py for MixInstruct.

V1.2 FP: same LLM-backbone expertise-summary approach as build_v12_fp.py,
using real (prompt, model_response) pairs from Set A, Qwen2.5-1.5B-Instruct
summarizer with the same role-separated prompt, MiniLM mean-pooled embedding.

Then: same kNN unseen-recovery test as knn_unseen_recovery.py on held-out Set B.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

MODELS = [
    'claude-instant-v1', 'claude-v1', 'claude-v2', 'gpt-3.5-turbo-1106',
    'gpt-4-1106-preview', 'meta/code-llama-instruct-34b-chat',
    'meta/llama-2-70b-chat', 'mistralai/mistral-7b-chat',
    'mistralai/mixtral-8x7b-chat', 'zero-one-ai/Yi-34B-Chat',
    'WizardLM/WizardLM-13B-V1.2',
]
NAMES = [m.replace("/", "__") for m in MODELS]
M2COL = dict(zip(NAMES, MODELS))

OUT_DIR = Path("local_descriptors/routerbench-analysis")
CEILING_DIR = Path("local_descriptors/routerbench-ceiling")
V12_DIR = Path("local_descriptors/routerbench-v1.2")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARIZER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SET_A_FRACTION = 0.8
SPLIT_SEED = 42
N_PROBES = 32


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def load_data():
    print("Loading RouterBench...")
    f = hf_hub_download(repo_id="withmartian/routerbench", repo_type="dataset", filename="routerbench_0shot.pkl")
    import pandas as pd
    df = pd.read_pickle(f)
    df = df.dropna(subset=[m for m in MODELS] + ["eval_name", "prompt"])
    rng = np.random.RandomState(SPLIT_SEED)
    idx = rng.permutation(len(df))
    n_a = int(len(df) * SET_A_FRACTION)
    set_a = df.iloc[idx[:n_a]].reset_index(drop=True)
    set_b = df.iloc[idx[n_a:]].reset_index(drop=True)
    print(f"Total {len(df)} rows -> Set A {len(set_a)}, Set B {len(set_b)}")
    return set_a, set_b


def build_ceiling_fp(set_a):
    print("\nBuilding Ceiling FP (eval_name-clustered, mean-centered)...")
    eval_names = sorted(set_a["eval_name"].unique())
    print(f"  {len(eval_names)} distinct eval tasks")
    name_to_idx = {e: i for i, e in enumerate(eval_names)}

    raw = {}
    for name, model_col in zip(NAMES, MODELS):
        vec = np.zeros(len(eval_names))
        counts = np.zeros(len(eval_names))
        for ev, score in zip(set_a["eval_name"], set_a[model_col]):
            vec[name_to_idx[ev]] += float(score)
            counts[name_to_idx[ev]] += 1
        vec = vec / np.maximum(counts, 1)
        raw[name] = vec

    pool_matrix = np.stack([raw[n] for n in NAMES])
    pool_mean = pool_matrix.mean(axis=0)

    CEILING_DIR.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        centered = raw[name] - pool_mean
        vec = (centered / (np.linalg.norm(centered) + 1e-12)).astype(np.float32)
        np.save(CEILING_DIR / f"{name}.npy", vec)

    E = np.stack([np.load(CEILING_DIR / f"{n}.npy") for n in NAMES])
    sim = E @ E.T
    off = sim[~np.eye(len(NAMES), dtype=bool)]
    print(f"  pairwise cosine sim after centering: mean={off.mean():.4f} std={off.std():.4f} "
          f"min={off.min():.4f} max={off.max():.4f}")


SYSTEM_PROMPT = (
    "You are a technical analyst reviewing transcripts from a THIRD-PARTY AI system "
    "called 'System X'. You are not System X and must never describe yourself. Your "
    "only job is to characterize System X's behavior, in the third person, based "
    "strictly on the transcript excerpts you are given."
)


def build_summary_prompt(qa_pairs):
    lines = [f"Here are {len(qa_pairs)} transcript excerpts of System X answering different questions.\n"]
    for i, (q, a, correct) in enumerate(qa_pairs, 1):
        q_trunc = str(q).strip()[:300]
        a_trunc = str(a).strip()[:300]
        outcome = "CORRECT" if correct else "INCORRECT"
        lines.append(f"\n[Excerpt {i}, System X was {outcome}]\nQuestion: {q_trunc}\nSystem X answered: {a_trunc}")
    lines.append(
        "\n\nWrite a short paragraph (3-5 sentences) describing System X's apparent "
        "strengths, weaknesses, and response style, based only on the excerpts above. "
        "Start your answer with the words \"System X\". Do not describe yourself or "
        "any AI assistant other than System X."
    )
    return "".join(lines)


def build_v12_fp(set_a):
    print("\nBuilding V1.2 FP (LLM expertise-summary embedding)...")
    rng = random.Random(SPLIT_SEED)
    idx_pool = list(range(len(set_a)))
    rng.shuffle(idx_pool)
    probe_idx = idx_pool[:N_PROBES]
    probes = set_a.iloc[probe_idx]

    print(f"  Loading summarizer {SUMMARIZER_MODEL} ...")
    sum_tok = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
    sum_model = AutoModelForCausalLM.from_pretrained(SUMMARIZER_MODEL, torch_dtype=torch.float16).to(DEVICE)
    sum_model.eval()
    print(f"  Loading embedder {EMBED_MODEL} ...")
    emb_tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    emb_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    emb_model.eval()

    V12_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, model_col in zip(NAMES, MODELS):
        resp_col = f"{model_col}|model_response"
        qa_pairs = [(r["prompt"], r[resp_col], r[model_col]) for _, r in probes.iterrows()]
        user_prompt = build_summary_prompt(qa_pairs)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
        chat_text = sum_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = sum_tok(chat_text, return_tensors="pt", truncation=True, max_length=4096).to(DEVICE)
        with torch.no_grad():
            out = sum_model.generate(**enc, max_new_tokens=180, do_sample=False, pad_token_id=sum_tok.eos_token_id)
        gen_text = sum_tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        summaries[name] = gen_text
        print(f"\n=== {name} ===\n{gen_text}\n")

        with torch.no_grad():
            e_enc = emb_tok(gen_text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
            out_e = emb_model(**e_enc)
            pooled = mean_pool(out_e.last_hidden_state, e_enc["attention_mask"])
            vec = (pooled / (pooled.norm(dim=1, keepdim=True) + 1e-9))[0].cpu().numpy().astype(np.float32)
        np.save(V12_DIR / f"{name}.npy", vec)

    with open(V12_DIR / "summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)

    del sum_model
    torch.cuda.empty_cache()

    E = np.stack([np.load(V12_DIR / f"{n}.npy") for n in NAMES])
    sim = E @ E.T
    off = sim[~np.eye(len(NAMES), dtype=bool)]
    print(f"  pairwise cosine sim: mean={off.mean():.4f} std={off.std():.4f} min={off.min():.4f} max={off.max():.4f}")


def knn_test(set_b, desc_dir, fp_name):
    print(f"\n{'='*60}\nkNN unseen-recovery test: {fp_name}\n{'='*60}")
    E = np.stack([np.load(desc_dir / f"{n}.npy") for n in NAMES])
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    sim_full = E @ E.T

    true_scores = np.stack([set_b[m].to_numpy(dtype=float) for m in MODELS], axis=1)  # (N, 11)

    fp_rhos, uniform_rhos = [], []
    for i, held_out in enumerate(NAMES):
        others_idx = [j for j in range(len(NAMES)) if j != i]
        sims = sim_full[i, others_idx]
        w = np.clip(sims, 0, None)
        if w.sum() < 1e-9:
            w = np.ones_like(w)
        w = w / w.sum()

        other_scores = true_scores[:, others_idx]
        true_m = true_scores[:, i]
        fp_proxy = other_scores @ w
        uniform_proxy = other_scores.mean(axis=1)

        fp_rho, _ = spearmanr(fp_proxy, true_m)
        uni_rho, _ = spearmanr(uniform_proxy, true_m)
        fp_rhos.append(fp_rho)
        uniform_rhos.append(uni_rho)
        print(f"  held out {held_out:35s} FP-proxy rho={fp_rho:.4f}  uniform-proxy rho={uni_rho:.4f}")

    fp_rhos, uniform_rhos = np.array(fp_rhos), np.array(uniform_rhos)
    delta = fp_rhos - uniform_rhos
    from scipy import stats
    t, p = stats.ttest_rel(fp_rhos, uniform_rhos)
    print(f"\nmean FP rho={fp_rhos.mean():.4f}  mean uniform rho={uniform_rhos.mean():.4f}  "
          f"mean delta={delta.mean():+.4f}  paired t-test p={p:.4f}  ({(delta>0).sum()}/11 folds improved)")
    return {
        "fp_rho": fp_rhos.tolist(), "uniform_rho": uniform_rhos.tolist(),
        "mean_fp_rho": float(fp_rhos.mean()), "mean_uniform_rho": float(uniform_rhos.mean()),
        "mean_delta": float(delta.mean()), "paired_t_p": float(p),
        "n_folds_improved": int((delta > 0).sum()),
    }


def main():
    print(f"DEVICE: {DEVICE}")
    set_a, set_b = load_data()
    build_ceiling_fp(set_a)
    build_v12_fp(set_a)

    results = {}
    results["Ceiling"] = knn_test(set_b, CEILING_DIR, "Ceiling")
    results["V1.2"] = knn_test(set_b, V12_DIR, "V1.2")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "routerbench_knn_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}\nFINAL SUMMARY (RouterBench)\n{'='*60}")
    print(f"{'FP':12s} {'mean FP rho':>12s} {'mean uniform rho':>18s} {'mean delta':>12s} {'p':>8s} {'folds+':>8s}")
    for fp_name, r in results.items():
        print(f"{fp_name:12s} {r['mean_fp_rho']:12.4f} {r['mean_uniform_rho']:18.4f} "
              f"{r['mean_delta']:+12.4f} {r['paired_t_p']:8.4f} {r['n_folds_improved']:6d}/11")
    print(f"\nSaved -> {OUT_DIR / 'routerbench_knn_results.json'}")


if __name__ == "__main__":
    main()
