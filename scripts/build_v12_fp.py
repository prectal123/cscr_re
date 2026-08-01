"""Build v1.2 FP: LLM-backbone expertise-summary embedding (see FP_IDEAS.md).

Idea: instead of measuring a model's *token statistics* (Logit/Perplexity/
Lexical FP) or a *ground-truth score vector* (Ceiling FP), ask a small LLM
backbone to directly summarize, in natural language, what each pool model
seems good/bad at -- based on its actual (probe prompt, response) pairs --
then embed that summary text. One summarization CALL PER MODEL (not per
prompt), so cost stays tiny even though it's a "capability description"
rather than a token-statistic FP.

Steps:
1. Load the existing 32-probe set (local_data/probes_mix-instruct-32.json)
   -- these prompt_ids are literal MixInstruct dataset ids, so each pool
   model's real candidate response text can be looked up directly, no new
   generation needed.
2. For each of the 11 pool models, bundle all (probe, response) pairs it has
   into one prompt, ask a small local instruct LLM to summarize the model's
   apparent strengths/weaknesses/style in a short paragraph.
3. Embed that summary paragraph with the same MiniLM backbone used
   elsewhere (CLS pooling) -> this is the model's v1.2 FP (384-dim, MiniLM's
   native hidden size -- each FP type gets its own proj_dim in training, so
   this doesn't need to match Logit/Perplexity/Ceiling's 192).
"""
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

PROBES_PATH = "local_data/probes_mix-instruct-32.json"
OUT_DIR = Path("local_descriptors/mix-instruct-v1.2")
POOL_PATH = "experts/pool-mix-instruct-11.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# 0.5B-Instruct was tried first but confused the ROLE ("you are analyzing another
# system") with ITSELF -- every summary opened with "Qwen, the AI assistant created
# by Alibaba Cloud, is capable of..." regardless of the actual input examples, making
# all 11 summaries near-identical boilerplate. 1.5B follows the role separation
# correctly in spot checks.
SUMMARIZER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


SYSTEM_PROMPT = (
    "You are a technical analyst reviewing transcripts from a THIRD-PARTY AI system "
    "called 'System X'. You are not System X and must never describe yourself. Your "
    "only job is to characterize System X's behavior, in the third person, based "
    "strictly on the transcript excerpts you are given."
)


def build_summary_prompt(qa_pairs: list[tuple[str, str]]) -> str:
    lines = [
        f"Here are {len(qa_pairs)} transcript excerpts of System X answering different "
        f"questions.\n"
    ]
    for i, (q, a) in enumerate(qa_pairs, 1):
        q_trunc = q.strip()[:300]
        a_trunc = a.strip()[:300]
        lines.append(f"\n[Excerpt {i}]\nQuestion to System X: {q_trunc}\nSystem X answered: {a_trunc}")
    lines.append(
        "\n\nWrite a short paragraph (3-5 sentences) describing System X's apparent "
        "strengths, weaknesses, and response style, based only on the excerpts above. "
        "Start your answer with the words \"System X\". Do not describe yourself or "
        "any AI assistant other than System X."
    )
    return "".join(lines)


def main():
    print(f"DEVICE: {DEVICE}")
    pool = json.load(open(POOL_PATH))
    pool_set = set(pool)
    probes = json.load(open(PROBES_PATH))
    probe_ids = {p["prompt_id"] for p in probes}
    print(f"Pool ({len(pool)}), probes: {len(probe_ids)}")

    print("Loading mix-instruct to recover each pool model's response to the probes...")
    raw = concatenate_datasets([
        load_dataset("llm-blender/mix-instruct", split="train"),
        load_dataset("llm-blender/mix-instruct", split="validation"),
    ])

    qa_by_model: dict[str, dict[str, str]] = {m: {} for m in pool}  # model -> {prompt: response}
    n_found = 0
    for rec in raw:
        pid = rec["id"]
        if pid not in probe_ids:
            continue
        n_found += 1
        prompt_text = f"{rec['instruction']} {rec['input']}".strip()
        for cand in rec["candidates"]:
            hf_name = NAME_TO_HF.get(cand["model"])
            if hf_name is None or hf_name not in pool_set:
                continue
            # NAME_TO_HF aliases both "mpt-7b" and "mpt-7b-instruct" to the same HF name,
            # so a single prompt can otherwise contribute 2 near-duplicate entries -- keep
            # one response per (model, prompt) pair, which also caps every model at the
            # same 32-probe budget instead of one silently getting 64.
            qa_by_model[hf_name][prompt_text] = cand["text"]
    qa_by_model = {m: list(d.items()) for m, d in qa_by_model.items()}
    print(f"Recovered {n_found}/{len(probe_ids)} probe prompts from the dataset")
    for m in pool:
        print(f"  {m:55s} {len(qa_by_model[m])} responses")

    print(f"\nLoading summarizer backbone: {SUMMARIZER_MODEL} ...")
    sum_tok = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
    sum_model = AutoModelForCausalLM.from_pretrained(SUMMARIZER_MODEL, torch_dtype=torch.float16).to(DEVICE)
    sum_model.eval()

    print(f"Loading embedder: {EMBED_MODEL} ...")
    emb_tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    emb_model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
    emb_model.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for model_name in pool:
        qa_pairs = qa_by_model[model_name]
        if not qa_pairs:
            print(f"  SKIP {model_name}: no responses found")
            continue
        user_prompt = build_summary_prompt(qa_pairs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        chat_text = sum_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = sum_tok(chat_text, return_tensors="pt", truncation=True, max_length=4096).to(DEVICE)
        with torch.no_grad():
            out = sum_model.generate(
                **enc, max_new_tokens=180, do_sample=False,
                pad_token_id=sum_tok.eos_token_id,
            )
        gen_text = sum_tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        summaries[model_name] = gen_text
        print(f"\n=== {model_name} ===\n{gen_text}\n")

        with torch.no_grad():
            e_enc = emb_tok(gen_text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
            out_e = emb_model(**e_enc)
            # all-MiniLM-L6-v2 is trained for mean pooling, not CLS-token pooling -- see
            # loo_unseen_recovery.py's mean_pool() for the full story (CLS pooling gave
            # 0.62 baseline anisotropy among unrelated prompts vs 0.05 for mean pooling).
            mask = e_enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out_e.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vec = (pooled / (pooled.norm(dim=1, keepdim=True) + 1e-9))[0].cpu().numpy().astype(np.float32)
        np.save(OUT_DIR / f"{model_name}.npy", vec)

    with open(OUT_DIR / "summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nDone. Saved {len(summaries)} v1.2 FPs + summaries -> {OUT_DIR}")

    E = np.stack([np.load(OUT_DIR / f"{m}.npy") for m in summaries])
    sim = E @ E.T
    off = sim[~np.eye(len(summaries), dtype=bool)]
    print(f"Pairwise cosine sim: mean={off.mean():.4f} std={off.std():.4f} min={off.min():.4f} max={off.max():.4f}")


if __name__ == "__main__":
    main()
