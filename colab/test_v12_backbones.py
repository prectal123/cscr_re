# %% [markdown]
# # v1.2 FP methodology — backbone & instruction sweep
#
# Paste each `# %%` block below into its own Colab cell, in order.
#
# Background: v1.2 (FP_IDEAS.md) proposes summarizing each model's
# capability from a pool of its real probe responses using a lightweight
# LLM "backbone", then embedding that summary (MiniLM) as the model's
# descriptor. The prototype in PROGRESS.md hand-wrote these summaries by
# reading real responses (rho=+0.43 vs capability on the 7-model pool,
# not statistically significant at n=7, see local_descriptors/analysis/
# rsa_scatter_v12_vs_capability.png) — the open question is whether an
# ACTUAL small backbone model, called once per target model, reproduces
# that signal, and whether the exact instruction wording matters.
#
# This script tries every (backbone, instruction) combination against all
# 11 MixInstruct models and reports RSA (vs the existing bartscore
# capability vectors, exact Mantel test) for each combination, so you can
# see which setup actually tracks capability best -- not just eyeball one
# run.
#
# Needs GPU runtime (T4 is enough -- backbones here are <=8B, 4-bit).
# Total run time dominated by backbone download (~2x, ~15-30GB) + ~11
# generate() calls per (backbone, instruction) pair -- budget ~30-60 min
# for the default 2 backbones x 3 instructions = 6 combinations.

# %% Cell 1 — mount Drive, set up persistent OUTPUT paths (not model cache)
from google.colab import drive
drive.mount('/content/drive')

import os

DRIVE_ROOT = "/content/drive/MyDrive/cscr_repro"
OUT_DIR = f"{DRIVE_ROOT}/v12_backbone_sweep"
os.makedirs(OUT_DIR, exist_ok=True)

os.environ["HF_HOME"] = "/content/hf_cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# %% Cell 2 — clone repo + install deps
get_ipython().system('git clone -q https://github.com/prectal123/cscr_re.git /content/cscr')
get_ipython().run_line_magic('cd', '/content/cscr')
get_ipython().system('pip install -e . -q')
get_ipython().system('pip install -q bitsandbytes accelerate sentencepiece protobuf sentence-transformers datasets scipy')

import sys
sys.path.insert(0, "/content/cscr/src")

# %% Cell 3 — pull real probe responses for all 11 pool models (no model
# download needed -- MixInstruct's dataset already stores each model's
# response text, same trick used for the "free" perplexity descriptors)
import json
from datasets import load_dataset

NAME_TO_HF = {
    'vicuna-13b-1.1': 'eachadea__vicuna-13b-1.1',
    'alpaca-native': 'chavinlo__alpaca-native',
    'dolly-v2-12b': 'databricks__dolly-v2-12b',
    'stablelm-tuned-alpha-7b': 'stabilityai__stablelm-tuned-alpha-7b',
    'oasst-sft-4-pythia-12b-epoch-3.5': 'OpenAssistant__oasst-sft-4-pythia-12b-epoch-3.5',
    'koala-7B-HF': 'TheBloke__koala-7B-HF',
    'llama-7b-hf-baize-lora-bf16': 'mosesjun0h__llama-7b-hf-baize-lora-bf16',
    'flan-t5-xxl': 'google__flan-t5-xxl',
    'chatglm-6b': 'THUDM__chatglm-6b',
    'moss-moon-003-sft': 'fnlp__moss-moon-003-sft',
    'mpt-7b-instruct': 'mosaicml__mpt-7b-instruct',
}
ALL_11 = sorted(set(NAME_TO_HF.values()))
N_PROBES_PER_MODEL = 25

probes_path = f"{OUT_DIR}/probe_responses_11.json"
if os.path.exists(probes_path):
    probe_data = json.load(open(probes_path))
    print(f"reusing cached probe responses -> {probes_path}")
else:
    ds = load_dataset('llm-blender/mix-instruct', split='train', streaming=True)
    found = []
    for rec in ds:
        texts = {}
        for cand in rec['candidates']:
            hf = NAME_TO_HF.get(cand['model'])
            if hf:
                texts[hf] = cand['text']
        if len(texts) == len(ALL_11):
            prompt = ((rec['instruction'] or '') + ' ' + (rec['input'] or '')).strip()
            found.append({'prompt': prompt, 'texts': texts})
        if len(found) >= N_PROBES_PER_MODEL:
            break
    probe_data = found
    json.dump(probe_data, open(probes_path, 'w'))
    print(f"saved {len(probe_data)} probe rows -> {probes_path}")

# %% Cell 4 — instruction variants + backbone candidates
# Variant A: free-form, mirrors v1.2's original "summarize expertise" framing.
# Variant B: structured JSON schema, matches the hand-written prototype's
#   fields exactly (strengths/performance/flaws/traits) for apples-to-apples
#   comparison against the PROGRESS.md baseline.
# Variant C: reliability-focused, built from what the hand-written prototype
#   actually found mattered in this pool (direct-answering, factual
#   accuracy, format/language consistency) rather than "domain" -- this
#   pool showed little genuine domain specialization (see PROGRESS.md
#   section on label balance), so a domain-first prompt may just add noise.

INSTRUCTIONS = {
    "A_freeform": (
        "Below are several (prompt, response) pairs from the same AI model. "
        "In 3-4 sentences, describe this model's capabilities: what it does "
        "well, what it struggles with, and any distinctive patterns in how "
        "it responds."
    ),
    "B_structured_json": (
        "Below are several (prompt, response) pairs from the same AI model. "
        "Analyze them and output ONLY a JSON object with exactly these keys: "
        '"strengths" (1 sentence), "performance" (1 sentence, an overall '
        'quality level and why), "flaws" (1 sentence), "traits" (1 sentence, '
        "distinctive stylistic or behavioral patterns). No text outside the "
        "JSON object."
    ),
    "C_reliability_focused": (
        "Below are several (prompt, response) pairs from the same AI model. "
        "In 3-4 sentences, describe: (1) does it directly answer the "
        "question asked, or does it drift/refuse/list unrelated things "
        "instead; (2) how often its factual claims seem accurate vs "
        "confidently wrong; (3) whether its output format/language stays "
        "consistent and well-formed. Do not guess at its knowledge domain "
        "if the sample doesn't show one clearly."
    ),
}

BACKBONES = {
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",       # 3.8B, fast, ungated
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",  # 7B, ungated
}

MAX_PROMPTS_IN_CONTEXT = 8  # cap per-model probe block to keep backbone context short/fast

# %% Cell 5 — run every (backbone, instruction) combo over all 11 models
import gc
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def free_model_cache(hf_id):
    from huggingface_hub import scan_cache_dir
    cache = scan_cache_dir()
    hashes = [rev.commit_hash for repo in cache.repos if repo.repo_id == hf_id
              for rev in repo.revisions]
    if hashes:
        cache.delete_revisions(*hashes).execute()


def build_probe_block(model_hf_name, probe_data, max_n=MAX_PROMPTS_IN_CONTEXT):
    rows = [r for r in probe_data if model_hf_name in r["texts"]][:max_n]
    lines = []
    for i, r in enumerate(rows):
        resp = r["texts"][model_hf_name][:600]  # cap response length, avoid runaway context
        lines.append(f"[Prompt {i+1}]: {r['prompt'][:300]}\n[Response {i+1}]: {resp}")
    return "\n\n".join(lines)


def summarize_one_model(model, tokenizer, instruction, probe_block, device):
    chat = [
        {"role": "user", "content": f"{instruction}\n\n{probe_block}"},
    ]
    inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=200, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    return text.strip()


results = {}  # results[(backbone_key, instr_key)][model_hf_name] = summary text
log_path = f"{OUT_DIR}/sweep_progress.log"


def log(msg):
    print(msg)
    with open(log_path, "a") as f:
        f.write(msg + "\n")


for backbone_key, hf_id in BACKBONES.items():
    log(f"\n=== loading backbone {backbone_key} ({hf_id}) ===")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, quantization_config=bnb_config, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map="auto",
    )
    device = next(model.parameters()).device

    for instr_key, instruction in INSTRUCTIONS.items():
        combo_key = f"{backbone_key}__{instr_key}"
        out_path = f"{OUT_DIR}/summaries_{combo_key}.json"
        if os.path.exists(out_path):
            log(f"[skip] {combo_key} already done")
            results[combo_key] = json.load(open(out_path))
            continue

        summaries = {}
        for model_hf_name in ALL_11:
            probe_block = build_probe_block(model_hf_name, probe_data)
            try:
                summary = summarize_one_model(model, tokenizer, instruction, probe_block, device)
            except Exception as e:
                log(f"  [FAILED] {combo_key} / {model_hf_name}: {e}")
                summary = ""
            summaries[model_hf_name] = summary
            log(f"  [{combo_key}] {model_hf_name}: {summary[:80]!r}...")

        json.dump(summaries, open(out_path, "w"), indent=2)
        results[combo_key] = summaries
        log(f"[done] {combo_key} -> {out_path}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    free_model_cache(hf_id)

log("\nAll (backbone, instruction) combinations finished.")

# %% Cell 6 — embed each combo's summaries (MiniLM) and RSA vs capability
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr
from itertools import permutations

pool11 = ALL_11
minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def try_parse_json_summary(text):
    """For the B_structured_json variant: pull out the JSON object if the
    backbone wrapped it in extra text; fall back to raw text otherwise."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return text
    try:
        d = json.loads(match.group(0))
        return " ".join(str(v) for v in d.values())
    except Exception:
        return text


def cosine_sim_matrix(vecs, order):
    n = len(order)
    M = np.zeros((n, n))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            va, vb = vecs[a], vecs[b]
            M[i, j] = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
    return M


def upper_tri(M):
    return M[np.triu_indices(M.shape[0], k=1)]


def exact_or_mc_mantel(M_a, M_b, exact_limit=8, n_perm=20000, seed=0):
    n = M_a.shape[0]
    a_flat = upper_tri(M_a)
    rho_obs, _ = spearmanr(a_flat, upper_tri(M_b))
    if n <= exact_limit:
        vals = []
        for perm in permutations(range(n)):
            perm = np.array(perm)
            rho, _ = spearmanr(a_flat, upper_tri(M_b[np.ix_(perm, perm)]))
            vals.append(rho)
        p = float(np.mean(np.abs(np.array(vals)) >= abs(rho_obs) - 1e-12))
    else:
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        count_ge = 0
        for _ in range(n_perm):
            perm = rng.permutation(idx)
            rho, _ = spearmanr(a_flat, upper_tri(M_b[np.ix_(perm, perm)]))
            if abs(rho) >= abs(rho_obs) - 1e-12:
                count_ge += 1
        p = (count_ge + 1) / (n_perm + 1)
    return rho_obs, p


cap_vecs = {m: np.load(f"local_descriptors/mix-instruct-capability-11/{m}.npy") for m in pool11}
M_cap = cosine_sim_matrix(cap_vecs, pool11)

summary_table = []
for combo_key, summaries in results.items():
    texts = [try_parse_json_summary(summaries.get(m, "")) or "(empty)" for m in pool11]
    emb = minilm.encode(texts, normalize_embeddings=True)
    v12_vecs = {m: emb[i] for i, m in enumerate(pool11)}
    M_v12 = cosine_sim_matrix(v12_vecs, pool11)
    rho, p = exact_or_mc_mantel(M_v12, M_cap)
    summary_table.append((combo_key, rho, p))
    print(f"{combo_key:35s} rho={rho:+.4f}  p={p:.4f}  {'SIGNIFICANT' if p < 0.05 else 'not significant'}")

summary_table.sort(key=lambda row: -row[1])
print("\n=== ranked by rho (best-aligned-with-capability first) ===")
for combo_key, rho, p in summary_table:
    print(f"{combo_key:35s} rho={rho:+.4f}  p={p:.4f}")

with open(f"{OUT_DIR}/sweep_results.json", "w") as f:
    json.dump([{"combo": k, "rho": r, "p": p} for k, r, p in summary_table], f, indent=2)
print(f"\nsaved -> {OUT_DIR}/sweep_results.json")
