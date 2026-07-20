# %% [markdown]
# # CSCR small-scale reproduction — MixInstruct pool
#
# Paste each `# %%` block below into its own Colab cell, in order.
#
# Mirrors the original repo's `compute_descriptors.py` (logit) and
# `compute_descriptors_perplexity.py` (perplexity), scaled down to the
# 11-model MixInstruct pool (the only benchmark where the paper computes
# both descriptor types — see Table 4 / Section 4.3.1).
#
# IMPORTANT — read before running:
# 4-bit quantization (bitsandbytes) reduces VRAM usage, NOT download size.
# `from_pretrained(..., load_in_4bit=True)` still pulls the full fp16/bf16
# checkpoint from the Hub and quantizes it on load. All 11 MixInstruct
# models at native precision total roughly 200GB+ (vicuna-13b ~26GB,
# dolly-v2-12b ~24GB, oasst-pythia-12b ~24GB, flan-t5-xxl ~44GB,
# moss-moon-003-sft ~32GB, the rest ~13-14GB each). To avoid filling
# Colab's disk, this script downloads ONE model at a time, computes its
# descriptor, then deletes that model's local HF cache before moving to
# the next. Only the tiny descriptor `.npy` outputs (KB-MB range) are
# persisted to Drive — raw model weights are treated as scratch.
#
# This has NOT been run end-to-end in a live Colab session — treat it as
# a first draft. A few of these 11 repos use older custom modeling code
# (chatglm-6b, moss-moon-003-sft) and may need per-model debugging; the
# loop is written to skip failures and keep going rather than crash.

# %% Cell 1 — mount Drive, set up persistent OUTPUT paths (not model cache)
from google.colab import drive
drive.mount('/content/drive')

import os

DRIVE_ROOT = "/content/drive/MyDrive/cscr_repro"
DATA_DIR = f"{DRIVE_ROOT}/data"
DESC_LOGIT_DIR = f"{DRIVE_ROOT}/experts/descriptors/mix-instruct-logit"
DESC_PPL_DIR = f"{DRIVE_ROOT}/experts/descriptors/mix-instruct-perplexity"
for d in (DATA_DIR, DESC_LOGIT_DIR, DESC_PPL_DIR):
    os.makedirs(d, exist_ok=True)

# Model weight cache stays LOCAL to the Colab VM (ephemeral) and gets
# cleared per-model right after use — do NOT point this at Drive or you
# will blow through your Drive quota downloading ~200GB of checkpoints.
os.environ["HF_HOME"] = "/content/hf_cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# %% Cell 2 — clone repo + install deps
# Cloning the fork (prectal123/cscr_re), not the original repo — it already
# has the padding-bug fix (and, once applied, the perplexity NaN fix) baked
# in, so Cell 2b's monkey-patch below is now just a harmless no-op safety
# net rather than doing the real work.
get_ipython().system('git clone -q https://github.com/prectal123/cscr_re.git /content/cscr')
get_ipython().run_line_magic('cd', '/content/cscr')
get_ipython().system('pip install -e . -q')
get_ipython().system('pip install -q bitsandbytes accelerate')
# sentencepiece: required to load/convert LLaMA-family (SentencePiece-based)
# tokenizers, e.g. the baize model in this pool. Not always present on a
# fresh Colab VM and not pulled in automatically by pyproject.toml.
get_ipython().system('pip install -q sentencepiece protobuf')

import sys
sys.path.insert(0, "/content/cscr/src")  # safety net in case the editable install doesn't register `router`

# %% Cell 2b — patch a real correctness bug in the cloned repo
# router/descriptors.py sets tokenizer.pad_token but never tokenizer.padding_side.
# Most tokenizers default to right-padding, which silently corrupts batched
# causal-LM generate() calls: for any prompt shorter than the longest one in
# its batch, the model starts generating from a PAD position instead of the
# real last token, producing garbage scores for that prompt. Cell 8 uses
# batch_size=4, so this affects most probes in most batches. Left-padding
# fixes it — this patch must run BEFORE `from router.descriptors import
# save_descriptors` is first executed (i.e. before Cell 8 runs).
_desc_path = "/content/cscr/src/router/descriptors.py"
with open(_desc_path) as _f:
    _content = _f.read()
_old = "tokenizer.pad_token = tokenizer.eos_token \n"
_new = "tokenizer.pad_token = tokenizer.eos_token\n    tokenizer.padding_side = \"left\"  # patched: fixes right-padding bug\n"
if _old in _content and "padding_side" not in _content:
    with open(_desc_path, "w") as _f:
        _f.write(_content.replace(_old, _new))
    print("patched descriptors.py: added tokenizer.padding_side = 'left'")
elif "padding_side" in _content:
    print("descriptors.py already patched")
else:
    print("WARNING: expected string not found — patch not applied, check descriptors.py manually")

# %% Cell 3 — HF login (usually NOT needed)
# None of the 11 MixInstruct models are gated as of writing. Only uncomment
# this if you hit a 401/403 on one of them.
# from huggingface_hub import login
# login()

# %% Cell 4 — [STEP 1] asset list: MixInstruct's fixed 11-model pool
# Kept identical to router.mix_instruct.MixInstructOracle.NAME_TO_HF so the
# existing prompt/candidate/bartscore labels already in the mix-instruct
# HF dataset stay usable without any extra labeling work.
MIXINSTRUCT_POOL = {
    "eachadea__vicuna-13b-1.1":                        "eachadea/vicuna-13b-1.1",
    "chavinlo__alpaca-native":                         "chavinlo/alpaca-native",
    "stabilityai__stablelm-tuned-alpha-7b":            "stabilityai/stablelm-tuned-alpha-7b",
    "OpenAssistant__oasst-sft-4-pythia-12b-epoch-3.5": "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
    "TheBloke__koala-7B-HF":                           "TheBloke/koala-7B-HF",
    "mosesjun0h__llama-7b-hf-baize-lora-bf16":         "mosesjun0h/llama-7b-hf-baize-lora-bf16",
    "google__flan-t5-xxl":                             "google/flan-t5-xxl",
}
# Dropped from the original 11, all confirmed via actual load/download
# attempts in Colab (not just guessed):
# - databricks__dolly-v2-12b, mosaicml__mpt-7b-instruct: identical "not a
#   valid model identifier" failure on repeated attempts, and their HF pages
#   don't load at all (even via WebFetch) — these two repos look removed/
#   restricted from the Hub, not gated. Community re-uploads exist for dolly
#   (e.g. RichardErkhov/databricks_-_dolly-v2-12b-*bits) but are pre-quantized
#   in a different format (likely GGUF), not a clean fit for this pipeline's
#   uniform on-the-fly bnb 4-bit loading.
# - THUDM__chatglm-6b: fails with "'property' object cannot be interpreted
#   as an integer" — its 2023-era trust_remote_code is incompatible with the
#   transformers version installed here.
# - fnlp__moss-moon-003-sft: fails with "cannot import name 'is_tf_available'
#   from 'transformers.utils'" — same kind of version-compatibility issue.
# Both could theoretically be fixed by pinning an older transformers version,
# but that risks breaking the other models that are already working, so not
# worth it for a 3-week-deadline reproduction. 7 models is still plenty for
# the vector-diversity analysis.

# %% Cell 5 — quantized model loader
# router.utils.load_model_and_tokenizer has no quantization support, so
# this is a small variant of it with a BitsAndBytesConfig added.
import torch
from transformers import (
    AutoModelForCausalLM, AutoModel, AutoTokenizer,
    T5ForConditionalGeneration, T5Tokenizer, BitsAndBytesConfig,
)

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def load_quantized(model_name: str):
    # low_cpu_mem_usage=True matters a lot here: several repos in this pool
    # (e.g. stablelm-tuned-alpha-7b) only ship old-style `.bin` checkpoints,
    # not `.safetensors`. `.bin` has to be fully deserialized on CPU before
    # dispatch/quantization, which can spike *system RAM* (not GPU VRAM) well
    # past Colab's free-tier ~12GB and crash the runtime, even though the
    # model comfortably fits on the GPU once quantized. This flag streams
    # the load instead of materializing the whole checkpoint in RAM at once.
    #
    # device_map={"": 0} (rather than "auto"): accelerate's automatic device
    # planner can decide a model needs CPU/disk offload based on its
    # *pre-quantization* size, even when the actual 4-bit result would fit
    # on the GPU alone — and bitsandbytes 4-bit doesn't support mixing
    # GPU-quantized layers with CPU/disk-offloaded ones by default, which
    # raises "Some modules are dispatched on the CPU or the disk". Every
    # model kept in this pool is small enough post-quantization (~4-8GB) to
    # just force it entirely onto GPU 0 and skip that planning step.
    if "flan-t5" in model_name:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(
            model_name, quantization_config=BNB_CONFIG, device_map={"": 0},
            low_cpu_mem_usage=True,
        )
    elif "chatglm" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name, quantization_config=BNB_CONFIG, device_map={"": 0},
            trust_remote_code=True, low_cpu_mem_usage=True,
        )
    elif "moss-moon" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, revision="refs/pr/6",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=BNB_CONFIG, device_map={"": 0},
            trust_remote_code=True, low_cpu_mem_usage=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=BNB_CONFIG, device_map={"": 0},
            trust_remote_code=True, low_cpu_mem_usage=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def free_model_cache(hf_id: str):
    """Delete this model's local HF cache snapshot so disk usage stays bounded."""
    from huggingface_hub import scan_cache_dir
    cache = scan_cache_dir()
    hashes = [
        rev.commit_hash
        for repo in cache.repos
        if repo.repo_id == hf_id
        for rev in repo.revisions
    ]
    if hashes:
        cache.delete_revisions(*hashes).execute()

# %% Cell 6 — sanity check with ONE model before looping over all 11
_test_id = MIXINSTRUCT_POOL["stabilityai__stablelm-tuned-alpha-7b"]
model, tok = load_quantized(_test_id)
print(f"loaded OK — memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
del model, tok
torch.cuda.empty_cache()
free_model_cache(_test_id)

# %% Cell 7 — [STEP 2] probe prompts (list + actual text)
# Start small to validate the full pipeline runs cleanly, then bump N up
# to ~150-192 (closer to the paper's default of 192) once the loop below
# has run without errors.
N_PROBES = 32
get_ipython().system(
    f'python scripts/generate_probes.py --n_mix-instruct {N_PROBES} --seed 42 --out_dir {DATA_DIR}'
)

import json

probes_path = f"{DATA_DIR}/probes_mix-instruct-{N_PROBES}.json"
probes = json.load(open(probes_path))
print(f"{len(probes)} probes loaded, e.g.: {probes[0]}")

# %% Cell 8 — [STEP 3] Logit descriptor — one model at a time, then free disk
from router.descriptors import save_descriptors
import time

TOPK = 256      # matches the paper's default; lower (e.g. 64) for a faster first pass
N_TOKENS = 10   # matches the paper's default

# Persistent, append-as-you-go log on Drive. Print statements alone only
# live in the cell's output and are lost if the process is killed outright
# (e.g. an OOM kill, unlike a caught Python exception) rather than cleanly
# disconnected. Writing a line to Drive right when each event happens means
# even a hard crash leaves a record of exactly which model was in progress.
LOG_PATH = f"{DRIVE_ROOT}/logit_progress.log"


def log(msg: str):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


failed = []
for label, hf_id in MIXINSTRUCT_POOL.items():
    out_path = f"{DESC_LOGIT_DIR}/{label}"
    if os.path.exists(out_path + ".npy"):
        log(f"[skip] {label} already computed")
        continue

    log(f"[start] {label} ({hf_id}) ...")
    success = False
    try:
        model, tok = load_quantized(hf_id)
        save_descriptors(
            model, tok, [probes_path], out_path,
            topk=TOPK, n_tokens=N_TOKENS, batch_size=4,
        )
        log(f"[done] {label}")
        success = True
    except Exception as e:
        log(f"[FAILED] {label}: {e}")
        failed.append((label, str(e)))
    finally:
        try:
            del model, tok
        except NameError:
            pass
        torch.cuda.empty_cache()
        # Only clear the downloaded checkpoint on success. If the failure
        # happened after the (often 30+ minute) download finished — e.g. a
        # device_map/config issue during load — the cache is left in place
        # so a retry with fixed code doesn't have to re-download the model.
        if success:
            free_model_cache(hf_id)
        else:
            log(f"  (leaving {hf_id} cache on disk for retry)")

log(f"Loop finished. Failed: {failed}")

# %% Cell 9 — [STEP 4] Perplexity descriptor
# Cheap: only needs GPT2 (as the fixed judge) plus the candidate answers
# already stored in the mix-instruct HF dataset. Does NOT need any of the
# 11 big models downloaded again — this step is independent of Cell 8.
get_ipython().system(
    f'python scripts/compute_descriptors_perplexity.py '
    f'--probe_ids {probes_path} --dataset mix-instruct '
    f'--out {DESC_PPL_DIR} --plot'
)

# %% Cell 10 — sanity check both outputs
import numpy as np

for d, kind in [(DESC_LOGIT_DIR, "logit"), (DESC_PPL_DIR, "perplexity")]:
    files = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
    shape = np.load(os.path.join(d, files[0])).shape if files else "none"
    print(f"{kind}: {len(files)} descriptors computed, e.g. shape {shape}")
