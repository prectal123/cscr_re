#!/usr/bin/env bash
# One-shot setup + experiment runner for a fresh GPU server.
#
# Usage (after `git clone` and `cd` into the repo):
#   bash scripts/setup_and_run_gpu_server.sh
#
# What this does, in order:
#   1. Creates a Python venv and installs dependencies.
#   2. Checks GPU VRAM and decides bf16 vs 4-bit quantization automatically.
#   3. Computes the logit descriptor for moss-moon-003-sft (16B, blocked
#      locally by VRAM only) and baize (blocked by a missing tokenizer file
#      in its HF repo - the code has a Llama-2-tokenizer fallback that MAY
#      recover it, not guaranteed) using the existing 192-probe set.
#
#   NOTE: dolly-v2-12b and mpt-7b-instruct are NOT attempted here - their
#   HF Hub repos are gone entirely (verified 2026-07-25, 401/Repository Not
#   Found). No amount of GPU/VRAM fixes a repo that no longer exists; see
#   PROGRESS.md section 6/14. Realistic ceiling for the logit-descriptor
#   pool is 9 models (current 7 + moss-moon-003-sft, +baize if the
#   tokenizer fallback works), not the full 11.
#
#   Perplexity descriptors for all 11 (including dolly/mpt) are already in
#   the repo - those never needed live model access, just the response
#   text already stored in the MixInstruct dataset. See PROGRESS.md #13.
#
#   Safe to re-run: already-completed steps and already-computed
#   descriptors are skipped.

set -uo pipefail
cd "$(dirname "$0")/.."   # repo root
REPO_ROOT="$(pwd)"

echo "=============================================="
echo " CSCR reimplementation - GPU server setup"
echo " Repo root: $REPO_ROOT"
echo "=============================================="

# ---------------------------------------------------------------------
# 1. Python venv + dependencies
# ---------------------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating venv (.venv) ..."
    python3 -m venv .venv
else
    echo "[1/4] .venv already exists, reusing it."
fi
source .venv/bin/activate

echo "[1/4] Installing dependencies (this can take a few minutes the first time) ..."
pip install --quiet --upgrade pip
pip install --quiet -e .
pip install --quiet bitsandbytes sentencepiece protobuf huggingface_hub
echo "[1/4] Done. Python: $(python --version), pip packages installed."

if [ -n "${HF_TOKEN:-}" ]; then
    echo "[1/4] HF_TOKEN is set - HuggingFace downloads will use it."
else
    echo "[1/4] NOTE: HF_TOKEN is not set. Downloads will still work but may be"
    echo "      rate-limited. To set one: export HF_TOKEN=hf_xxxxxxxx"
fi

# ---------------------------------------------------------------------
# 2. GPU / VRAM check
# ---------------------------------------------------------------------
echo ""
echo "[2/4] Checking GPU ..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Is this actually a GPU machine / are drivers installed?"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
echo "[2/4] Detected ${VRAM_MB} MiB total VRAM on GPU 0."

if [ "$VRAM_MB" -ge 40000 ]; then
    FOUR_BIT_FLAG=""
    echo "[2/4] >=40GB VRAM: loading all 4 remaining models in bf16 (no quantization needed)."
else
    FOUR_BIT_FLAG="--four_bit"
    echo "[2/4] <40GB VRAM: using 4-bit NF4 quantization for all 4 remaining models to be safe."
fi

# ---------------------------------------------------------------------
# 3. Check torch/CUDA works before committing to long downloads
# ---------------------------------------------------------------------
echo ""
echo "[3/4] Verifying torch sees the GPU ..."
python -c "
import torch
assert torch.cuda.is_available(), 'torch does not see a CUDA GPU - check the torch install matches this server CUDA version.'
print(f'  torch {torch.__version__}, CUDA available, device: {torch.cuda.get_device_name(0)}')
"
if [ $? -ne 0 ]; then
    echo "ERROR: torch/CUDA check failed. This usually means the pip-installed torch"
    echo "       build doesn't match this server's CUDA driver. Check with:"
    echo "         nvidia-smi   (driver's max supported CUDA version)"
    echo "       then install a matching torch build, e.g.:"
    echo "         pip install torch --index-url https://download.pytorch.org/whl/cu124"
    exit 1
fi

# ---------------------------------------------------------------------
# 4. Compute logit descriptors for the 4 missing MixInstruct models
# ---------------------------------------------------------------------
echo ""
echo "[4/4] Computing logit descriptors for the 4 models we couldn't run locally ..."

PROBES="local_data/probes_mix-instruct-192.json"
OUT_DIR="local_descriptors/mix-instruct-logit"
mkdir -p "$OUT_DIR"

MODELS=(
    "fnlp/moss-moon-003-sft"
    "mosesjun0h/llama-7b-hf-baize-lora-bf16"
)

FAILED=()
for MODEL in "${MODELS[@]}"; do
    OUT_NAME="${MODEL//\//__}"
    OUT_PATH="$OUT_DIR/${OUT_NAME}.npy"
    if [ -f "$OUT_PATH" ]; then
        echo "  [skip] $MODEL -> $OUT_PATH already exists."
        continue
    fi
    echo "  [running] $MODEL (this downloads the model first, can take a while) ..."
    if python scripts/compute_descriptors.py \
        --model "$MODEL" \
        --probes_files "$PROBES" \
        --out "$OUT_PATH" \
        --topk 192 --n_tokens 10 \
        $FOUR_BIT_FLAG; then
        echo "  [done] $MODEL -> $OUT_PATH"
    else
        echo "  [FAILED] $MODEL - see error above."
        FAILED+=("$MODEL")
    fi
done

echo ""
echo "=============================================="
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All done. Logit descriptors now cover up to 9/11 MixInstruct models"
    echo "(7 original + moss-moon-003-sft + baize) in:"
    echo "  $OUT_DIR/"
    echo "(dolly-v2-12b and mpt-7b-instruct are permanently unreachable - see"
    echo " the note at the top of this script / PROGRESS.md section 14.)"
else
    echo "Finished with ${#FAILED[@]} failure(s):"
    for M in "${FAILED[@]}"; do echo "  - $M"; done
    echo "Re-running this script will retry only the failed/missing ones."
fi
echo ""
echo "Next steps (not run automatically - these are judgment calls):"
echo "  - Decide whether to expand experts/pool-mix-instruct-*.json to 11 models"
echo "    and rebuild the FAISS index (scripts/build_faiss.py) before retraining."
echo "  - See PROGRESS.md section 13 for the multi-seed n_bands follow-up."
echo "=============================================="
