#!/usr/bin/env bash
# One-shot setup + EmbedLLM descriptor batch runner for a multi-3090 GPU server.
#
# Usage (after `git clone` and `cd` into the repo):
#   bash scripts/run_embedllm_batch.sh
#
# What this does, in order:
#   1. Creates/reuses a venv and installs dependencies (same as
#      setup_and_run_gpu_server.sh).
#   2. Detects GPU count via nvidia-smi and picks the matching pool file:
#        2 GPUs  -> experts/pool-embedllm-3090x2.json  (105 models)
#        4+ GPUs -> experts/pool-embedllm-3090x4.json  (115 models, superset)
#      Both pool files were derived from experts/registry-embedllm.json by
#      estimating 4-bit NF4 VRAM per model (~0.6 GB/param + ~2.5GB overhead)
#      and keeping only models that fit under (n_gpus * 24GB - 2GB/gpu
#      headroom). This is an ESTIMATE calibrated against this project's own
#      measured 4-bit footprints (see PROGRESS.md section 10), not a
#      per-model hardware test -- if a specific model still OOMs, that's a
#      single failed entry the batch script logs and skips, not a crash.
#   3. Generates the EmbedLLM probe set (192 probes) if not already present.
#   4. Runs scripts/compute_embedllm_descriptors_batch.py over the chosen
#      pool: for each model, downloads it, computes BOTH the logit and
#      perplexity descriptor in the same pass (see that script's docstring
#      for why: unlike MixInstruct, EmbedLLM has no pre-stored response text
#      to score for free), saves both, deletes the HF cache, moves on.
#
#   NOTE: this does NOT touch MixInstruct's moss-moon-003-sft / baize --
#   those are still handled by the existing scripts/setup_and_run_gpu_server.sh
#   (unchanged). Run that separately if you want both pools filled in.
#
#   Safe to re-run: already-completed models (both descriptors present) are
#   skipped, and failed models leave their partial HF cache in place for a
#   faster retry rather than triggering a redownload.

set -uo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

echo "=============================================="
echo " CSCR reimplementation - EmbedLLM batch runner"
echo " Repo root: $REPO_ROOT"
echo "=============================================="

# ---------------------------------------------------------------------
# 1. Python venv + dependencies
# ---------------------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "[1/5] Creating venv (.venv) ..."
    python3 -m venv .venv
else
    echo "[1/5] .venv already exists, reusing it."
fi
source .venv/bin/activate

echo "[1/5] Installing dependencies ..."
pip install --quiet --upgrade pip
pip install --quiet -e .
pip install --quiet bitsandbytes sentencepiece protobuf huggingface_hub

if [ -n "${HF_TOKEN:-}" ]; then
    echo "[1/5] HF_TOKEN is set."
else
    echo "[1/5] NOTE: HF_TOKEN is not set. Gated repos (meta-llama__* entries"
    echo "      in the pool) will fail and be skipped/logged unless you export"
    echo "      HF_TOKEN=hf_xxxx for an account that has accepted their license."
fi

# ---------------------------------------------------------------------
# 2. GPU detection -> pick pool file
# ---------------------------------------------------------------------
echo ""
echo "[2/5] Checking GPUs ..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found."
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
echo "[2/5] Detected ${N_GPUS} GPU(s)."

if [ "$N_GPUS" -ge 4 ]; then
    POOL_FILE="experts/pool-embedllm-3090x4.json"
    echo "[2/5] >=4 GPUs: using $POOL_FILE (full 115-model pool)."
elif [ "$N_GPUS" -ge 2 ]; then
    POOL_FILE="experts/pool-embedllm-3090x2.json"
    echo "[2/5] 2-3 GPUs: using $POOL_FILE (105-model pool)."
else
    echo "ERROR: this pool was sized for 2-4x RTX 3090 (24GB each)."
    echo "       Only $N_GPUS GPU(s) detected -- the 2x3090 pool (105 models,"
    echo "       up to ~40B params under 4-bit) will not reliably fit on a"
    echo "       single card. Re-derive a 1-GPU pool from"
    echo "       experts/registry-embedllm.json before proceeding, or attach"
    echo "       more GPUs."
    exit 1
fi

# ---------------------------------------------------------------------
# 3. Probes
# ---------------------------------------------------------------------
echo ""
echo "[3/5] Checking probe set ..."
PROBES="local_data/probes_embedllm-192.json"
if [ -f "$PROBES" ]; then
    echo "[3/5] $PROBES already exists, reusing it."
else
    echo "[3/5] Generating 192 EmbedLLM probes ..."
    python scripts/generate_probes.py --n_embedllm 192 --seed 42 --out_dir local_data
fi

# ---------------------------------------------------------------------
# 4. Torch/CUDA sanity check
# ---------------------------------------------------------------------
echo ""
echo "[4/5] Verifying torch sees the GPU(s) ..."
python -c "
import torch
assert torch.cuda.is_available(), 'torch does not see a CUDA GPU - check the torch install matches this server CUDA version.'
print(f'  torch {torch.__version__}, {torch.cuda.device_count()} device(s) visible')
"
if [ $? -ne 0 ]; then
    echo "ERROR: torch/CUDA check failed. See scripts/setup_and_run_gpu_server.sh"
    echo "       for the matching-CUDA-build troubleshooting note."
    exit 1
fi

# ---------------------------------------------------------------------
# 5. Run the batch
# ---------------------------------------------------------------------
echo ""
echo "[5/5] Running EmbedLLM descriptor batch over $POOL_FILE ..."
echo "      (this can take a long time -- see PROGRESS.md for disk/time"
echo "       estimates. Safe to Ctrl-C and re-run; completed models skip.)"

python scripts/compute_embedllm_descriptors_batch.py \
    --pool "$POOL_FILE" \
    --probes "$PROBES" \
    --out_logit_dir local_descriptors/embedllm-logit \
    --out_perplexity_dir local_descriptors/embedllm-perplexity \
    --topk 192 --n_tokens 10 --batch_size 8 \
    --progress_log local_descriptors/embedllm_batch_progress.log \
    --min_disk_headroom_gb 15

echo ""
echo "=============================================="
echo "Done (or stopped early -- check the summary line above and"
echo "local_descriptors/embedllm_batch_progress.log for per-model detail)."
echo "Next steps (not automatic, judgment calls):"
echo "  - Review the progress log for [FAILED] entries (gated repos, OOM, etc.)"
echo "  - Build an experts/pool-embedllm-<final-N>.json from whichever models"
echo "    actually succeeded before rebuilding the FAISS index / retraining."
echo "=============================================="
