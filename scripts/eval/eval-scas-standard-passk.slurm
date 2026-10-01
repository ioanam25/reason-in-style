#!/usr/bin/env bash
#SBATCH --job-name=scas-eval-std
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu  # set to your site partition
#SBATCH --qos=normal  # set to your site QoS (or remove)
#SBATCH --time=24:00:00
#
# SCAS validation Pass@k for standard (question-only) decoder SFT.
# Usage:
#   sbatch scripts/eval/eval-scas-standard-passk.slurm
#   sbatch scripts/eval/eval-scas-standard-passk.slurm qwen3-4b checkpoint-18748

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"

SIZE="${1:-qwen3-4b}"
CKPT_STEP="${2:-}"
source scripts/scas_model_size.sh "${SIZE}"

if [ -z "${CKPT_STEP}" ]; then
  CKPT_STEP="$(ls -d "${STANDARD_CKPT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 | xargs basename)"
fi

STANDARD_CKPT="${STANDARD_CKPT_DIR}/${CKPT_STEP}"
CKPT_TAG="scas-standard-${MODEL_SIZE}_${CKPT_STEP}"
OUT_DIR="model-evals/${CKPT_TAG}-scas-val-passk"

VENV_PY="./.venv/bin/python"
mkdir -p slurm_logs

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${HF_CACHE}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

echo "Host: $(hostname)"
echo "Checkpoint: ${STANDARD_CKPT}"
echo "Test: ${SCAS_VAL_PARQUET}"
nvidia-smi -L || true

if [ ! -d "${STANDARD_CKPT}" ]; then
  echo "Missing checkpoint: ${STANDARD_CKPT}"
  exit 1
fi

if [[ -f "${OUT_DIR}/pass_at_k.json" ]]; then
  echo "=== SKIP (already done): ${OUT_DIR}/pass_at_k.json ==="
  exit 0
fi

echo "=== SCAS validation standard (question-only) ==="
"${VENV_PY}" scripts/eval/eval_math_standard_passk_qwen.py \
  --checkpoint "${STANDARD_CKPT}" \
  --test-parquet "${SCAS_VAL_PARQUET}" \
  --output-dir "${OUT_DIR}" \
  --samples-per-problem 256 \
  --tensor-parallel-size 1 \
  --data-parallel-replicas -1 \
  --max-tokens "${SCAS_MAX_LEN}" \
  --temperature 1.0 \
  --top-p 1.0 \
  --budgets 1 2 4 8 16 32 64 128 256 \
  --grade-timeout 2.0 \
  --grade-flush-every 100

echo ""
echo "Done: ${OUT_DIR}/pass_at_k.json"
