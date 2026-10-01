#!/usr/bin/env bash
#SBATCH --job-name=scas-extract-z
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu  # set to your site partition
#SBATCH --qos=normal  # set to your site QoS (or remove)
#SBATCH --time=12:00:00
#
# Extract z(s) for SCAS trace-only AE train split (8 GPU shards + merge).
# Usage:
#   sbatch scripts/discovery/extract-z-embeddings-scas.slurm
#   sbatch scripts/discovery/extract-z-embeddings-scas.slurm qwen3-4b /path/to.ckpt

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"

SIZE="${1:-qwen3-4b}"
CHECKPOINT_OVERRIDE="${2:-}"
NUM_SHARDS="${3:-8}"
source scripts/scas_model_size.sh "${SIZE}"

VENV_PY="./.venv/bin/python"
OUT_DIR="${GMM_ASSIGN_DIR}"
SHARD_DIR="${OUT_DIR}/embedding_shards"
mkdir -p slurm_logs "${OUT_DIR}" "${SHARD_DIR}"

TRACE_CKPT="${CHECKPOINT_OVERRIDE:-${TRACE_CKPT_DIR}/last.ckpt}"
if [[ ! -f "${TRACE_CKPT}" ]]; then
  echo "Missing checkpoint: ${TRACE_CKPT}"
  exit 1
fi

echo "Host: $(hostname)"
echo "Checkpoint: ${TRACE_CKPT}"
echo "Dataset: ${TRACE_DATA_PATH}"
echo "Output: ${OUT_DIR}"
nvidia-smi -L || true

pids=()
for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
  echo "Starting shard ${SHARD_ID}/${NUM_SHARDS} on GPU ${SHARD_ID}"
  CUDA_VISIBLE_DEVICES="${SHARD_ID}" "${VENV_PY}" -u scripts/discovery/extract_z_embeddings.py \
    --checkpoint "${TRACE_CKPT}" \
    --hf-dataset "${TRACE_DATA_PATH}" \
    --raw-json "${SCAS_TRACE_JSON}" \
    --output-dir "${OUT_DIR}" \
    --shard-id "${SHARD_ID}" \
    --num-shards "${NUM_SHARDS}" \
    --batch-size 8 \
    --device cuda \
    > "slurm_logs/scas-extract-z-shard${SHARD_ID}-${SLURM_JOB_ID}.log" \
    2>&1 &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    echo "Shard process ${pid} failed"
    fail=1
  fi
done
if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi

echo "Merging shards..."
"${VENV_PY}" -u scripts/discovery/merge_z_embedding_shards.py --output-dir "${OUT_DIR}"
echo "Done: ${OUT_DIR}/embeddings_train_z.npz"
