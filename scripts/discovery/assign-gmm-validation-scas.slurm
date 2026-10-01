#!/usr/bin/env bash
#SBATCH --job-name=scas-gmm-val
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu  # set to your site partition
#SBATCH --time=4:00:00
#
# Assign train-fitted GMM labels to SCAS validation split.
# Usage: sbatch scripts/discovery/assign-gmm-validation-scas.slurm [qwen3-4b] [checkpoint.ckpt]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"

SIZE="${1:-qwen3-4b}"
CHECKPOINT="${2:-}"
source scripts/scas_model_size.sh "${SIZE}"

VENV_PY="./.venv/bin/python"
mkdir -p slurm_logs

TRACE_CKPT="${CHECKPOINT:-${TRACE_CKPT_DIR}/last.ckpt}"
if [ ! -f "${TRACE_CKPT}" ]; then
  echo "Missing checkpoint: ${TRACE_CKPT}"
  exit 1
fi
if [ ! -f "${GMM_ASSIGN_DIR}/cluster_meta.json" ]; then
  echo "Missing GMM outputs in ${GMM_ASSIGN_DIR}"
  exit 1
fi

echo "Host: $(hostname)"
echo "GMM dir: ${GMM_ASSIGN_DIR}"
echo "Trace ckpt: ${TRACE_CKPT}"
nvidia-smi -L || true

"${VENV_PY}" -u scripts/discovery/assign_gmm_validation_split.py \
  --assignments-dir "${GMM_ASSIGN_DIR}" \
  --checkpoint "${TRACE_CKPT}" \
  --hf-dataset "${TRACE_DATA_PATH}"

echo "Done: ${GMM_ASSIGN_DIR}/validation.parquet"
