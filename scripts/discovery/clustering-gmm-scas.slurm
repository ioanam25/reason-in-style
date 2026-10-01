#!/usr/bin/env bash
#SBATCH --job-name=scas-gmm-z
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:0
#SBATCH --partition=cpu  # set to your site partition
#SBATCH --time=4:00:00
#
# GMM sweep K=2..10 on SCAS z embeddings (requires merged embeddings_train_z.npz).
# Usage:
#   sbatch scripts/discovery/clustering-gmm-scas.slurm
#   sbatch scripts/discovery/clustering-gmm-scas.slurm qwen3-4b "" --gmm-only

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"

SIZE="${1:-qwen3-4b}"
CHECKPOINT_OVERRIDE="${2:-}"
GMM_ONLY="${3:-}"
source scripts/scas_model_size.sh "${SIZE}"

VENV_PY="./.venv/bin/python"
OUT_DIR="${GMM_ASSIGN_DIR}"
mkdir -p slurm_logs "${OUT_DIR}"

TRACE_CKPT="${CHECKPOINT_OVERRIDE:-${TRACE_CKPT_DIR}/last.ckpt}"

echo "Host: $(hostname)"
echo "Checkpoint: ${TRACE_CKPT}"
echo "Dataset: ${TRACE_DATA_PATH}"
echo "Output: ${OUT_DIR}"

EXTRA=()
if [[ "${GMM_ONLY}" == "--gmm-only" ]]; then
  EXTRA+=(--gmm-only)
fi

"${VENV_PY}" -u scripts/discovery/clustering_gmm.py \
  --checkpoint "${TRACE_CKPT}" \
  --hf-dataset "${TRACE_DATA_PATH}" \
  --raw-json "${SCAS_TRACE_JSON}" \
  --output-dir "${OUT_DIR}" \
  --min-clusters 2 \
  --max-clusters 10 \
  --skip-val-split \
  "${EXTRA[@]}"

echo "Done: ${OUT_DIR}/cluster_meta.json"
