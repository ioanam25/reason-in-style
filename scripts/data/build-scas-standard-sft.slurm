#!/usr/bin/env bash
#SBATCH --job-name=scas-sft-build
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu  # set to your site partition
#SBATCH --time=4:00:00
#
# Build SCAS vanilla SFT parquet from HuggingFace.
# Usage: sbatch scripts/data/build-scas-standard-sft.slurm [qwen3-4b]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"

SIZE="${1:-qwen3-4b}"
source scripts/scas_model_size.sh "${SIZE}"

VENV_PY="./.venv/bin/python"
mkdir -p slurm_logs "${STANDARD_SFT_DIR}"

echo "Host: $(hostname)"
echo "HF dataset: ${SCAS_HF_DATASET}"
echo "Standard out: ${STANDARD_SFT_DIR}"

"${VENV_PY}" scripts/data/build_scas_standard_sft.py \
  --config all \
  --output-dir "${STANDARD_SFT_DIR}" \
  --json-out "${SCAS_SFT_JSON}" \
  --tokenizer "${DECODER_MODEL}" \
  --verify-chat-tokens 0

echo "Done."
