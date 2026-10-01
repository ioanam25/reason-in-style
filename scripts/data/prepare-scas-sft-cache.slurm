#!/usr/bin/env bash
#SBATCH --job-name=scas-sft-tok
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=cpu  # set to your site partition
#SBATCH --time=4:00:00
#
# Pre-tokenize SCAS SFT parquet once (chat template + truncate) before DDP training.
# Usage: sbatch scripts/data/prepare-scas-sft-cache.slurm [qwen3-4b]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
source ./.venv/bin/activate

SIZE="${1:-qwen3-4b}"
source scripts/scas_model_size.sh "${SIZE}"

PY=./.venv/bin/python
mkdir -p slurm_logs

if [ ! -f "${STANDARD_SFT_DIR}/train.parquet" ]; then
  echo "Missing ${STANDARD_SFT_DIR}/train.parquet"
  exit 1
fi

echo "Tokenizing ${STANDARD_SFT_DIR} (max_seq_length=${SCAS_MAX_LEN})"
"${PY}" scripts/prepare_sft_hf_cache.py "${STANDARD_SFT_DIR}" \
  --tokenizer "${DECODER_MODEL}" \
  --max-seq-length "${SCAS_MAX_LEN}" \
  --num-proc 32 \
  --force

echo "Done."
