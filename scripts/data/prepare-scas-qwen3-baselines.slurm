#!/usr/bin/env bash
#SBATCH --job-name=scas-q3-base-tok
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=cpu  # set to your site partition
#SBATCH --time=12:00:00
#
# Build ModC [style_1..9] parquet + tokenize vanilla + ModC with Qwen3-0.6B.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
source ./.venv/bin/activate
source scripts/scas_model_size.sh qwen3-0p6b
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
PY=./.venv/bin/python
SROOT="${ARCHIVE_ROOT}"
STD_OUT="${SROOT}/baselines/standard-qwen3"
MODC_OUT="${SROOT}/baselines/modc-style-qwen3"

echo "=== vanilla: link parquet + tokenize with ${DECODER_MODEL} ==="
mkdir -p "${STD_OUT}"
ln -sfn "$(pwd)/data/scas/standard_full_sft/train.parquet" "${STD_OUT}/train.parquet"
ln -sfn "$(pwd)/data/scas/standard_full_sft/validation.parquet" "${STD_OUT}/validation.parquet"
"${PY}" scripts/prepare_sft_hf_cache.py "${STD_OUT}" \
  --tokenizer "${DECODER_MODEL}" \
  --max-seq-length "${SCAS_MAX_LEN}" \
  --num-proc 32 \
  --force

echo "=== modc style_1..9: remap prefixes + tokenize ==="
"${PY}" scripts/build_scas_modc_style_prefix_sft.py \
  --src-dir data/scas/modc_prefix_full_sft \
  --out-dir "${MODC_OUT}"
"${PY}" scripts/prepare_sft_hf_cache.py "${MODC_OUT}" \
  --tokenizer "${DECODER_MODEL}" \
  --max-seq-length "${SCAS_MAX_LEN}" \
  --num-proc 32 \
  --force

echo "DONE baselines tokenize"
ls -la "${STD_OUT}" "${MODC_OUT}"
