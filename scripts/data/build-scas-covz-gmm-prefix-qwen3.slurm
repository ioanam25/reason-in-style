#!/usr/bin/env bash
#SBATCH --job-name=covz-gmm-pfx
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=cpu  # set to your site partition
#SBATCH --time=12:00:00
#
# Build GMM cluster-prefix parquets + tokenize with Qwen3-0.6B.
#   sbatch --export=ALL,KLIST="21 29" scripts/data/build-scas-covz-gmm-prefix-qwen3.slurm
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
GMM_DIR="${SROOT}/zscore-variants/covz/gmm_from_kmeans"
OUT_ROOT="${SROOT}/cluster_prefix/covz-gmm-qwen3"
KLIST="${KLIST:-21 29}"

echo "Tokenizer=${DECODER_MODEL}"
echo "GMM_DIR=${GMM_DIR} OUT_ROOT=${OUT_ROOT} KLIST=${KLIST}"

for K in ${KLIST}; do
  OUT="${OUT_ROOT}/k${K}"
  echo "=== build GMM prefix K=${K} -> ${OUT} ==="
  mkdir -p "${OUT}"
  "${PY}" scripts/data/build_scas_gmm_cluster_prefix_sft_dataset.py \
    --k "${K}" \
    --prefix-dir "${MODC_SFT_DIR}" \
    --gmm-dir "${GMM_DIR}" \
    --output-dir "${OUT}"

  echo "--- tokenize K=${K} with ${DECODER_MODEL} ---"
  "${PY}" scripts/prepare_sft_hf_cache.py "${OUT}" \
    --tokenizer "${DECODER_MODEL}" \
    --max-seq-length "${SCAS_MAX_LEN}" \
    --num-proc 32 \
    --force
done
echo "DONE GMM prefix+tokenize KLIST=${KLIST}"
