#!/usr/bin/env bash
#SBATCH --job-name=ae-gmm-pfx-build
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=cpu  # set to your site partition
#SBATCH --time=12:00:00
#
# Build tokenizer-agnostic [style_i] parquets from AE-decoder GMM K=6.
#   sbatch scripts/data/build-scas-ae-decoder-gmm-prefix.slurm
#   sbatch --export=ALL,K=6,STYLES=covz-qwen3-1p7b+covz-qwen3-4b scripts/...
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
source ./.venv/bin/activate
source scripts/scas_model_size.sh qwen3-0p6b
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
PY=./.venv/bin/python

SROOT="${ARCHIVE_ROOT}"
K="${K:-6}"
STYLES="${STYLES:-covz-qwen3-1p7b+covz-qwen3-4b}"
STYLES="${STYLES//+/ }"

echo "Build AE GMM prefix K=${K} STYLES=${STYLES}"
for STYLE in ${STYLES}; do
  GMM_DIR="${SROOT}/zscore-variants/${STYLE}/gmm_from_kmeans"
  OUT="${SROOT}/cluster_prefix/${STYLE}-gmm/k${K}"
  KK=$(printf '%02d' "${K}")
  [[ -f "${GMM_DIR}/assignments_gmm_k${KK}.parquet" ]] || {
    echo "Missing GMM assignments in ${GMM_DIR}"; exit 1;
  }
  echo "=== ${STYLE} -> ${OUT} ==="
  mkdir -p "${OUT}"
  "${PY}" scripts/data/build_scas_gmm_cluster_prefix_sft_dataset.py \
    --k "${K}" \
    --prefix-dir "${MODC_SFT_DIR}" \
    --gmm-dir "${GMM_DIR}" \
    --output-dir "${OUT}"
done
echo "DONE AE GMM prefix build"
