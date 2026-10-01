#!/usr/bin/env bash
#SBATCH --job-name=aez-pfx-build
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=cpu  # set to your site partition
#SBATCH --time=12:00:00
#
# Build + tokenize [style_i] prefix SFT data from an AE variant's k-means sweep.
# All outputs go to scratch (project dir is over quota).
#   sbatch --export=ALL,VAR=b8,KMIN=2,KMAX=10 scripts/data/build-scas-aez-cluster-prefix.slurm

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
source ./.venv/bin/activate
source scripts/scas_model_size.sh qwen3-4b
# tqdm's monitor thread aborts the interpreter mid-map on this stack.
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
PY=./.venv/bin/python

VAR="${VAR:?set VAR=b8|d64}"
KMIN="${KMIN:-2}"
KMAX="${KMAX:-10}"

SROOT="${ARCHIVE_ROOT}"
ZROOT="${SROOT}/zscore-variants"
KMEANS_DIR="${ZROOT}/${VAR}/kmeans_sweep"
VAL_EMB="${ZROOT}/${VAR}-validation/embeddings_validation_z.npz"

[[ -d "${KMEANS_DIR}" ]] || { echo "Missing ${KMEANS_DIR}"; exit 1; }
[[ -f "${VAL_EMB}" ]] || { echo "Missing ${VAL_EMB}"; exit 1; }

for K in $(seq "${KMIN}" "${KMAX}"); do
  OUT="${SROOT}/cluster_prefix/${VAR}/k${K}"
  if [[ -f "${OUT}/.hf_tokenized/train/dataset_info.json" ]]; then
    echo "=== ${VAR} K=${K} already tokenized, skipping ==="
    continue
  fi
  echo "=== ${VAR} K=${K} -> ${OUT} ==="
  mkdir -p "${OUT}"
  "${PY}" scripts/data/build_scas_kmeans_cluster_prefix_sft_dataset.py \
    --k "${K}" \
    --prefix-dir "${MODC_SFT_DIR}" \
    --kmeans-dir "${KMEANS_DIR}" \
    --val-embeddings-npz "${VAL_EMB}" \
    --output-dir "${OUT}"

  echo "--- tokenizing ${VAR} K=${K} ---"
  "${PY}" scripts/prepare_sft_hf_cache.py "${OUT}" \
    --tokenizer "${DECODER_MODEL}" \
    --max-seq-length "${SCAS_MAX_LEN}" \
    --num-proc 32 \
    --force
done
echo "DONE ${VAR} K=${KMIN}..${KMAX}"
