#!/usr/bin/env bash
#SBATCH --job-name=scas-rnd-q3-tok
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=cpu  # set to your site partition
#SBATCH --time=12:00:00
#
# Random-cluster assign + [style_i] prefix build + Qwen3-0.6B tokenize (K=2..10).
# Outputs under scratch (project dir quota).
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
RAND_DIR="${SROOT}/cluster_random/assignments-qwen3"
PFX_ROOT="${SROOT}/cluster_prefix/random-qwen3"
KMIN="${KMIN:-2}"
KMAX="${KMAX:-10}"
SEED="${SEED:-42}"

echo "=== Random assignments seed=${SEED} -> ${RAND_DIR} ==="
"${PY}" scripts/data/assign_scas_random_clusters.py \
  --prefix-dir "${MODC_SFT_DIR}" \
  --output-dir "${RAND_DIR}" \
  --k-min "${KMIN}" \
  --k-max "${KMAX}" \
  --seed "${SEED}"

for K in $(seq "${KMIN}" "${KMAX}"); do
  OUT="${PFX_ROOT}/k${K}"
  echo "=== Build K=${K} -> ${OUT} ==="
  mkdir -p "${OUT}"
  "${PY}" scripts/data/build_scas_random_cluster_prefix_sft_dataset.py \
    --k "${K}" \
    --prefix-dir "${MODC_SFT_DIR}" \
    --random-dir "${RAND_DIR}" \
    --output-dir "${OUT}"

  echo "=== Tokenize K=${K} with ${DECODER_MODEL} ==="
  "${PY}" scripts/prepare_sft_hf_cache.py "${OUT}" \
    --tokenizer "${DECODER_MODEL}" \
    --max-seq-length "${SCAS_MAX_LEN}" \
    --num-proc 32 \
    --force
done
echo "DONE random-qwen3 build+tokenize K=${KMIN}..${KMAX}"
