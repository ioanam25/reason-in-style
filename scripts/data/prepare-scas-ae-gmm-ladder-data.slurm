#!/usr/bin/env bash
#SBATCH --job-name=ae-gmm-tok
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=cpu  # set to your site partition
#SBATCH --time=12:00:00
#
# Retokenize AE-GMM K=6 prefix data for one student SIZE.
#   sbatch --export=ALL,SIZE=qwen3-4b-instruct,K=6,STYLES=covz-qwen3-1p7b+covz-qwen3-4b \
#     scripts/data/prepare-scas-ae-gmm-ladder-data.slurm
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
source ./.venv/bin/activate
SIZE="${SIZE:?set SIZE}"
source scripts/scas_model_size.sh "${SIZE}"
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
PY=./.venv/bin/python
SROOT="${ARCHIVE_ROOT}"
K="${K:-6}"
STYLES="${STYLES:-covz-qwen3-1p7b+covz-qwen3-4b}"
STYLES="${STYLES//+/ }"

echo "Tokenizer=${DECODER_MODEL} SIZE=${MODEL_SIZE} K=${K} STYLES=${STYLES}"
"${PY}" -c "import runpy; runpy.run_path('mod_c/train.py')['ensure_model_cached']('${DECODER_MODEL}')"

for STYLE in ${STYLES}; do
  SRC="${SROOT}/cluster_prefix/${STYLE}-gmm/k${K}"
  OUT="${SROOT}/cluster_prefix/${STYLE}-gmm-${MODEL_SIZE}/k${K}"
  [[ -f "${SRC}/train.parquet" ]] || { echo "Missing ${SRC}/train.parquet"; exit 1; }
  mkdir -p "${OUT}"
  ln -sfn "${SRC}/train.parquet" "${OUT}/train.parquet"
  ln -sfn "${SRC}/validation.parquet" "${OUT}/validation.parquet"
  for f in meta.json dataset_meta.json build_meta.json style_map.json; do
    [[ -f "${SRC}/${f}" ]] && ln -sfn "${SRC}/${f}" "${OUT}/${f}" || true
  done
  echo "=== tokenize ${STYLE}-gmm K=${K} -> ${OUT} ==="
  "${PY}" scripts/prepare_sft_hf_cache.py "${OUT}" \
    --tokenizer "${DECODER_MODEL}" \
    --max-seq-length "${SCAS_MAX_LEN}" \
    --num-proc 32 \
    --force
done
echo "DONE AE-GMM tokenize SIZE=${MODEL_SIZE}"
