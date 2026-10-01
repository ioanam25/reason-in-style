#!/usr/bin/env bash
#SBATCH --job-name=pfx-tok
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=cpu  # set to your site partition
#SBATCH --time=12:00:00
#
# Generic per-student-size tokenization of any [style_i] prefix dataset.
# Mirrors prepare-scas-ae-gmm-ladder-data.slurm but takes an explicit SRC tag.
#
#   sbatch --export=ALL,SRC=random-qwen3,SIZE=qwen3-4b-thinking,K=6 \
#     scripts/data/prepare-scas-prefix-size.slurm
#
# Reads : ${SROOT}/cluster_prefix/${SRC}/k${K}
# Writes: ${SROOT}/cluster_prefix/${SRC}-${MODEL_SIZE}/k${K}
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
source ./.venv/bin/activate
SIZE="${SIZE:?set SIZE}"
SRC="${SRC:?set SRC e.g. random-qwen3}"
source scripts/scas_model_size.sh "${SIZE}"
# Allow pointing at a modified base (e.g. one with <style_i> special tokens).
DECODER_MODEL="${MODEL_OVERRIDE:-$DECODER_MODEL}"
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
PY=./.venv/bin/python
SROOT="${ARCHIVE_ROOT}"
K="${K:-6}"

SRC_DIR="${SROOT}/cluster_prefix/${SRC}/k${K}"
OUT="${SROOT}/cluster_prefix/${SRC}-${MODEL_SIZE}/k${K}"

echo "Tokenizer=${DECODER_MODEL} SIZE=${MODEL_SIZE} SRC=${SRC} K=${K}"
[[ -f "${SRC_DIR}/train.parquet" ]] || { echo "Missing ${SRC_DIR}/train.parquet"; exit 1; }

"${PY}" -c "import runpy; runpy.run_path('mod_c/train.py')['ensure_model_cached']('${DECODER_MODEL}')"

mkdir -p "${OUT}"
ln -sfn "${SRC_DIR}/train.parquet" "${OUT}/train.parquet"
ln -sfn "${SRC_DIR}/validation.parquet" "${OUT}/validation.parquet"
for f in meta.json dataset_meta.json build_meta.json style_map.json; do
  [[ -f "${SRC_DIR}/${f}" ]] && ln -sfn "${SRC_DIR}/${f}" "${OUT}/${f}" || true
done

echo "=== tokenize ${SRC} K=${K} -> ${OUT} ==="
"${PY}" scripts/prepare_sft_hf_cache.py "${OUT}" \
  --tokenizer "${DECODER_MODEL}" \
  --max-seq-length "${SCAS_MAX_LEN}" \
  --num-proc 32 \
  --force

echo "DONE tokenize SRC=${SRC} SIZE=${MODEL_SIZE}"
