#!/usr/bin/env bash
#SBATCH --job-name=scas-init-tok
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=cpu  # set to your site partition
#SBATCH --time=12:00:00
#
# Retokenize covz K + vanilla for an init-ladder SIZE.
#   sbatch --export=ALL,SIZE=qwen3-1p7b-base,K=6 scripts/data/prepare-scas-init-ladder-data.slurm
#   sbatch --export=ALL,SIZE=qwen3-4b-instruct,K=6,ARMS=covz,standard scripts/...
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
source ./.venv/bin/activate
SIZE="${SIZE:?set SIZE=qwen3-1p7b-base|qwen3-4b-base|qwen3-4b-instruct|qwen3-4b-thinking}"
source scripts/scas_model_size.sh "${SIZE}"
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
PY=./.venv/bin/python
SROOT="${ARCHIVE_ROOT}"
K="${K:-6}"
ARMS="${ARMS:-covz,standard}"

echo "Tokenizer=${DECODER_MODEL} SIZE=${MODEL_SIZE} K=${K} ARMS=${ARMS}"
"${PY}" -c "import runpy; runpy.run_path('mod_c/train.py')['ensure_model_cached']('${DECODER_MODEL}')"

# Allow comma or plus separators (Slurm --export splits on commas).
ARMS="${ARMS//+/\,}"
IFS=',' read -r -a ARM_ARR <<< "${ARMS}"
for ARM in "${ARM_ARR[@]}"; do
  case "${ARM}" in
    covz)
      SRC="${SROOT}/cluster_prefix/covz/k${K}"
      OUT="${SROOT}/cluster_prefix/covz-${MODEL_SIZE}/k${K}"
      [[ -f "${SRC}/train.parquet" ]] || { echo "Missing ${SRC}/train.parquet"; exit 1; }
      mkdir -p "${OUT}"
      ln -sfn "${SRC}/train.parquet" "${OUT}/train.parquet"
      ln -sfn "${SRC}/validation.parquet" "${OUT}/validation.parquet"
      for f in meta.json dataset_meta.json build_meta.json style_map.json; do
        [[ -f "${SRC}/${f}" ]] && ln -sfn "${SRC}/${f}" "${OUT}/${f}" || true
      done
      echo "=== tokenize covz K=${K} -> ${OUT} ==="
      "${PY}" scripts/prepare_sft_hf_cache.py "${OUT}" \
        --tokenizer "${DECODER_MODEL}" \
        --max-seq-length "${SCAS_MAX_LEN}" \
        --num-proc 32 \
        --force
      ;;
    standard)
      OUT="${SROOT}/baselines/standard-${MODEL_SIZE}"
      mkdir -p "${OUT}"
      ln -sfn "$(pwd)/data/scas/standard_full_sft/train.parquet" "${OUT}/train.parquet"
      ln -sfn "$(pwd)/data/scas/standard_full_sft/validation.parquet" "${OUT}/validation.parquet"
      echo "=== tokenize vanilla -> ${OUT} ==="
      "${PY}" scripts/prepare_sft_hf_cache.py "${OUT}" \
        --tokenizer "${DECODER_MODEL}" \
        --max-seq-length "${SCAS_MAX_LEN}" \
        --num-proc 32 \
        --force
      ;;
    *)
      echo "Unknown ARM in ARMS: ${ARM}"; exit 1
      ;;
  esac
done
echo "DONE init-ladder tokenize SIZE=${MODEL_SIZE}"
