#!/usr/bin/env bash
# Run one question-only (vanilla SFT) MATH-500 Pass@k eval. Requires: VAR, SIZE.
#
#   VAR       baseline checkpoint stem under ${SROOT}/baselines/ckpt (e.g. standard-qwen3-4b-thinking)
#   SIZE      student size key understood by scripts/scas_model_size.sh
#   CKPT_TAG  suffix on the checkpoint dir (default: opt)
#   EOS_FIX   1 (default) to pass explicit stop_token_ids
#   OUT_TAG   suffix appended to OUT_DIR (default: -eosfix when EOS_FIX=1)
#
# CKPT_DIR / CKPT_STEP / OUT_DIR may be set explicitly to override discovery.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
SIZE="${SIZE:?set SIZE}"
source scripts/scas_model_size.sh "${SIZE}"
# Optional decode budget; SFT context stays 4096.
if [[ -n "${MAX_TOKENS:-}" ]]; then
  SCAS_MAX_LEN="${MAX_TOKENS}"
fi
VAR="${VAR:?set VAR}"
CKPT_TAG="${CKPT_TAG:-opt}"
SROOT="${ARCHIVE_ROOT}"
export EOS_FIX="${EOS_FIX:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"

if [[ -z "${CKPT_DIR:-}" ]]; then
  for d in "${SROOT}/baselines/ckpt/${VAR}-${CKPT_TAG}" "${SROOT}/baselines/ckpt/${VAR}-opt" "${SROOT}/baselines/ckpt/${VAR}"; do
    [[ -d "$d" ]] && { CKPT_DIR="$d"; break; }
  done
fi
[[ -n "${CKPT_DIR:-}" ]] || { echo "No checkpoint dir for VAR=${VAR} CKPT_TAG=${CKPT_TAG}"; exit 1; }

if [[ -z "${CKPT_STEP:-}" ]]; then
  CKPT_STEP="$(ls -1 "${CKPT_DIR}" | grep -E '^checkpoint-[0-9]+$' | sort -t- -k2 -n | tail -1)"
fi
CKPT="${CKPT_DIR}/${CKPT_STEP}"

if [[ "${EOS_FIX}" == "1" ]]; then
  OUT_TAG="${OUT_TAG--eosfix}"
else
  OUT_TAG="${OUT_TAG-}"
fi
# Keep seed replicates (opt-s2/opt-s3) from colliding: they share a step number.
if [[ "${CKPT_TAG}" != "opt" ]]; then
  _VAR_TAG="${VAR}-${CKPT_TAG}"
else
  _VAR_TAG="${VAR}"
fi
OUT_DIR="${OUT_DIR:-${SROOT}/model-evals/scas-${_VAR_TAG}_${CKPT_STEP}-math500-passk${OUT_TAG}}"

VENV_PY="./.venv/bin/python"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export HF_HOME="${HF_HOME:-${HF_CACHE}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

echo "==== $(date -Is) VAR=${VAR} SIZE=${MODEL_SIZE} ckpt=${CKPT} EOS_FIX=${EOS_FIX} T=${TEMPERATURE} top_p=${TOP_P}"
echo "==== OUT_DIR=${OUT_DIR}"
[[ -f "${OUT_DIR}/pass_at_k.json" ]] && { echo "SKIP already evaluated: ${OUT_DIR}"; exit 0; }
[[ -d "${CKPT}" ]] || { echo "Missing checkpoint: ${CKPT}"; exit 1; }
mkdir -p "${OUT_DIR}"

"${VENV_PY}" scripts/eval/eval_math_standard_passk_qwen.py \
  --checkpoint "${CKPT}" \
  --hf-dataset HuggingFaceH4/MATH-500 \
  --output-dir "${OUT_DIR}" \
  --samples-per-problem 256 \
  --tensor-parallel-size 1 \
  --data-parallel-replicas -1 \
  --max-tokens "${SCAS_MAX_LEN}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --budgets 1 2 4 8 16 32 64 128 256 \
  --grade-timeout 2.0 \
  --grade-flush-every 100
echo "DONE ${OUT_DIR}"
