#!/usr/bin/env bash
# Question-only Pass@k on an arbitrary HF bench alias.
# Requires: VAR, SIZE, BENCH (amc|olympiadbench|math500|...)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
SIZE="${SIZE:?set SIZE}"
source scripts/scas_model_size.sh "${SIZE}"
VAR="${VAR:?set VAR}"
BENCH="${BENCH:?set BENCH}"
CKPT_TAG="${CKPT_TAG:-opt}"
SROOT="${ARCHIVE_ROOT}"
export EOS_FIX="${EOS_FIX:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"

STAGE="${STAGE:-all}"
case "${BENCH}" in
  amc|amc-mid) HF_DATASET="amc"; BENCH_TAG="amc" ;;
  amc25|amc-2025|amc12-2025) HF_DATASET="amc25"; BENCH_TAG="amc12-2025" ;;
  olympiadbench|olympiad|olympiadbench-en) HF_DATASET="olympiadbench"; BENCH_TAG="olympiadbench-en" ;;
  math500|math-500) HF_DATASET="HuggingFaceH4/MATH-500"; BENCH_TAG="math500" ;;
  aime|aime2024-2025) HF_DATASET="aime"; BENCH_TAG="aime2024-2025" ;;
  hmmt|hmmt25|hmmt-feb-2025|hmmt_feb_2025) HF_DATASET="hmmt"; BENCH_TAG="hmmt-feb-2025" ;;
  *) echo "Unknown BENCH=${BENCH}"; exit 1 ;;
esac

if [[ -z "${CKPT_DIR:-}" ]]; then
  for d in "${SROOT}/baselines/ckpt/${VAR}-${CKPT_TAG}" "${SROOT}/baselines/ckpt/${VAR}-opt" "${SROOT}/baselines/ckpt/${VAR}"; do
    [[ -d "$d" ]] && { CKPT_DIR="$d"; break; }
  done
fi
[[ -n "${CKPT_DIR:-}" ]] || { echo "No ckpt for ${VAR}"; exit 1; }
CKPT_STEP="${CKPT_STEP:-$(ls -1 "${CKPT_DIR}" | grep -E '^checkpoint-[0-9]+$' | sort -t- -k2 -n | tail -1)}"
CKPT="${CKPT_DIR}/${CKPT_STEP}"

if [[ "${EOS_FIX}" == "1" ]]; then
  OUT_TAG="${OUT_TAG--eosfix}"
else
  OUT_TAG="${OUT_TAG-}"
fi
if [[ "${CKPT_TAG}" != "opt" ]]; then
  _VT="${VAR}-${CKPT_TAG}"
else
  _VT="${VAR}"
fi
OUT_DIR="${OUT_DIR:-${SROOT}/model-evals/scas-${_VT}_${CKPT_STEP}-${BENCH_TAG}-passk${OUT_TAG}}"

VENV_PY="./.venv/bin/python"
export TOKENIZERS_PARALLELISM=false HF_DATASETS_DISABLE_PROGRESS_BARS=1
export HF_HOME="${HF_HOME:-${HF_CACHE}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1

echo "==== $(date -Is) VANILLA BENCH=${BENCH_TAG} VAR=${VAR} ckpt=${CKPT} EOS_FIX=${EOS_FIX} T=${TEMPERATURE} top_p=${TOP_P} STAGE=${STAGE}"
echo "==== OUT_DIR=${OUT_DIR}"
if [[ -f "${OUT_DIR}/pass_at_k.json" ]]; then
  echo SKIP graded; exit 0
fi
if [[ "${STAGE}" == "generate" && -f "${OUT_DIR}/generations.done" ]]; then
  echo SKIP generate; exit 0
fi
if [[ "${STAGE}" == "grade" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  [[ -f "${OUT_DIR}/generations.jsonl" ]] || { echo "MISSING ${OUT_DIR}/generations.jsonl"; exit 1; }
else
  [[ -d "${CKPT}" ]] || { echo "Missing ${CKPT}"; exit 1; }
fi
mkdir -p "${OUT_DIR}"

"${VENV_PY}" scripts/eval/eval_math_standard_passk_qwen.py \
  --checkpoint "${CKPT}" \
  --hf-dataset "${HF_DATASET}" \
  --output-dir "${OUT_DIR}" \
  --samples-per-problem 256 \
  --tensor-parallel-size 1 \
  --data-parallel-replicas -1 \
  --max-tokens "${SCAS_MAX_LEN}" \
  --temperature "${TEMPERATURE}" --top-p "${TOP_P}" \
  --budgets 1 2 4 8 16 32 64 128 256 \
  --grade-timeout 2.0 --grade-flush-every 100 \
  --stage "${STAGE}"
echo "DONE ${OUT_DIR}"
