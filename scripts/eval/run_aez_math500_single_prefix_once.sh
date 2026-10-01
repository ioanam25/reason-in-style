#!/usr/bin/env bash
# Single-prefix MATH-500 Pass@k (all 256 samples from ONE style).
# Requires: VAR, SIZE; optional: K, STYLE (default style_1), CKPT_TAG, EOS_FIX=1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
SIZE="${SIZE:?set SIZE}"
source scripts/scas_model_size.sh "${SIZE}"
VAR="${VAR:?set VAR}"
K="${K:-6}"
STYLE="${STYLE:-style_1}"
CKPT_TAG="${CKPT_TAG:-opt}"
SROOT="${ARCHIVE_ROOT}"
export EOS_FIX="${EOS_FIX:-1}"

if [[ -z "${CKPT_DIR:-}" || -z "${CKPT_STEP:-}" ]]; then
  BEST_N=-1
  CKPT_DIR=""
  CKPT_STEP=""
  for d in "${SROOT}/cluster_ckpt/${VAR}/k${K}-${CKPT_TAG}" \
           "${SROOT}/cluster_ckpt/${VAR}/k${K}-opt" \
           "${SROOT}/cluster_ckpt/${VAR}/k${K}"; do
    [[ -d "$d" ]] || continue
    step="$(ls -1 "$d" 2>/dev/null | grep -E "^checkpoint-[0-9]+$" | sort -t- -k2 -n | tail -1 || true)"
    [[ -n "$step" ]] || continue
    n="${step##*-}"
    if (( n > BEST_N )); then BEST_N=$n; CKPT_DIR=$d; CKPT_STEP=$step; fi
  done
fi
CKPT="${CKPT_DIR}/${CKPT_STEP}"
OUT_DIR="${OUT_DIR:-${SROOT}/model-evals/scas-aez-${VAR}-${MODEL_SIZE}-k${K}_${CKPT_STEP}-math500-passk-single-${STYLE}-eosfix}"

VENV_PY="./.venv/bin/python"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export HF_HOME="${HF_HOME:-${HF_CACHE}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

echo "==== $(date -Is) SINGLE-PREFIX VAR=${VAR} STYLE=${STYLE} ckpt=${CKPT}"
echo "==== OUT_DIR=${OUT_DIR} EOS_FIX=${EOS_FIX}"
[[ -f "${OUT_DIR}/pass_at_k.json" ]] && { echo "SKIP ${OUT_DIR}"; exit 0; }
[[ -d "${CKPT}" ]] || { echo "Missing ${CKPT}"; exit 1; }
mkdir -p "${OUT_DIR}"

"${VENV_PY}" scripts/eval/eval_math_cluster_prefix_qwen.py \
  --checkpoint "${CKPT}" \
  --oracle-styles "${STYLE}" \
  --style-token-format "${STYLE_TOKEN_FORMAT:-hard}" \
  --hf-dataset HuggingFaceH4/MATH-500 \
  --output-dir "${OUT_DIR}" \
  --sampling-mode balanced \
  --samples-total 256 \
  --tensor-parallel-size 1 \
  --data-parallel-replicas -1 \
  --max-tokens "${SCAS_MAX_LEN}" \
  --temperature 1.0 \
  --top-p 1.0 \
  --budgets 1 2 4 8 16 32 64 128 256 \
  --grade-timeout 2.0 \
  --grade-flush-every 100
echo "DONE ${OUT_DIR}"
