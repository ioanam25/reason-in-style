#!/usr/bin/env bash
# Run one AE-GMM MATH-500 Pass@k eval. Requires: VAR, SIZE; K default 6.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
SIZE="${SIZE:?set SIZE}"
source scripts/scas_model_size.sh "${SIZE}"
VAR="${VAR:?set VAR}"
K="${K:-6}"
CKPT_TAG="${CKPT_TAG:-opt}"
SROOT="${ARCHIVE_ROOT}"

pick_best_ckpt() {
  local best_dir="" best_step="" best_n=-1 d step n
  CKPT_TAG="${CKPT_TAG:-opt}"
  for d in "${SROOT}/cluster_ckpt/${VAR}/k${K}-${CKPT_TAG}" "${SROOT}/cluster_ckpt/${VAR}/k${K}-opt" "${SROOT}/cluster_ckpt/${VAR}/k${K}" "${SROOT}/baselines/ckpt/${VAR}-${CKPT_TAG}" "${SROOT}/baselines/ckpt/${VAR}-opt"; do
    [[ -d "$d" ]] || continue
    step="$(ls -1 "$d" 2>/dev/null | grep -E "^checkpoint-[0-9]+$" | sort -t- -k2 -n | tail -1 || true)"
    [[ -n "$step" ]] || continue
    n="${step##*-}"
    if (( n > best_n )); then best_n=$n; best_step=$step; best_dir=$d; fi
  done
  echo "${best_dir}|${best_step}"
}

if [[ -z "${CKPT_DIR:-}" || -z "${CKPT_STEP:-}" ]]; then
  BEST="$(pick_best_ckpt)"
  CKPT_DIR="${BEST%%|*}"
  CKPT_STEP="${BEST##*|}"
fi
CKPT="${CKPT_DIR}/${CKPT_STEP}"
if [[ -n "${FORCE_STYLES:-}" ]]; then
  read -ra STYLES <<< "${FORCE_STYLES}"
else
  STYLES=()
  for i in $(seq 1 "${K}"); do STYLES+=("style_${i}"); done
fi
_STYLE_TAG=""
if [[ -n "${FORCE_STYLES:-}" ]]; then
  _STYLE_TAG="-$(echo "${FORCE_STYLES}" | tr " " "+" | tr -cd "A-Za-z0-9+_")"
fi
if [[ -n "${OUT_DIR:-}" ]]; then
  :
elif [[ "${CKPT_TAG:-opt}" != "opt" ]]; then
  OUT_DIR="${SROOT}/model-evals/scas-aez-${VAR}-${MODEL_SIZE}-k${K}-${CKPT_TAG}_${CKPT_STEP}-math500-passk-balanced${_STYLE_TAG}"
else
  OUT_DIR="${SROOT}/model-evals/scas-aez-${VAR}-${MODEL_SIZE}-k${K}_${CKPT_STEP}-math500-passk-balanced${_STYLE_TAG}"
fi
VENV_PY="./.venv/bin/python"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export HF_HOME="${HF_HOME:-${HF_CACHE}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export EOS_FIX="${EOS_FIX:-1}"
echo "EOS_FIX=${EOS_FIX} STYLE_TOKEN_FORMAT=${STYLE_TOKEN_FORMAT:-hard}"

echo "==== $(date -Is) VAR=${VAR} SIZE=${MODEL_SIZE} K=${K} ckpt=${CKPT}"
[[ -f "${OUT_DIR}/pass_at_k.json" ]] && { echo "SKIP already evaluated: ${OUT_DIR}"; exit 0; }
[[ -d "${CKPT}" ]] || { echo "Missing checkpoint: ${CKPT}"; exit 1; }
if [[ "${VAR}" == *final* && "${ALLOW_PARTIAL:-0}" != "1" ]]; then
  ep=$(${VENV_PY} -c "import json;print(float(json.load(open(\"${CKPT}/trainer_state.json\")).get(\"epoch\",0)))")
  ${VENV_PY} -c "import sys; e=float(\"${ep}\"); sys.exit(0 if e>=49 else 1)" || {
    echo "CKPT only epoch ${ep}; refuse (set ALLOW_PARTIAL=1)"; exit 1
  }
fi
mkdir -p "${OUT_DIR}"
"${VENV_PY}" scripts/eval/eval_math_cluster_prefix_qwen.py \
  --checkpoint "${CKPT}" \
  --oracle-styles "${STYLES[@]}" \
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
