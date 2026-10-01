#!/usr/bin/env bash
#SBATCH --job-name=aez-scasval-ev
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu  # set to your site partition
#SBATCH --qos=normal  # set to your site QoS (or remove)
#SBATCH --time=24:00:00
#
# SCAS SFT-validation Pass@k (unique questions) for AE cluster-prefix models.
# Used to pick K without relying on MATH-500 / AMC / Olympiad.
#
#   sbatch --export=ALL,VAR=covz-qwen3,K=6,SIZE=qwen3-0p6b,CKPT_TAG=qwen3-opt,CKPT_STEP=checkpoint-5400 \
#     scripts/eval/eval-scas-aez-cluster-prefix-scasval-passk.slurm
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
SIZE="${SIZE:-qwen3-0p6b}"
source scripts/scas_model_size.sh "${SIZE}"

VAR="${VAR:?set VAR}"
K="${K:?set K}"
CKPT_STEP="${CKPT_STEP:-}"
SROOT="${ARCHIVE_ROOT}"

pick_best_ckpt() {
  local best_dir="" best_step="" best_n=-1
  local d step n
  CKPT_TAG="${CKPT_TAG:-opt}"
  for d in "${SROOT}/cluster_ckpt/${VAR}/k${K}-${CKPT_TAG}" \
           "${SROOT}/cluster_ckpt/${VAR}/k${K}-opt" \
           "${SROOT}/cluster_ckpt/${VAR}/k${K}"; do
    [[ -d "$d" ]] || continue
    step="$(ls -1 "$d" 2>/dev/null | grep -E "^checkpoint-[0-9]+$" | sort -t- -k2 -n | tail -1 || true)"
    [[ -n "$step" ]] || continue
    n="${step##*-}"
    if (( n > best_n )); then
      best_n=$n; best_step=$step; best_dir=$d
    fi
  done
  echo "${best_dir}|${best_step}"
}

if [[ -z "${CKPT_STEP:-}" || -z "${CKPT_DIR:-}" ]]; then
  BEST="$(pick_best_ckpt)"
  CKPT_DIR="${BEST%%|*}"
  CKPT_STEP="${BEST##*|}"
fi
CKPT="${CKPT_DIR}/${CKPT_STEP}"
OUT_DIR="${SROOT}/model-evals/scas-aez-${VAR}-${MODEL_SIZE}-k${K}_${CKPT_STEP}-scas-val-passk-balanced"

# Unique questions from standard (no-prefix) SCAS validation parquet — same split as SFT val.
VAL_PARQUET="${VAL_PARQUET:-data/scas/standard_full_sft/validation.parquet}"

STYLES=()
for i in $(seq 1 "${K}"); do STYLES+=("style_${i}"); done

VENV_PY="./.venv/bin/python"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export HF_HOME="${HF_HOME:-${HF_CACHE}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

echo "Host: $(hostname)"
echo "SCAS-val Pass@k VAR=${VAR} K=${K} ckpt=${CKPT}"
echo "val_parquet=${VAL_PARQUET}"
echo "Output: ${OUT_DIR}"
nvidia-smi -L || true

[[ -f "${OUT_DIR}/pass_at_k.json" ]] && { echo "SKIP: already evaluated"; exit 0; }
[[ -d "${CKPT}" ]] || { echo "Missing checkpoint: ${CKPT}"; exit 1; }
[[ -f "${VAL_PARQUET}" ]] || { echo "Missing ${VAL_PARQUET}"; exit 1; }
mkdir -p "${OUT_DIR}"

"${VENV_PY}" scripts/eval/eval_math_cluster_prefix_qwen.py \
  --checkpoint "${CKPT}" \
  --oracle-styles "${STYLES[@]}" \
  --test-parquet "${VAL_PARQUET}" \
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

echo "Done: ${OUT_DIR}/pass_at_k.json"
