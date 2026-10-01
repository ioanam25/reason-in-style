#!/usr/bin/env bash
#SBATCH --job-name=scas-init-ev
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu  # set to your site partition
#SBATCH --qos=normal  # set to your site QoS (or remove)
#SBATCH --time=24:00:00
#
# Init-ladder Pass@k on all 8 GPUs (DP replicas auto).
#   SIZE=qwen3-1p7b-base|qwen3-4b-base|qwen3-4b-instruct|qwen3-4b-thinking
#   ARM=covz|standard
#   BENCH=math500|amc|olympiadbench|scasval
#   K=6 for covz
#
#   sbatch --dependency=afterok:JOB --export=ALL,SIZE=...,ARM=covz,K=6,BENCH=math500 \
#     scripts/eval/eval-scas-init-ladder-passk.slurm
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
SIZE="${SIZE:?set SIZE}"
source scripts/scas_model_size.sh "${SIZE}"

ARM="${ARM:?set ARM=covz|standard}"
BENCH="${BENCH:?set BENCH=math500|amc|olympiadbench|scasval}"
SROOT="${ARCHIVE_ROOT}"
VAL_PARQUET="${VAL_PARQUET:-data/scas/standard_full_sft/validation.parquet}"

case "${BENCH}" in
  math500|math-500|m500)
    HF_DATASET="HuggingFaceH4/MATH-500"
    BENCH_TAG="math500"
    USE_PARQUET=0
    ;;
  amc|amc-mid)
    HF_DATASET="amc"
    BENCH_TAG="amc"
    USE_PARQUET=0
    ;;
  amc25|amc-2025|amc12-2025)
    HF_DATASET="amc25"
    BENCH_TAG="amc12-2025"
    USE_PARQUET=0
    ;;
  olympiadbench|olympiad|olympiadbench-en)
    HF_DATASET="olympiadbench"
    BENCH_TAG="olympiadbench-en"
    USE_PARQUET=0
    ;;
  scasval|scas-val|scas_val)
    HF_DATASET=""
    BENCH_TAG="scas-val"
    USE_PARQUET=1
    ;;
  *)
    echo "Unknown BENCH=${BENCH}"; exit 1
    ;;
esac

case "${ARM}" in
  covz)
    K="${K:?set K for covz}"
    CKPT_DIR="${CKPT_DIR:-${SROOT}/cluster_ckpt/covz-${MODEL_SIZE}/k${K}-opt}"
    TAG="scas-aez-covz-${MODEL_SIZE}-k${K}"
    N_STYLES="${K}"
    MODE="prefix"
    ;;
  standard)
    CKPT_DIR="${CKPT_DIR:-${SROOT}/baselines/ckpt/standard-${MODEL_SIZE}-opt}"
    TAG="scas-standard-${MODEL_SIZE}"
    MODE="standard"
    ;;
  *)
    echo "Unknown ARM=${ARM}"; exit 1
    ;;
esac

if [[ -z "${CKPT_STEP:-}" ]]; then
  CKPT_STEP="$(ls -1 "${CKPT_DIR}" 2>/dev/null | grep -E '^checkpoint-[0-9]+$' | sort -t- -k2 -n | tail -1 || true)"
fi
CKPT="${CKPT_DIR}/${CKPT_STEP}"
OUT_DIR="${SROOT}/model-evals/${TAG}_${CKPT_STEP}-${BENCH_TAG}-passk"
if [[ "${MODE}" == "prefix" ]]; then
  OUT_DIR="${OUT_DIR}-balanced"
fi

export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export HF_HOME="${HF_HOME:-${HF_CACHE}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

echo "Host: $(hostname)"
echo "SIZE=${MODEL_SIZE} ARM=${ARM} K=${K:--} BENCH=${BENCH_TAG} ckpt=${CKPT}"
echo "out=${OUT_DIR}"
echo "GPUs requested: 8 (data-parallel-replicas=-1)"
nvidia-smi -L || true

[[ -d "${CKPT}" ]] || { echo "Missing checkpoint: ${CKPT}"; exit 1; }
[[ -f "${OUT_DIR}/pass_at_k.json" ]] && { echo "SKIP: already evaluated"; exit 0; }
if [[ "${USE_PARQUET}" -eq 1 ]]; then
  [[ -f "${VAL_PARQUET}" ]] || { echo "Missing ${VAL_PARQUET}"; exit 1; }
fi
mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  --checkpoint "${CKPT}"
  --output-dir "${OUT_DIR}"
  --tensor-parallel-size 1
  --data-parallel-replicas -1
  --max-tokens "${SCAS_MAX_LEN}"
  --temperature 1.0
  --top-p 1.0
  --budgets 1 2 4 8 16 32 64 128 256
  --grade-timeout 2.0
  --grade-flush-every 100
)
if [[ "${USE_PARQUET}" -eq 1 ]]; then
  COMMON_ARGS+=(--test-parquet "${VAL_PARQUET}")
else
  COMMON_ARGS+=(--hf-dataset "${HF_DATASET}")
fi

if [[ "${MODE}" == "standard" ]]; then
  ./.venv/bin/python scripts/eval/eval_math_standard_passk_qwen.py \
    "${COMMON_ARGS[@]}" \
    --samples-per-problem 256
else
  STYLES=()
  for i in $(seq 1 "${N_STYLES}"); do STYLES+=("style_${i}"); done
  ./.venv/bin/python scripts/eval/eval_math_cluster_prefix_qwen.py \
    "${COMMON_ARGS[@]}" \
    --oracle-styles "${STYLES[@]}" \
    --sampling-mode balanced \
    --samples-total 256
fi

echo "Done: ${OUT_DIR}/pass_at_k.json"
