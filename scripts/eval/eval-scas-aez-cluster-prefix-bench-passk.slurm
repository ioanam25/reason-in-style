#!/usr/bin/env bash
#SBATCH --job-name=aez-bench-ev
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu  # set to your site partition
#SBATCH --qos=normal  # set to your site QoS (or remove)
#SBATCH --time=12:00:00
#
# Pass@k on AMC / OlympiadBench-EN (or other aliases) for AE cluster-prefix models.
#   sbatch --export=ALL,BENCH=amc,VAR=covz-qwen3,K=6,SIZE=qwen3-0p6b,CKPT_TAG=qwen3-opt,CKPT_DIR=...,CKPT_STEP=checkpoint-5400 \
#     scripts/eval/eval-scas-aez-cluster-prefix-bench-passk.slurm
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
SIZE="${SIZE:-qwen3-4b}"
source scripts/scas_model_size.sh "${SIZE}"

BENCH="${BENCH:?set BENCH=amc|olympiadbench}"
VAR="${VAR:?set VAR}"
K="${K:?set K}"
CKPT_STEP="${CKPT_STEP:-}"

case "${BENCH}" in
  amc|amc-mid)
    HF_DATASET="amc"
    BENCH_TAG="amc"
    ;;
  amc25|amc-2025|amc12-2025)
    HF_DATASET="amc25"
    BENCH_TAG="amc12-2025"
    ;;
  olympiadbench|olympiad|olympiadbench-en)
    HF_DATASET="olympiadbench"
    BENCH_TAG="olympiadbench-en"
    ;;
  gsm8k)
    HF_DATASET="gsm8k"
    BENCH_TAG="gsm8k"
    ;;
  aime|aime2024-2025)
    HF_DATASET="aime"
    BENCH_TAG="aime2024-2025"
    ;;
  hmmt|hmmt25|hmmt-feb-2025|hmmt_feb_2025)
    HF_DATASET="hmmt"
    BENCH_TAG="hmmt-feb-2025"
    ;;
  math500|math-500)
    HF_DATASET="HuggingFaceH4/MATH-500"
    BENCH_TAG="math500"
    ;;
  *)
    echo "Unknown BENCH=${BENCH}"; exit 1
    ;;
esac

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

if [[ -n "${CKPT_STEP:-}" && -n "${CKPT_DIR:-}" ]]; then
  :
else
  BEST="$(pick_best_ckpt)"
  CKPT_DIR="${BEST%%|*}"
  CKPT_STEP="${BEST##*|}"
fi
CKPT="${CKPT_DIR}/${CKPT_STEP}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
if [[ "${EOS_FIX:-0}" == "1" ]]; then
  OUT_TAG="${OUT_TAG--eosfix}"
else
  OUT_TAG="${OUT_TAG-}"
fi
if [[ -z "${OUT_DIR:-}" ]]; then
  if [[ "${CKPT_TAG:-opt}" != "opt" && "${CKPT_TAG}" != "qwen3-opt" ]]; then
    OUT_DIR="${SROOT}/model-evals/scas-aez-${VAR}-${MODEL_SIZE}-k${K}-${CKPT_TAG}_${CKPT_STEP}-${BENCH_TAG}-passk-balanced${OUT_TAG}"
  else
    OUT_DIR="${SROOT}/model-evals/scas-aez-${VAR}-${MODEL_SIZE}-k${K}_${CKPT_STEP}-${BENCH_TAG}-passk-balanced${OUT_TAG}"
  fi
fi

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
export EOS_FIX="${EOS_FIX:-1}"

STAGE="${STAGE:-all}"
echo "Host: $(hostname)"
echo "BENCH=${BENCH_TAG} HF=${HF_DATASET} VAR=${VAR} K=${K} ckpt=${CKPT} STAGE=${STAGE}"
echo "Output: ${OUT_DIR}"
if [[ "${STAGE}" == "grade" ]]; then
  export CUDA_VISIBLE_DEVICES=""
else
  nvidia-smi -L || true
fi

if [[ -f "${OUT_DIR}/pass_at_k.json" ]]; then
  echo "SKIP: already evaluated"; exit 0
fi
if [[ "${STAGE}" == "generate" && -f "${OUT_DIR}/generations.done" ]]; then
  echo "SKIP: generate already dumped"; exit 0
fi
if [[ "${STAGE}" == "grade" ]]; then
  [[ -f "${OUT_DIR}/generations.jsonl" ]] || { echo "MISSING ${OUT_DIR}/generations.jsonl"; exit 1; }
else
  [[ -d "${CKPT}" ]] || { echo "Missing checkpoint: ${CKPT}"; exit 1; }
fi
mkdir -p "${OUT_DIR}"

SAMPLES_PER_STYLE="${SAMPLES_PER_STYLE:-0}"
echo "EOS_FIX=${EOS_FIX} STYLE_TOKEN_FORMAT=${STYLE_TOKEN_FORMAT:-hard} SAMPLES_PER_STYLE=${SAMPLES_PER_STYLE}"
SAMPLE_ARGS=(--samples-total 256)
if [[ "${SAMPLES_PER_STYLE}" -gt 0 ]]; then
  SAMPLE_ARGS=(--samples-per-style "${SAMPLES_PER_STYLE}")
fi
"${VENV_PY}" scripts/eval/eval_math_cluster_prefix_qwen.py \
  --checkpoint "${CKPT}" \
  --oracle-styles "${STYLES[@]}" \
  --style-token-format "${STYLE_TOKEN_FORMAT:-hard}" \
  --hf-dataset "${HF_DATASET}" \
  --output-dir "${OUT_DIR}" \
  --sampling-mode balanced \
  "${SAMPLE_ARGS[@]}" \
  --tensor-parallel-size 1 \
  --data-parallel-replicas -1 \
  --max-tokens "${SCAS_MAX_LEN}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --budgets 1 2 4 8 16 32 64 128 256 \
  --grade-timeout 2.0 \
  --grade-flush-every 100 \
  --stage "${STAGE}"

echo "Done STAGE=${STAGE} ${OUT_DIR}"
