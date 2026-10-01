#!/usr/bin/env bash
#SBATCH --job-name=rand-bench-ev
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
# Random-cluster control Pass@k for remaining benches (AMC, Olympiad, SCAS-val).
#   sbatch --export=ALL,K=3 scripts/eval/eval-scas-random-qwen3-benches-passk.slurm
#   sbatch --export=ALL,K=6 scripts/eval/eval-scas-random-qwen3-benches-passk.slurm
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
source scripts/scas_model_size.sh qwen3-0p6b

K="${K:?set K}"
SROOT="${ARCHIVE_ROOT}"
CKPT_DIR="${SROOT}/cluster_ckpt/random-qwen3/k${K}-opt"
CKPT_STEP="${CKPT_STEP:-checkpoint-5400}"
CKPT="${CKPT_DIR}/${CKPT_STEP}"
VAL_PARQUET="${VAL_PARQUET:-data/scas/standard_full_sft/validation.parquet}"
N_STYLES="${K}"

[[ -d "${CKPT}" ]] || { echo "Missing ${CKPT}"; exit 1; }

export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export HF_HOME="${HF_HOME:-${HF_CACHE}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

STYLES=()
for i in $(seq 1 "${N_STYLES}"); do STYLES+=("style_${i}"); done

echo "Host: $(hostname)"
echo "random-qwen3 K=${K} ckpt=${CKPT}"
nvidia-smi -L || true

run_bench() {
  local BENCH_TAG="$1"
  local HF_OR_PARQUET="$2"  # hf name or parquet path
  local MODE="$3"           # hf|parquet
  local OUT_DIR="${SROOT}/model-evals/scas-aez-random-qwen3-qwen3-0p6b-k${K}_${CKPT_STEP}-${BENCH_TAG}-passk-balanced"
  if [[ -f "${OUT_DIR}/pass_at_k.json" ]]; then
    echo "SKIP ${BENCH_TAG}: already done"
    return 0
  fi
  mkdir -p "${OUT_DIR}"
  echo "=== ${BENCH_TAG} -> ${OUT_DIR} ==="
  local EXTRA=()
  if [[ "${MODE}" == "hf" ]]; then
    EXTRA+=(--hf-dataset "${HF_OR_PARQUET}")
  else
    EXTRA+=(--test-parquet "${HF_OR_PARQUET}")
  fi
  ./.venv/bin/python scripts/eval/eval_math_cluster_prefix_qwen.py \
    --checkpoint "${CKPT}" \
    --oracle-styles "${STYLES[@]}" \
    "${EXTRA[@]}" \
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
  echo "Done ${OUT_DIR}/pass_at_k.json"
}

run_bench amc amc hf
run_bench olympiadbench-en olympiadbench hf
run_bench scas-val "${VAL_PARQUET}" parquet

echo "ALL BENCHES DONE K=${K}"
