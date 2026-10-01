#!/usr/bin/env bash
#SBATCH --job-name=scas-bench-std
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu  # set to your site partition
#SBATCH --qos=normal  # set to your site QoS (or remove)
#SBATCH --time=12:00:00
#
# Vanilla (no-prefix) Qwen3 baseline Pass@k on AMC / OlympiadBench-EN.
#   sbatch --export=ALL,BENCH=amc scripts/eval/eval-scas-standard-qwen3-bench-passk.slurm
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
source scripts/scas_model_size.sh qwen3-0p6b
S=${ARCHIVE_ROOT}
BENCH="${BENCH:?set BENCH=amc|olympiadbench}"
CKPT_DIR="${CKPT_DIR:-$S/baselines/ckpt/standard-qwen3-opt}"
CKPT_STEP="${CKPT_STEP:-$(ls -1 "$CKPT_DIR" | grep -E "^checkpoint-[0-9]+$" | sort -t- -k2 -n | tail -1)}"

case "${BENCH}" in
  amc|amc-mid) HF_DATASET="amc"; BENCH_TAG="amc" ;;
  olympiadbench|olympiad|olympiadbench-en) HF_DATASET="olympiadbench"; BENCH_TAG="olympiadbench-en" ;;
  *) echo "Unknown BENCH=${BENCH}"; exit 1 ;;
esac

CKPT="${CKPT_DIR}/${CKPT_STEP}"
OUT="$S/model-evals/scas-standard-qwen3-0p6b_${CKPT_STEP}-${BENCH_TAG}-passk"
export TOKENIZERS_PARALLELISM=false HF_DATASETS_DISABLE_PROGRESS_BARS=1
export HF_HOME=${HF_CACHE} HF_HUB_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1

[[ -d "$CKPT" ]] || { echo "missing $CKPT"; exit 1; }
[[ -f "$OUT/pass_at_k.json" ]] && { echo SKIP; exit 0; }
mkdir -p "$OUT"
echo "bench=$BENCH_TAG ckpt=$CKPT out=$OUT"
./.venv/bin/python scripts/eval/eval_math_standard_passk_qwen.py \
  --checkpoint "$CKPT" --hf-dataset "$HF_DATASET" --output-dir "$OUT" \
  --samples-per-problem 256 --tensor-parallel-size 1 --data-parallel-replicas -1 \
  --max-tokens "$SCAS_MAX_LEN" --temperature 1.0 --top-p 1.0 \
  --budgets 1 2 4 8 16 32 64 128 256 --grade-timeout 2.0 --grade-flush-every 100
echo Done
