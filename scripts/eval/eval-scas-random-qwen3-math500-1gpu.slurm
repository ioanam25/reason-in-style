#!/usr/bin/env bash
#SBATCH --job-name=rand-m500-1g
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu  # set to your site partition
#SBATCH --qos=normal  # set to your site QoS (or remove)
#SBATCH --time=8:00:00
#
#   sbatch --export=ALL,K=2 scripts/eval/eval-scas-random-qwen3-math500-1gpu.slurm
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
OUT_DIR="${SROOT}/model-evals/scas-aez-random-qwen3-qwen3-0p6b-k${K}_${CKPT_STEP}-math500-passk-balanced"

export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export HF_HOME="${HF_HOME:-${HF_CACHE}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

echo "Host: $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-all}"
echo "random K=${K} ckpt=${CKPT}"
nvidia-smi -L || true

[[ -d "${CKPT}" ]] || { echo "Missing ${CKPT}"; exit 1; }
[[ -f "${OUT_DIR}/pass_at_k.json" ]] && { echo "SKIP done"; exit 0; }
mkdir -p "${OUT_DIR}"

STYLES=()
for i in $(seq 1 "${K}"); do STYLES+=("style_${i}"); done

./.venv/bin/python scripts/eval/eval_math_cluster_prefix_qwen.py \
  --checkpoint "${CKPT}" \
  --oracle-styles "${STYLES[@]}" \
  --hf-dataset HuggingFaceH4/MATH-500 \
  --output-dir "${OUT_DIR}" \
  --sampling-mode balanced \
  --samples-total 256 \
  --tensor-parallel-size 1 \
  --data-parallel-replicas 1 \
  --max-tokens "${SCAS_MAX_LEN}" \
  --temperature 1.0 \
  --top-p 1.0 \
  --budgets 1 2 4 8 16 32 64 128 256 \
  --grade-timeout 2.0 \
  --grade-flush-every 100 \
  --gpu-memory-utilization 0.90

echo "Done ${OUT_DIR}/pass_at_k.json"
