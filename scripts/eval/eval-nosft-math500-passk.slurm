#!/usr/bin/env bash
#SBATCH --job-name=m500-nosft
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu  # set to your site partition
#SBATCH --qos=normal  # set to your site QoS (or remove)
#SBATCH --time=48:00:00
#
# No-SFT MATH-500 Pass@k (EOS_FIX) for a HuggingFace Qwen3 checkpoint.
# Same question-only protocol as vanilla SFT evals.
#
#   sbatch --export=ALL,HF_MODEL=Qwen/Qwen3-4B-Instruct-2507,SIZE=qwen3-4b-instruct,TAG=instruct \
#     scripts/eval/eval-nosft-math500-passk.slurm
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
HF_MODEL="${HF_MODEL:?set HF_MODEL}"
SIZE="${SIZE:?set SIZE}"
source scripts/scas_model_size.sh "${SIZE}"
if [[ -n "${MAX_TOKENS:-}" ]]; then
  SCAS_MAX_LEN="${MAX_TOKENS}"
fi

SROOT="${ARCHIVE_ROOT}"
export EOS_FIX="${EOS_FIX:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export HF_HOME="${HF_HOME:-${HF_CACHE}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

# Materialize a local snapshot so load_model_config finds config.json (HF ids
# are not a filesystem path). Instant if already in the hub cache.
CKPT="$(./.venv/bin/python - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download("${HF_MODEL}"))
PY
)"
OUT_DIR="${OUT_DIR:-${SROOT}/model-evals/scas-nosft-${SIZE}_pretrained-math500-passk-eosfix}"

echo "Host=$(hostname) HF_MODEL=${HF_MODEL} SIZE=${SIZE} CKPT=${CKPT}"
echo "OUT_DIR=${OUT_DIR} EOS_FIX=${EOS_FIX} SCAS_MAX_LEN=${SCAS_MAX_LEN} T=${TEMPERATURE} top_p=${TOP_P}"
nvidia-smi -L || true
[[ -f "${OUT_DIR}/pass_at_k.json" ]] && { echo "SKIP already evaluated: ${OUT_DIR}"; exit 0; }
mkdir -p "${OUT_DIR}"

./.venv/bin/python scripts/eval/eval_math_standard_passk_qwen.py \
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
