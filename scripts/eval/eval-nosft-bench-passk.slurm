#!/usr/bin/env bash
#SBATCH --job-name=bench-nosft
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
# No-SFT AMC + OlympiadBench Pass@k (EOS_FIX) for a HuggingFace Qwen3 checkpoint.
#   sbatch --export=ALL,HF_MODEL=Qwen/Qwen3-4B-Instruct-2507,SIZE=qwen3-4b-instruct \
#     scripts/eval/eval-nosft-bench-passk.slurm
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
HF_MODEL="${HF_MODEL:?set HF_MODEL}"
SIZE="${SIZE:?set SIZE}"
source scripts/scas_model_size.sh "${SIZE}"

SROOT="${ARCHIVE_ROOT}"
export EOS_FIX="${EOS_FIX:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
if [[ "${EOS_FIX}" == "1" ]]; then
  OUT_TAG="${OUT_TAG--eosfix}"
else
  OUT_TAG="${OUT_TAG-}"
fi
BENCHES="${BENCHES:-amc:amc olympiadbench:olympiadbench-en amc25:amc12-2025}"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export HF_HOME="${HF_HOME:-${HF_CACHE}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

CKPT="$(./.venv/bin/python - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download("${HF_MODEL}"))
PY
)"
STAGE="${STAGE:-all}"
echo "Host=$(hostname) HF_MODEL=${HF_MODEL} SIZE=${SIZE} CKPT=${CKPT} STAGE=${STAGE}"
if [[ "${STAGE}" == "grade" ]]; then
  export CUDA_VISIBLE_DEVICES=""
else
  nvidia-smi -L || true
fi

echo "Host=$(hostname) HF_MODEL=${HF_MODEL} SIZE=${SIZE} T=${TEMPERATURE} top_p=${TOP_P} BENCHES=${BENCHES} OUT_TAG=${OUT_TAG} STAGE=${STAGE}"
fail=0
for spec in ${BENCHES}; do
  HF_DATASET="${spec%%:*}"
  BENCH_TAG="${spec##*:}"
  THIS_OUT="${SROOT}/model-evals/scas-nosft-${SIZE}_pretrained-${BENCH_TAG}-passk${OUT_TAG}"
  echo "======== BENCH=${BENCH_TAG} OUT=${THIS_OUT} ========"
  if [[ -f "${THIS_OUT}/pass_at_k.json" ]]; then
    echo "SKIP graded ${THIS_OUT}"
    continue
  fi
  if [[ "${STAGE}" == "generate" && -f "${THIS_OUT}/generations.done" ]]; then
    echo "SKIP generate ${THIS_OUT}"
    continue
  fi
  if [[ "${STAGE}" == "grade" && ! -f "${THIS_OUT}/generations.jsonl" ]]; then
    echo "MISSING generations ${THIS_OUT}"
    fail=1
    continue
  fi
  mkdir -p "${THIS_OUT}"
  if ! ./.venv/bin/python scripts/eval/eval_math_standard_passk_qwen.py \
      --checkpoint "${CKPT}" \
      --hf-dataset "${HF_DATASET}" \
      --output-dir "${THIS_OUT}" \
      --samples-per-problem 256 \
      --tensor-parallel-size 1 \
      --data-parallel-replicas -1 \
      --max-tokens "${SCAS_MAX_LEN}" \
      --temperature "${TEMPERATURE}" \
      --top-p "${TOP_P}" \
      --budgets 1 2 4 8 16 32 64 128 256 \
      --grade-timeout 2.0 \
      --grade-flush-every 100 \
      --stage "${STAGE}"; then
    echo "FAIL ${BENCH_TAG}"
    fail=1
  fi
done
exit ${fail}
