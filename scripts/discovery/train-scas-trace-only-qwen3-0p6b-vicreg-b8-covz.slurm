#!/usr/bin/env bash
#SBATCH --job-name=scas-trace-q3-covz
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu  # set to your site partition
#SBATCH --qos=normal  # set to your site QoS (or remove)
#SBATCH --time=72:00:00
#
# Setup A v4 ARM 5 - VICReg variance/covariance on z(s) ONLY (d=256).
#   src/module.py applied both regularizers to c(s) as well as z(s). Covariance
#   has d(d-1)/2 off-diagonal terms: 496 at d=32 but 32640 at d=256. At d=256 it
#   swamps the InfoNCE gradient and the content head collapses - measured l_c
#   rose 5.05 -> 5.63 over epochs 1-6, while chance is ln(127) = 4.84, i.e. it
#   ended worse than random. z32 (d=32) held l_c at 4.21, below chance.
#   This arm keeps the wide d=256 latent but takes the VICReg pressure off
#   c(s), isolating the mechanism. If l_c now falls below chance, dimension was
#   never the real variable.
#
# Budget: 4 epochs. Across z32/d64/b8-cont, l_rec is 90% converged after epoch 0
# and moves <0.02 thereafter, and z_eff_rank_frac reaches ~94% of its epoch-5
# value by epoch 3. Epochs 5-8 bought nothing measurable.
#
# save-top-k -1 keeps every epoch. val/loss is a broken selection metric here:
# it RISES monotonically because l_cov grows as the representation expands, so
# monitor=val/loss with top-1 silently discarded epochs 1-5 of the z32 run.
#
# Usage: sbatch scripts/discovery/train-scas-trace-only-qwen3-0p6b-vicreg-b8-covz.slurm

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"

source scripts/scas_model_size.sh qwen3-0p6b

TRACE_MAX_EPOCHS=4
TRACE_BATCH=8
TRACE_ACCUM=1
TRACE_MICRO_BATCH=4
LAMBDA_COV=0.15
LAMBDA_C=1.0
VICREG_APPLY_TO=z
D_C=256
D_Z=256
TRACE_CKPT_DIR="${ARCHIVE_ROOT}/checkpoints/scas-trace-only-qwen3-0p6b-vicreg-b8-covz"
WANDB_RUN_NAME="trace-only-qwen3-0p6b-scas-vicreg-b8-covz-8gpu"

unset PYTHONPATH PYTHONPATH_modshare 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR=${WANDB_DIR}
mkdir -p "$WANDB_DIR" ${SLURM_LOG_DIR}
export TOKENIZERS_PARALLELISM=false
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"

if [ -z "${HF_TOKEN:-}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
  export HF_TOKEN="$(tr -d "\n" < "${HOME}/.cache/huggingface/token")"
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set."
  exit 1
fi

VENV_PY="./.venv/bin/python"
DATA_PATH="${TRACE_DATA_PATH}"
mkdir -p "${TRACE_CKPT_DIR}"

if [ ! -d "${DATA_PATH}" ]; then
  echo "Dataset missing: ${DATA_PATH}."
  exit 1
fi

RESUME_ARGS=()
if [ -f "${TRACE_CKPT_DIR}/last.ckpt" ]; then
  echo "Resuming from ${TRACE_CKPT_DIR}/last.ckpt"
  RESUME_ARGS=(--resume-from "${TRACE_CKPT_DIR}/last.ckpt")
fi

MICRO_BATCH_ARGS=()
if [ "${TRACE_MICRO_BATCH:-0}" -gt 0 ]; then
  MICRO_BATCH_ARGS=(--micro-batch-size "${TRACE_MICRO_BATCH}")
fi

echo "Host: $(hostname)"
echo "Setup A v4 ARM5 (covz, d=256, vicreg on z only) | Size: ${MODEL_SIZE}  Decoder: ${DECODER_MODEL}"
echo "batch=${TRACE_BATCH} accum=${TRACE_ACCUM} micro=${TRACE_MICRO_BATCH} d=${D_Z} lambda_cov=${LAMBDA_COV} lambda_c=${LAMBDA_C} vicreg=${VICREG_APPLY_TO} epochs=${TRACE_MAX_EPOCHS}"
echo "Checkpoint dir: ${TRACE_CKPT_DIR}"
nvidia-smi -L || true

"${VENV_PY}" -c "
import runpy
ensure = runpy.run_path('mod_c/train.py')['ensure_model_cached']
ensure('Alibaba-NLP/gte-Qwen2-1.5B-instruct')
ensure('${DECODER_MODEL}')
"

srun "${VENV_PY}" main.py \
  --data-path "${DATA_PATH}" \
  --training-mode trace \
  --accelerator cuda \
  --devices 8 \
  --num-nodes 1 \
  --strategy "${TRACE_STRATEGY}" \
  --dist-timeout-seconds 7200 \
  --precision "${TRACE_PRECISION}" \
  --batch-size "${TRACE_BATCH}" \
  --accumulate-grad-batches "${TRACE_ACCUM}" \
  ${MICRO_BATCH_ARGS[@]+"${MICRO_BATCH_ARGS[@]}"} \
  --max-epochs "${TRACE_MAX_EPOCHS}" \
  --max-traces-per-question 0 \
  --num-workers 4 \
  --max-len "${TRACE_MAX_LEN}" \
  --val-max-len "${TRACE_MAX_LEN}" \
  --model-name "Alibaba-NLP/gte-Qwen2-1.5B-instruct" \
  --lora-r 8 \
  --gradient-checkpointing true \
  --decoder-type pretrained \
  --decoder-model-name "${DECODER_MODEL}" \
  --decoder-freeze-lm true \
  --decoder-lora-r 16 \
  --decoder-lora-alpha 32 \
  --decoder-lora-dropout 0.05 \
  --decoder-lora-target-modules q_proj k_proj v_proj o_proj \
  --decoder-gradient-checkpointing true \
  --num-prefix-tokens 8 \
  --d-c "${D_C}" \
  --d-z "${D_Z}" \
  --projection-layers 4 \
  --freeze-encoder-epochs 0 \
  --checkpoint-dir "${TRACE_CKPT_DIR}" \
  --save-top-k -1 \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
  --wandb true \
  --wandb-project disentangling-reasoning-scas \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --log-every-n-steps 50 \
  --lambda-c "${LAMBDA_C}" \
  --lambda-z 1.0 \
  --lambda-rec 1.0 \
  --lambda-var 1.0 \
  --lambda-div 1.0 \
  --vicreg-apply-to "${VICREG_APPLY_TO}" \
  --lambda-cov "${LAMBDA_COV}"

echo "Done: ${TRACE_CKPT_DIR}"
