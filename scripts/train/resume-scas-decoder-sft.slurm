#!/usr/bin/env bash
#SBATCH --job-name=scas-resume
#SBATCH --output=slurm_logs/scas-resume-%j.out
#SBATCH --error=slurm_logs/scas-resume-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu  # set to your site partition
#SBATCH --qos=normal  # set to your site QoS (or remove)
#SBATCH --time=48:00:00
#
# Resume SCAS decoder SFT for EXTRA_EPOCHS more epochs (default 10).
# Also appends to the original parent job log files.
#
# Usage:
#   sbatch --export=ALL,EXTRA_EPOCHS=10,PARENT_JOB=25085277 \
#     scripts/train/resume-scas-decoder-sft.slurm standard qwen3-4b

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve repo root whether script lives in scripts/ or scripts/pipelines/
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then source "${SCRIPT_DIR}/env.sh"; elif [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; fi
cd "${REPO_ROOT}"

VARIANT="${1:-${VARIANT:-}}"
SIZE="${2:-${SIZE:-qwen3-4b}}"
PARENT_JOB="${PARENT_JOB:-}"
EXTRA_EPOCHS="${EXTRA_EPOCHS:-10}"
BASE_EPOCHS="${BASE_EPOCHS:-4}"
if [ -n "${TOTAL_EPOCHS:-}" ]; then
  : # use explicit total (e.g. TOTAL_EPOCHS=40)
else
  TOTAL_EPOCHS=$((BASE_EPOCHS + EXTRA_EPOCHS))
fi

if [ -z "${VARIANT}" ] || { [ "${VARIANT}" != "standard" ] && [ "${VARIANT}" != "modc" ]; }; then
  echo "ERROR: VARIANT must be standard or modc (got: ${VARIANT:-<unset>})"
  exit 1
fi

if [ -z "${PARENT_JOB}" ]; then
  echo "ERROR: PARENT_JOB not set. Use sbatch --export=ALL,PARENT_JOB=<id> ..."
  exit 1
fi

case "${VARIANT}" in
  standard)
    LOG_OUT="slurm_logs/scas-std-q25-${PARENT_JOB}.out"
    LOG_ERR="slurm_logs/scas-std-q25-${PARENT_JOB}.err"
    ;;
  modc)
    LOG_OUT="slurm_logs/scas-modc-q25-${PARENT_JOB}.out"
    LOG_ERR="slurm_logs/scas-modc-q25-${PARENT_JOB}.err"
    ;;
esac

mkdir -p slurm_logs
{
  echo ""
  echo "========== RESUME ${VARIANT} total_epochs=${TOTAL_EPOCHS} (base=${BASE_EPOCHS}) $(date -Is) job=${SLURM_JOB_ID} parent=${PARENT_JOB} =========="
} >> "${LOG_OUT}"

source scripts/scas_model_size.sh "${SIZE}"
source ./.venv/bin/activate
unset PYTHONPATH PYTHONPATH_modshare 2>/dev/null || true
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-disentangling-reasoning-scas}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

case "${VARIANT}" in
  standard)
    DATA_DIR="${STANDARD_SFT_DIR}"
    CKPT_DIR="${STANDARD_CKPT_DIR}"
    export WANDB_RUN_NAME="scas-standard-${MODEL_SIZE}"
    ;;
  modc)
    DATA_DIR="${MODC_SFT_DIR}"
    CKPT_DIR="${MODC_CKPT_DIR}"
    export WANDB_RUN_NAME="scas-modc-prefix-${MODEL_SIZE}"
    ;;
esac

{
  echo "Host: $(hostname)"
  echo "Variant=${VARIANT} size=${MODEL_SIZE} total_epochs=${TOTAL_EPOCHS} (base=${BASE_EPOCHS})"
  echo "Checkpoint dir: ${CKPT_DIR}"
  echo "Data: ${DATA_DIR}"
} >> "${LOG_OUT}" 2>> "${LOG_ERR}"

if ! compgen -G "${CKPT_DIR}/checkpoint-*" > /dev/null; then
  echo "No checkpoints in ${CKPT_DIR}" >> "${LOG_OUT}"
  exit 1
fi

for f in "${DATA_DIR}/.hf_tokenized/train/dataset_info.json" \
         "${DATA_DIR}/.hf_tokenized/validation/dataset_info.json"; do
  if [ ! -f "${f}" ]; then
    echo "Missing tokenized cache ${f}" >> "${LOG_OUT}"
    exit 1
  fi
done

VENV_PY="./.venv/bin/python"
TORCHRUN="${VENV_PY} -m torch.distributed.run"
mkdir -p "${CKPT_DIR}"

[ -z "${HF_TOKEN:-}" ] && [ -f "${HOME}/.cache/huggingface/token" ] && \
  export HF_TOKEN="$(tr -d '\n' < "${HOME}/.cache/huggingface/token")"

"${VENV_PY}" -c "import runpy; runpy.run_path('mod_c/train.py')['ensure_model_cached']('${DECODER_MODEL}')" \
  >> "${LOG_OUT}" 2>> "${LOG_ERR}"

${TORCHRUN} --standalone --nproc_per_node=8 mod_c/train.py \
  --dataset_name "${DATA_DIR}" \
  --dataset_train_split train.parquet \
  --dataset_test_split validation.parquet \
  --model_name_or_path "${DECODER_MODEL}" \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --max_seq_length "${SCAS_MAX_LEN}" \
  --learning_rate 1e-5 \
  --max_grad_norm 1.0 \
  --num_train_epochs "${TOTAL_EPOCHS}" \
  --gradient_checkpointing true \
  --gradient_accumulation_steps 8 \
  --logging_steps 10 \
  --save_strategy epoch \
  --eval_strategy epoch \
  --output_dir "${CKPT_DIR}" \
  --optim adamw_torch \
  --lr_scheduler_type constant \
  --warmup_steps 0 \
  --report_to wandb \
  --run_name "${WANDB_RUN_NAME}" \
  --bf16 true \
  --torch_dtype bfloat16 \
  --logging_first_step \
  --attn_implementation sdpa \
  --dataloader_num_workers 4 \
  --skip_upload_optimizer_states true \
  >> "${LOG_OUT}" 2>> "${LOG_ERR}"

echo "Done: ${CKPT_DIR} (epochs 1-${TOTAL_EPOCHS})" >> "${LOG_OUT}"
