#!/usr/bin/env bash
#SBATCH --job-name=scas-base-res
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu  # set to your site partition
#SBATCH --qos=normal  # set to your site QoS (or remove)
#SBATCH --time=1-12:00:00
#
# Resume vanilla or modc-style Qwen3 baseline to NUM_EPOCHS (default 20).
#   sbatch --export=ALL,ARM=standard,NUM_EPOCHS=20 scripts/train/resume-scas-qwen3-baseline.slurm
#   sbatch --export=ALL,ARM=modc-style,NUM_EPOCHS=20 scripts/train/resume-scas-qwen3-baseline.slurm
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
source ./.venv/bin/activate
source scripts/scas_model_size.sh qwen3-0p6b
unset PYTHONPATH PYTHONPATH_modshare 2>/dev/null || true

ARM="${ARM:?set ARM=standard|modc-style}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
SROOT="${ARCHIVE_ROOT}"

case "${ARM}" in
  standard)
    DATA_DIR="${SROOT}/baselines/standard-qwen3"
    CKPT_DIR="${SROOT}/baselines/ckpt/standard-qwen3-opt"
    RUN="scas-standard-qwen3-0p6b-opt-e${NUM_EPOCHS}"
    ;;
  modc-style)
    DATA_DIR="${SROOT}/baselines/modc-style-qwen3"
    CKPT_DIR="${SROOT}/baselines/ckpt/modc-style-qwen3-opt"
    RUN="scas-modc-style-qwen3-0p6b-opt-e${NUM_EPOCHS}"
    ;;
  *)
    echo "Unknown ARM=${ARM}"; exit 1
    ;;
esac

export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-disentangling-reasoning-scas}"
export WANDB_DIR=${WANDB_DIR}
export WANDB_RUN_NAME="${RUN}"
mkdir -p "$WANDB_DIR" ${SLURM_LOG_DIR} "${CKPT_DIR}"

LAST="$(ls -1 "${CKPT_DIR}" 2>/dev/null | grep -E "^checkpoint-[0-9]+$" | sort -t- -k2 -n | tail -1 || true)"
[[ -n "${LAST}" ]] || { echo "No checkpoint in ${CKPT_DIR}"; exit 1; }

[ -z "${HF_TOKEN:-}" ] && [ -f "${HOME}/.cache/huggingface/token" ] && \
  export HF_TOKEN="$(tr -d '\n' < "${HOME}/.cache/huggingface/token")"

echo "RESUME ARM=${ARM} from=${CKPT_DIR}/${LAST} epochs=${NUM_EPOCHS}"

./.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 mod_c/train.py \
  --dataset_name "${DATA_DIR}" \
  --dataset_train_split train.parquet \
  --dataset_test_split validation.parquet \
  --model_name_or_path "${DECODER_MODEL}" \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --max_seq_length "${SCAS_MAX_LEN}" \
  --learning_rate 1e-5 \
  --max_grad_norm 1.0 \
  --num_train_epochs "${NUM_EPOCHS}" \
  --gradient_checkpointing false \
  --gradient_accumulation_steps 16 \
  --logging_steps 10 \
  --save_strategy epoch \
  --save_total_limit 3 \
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
  --attn_implementation flash_attention_2 \
  --packing true \
  --dataloader_num_workers 4 \
  --dataloader_pin_memory true \
  --skip_upload_optimizer_states true

echo "Done: ${CKPT_DIR}"
