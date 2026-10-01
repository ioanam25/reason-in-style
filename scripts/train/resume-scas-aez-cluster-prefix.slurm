#!/usr/bin/env bash
#SBATCH --job-name=aez-pfx-tr
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
# Resume cluster-prefix SFT for more epochs (auto-resumes if checkpoint-* exists).
#   sbatch --export=ALL,VAR=covz-qwen3,K=4,SIZE=qwen3-0p6b,CKPT_TAG=qwen3-opt,NUM_EPOCHS=20 \
#     scripts/train/train-scas-aez-cluster-prefix.slurm
#
# (This file intentionally mirrors train-scas-aez-cluster-prefix.slurm with NUM_EPOCHS.)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
source ./.venv/bin/activate
SIZE="${SIZE:-qwen3-4b}"
source scripts/scas_model_size.sh "${SIZE}"
unset PYTHONPATH PYTHONPATH_modshare 2>/dev/null || true

VAR="${VAR:?set VAR}"
K="${K:?set K}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"

export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-disentangling-reasoning-scas}"
export WANDB_DIR=${WANDB_DIR}
export WANDB_RUN_NAME="scas-aez-${VAR}-k${K}-${MODEL_SIZE}-opt-e${NUM_EPOCHS}"
mkdir -p "$WANDB_DIR" ${SLURM_LOG_DIR}

SROOT="${ARCHIVE_ROOT}"
CLUSTER_SFT_DIR="${SROOT}/cluster_prefix/${VAR}/k${K}"
CKPT_TAG="${CKPT_TAG:-opt}"
CLUSTER_CKPT_DIR="${SROOT}/cluster_ckpt/${VAR}/k${K}-${CKPT_TAG}"

VENV_PY="./.venv/bin/python"
TORCHRUN="${VENV_PY} -m torch.distributed.run"
mkdir -p "${CLUSTER_CKPT_DIR}"

for f in "${CLUSTER_SFT_DIR}/train.parquet" \
         "${CLUSTER_SFT_DIR}/.hf_tokenized/train/dataset_info.json" \
         "${CLUSTER_SFT_DIR}/.hf_tokenized/validation/dataset_info.json"; do
  [[ -f "${f}" ]] || { echo "Missing ${f}"; exit 1; }
done

LAST="$(ls -1 "${CLUSTER_CKPT_DIR}" 2>/dev/null | grep -E "^checkpoint-[0-9]+$" | sort -t- -k2 -n | tail -1 || true)"
[[ -n "${LAST}" ]] || { echo "No checkpoint in ${CLUSTER_CKPT_DIR} to resume from"; exit 1; }

[ -z "${HF_TOKEN:-}" ] && [ -f "${HOME}/.cache/huggingface/token" ] && \
  export HF_TOKEN="$(tr -d '\n' < "${HOME}/.cache/huggingface/token")"

echo "Host: $(hostname)"
echo "RESUME VAR=${VAR} K=${K} model=${DECODER_MODEL} epochs=${NUM_EPOCHS}"
echo "from=${CLUSTER_CKPT_DIR}/${LAST}"
echo "data=${CLUSTER_SFT_DIR}"

${TORCHRUN} --standalone --nproc_per_node=8 mod_c/train.py \
  --dataset_name "${CLUSTER_SFT_DIR}" \
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
  --output_dir "${CLUSTER_CKPT_DIR}" \
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

echo "Done resume: ${CLUSTER_CKPT_DIR}"
