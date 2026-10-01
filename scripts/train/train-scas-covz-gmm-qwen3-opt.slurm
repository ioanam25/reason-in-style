#!/usr/bin/env bash
#SBATCH --job-name=covz-gmm-q3-tr
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
# GMM cluster-prefix SFT on Qwen3-0.6B (same opt recipe as covz-qwen3).
#   sbatch --export=ALL,K=21,NUM_EPOCHS=50 scripts/train/train-scas-covz-gmm-qwen3-opt.slurm
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
source ./.venv/bin/activate
source scripts/scas_model_size.sh qwen3-0p6b
unset PYTHONPATH PYTHONPATH_modshare 2>/dev/null || true

K="${K:?set K}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"

export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-disentangling-reasoning-scas}"
export WANDB_DIR=${WANDB_DIR}
export WANDB_RUN_NAME="scas-covz-gmm-k${K}-qwen3-0p6b-opt-e${NUM_EPOCHS}"
mkdir -p "$WANDB_DIR" ${SLURM_LOG_DIR}

SROOT="${ARCHIVE_ROOT}"
DATA_DIR="${SROOT}/cluster_prefix/covz-gmm-qwen3/k${K}"
CKPT_DIR="${SROOT}/cluster_ckpt/covz-gmm-qwen3/k${K}-opt"
VENV_PY="./.venv/bin/python"
TORCHRUN="${VENV_PY} -m torch.distributed.run"
mkdir -p "${CKPT_DIR}"

for f in "${DATA_DIR}/train.parquet" \
         "${DATA_DIR}/.hf_tokenized/train/dataset_info.json" \
         "${DATA_DIR}/.hf_tokenized/validation/dataset_info.json"; do
  [[ -f "${f}" ]] || { echo "Missing ${f}"; exit 1; }
done

[ -z "${HF_TOKEN:-}" ] && [ -f "${HOME}/.cache/huggingface/token" ] && \
  export HF_TOKEN="$(tr -d '\n' < "${HOME}/.cache/huggingface/token")"

echo "Host: $(hostname)"
echo "COVZ-GMM SFT K=${K} model=${DECODER_MODEL} epochs=${NUM_EPOCHS}"
echo "data=${DATA_DIR} out=${CKPT_DIR}"
echo "batch=2 accum=16 eff=256 gc=off flash_attn packing=1"

${TORCHRUN} --standalone --nproc_per_node=8 mod_c/train.py \
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
