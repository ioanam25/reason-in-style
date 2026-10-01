#!/usr/bin/env bash
#SBATCH --job-name=pfx-arm-sft
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu  # set to your site partition
#SBATCH --qos=normal  # set to your site QoS (or remove)
#SBATCH --time=2-00:00:00
#
# Generic style-prefix SFT for any control / method arm, any student size.
# Same optimizer recipe as train-scas-ae-gmm-ladder-opt.slurm so arms are comparable.
#
#   sbatch --export=ALL,ARM=random-qwen3,SIZE=qwen3-4b-thinking,K=6 \
#     scripts/train/train-scas-prefix-arm-opt.slurm
#
# Reads : ${SROOT}/cluster_prefix/${ARM}-${MODEL_SIZE}/k${K}
# Writes: ${SROOT}/cluster_ckpt/${ARM}-${MODEL_SIZE}/k${K}-opt
# Auto-resumes from checkpoint-* in the output dir (mod_c/train.py).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
source ./.venv/bin/activate
SIZE="${SIZE:?set SIZE}"
ARM="${ARM:?set ARM e.g. random-qwen3|const1|modc9|covz-qwen3-4b-final-gmm-bal}"
source scripts/scas_model_size.sh "${SIZE}"
# Allow pointing at a modified base (e.g. one with <style_i> special tokens).
DECODER_MODEL="${MODEL_OVERRIDE:-$DECODER_MODEL}"
unset PYTHONPATH PYTHONPATH_modshare 2>/dev/null || true

K="${K:-6}"
SEED="${SEED:-42}"
SROOT="${ARCHIVE_ROOT}"
DATA_DIR="${SROOT}/cluster_prefix/${ARM}-${MODEL_SIZE}/k${K}"
SUFFIX="opt"
[[ "${SEED}" != "42" ]] && SUFFIX="opt-s${SEED}"
CKPT_DIR="${CKPT_DIR:-${SROOT}/cluster_ckpt/${ARM}-${MODEL_SIZE}/k${K}-${SUFFIX}}"

case "${MODEL_SIZE}" in
  qwen3-0p6b|qwen3-0p6b-base)
    BATCH=2; ACCUM=16; GC=false; NUM_EPOCHS="${NUM_EPOCHS:-15}"; EVAL_BS=2 ;;
  *)
    BATCH=1; ACCUM=32; GC=true; NUM_EPOCHS="${NUM_EPOCHS:-50}"; EVAL_BS=1 ;;
esac

export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-disentangling-reasoning-scas}"
export WANDB_DIR=${WANDB_DIR}
export WANDB_RUN_NAME="scas-${ARM}-k${K}-${MODEL_SIZE}-${SUFFIX}-e${NUM_EPOCHS}"
mkdir -p "$WANDB_DIR" ${SLURM_LOG_DIR} "${CKPT_DIR}"

for f in "${DATA_DIR}/train.parquet" \
         "${DATA_DIR}/.hf_tokenized/train/dataset_info.json" \
         "${DATA_DIR}/.hf_tokenized/validation/dataset_info.json"; do
  [[ -f "${f}" ]] || { echo "Missing ${f}"; exit 1; }
done

[ -z "${HF_TOKEN:-}" ] && [ -f "${HOME}/.cache/huggingface/token" ] && \
  export HF_TOKEN="$(tr -d '\n' < "${HOME}/.cache/huggingface/token")"

echo "Host: $(hostname)"
echo "ARM=${ARM} SIZE=${MODEL_SIZE} K=${K} SEED=${SEED} model=${DECODER_MODEL} epochs=${NUM_EPOCHS}"
echo "data=${DATA_DIR} out=${CKPT_DIR}"
PACKING="${PACKING:-true}"
if [[ "${PACKING}" == "false" || "${PACKING}" == "0" ]]; then
  GROUP_BY_LENGTH="${GROUP_BY_LENGTH:-true}"
else
  GROUP_BY_LENGTH="${GROUP_BY_LENGTH:-false}"
fi
echo "batch=${BATCH} accum=${ACCUM} eff=$((BATCH*ACCUM*8)) gc=${GC} packing=${PACKING} group_by_length=${GROUP_BY_LENGTH}"
nvidia-smi -L || true

./.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 mod_c/train.py \
  --dataset_name "${DATA_DIR}" \
  --dataset_train_split train.parquet \
  --dataset_test_split validation.parquet \
  --model_name_or_path "${DECODER_MODEL}" \
  --per_device_train_batch_size "${BATCH}" \
  --per_device_eval_batch_size "${EVAL_BS}" \
  --max_seq_length "${SCAS_MAX_LEN}" \
  --learning_rate 1e-5 \
  --max_grad_norm 1.0 \
  --num_train_epochs "${NUM_EPOCHS}" \
  --gradient_checkpointing "${GC}" \
  --gradient_accumulation_steps "${ACCUM}" \
  --logging_steps 10 \
  --save_strategy epoch \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --eval_strategy epoch \
  --load_best_model_at_end true \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --output_dir "${CKPT_DIR}" \
  --optim adamw_torch \
  --lr_scheduler_type constant \
  --warmup_steps 0 \
  --seed "${SEED}" \
  --report_to wandb \
  --run_name "${WANDB_RUN_NAME}" \
  --bf16 true \
  --torch_dtype bfloat16 \
  --logging_first_step \
  --attn_implementation flash_attention_2 \
  --packing "${PACKING}" \
  --group_by_length "${GROUP_BY_LENGTH}" \
  --dataloader_num_workers 4 \
  --dataloader_pin_memory true \
  --skip_upload_optimizer_states true

echo "Done: ${CKPT_DIR}"
