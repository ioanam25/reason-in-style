#!/usr/bin/env bash
#SBATCH --job-name=scas-init-tr
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
# Init-ladder SFT (base / instruct / thinking).
#   sbatch --export=ALL,SIZE=qwen3-1p7b-base,ARM=covz,K=6,NUM_EPOCHS=50 scripts/train/train-scas-init-ladder-opt.slurm
#   sbatch --export=ALL,SIZE=qwen3-4b-instruct,ARM=standard,NUM_EPOCHS=50 scripts/train/train-scas-init-ladder-opt.slurm
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
source ./.venv/bin/activate
SIZE="${SIZE:?set SIZE}"
source scripts/scas_model_size.sh "${SIZE}"
unset PYTHONPATH PYTHONPATH_modshare 2>/dev/null || true

ARM="${ARM:?set ARM=covz|standard}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
SEED="${SEED:-42}"
SROOT="${ARCHIVE_ROOT}"

case "${ARM}" in
  covz)
    K="${K:?set K for covz}"
    DATA_DIR="${SROOT}/cluster_prefix/covz-${MODEL_SIZE}/k${K}"
    SUFFIX="opt"; [[ "${SEED}" != "42" ]] && SUFFIX="opt-s${SEED}"
    CKPT_DIR="${SROOT}/cluster_ckpt/covz-${MODEL_SIZE}/k${K}-${SUFFIX}"
    RUN="scas-aez-covz-k${K}-${MODEL_SIZE}-${SUFFIX}-e${NUM_EPOCHS}"
    ;;
  standard)
    DATA_DIR="${SROOT}/baselines/standard-${MODEL_SIZE}"
    SUFFIX="opt"; [[ "${SEED}" != "42" ]] && SUFFIX="opt-s${SEED}"
    CKPT_DIR="${SROOT}/baselines/ckpt/standard-${MODEL_SIZE}-${SUFFIX}"
    RUN="scas-standard-${MODEL_SIZE}-${SUFFIX}-e${NUM_EPOCHS}"
    ;;
  *)
    echo "Unknown ARM=${ARM}"; exit 1
    ;;
esac

# Memory recipe: 1.7B and 4B both use microbatch=1 + grad ckpt (DDP full weights).
BATCH=1
ACCUM=32
GC=true

export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-disentangling-reasoning-scas}"
export WANDB_DIR=${WANDB_DIR}
export WANDB_RUN_NAME="${RUN}"
mkdir -p "$WANDB_DIR" ${SLURM_LOG_DIR} "${CKPT_DIR}"

for f in "${DATA_DIR}/train.parquet" \
         "${DATA_DIR}/.hf_tokenized/train/dataset_info.json" \
         "${DATA_DIR}/.hf_tokenized/validation/dataset_info.json"; do
  [[ -f "${f}" ]] || { echo "Missing ${f}"; exit 1; }
done

[ -z "${HF_TOKEN:-}" ] && [ -f "${HOME}/.cache/huggingface/token" ] && \
  export HF_TOKEN="$(tr -d '\n' < "${HOME}/.cache/huggingface/token")"

echo "Host: $(hostname)"
echo "SIZE=${MODEL_SIZE} ARM=${ARM} K=${K:--} SEED=${SEED} model=${DECODER_MODEL} epochs=${NUM_EPOCHS}"
echo "data=${DATA_DIR} out=${CKPT_DIR}"
echo "MEMSAFE: batch=${BATCH} accum=${ACCUM} eff=$((BATCH*ACCUM*8)) gc=${GC} flash_attn packing=1"
nvidia-smi -L || true

./.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 mod_c/train.py \
  --dataset_name "${DATA_DIR}" \
  --dataset_train_split train.parquet \
  --dataset_test_split validation.parquet \
  --model_name_or_path "${DECODER_MODEL}" \
  --per_device_train_batch_size "${BATCH}" \
  --per_device_eval_batch_size 1 \
  --max_seq_length "${SCAS_MAX_LEN}" \
  --learning_rate 1e-5 \
  --max_grad_norm 1.0 \
  --num_train_epochs "${NUM_EPOCHS}" \
  --gradient_checkpointing "${GC}" \
  --gradient_accumulation_steps "${ACCUM}" \
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
  --seed "${SEED}" \
  --bf16 true \
  --torch_dtype bfloat16 \
  --logging_first_step \
  --attn_implementation flash_attention_2 \
  --packing true \
  --dataloader_num_workers 4 \
  --dataloader_pin_memory true \
  --skip_upload_optimizer_states true

echo "Done: ${CKPT_DIR}"
