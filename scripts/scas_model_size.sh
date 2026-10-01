#!/usr/bin/env bash
# Source after: SIZE=qwen3-4b  or  source scripts/scas_model_size.sh qwen3-4b
# Sets decoder model + SCAS SFT paths.

_scas_model_size() {
  # shellcheck source=scripts/repo_paths.sh
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/repo_paths.sh"
  _repo_paths_init

  local SIZE="${1:-qwen3-4b}"
  case "${SIZE}" in
    0p6b|qwen3-0p6b|qwen3_0p6b)
      MODEL_SIZE="qwen3-0p6b"
      DECODER_MODEL="Qwen/Qwen3-0.6B"
      MODEL_LABEL="Qwen3-0.6B"
      ;;
    0p6b-base|qwen3-0p6b-base|qwen3_0p6b_base)
      MODEL_SIZE="qwen3-0p6b-base"
      DECODER_MODEL="Qwen/Qwen3-0.6B-Base"
      MODEL_LABEL="Qwen3-0.6B-Base"
      ;;
    1p7b|qwen3-1p7b|qwen3_1p7b|1.7b)
      MODEL_SIZE="qwen3-1p7b"
      DECODER_MODEL="Qwen/Qwen3-1.7B"
      MODEL_LABEL="Qwen3-1.7B"
      ;;
    # --- init-ladder: true base / instruct / RL-thinking ---
    1p7b-base|qwen3-1p7b-base|qwen3_1p7b_base)
      MODEL_SIZE="qwen3-1p7b-base"
      DECODER_MODEL="Qwen/Qwen3-1.7B-Base"
      MODEL_LABEL="Qwen3-1.7B-Base"
      ;;
    4b|qwen3-4b|qwen3_4b)
      MODEL_SIZE="qwen3-4b"
      DECODER_MODEL="Qwen/Qwen3-4B"
      MODEL_LABEL="Qwen3-4B"
      ;;
    4b-base|qwen3-4b-base|qwen3_4b_base)
      MODEL_SIZE="qwen3-4b-base"
      DECODER_MODEL="Qwen/Qwen3-4B-Base"
      MODEL_LABEL="Qwen3-4B-Base"
      ;;
    4b-instruct|qwen3-4b-instruct|qwen3_4b_instruct)
      MODEL_SIZE="qwen3-4b-instruct"
      DECODER_MODEL="Qwen/Qwen3-4B-Instruct-2507"
      MODEL_LABEL="Qwen3-4B-Instruct-2507"
      ;;
    4b-thinking|qwen3-4b-thinking|qwen3_4b_thinking)
      MODEL_SIZE="qwen3-4b-thinking"
      DECODER_MODEL="Qwen/Qwen3-4B-Thinking-2507"
      MODEL_LABEL="Qwen3-4B-Thinking-2507"
      ;;
    *)
      echo "Unknown size: ${SIZE} (use qwen3-0p6b[-base], qwen3-1p7b[-base], qwen3-4b[-base|-instruct|-thinking])" >&2
      return 1
      ;;
  esac

  STANDARD_SFT_DIR="data/scas/standard_full_sft"
  MODC_SFT_DIR="data/scas/modc_prefix_full_sft"
  TRACE_DATA_PATH="${DATA_ROOT:-.}/scas_traces_dataset"
  _SCAS_TAG="${MODEL_SIZE}"
  STANDARD_CKPT_DIR="checkpoints/scas-standard-${_SCAS_TAG}"
  MODC_CKPT_DIR="checkpoints/scas-modc-prefix-${_SCAS_TAG}"
  TRACE_CKPT_DIR="checkpoints/scas-trace-only-${_SCAS_TAG}"
  GMM_ASSIGN_DIR="data/scas/cluster_gmm_assignments-${_SCAS_TAG}"
  SCAS_VAL_PARQUET="data/scas/standard_full_sft/validation.parquet"
  SCAS_SFT_JSON="${CONFIG_SCAS}/scas_standard_full_sft.json"
  SCAS_MODC_SFT_JSON="${CONFIG_SCAS}/scas_modc_prefix_full_sft.json"
  SCAS_TRACE_JSON="${CONFIG_SCAS}/scas_modc_dataset.json"
  SCAS_HF_DATASET="Student-Centric-Answer-Sampling/scas_verified_teacher_pool"
  SCAS_MAX_LEN=4096
  TRACE_MAX_EPOCHS=4
  TRACE_MAX_LEN=4096
  TRACE_ACCUM=2
  TRACE_STRATEGY="ddp_find_unused_parameters_true"
  TRACE_PRECISION="bf16-mixed"
  TRACE_MICRO_BATCH=0

  USER_HF_CACHE="${HF_CACHE}"
  export HF_HOME="${USER_HF_CACHE}"
  export HF_HUB_CACHE="${USER_HF_CACHE}"
  export HF_DATASETS_CACHE="${USER_HF_CACHE}/datasets"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  _scas_model_size "${1:-qwen3-4b}"
  echo "MODEL_SIZE=${MODEL_SIZE}"
  echo "DECODER_MODEL=${DECODER_MODEL}"
  echo "STANDARD_SFT_DIR=${STANDARD_SFT_DIR}"
  echo "MODC_SFT_DIR=${MODC_SFT_DIR}"
  echo "TRACE_DATA_PATH=${TRACE_DATA_PATH}"
  echo "STANDARD_CKPT_DIR=${STANDARD_CKPT_DIR}"
  echo "MODC_CKPT_DIR=${MODC_CKPT_DIR}"
  echo "TRACE_CKPT_DIR=${TRACE_CKPT_DIR}"
else
  _scas_model_size "${1:-${SIZE:-qwen3-4b}}"
fi
