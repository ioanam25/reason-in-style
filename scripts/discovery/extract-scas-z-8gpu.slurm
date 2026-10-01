#!/usr/bin/env bash
#SBATCH --job-name=scas-z8
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu  # set to your site partition
#SBATCH --qos=normal  # set to your site QoS (or remove)
#SBATCH --time=4:00:00
#
# Extract z(s) for one AE run: 8 shards, one per GPU, then merge.
#
# Primary 4B covz run:
#   sbatch --export=ALL,TAG=covz-qwen3-4b-final,CKPT=${ARCHIVE_ROOT}/checkpoints/scas-trace-only-qwen3-4b-vicreg-b8-covz/last.ckpt \
#     scripts/discovery/extract-scas-z-8gpu.slurm
#
# Or let TAG imply the checkpoint:
#   sbatch --export=ALL,TAG=covz-qwen3-4b \
#     scripts/discovery/extract-scas-z-8gpu.slurm

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
source scripts/scas_model_size.sh "${SIZE:-qwen3-4b}"
unset PYTHONPATH PYTHONPATH_modshare 2>/dev/null || true
export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1

TAG="${TAG:?set TAG e.g. covz-qwen3-4b-final}"

# Resolve checkpoint: explicit CKPT wins; else map common paper tags.
if [[ -z "${CKPT:-}" ]]; then
  case "$TAG" in
    covz-qwen3-4b|covz-qwen3-4b-final)
      CKPT_DIR="${ARCHIVE_ROOT}/checkpoints/scas-trace-only-qwen3-4b-vicreg-b8-covz" ;;
    covz-qwen3-1p7b|covz-qwen3-1p7b-final)
      CKPT_DIR="${ARCHIVE_ROOT}/checkpoints/scas-trace-only-qwen3-1p7b-vicreg-b8-covz" ;;
    covz-qwen3-0p6b|covz-qwen3-0p6b-final)
      CKPT_DIR="${ARCHIVE_ROOT}/checkpoints/scas-trace-only-qwen3-0p6b-vicreg-b8-covz" ;;
    *)
      CKPT_DIR="${ARCHIVE_ROOT}/checkpoints/scas-trace-only-qwen3-0p6b-vicreg-b8-${TAG}" ;;
  esac
  CKPT="${CKPT_DIR}/last.ckpt"
fi
[[ -f "$CKPT" ]] || { echo "Missing checkpoint: $CKPT"; exit 1; }

VENV_PY="./.venv/bin/python"
NUM_SHARDS=8
SPLIT="${SPLIT:-train}"
OUT_DIR="${ARCHIVE_ROOT}/zscore-variants/${TAG}"
mkdir -p "${OUT_DIR}/embedding_shards" slurm_logs eval_results

# Prefer baked-in style labels from scas_traces_dataset; raw JSON optional.
RAW_JSON="${SCAS_TRACE_JSON:-configs/scas/scas_traces_dataset.json}"
[[ -f "${RAW_JSON}" ]] || RAW_JSON="/dev/null"

echo "Host: $(hostname)  TAG=${TAG}"
echo "Checkpoint: ${CKPT}"
echo "Traces: ${TRACE_DATA_PATH}"
echo "Output: ${OUT_DIR}"
nvidia-smi -L

TMPDIR_LOCAL="${SLURM_TMPDIR:-/tmp}"
TMP_CKPT="${TMPDIR_LOCAL}/scas-z-${TAG}-${SLURM_JOB_ID}.ckpt"
cp -f "$CKPT" "$TMP_CKPT"
trap 'rm -f "$TMP_CKPT"' EXIT

run_shard() {
  local sid="$1"
  CUDA_VISIBLE_DEVICES="${sid}" "${VENV_PY}" -u scripts/discovery/extract_z_embeddings.py \
    --checkpoint "${TMP_CKPT}" \
    --hf-dataset "${TRACE_DATA_PATH}" \
    --raw-json "${RAW_JSON}" \
    --output-dir "${OUT_DIR}" \
    --split "${SPLIT}" \
    --shard-id "${sid}" \
    --num-shards "${NUM_SHARDS}" \
    --batch-size 8 \
    --device cuda \
    > "slurm_logs/scas-z8-${TAG}-shard${sid}-${SLURM_JOB_ID}.log" 2>&1
}

pids=()
for sid in $(seq 0 $((NUM_SHARDS - 1))); do
  run_shard "${sid}" &
  pids+=("$!")
  sleep 10
done

fail=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "shard ${i} failed"; fail=1; }
done

if [[ "${fail}" -ne 0 ]]; then
  for sid in $(seq 0 $((NUM_SHARDS - 1))); do
    f="${OUT_DIR}/embedding_shards/shard_$(printf '%04d' "${sid}")_of_$(printf '%04d' "${NUM_SHARDS}").npz"
    [[ -f "$f" ]] && continue
    echo "retrying shard ${sid}"
    run_shard "${sid}" || echo "shard ${sid} failed again"
  done
fi

missing=0
for sid in $(seq 0 $((NUM_SHARDS - 1))); do
  f="${OUT_DIR}/embedding_shards/shard_$(printf '%04d' "${sid}")_of_$(printf '%04d' "${NUM_SHARDS}").npz"
  [[ -f "$f" ]] || { echo "MISSING ${f}"; missing=1; }
done
[[ "${missing}" -eq 0 ]] || exit 1

echo "Merging shards..."
"${VENV_PY}" -u scripts/discovery/merge_z_embedding_shards.py --output-dir "${OUT_DIR}" \
  --output-npz "${OUT_DIR}/embeddings_${SPLIT}_z.npz"

echo "DONE TAG=${TAG} split=${SPLIT} -> ${OUT_DIR}/embeddings_${SPLIT}_z.npz"
