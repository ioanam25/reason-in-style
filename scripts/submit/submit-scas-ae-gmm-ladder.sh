#!/usr/bin/env bash
# Submit AE-GMM K=6 prefix SFT ladder: build -> tokenize(SIZE) -> train(STYLE x SIZE).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve repo root whether script lives in scripts/ or scripts/pipelines/
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then source "${SCRIPT_DIR}/env.sh"; elif [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; fi
cd "${REPO_ROOT}"
mkdir -p ${SLURM_LOG_DIR}

K="${K:-6}"
STYLES_PLUS="${STYLES_PLUS:-covz-qwen3-1p7b+covz-qwen3-4b}"
# Students: 0.6B base+chat, 1.7B base+chat, 4B plain+base+instruct+thinking
SIZES=(
  qwen3-0p6b
  qwen3-0p6b-base
  qwen3-1p7b
  qwen3-1p7b-base
  qwen3-4b
  qwen3-4b-base
  qwen3-4b-instruct
  qwen3-4b-thinking
)
STYLES=(covz-qwen3-1p7b covz-qwen3-4b)

echo "Submitting AE-GMM K=${K} styles=${STYLES_PLUS}"
echo "SIZES=${SIZES[*]}"

BUILD_ID=$(sbatch --parsable \
  --export=ALL,K="${K}",STYLES="${STYLES_PLUS}" \
  scripts/data/build-scas-ae-decoder-gmm-prefix.slurm)
echo "build job ${BUILD_ID}"

declare -A TOK_IDS
for SIZE in "${SIZES[@]}"; do
  TOK_ID=$(sbatch --parsable \
    --dependency=afterok:${BUILD_ID} \
    --job-name="ae-gmm-tok-${SIZE}" \
    --export=ALL,SIZE="${SIZE}",K="${K}",STYLES="${STYLES_PLUS}" \
    scripts/data/prepare-scas-ae-gmm-ladder-data.slurm)
  TOK_IDS["${SIZE}"]="${TOK_ID}"
  echo "tokenize ${SIZE} -> ${TOK_ID} (afterok:${BUILD_ID})"
done

for SIZE in "${SIZES[@]}"; do
  for STYLE in "${STYLES[@]}"; do
    SHORT="${STYLE#covz-}"
    TRAIN_ID=$(sbatch --parsable \
      --dependency=afterok:${TOK_IDS[${SIZE}]} \
      --job-name="ae-gmm-${SHORT}-${SIZE}" \
      --export=ALL,SIZE="${SIZE}",STYLE="${STYLE}",K="${K}" \
      scripts/train/train-scas-ae-gmm-ladder-opt.slurm)
    echo "train STYLE=${STYLE} SIZE=${SIZE} -> ${TRAIN_ID} (afterok:${TOK_IDS[${SIZE}]})"
  done
done

echo "Queued: 1 build + ${#SIZES[@]} tokenize + $((${#SIZES[@]}*${#STYLES[@]})) train"
