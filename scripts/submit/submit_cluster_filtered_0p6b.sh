#!/usr/bin/env bash
# Build GMM-filtered vanilla SFT parquets, tokenize Qwen3-0.6B, optionally train.
#
# Default: CPU build + tokenize only. 8-GPU trains are NOT submitted unless
#   SUBMIT_TRAIN=1
# because the exclusive 8-GPU eval queue is still draining.
#
#   bash scripts/submit/submit_cluster_filtered_0p6b.sh
#   SUBMIT_TRAIN=1 bash scripts/submit/submit_cluster_filtered_0p6b.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve repo root whether script lives in scripts/ or scripts/pipelines/
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then source "${SCRIPT_DIR}/env.sh"; elif [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; fi
cd "${REPO_ROOT}"
LOG="${SLURM_LOG_DIR}"/submit_cluster_filtered_0p6b.log
mkdir -p ${SLURM_LOG_DIR}
exec > >(tee -a "$LOG") 2>&1
echo "=== cluster-filtered 0.6B $(date -Is) SUBMIT_TRAIN=${SUBMIT_TRAIN:-0} ==="

SIZE="${SIZE:-qwen3-0p6b}"
K="${K:-6}"
ARMS=(
  gmm-filter-style1
  gmm-filter-style6
  gmm-filter-rebal
  gmm-filter-rand-rebal
  gmm-filter-rand-style1
)

BUILD=$(sbatch --parsable --job-name="gmm-filt-build" \
  scripts/data/build-cluster-filtered-sft.slurm)
echo "build=$BUILD"

TOK_JOBS=()
for ARM in "${ARMS[@]}"; do
  TOK=$(sbatch --parsable --dependency=afterok:"${BUILD}" \
    --job-name="tok-${ARM}" \
    --export=ALL,SRC="${ARM}",SIZE="${SIZE}",K="${K}" \
    scripts/data/prepare-scas-prefix-size.slurm)
  TOK_JOBS+=("$TOK")
  echo "  tokenize ${ARM}: $TOK"
done
TOK_GATE=$(IFS=:; echo "${TOK_JOBS[*]}")

echo
echo "Train commands (same opt recipe as prefix 0.6B, 15 epochs):"
for ARM in "${ARMS[@]}"; do
  echo "  sbatch --export=ALL,ARM=${ARM},SIZE=${SIZE},K=${K} scripts/train/train-scas-prefix-arm-opt.slurm"
done

if [[ "${SUBMIT_TRAIN:-0}" == "1" ]]; then
  echo "SUBMIT_TRAIN=1 — queuing 8-GPU trains after tokenize"
  for ARM in "${ARMS[@]}"; do
    TR=$(sbatch --parsable --exclusive --partition="${PARTITION}" --qos="${QOS:-normal}" \
      --dependency=afterok:"${TOK_GATE}" \
      --job-name="tr-${ARM}" \
      --export=ALL,ARM="${ARM}",SIZE="${SIZE}",K="${K}" \
      scripts/train/train-scas-prefix-arm-opt.slurm)
    echo "  train ${ARM}: $TR"
  done
else
  echo "Not submitting 8-GPU trains. Re-run with SUBMIT_TRAIN=1 after the eval queue drains."
fi
