#!/usr/bin/env bash
# Submit MATH-500 Pass@k for every AE-variant cluster-prefix run that has finished SFT.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve repo root whether script lives in scripts/ or scripts/pipelines/
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then source "${SCRIPT_DIR}/env.sh"; elif [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; fi
cd "${REPO_ROOT}"
SROOT="${ARCHIVE_ROOT}"
for VAR in b8 d64; do
  for K in $(seq 2 10); do
    CKPT_DIR="${SROOT}/cluster_ckpt/${VAR}/k${K}"
    last="$(ls -1 "${CKPT_DIR}" 2>/dev/null | grep -E '^checkpoint-[0-9]+$' | sort -t- -k2 -n | tail -1)"
    [[ -n "${last}" ]] || { echo "skip ${VAR} k${K}: no checkpoint yet"; continue; }
    J=$(sbatch --parsable --export=ALL,VAR=$VAR,K=$K scripts/eval/eval-scas-aez-cluster-prefix-math500-passk.slurm)
    echo "eval ${VAR} k${K} (${last}) -> ${J}"
  done
done
