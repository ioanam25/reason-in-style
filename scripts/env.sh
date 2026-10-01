#!/usr/bin/env bash
# Cluster-agnostic path defaults for training / eval / SLURM jobs.
# Override any variable before sourcing, or export in your shell / sbatch script.
#
# Example:
#   export DATA_ROOT=/path/to/datasets
#   export SCRATCH_ROOT=/path/to/scratch
#   source scripts/env.sh

_ENV_SRC="${BASH_SOURCE[0]:-$0}"
_ENV_DIR="$(cd "$(dirname "${_ENV_SRC}")" && pwd)"
: "${REPO_ROOT:=$(cd "${_ENV_DIR}/.." && pwd)}"

# Datasets, HF caches, checkpoints, and run artifacts (override on your cluster)
: "${DATA_ROOT:=${REPO_ROOT}/data}"
: "${SCRATCH_ROOT:=${REPO_ROOT}/scratch}"
: "${HF_HOME:=${DATA_ROOT}/hf_home}"
: "${HF_CACHE:=${HF_HOME}}"
: "${CKPT_ROOT:=${SCRATCH_ROOT}/checkpoints}"
: "${ARCHIVE_ROOT:=${SCRATCH_ROOT}/archive}"
: "${SLURM_LOG_DIR:=${SCRATCH_ROOT}/slurm_logs}"
: "${WANDB_DIR:=${SCRATCH_ROOT}/wandb}"

# SLURM defaults — change for your site (these are placeholders, not lab names)
: "${PARTITION:=gpu}"
: "${QOS:=}"
: "${ACCOUNT:=}"

export REPO_ROOT DATA_ROOT SCRATCH_ROOT HF_HOME HF_CACHE CKPT_ROOT ARCHIVE_ROOT
export SLURM_LOG_DIR WANDB_DIR PARTITION QOS ACCOUNT

# Convenience aliases used by older scripts
: "${SROOT:=${ARCHIVE_ROOT}}"
export SROOT

mkdir -p "${SLURM_LOG_DIR}" "${WANDB_DIR}" "${CKPT_ROOT}" "${ARCHIVE_ROOT}" 2>/dev/null || true
