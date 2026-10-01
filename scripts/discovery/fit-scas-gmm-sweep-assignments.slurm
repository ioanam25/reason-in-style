#!/usr/bin/env bash
#SBATCH --job-name=scas-gmm-swp
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu  # set to your site partition
#SBATCH --time=6:00:00
#
# Fit GMM on AE z for K=2..10 (paper uses K=6) and save assignments.
#
#   sbatch --export=ALL,TAG=covz-qwen3-4b-final \
#     scripts/discovery/fit-scas-gmm-sweep-assignments.slurm
#
# Reads:  ${ARCHIVE_ROOT}/zscore-variants/${TAG}/embeddings_train_z.npz
# Writes: ${ARCHIVE_ROOT}/zscore-variants/${TAG}/gmm_from_kmeans/

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then source "${SCRIPT_DIR}/env.sh"; elif [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; fi
cd "${REPO_ROOT}"

TAG="${TAG:?set TAG e.g. covz-qwen3-4b-final}"
ZDIR="${ZDIR:-${ARCHIVE_ROOT}/zscore-variants/${TAG}}"
OUT="${OUT:-${ZDIR}/gmm_from_kmeans}"
mkdir -p slurm_logs "${OUT}"

[[ -f "${ZDIR}/embeddings_train_z.npz" ]] || {
  echo "Missing ${ZDIR}/embeddings_train_z.npz — run extract-scas-z-8gpu.slurm first"
  exit 1
}

./.venv/bin/python -u scripts/discovery/fit_scas_gmm_sweep_assignments.py \
  --gmm-dir "${ZDIR}" \
  --output-dir "${OUT}" \
  --k-min "${K_MIN:-2}" \
  --k-max "${K_MAX:-10}" \
  --seed 42

echo "Done. Outputs in ${OUT}/"
ls -la "${OUT}/"
echo "Style-SFT uses K=6: ${OUT}/assignments_gmm_k06.parquet"
