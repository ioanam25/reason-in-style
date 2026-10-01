#!/usr/bin/env bash
#SBATCH --job-name=scas-z-score
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu  # set to your site partition
#SBATCH --time=2:00:00
#
# Score AE z: length predictability + k-means ARI/NMI vs teacher.
# Usage:
#   sbatch scripts/discovery/score-scas-z-representation.slurm \
#     data/scas/cluster_gmm_assignments-qwen3-0p6b/embeddings_train_z.npz \
#     eval_results/scas_z_score-qwen3-0p6b.json \
#     data/scas/cluster_kmeans_sweep-qwen3-0p6b \
#     baseline-qwen3-0p6b

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve repo root whether script lives in scripts/ or scripts/pipelines/
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then source "${SCRIPT_DIR}/env.sh"; elif [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; fi
cd "${REPO_ROOT}"

EMB="${1:?embeddings npz}"
OUT_JSON="${2:?output json}"
KMEANS_DIR="${3:-}"
LABEL="${4:-}"

mkdir -p slurm_logs eval_results
ARGS=(
  --embeddings-npz "${EMB}"
  --hf-dataset scas_traces_dataset
  --raw-json configs/scas/scas_modc_dataset.json
  --output-json "${OUT_JSON}"
  --min-k 2 --max-k 9
)
[[ -n "${KMEANS_DIR}" ]] && ARGS+=(--kmeans-out-dir "${KMEANS_DIR}")
[[ -n "${LABEL}" ]] && ARGS+=(--label "${LABEL}")

echo "Host: $(hostname)"
echo "Emb: ${EMB}"
./.venv/bin/python -u scripts/discovery/score_scas_z_representation.py "${ARGS[@]}"
