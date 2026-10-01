#!/usr/bin/env bash
#SBATCH --job-name=scas-rnd-pfx-b
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cpu  # set to your site partition,cpu_long
#SBATCH --time=4:00:00
#
# Assign random clusters K=2..10 and build [style_i] prefix SFT parquets.
# Usage: sbatch scripts/data/build-scas-random-cluster-prefix-all.slurm [qwen3-4b]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"

SIZE="${1:-qwen3-4b}"
source scripts/scas_model_size.sh "${SIZE}"
PY=./.venv/bin/python
mkdir -p slurm_logs

RAND_DIR="data/scas/cluster_random_assignments-${MODEL_SIZE}"

echo "=== Random assignments ==="
"${PY}" scripts/data/assign_scas_random_clusters.py \
  --prefix-dir "${MODC_SFT_DIR}" \
  --output-dir "${RAND_DIR}" \
  --k-min 2 \
  --k-max 10 \
  --seed 42

for K in $(seq 2 10); do
  OUT="data/scas/cluster_random_prefix_k${K}-${MODEL_SIZE}"
  echo "=== Build K=${K} -> ${OUT} ==="
  "${PY}" scripts/data/build_scas_random_cluster_prefix_sft_dataset.py \
    --k "${K}" \
    --prefix-dir "${MODC_SFT_DIR}" \
    --random-dir "${RAND_DIR}" \
    --output-dir "${OUT}"
done
echo "Done random assign+build K=2..10"
