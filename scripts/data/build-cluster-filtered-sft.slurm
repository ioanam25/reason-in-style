#!/usr/bin/env bash
#SBATCH --job-name=gmm-filter-sft
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu  # set to your site partition
#SBATCH --time=2:00:00
#
# Strip [style_i] prefixes and write GMM-filtered vanilla SFT parquets.
# Does not tokenize or train.
#
#   sbatch scripts/data/build-cluster-filtered-sft.slurm
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
source ./.venv/bin/activate
mkdir -p ${SLURM_LOG_DIR}

./.venv/bin/python -u scripts/data/build_cluster_filtered_sft.py --all --seed 42 --k 6

echo "DONE cluster-filtered SFT parquets"
echo "Next: tokenize 0.6B with scripts/submit/submit_cluster_filtered_0p6b.sh"
