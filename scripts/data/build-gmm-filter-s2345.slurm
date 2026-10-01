#!/usr/bin/env bash
#SBATCH --job-name=gmm-filter-s2345
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu  # set to your site partition
#SBATCH --time=2:00:00
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
source ./.venv/bin/activate
mkdir -p ${SLURM_LOG_DIR}
for v in style2 style3 style4 style5; do
  ./.venv/bin/python -u scripts/data/build_cluster_filtered_sft.py --variant "$v" --seed 42 --k 6
done
echo DONE
