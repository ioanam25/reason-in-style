#!/usr/bin/env bash
#SBATCH --job-name=scas-rnd-pfx-t
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=cpu  # set to your site partition
#SBATCH --time=12:00:00
#
# Tokenize random-cluster-prefix SFT for K=2..10.
# Usage: sbatch scripts/data/prepare-scas-random-cluster-prefix-cache-all.slurm [qwen3-4b]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
source ./.venv/bin/activate
SIZE="${1:-qwen3-4b}"
source scripts/scas_model_size.sh "${SIZE}"
PY=./.venv/bin/python
mkdir -p slurm_logs

for K in $(seq 2 10); do
  DIR="data/scas/cluster_random_prefix_k${K}-${MODEL_SIZE}"
  echo "=== Tokenizing K=${K} ${DIR} ==="
  "${PY}" scripts/prepare_sft_hf_cache.py "${DIR}" \
    --tokenizer "${DECODER_MODEL}" \
    --max-seq-length "${SCAS_MAX_LEN}" \
    --num-proc 32 \
    --force
done
echo "Done tokenizing random K=2..10"
