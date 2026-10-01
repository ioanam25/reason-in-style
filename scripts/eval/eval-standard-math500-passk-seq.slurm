#!/usr/bin/env bash
#SBATCH --job-name=std-m500-seq
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu  # set to your site partition
#SBATCH --qos=normal  # set to your site QoS (or remove)
#SBATCH --time=48:00:00
#
# Sequential question-only MATH-500 Pass@k evals on one node.
#   sbatch --export=ALL,EVAL_SPECS="VAR1:SIZE1+VAR2:SIZE2" scripts/eval/eval-standard-math500-passk-seq.slurm
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve repo root whether script lives in scripts/ or scripts/pipelines/
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then source "${SCRIPT_DIR}/env.sh"; elif [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; fi
cd "${REPO_ROOT}"
EVAL_SPECS="${EVAL_SPECS:?set EVAL_SPECS=VAR:SIZE+...}"
export EOS_FIX="${EOS_FIX:-1}"
echo "Host=$(hostname) EVAL_SPECS=${EVAL_SPECS} EOS_FIX=${EOS_FIX}"
nvidia-smi -L || true
IFS="+" read -ra ITEMS <<< "${EVAL_SPECS}"
fail=0
for item in "${ITEMS[@]}"; do
  item="${item// /}"
  [[ -z "$item" ]] && continue
  export VAR="${item%%:*}"
  export SIZE="${item##*:}"
  echo "======== START ${VAR} / ${SIZE} ========"
  if ! bash scripts/eval/run_standard_math500_passk_once.sh; then
    echo "FAIL ${VAR}/${SIZE}"
    fail=1
  fi
done
exit ${fail}
