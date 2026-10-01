#!/usr/bin/env bash
#SBATCH --job-name=m500-single
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
# EVAL_SPECS="VAR:SIZE+..."  STYLE=style_1  K=6  EOS_FIX=1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve repo root whether script lives in scripts/ or scripts/pipelines/
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then source "${SCRIPT_DIR}/env.sh"; elif [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; fi
cd "${REPO_ROOT}"
EVAL_SPECS="${EVAL_SPECS:?}"
export EOS_FIX="${EOS_FIX:-1}"
export STYLE="${STYLE:-style_1}"
export K="${K:-6}"
echo "Host=$(hostname) SINGLE STYLE=${STYLE} SPECS=${EVAL_SPECS}"
nvidia-smi -L || true
IFS="+" read -ra ITEMS <<< "${EVAL_SPECS}"
fail=0
for item in "${ITEMS[@]}"; do
  item="${item// /}"
  [[ -z "$item" ]] && continue
  export VAR="${item%%:*}"
  export SIZE="${item##*:}"
  echo "======== START single ${STYLE} ${VAR}/${SIZE} ========"
  bash scripts/eval/run_aez_math500_single_prefix_once.sh || fail=1
done
exit ${fail}
