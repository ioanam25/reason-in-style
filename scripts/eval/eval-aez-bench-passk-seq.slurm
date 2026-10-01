#!/usr/bin/env bash
#SBATCH --job-name=aez-bench-seq
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
# BENCH_SPECS="VAR:SIZE:K:TAG:BENCH+..."
# e.g. random-qwen3-qwen3-4b-thinking:qwen3-4b-thinking:6:opt:amc
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve repo root whether script lives in scripts/ or scripts/pipelines/
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then source "${SCRIPT_DIR}/env.sh"; elif [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; fi
cd "${REPO_ROOT}"
BENCH_SPECS="${BENCH_SPECS:?}"
export EOS_FIX="${EOS_FIX:-1}"
export STYLE_TOKEN_FORMAT="${STYLE_TOKEN_FORMAT:-hard}"
echo "Host=$(hostname) AEZ BENCH_SPECS=${BENCH_SPECS}"
nvidia-smi -L || true
IFS="+" read -ra ITEMS <<< "${BENCH_SPECS}"
fail=0
for item in "${ITEMS[@]}"; do
  item="${item// /}"
  [[ -z "$item" ]] && continue
  IFS=":" read -r VAR SIZE K TAG BENCH <<< "$item"
  export VAR SIZE K CKPT_TAG="$TAG" BENCH
  echo "======== START ${BENCH} ${VAR}/${SIZE} K=${K} ========"
  bash scripts/eval/eval-scas-aez-cluster-prefix-bench-passk.slurm || fail=1
done
exit ${fail}
