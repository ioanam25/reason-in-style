#!/usr/bin/env bash
#SBATCH --job-name=aez-m500-seq
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
# Sequential MATH-500 Pass@k evals on one node.
#   sbatch --export=ALL,EVAL_SPECS="VAR1:SIZE1,VAR2:SIZE2" scripts/eval/eval-aez-math500-passk-seq.slurm
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
  export K="${K:-6}"
  echo "======== START ${VAR} / ${SIZE} ========"
  if ! bash scripts/eval/run_aez_math500_passk_once.sh; then
    echo "FAIL ${VAR}/${SIZE}"
    fail=1
  fi
done
if [[ "${RUN_FORCED_STYLE:-0}" == "1" ]]; then
  echo "======== forced-style sweep ========"
  OUT="${FORCED_OUT:-${ARCHIVE_ROOT}/forced_style_sweep_math500_gmm_final}"
  ./.venv/bin/python scripts/analysis/analyze_forced_style_sweep.py \
    --eval-root ${ARCHIVE_ROOT}/model-evals \
    --out-dir "${OUT}" \
    --bench-regex math500 \
    --budgets 1 8 32 128 256 || fail=1
fi
exit ${fail}
