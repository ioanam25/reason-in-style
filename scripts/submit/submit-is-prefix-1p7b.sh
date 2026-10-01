#!/usr/bin/env bash
# Submit 1.7B-Base prefix+IS: tokenize → 40e valbest SFT. No evals.
# Same traces/weights as 0.6B prefix+IS (gmm-filter-is-{rebal,obs,global}-pfx).
# Train recipe matches no-prefix 1.7B IS: packing off, batch 1, accum 32, GC on, 40e.
# Evals (GPU generate + CPU grade, 5 benches) are in scripts/submit/submit-is-prefix-1p7b-eval.sh
#
#   bash scripts/submit/submit-is-prefix-1p7b.sh dry      # print only (default)
#   bash scripts/submit/submit-is-prefix-1p7b.sh submit   # sbatch tokenize + train
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve repo root whether script lives in scripts/ or scripts/pipelines/
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then source "${SCRIPT_DIR}/env.sh"; elif [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; fi
cd "${REPO_ROOT}"

MODE="${1:-dry}"
SROOT="${ARCHIVE_ROOT}"
LOG="${SLURM_LOG_DIR}"
SIZE=qwen3-1p7b-base
K=6
ARMS=(rebal obs global)

NODES="${NODES:-}"  # optional: --nodelist
EXCL="${EXCL:-}"  # optional: --exclude

GPU_COMMON=(
  --partition="${PARTITION}" --qos="${QOS:-normal}"
  --gres=gpu:8 --cpus-per-task=32 --mem=240G
  --nodes=1 --ntasks-per-node=1
  ${NODES:+--nodelist="${NODES}"} ${EXCL:+--exclude="${EXCL}"}
  --output="${LOG}/%x-%j.out" --error="${LOG}/%x-%j.err"
)

run() {
  if [[ "${MODE}" == "submit" ]]; then
    "$@"
  else
    printf '[dry] ' >&2
    printf '%q ' "$@" >&2
    printf '\n' >&2
    echo DRY
  fi
}

echo "MODE=${MODE} SIZE=${SIZE} pool=${NODES} exclude=${EXCL}  (train only, no evals)"

for arm in "${ARMS[@]}"; do
  parq="${SROOT}/cluster_prefix/gmm-filter-is-${arm}-pfx/k${K}/train.parquet"
  [[ -f "${parq}" ]] || { echo "MISSING ${parq}"; exit 1; }
done

declare -A TOK=()
declare -A TR=()

for arm in "${ARMS[@]}"; do
  src="gmm-filter-is-${arm}-pfx"
  tok_out="${SROOT}/cluster_prefix/${src}-${SIZE}/k${K}"
  ckpt="${SROOT}/cluster_ckpt/${src}-${SIZE}/k${K}-opt-valbest"

  if [[ "${MODE}" == "submit" ]]; then
    if [[ -d "${ckpt}/best" ]] || ls -d "${ckpt}"/checkpoint-* >/dev/null 2>&1; then
      echo "REFUSING: ${ckpt} already has checkpoints"
      exit 1
    fi
    mkdir -p "${ckpt}"
  fi

  echo "=== tokenize ${arm} ==="
  TOK[${arm}]=$(run sbatch --parsable \
    --job-name="tok-is-${arm}-pfx-1p7b" \
    --export=ALL,SRC="${src}",SIZE="${SIZE}",K="${K}" \
    scripts/data/prepare-scas-prefix-size.slurm)
  echo "tok ${arm} -> ${TOK[${arm}]:-DRY}  out=${tok_out}"

  dep=()
  if [[ "${MODE}" == "submit" ]]; then
    dep=(--dependency="afterok:${TOK[${arm}]}")
  fi

  echo "=== train ${arm} ==="
  TR[${arm}]=$(run sbatch --parsable "${GPU_COMMON[@]}" \
    --time=2-00:00:00 \
    "${dep[@]}" \
    --job-name="tr-is-${arm}-pfx-1p7b-vb" \
    --export=ALL,ARM="${src}",SIZE="${SIZE}",K="${K}",NUM_EPOCHS=40,PACKING=false,CKPT_DIR="${ckpt}" \
    scripts/train/train-scas-prefix-arm-opt.slurm)
  echo "train ${arm} -> ${TR[${arm}]:-DRY}  ckpt=${ckpt}"
done

echo
echo "done MODE=${MODE}"
echo "Init: Qwen/Qwen3-1.7B-Base (not hybrid). Packing off, batch 1 x accum 32 x 8 = 256, GC on, 40e valbest."
echo "No evals queued. Later: bash scripts/submit/submit-is-prefix-1p7b-eval.sh dry|submit"
