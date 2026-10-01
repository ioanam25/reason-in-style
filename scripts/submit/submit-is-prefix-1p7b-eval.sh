#!/usr/bin/env bash
# Prefix+IS 1.7B-Base evals: GPU generate then CPU grade (--gres=gpu:0).
# Benches: MATH-500, AMC12-2025, OlympiadBench-EN, AIME 2024+2025, HMMT Feb 2025.
# 256 samples per [style_i], T=0.6 / top_p=0.95 / EOS_FIX / cap 4096.
#
#   bash scripts/submit/submit-is-prefix-1p7b-eval.sh dry
#   bash scripts/submit/submit-is-prefix-1p7b-eval.sh submit
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
# Train jobs already queued; generate waits so evals cannot start before valbest exists.
declare -A TRAIN_JID=([rebal]="${TRAIN_REBAL:-27307752}" [obs]="${TRAIN_OBS:-27307754}" [global]="${TRAIN_GLOBAL:-27307756}")

GPU_COMMON=(
  --partition="${PARTITION}" --qos="${QOS:-normal}"
  --gres=gpu:8 --cpus-per-task=32 --mem=240G
  --nodes=1 --ntasks-per-node=1
  ${NODES:+--nodelist="${NODES}"} ${EXCL:+--exclude="${EXCL}"}
  --time=48:00:00
  --output="${LOG}/%x-%j.out" --error="${LOG}/%x-%j.err"
)
CPU_COMMON=(
  --partition="${PARTITION}" --qos="${QOS:-normal}"
  --gres=gpu:0 --cpus-per-task=8 --mem=48G
  --nodes=1 --ntasks-per-node=1
  ${NODES:+--nodelist="${NODES}"} ${EXCL:+--exclude="${EXCL}"}
  --time=12:00:00
  --output="${LOG}/%x-%j.out" --error="${LOG}/%x-%j.err"
)

BENCHES=(
  "math500:math500:m500"
  "amc25:amc12-2025:amc25"
  "olympiadbench:olympiadbench-en:olymp"
  "aime:aime2024-2025:aime"
  "hmmt:hmmt-feb-2025:hmmt"
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

echo "MODE=${MODE} SIZE=${SIZE} pool=${NODES}  GPU generate + CPU grade, 5 benches"

for arm in "${ARMS[@]}"; do
  var="gmm-filter-is-${arm}-pfx-${SIZE}"
  ckpt="${SROOT}/cluster_ckpt/${var}/k${K}-opt-valbest"
  tr_jid="${TRAIN_JID[${arm}]}"
  echo "=== ${arm} train=${tr_jid} ckpt=${ckpt} ==="
  for spec in "${BENCHES[@]}"; do
    IFS=':' read -r bench tag short <<<"${spec}"
    out="${SROOT}/model-evals/scas-${var}_best-${tag}-passk-n256x6-t0p6-eosfix"
    common="ALL,BENCH=${bench},VAR=${var},SIZE=${SIZE},K=${K},CKPT_DIR=${ckpt},CKPT_STEP=best,CKPT_TAG=opt-valbest,TEMPERATURE=0.6,TOP_P=0.95,EOS_FIX=1,OUT_TAG=-t0p6-eosfix,OUT_DIR=${out},STYLE_TOKEN_FORMAT=hard,SAMPLES_PER_STYLE=256"
    gen_name="gen-${short}-is-${arm}-pfx-1p7b-t06"
    grd_name="grd-${short}-is-${arm}-pfx-1p7b-t06"
    gen_dep=()
    if [[ "${MODE}" == "submit" ]]; then
      gen_dep=(--dependency="afterok:${tr_jid}")
    fi
    gen_jid=$(run sbatch --parsable "${GPU_COMMON[@]}" "${gen_dep[@]}" --job-name="${gen_name}" \
      --export="${common},STAGE=generate" \
      scripts/eval/eval-scas-aez-cluster-prefix-bench-passk.slurm)
    echo "GEN ${gen_name} -> ${gen_jid}  afterok:${tr_jid}  ${out}"
    grd_dep=()
    if [[ "${MODE}" == "submit" ]]; then
      grd_dep=(--dependency="afterok:${gen_jid}")
    fi
    grd_jid=$(run sbatch --parsable "${CPU_COMMON[@]}" "${grd_dep[@]}" --job-name="${grd_name}" \
      --export="${common},STAGE=grade" \
      scripts/eval/eval-scas-aez-cluster-prefix-bench-passk.slurm)
    echo "GRD ${grd_name} -> ${grd_jid}  afterok:${gen_jid}"
  done
done

echo
echo "done MODE=${MODE}"
echo "3 arms x 5 benches x (generate + grade). Prefix: 256/style. T=0.6/top_p=0.95/EOS_FIX."
