#!/usr/bin/env bash
# AIME 2024+2025 + HMMT Feb 2025 for Qwen3-0.6B-Base:
#   no-SFT, vanilla checkpoint-5250, prefix+IS {rebal,obs,global}.
# GPU generate (8xH100) then CPU grade (--gres=gpu:0) so grading does not hold GPUs.
# Optional NODES/EXCL env vars select/exclude hosts on your cluster.
#
#   bash scripts/submit/submit-aime-hmmt-0p6b.sh dry
#   bash scripts/submit/submit-aime-hmmt-0p6b.sh submit
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve repo root whether script lives in scripts/ or scripts/pipelines/
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then source "${SCRIPT_DIR}/env.sh"; elif [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; fi
cd "${REPO_ROOT}"

MODE="${1:-dry}"
SROOT="${ARCHIVE_ROOT}"
LOG="${SLURM_LOG_DIR}"
SIZE=qwen3-0p6b-base
K=6
NODES="${NODES:-}"  # optional: --nodelist
EXCL="${EXCL:-}"  # optional: --exclude

GPU_COMMON=(
  --partition="${PARTITION}" --qos="${QOS:-normal}"
  --gres=gpu:8 --cpus-per-task=32 --mem=240G
  --nodes=1 --ntasks-per-node=1
  --nodelist="${NODES}" --exclude="${EXCL}"
  --time=12:00:00
  --output="${LOG}/%x-%j.out" --error="${LOG}/%x-%j.err"
)
# gpu:0: GPU node, zero H100s allocated, so the next generate can start.
CPU_COMMON=(
  --partition="${PARTITION}" --qos="${QOS:-normal}"
  --gres=gpu:0 --cpus-per-task=8 --mem=48G
  --nodes=1 --ntasks-per-node=1
  --nodelist="${NODES}" --exclude="${EXCL}"
  --time=8:00:00
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

echo "MODE=${MODE} pool=${NODES} exclude=${EXCL}"

submit_gen_grade() {
  local gen_name="$1" grd_name="$2" script="$3" gen_export="$4" grd_export="$5"
  local gen_jid grd_jid
  gen_jid=$(run sbatch --parsable "${GPU_COMMON[@]}" --job-name="${gen_name}" --export="${gen_export}" "${script}")
  echo "GEN ${gen_name} -> ${gen_jid}"
  local dep=()
  if [[ "${MODE}" == "submit" ]]; then
    dep=(--dependency="afterok:${gen_jid}")
  fi
  grd_jid=$(run sbatch --parsable "${CPU_COMMON[@]}" "${dep[@]}" --job-name="${grd_name}" --export="${grd_export}" "${script}")
  echo "GRD ${grd_name} -> ${grd_jid}  afterok:${gen_jid}"
}

# --- no-SFT Base (question-only, n=256) ---
for spec in "aime:aime2024-2025:aime" "hmmt:hmmt-feb-2025:hmmt"; do
  IFS=':' read -r bench tag short <<<"${spec}"
  common="ALL,HF_MODEL=Qwen/Qwen3-0.6B-Base,SIZE=${SIZE},BENCHES=${bench}:${tag},TEMPERATURE=0.6,TOP_P=0.95,EOS_FIX=1,OUT_TAG=-t0p6-eosfix"
  submit_gen_grade \
    "gen-${short}-nosft-0p6b-t06" \
    "grd-${short}-nosft-0p6b-t06" \
    scripts/eval/eval-nosft-bench-passk.slurm \
    "${common},STAGE=generate" \
    "${common},STAGE=grade"
done

# --- vanilla SFT checkpoint-5250 (question-only, n=256) ---
VAN_DIR="${SROOT}/baselines/ckpt/standard-qwen3-0p6b-base-opt"
for spec in "aime:aime2024-2025:aime" "hmmt:hmmt-feb-2025:hmmt"; do
  IFS=':' read -r bench tag short <<<"${spec}"
  common="ALL,VAR=standard-qwen3-0p6b-base,SIZE=${SIZE},BENCH=${bench},CKPT_DIR=${VAN_DIR},CKPT_STEP=checkpoint-5250,CKPT_TAG=opt,TEMPERATURE=0.6,TOP_P=0.95,EOS_FIX=1,OUT_TAG=-t0p6-eosfix"
  submit_gen_grade \
    "gen-${short}-vanilla-0p6b-t06" \
    "grd-${short}-vanilla-0p6b-t06" \
    scripts/eval/eval-standard-bench-passk-seq.slurm \
    "${common},STAGE=generate,BENCH_SPECS=standard-qwen3-0p6b-base:${SIZE}:${bench}" \
    "${common},STAGE=grade,BENCH_SPECS=standard-qwen3-0p6b-base:${SIZE}:${bench}"
done

# --- prefix+IS valbest, 256 samples per [style_i] ---
for arm in rebal obs global; do
  var="gmm-filter-is-${arm}-pfx-${SIZE}"
  ckpt="${SROOT}/cluster_ckpt/${var}/k${K}-opt-valbest"
  for spec in "aime:aime2024-2025:aime" "hmmt:hmmt-feb-2025:hmmt"; do
    IFS=':' read -r bench tag short <<<"${spec}"
    out="${SROOT}/model-evals/scas-${var}_best-${tag}-passk-n256x6-t0p6-eosfix"
    common="ALL,BENCH=${bench},VAR=${var},SIZE=${SIZE},K=${K},CKPT_DIR=${ckpt},CKPT_STEP=best,CKPT_TAG=opt-valbest,TEMPERATURE=0.6,TOP_P=0.95,EOS_FIX=1,OUT_TAG=-t0p6-eosfix,OUT_DIR=${out},STYLE_TOKEN_FORMAT=hard,SAMPLES_PER_STYLE=256"
    submit_gen_grade \
      "gen-${short}-is-${arm}-pfx-0p6b-t06" \
      "grd-${short}-is-${arm}-pfx-0p6b-t06" \
      scripts/eval/eval-scas-aez-cluster-prefix-bench-passk.slurm \
      "${common},STAGE=generate" \
      "${common},STAGE=grade"
  done
done

echo
echo "done MODE=${MODE}"
echo "10 GPU generate + 10 CPU grade. Decode: T=0.6 / top_p=0.95 / EOS_FIX / cap 4096."
echo "Prefix: 256/style (1536/problem). Vanilla/no-SFT: 256 question-only."
