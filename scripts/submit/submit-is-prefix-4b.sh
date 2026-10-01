#!/usr/bin/env bash
# Qwen3-4B-Base prefix+IS: tokenize → 40e valbest SFT → GPU generate + CPU grade.
# Same traces/weights as 0.6B/1.7B prefix+IS (gmm-filter-is-{rebal,obs,global}-pfx).
# Train recipe matches 1.7B IS: packing off, batch 1, accum 32, GC on, 40e.
# Optional NODES/EXCL env vars select/exclude hosts on your cluster.
#
#   bash scripts/submit/submit-is-prefix-4b.sh dry
#   bash scripts/submit/submit-is-prefix-4b.sh submit
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve repo root whether script lives in scripts/ or scripts/pipelines/
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then source "${SCRIPT_DIR}/env.sh"; elif [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; fi
cd "${REPO_ROOT}"

MODE="${1:-dry}"
SROOT="${ARCHIVE_ROOT}"
LOG="${SLURM_LOG_DIR}"
SIZE=qwen3-4b-base
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

echo "MODE=${MODE} SIZE=${SIZE} pool=${NODES} exclude=${EXCL}"

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
    --job-name="tok-is-${arm}-pfx-4b" \
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
    --job-name="tr-is-${arm}-pfx-4b-vb" \
    --export=ALL,ARM="${src}",SIZE="${SIZE}",K="${K}",NUM_EPOCHS=40,PACKING=false,CKPT_DIR="${ckpt}" \
    scripts/train/train-scas-prefix-arm-opt.slurm)
  echo "train ${arm} -> ${TR[${arm}]:-DRY}  ckpt=${ckpt}"
done

echo
echo "===== prefix evals (afterok train) ====="
for arm in "${ARMS[@]}"; do
  var="gmm-filter-is-${arm}-pfx-${SIZE}"
  ckpt="${SROOT}/cluster_ckpt/${var}/k${K}-opt-valbest"
  tr_jid="${TR[${arm}]:-DRY}"
  echo "=== ${arm} train=${tr_jid} ckpt=${ckpt} ==="
  for spec in "${BENCHES[@]}"; do
    IFS=':' read -r bench tag short <<<"${spec}"
    out="${SROOT}/model-evals/scas-${var}_best-${tag}-passk-n256x6-t0p6-eosfix"
    common="ALL,BENCH=${bench},VAR=${var},SIZE=${SIZE},K=${K},CKPT_DIR=${ckpt},CKPT_STEP=best,CKPT_TAG=opt-valbest,TEMPERATURE=0.6,TOP_P=0.95,EOS_FIX=1,OUT_TAG=-t0p6-eosfix,OUT_DIR=${out},STYLE_TOKEN_FORMAT=hard,SAMPLES_PER_STYLE=256"
    gen_name="gen-${short}-is-${arm}-pfx-4b-t06"
    grd_name="grd-${short}-is-${arm}-pfx-4b-t06"
    gen_dep=()
    if [[ "${MODE}" == "submit" ]]; then
      gen_dep=(--dependency="afterok:${tr_jid}")
    fi
    gen_jid=$(run sbatch --parsable "${GPU_COMMON[@]}" --time=48:00:00 "${gen_dep[@]}" --job-name="${gen_name}" \
      --export="${common},STAGE=generate" \
      scripts/eval/eval-scas-aez-cluster-prefix-bench-passk.slurm)
    echo "GEN ${gen_name} -> ${gen_jid}  afterok:${tr_jid}"
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
echo "===== no-SFT + vanilla AIME/HMMT (question-only n=256) ====="
for spec in "aime:aime2024-2025:aime" "hmmt:hmmt-feb-2025:hmmt"; do
  IFS=':' read -r bench tag short <<<"${spec}"
  common="ALL,HF_MODEL=Qwen/Qwen3-4B-Base,SIZE=${SIZE},BENCHES=${bench}:${tag},TEMPERATURE=0.6,TOP_P=0.95,EOS_FIX=1,OUT_TAG=-t0p6-eosfix"
  gen_jid=$(run sbatch --parsable "${GPU_COMMON[@]}" --time=12:00:00 --job-name="gen-${short}-nosft-4b-t06" \
    --export="${common},STAGE=generate" \
    scripts/eval/eval-nosft-bench-passk.slurm)
  echo "GEN gen-${short}-nosft-4b-t06 -> ${gen_jid}"
  grd_dep=()
  if [[ "${MODE}" == "submit" ]]; then
    grd_dep=(--dependency="afterok:${gen_jid}")
  fi
  grd_jid=$(run sbatch --parsable "${CPU_COMMON[@]}" "${grd_dep[@]}" --job-name="grd-${short}-nosft-4b-t06" \
    --export="${common},STAGE=grade" \
    scripts/eval/eval-nosft-bench-passk.slurm)
  echo "GRD grd-${short}-nosft-4b-t06 -> ${grd_jid}  afterok:${gen_jid}"
done

VAN_DIR="${SROOT}/baselines/ckpt/standard-qwen3-4b-base-opt"
[[ -f "${VAN_DIR}/checkpoint-5250/config.json" ]] || { echo "MISSING vanilla ${VAN_DIR}/checkpoint-5250"; exit 1; }
for spec in "aime:aime2024-2025:aime" "hmmt:hmmt-feb-2025:hmmt"; do
  IFS=':' read -r bench tag short <<<"${spec}"
  common="ALL,VAR=standard-qwen3-4b-base,SIZE=${SIZE},BENCH=${bench},CKPT_DIR=${VAN_DIR},CKPT_STEP=checkpoint-5250,CKPT_TAG=opt,TEMPERATURE=0.6,TOP_P=0.95,EOS_FIX=1,OUT_TAG=-t0p6-eosfix"
  gen_jid=$(run sbatch --parsable "${GPU_COMMON[@]}" --time=12:00:00 --job-name="gen-${short}-vanilla-4b-t06" \
    --export="${common},STAGE=generate,BENCH_SPECS=standard-qwen3-4b-base:${SIZE}:${bench}" \
    scripts/eval/eval-standard-bench-passk-seq.slurm)
  echo "GEN gen-${short}-vanilla-4b-t06 -> ${gen_jid}"
  grd_dep=()
  if [[ "${MODE}" == "submit" ]]; then
    grd_dep=(--dependency="afterok:${gen_jid}")
  fi
  grd_jid=$(run sbatch --parsable "${CPU_COMMON[@]}" "${grd_dep[@]}" --job-name="grd-${short}-vanilla-4b-t06" \
    --export="${common},STAGE=grade,BENCH_SPECS=standard-qwen3-4b-base:${SIZE}:${bench}" \
    scripts/eval/eval-standard-bench-passk-seq.slurm)
  echo "GRD grd-${short}-vanilla-4b-t06 -> ${grd_jid}  afterok:${gen_jid}"
done

echo
echo "done MODE=${MODE}"
echo "Init: Qwen/Qwen3-4B-Base. Packing off, batch 1 x accum 32 x 8 = 256, GC on, 40e valbest."
echo "Prefix evals: 256/style, T=0.6/top_p=0.95/EOS_FIX. Nodes: ${NODES} only."
