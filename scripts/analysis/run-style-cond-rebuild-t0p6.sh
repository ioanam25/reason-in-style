#!/usr/bin/env bash
#SBATCH --job-name=style-cond-t06
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu  # set to your site partition
#SBATCH --qos=normal  # set to your site QoS (or remove)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --gres=gpu:0
#SBATCH --time=06:00:00
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve repo root whether script lives in scripts/ or scripts/pipelines/
if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then source "${SCRIPT_DIR}/env.sh"; elif [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; fi
cd "${REPO_ROOT}"
export OMP_NUM_THREADS=1
echo "==== $(date -Is) host=$(hostname) ===="

./.venv/bin/python -u << 'PY'
from pathlib import Path
from scripts.analysis.extract_style_records import process_dir

SROOT = Path("${ARCHIVE_ROOT}")
OUT = SROOT / "style_records"
EVALS = [
    ("scas-nosft-qwen3-4b-base_pretrained-math500-passk-t0p6-eosfix", "base", "4B-Base"),
    ("scas-standard-qwen3-4b-base_checkpoint-5250-math500-passk-t0p6-eosfix", "vanilla", "4B-Base"),
    ("scas-gmm-filter-is-obs-pfx-qwen3-4b-base_best-math500-passk-n256x6-t0p6-eosfix", "is_obs", "4B-Base"),
]
for name, arm, student in EVALS:
    EVAL = SROOT / "model-evals" / name
    pq = OUT / name / "records.parquet"
    assert (EVAL / "generations.jsonl").is_file(), EVAL
    if pq.is_file():
        print(f"records exist {pq}", flush=True)
        continue
    meta = {"arm": arm, "student": student, "benchmark": "math500", "protocol": "t0p6-eosfix"}
    print(f"extract {name}", flush=True)
    info = process_dir(EVAL, meta, OUT, workers=32, qmod=4, per_cell=4)
    print(info, flush=True)
print("done extract", flush=True)
PY

./.venv/bin/python -u scripts/analysis/analyze_style_conditioned_correctness_t0p6.py
echo "==== DONE $(date -Is) ===="
