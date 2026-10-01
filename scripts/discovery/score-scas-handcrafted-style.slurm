#!/usr/bin/env bash
#SBATCH --job-name=scas-handcrafted-style
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --partition=cpu  # set to your site partition
#SBATCH --time=2:00:00
#
# Part A: interpretable style-feature baseline (pivots/length/backtracks) vs AE z.
# Usage: sbatch scripts/discovery/score-scas-handcrafted-style.slurm

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../env.sh" ]]; then source "${SCRIPT_DIR}/../env.sh"; elif [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then source "${SCRIPT_DIR}/../../env.sh"; fi
cd "${REPO_ROOT}"
source scripts/scas_model_size.sh qwen3-0p6b
source ./.venv/bin/activate
unset PYTHONPATH PYTHONPATH_modshare 2>/dev/null || true

VENV_PY="./.venv/bin/python"
mkdir -p slurm_logs eval_results

RAW_JSON="${SCAS_TRACE_JSON}"
HF_DS="${TRACE_DATA_PATH}"
KMEANS_OUT="data/scas/cluster_handcrafted_sweep-qwen3-0p6b"

echo "=== Part A: handcrafted style features ==="
"${VENV_PY}" -u scripts/discovery/score_scas_handcrafted_style.py \
  --hf-dataset "${HF_DS}" \
  --raw-json "${RAW_JSON}" \
  --output-json eval_results/scas_handcrafted_style-qwen3-0p6b.json \
  --kmeans-out-dir "${KMEANS_OUT}" \
  --features-npz "${KMEANS_OUT}/handcrafted_features.npz" \
  --min-k 2 --max-k 9 \
  --label "handcrafted-style-scas"

echo "=== summary ==="
"${VENV_PY}" - <<'PY'
import json
from pathlib import Path
d = json.load(open("eval_results/scas_handcrafted_style-qwen3-0p6b.json"))
print("teachers:", len(d["teacher_names"]))
pf, pd_ = d["teacher_probe_full"], d["teacher_probe_density"]
print(f"probe FULL   : acc={pf['val_accuracy']:.4f} macroF1={pf['val_macro_f1']:.4f} (maj={pf['majority_baseline_accuracy']:.4f})")
print(f"probe DENSITY: acc={pd_['val_accuracy']:.4f} macroF1={pd_['val_macro_f1']:.4f}")
print("top features (FULL):", [f["feature"] for f in pf["feature_importance"][:8]])
bf, bd = d["best_by_ari_full"], d["best_by_ari_density"]
print(f"best kmeans FULL   : K={bf['k']} ARI={bf['ari_vs_teacher']:.4f} NMI={bf['nmi_vs_teacher']:.4f} maj={bf['majority_teacher_accuracy']:.4f}")
print(f"best kmeans DENSITY: K={bd['k']} ARI={bd['ari_vs_teacher']:.4f} NMI={bd['nmi_vs_teacher']:.4f} maj={bd['majority_teacher_accuracy']:.4f}")
PY

echo "Done."
