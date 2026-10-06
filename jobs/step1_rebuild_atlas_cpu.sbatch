#!/bin/bash
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --job-name=pdnpj_step1
#SBATCH --output=logs/step1_rebuild_%j.out
#SBATCH --error=logs/step1_rebuild_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=220G

set -euo pipefail

PROJECT_ROOT="${PDNPJ_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PATH="${PDNPJ_STEP_CPU_VENV:?Set PDNPJ_STEP_CPU_VENV to your CPU venv path}"
VENV_SITE_PACKAGES="$VENV_PATH/lib/python3.12/site-packages"
RUN_QC_REVIEW="${PDNPJ_STEP1_RUN_QC_REVIEW:-0}"

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

# Adapt module loads to your cluster environment, or activate a virtualenv directly.
module --force purge 2>/dev/null || true
source "$VENV_PATH/bin/activate" 2>/dev/null || true
PYTHON_BIN="$(command -v python)"

echo "[step1] python: $PYTHON_BIN"
echo "[step1] executable check: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "[step1] starting atlas rebuild"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_step1b_atlas.py"

echo "[step1] repairing atlas indices"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/repair_step1b_atlas.py"

if [[ "$RUN_QC_REVIEW" == "1" ]]; then
  echo "[step1] rendering atlas QC views"
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/visualize_step1b_atlas.py"
  echo "[step1] running joint atlas QC"
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/joint_check_step1b_atlas.py"
fi

echo "[step1] completed"
