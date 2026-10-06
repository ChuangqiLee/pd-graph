#!/bin/bash
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --job-name=pdnpj_step2cpu
#SBATCH --output=logs/step2_refresh_cpu_%j.out
#SBATCH --error=logs/step2_refresh_cpu_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G

set -euo pipefail

PROJECT_ROOT="${PDNPJ_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PATH="${PDNPJ_STEP_CPU_VENV:?Set PDNPJ_STEP_CPU_VENV to your CPU venv path}"
VENV_SITE_PACKAGES="$VENV_PATH/lib/python3.12/site-packages"

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

# Adapt module loads to your cluster environment, or activate a virtualenv directly.
module --force purge 2>/dev/null || true
source "$VENV_PATH/bin/activate" 2>/dev/null || true
PYTHON_BIN="$(command -v python)"

echo "[step2-cpu] python: $PYTHON_BIN"
echo "[step2-cpu] executable check: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "[step2-cpu] freezing backbone"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_step2_backbone_manifest.py"

echo "[step2-cpu] building communication gene panel"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_step2_comm_gene_panel.py"

echo "[step2-cpu] rehydrating expression matrix"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/rehydrate_step2_expression.py"

echo "[step2-cpu] applying dopaminergic rescue"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/apply_step2_dopaminergic_rescue.py"

echo "[step2-cpu] building task cohorts"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_step2_cohorts.py"

echo "[step2-cpu] building ligand-receptor universe"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_step2_lr_universe.py"

echo "[step2-cpu] building graphs"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_step2_graphs.py"

echo "[step2-cpu] completed"
