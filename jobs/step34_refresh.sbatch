#!/bin/bash
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --job-name=pdnpj_step34
#SBATCH --output=logs/step34_refresh_%j.out
#SBATCH --error=logs/step34_refresh_%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --gpus-per-node=1

set -euo pipefail

PROJECT_ROOT="${PDNPJ_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="$PROJECT_ROOT/logs"
HOST_STEP4_VENV="${PDNPJ_STEP4_VENV:?Set PDNPJ_STEP4_VENV to your Step 4 host venv path}"
HOST_PYTHON="$HOST_STEP4_VENV/bin/python"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

# Adapt these module loads and container path to your cluster environment.
module load PyTorch/2.8.0 2>/dev/null || true

CONTAINER_PATH="${PYTORCH_CONTAINER:-}"
export PYTHONUNBUFFERED=1

if [[ -n "$CONTAINER_PATH" && -f "$CONTAINER_PATH" ]]; then
  export APPTAINERENV_PYTHONNOUSERSITE=1
  export APPTAINERENV_PYTHONPATH=
  export APPTAINERENV_PYTHONUNBUFFERED=1
  run_in_container() {
    apptainer exec --cleanenv --nv --bind "$HOME:$HOME" "$CONTAINER_PATH" "$@"
  }
else
  run_in_container() { "$@"; }
fi

if [[ -x "$HOST_PYTHON" ]]; then
  run_host_python() {
    env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "$HOST_PYTHON" "$@"
  }
else
  run_host_python() { python "$@"; }
fi

echo "[step34] starting environment probe"
run_in_container python -c "import sys, pandas, numpy, scipy; print('python', sys.version); print('pandas', pandas.__version__); print('numpy', numpy.__version__); print('scipy', scipy.__version__)"

echo "[step34] step3 candidate registry"
run_in_container python "$PROJECT_ROOT/scripts/build_step3_candidate_registry.py"

echo "[step34] step3 ranking"
run_in_container python "$PROJECT_ROOT/scripts/rank_step3_programs.py"

echo "[step34] step3 handoff"
run_in_container python "$PROJECT_ROOT/scripts/export_step3_handoff.py"

echo "[step34] freeze step4 inputs"
run_in_container python "$PROJECT_ROOT/scripts/freeze_step4_inputs.py"

echo "[step34] build step4 validation registry"
run_in_container python "$PROJECT_ROOT/scripts/build_step4_validation_registry.py"

echo "[step34] refresh step4 genetic resources"
run_in_container python "$PROJECT_ROOT/scripts/build_step4_genetic_resources.py"

echo "[step34] rerun step4 gene-level genetics"
run_in_container python "$PROJECT_ROOT/scripts/run_step4_gene_level_analyses.py"

echo "[step34] rerun step4 program-level genetics"
run_in_container python "$PROJECT_ROOT/scripts/run_step4_program_level_genetics.py"

echo "[step34] rescore step4 spatial programs"
run_host_python "$PROJECT_ROOT/scripts/score_step4_spatial_programs.py"

echo "[step34] rerun step4 primary spatial tests"
run_in_container python "$PROJECT_ROOT/scripts/run_step4_primary_spatial_tests.py"

echo "[step34] rerun step4 secondary spatial replication"
run_in_container python "$PROJECT_ROOT/scripts/run_step4_secondary_spatial_replication.py"

echo "[step34] rerun step4 internal robustness"
run_in_container python "$PROJECT_ROOT/scripts/run_step4_internal_robustness.py"

echo "[step34] integrate step4 evidence"
run_in_container python "$PROJECT_ROOT/scripts/integrate_step4_evidence.py"

echo "[step34] build step4 reports"
run_in_container python "$PROJECT_ROOT/scripts/build_step4_reports.py"

echo "[step34] completed"
