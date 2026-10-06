#!/bin/bash
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --job-name=pdnpj_step2pd
#SBATCH --output=logs/step2_pd_refresh_%j.out
#SBATCH --error=logs/step2_pd_refresh_%j.err
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=110G
#SBATCH --gpus-per-node=1

set -euo pipefail

PROJECT_ROOT="${PDNPJ_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

# Adapt these module loads and container path to your cluster environment.
module load PyTorch/2.8.0 2>/dev/null || true

CONTAINER_PATH="${PYTORCH_CONTAINER:-}"
if [[ -n "$CONTAINER_PATH" && -f "$CONTAINER_PATH" ]]; then
  export APPTAINERENV_PYTHONNOUSERSITE=1
  export APPTAINERENV_PYTHONPATH=
  run_in_container() {
    apptainer exec --cleanenv --nv --bind "$HOME:$HOME" "$CONTAINER_PATH" "$@"
  }
else
  run_in_container() { "$@"; }
fi

echo "[step2-pd] starting environment probe"
run_in_container python -c "import sys, torch, pandas, numpy; print('python', sys.version); print('torch', torch.__version__); print('pandas', pandas.__version__); print('numpy', numpy.__version__); print('cuda_available', torch.cuda.is_available()); print('gpu_count', torch.cuda.device_count())"

echo "[step2-pd] training pd model"
run_in_container python "$PROJECT_ROOT/scripts/train_step2_pd_model.py"

echo "[step2-pd] running robustness"
run_in_container python "$PROJECT_ROOT/scripts/run_step2_robustness.py"

echo "[step2-pd] interpreting models"
run_in_container python "$PROJECT_ROOT/scripts/interpret_step2_models.py"

echo "[step2-pd] exporting downstream artifacts"
run_in_container python "$PROJECT_ROOT/scripts/export_step2_results.py"

echo "[step2-pd] completed"
