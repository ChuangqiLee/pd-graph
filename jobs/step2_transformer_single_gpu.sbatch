#!/bin/bash
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --job-name=pdnpj_step2_ht
#SBATCH --output=logs/step2_transformer_%j.out
#SBATCH --error=logs/step2_transformer_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=110G
#SBATCH --gpus-per-node=1

set -euo pipefail

PROJECT_ROOT="${PDNPJ_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="$PROJECT_ROOT/logs"
AGE_CONFIG="${STEP2_AGE_CONFIG:-$PROJECT_ROOT/configs/step2/step2_model_age.yaml}"
PD_CONFIG="${STEP2_PD_CONFIG:-$PROJECT_ROOT/configs/step2/step2_model_pd.yaml}"
SKIP_DOWNSTREAM="${STEP2_SKIP_DOWNSTREAM:-0}"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

# Adapt these module loads and container path to your cluster environment.
# Example: module load cuda/12.x pytorch/2.x
# If not using containers, replace run_in_container() with direct python calls.
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

echo "[step2] starting environment probe"
run_in_container python -c "import sys, torch, pandas, numpy; print('python', sys.version); print('torch', torch.__version__); print('pandas', pandas.__version__); print('numpy', numpy.__version__); print('cuda_available', torch.cuda.is_available()); print('gpu_count', torch.cuda.device_count())"

echo "[step2] training age model"
run_in_container python "$PROJECT_ROOT/scripts/train_step2_age_model.py" --config "$AGE_CONFIG"

echo "[step2] training pd model"
run_in_container python "$PROJECT_ROOT/scripts/train_step2_pd_model.py" --config "$PD_CONFIG"

if [[ "$SKIP_DOWNSTREAM" == "1" ]]; then
  echo "[step2] skipping robustness, interpretation, and export"
  echo "[step2] completed"
  exit 0
fi

echo "[step2] running robustness and external holdout scoring"
run_in_container python "$PROJECT_ROOT/scripts/run_step2_robustness.py" --age-config "$AGE_CONFIG" --pd-config "$PD_CONFIG"

echo "[step2] interpreting upgraded models"
run_in_container python "$PROJECT_ROOT/scripts/interpret_step2_models.py"

echo "[step2] exporting downstream artifacts"
run_in_container python "$PROJECT_ROOT/scripts/export_step2_results.py"

echo "[step2] completed"
