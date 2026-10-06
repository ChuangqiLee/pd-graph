#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH_SCRIPT="$PROJECT_ROOT/jobs/step2_transformer_single_gpu.sbatch"
AGE_CONFIG="${1:-$PROJECT_ROOT/configs/step2/step2_model_age.yaml}"
PD_CONFIG="${2:-$PROJECT_ROOT/configs/step2/step2_model_pd.yaml}"
SKIP_DOWNSTREAM="${3:-0}"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

sbatch \
  --export=ALL,STEP2_AGE_CONFIG="$AGE_CONFIG",STEP2_PD_CONFIG="$PD_CONFIG",STEP2_SKIP_DOWNSTREAM="$SKIP_DOWNSTREAM" \
  "$SBATCH_SCRIPT"
