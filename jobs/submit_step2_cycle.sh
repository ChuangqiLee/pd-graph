#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEP2_SCRIPT="$PROJECT_ROOT/jobs/step2_pd_refresh.sbatch"
STEP34_SCRIPT="$PROJECT_ROOT/jobs/step34_refresh.sbatch"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

STEP2_JOB_ID=$(sbatch --parsable "$STEP2_SCRIPT")
STEP34_JOB_ID=$(sbatch --parsable --dependency=afterok:${STEP2_JOB_ID} "$STEP34_SCRIPT")

echo "step2_job=${STEP2_JOB_ID}"
echo "step34_job=${STEP34_JOB_ID}"
