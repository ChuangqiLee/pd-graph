#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEP1_SBATCH="$PROJECT_ROOT/jobs/step1_rebuild_atlas_cpu.sbatch"
STEP2_CPU_SBATCH="$PROJECT_ROOT/jobs/step2_refresh_cpu.sbatch"
STEP2_GPU_SBATCH="$PROJECT_ROOT/jobs/step2_transformer_single_gpu.sbatch"
STEP34_SBATCH="$PROJECT_ROOT/jobs/step34_refresh.sbatch"

AGE_CONFIG="${1:-$PROJECT_ROOT/configs/step2/step2_model_age.yaml}"
PD_CONFIG="${2:-$PROJECT_ROOT/configs/step2/step2_model_pd.yaml}"
STEP1_RUN_QC_REVIEW="${3:-0}"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

step1_job_id=$(
  sbatch --parsable \
    --export=ALL,PDNPJ_STEP1_RUN_QC_REVIEW="$STEP1_RUN_QC_REVIEW" \
    "$STEP1_SBATCH"
)
echo "[submit] step1_rebuild_atlas_cpu: $step1_job_id"

step2_cpu_job_id=$(
  sbatch --parsable \
    --dependency=afterok:"$step1_job_id" \
    "$STEP2_CPU_SBATCH"
)
echo "[submit] step2_refresh_cpu: $step2_cpu_job_id"

step2_gpu_job_id=$(
  sbatch --parsable \
    --dependency=afterok:"$step2_cpu_job_id" \
    --export=ALL,STEP2_AGE_CONFIG="$AGE_CONFIG",STEP2_PD_CONFIG="$PD_CONFIG",STEP2_SKIP_DOWNSTREAM=0 \
    "$STEP2_GPU_SBATCH"
)
echo "[submit] step2_transformer_gpu: $step2_gpu_job_id"

step34_job_id=$(
  sbatch --parsable \
    --dependency=afterok:"$step2_gpu_job_id" \
    "$STEP34_SBATCH"
)
echo "[submit] step34_refresh: $step34_job_id"

echo "[submit] full_refresh_chain_complete"
