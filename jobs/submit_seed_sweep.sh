#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEP2_JOB="$PROJECT_ROOT/jobs/step2_transformer_single_gpu.sbatch"
STEP34_JOB="$PROJECT_ROOT/jobs/step34_refresh.sbatch"
ARCHIVE_JOB="$PROJECT_ROOT/jobs/archive_full_refresh.sbatch"

INITIAL_AFTER_JOB=""
if [[ "${1:-}" == "--after-job" ]]; then
  INITIAL_AFTER_JOB="${2:-}"
  shift 2
fi

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 [--after-job <jobid>] <base-tag> <seed1> [seed2 ...]" >&2
  exit 1
fi

BASE_TAG="$1"
shift

PREV_ARCHIVE_JOB="$INITIAL_AFTER_JOB"

for SEED in "$@"; do
  TAG="${BASE_TAG}_seed${SEED}"
  AGE_CONFIG="$PROJECT_ROOT/configs/step2/experiments/$TAG/step2_model_age.yaml"
  PD_CONFIG="$PROJECT_ROOT/configs/step2/experiments/$TAG/step2_model_pd.yaml"

  STEP2_ARGS=()
  if [[ -n "$PREV_ARCHIVE_JOB" ]]; then
    STEP2_ARGS+=(--dependency "afterok:${PREV_ARCHIVE_JOB}")
  fi

  STEP2_SUBMIT=$(
    sbatch --parsable \
      "${STEP2_ARGS[@]}" \
      --export=ALL,STEP2_AGE_CONFIG="$AGE_CONFIG",STEP2_PD_CONFIG="$PD_CONFIG" \
      "$STEP2_JOB"
  )
  STEP2_ID="${STEP2_SUBMIT%%;*}"

  STEP34_SUBMIT=$(
    sbatch --parsable \
      --dependency "afterok:${STEP2_ID}" \
      "$STEP34_JOB"
  )
  STEP34_ID="${STEP34_SUBMIT%%;*}"

  ARCHIVE_SUBMIT=$(
    sbatch --parsable \
      --dependency "afterok:${STEP34_ID}" \
      --export=ALL,ARCHIVE_TAG="$TAG" \
      "$ARCHIVE_JOB"
  )
  ARCHIVE_ID="${ARCHIVE_SUBMIT%%;*}"

  echo "${TAG},step2=${STEP2_ID},step34=${STEP34_ID},archive=${ARCHIVE_ID}"
  PREV_ARCHIVE_JOB="$ARCHIVE_ID"
done
