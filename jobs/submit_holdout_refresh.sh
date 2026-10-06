#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH_SCRIPT="$PROJECT_ROOT/jobs/holdout_refresh.sbatch"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

sbatch "$SBATCH_SCRIPT"
