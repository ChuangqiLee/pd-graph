#!/bin/bash
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --job-name=pdnpj_archive
#SBATCH --output=logs/archive_full_refresh_%j.out
#SBATCH --error=logs/archive_full_refresh_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

PROJECT_ROOT="${PDNPJ_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARCHIVE_TAG="${ARCHIVE_TAG:?ARCHIVE_TAG must be set}"

cd "$PROJECT_ROOT"
# Adapt module loads to your cluster environment, or activate a virtualenv directly.
module --force purge 2>/dev/null || true
python "$PROJECT_ROOT/scripts/archive_full_refresh_outputs.py" --tag "$ARCHIVE_TAG"
