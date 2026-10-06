#!/usr/bin/env python3.12
"""Run the full Step 4 pipeline in sequence."""

from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPTS = [
    "freeze_step4_inputs.py",
    "build_step4_validation_registry.py",
    "build_step4_genetic_resources.py",
    "run_step4_gene_level_analyses.py",
    "run_step4_program_level_genetics.py",
    "ingest_step4_primary_visium.py",
    "map_step4_states_to_spatial.py",
    "score_step4_spatial_programs.py",
    "run_step4_primary_spatial_tests.py",
    "run_step4_secondary_spatial_replication.py",
    "run_step4_internal_robustness.py",
    "integrate_step4_evidence.py",
    "build_step4_reports.py",
]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    for script_name in SCRIPTS:
        script_path = project_root / "scripts" / script_name
        print(f"[step4] running {script_name}", flush=True)
        subprocess.run(["python3.12", str(script_path)], cwd=project_root, check=True)


if __name__ == "__main__":
    main()
