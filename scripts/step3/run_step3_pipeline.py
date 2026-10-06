#!/usr/bin/env python3.12
"""Run the full Step 3 pipeline in sequence."""

from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPTS = [
    "freeze_step3_inputs.py",
    "build_step3_candidate_registry.py",
    "refine_step3_programs.py",
    "compute_step3_pair_scores.py",
    "model_step3_donor_trajectories.py",
    "map_step3_task_support.py",
    "build_step3_lineage_states.py",
    "map_step3_state_support.py",
    "extract_step3_exemplars.py",
    "run_step3_confound_checks.py",
    "build_step3_hypothesis_reconciliation.py",
    "rank_step3_programs.py",
    "render_step3_figures.py",
    "export_step3_handoff.py",
]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    for script_name in SCRIPTS:
        script_path = project_root / "scripts" / script_name
        print(f"[step3] running {script_name}", flush=True)
        subprocess.run(["python3.12", str(script_path)], cwd=project_root, check=True)


if __name__ == "__main__":
    main()

