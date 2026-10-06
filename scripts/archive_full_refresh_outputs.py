#!/usr/bin/env python3
"""Archive the current Step 2 and Step 4 refresh outputs under a stable experiment tag."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_ROOT = PROJECT_ROOT / "data/processed/full_refresh_archives"

FILES_TO_COPY = {
    "step2_top_candidates_for_step3.csv": PROJECT_ROOT / "data/processed/step2/09_exports/step2_top_candidates_for_step3.csv",
    "step3_top10_refined_programs.csv": PROJECT_ROOT / "data/processed/step3/09_top10/step3_top10_refined_programs.csv",
    "step3_pair_appendix.csv": PROJECT_ROOT / "data/processed/step3/10_reports/step3_pair_appendix.csv",
    "step3_refined_program_map.csv": PROJECT_ROOT / "data/processed/step3/02_programs/step3_refined_program_map.csv",
    "step3_task_support_matrix.csv": PROJECT_ROOT / "data/processed/step3/04_task_support/step3_task_support_matrix.csv",
    "step3_candidate_state_support.csv": PROJECT_ROOT / "data/processed/step3/05_states/step3_candidate_state_support.csv",
    "step3_state_marker_summary.csv": PROJECT_ROOT / "data/processed/step3/05_states/step3_state_marker_summary.csv",
    "step4_validation_registry.csv": PROJECT_ROOT / "data/processed/step4/01_registry/step4_validation_registry.csv",
    "step4_final_evidence_matrix.csv": PROJECT_ROOT / "data/processed/step4/05_integration/step4_final_evidence_matrix.csv",
    "STEP4_VALIDATION_SUMMARY.md": PROJECT_ROOT / "data/processed/step4/06_reports/STEP4_VALIDATION_SUMMARY.md",
    "pd_model_report.md": PROJECT_ROOT / "data/processed/step2/06_models/pd_model/pd_model_report.md",
    "pd_model_fold_metrics.csv": PROJECT_ROOT / "data/processed/step2/06_models/pd_model/pd_model_fold_metrics.csv",
    "pd_model_thresholds.csv": PROJECT_ROOT / "data/processed/step2/06_models/pd_model/pd_model_thresholds.csv",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive current refresh outputs.")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_root = Path(args.archive_root)
    archive_dir = archive_root / args.tag
    archive_dir.mkdir(parents=True, exist_ok=True)

    for name, source in FILES_TO_COPY.items():
        if source.exists():
            shutil.copy2(source, archive_dir / name)

    config_archive_dir = archive_dir / "configs"
    config_archive_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = PROJECT_ROOT / "configs/step2/experiments" / args.tag
    if experiment_dir.exists():
        shutil.copy2(experiment_dir / "step2_model_age.yaml", config_archive_dir / "step2_model_age.yaml")
        shutil.copy2(experiment_dir / "step2_model_pd.yaml", config_archive_dir / "step2_model_pd.yaml")

    metadata_lines = [
        f"tag={args.tag}",
        f"archive_dir={archive_dir}",
    ]
    (archive_dir / "archive_metadata.txt").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/summarize_full_refresh_archives.py")],
        check=True,
        cwd=str(PROJECT_ROOT),
    )
    print(archive_dir)


if __name__ == "__main__":
    main()
