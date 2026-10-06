#!/usr/bin/env python3.12
"""Freeze Step 3 inputs and record provenance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import markdown_table, project_path
from scripts.step3.common import stage_file, write_simple_report


DEFAULT_CONFIG = Path("configs/step3/step3_inputs.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze Step 3 inputs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    output_dir = project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    for entry in config["inputs"]:
        source_path = project_path(entry["source"])
        if not source_path.exists():
            missing_rows.append(
                {
                    "source_path": str(source_path),
                    "source_phase": entry.get("source_phase", "unknown"),
                    "reason": "missing",
                }
            )
            continue
        frozen_path = output_dir / entry["frozen_relpath"]
        staged = stage_file(source_path, frozen_path, mode=str(entry.get("mode", "copy")))
        manifest_rows.append(
            {
                **staged,
                "source_phase": entry.get("source_phase", "unknown"),
                "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
            }
        )

    manifest = pd.DataFrame(manifest_rows).sort_values(["source_phase", "source_path"]).reset_index(drop=True)
    manifest.to_csv(output_dir / "step3_input_manifest.csv", index=False)

    notes = config.get("notes", [])
    summary = [
        ("Frozen files", f"Staged {len(manifest_rows)} files into `{output_dir}`."),
        (
            "Manifest preview",
            markdown_table(manifest.head(12).fillna("NA")) if not manifest.empty else "_No staged files._",
        ),
        (
            "Missing artifacts",
            markdown_table(pd.DataFrame(missing_rows).fillna("NA")) if missing_rows else "_No required files were missing._",
        ),
        (
            "Notes",
            "\n".join(f"- {note}" for note in notes) if notes else "_No additional notes._",
        ),
    ]
    write_simple_report(
        output_dir / "step3_missing_artifacts_report.md",
        "Step 3 Missing Artifacts Report",
        summary,
    )


if __name__ == "__main__":
    main()
