#!/usr/bin/env python3.12
"""Build the manuscript-ready Step 4 summary package."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import markdown_table, write_markdown
from scripts.step4.common import STEP4_FINAL_MATRIX_COLUMNS, ensure_step4_dir, read_csv_required_columns


DEFAULT_OUTPUT_DIR = Path("data/processed/step4/06_reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Step 4 report bundle.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_step4_dir("06_reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    final_matrix = read_csv_required_columns(
        PROJECT_ROOT / "data/processed/step4/05_integration/step4_final_evidence_matrix.csv",
        STEP4_FINAL_MATRIX_COLUMNS,
        context="Step 4 final evidence matrix",
    )

    class_column = "claim_tier" if "claim_tier" in final_matrix.columns else "final_evidence_class"
    class_counts = final_matrix[class_column].value_counts().rename_axis(class_column).reset_index(name="n_programs")
    top_summary = final_matrix.sort_values("final_step4_score", ascending=False)[
        [
            "program_id",
            "program_name",
            "final_step4_score",
            class_column,
            *(["promotion_decision"] if "promotion_decision" in final_matrix.columns else []),
            "primary_spatial_support_bin",
            "secondary_spatial_support_bin",
            "program_genetic_support_bin",
        ]
    ]
    summary_lines = [
        "# Step 4 Validation Summary",
        "",
        "This report summarizes the complete external validation layer built from the frozen Step 3 handoff, the primary human Visium substantia nigra dataset, the secondary GeoMx SNpc replication workbook, and the public PD genetics resources.",
        "",
        "## Final class counts",
        markdown_table(class_counts.astype(object).fillna("NA")),
        "",
        "## Program table",
        markdown_table(top_summary.astype(object).fillna("NA")),
        "",
    ]
    write_markdown(output_dir / "STEP4_VALIDATION_SUMMARY.md", "\n".join(summary_lines))

    figure_dir = PROJECT_ROOT / "data/processed/step4/03_spatial/04_primary_tests/program_primary_spatial_figures"
    figure_rows = [{"figure_file": path.name, "figure_path": str(path.relative_to(PROJECT_ROOT))} for path in sorted(figure_dir.glob("*.png"))]
    figure_index = pd.DataFrame(figure_rows)
    write_markdown(
        output_dir / "STEP4_FIGURE_INDEX.md",
        "\n".join(
            [
                "# Step 4 Figure Index",
                "",
                markdown_table(figure_index.astype(object).fillna("NA")) if not figure_index.empty else "_No figures found._",
                "",
            ]
        ),
    )

    manuscript = final_matrix[
        [
            "program_id",
            "program_name",
            class_column,
            *(["promotion_decision"] if "promotion_decision" in final_matrix.columns else []),
            "final_step4_score",
            "primary_spatial_support_bin",
            "secondary_spatial_support_bin",
            "program_genetic_support_bin",
            "decision_rationale",
        ]
    ].copy()
    manuscript.to_csv(output_dir / "step4_results_table_for_manuscript.csv", index=False)


if __name__ == "__main__":
    main()
