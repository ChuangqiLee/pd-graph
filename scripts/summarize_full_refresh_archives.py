#!/usr/bin/env python3
"""Summarize archived full refresh experiments into a compact leaderboard."""

import argparse
import csv
import math
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step4.common import STEP4_FINAL_MATRIX_COLUMNS, read_csv_required_columns, require_columns
DEFAULT_ARCHIVE_ROOT = PROJECT_ROOT / "data/processed/full_refresh_archives"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/full_refresh_archives_summary"


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize archived full refresh experiments.")
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def read_csv_rows(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean(values):
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def markdown_table(rows, columns):
    if not rows:
        return "| archive | status |\n| --- | --- |\n| none | no archives found |"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        formatted = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                if math.isnan(value):
                    formatted.append("nan")
                else:
                    formatted.append("{:.4f}".format(value))
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    archive_root = Path(args.archive_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if archive_root.exists():
        for archive_dir in sorted(path for path in archive_root.iterdir() if path.is_dir()):
            final_matrix_path = archive_dir / "step4_final_evidence_matrix.csv"
            pd_metrics_path = archive_dir / "pd_model_fold_metrics.csv"
            if not final_matrix_path.exists():
                continue

            try:
                final_matrix = read_csv_required_columns(
                    final_matrix_path,
                    STEP4_FINAL_MATRIX_COLUMNS,
                    context=f"Archived final matrix {archive_dir.name}",
                )
                final_rows = final_matrix.to_dict(orient="records")
                if pd_metrics_path.exists():
                    pd_metric_frame = pd.read_csv(pd_metrics_path)
                    require_columns(
                        pd_metric_frame,
                        ("auroc", "balanced_accuracy"),
                        context=f"Archived PD metrics {archive_dir.name}",
                    )
                    pd_metric_rows = pd_metric_frame.to_dict(orient="records")
                else:
                    pd_metric_rows = []
            except Exception:
                continue

            externally_supported = sum(
                1
                for row in final_rows
                if row.get("claim_tier") in {"externally_supported_program", "external_mechanistic_candidate"}
            )
            replication_priority = sum(
                1
                for row in final_rows
                if row.get("promotion_decision") in {"priority_replication_candidate", "main_figure_candidate"}
            )
            top_program_id = final_rows[0].get("program_id", "NA") if final_rows else "NA"
            top_claim_tier = final_rows[0].get("claim_tier", "NA") if final_rows else "NA"
            top_final_step4_score = max((safe_float(row.get("final_step4_score")) for row in final_rows), default=float("nan"))

            rows.append(
                {
                    "archive": archive_dir.name,
                    "n_programs": len(final_rows),
                    "externally_supported_programs": externally_supported,
                    "priority_replication_candidates": replication_priority,
                    "top_claim_tier": top_claim_tier,
                    "top_program_id": top_program_id,
                    "top_final_step4_score": top_final_step4_score,
                    "mean_pd_auroc": mean(safe_float(row.get("auroc")) for row in pd_metric_rows),
                    "mean_pd_balanced_accuracy": mean(safe_float(row.get("balanced_accuracy")) for row in pd_metric_rows),
                }
            )

    rows.sort(
        key=lambda row: (
            -int(row["externally_supported_programs"]),
            -int(row["priority_replication_candidates"]),
            -float(row["top_final_step4_score"]) if math.isfinite(float(row["top_final_step4_score"])) else float("inf"),
            -float(row["mean_pd_auroc"]) if math.isfinite(float(row["mean_pd_auroc"])) else float("inf"),
            -float(row["mean_pd_balanced_accuracy"]) if math.isfinite(float(row["mean_pd_balanced_accuracy"])) else float("inf"),
        )
    )

    columns = [
        "archive",
        "n_programs",
        "externally_supported_programs",
        "priority_replication_candidates",
        "top_claim_tier",
        "top_program_id",
        "top_final_step4_score",
        "mean_pd_auroc",
        "mean_pd_balanced_accuracy",
    ]

    write_csv(output_dir / "full_refresh_leaderboard.csv", rows, columns)
    report_lines = [
        "# Full Refresh Leaderboard",
        "",
        markdown_table(rows, columns),
        "",
    ]
    (output_dir / "full_refresh_leaderboard.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
