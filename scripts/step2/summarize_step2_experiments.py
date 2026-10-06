#!/usr/bin/env python3
"""Summarize archived Step 2 experiment outputs into a simple leaderboard."""

import argparse
import csv
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "data/processed/step2/experiments"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/step2/experiment_summaries"


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize Step 2 experiment metrics.")
    parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def read_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
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


def format_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def collect_experiment_rows(experiment_root):
    rows = []
    if not experiment_root.exists():
        return rows

    for experiment_dir in sorted(path for path in experiment_root.iterdir() if path.is_dir()):
        pd_dir = experiment_dir / "06_models" / "pd_model"
        metrics_path = pd_dir / "pd_model_fold_metrics.csv"
        bridge_path = pd_dir / "pd_model_lbd_bridge_scores.csv"
        report_path = pd_dir / "pd_model_report.md"
        if not metrics_path.exists():
            continue

        metrics_rows = read_csv_rows(metrics_path)
        bridge_rows = read_csv_rows(bridge_path) if bridge_path.exists() else []
        report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""

        row = {
            "experiment": experiment_dir.name,
            "n_folds": len(metrics_rows),
            "mean_auroc": mean([safe_float(value.get("auroc")) for value in metrics_rows]),
            "mean_balanced_accuracy": mean([safe_float(value.get("balanced_accuracy")) for value in metrics_rows]),
            "mean_brier_score": mean([safe_float(value.get("brier_score")) for value in metrics_rows]),
            "mean_log_loss": mean([safe_float(value.get("log_loss")) for value in metrics_rows]),
            "mean_threshold": mean([safe_float(value.get("decision_threshold")) for value in metrics_rows]),
            "bridge_mean_pd_probability": mean([safe_float(value.get("pd_probability")) for value in bridge_rows]),
            "completed": "Mean validation AUROC" in report_text,
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            not bool(row["completed"]),
            -row["mean_auroc"] if math.isfinite(row["mean_auroc"]) else float("inf"),
            -row["mean_balanced_accuracy"] if math.isfinite(row["mean_balanced_accuracy"]) else float("inf"),
            row["mean_brier_score"] if math.isfinite(row["mean_brier_score"]) else float("inf"),
        )
    )
    return rows


def markdown_table(rows, columns):
    if not rows:
        return "| experiment | status |\n| --- | --- |\n| none | no completed experiment outputs found |"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(format_value(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, separator] + body)


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_experiment_rows(experiment_root)
    columns = [
        "experiment",
        "completed",
        "n_folds",
        "mean_auroc",
        "mean_balanced_accuracy",
        "mean_brier_score",
        "mean_log_loss",
        "mean_threshold",
        "bridge_mean_pd_probability",
    ]

    write_csv(output_dir / "step2_experiment_leaderboard.csv", rows, columns)
    report_lines = [
        "# Step 2 Experiment Leaderboard",
        "",
        markdown_table(rows, columns),
    ]
    (output_dir / "step2_experiment_leaderboard.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
