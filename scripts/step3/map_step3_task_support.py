#!/usr/bin/env python3.12
"""Map candidate support across the four Step 2 graph tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import markdown_table, project_path, write_markdown
from scripts.step3.common import TASK_DISPLAY, TASK_ORDER


DEFAULT_CONFIG = Path("configs/step3/step3_task_support.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map Step 3 task support.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def summarize_task_support(frame: pd.DataFrame, *, level: str, shared_floor: float, axis_ratio: float) -> pd.DataFrame:
    task_support = (
        frame.groupby(["candidate_id", "task_name"], observed=False)["score_normalized"]
        .apply(lambda values: values.abs().mean())
        .rename("task_support")
        .reset_index()
    )
    pivot = task_support.pivot_table(index="candidate_id", columns="task_name", values="task_support", fill_value=0.0).reset_index()
    pivot.columns.name = None
    for task in TASK_ORDER:
        if task not in pivot.columns:
            pivot[task] = 0.0
    pivot["primary_task"] = pivot[TASK_ORDER].idxmax(axis=1)
    pivot["task_support_count"] = pivot[TASK_ORDER].apply(
        lambda row: int((row >= row.max() * shared_floor).sum()) if float(row.max()) > 0 else 0,
        axis=1,
    )
    pivot["task_specificity_score"] = pivot[TASK_ORDER].apply(
        lambda row: float(row.max()) / float(max(row.drop(labels=[row.idxmax()]).mean(), 1e-6)) if float(row.max()) > 0 else 0.0,
        axis=1,
    )
    pivot["receiver_specificity_task"] = pivot["primary_task"]
    pivot["task_consistency_status"] = [
        "axis_specific" if count <= 1 or specificity >= axis_ratio else "shared_multitask"
        for count, specificity in zip(pivot["task_support_count"], pivot["task_specificity_score"])
    ]
    pivot["candidate_level"] = level
    return pivot


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    output_dir = project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_floor = float(config["thresholds"]["support_fraction_for_shared"])
    axis_ratio = float(config["thresholds"]["axis_specific_ratio"])

    donor_dir = project_path("data/processed/step3/03_donor_patterns")
    pair_scores = pd.read_csv(donor_dir / "step3_pair_scores_by_donor.csv")
    program_scores = pd.read_csv(donor_dir / "step3_program_scores_by_donor.csv")

    pair_matrix = summarize_task_support(pair_scores, level="pair", shared_floor=shared_floor, axis_ratio=axis_ratio)
    program_matrix = summarize_task_support(program_scores, level="program", shared_floor=shared_floor, axis_ratio=axis_ratio)
    task_matrix = pd.concat([pair_matrix, program_matrix], ignore_index=True, sort=False)
    task_matrix.to_csv(output_dir / "step3_task_support_matrix.csv", index=False)

    interesting = task_matrix.loc[
        task_matrix["candidate_level"].eq("program") & task_matrix["primary_task"].ne("microglia_to_dopaminergic_neuron")
    ][["candidate_id", "primary_task", "task_support_count", "task_specificity_score"]].copy()
    for task in TASK_ORDER:
        interesting[task] = task_matrix.set_index("candidate_id").reindex(interesting["candidate_id"])[task].to_numpy()

    report = "\n".join(
        [
            "# Step 3 Task Specificity Report",
            "",
            "## Task Support Matrix Preview",
            markdown_table(task_matrix.head(20).fillna("NA")),
            "",
            "## Programs With Non-primary Microglia->DA Support",
            markdown_table(interesting.fillna("NA")) if not interesting.empty else "_All current programs remain strongest in the primary axis._",
            "",
            "## Task Labels",
            "\n".join(f"- `{task}`: {label}" for task, label in TASK_DISPLAY.items()),
        ]
    )
    write_markdown(output_dir / "step3_task_specificity_report.md", report)


if __name__ == "__main__":
    main()
