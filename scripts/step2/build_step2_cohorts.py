#!/usr/bin/env python3
"""Build Step 2 graph-unit cohorts and task manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import load_yaml, markdown_table, project_path, write_markdown


DEFAULT_CONFIG = Path("configs/step2/step2_cohorts.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step 2 graph-unit cohorts and task manifests.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def assign_cohort(row: pd.Series, config: dict[str, object]) -> str:
    disease = str(row["disease_group"])
    age_present = not pd.isna(row["age_years"])
    if disease == "control" and age_present:
        return "control_known_age"
    if disease == "pd" and age_present:
        return "pd_known_age"
    if disease == "lbd":
        return "lbd_bridge"
    return "unknown_age_auxiliary"


def resolve_task_minimum(task_config: dict[str, object], source_id: str, field: str) -> int:
    default_value = int(task_config[field])
    overrides = task_config.get(f"{field}_by_source", {})
    if not isinstance(overrides, dict):
        return default_value
    return int(overrides.get(str(source_id), default_value))


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_dir = project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    backbone = pd.read_parquet(project_path(config["paths"]["backbone_manifest"]))
    primary_cell_types = set(config["primary_cell_types"])
    default_threshold = float(config["transferred_label_confidence_default"])

    cell_type = backbone["cell_type_standardized_final"].astype(str)
    label_source = backbone["cell_type_label_source"].astype(str)
    confidence = pd.to_numeric(backbone["cell_type_prediction_confidence"], errors="coerce")
    backbone["included_primary_default"] = (
        backbone["pass_qc"].astype(bool)
        & cell_type.isin(primary_cell_types)
        & (
            label_source.ne("transferred")
            | confidence.ge(default_threshold)
        )
    )
    for threshold in config["transferred_label_confidence_sensitivity"]:
        threshold_name = str(threshold).replace(".", "_")
        backbone[f"included_primary_conf_{threshold_name}"] = (
            backbone["pass_qc"].astype(bool)
            & cell_type.isin(primary_cell_types)
            & (
                label_source.ne("transferred")
                | confidence.ge(float(threshold))
            )
        )

    inclusion_cells = backbone.loc[backbone["included_primary_default"]].copy()
    counts = pd.crosstab(inclusion_cells["graph_unit_id"], inclusion_cells["cell_type_standardized_final"])
    counts.columns = [str(column) for column in counts.columns]
    counts = counts.reset_index()
    for required in config["primary_cell_types"]:
        if required not in counts.columns:
            counts[required] = 0

    graph_meta = (
        backbone.groupby("graph_unit_id", observed=False)
        .agg(
            source_id=("source_id", "first"),
            sample_id=("sample_id", "first"),
            sample_accession=("sample_accession", "first"),
            donor_id=("donor_id", "first"),
            split_group_id=("split_group_id", "first"),
            disease_group=("disease_group", "first"),
            age_years=("age_years", "first"),
            age_group=("age_group", "first"),
            sex_standardized=("sex_standardized", "first"),
            brain_region_standardized=("brain_region_standardized", "first"),
            graph_unit_level=("graph_unit_level", "first"),
            split_group_level=("split_group_level", "first"),
            total_cells=("cell_id", "count"),
        )
        .reset_index()
    )
    graph_meta = graph_meta.merge(counts, on="graph_unit_id", how="left")
    count_columns = [cell_type for cell_type in config["primary_cell_types"] if cell_type in graph_meta.columns]
    graph_meta[count_columns] = graph_meta[count_columns].fillna(0).astype(int)
    graph_meta["cohort_name"] = graph_meta.apply(assign_cohort, axis=1, config=config["cohorts"])
    graph_meta["region_main_flag"] = graph_meta["brain_region_standardized"].astype(str).isin(config["region_sensitivity"]["main_regions"])
    graph_meta["region_snpc_flag"] = graph_meta["brain_region_standardized"].astype(str).isin(config["region_sensitivity"]["snpc_enriched_regions"])

    task_rows = []
    for task_name, task_config in config["tasks"].items():
        sender = task_config["sender"]
        receiver = task_config["receiver"]
        sender_counts = graph_meta[sender].astype(int)
        receiver_counts = graph_meta[receiver].astype(int)
        task_frame = graph_meta.copy()
        sender_mins = task_frame["source_id"].astype(str).map(lambda value: resolve_task_minimum(task_config, value, "min_sender_cells")).astype(int)
        receiver_mins = task_frame["source_id"].astype(str).map(lambda value: resolve_task_minimum(task_config, value, "min_receiver_cells")).astype(int)
        task_frame["task_name"] = task_name
        task_frame["sender_cell_type"] = sender
        task_frame["receiver_cell_type"] = receiver
        task_frame["sender_cell_count"] = sender_counts
        task_frame["receiver_cell_count"] = receiver_counts
        task_frame["min_sender_cells_required"] = sender_mins
        task_frame["min_receiver_cells_required"] = receiver_mins
        task_frame["included_task_default"] = (
            sender_counts.ge(sender_mins)
            & receiver_counts.ge(receiver_mins)
        )
        task_rows.append(task_frame)

    graph_manifest = pd.concat(task_rows, ignore_index=True)
    graph_manifest.to_csv(output_dir / "step2_graph_manifest.csv", index=False)

    coverage = (
        graph_manifest.groupby(["task_name", "cohort_name", "source_id", "disease_group"], observed=False)["included_task_default"]
        .sum()
        .rename("n_included_graph_units")
        .reset_index()
        .sort_values(["task_name", "cohort_name", "source_id", "disease_group"])
    )
    coverage.to_csv(output_dir / "step2_sample_coverage_by_task.csv", index=False)

    cell_summary = (
        backbone.assign(cell_type=cell_type)
        .groupby("cell_type", observed=False)
        .agg(
            total_cells=("cell_id", "count"),
            included_primary_default=("included_primary_default", "sum"),
            transferred_cells=("cell_type_label_source", lambda series: int((series.astype(str) == "transferred").sum())),
        )
        .reset_index()
        .sort_values("included_primary_default", ascending=False)
    )

    task_summary = (
        graph_manifest.groupby(["task_name", "cohort_name"], observed=False)
        .agg(
            graph_units=("graph_unit_id", "nunique"),
            included_graph_units=("included_task_default", "sum"),
            median_sender_cells=("sender_cell_count", "median"),
            median_receiver_cells=("receiver_cell_count", "median"),
        )
        .reset_index()
        .sort_values(["task_name", "cohort_name"])
    )

    report = "\n".join(
        [
            "# Step 2 Cohorts",
            "",
            "## Cell Inclusion",
            markdown_table(cell_summary),
            "",
            "## Task Coverage",
            markdown_table(task_summary),
            "",
            "## Rules",
            f"- Default transferred-label confidence threshold: `{default_threshold:.2f}`.",
            "- `control_known_age`, `pd_known_age`, `lbd_bridge`, and `unknown_age_auxiliary` are mutually exclusive graph cohorts.",
            "- The graph manifest is task-specific and contains one row per `graph_unit_id` and task.",
        ]
    )
    write_markdown(output_dir / "step2_cell_inclusion_report.md", report)


if __name__ == "__main__":
    main()
