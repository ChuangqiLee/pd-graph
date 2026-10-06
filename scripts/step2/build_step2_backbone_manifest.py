#!/usr/bin/env python3
"""Freeze the Step 1 atlas backbone for Step 2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import (
    infer_graph_unit_id,
    infer_graph_unit_level,
    infer_split_group_id,
    infer_split_group_level,
    load_yaml,
    markdown_table,
    project_path,
    write_json,
    write_markdown,
)


DEFAULT_CONFIG = Path("configs/step2/step2_backbone.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the Step 1 atlas backbone for Step 2.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_dir = project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    atlas = ad.read_h5ad(project_path(config["paths"]["step1_atlas"]), backed="r")
    obs = atlas.obs.copy()
    obs.insert(0, "obs_index", atlas.obs_names.astype(str))
    if "cell_id" not in obs.columns:
        obs["cell_id"] = obs["obs_index"].astype(str)
    manifest_columns = ["obs_index", *config["manifest_columns"]]
    manifest = obs[manifest_columns].copy()
    manifest["graph_unit_id"] = manifest.apply(infer_graph_unit_id, axis=1)
    manifest["graph_unit_level"] = manifest.apply(infer_graph_unit_level, axis=1)
    manifest["split_group_id"] = manifest.apply(infer_split_group_id, axis=1)
    manifest["split_group_level"] = manifest.apply(infer_split_group_level, axis=1)

    primary = set(config["primary_cell_types"])
    exploratory = set(config["exploratory_cell_types"])
    excluded = set(config["excluded_primary_cell_types"])
    manifest["is_primary_cell_type"] = manifest["cell_type_standardized_final"].astype(str).isin(primary)
    manifest["is_exploratory_cell_type"] = manifest["cell_type_standardized_final"].astype(str).isin(exploratory)
    manifest["is_excluded_primary_cell_type"] = manifest["cell_type_standardized_final"].astype(str).isin(excluded)

    manifest_path = output_dir / "step2_backbone_manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    primary_rows = []
    for cell_type in config["primary_cell_types"]:
        primary_rows.append({"cell_type": cell_type, "priority_tier": "primary", "included_in_primary_pass": True})
    for cell_type in config["exploratory_cell_types"]:
        primary_rows.append({"cell_type": cell_type, "priority_tier": "exploratory", "included_in_primary_pass": False})
    for cell_type in config["excluded_primary_cell_types"]:
        primary_rows.append({"cell_type": cell_type, "priority_tier": "excluded_primary", "included_in_primary_pass": False})
    primary_df = pd.DataFrame(primary_rows).drop_duplicates(subset=["cell_type"]).sort_values(["priority_tier", "cell_type"])
    primary_df.to_csv(output_dir / "step2_primary_cells.csv", index=False)

    graph_meta = (
        manifest.groupby("graph_unit_id", observed=False)
        .agg(
            source_id=("source_id", "first"),
            sample_id=("sample_id", "first"),
            sample_accession=("sample_accession", "first"),
            donor_id=("donor_id", "first"),
            disease_group=("disease_group", "first"),
            age_years=("age_years", "first"),
            age_group=("age_group", "first"),
            sex_standardized=("sex_standardized", "first"),
            brain_region_standardized=("brain_region_standardized", "first"),
            graph_unit_level=("graph_unit_level", "first"),
            split_group_id=("split_group_id", "first"),
            split_group_level=("split_group_level", "first"),
            n_cells=("cell_id", "count"),
        )
        .reset_index()
    )
    counts = pd.crosstab(manifest["graph_unit_id"], manifest["cell_type_standardized_final"]).reset_index()
    graph_units = graph_meta.merge(counts, on="graph_unit_id", how="left").sort_values(["source_id", "graph_unit_id"])
    graph_units.to_csv(output_dir / "step2_graph_units.csv", index=False)

    summary = {
        "atlas_cells": int(manifest.shape[0]),
        "graph_units": int(graph_units.shape[0]),
        "split_groups": int(manifest["split_group_id"].nunique()),
        "primary_cell_types": list(config["primary_cell_types"]),
        "exploratory_cell_types": list(config["exploratory_cell_types"]),
        "excluded_primary_cell_types": list(config["excluded_primary_cell_types"]),
    }
    write_json(output_dir / "step2_backbone_summary.json", summary)

    cell_type_counts = (
        manifest["cell_type_standardized_final"]
        .astype(str)
        .value_counts()
        .rename_axis("cell_type")
        .reset_index(name="n_cells")
    )
    report = "\n".join(
        [
            "# Step 2 Backbone Freeze",
            "",
            "## Summary",
            f"- atlas_cells: {summary['atlas_cells']}",
            f"- graph_units: {summary['graph_units']}",
            f"- split_groups: {summary['split_groups']}",
            "",
            "## Cell Type Counts",
            markdown_table(cell_type_counts),
            "",
            "## Notes",
            "- `graph_unit_id` defaults to `source_id::sample_id` with donor fallback when needed.",
            "- `split_group_id` defaults to `source_id::donor_id` with sample fallback when donor metadata are missing.",
            "- This manifest is the immutable Step 2 cell universe and preserves Step 1 cell order.",
        ]
    )
    write_markdown(output_dir / "step2_backbone_report.md", report)

    if hasattr(atlas, "file") and atlas.file is not None:
        atlas.file.close()


if __name__ == "__main__":
    main()
