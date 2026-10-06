#!/usr/bin/env python3
"""Rescue dopaminergic neurons from transferred generic-neuron labels in Step 2."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import load_yaml, markdown_table, project_path, write_markdown


DEFAULT_CONFIG = Path("configs/step2/step2_dopaminergic_rescue.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply marker-based dopaminergic rescue to Step 2 artifacts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def marker_positive_counts(matrix: sp.spmatrix | np.ndarray) -> np.ndarray:
    dense = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    return np.asarray((dense > 0).sum(axis=1)).reshape(-1).astype(int)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    rescue_config = dict(config["dopaminergic_rescue"])

    backbone_path = project_path(config["paths"]["backbone_manifest"])
    comm_panel_path = project_path(config["paths"]["comm_panel_h5ad"])
    output_dir = project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    backbone = pd.read_parquet(backbone_path)
    comm = ad.read_h5ad(comm_panel_path)
    obs = comm.obs.copy()
    if "cell_id" not in obs.columns:
        obs["cell_id"] = comm.obs_names.astype(str)
    obs["cell_id"] = obs["cell_id"].astype(str)
    for column in ["cell_type_standardized_final", "cell_type_label_source"]:
        if column in obs.columns:
            obs[column] = obs[column].astype(str)
        if column in backbone.columns:
            backbone[column] = backbone[column].astype(str)

    marker_genes = [str(gene) for gene in rescue_config["marker_genes"] if str(gene) in comm.var_names]
    if not marker_genes:
        raise RuntimeError("None of the configured dopaminergic marker genes are present in the Step 2 panel.")
    marker_positions = [int(comm.var_names.get_loc(gene)) for gene in marker_genes]

    target_sources = {str(value) for value in rescue_config["target_sources"]}
    from_labels = {str(value) for value in rescue_config["from_labels"]}
    allowed_label_sources = {str(value) for value in rescue_config.get("allowed_label_sources", [])}
    rescue_label = str(rescue_config["to_label"])
    rescue_label_source = str(rescue_config.get("label_source_name", "marker_rescue"))
    required_marker_count = int(rescue_config.get("required_marker_count", 2))
    confidence_floor = float(rescue_config.get("confidence_floor", 0.70))

    candidate_mask = (
        obs["source_id"].astype(str).isin(target_sources)
        & obs["cell_type_standardized_final"].astype(str).isin(from_labels)
    )
    if allowed_label_sources:
        candidate_mask &= obs["cell_type_label_source"].astype(str).isin(allowed_label_sources)

    candidate_obs = obs.loc[candidate_mask, ["cell_id", "source_id", "sample_id", "cell_type_standardized_final", "cell_type_prediction_confidence"]].copy()
    candidate_obs["marker_gene_count"] = 0
    candidate_obs["rescued"] = False

    if not candidate_obs.empty:
        row_positions = np.flatnonzero(candidate_mask.to_numpy())
        marker_matrix = comm.layers["norm"][row_positions][:, marker_positions]
        candidate_obs["marker_gene_count"] = marker_positive_counts(marker_matrix)
        candidate_obs["rescued"] = candidate_obs["marker_gene_count"].ge(required_marker_count)

        rescued_ids = set(candidate_obs.loc[candidate_obs["rescued"], "cell_id"].astype(str))
        rescue_confidence = np.clip(
            confidence_floor + 0.05 * (candidate_obs["marker_gene_count"].to_numpy(dtype=float) - float(required_marker_count)),
            confidence_floor,
            0.99,
        )
        candidate_obs["rescue_confidence"] = rescue_confidence

        rescued_mask = obs["cell_id"].astype(str).isin(rescued_ids)
        if rescued_mask.any():
            cell_to_conf = dict(zip(candidate_obs["cell_id"].astype(str), candidate_obs["rescue_confidence"].astype(float)))
            obs.loc[rescued_mask, "cell_type_standardized_final"] = rescue_label
            obs.loc[rescued_mask, "cell_type_label_source"] = rescue_label_source
            existing_conf = pd.to_numeric(obs.loc[rescued_mask, "cell_type_prediction_confidence"], errors="coerce").fillna(0.0)
            new_conf = obs.loc[rescued_mask, "cell_id"].astype(str).map(cell_to_conf).astype(float)
            obs.loc[rescued_mask, "cell_type_prediction_confidence"] = np.maximum(existing_conf.to_numpy(dtype=float), new_conf.to_numpy(dtype=float))

            backbone_mask = backbone["cell_id"].astype(str).isin(rescued_ids)
            backbone.loc[backbone_mask, "cell_type_standardized_final"] = rescue_label
            backbone.loc[backbone_mask, "cell_type_label_source"] = rescue_label_source
            backbone_existing_conf = pd.to_numeric(backbone.loc[backbone_mask, "cell_type_prediction_confidence"], errors="coerce").fillna(0.0)
            backbone_new_conf = backbone.loc[backbone_mask, "cell_id"].astype(str).map(cell_to_conf).astype(float)
            backbone.loc[backbone_mask, "cell_type_prediction_confidence"] = np.maximum(
                backbone_existing_conf.to_numpy(dtype=float),
                backbone_new_conf.to_numpy(dtype=float),
            )

    comm.obs = obs
    backbone.to_parquet(backbone_path, index=False)
    temp_comm_panel_path = comm_panel_path.with_suffix(".tmp.h5ad")
    comm.write_h5ad(temp_comm_panel_path, compression="gzip")
    os.replace(temp_comm_panel_path, comm_panel_path)

    rescue_table = (
        candidate_obs.groupby(["source_id", "sample_id"], observed=False)
        .agg(
            candidate_cells=("cell_id", "count"),
            rescued_cells=("rescued", "sum"),
            max_marker_gene_count=("marker_gene_count", "max"),
        )
        .reset_index()
        .sort_values(["rescued_cells", "candidate_cells", "source_id", "sample_id"], ascending=[False, False, True, True])
    )
    rescue_table.to_csv(output_dir / "step2_dopaminergic_rescue_summary.csv", index=False)

    report_lines = [
        "# Step 2 Dopaminergic Rescue",
        "",
        f"- marker_genes_used: `{', '.join(marker_genes)}`",
        f"- required_marker_count: `{required_marker_count}`",
        f"- rescued_cells_total: `{int(candidate_obs['rescued'].sum()) if not candidate_obs.empty else 0}`",
        "",
        "## Per-sample summary",
        markdown_table(rescue_table.head(30)),
    ]
    write_markdown(output_dir / "step2_dopaminergic_rescue_report.md", "\n".join(report_lines))


if __name__ == "__main__":
    main()
