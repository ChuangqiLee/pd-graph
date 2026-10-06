#!/usr/bin/env python3
"""Build the Step 2 ligand-receptor universe with coverage filtering."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import (
    canonical_gene_symbol,
    infer_graph_unit_id,
    load_yaml,
    markdown_table,
    project_path,
    write_markdown,
)


DEFAULT_CONFIG = Path("configs/step2/step2_lr_universe.yaml")
COHORT_CONFIG = Path("configs/step2/step2_cohorts.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step 2 ligand-receptor universe.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--cohort-config", default=str(COHORT_CONFIG))
    return parser.parse_args()


def assign_pathway_family(row: pd.Series, config: dict[str, Any]) -> str:
    text = " ".join(
        [
            str(row.get("pathway_names", "")),
            str(row.get("annotation", "")),
            str(row.get("source_databases", "")),
            str(row.get("original_ligands", "")),
            str(row.get("original_receptors", "")),
        ]
    ).lower()
    annotation = str(row.get("annotation", "")).lower()
    if "ecm" in annotation or "cell-cell contact" in annotation:
        return "extracellular matrix and adhesion"
    for family, keywords in config["family_keyword_rules"].items():
        if any(keyword in text for keyword in keywords):
            return family
    pathway_names = [item for item in str(row.get("pathway_names", "")).split("|") if item]
    if pathway_names:
        return pathway_names[0]
    return "unclassified"


def build_primary_inclusion_mask(obs: pd.DataFrame, threshold: float, primary_cell_types: set[str]) -> pd.Series:
    label_source = obs["cell_type_label_source"].astype(str)
    confidence = pd.to_numeric(obs["cell_type_prediction_confidence"], errors="coerce")
    return (
        obs["pass_qc"].astype(bool)
        & obs["cell_type_standardized_final"].astype(str).isin(primary_cell_types)
        & (
            label_source.ne("transferred")
            | confidence.ge(threshold)
        )
    )


def mean_and_pct(matrix: sp.spmatrix | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if sp.issparse(matrix):
        matrix = matrix.tocsr()
        mean = np.asarray(matrix.mean(axis=0)).ravel().astype(np.float32)
        pct = (matrix.getnnz(axis=0) / max(matrix.shape[0], 1)).astype(np.float32)
        return mean, np.asarray(pct).ravel()
    dense = np.asarray(matrix, dtype=np.float32)
    return dense.mean(axis=0).astype(np.float32), (dense > 0).mean(axis=0).astype(np.float32)


def min_component(values: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    return float(np.min(values[indices]))


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    cohort_config = load_yaml(args.cohort_config)
    output_dir = project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    comm = ad.read_h5ad(project_path(config["paths"]["comm_panel_h5ad"]))
    graph_manifest = pd.read_csv(project_path(config["paths"]["graph_manifest_csv"]))
    resource_dir = project_path("data/metadata/step2_resources")
    cellchat_pairs = pd.read_parquet(resource_dir / "cellchat_lr_pairs.parquet")
    nichenet_pairs = pd.read_parquet(resource_dir / "nichenet_lr_pairs.parquet")

    primary_cell_types = set(cohort_config["primary_cell_types"])
    default_threshold = float(cohort_config["transferred_label_confidence_default"])
    obs = comm.obs.copy()
    obs["graph_unit_id"] = obs.apply(infer_graph_unit_id, axis=1)
    obs["included_primary_default"] = build_primary_inclusion_mask(obs, default_threshold, primary_cell_types)
    obs["obs_position"] = np.arange(comm.n_obs)

    included_obs = obs.loc[obs["included_primary_default"]].copy()
    gene_names = comm.var_names.astype(str).tolist()
    gene_lookup = {gene: idx for idx, gene in enumerate(gene_names)}

    stats_rows: list[dict[str, object]] = []
    group_stats: dict[tuple[str, str], dict[str, Any]] = {}
    norm_matrix = comm.layers["norm"]
    for (graph_unit_id, cell_type), group in included_obs.groupby(["graph_unit_id", "cell_type_standardized_final"], observed=False):
        positions = group["obs_position"].to_numpy(dtype=int)
        matrix = norm_matrix[positions]
        mean, pct = mean_and_pct(matrix)
        group_stats[(str(graph_unit_id), str(cell_type))] = {
            "mean": mean,
            "pct": pct,
            "n_cells": int(len(positions)),
        }
        stats_rows.extend(
            {
                "graph_unit_id": str(graph_unit_id),
                "cell_type": str(cell_type),
                "gene_symbol": gene_names[index],
                "mean_norm": float(mean[index]),
                "pct_expr": float(pct[index]),
                "n_cells": int(len(positions)),
            }
            for index in range(len(gene_names))
        )
    gene_stats = pd.DataFrame(stats_rows)
    gene_stats.to_parquet(output_dir / "step2_graph_unit_celltype_gene_stats.parquet", index=False)

    combined_pairs = pd.concat([cellchat_pairs, nichenet_pairs], ignore_index=True)
    combined_pairs["pair_key"] = combined_pairs["ligand_genes"].astype(str) + "->" + combined_pairs["receptor_genes"].astype(str)
    consensus = (
        combined_pairs.groupby("pair_key", observed=False)
        .agg(
            ligand_genes=("ligand_genes", "first"),
            receptor_genes=("receptor_genes", "first"),
            ligand_multimeric=("ligand_multimeric", "max"),
            receptor_multimeric=("receptor_multimeric", "max"),
            databases=("database", lambda series: "|".join(sorted(set(series.astype(str))))),
            database_support_count=("database", lambda series: int(series.astype(str).nunique())),
            source_databases=("source_database", lambda series: "|".join(sorted(set(series.astype(str))))),
            sources=("source", lambda series: "|".join(sorted(set(series.astype(str))))),
            pathway_names=("pathway_name", lambda series: "|".join(sorted({value for value in series.astype(str) if value}))),
            annotation=("annotation", lambda series: "|".join(sorted({value for value in series.astype(str) if value}))),
            evidence=("evidence", lambda series: "|".join(sorted({value for value in series.astype(str) if value}))),
            original_ligands=("original_ligand", lambda series: "|".join(sorted(set(series.astype(str))))),
            original_receptors=("original_receptor", lambda series: "|".join(sorted(set(series.astype(str))))),
        )
        .reset_index()
    )
    consensus["pathway_family"] = consensus.apply(assign_pathway_family, axis=1, config=config)

    task_rows = graph_manifest.loc[graph_manifest["included_task_default"]].copy()
    min_sender_fraction = float(config["filters"]["min_sender_fraction"])
    min_receiver_fraction = float(config["filters"]["min_receiver_fraction"])
    min_graph_units = int(config["filters"]["min_graph_units"])

    filtered_rows: list[dict[str, Any]] = []
    for row in consensus.itertuples(index=False):
        ligand_components = [canonical_gene_symbol(gene) for gene in str(row.ligand_genes).split("|") if canonical_gene_symbol(gene)]
        receptor_components = [canonical_gene_symbol(gene) for gene in str(row.receptor_genes).split("|") if canonical_gene_symbol(gene)]
        if not ligand_components or not receptor_components:
            continue
        if any(gene not in gene_lookup for gene in ligand_components + receptor_components):
            continue

        ligand_indices = [gene_lookup[gene] for gene in ligand_components]
        receptor_indices = [gene_lookup[gene] for gene in receptor_components]
        task_supports: dict[str, int] = {}
        sender_pct_values: list[float] = []
        receiver_pct_values: list[float] = []
        sender_mean_values: list[float] = []
        receiver_mean_values: list[float] = []
        for task_name, task_frame in task_rows.groupby("task_name", observed=False):
            support_count = 0
            for task_record in task_frame.itertuples(index=False):
                sender_stats = group_stats.get((str(task_record.graph_unit_id), str(task_record.sender_cell_type)))
                receiver_stats = group_stats.get((str(task_record.graph_unit_id), str(task_record.receiver_cell_type)))
                if sender_stats is None or receiver_stats is None:
                    continue
                sender_pct = min_component(sender_stats["pct"], ligand_indices)
                receiver_pct = min_component(receiver_stats["pct"], receptor_indices)
                sender_mean = min_component(sender_stats["mean"], ligand_indices)
                receiver_mean = min_component(receiver_stats["mean"], receptor_indices)
                sender_pct_values.append(sender_pct)
                receiver_pct_values.append(receiver_pct)
                sender_mean_values.append(sender_mean)
                receiver_mean_values.append(receiver_mean)
                if sender_pct >= min_sender_fraction and receiver_pct >= min_receiver_fraction:
                    support_count += 1
            if support_count >= min_graph_units:
                task_supports[str(task_name)] = support_count

        if not task_supports:
            continue

        support_score = float(sum(task_supports.values())) * (
            0.5 * (np.mean(sender_pct_values) if sender_pct_values else 0.0)
            + 0.5 * (np.mean(receiver_pct_values) if receiver_pct_values else 0.0)
        ) * (1.0 + 0.25 * (int(row.database_support_count) - 1))
        filtered_rows.append(
            {
                **row._asdict(),
                "eligible_tasks": "|".join(sorted(task_supports.keys())),
                "graph_units_supported": int(sum(task_supports.values())),
                "task_supports": "|".join(f"{task}:{count}" for task, count in sorted(task_supports.items())),
                "avg_sender_pct_expr": float(np.mean(sender_pct_values)) if sender_pct_values else 0.0,
                "avg_receiver_pct_expr": float(np.mean(receiver_pct_values)) if receiver_pct_values else 0.0,
                "avg_sender_mean_norm": float(np.mean(sender_mean_values)) if sender_mean_values else 0.0,
                "avg_receiver_mean_norm": float(np.mean(receiver_mean_values)) if receiver_mean_values else 0.0,
                "support_score": support_score,
            }
        )

    full_universe = pd.DataFrame(filtered_rows).sort_values(
        ["support_score", "graph_units_supported", "database_support_count"],
        ascending=[False, False, False],
    )
    compact_universe = full_universe.head(512).copy()
    full_universe.to_parquet(output_dir / "step2_lr_universe_full.parquet", index=False)
    compact_universe.to_parquet(output_dir / "step2_lr_universe_compact.parquet", index=False)

    family_counts = (
        full_universe["pathway_family"]
        .value_counts()
        .rename_axis("pathway_family")
        .reset_index(name="n_pairs")
        .head(20)
    )
    report = "\n".join(
        [
            "# Step 2 LR Universe",
            "",
            "## Summary",
            f"- full_universe_pairs: {int(full_universe.shape[0])}",
            f"- compact_universe_pairs: {int(compact_universe.shape[0])}",
            "",
            "## Top Pathway Families",
            markdown_table(family_counts),
            "",
            "## Filtering",
            f"- min_sender_fraction: `{min_sender_fraction:.2f}`",
            f"- min_receiver_fraction: `{min_receiver_fraction:.2f}`",
            f"- min_graph_units: `{min_graph_units}`",
            "",
            "## Notes",
            "- The full universe preserves consensus provenance across CellChat and NicheNet.",
            "- The compact universe keeps the highest-support measurable pairs for tractable graph edge encoding.",
        ]
    )
    write_markdown(output_dir / "step2_lr_qc_report.md", report)


if __name__ == "__main__":
    main()
