#!/usr/bin/env python3
"""Run Step 2 robustness, sensitivity, and external holdout scoring."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import tarfile
from io import TextIOWrapper
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import (
    infer_graph_unit_id,
    load_yaml,
    log,
    markdown_table,
    project_path,
    read_table,
    safe_slug,
    save_pickle_gz,
    write_markdown,
)
from scripts.step2.graph_models import AgeConditionedPDModel, AgeModel, load_graph, tensorize_graph
from scripts.step2.graph_models import build_context_vocabulary, context_cardinalities, tensorize_graph_context
from scripts.step2.training_utils import apply_platt_scaler, apply_source_filters


DEFAULT_INTERP_CONFIG = Path("configs/step2/step2_interpretation.yaml")
DEFAULT_COHORT_CONFIG = Path("configs/step2/step2_cohorts.yaml")
DEFAULT_GRAPH_CONFIG = Path("configs/step2/step2_graphs.yaml")
DEFAULT_AGE_CONFIG = Path("configs/step2/step2_model_age.yaml")
DEFAULT_PD_CONFIG = Path("configs/step2/step2_model_pd.yaml")

HOLDOUT_SOURCE_ID = "GSE157783"
HOLDOUT_CELL_TAR = project_path("data/raw/pd_definitive/geo/GSE157783/GSE157783_IPDCO_hg_midbrain_cell.tar.gz")
HOLDOUT_GENE_TAR = project_path("data/raw/pd_definitive/geo/GSE157783/GSE157783_IPDCO_hg_midbrain_genes.tar.gz")
HOLDOUT_UMI_TAR = project_path("data/raw/pd_definitive/geo/GSE157783/GSE157783_IPDCO_hg_midbrain_UMI.tar.gz")
NIGRA_HOLDOUT_SOURCE_ID = "GSE253462"
NIGRA_HOLDOUT_TAR = project_path("data/raw/pd_definitive/geo/GSE253462/GSE253462_RAW.tar")
NIGRA_HOLDOUT_MANIFEST = project_path("data/metadata/gse253462_nigra_holdout/gse253462_nigra_sample_manifest.csv")
HOLDOUT_SOURCE_WEIGHTS = {
    HOLDOUT_SOURCE_ID: 1.0,
    NIGRA_HOLDOUT_SOURCE_ID: 2.0,
}
MICROGLIA_MARKERS = ["P2RY12", "CX3CR1", "CSF1R", "AIF1", "TREM2", "C1QA", "C1QB", "C1QC", "TMEM119", "HEXB"]
MICROGLIA_CORE_MARKERS = ["P2RY12", "CX3CR1", "CSF1R", "TREM2", "C1QA"]
DA_MARKERS = ["TH", "SLC6A3", "DDC", "ALDH1A1", "GCH1", "DRD2", "SOX6"]
DA_CORE_MARKERS = ["TH", "SLC6A3", "DDC", "ALDH1A1", "GCH1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 2 robustness analyses.")
    parser.add_argument("--interpretation-config", default=str(DEFAULT_INTERP_CONFIG))
    parser.add_argument("--cohort-config", default=str(DEFAULT_COHORT_CONFIG))
    parser.add_argument("--graph-config", default=str(DEFAULT_GRAPH_CONFIG))
    parser.add_argument("--age-config", default=str(DEFAULT_AGE_CONFIG))
    parser.add_argument("--pd-config", default=str(DEFAULT_PD_CONFIG))
    return parser.parse_args()


def disease_effect_ols(frame: pd.DataFrame, value_col: str) -> float:
    local = frame[[value_col, "age_years", "disease_group"]].dropna().copy()
    if local["disease_group"].nunique() < 2 or len(local) < 4:
        return float("nan")
    local["disease_binary"] = local["disease_group"].map({"control": 0.0, "pd": 1.0}).astype(float)
    design = np.column_stack(
        [
            np.ones(len(local), dtype=float),
            local["age_years"].astype(float).to_numpy(),
            local["disease_binary"].to_numpy(),
        ]
    )
    target = local[value_col].astype(float).to_numpy()
    coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return float(coeffs[-1])


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    local = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(local) < 3 or local["x"].nunique() < 2 or local["y"].nunique() < 2:
        return float("nan")
    return float(spearmanr(local["x"], local["y"]).statistic)


def compute_label_confidence_sensitivity(comm_obs: pd.DataFrame, cohort_config: dict[str, Any]) -> pd.DataFrame:
    obs = comm_obs.copy()
    obs["graph_unit_id"] = obs.apply(infer_graph_unit_id, axis=1)
    obs["confidence"] = pd.to_numeric(obs["cell_type_prediction_confidence"], errors="coerce")
    obs["label_source"] = obs["cell_type_label_source"].astype(str)
    thresholds = list(cohort_config["transferred_label_confidence_sensitivity"])
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        included = (
            obs["pass_qc"].astype(bool)
            & obs["cell_type_standardized_final"].astype(str).isin(set(cohort_config["primary_cell_types"]))
            & (obs["label_source"].ne("transferred") | obs["confidence"].ge(float(threshold)))
        )
        filtered = obs.loc[included].copy()
        counts = (
            filtered.groupby(["graph_unit_id", "cell_type_standardized_final"], observed=False)
            .size()
            .rename("n_cells")
            .reset_index()
        )
        meta = (
            filtered.groupby("graph_unit_id", observed=False)
            .agg(
                disease_group=("disease_group", "first"),
                age_years=("age_years", "first"),
            )
            .reset_index()
        )
        wide = counts.pivot_table(
            index="graph_unit_id",
            columns="cell_type_standardized_final",
            values="n_cells",
            fill_value=0,
        ).reset_index()
        graph_table = meta.merge(wide, on="graph_unit_id", how="left")
        count_columns = [column for column in wide.columns if column != "graph_unit_id"]
        graph_table[count_columns] = graph_table[count_columns].fillna(0)
        for task_name, task in cohort_config["tasks"].items():
            sender = task["sender"]
            receiver = task["receiver"]
            sender_counts = pd.to_numeric(graph_table.get(sender, 0), errors="coerce").fillna(0)
            receiver_counts = pd.to_numeric(graph_table.get(receiver, 0), errors="coerce").fillna(0)
            eligible = sender_counts.ge(int(task["min_sender_cells"])) & receiver_counts.ge(int(task["min_receiver_cells"]))
            eligible_frame = graph_table.loc[eligible].copy()
            def cohort_name_for_row(row: pd.Series) -> str:
                age = pd.to_numeric(row["age_years"], errors="coerce")
                disease = str(row["disease_group"])
                if disease == "control" and not pd.isna(age):
                    return "control_known_age"
                if disease == "pd" and not pd.isna(age):
                    return "pd_known_age"
                if disease == "lbd":
                    return "lbd_bridge"
                return "unknown_age_auxiliary"
            if not eligible_frame.empty:
                eligible_frame["cohort_name"] = eligible_frame.apply(cohort_name_for_row, axis=1)
                summary = eligible_frame.groupby("cohort_name", observed=False)["graph_unit_id"].nunique()
                for cohort_name, n_units in summary.items():
                    rows.append(
                        {
                            "confidence_threshold": float(threshold),
                            "task_name": task_name,
                            "cohort_name": str(cohort_name),
                            "n_graph_units": int(n_units),
                        }
                    )
    return pd.DataFrame(rows).sort_values(["task_name", "confidence_threshold", "cohort_name"])


def compute_source_task_coverage(graph_manifest: pd.DataFrame) -> pd.DataFrame:
    return (
        graph_manifest[["task_name", "source_id", "cohort_name", "split_group_id"]]
        .drop_duplicates()
        .groupby(["task_name", "source_id", "cohort_name"], observed=False)["split_group_id"]
        .nunique()
        .rename("n_donors")
        .reset_index()
        .sort_values(["task_name", "source_id", "cohort_name"])
    )


def compute_receiver_specificity(pathway_scores: pd.DataFrame) -> pd.DataFrame:
    disease_frame = pathway_scores.loc[
        pathway_scores["cohort_name"].isin(["control_known_age", "pd_known_age"]) & pathway_scores["age_years"].notna()
    ].copy()
    rows: list[dict[str, Any]] = []
    for (task_name, pathway_family), frame in disease_frame.groupby(["task_name", "pathway_family"], observed=False):
        rows.append(
            {
                "task_name": str(task_name),
                "pathway_family": str(pathway_family),
                "pd_age_adjusted_effect": disease_effect_ols(frame, "pathway_score"),
            }
        )
    effect_df = pd.DataFrame(rows)
    mg_da = effect_df.loc[effect_df["task_name"].eq("microglia_to_dopaminergic_neuron")].rename(
        columns={"pd_age_adjusted_effect": "mg_da_effect"}
    )[["pathway_family", "mg_da_effect"]]
    mg_neuron = effect_df.loc[effect_df["task_name"].eq("microglia_to_neuron")].rename(
        columns={"pd_age_adjusted_effect": "mg_neuron_effect"}
    )[["pathway_family", "mg_neuron_effect"]]
    merged = mg_da.merge(mg_neuron, on="pathway_family", how="outer")
    merged["receiver_specificity_ratio"] = np.abs(merged["mg_da_effect"]) / (np.abs(merged["mg_neuron_effect"]) + 1e-6)
    return merged.sort_values("receiver_specificity_ratio", ascending=False)


def compute_cell_count_robustness(graph_manifest: pd.DataFrame, age_predictions: pd.DataFrame, pd_predictions: pd.DataFrame) -> pd.DataFrame:
    age_graphs = age_predictions.loc[age_predictions["split"].eq("val")].merge(
        graph_manifest[["graph_id", "n_nodes", "n_edges"]], on="graph_id", how="left"
    )
    age_grouped = age_graphs.groupby("split_group_id", observed=False).agg(
        pred_age_years=("pred_age_years", "mean"),
        n_nodes=("n_nodes", "mean"),
        n_edges=("n_edges", "mean"),
    )
    pd_graphs = pd_predictions.loc[pd_predictions["split"].eq("val")].merge(
        graph_manifest[["graph_id", "n_nodes", "n_edges"]], on="graph_id", how="left"
    )
    pd_grouped = pd_graphs.groupby("split_group_id", observed=False).agg(
        pd_probability=("pd_probability", "mean"),
        n_nodes=("n_nodes", "mean"),
        n_edges=("n_edges", "mean"),
    )
    rows = [
        {
            "analysis": "age_model",
            "metric": "pred_age_vs_n_nodes",
            "spearman_rho": safe_spearman(age_grouped["pred_age_years"], age_grouped["n_nodes"]),
        },
        {
            "analysis": "age_model",
            "metric": "pred_age_vs_n_edges",
            "spearman_rho": safe_spearman(age_grouped["pred_age_years"], age_grouped["n_edges"]),
        },
        {
            "analysis": "pd_model",
            "metric": "pd_prob_vs_n_nodes",
            "spearman_rho": safe_spearman(pd_grouped["pd_probability"], pd_grouped["n_nodes"]),
        },
        {
            "analysis": "pd_model",
            "metric": "pd_prob_vs_n_edges",
            "spearman_rho": safe_spearman(pd_grouped["pd_probability"], pd_grouped["n_edges"]),
        },
    ]
    return pd.DataFrame(rows)


def compute_lbd_bridge_summary(pd_predictions: pd.DataFrame, bridge_scores: pd.DataFrame) -> pd.DataFrame:
    val_donors = (
        pd_predictions.loc[pd_predictions["split"].eq("val")]
        .groupby("split_group_id", observed=False)
        .agg(
            disease_target=("disease_target", "first"),
            pd_probability=("pd_probability", "mean"),
        )
        .reset_index()
    )
    val_donors["cohort_name"] = np.where(val_donors["disease_target"].eq(1), "pd_known_age", "control_known_age")
    bridge = (
        bridge_scores.groupby("split_group_id", observed=False)["pd_probability"]
        .mean()
        .rename("pd_probability")
        .reset_index()
    )
    bridge["cohort_name"] = "lbd_bridge"
    combined = pd.concat([val_donors[["split_group_id", "cohort_name", "pd_probability"]], bridge], ignore_index=True)
    summary = (
        combined.groupby("cohort_name", observed=False)["pd_probability"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )
    summary.columns = ["cohort_name", "n_donors", "mean_pd_probability", "median_pd_probability", "min_pd_probability", "max_pd_probability"]
    return summary


def load_holdout_cells() -> pd.DataFrame:
    with tarfile.open(HOLDOUT_CELL_TAR, "r:gz") as tar:
        handle = tar.extractfile("IPDCO_hg_midbrain_cell.tsv")
        if handle is None:
            raise FileNotFoundError("Missing holdout cell metadata file")
        cells = pd.read_csv(TextIOWrapper(handle, encoding="utf-8"), sep="\t")
    mapping = {
        "Microglia": "microglia",
        "DaNs": "dopaminergic neuron",
        "Astrocytes": "astrocyte",
        "Endothelial cells": "endothelial cell",
        "Excitatory": "neuron",
        "Inhibitory": "neuron",
        "GABA": "neuron",
        "CADPS2+ neurons": "neuron",
        "OPCs": "oligodendrocyte precursor cell",
        "Oligodendrocytes": "oligodendrocyte",
    }
    cells["cell_type_standardized_final"] = cells["cell_ontology"].map(mapping).fillna("other")
    cells["disease_group"] = np.where(cells["patient"].astype(str).str.startswith("PD"), "pd", "control")
    cells["graph_unit_id"] = cells["patient"].map(lambda patient: f"{HOLDOUT_SOURCE_ID}::{patient}")
    cells["split_group_id"] = cells["graph_unit_id"]
    cells["source_id"] = HOLDOUT_SOURCE_ID
    cells["sample_id"] = cells["patient"].astype(str)
    cells["donor_id"] = cells["patient"].astype(str)
    cells["age_years"] = np.nan
    cells["brain_region_standardized"] = "midbrain"
    cells["sex_standardized"] = "unknown"
    return cells


def load_cached_holdout_outputs(holdout_dir: Path) -> dict[str, Any] | None:
    cache_paths = {
        "patient_counts": holdout_dir / "gse157783_patient_celltype_counts.csv",
        "lr_scores": holdout_dir / "gse157783_mgda_lr_scores.parquet",
        "pathway_scores": holdout_dir / "gse157783_mgda_pathway_scores.parquet",
        "gnn_scores": holdout_dir / "gse157783_mgda_gnn_scores.csv",
        "baseline_scores": holdout_dir / "gse157783_mgda_baseline_scores.csv",
    }
    if not all(path.exists() for path in cache_paths.values()):
        return None
    return {
        "patient_counts": pd.read_csv(cache_paths["patient_counts"]),
        "lr_scores": read_table(cache_paths["lr_scores"]),
        "pathway_scores": read_table(cache_paths["pathway_scores"]),
        "gnn_scores": pd.read_csv(cache_paths["gnn_scores"]),
        "baseline_scores": pd.read_csv(cache_paths["baseline_scores"]),
    }


def load_cached_nigra_holdout_outputs(holdout_dir: Path) -> dict[str, Any] | None:
    cache_paths = {
        "patient_counts": holdout_dir / "gse253462_patient_celltype_counts.csv",
        "lr_scores": holdout_dir / "gse253462_mgda_lr_scores.parquet",
        "pathway_scores": holdout_dir / "gse253462_mgda_pathway_scores.parquet",
        "assignment_summary": holdout_dir / "gse253462_assignment_summary.csv",
    }
    if not all(path.exists() for path in cache_paths.values()):
        return None
    patient_counts = pd.read_csv(cache_paths["patient_counts"])
    assignment_summary = pd.read_csv(cache_paths["assignment_summary"])
    if patient_counts["eligible_mg_da"].fillna(False).sum() == 0 or assignment_summary["n_qc_pass"].fillna(0).sum() == 0:
        return None
    return {
        "patient_counts": patient_counts,
        "lr_scores": read_table(cache_paths["lr_scores"]),
        "pathway_scores": read_table(cache_paths["pathway_scores"]),
        "gnn_scores": pd.DataFrame(),
        "baseline_scores": pd.DataFrame(),
        "assignment_summary": assignment_summary,
    }


def load_gse253462_manifest() -> pd.DataFrame:
    if not NIGRA_HOLDOUT_MANIFEST.exists():
        raise FileNotFoundError(f"Missing GSE253462 manifest at {NIGRA_HOLDOUT_MANIFEST}")
    manifest = pd.read_csv(NIGRA_HOLDOUT_MANIFEST)
    manifest = manifest.loc[manifest["download_complete"].fillna(False)].copy()
    manifest["source_id"] = NIGRA_HOLDOUT_SOURCE_ID
    manifest["sample_id"] = manifest["gsm_accession"].astype(str)
    manifest["disease_group"] = manifest["condition_standardized"].astype(str).str.lower()
    manifest["graph_unit_id"] = manifest["donor_id"].astype(str).map(lambda donor_id: f"{NIGRA_HOLDOUT_SOURCE_ID}::{donor_id}")
    manifest["split_group_id"] = manifest["graph_unit_id"]
    manifest["brain_region_standardized"] = "substantia_nigra"
    return manifest.sort_values("gsm_accession").reset_index(drop=True)


def open_nigra_holdout_member(tar_handle: tarfile.TarFile, member_name: str) -> gzip.GzipFile:
    extracted = tar_handle.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(f"Missing member {member_name} in {NIGRA_HOLDOUT_TAR}")
    return gzip.open(extracted, "rt", encoding="utf-8", newline="")


def collect_nigra_marker_pass(
    tar_handle: tarfile.TarFile,
    member_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray | None]]:
    marker_genes = sorted(set(MICROGLIA_MARKERS + DA_MARKERS))
    marker_values: dict[str, np.ndarray | None] = {gene: None for gene in marker_genes}
    with open_nigra_holdout_member(tar_handle, member_name) as handle:
        reader = csv.reader(handle, delimiter=" ", quotechar='"', skipinitialspace=True)
        header = np.asarray([token for token in next(reader) if token], dtype=object)
        total_counts = np.zeros((len(header),), dtype=np.float64)
        genes_detected = np.zeros((len(header),), dtype=np.int32)
        mt_counts = np.zeros((len(header),), dtype=np.float64)
        for row in reader:
            if not row:
                continue
            gene_name = str(row[0]).upper()
            values = np.asarray(row[1:], dtype=np.float32)
            if values.size != len(header):
                continue
            total_counts += values
            genes_detected += values > 0
            if gene_name.startswith("MT-"):
                mt_counts += values
            if gene_name in marker_values:
                marker_values[gene_name] = values
    return header, total_counts, genes_detected, mt_counts, marker_values


def average_log_module_score(
    lognorm_values: np.ndarray,
    gene_names: list[str],
    gene_order: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    present = [gene for gene in gene_names if gene in gene_order]
    if not present:
        zeros = np.zeros((lognorm_values.shape[0],), dtype=np.float32)
        return zeros, zeros.astype(np.int32), zeros
    positions = [gene_order.index(gene) for gene in present]
    subset = lognorm_values[:, positions]
    return (
        subset.mean(axis=1).astype(np.float32),
        (subset > 0).sum(axis=1).astype(np.int32),
        subset.max(axis=1).astype(np.float32),
    )


def classify_nigra_holdout_cells(
    barcodes: np.ndarray,
    total_counts: np.ndarray,
    genes_detected: np.ndarray,
    mt_counts: np.ndarray,
    marker_values: dict[str, np.ndarray | None],
    metadata_row: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    marker_gene_order = [gene for gene, values in marker_values.items() if values is not None]
    if not marker_gene_order:
        empty = pd.DataFrame(
            columns=[
                "barcode",
                "cell_type_standardized_final",
                "cell_type_prediction_confidence",
                "graph_unit_id",
                "split_group_id",
                "source_id",
                "sample_id",
                "donor_id",
                "disease_group",
                "age_years",
                "brain_region_standardized",
                "sex_standardized",
                "qc_total_counts",
                "qc_n_genes_by_counts",
                "qc_pct_mt",
            ]
        )
        return empty, {"n_total_barcodes": int(len(barcodes)), "n_qc_pass": 0, "n_microglia": 0, "n_dopaminergic_neuron": 0}

    marker_matrix = np.column_stack([marker_values[gene] for gene in marker_gene_order]).astype(np.float32)
    safe_total = np.maximum(total_counts, 1.0)
    lognorm = np.log1p(marker_matrix / safe_total[:, None] * 1e4)

    positive_totals = total_counts[total_counts > 0]
    min_total = max(500.0, float(np.quantile(positive_totals, 0.10)) if positive_totals.size else 500.0)
    positive_detected = genes_detected[genes_detected > 0]
    min_genes = max(200, int(np.quantile(positive_detected, 0.10)) if positive_detected.size else 200)
    qc_mask = (total_counts >= min_total) & (genes_detected >= min_genes)

    mg_score, mg_detect, mg_core = average_log_module_score(lognorm, MICROGLIA_MARKERS, marker_gene_order)
    da_score, da_detect, da_core = average_log_module_score(lognorm, DA_MARKERS, marker_gene_order)
    mg_margin = mg_score - da_score
    da_margin = da_score - mg_score

    mg_positive = (mg_detect >= 2) & (mg_core >= 0.9) & (mg_margin >= 0.15)
    da_positive = (da_detect >= 1) & (da_core >= 1.0) & (da_margin >= 0.15)

    mg_reference = mg_score[qc_mask & mg_positive]
    da_reference = da_score[qc_mask & da_positive]
    mg_threshold = max(0.45, float(np.quantile(mg_reference, 0.75)) if mg_reference.size else 0.45)
    da_threshold = max(0.50, float(np.quantile(da_reference, 0.70)) if da_reference.size else 0.50)

    mg_mask = qc_mask & mg_positive & (mg_score >= mg_threshold)
    da_mask = qc_mask & da_positive & (da_score >= da_threshold)
    overlap = mg_mask & da_mask
    mg_mask = mg_mask & ~overlap
    da_mask = da_mask & ~overlap
    mg_mask = mg_mask | (overlap & (mg_margin >= da_margin))
    da_mask = da_mask | (overlap & (da_margin > mg_margin))

    confidence = np.zeros((len(barcodes),), dtype=np.float32)
    confidence[mg_mask] = np.clip(0.45 + 0.25 * mg_margin[mg_mask] + 0.05 * mg_detect[mg_mask], 0.0, 1.0)
    confidence[da_mask] = np.clip(0.45 + 0.30 * da_margin[da_mask] + 0.08 * da_detect[da_mask], 0.0, 1.0)

    cell_types = np.full((len(barcodes),), "other", dtype=object)
    cell_types[mg_mask] = "microglia"
    cell_types[da_mask] = "dopaminergic neuron"
    selected_mask = mg_mask | da_mask

    selected = pd.DataFrame(
        {
            "barcode": barcodes[selected_mask].astype(str),
            "cell_type_standardized_final": cell_types[selected_mask],
            "cell_type_prediction_confidence": confidence[selected_mask].astype(float),
            "graph_unit_id": str(metadata_row["graph_unit_id"]),
            "split_group_id": str(metadata_row["split_group_id"]),
            "source_id": NIGRA_HOLDOUT_SOURCE_ID,
            "sample_id": str(metadata_row["sample_id"]),
            "donor_id": str(metadata_row["donor_id"]),
            "disease_group": str(metadata_row["disease_group"]),
            "age_years": pd.to_numeric(metadata_row.get("age_years"), errors="coerce"),
            "brain_region_standardized": str(metadata_row["brain_region_standardized"]),
            "sex_standardized": str(metadata_row.get("sex_standardized", "unknown")),
            "qc_total_counts": total_counts[selected_mask].astype(float),
            "qc_n_genes_by_counts": genes_detected[selected_mask].astype(int),
            "qc_pct_mt": np.where(total_counts[selected_mask] > 0, mt_counts[selected_mask] / total_counts[selected_mask] * 100.0, 0.0),
        }
    )
    summary = {
        "n_total_barcodes": int(len(barcodes)),
        "n_qc_pass": int(qc_mask.sum()),
        "n_microglia": int(mg_mask.sum()),
        "n_dopaminergic_neuron": int(da_mask.sum()),
        "microglia_threshold": float(mg_threshold),
        "dopaminergic_threshold": float(da_threshold),
        "min_total_counts_threshold": float(min_total),
        "min_genes_threshold": int(min_genes),
    }
    return selected, summary


def collect_nigra_selected_gene_matrix(
    tar_handle: tarfile.TarFile,
    member_name: str,
    selected_indices: np.ndarray,
    gene_set: set[str],
) -> tuple[np.ndarray, list[str]]:
    gene_names: list[str] = []
    selected_rows: list[np.ndarray] = []
    with open_nigra_holdout_member(tar_handle, member_name) as handle:
        reader = csv.reader(handle, delimiter=" ", quotechar='"', skipinitialspace=True)
        _ = next(reader)
        for row in reader:
            if not row:
                continue
            gene_name = str(row[0]).upper()
            if gene_name not in gene_set:
                continue
            values = np.asarray(row[1:], dtype=np.float32)
            if values.size == 0:
                continue
            gene_names.append(gene_name)
            selected_rows.append(values[selected_indices])
    if not selected_rows:
        return np.zeros((len(selected_indices), 0), dtype=np.float32), []
    matrix = np.vstack(selected_rows).T.astype(np.float32)
    return matrix, gene_names


def ensembl_lookup(ids: list[str], cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    session = requests.Session()
    rows: list[dict[str, str]] = []
    batch_size = 400
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        payload = {"ids": batch}
        response = session.post(
            "https://rest.ensembl.org/lookup/id",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        for gene_id in batch:
            record = data.get(gene_id) or {}
            rows.append(
                {
                    "gene_id": gene_id,
                    "gene_symbol": str(record.get("display_name") or ""),
                }
            )
    mapping = pd.DataFrame(rows)
    mapping.to_csv(cache_path, index=False)
    return mapping


def component_geometric_expression(matrix: np.ndarray, component_positions: list[int]) -> np.ndarray:
    if not component_positions:
        return np.zeros((matrix.shape[0],), dtype=np.float32)
    values = matrix[:, component_positions]
    zero_mask = (values <= 0).any(axis=1)
    clipped = np.clip(values, 1e-12, None)
    result = np.exp(np.mean(np.log(clipped), axis=1))
    result[zero_mask] = 0.0
    return result.astype(np.float32)


def reconstruct_age_cv(graph_manifest: pd.DataFrame, age_config: dict[str, Any]) -> list[dict[str, Any]]:
    metadata_path = project_path(age_config["paths"]["output_dir"]) / "age_model_fold_metadata.csv"
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        return metadata.to_dict("records")
    filtered = graph_manifest.loc[
        graph_manifest["task_name"].eq(age_config["task_name"])
        & graph_manifest["cohort_name"].eq(age_config["cohort_name"])
        & graph_manifest["age_years"].notna()
    ].copy()
    data_filters = dict(age_config.get("data_filters", {}))
    filtered = apply_source_filters(
        filtered,
        include_sources=data_filters.get("include_source_ids"),
        exclude_sources=data_filters.get("exclude_source_ids"),
    )
    groups = filtered["split_group_id"].astype(str)
    splitter = GroupKFold(n_splits=min(int(age_config["cv"]["n_splits"]), groups.nunique()))
    folds = []
    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(filtered, groups=groups), start=1):
        train_df = filtered.iloc[train_idx]
        folds.append(
            {
                "fold": fold_index,
                "age_mean": float(train_df["age_years"].mean()),
                "age_std": float(train_df["age_years"].std(ddof=0) or 1.0),
            }
        )
    return folds


def reconstruct_pd_cv(graph_manifest: pd.DataFrame, pd_config: dict[str, Any]) -> list[dict[str, Any]]:
    metadata_path = project_path(pd_config["paths"]["output_dir"]) / "pd_model_fold_metadata.csv"
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        return metadata.to_dict("records")
    filtered = graph_manifest.loc[
        graph_manifest["task_name"].eq(pd_config["task_name"])
        & graph_manifest["cohort_name"].isin(pd_config["cohort_names"])
        & graph_manifest["age_years"].notna()
    ].copy()
    data_filters = dict(pd_config.get("data_filters", {}))
    filtered = apply_source_filters(
        filtered,
        include_sources=data_filters.get("include_source_ids"),
        exclude_sources=data_filters.get("exclude_source_ids"),
    )
    groups = filtered["split_group_id"].astype(str)
    splitter = GroupKFold(n_splits=min(int(pd_config["cv"]["n_splits"]), groups.nunique()))
    folds = []
    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(filtered, groups=groups), start=1):
        train_df = filtered.iloc[train_idx]
        folds.append(
            {
                "fold": fold_index,
                "age_mean": float(train_df["age_years"].mean()),
                "age_std": float(train_df["age_years"].std(ddof=0) or 1.0),
            }
        )
    return folds


def age_training_manifest(graph_manifest: pd.DataFrame, age_config: dict[str, Any]) -> pd.DataFrame:
    filtered = graph_manifest.loc[
        graph_manifest["task_name"].eq(age_config["task_name"])
        & graph_manifest["cohort_name"].eq(age_config["cohort_name"])
        & graph_manifest["age_years"].notna()
    ].copy()
    data_filters = dict(age_config.get("data_filters", {}))
    return apply_source_filters(
        filtered,
        include_sources=data_filters.get("include_source_ids"),
        exclude_sources=data_filters.get("exclude_source_ids"),
    )


def pd_training_manifest_for_vocab(graph_manifest: pd.DataFrame, pd_config: dict[str, Any]) -> pd.DataFrame:
    allowed_disease_groups = [str(value) for value in pd_config.get("disease_groups", ["control", "pd"])]
    include_age_missing = bool(pd_config.get("include_age_missing", False))
    data_filters = dict(pd_config.get("data_filters", {}))
    train_manifest = graph_manifest.loc[
        graph_manifest["task_name"].eq(pd_config["task_name"])
        & graph_manifest["cohort_name"].isin(pd_config["cohort_names"])
        & graph_manifest["disease_group"].isin(allowed_disease_groups)
        & (graph_manifest["age_years"].notna() | include_age_missing)
    ].copy()
    train_manifest = apply_source_filters(
        train_manifest,
        include_sources=data_filters.get("include_source_ids"),
        exclude_sources=data_filters.get("exclude_source_ids"),
    )
    bridge_manifest = graph_manifest.loc[
        graph_manifest["task_name"].eq(pd_config["task_name"])
        & graph_manifest["cohort_name"].eq(pd_config["bridge_cohort_name"])
    ].copy()
    if bridge_manifest.empty:
        return train_manifest
    return pd.concat([train_manifest, bridge_manifest], ignore_index=True)


def build_gse157783_holdout(
    output_dir: Path,
    cohort_config: dict[str, Any],
    graph_config: dict[str, Any],
    age_config: dict[str, Any],
    pd_config: dict[str, Any],
    graph_manifest: pd.DataFrame,
    compact_universe: pd.DataFrame,
    lr_universe_full: pd.DataFrame,
    comm_genes: list[str],
) -> dict[str, Any]:
    log("Preparing GSE157783 external holdout...")
    holdout_dir = output_dir / "gse157783_holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)

    cached_holdout = load_cached_holdout_outputs(holdout_dir)
    try:
        cells = load_holdout_cells()
    except FileNotFoundError:
        if cached_holdout is not None:
            log("Holdout raw files were unavailable; reusing cached GSE157783 outputs.")
            return cached_holdout
        raise
    mgda_task = cohort_config["tasks"]["microglia_to_dopaminergic_neuron"]
    patient_counts = (
        cells.loc[cells["cell_type_standardized_final"].isin(["microglia", "dopaminergic neuron"])]
        .groupby(["patient", "cell_type_standardized_final"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if "microglia" not in patient_counts.columns:
        patient_counts["microglia"] = 0
    if "dopaminergic neuron" not in patient_counts.columns:
        patient_counts["dopaminergic neuron"] = 0
    patient_counts["eligible_mg_da"] = (
        patient_counts["microglia"].ge(int(mgda_task["min_sender_cells"]))
        & patient_counts["dopaminergic neuron"].ge(int(mgda_task["min_receiver_cells"]))
    )
    patient_counts.to_csv(holdout_dir / "gse157783_patient_celltype_counts.csv", index=False)
    eligible_patients = patient_counts.loc[patient_counts["eligible_mg_da"], "patient"].astype(str).tolist()
    if not eligible_patients:
        return {
            "patient_counts": patient_counts,
            "lr_scores": pd.DataFrame(),
            "pathway_scores": pd.DataFrame(),
            "gnn_scores": pd.DataFrame(),
        }

    selected_cells = cells.loc[
        cells["patient"].astype(str).isin(eligible_patients)
        & cells["cell_type_standardized_final"].isin(["microglia", "dopaminergic neuron"])
    ].copy()
    selected_barcodes = selected_cells["barcode"].astype(str).tolist()
    selected_barcode_set = set(selected_barcodes)

    with tarfile.open(HOLDOUT_GENE_TAR, "r:gz") as tar:
        handle = tar.extractfile("IPDCO_hg_midbrain_genes.tsv")
        if handle is None:
            raise FileNotFoundError("Missing holdout gene file")
        genes = pd.read_csv(TextIOWrapper(handle, encoding="utf-8"), sep="\t")
    genes["gene_id"] = genes["gene"].astype(str).str.replace(r"\.\d+$", "", regex=True)
    mapping_cache = holdout_dir / "gse157783_ensembl_to_symbol.csv"
    mapping = ensembl_lookup(genes["gene_id"].astype(str).tolist(), mapping_cache)
    genes = genes.merge(mapping, on="gene_id", how="left")
    genes["gene_symbol"] = genes["gene_symbol"].fillna("").astype(str).str.upper()
    genes["row"] = pd.to_numeric(genes["row"], errors="coerce").astype(int)

    comm_gene_set = {gene.upper() for gene in comm_genes}
    selected_gene_rows = genes.loc[genes["gene_symbol"].isin(comm_gene_set) & genes["gene_symbol"].ne("")].copy()
    symbol_order = sorted(selected_gene_rows["gene_symbol"].unique().tolist())
    symbol_to_col = {symbol: idx for idx, symbol in enumerate(symbol_order)}
    row_to_col: dict[int, int] = {}
    mt_rows: set[int] = set()
    for row in selected_gene_rows.itertuples(index=False):
        row_to_col[int(row.row)] = symbol_to_col[str(row.gene_symbol)]
        if str(row.gene_symbol).startswith("MT-"):
            mt_rows.add(int(row.row))

    with tarfile.open(HOLDOUT_UMI_TAR, "r:gz") as tar:
        handle = tar.extractfile("IPDCO_hg_midbrain_UMI.tsv")
        if handle is None:
            raise FileNotFoundError("Missing holdout UMI file")
        text_handle = TextIOWrapper(handle, encoding="utf-8")
        header = text_handle.readline().rstrip("\n")
        all_barcodes = header.split("\t")
        selected_positions = [idx for idx, barcode in enumerate(all_barcodes) if barcode in selected_barcode_set]
        selected_order = [all_barcodes[idx] for idx in selected_positions]
        selected_cells = selected_cells.set_index("barcode").loc[selected_order].reset_index()
        n_cells = len(selected_positions)
        matrix = np.zeros((n_cells, len(symbol_order)), dtype=np.float32)
        total_counts = np.zeros((n_cells,), dtype=np.float64)
        genes_detected = np.zeros((n_cells,), dtype=np.int32)
        mt_counts = np.zeros((n_cells,), dtype=np.float64)
        for row_index, line in enumerate(text_handle, start=1):
            values = np.fromstring(line.rstrip("\n"), sep="\t", dtype=np.float32)
            if values.size == 0:
                continue
            chosen = values[selected_positions]
            total_counts += chosen
            genes_detected += (chosen > 0).astype(np.int32)
            if row_index in mt_rows:
                mt_counts += chosen
            if row_index in row_to_col:
                matrix[:, row_to_col[row_index]] += chosen

    selected_cells["qc_total_counts"] = total_counts
    selected_cells["qc_n_genes_by_counts"] = genes_detected
    selected_cells["qc_pct_mt"] = np.where(total_counts > 0, mt_counts / total_counts * 100.0, 0.0)
    selected_cells["cell_type_prediction_confidence"] = 1.0
    counts_sparse = sp.csr_matrix(matrix)
    norm = matrix / np.maximum(total_counts[:, None], 1.0) * 1e4
    lognorm = np.log1p(norm)

    holdout_gene_stats_rows: list[dict[str, Any]] = []
    for (graph_unit_id, cell_type), frame in selected_cells.groupby(["graph_unit_id", "cell_type_standardized_final"], observed=False):
        positions = frame.index.to_numpy(dtype=int)
        sub = norm[positions]
        pct_expr = (sub > 0).mean(axis=0)
        mean_norm = sub.mean(axis=0)
        for gene_idx, gene_symbol in enumerate(symbol_order):
            holdout_gene_stats_rows.append(
                {
                    "graph_unit_id": str(graph_unit_id),
                    "cell_type": str(cell_type),
                    "gene_symbol": str(gene_symbol),
                    "mean_norm": float(mean_norm[gene_idx]),
                    "pct_expr": float(pct_expr[gene_idx]),
                    "n_cells": int(len(positions)),
                }
            )
    holdout_gene_stats = pd.DataFrame(holdout_gene_stats_rows)
    holdout_gene_stats.to_parquet(holdout_dir / "gse157783_mgda_gene_stats.parquet", index=False)

    task_name = "microglia_to_dopaminergic_neuron"
    lr_rows: list[dict[str, Any]] = []
    stats_index = holdout_gene_stats.set_index(["graph_unit_id", "cell_type", "gene_symbol"]).sort_index()
    task_pairs_full = lr_universe_full.loc[lr_universe_full["eligible_tasks"].astype(str).str.contains(task_name)].copy()
    for patient in eligible_patients:
        graph_unit_id = f"{HOLDOUT_SOURCE_ID}::{patient}"
        disease_group = "pd" if patient.startswith("PD") else "control"
        for pair in task_pairs_full.itertuples(index=False):
            ligand_genes = [gene for gene in str(pair.ligand_genes).split("|") if gene]
            receptor_genes = [gene for gene in str(pair.receptor_genes).split("|") if gene]
            try:
                sender_expr = [float(stats_index.loc[(graph_unit_id, "microglia", gene), "mean_norm"]) for gene in ligand_genes]
                sender_pct = [float(stats_index.loc[(graph_unit_id, "microglia", gene), "pct_expr"]) for gene in ligand_genes]
                receiver_expr = [float(stats_index.loc[(graph_unit_id, "dopaminergic neuron", gene), "mean_norm"]) for gene in receptor_genes]
                receiver_pct = [float(stats_index.loc[(graph_unit_id, "dopaminergic neuron", gene), "pct_expr"]) for gene in receptor_genes]
            except KeyError:
                continue
            sender_value = float(np.prod(np.maximum(sender_expr, 1e-12)) ** (1.0 / len(sender_expr)))
            receiver_value = float(np.prod(np.maximum(receiver_expr, 1e-12)) ** (1.0 / len(receiver_expr)))
            lr_rows.append(
                {
                    "graph_unit_id": graph_unit_id,
                    "split_group_id": graph_unit_id,
                    "source_id": HOLDOUT_SOURCE_ID,
                    "sample_id": patient,
                    "donor_id": patient,
                    "disease_group": disease_group,
                    "task_name": task_name,
                    "pair_key": pair.pair_key,
                    "pathway_family": pair.pathway_family,
                    "score": float(np.sqrt(sender_value * receiver_value)),
                    "sender_mean_norm": sender_value,
                    "receiver_mean_norm": receiver_value,
                    "sender_pct_expr": float(np.min(sender_pct)),
                    "receiver_pct_expr": float(np.min(receiver_pct)),
                }
            )
    holdout_lr_scores = pd.DataFrame(lr_rows)
    holdout_lr_scores.to_parquet(holdout_dir / "gse157783_mgda_lr_scores.parquet", index=False)
    holdout_pathway_scores = (
        holdout_lr_scores.groupby(
            ["graph_unit_id", "split_group_id", "source_id", "sample_id", "donor_id", "disease_group", "task_name", "pathway_family"],
            observed=False,
        )["score"]
        .mean()
        .rename("pathway_score")
        .reset_index()
    )
    holdout_pathway_scores.to_parquet(holdout_dir / "gse157783_mgda_pathway_scores.parquet", index=False)

    train_pathway_scores = pd.read_parquet(project_path("data/processed/step2/04_baselines/step2_pathway_scores.parquet"))
    train_wide = train_pathway_scores.loc[
        train_pathway_scores["task_name"].eq(task_name)
        & train_pathway_scores["cohort_name"].isin(["control_known_age", "pd_known_age"])
        & train_pathway_scores["age_years"].notna()
    ].pivot_table(
        index=["graph_unit_id", "split_group_id", "cohort_name", "source_id", "disease_group"],
        columns="pathway_family",
        values="pathway_score",
        fill_value=0.0,
    ).reset_index()
    feature_columns = [column for column in train_wide.columns if column not in {"graph_unit_id", "split_group_id", "cohort_name", "source_id", "disease_group"}]
    holdout_wide = holdout_pathway_scores.pivot_table(
        index=["graph_unit_id", "split_group_id", "source_id", "sample_id", "donor_id", "disease_group"],
        columns="pathway_family",
        values="pathway_score",
        fill_value=0.0,
    ).reset_index()
    for column in feature_columns:
        if column not in holdout_wide.columns:
            holdout_wide[column] = 0.0
    holdout_wide = holdout_wide[[col for col in holdout_wide.columns if col not in feature_columns] + feature_columns]
    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=13)),
        ]
    )
    train_target = train_wide["disease_group"].map({"control": 0, "pd": 1}).astype(int)
    logistic.fit(train_wide[feature_columns], train_target)
    holdout_wide["baseline_pd_probability"] = logistic.predict_proba(holdout_wide[feature_columns])[:, 1]

    compact_pairs = compact_universe.loc[compact_universe["eligible_tasks"].astype(str).str.contains(task_name)].copy().reset_index(drop=True)
    compact_pairs["compact_pair_index"] = compact_pairs.index.astype(int)
    pathway_names = sorted(compact_pairs["pathway_family"].astype(str).unique().tolist())
    pathway_lookup = {name: idx for idx, name in enumerate(pathway_names)}
    pair_family_codes = np.array([pathway_lookup[name] for name in compact_pairs["pathway_family"].astype(str)], dtype=int)
    family_indicator = np.zeros((len(compact_pairs), len(pathway_names)), dtype=np.float32)
    for pair_position, family_code in enumerate(pair_family_codes):
        family_indicator[pair_position, family_code] = 1.0
    family_denominator = np.maximum(family_indicator.sum(axis=0, keepdims=True), 1.0)
    task_gene_union = sorted(
        {
            gene
            for value in pd.concat([compact_pairs["ligand_genes"], compact_pairs["receptor_genes"]]).astype(str)
            for gene in value.split("|")
            if gene and gene in symbol_to_col
        }
    )
    task_gene_positions = [symbol_to_col[gene] for gene in task_gene_union]
    task_gene_lookup = {gene: idx for idx, gene in enumerate(task_gene_union)}
    task_pair_indices: list[tuple[int, list[int], list[int]]] = []
    for pair in compact_pairs.itertuples(index=False):
        ligand_components = [task_gene_lookup[gene] for gene in str(pair.ligand_genes).split("|") if gene in task_gene_lookup]
        receptor_components = [task_gene_lookup[gene] for gene in str(pair.receptor_genes).split("|") if gene in task_gene_lookup]
        task_pair_indices.append((int(pair.compact_pair_index), ligand_components, receptor_components))

    pca = PCA(n_components=min(48, lognorm.shape[0] - 1, lognorm.shape[1]), random_state=13)
    embedding = pca.fit_transform(lognorm)
    if embedding.shape[1] < 48:
        embedding = np.pad(embedding, ((0, 0), (0, 48 - embedding.shape[1])), mode="constant")
    pseudo_harmony = embedding[:, :24].astype(np.float32)
    pseudo_comm = embedding[:, 24:48].astype(np.float32)
    primary_order = list(graph_config["node_caps"].keys())
    cell_type_lookup = {cell_type: idx for idx, cell_type in enumerate(primary_order)}
    node_features = []
    for row in selected_cells.itertuples(index=False):
        one_hot = np.zeros((len(primary_order),), dtype=np.float32)
        if row.cell_type_standardized_final in cell_type_lookup:
            one_hot[cell_type_lookup[row.cell_type_standardized_final]] = 1.0
        qc_features = np.array(
            [
                np.log1p(float(row.qc_total_counts)),
                np.log1p(float(row.qc_n_genes_by_counts)),
                float(row.qc_pct_mt) / 100.0,
                1.0,
            ],
            dtype=np.float32,
        )
        node_features.append(np.concatenate([pseudo_harmony[len(node_features)], pseudo_comm[len(node_features)], one_hot, qc_features]))
    node_feature_matrix = np.vstack(node_features).astype(np.float32)

    graph_paths: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(graph_config["random_seed"]))
    for patient in eligible_patients:
        patient_cells = selected_cells.loc[selected_cells["patient"].astype(str).eq(patient)].copy()
        sender_positions_all = patient_cells.loc[patient_cells["cell_type_standardized_final"].eq("microglia")].index.to_numpy(dtype=int)
        receiver_positions_all = patient_cells.loc[patient_cells["cell_type_standardized_final"].eq("dopaminergic neuron")].index.to_numpy(dtype=int)
        sender_cap = int(graph_config["node_caps"]["microglia"])
        receiver_cap = int(graph_config["node_caps"]["dopaminergic neuron"])
        repeats = int(graph_config["subgraphs"]["per_graph_unit"]) if (len(sender_positions_all) > sender_cap or len(receiver_positions_all) > receiver_cap) else 1
        for repeat_index in range(repeats):
            sender_positions = sender_positions_all
            receiver_positions = receiver_positions_all
            if len(sender_positions_all) > sender_cap:
                sender_positions = np.sort(rng.choice(sender_positions_all, size=sender_cap, replace=False))
            if len(receiver_positions_all) > receiver_cap:
                receiver_positions = np.sort(rng.choice(receiver_positions_all, size=receiver_cap, replace=False))
            sender_harmony = pseudo_harmony[sender_positions]
            receiver_harmony = pseudo_harmony[receiver_positions]
            neighbors = min(int(graph_config["graph_construction"]["k_neighbors"]), len(receiver_positions))
            knn = NearestNeighbors(n_neighbors=neighbors, metric="euclidean")
            knn.fit(receiver_harmony)
            distances, receiver_neighbor_indices = knn.kneighbors(sender_harmony)
            sender_gene_matrix = norm[sender_positions][:, task_gene_positions]
            receiver_gene_matrix = norm[receiver_positions][:, task_gene_positions]
            sender_pair_expr = np.column_stack(
                [component_geometric_expression(sender_gene_matrix, ligand_components) for _, ligand_components, _ in task_pair_indices]
            ).astype(np.float32)
            receiver_pair_expr = np.column_stack(
                [component_geometric_expression(receiver_gene_matrix, receptor_components) for _, _, receptor_components in task_pair_indices]
            ).astype(np.float32)
            edge_index_rows = []
            edge_attrs = []
            edge_top_pair_indices = []
            edge_top_pair_scores = []
            pair_ids = np.array([pair_index for pair_index, _, _ in task_pair_indices], dtype=np.int32)
            for sender_local_index, receiver_local_choices in enumerate(receiver_neighbor_indices):
                for neighbor_rank, receiver_local_index in enumerate(receiver_local_choices):
                    pair_scores = np.sqrt(
                        np.maximum(sender_pair_expr[sender_local_index], 0.0)
                        * np.maximum(receiver_pair_expr[receiver_local_index], 0.0)
                    )
                    pathway_features = (pair_scores @ family_indicator / family_denominator.ravel()).astype(np.float32)
                    if pathway_features.max(initial=0.0) < float(graph_config["graph_construction"]["min_active_edge_score"]):
                        continue
                    edge_attr = np.concatenate([pathway_features, np.array([distances[sender_local_index, neighbor_rank]], dtype=np.float32)])
                    top_k = min(int(graph_config["graph_construction"]["top_lr_features_per_edge"]), len(pair_scores))
                    top_indices_local = np.argpartition(pair_scores, -top_k)[-top_k:]
                    top_indices_local = top_indices_local[np.argsort(pair_scores[top_indices_local])[::-1]]
                    edge_index_rows.append([sender_local_index, len(sender_positions) + receiver_local_index])
                    edge_attrs.append(edge_attr.astype(np.float32))
                    edge_top_pair_indices.append(pair_ids[top_indices_local])
                    edge_top_pair_scores.append(pair_scores[top_indices_local].astype(np.float32))
            if not edge_index_rows:
                continue
            graph_id = safe_slug(f"gse157783_mgda_{patient}_subgraph_{repeat_index:02d}")
            node_positions = np.concatenate([sender_positions, receiver_positions])
            graph = {
                "graph_id": graph_id,
                "task_name": task_name,
                "node_ids": selected_cells.loc[node_positions, "barcode"].astype(str).to_numpy(),
                "node_positions": node_positions.astype(np.int32),
                "node_features": node_feature_matrix[node_positions],
                "edge_index": np.asarray(edge_index_rows, dtype=np.int64).T,
                "edge_attr": np.asarray(edge_attrs, dtype=np.float32),
                "edge_top_pair_indices": np.asarray(edge_top_pair_indices, dtype=np.int32),
                "edge_top_pair_scores": np.asarray(edge_top_pair_scores, dtype=np.float32),
                "sender_mask": np.array([True] * len(sender_positions) + [False] * len(receiver_positions)),
                "receiver_mask": np.array([False] * len(sender_positions) + [True] * len(receiver_positions)),
                "pathway_names": pathway_names,
                "metadata": {
                    "graph_unit_id": f"{HOLDOUT_SOURCE_ID}::{patient}",
                    "split_group_id": f"{HOLDOUT_SOURCE_ID}::{patient}",
                    "source_id": HOLDOUT_SOURCE_ID,
                    "sample_id": patient,
                    "donor_id": patient,
                    "cohort_name": "external_holdout",
                    "disease_group": "pd" if patient.startswith("PD") else "control",
                    "age_years": None,
                    "age_missing_mask": True,
                    "brain_region_standardized": "midbrain",
                    "sex_standardized": "unknown",
                },
            }
            graph_path = holdout_dir / "graphs" / f"{graph_id}.pkl.gz"
            save_pickle_gz(graph_path, graph)
            graph_paths.append(
                {
                    "patient": patient,
                    "graph_id": graph_id,
                    "graph_path": str(graph_path),
                    "graph_unit_id": f"{HOLDOUT_SOURCE_ID}::{patient}",
                    "disease_group": "pd" if patient.startswith("PD") else "control",
                }
            )
    holdout_graph_manifest = pd.DataFrame(graph_paths)
    holdout_graph_manifest.to_csv(holdout_dir / "gse157783_mgda_graph_manifest.csv", index=False)
    if holdout_graph_manifest.empty:
        return {
            "patient_counts": patient_counts,
            "lr_scores": holdout_lr_scores,
            "pathway_scores": holdout_pathway_scores,
            "gnn_scores": pd.DataFrame(),
            "baseline_scores": holdout_wide,
        }

    sample_graph = load_graph(holdout_graph_manifest.iloc[0]["graph_path"])
    node_dim = sample_graph["node_features"].shape[1]
    edge_dim = sample_graph["edge_attr"].shape[1]
    age_folds = reconstruct_age_cv(graph_manifest, age_config)
    pd_folds = reconstruct_pd_cv(graph_manifest, pd_config)
    age_vocab = build_context_vocabulary(age_training_manifest(graph_manifest, age_config))
    age_context_sizes = context_cardinalities(age_vocab)
    pd_vocab = build_context_vocabulary(pd_training_manifest_for_vocab(graph_manifest, pd_config))
    pd_context_sizes = context_cardinalities(pd_vocab)
    age_model_config = dict(age_config["model"])
    pd_model_config = dict(pd_config["model"])
    source_invariance = dict(pd_config.get("source_invariance", {}))
    gnn_rows: list[dict[str, Any]] = []
    device = torch.device("cpu")
    for fold in age_folds:
        age_model = AgeModel(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=int(age_model_config["hidden_dim"]),
            num_layers=int(age_model_config["num_layers"]),
            dropout=float(age_model_config["dropout"]),
            model_family=str(age_model_config.get("family", "legacy_gnn")),
            num_heads=int(age_model_config.get("num_heads", 4)),
            context_cardinalities=age_context_sizes,
            context_hidden_dim=int(age_model_config.get("context_hidden_dim", 48)),
        ).to(device)
        age_model.load_state_dict(torch.load(project_path(age_config["paths"]["output_dir"]) / f"fold_{fold['fold']:02d}_model.pt", map_location=device))
        age_model.eval()

        pd_fold = next(item for item in pd_folds if item["fold"] == fold["fold"])
        pd_model = AgeConditionedPDModel(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=int(pd_model_config["hidden_dim"]),
            age_hidden_dim=int(pd_model_config["age_hidden_dim"]),
            num_layers=int(pd_model_config["num_layers"]),
            dropout=float(pd_model_config["dropout"]),
            model_family=str(pd_model_config.get("family", "legacy_gnn")),
            num_heads=int(pd_model_config.get("num_heads", 4)),
            context_cardinalities=pd_context_sizes,
            context_hidden_dim=int(pd_model_config.get("context_hidden_dim", 48)),
            domain_num_classes=pd_context_sizes["source_id"] if bool(source_invariance.get("enabled", True)) else 0,
            domain_hidden_dim=int(pd_model_config.get("domain_hidden_dim", 32)),
            gradient_reversal_lambda=float(source_invariance.get("gradient_reversal_lambda", 0.25)),
        ).to(device)
        pd_model.load_state_dict(torch.load(project_path(pd_config["paths"]["output_dir"]) / f"fold_{fold['fold']:02d}_model.pt", map_location=device))
        pd_model.eval()

        with torch.no_grad():
            for row in holdout_graph_manifest.itertuples(index=False):
                graph_obj = load_graph(row.graph_path)
                graph = tensorize_graph(graph_obj, device)
                age_graph_context = tensorize_graph_context(graph_obj["metadata"], graph, age_vocab, device)
                age_pred_norm, _, _ = age_model(graph, age_graph_context)
                age_pred_years = float(age_pred_norm.cpu().item() * fold["age_std"] + fold["age_mean"])
                age_condition = torch.tensor(
                    (age_pred_years - pd_fold["age_mean"]) / pd_fold["age_std"],
                    dtype=torch.float32,
                    device=device,
                )
                pd_graph_context = tensorize_graph_context(graph_obj["metadata"], graph, pd_vocab, device)
                disease_logit, _, _, _, _ = pd_model(graph, age_condition, pd_graph_context)
                disease_logit_value = float(disease_logit.cpu().item())
                calibration = {
                    "calibration_method": pd_fold.get("calibration_method", "identity"),
                    "calibration_intercept": pd_fold.get("calibration_intercept", 0.0),
                    "calibration_slope": pd_fold.get("calibration_slope", 1.0),
                }
                gnn_rows.append(
                    {
                        "fold": int(fold["fold"]),
                        "patient": row.patient,
                        "graph_id": row.graph_id,
                        "graph_unit_id": row.graph_unit_id,
                        "disease_group": row.disease_group,
                        "predicted_age_years": age_pred_years,
                        "pd_logit": disease_logit_value,
                        "pd_probability_raw": float(torch.sigmoid(disease_logit).cpu().item()),
                        "pd_probability": float(apply_platt_scaler(np.asarray([disease_logit_value], dtype=float), calibration)[0]),
                    }
                )
    holdout_gnn_scores = pd.DataFrame(gnn_rows)
    holdout_gnn_scores.to_csv(holdout_dir / "gse157783_mgda_gnn_scores.csv", index=False)
    holdout_wide.to_csv(holdout_dir / "gse157783_mgda_baseline_scores.csv", index=False)

    return {
        "patient_counts": patient_counts,
        "lr_scores": holdout_lr_scores,
        "pathway_scores": holdout_pathway_scores,
        "gnn_scores": holdout_gnn_scores,
        "baseline_scores": holdout_wide,
    }


def build_gse253462_holdout(
    output_dir: Path,
    cohort_config: dict[str, Any],
    lr_universe_full: pd.DataFrame,
    comm_genes: list[str],
) -> dict[str, Any]:
    log("Preparing GSE253462 nigra holdout...")
    holdout_dir = output_dir / "gse253462_holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)

    cached_holdout = load_cached_nigra_holdout_outputs(holdout_dir)
    if cached_holdout is not None:
        return cached_holdout

    if not NIGRA_HOLDOUT_TAR.exists():
        raise FileNotFoundError(f"Missing GSE253462 tarball at {NIGRA_HOLDOUT_TAR}")

    manifest = load_gse253462_manifest()
    if manifest.empty:
        empty = pd.DataFrame()
        return {
            "patient_counts": empty,
            "lr_scores": empty,
            "pathway_scores": empty,
            "gnn_scores": empty,
            "baseline_scores": empty,
            "assignment_summary": empty,
        }

    mgda_task = cohort_config["tasks"]["microglia_to_dopaminergic_neuron"]
    task_name = "microglia_to_dopaminergic_neuron"
    comm_gene_set = {gene.upper() for gene in comm_genes}
    task_pairs_full = lr_universe_full.loc[lr_universe_full["eligible_tasks"].astype(str).str.contains(task_name)].copy()

    patient_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    holdout_gene_stats_rows: list[dict[str, Any]] = []
    lr_rows: list[dict[str, Any]] = []

    with tarfile.open(NIGRA_HOLDOUT_TAR, "r") as tar_handle:
        for sample in manifest.itertuples(index=False):
            barcodes, total_counts, genes_detected, mt_counts, marker_values = collect_nigra_marker_pass(
                tar_handle,
                sample.member_name,
            )
            selected_cells, summary = classify_nigra_holdout_cells(
                barcodes=barcodes,
                total_counts=total_counts,
                genes_detected=genes_detected,
                mt_counts=mt_counts,
                marker_values=marker_values,
                metadata_row=pd.Series(sample._asdict()),
            )
            microglia_count = int(selected_cells["cell_type_standardized_final"].eq("microglia").sum())
            da_count = int(selected_cells["cell_type_standardized_final"].eq("dopaminergic neuron").sum())
            eligible = microglia_count >= int(mgda_task["min_sender_cells"]) and da_count >= int(mgda_task["min_receiver_cells"])
            patient_rows.append(
                {
                    "gsm_accession": str(sample.gsm_accession),
                    "donor_id": str(sample.donor_id),
                    "disease_group": str(sample.disease_group),
                    "microglia": microglia_count,
                    "dopaminergic neuron": da_count,
                    "eligible_mg_da": bool(eligible),
                }
            )
            assignment_rows.append(
                {
                    "gsm_accession": str(sample.gsm_accession),
                    "donor_id": str(sample.donor_id),
                    "disease_group": str(sample.disease_group),
                    **summary,
                }
            )
            if not eligible or selected_cells.empty:
                continue

            selected_positions = pd.Index(barcodes.astype(str)).get_indexer(selected_cells["barcode"].astype(str))
            selected_positions = selected_positions[selected_positions >= 0]
            if selected_positions.size == 0:
                continue
            selected_matrix, gene_order = collect_nigra_selected_gene_matrix(
                tar_handle=tar_handle,
                member_name=str(sample.member_name),
                selected_indices=selected_positions.astype(int),
                gene_set=comm_gene_set,
            )
            if selected_matrix.shape[1] == 0:
                continue
            selected_totals = selected_cells["qc_total_counts"].to_numpy(dtype=float)
            norm = selected_matrix / np.maximum(selected_totals[:, None], 1.0) * 1e4

            for cell_type, frame in selected_cells.groupby("cell_type_standardized_final", observed=False):
                positions = frame.index.to_numpy(dtype=int)
                sub = norm[positions]
                pct_expr = (sub > 0).mean(axis=0)
                mean_norm = sub.mean(axis=0)
                for gene_idx, gene_symbol in enumerate(gene_order):
                    holdout_gene_stats_rows.append(
                        {
                            "graph_unit_id": str(sample.graph_unit_id),
                            "cell_type": str(cell_type),
                            "gene_symbol": str(gene_symbol),
                            "mean_norm": float(mean_norm[gene_idx]),
                            "pct_expr": float(pct_expr[gene_idx]),
                            "n_cells": int(len(positions)),
                        }
                    )

            gene_index = {
                (str(sample.graph_unit_id), str(cell_type), str(gene_symbol)): (float(mean_norm), float(pct_expr))
                for cell_type, frame in selected_cells.groupby("cell_type_standardized_final", observed=False)
                for gene_symbol, mean_norm, pct_expr in zip(
                    gene_order,
                    norm[frame.index.to_numpy(dtype=int)].mean(axis=0),
                    (norm[frame.index.to_numpy(dtype=int)] > 0).mean(axis=0),
                    strict=False,
                )
            }
            for pair in task_pairs_full.itertuples(index=False):
                ligand_genes = [gene for gene in str(pair.ligand_genes).split("|") if gene]
                receptor_genes = [gene for gene in str(pair.receptor_genes).split("|") if gene]
                try:
                    sender_expr = [gene_index[(str(sample.graph_unit_id), "microglia", gene)][0] for gene in ligand_genes]
                    sender_pct = [gene_index[(str(sample.graph_unit_id), "microglia", gene)][1] for gene in ligand_genes]
                    receiver_expr = [gene_index[(str(sample.graph_unit_id), "dopaminergic neuron", gene)][0] for gene in receptor_genes]
                    receiver_pct = [gene_index[(str(sample.graph_unit_id), "dopaminergic neuron", gene)][1] for gene in receptor_genes]
                except KeyError:
                    continue
                sender_value = float(np.prod(np.maximum(sender_expr, 1e-12)) ** (1.0 / len(sender_expr)))
                receiver_value = float(np.prod(np.maximum(receiver_expr, 1e-12)) ** (1.0 / len(receiver_expr)))
                lr_rows.append(
                    {
                        "graph_unit_id": str(sample.graph_unit_id),
                        "split_group_id": str(sample.split_group_id),
                        "source_id": NIGRA_HOLDOUT_SOURCE_ID,
                        "sample_id": str(sample.sample_id),
                        "donor_id": str(sample.donor_id),
                        "disease_group": str(sample.disease_group),
                        "task_name": task_name,
                        "pair_key": pair.pair_key,
                        "pathway_family": pair.pathway_family,
                        "score": float(np.sqrt(sender_value * receiver_value)),
                        "sender_mean_norm": sender_value,
                        "receiver_mean_norm": receiver_value,
                        "sender_pct_expr": float(np.min(sender_pct)),
                        "receiver_pct_expr": float(np.min(receiver_pct)),
                    }
                )

    patient_counts = pd.DataFrame(patient_rows).sort_values(["disease_group", "donor_id", "gsm_accession"])
    assignment_summary = pd.DataFrame(assignment_rows).sort_values(["disease_group", "donor_id", "gsm_accession"])
    holdout_lr_scores = pd.DataFrame(lr_rows)
    holdout_pathway_scores = (
        holdout_lr_scores.groupby(
            ["graph_unit_id", "split_group_id", "source_id", "sample_id", "donor_id", "disease_group", "task_name", "pathway_family"],
            observed=False,
        )["score"]
        .mean()
        .rename("pathway_score")
        .reset_index()
        if not holdout_lr_scores.empty
        else pd.DataFrame()
    )

    patient_counts.to_csv(holdout_dir / "gse253462_patient_celltype_counts.csv", index=False)
    assignment_summary.to_csv(holdout_dir / "gse253462_assignment_summary.csv", index=False)
    holdout_lr_scores.to_parquet(holdout_dir / "gse253462_mgda_lr_scores.parquet", index=False)
    holdout_pathway_scores.to_parquet(holdout_dir / "gse253462_mgda_pathway_scores.parquet", index=False)
    if holdout_gene_stats_rows:
        pd.DataFrame(holdout_gene_stats_rows).to_parquet(holdout_dir / "gse253462_mgda_gene_stats.parquet", index=False)

    return {
        "patient_counts": patient_counts,
        "lr_scores": holdout_lr_scores,
        "pathway_scores": holdout_pathway_scores,
        "gnn_scores": pd.DataFrame(),
        "baseline_scores": pd.DataFrame(),
        "assignment_summary": assignment_summary,
    }


def summarize_holdout_effects_by_source(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["holdout_source_id", *group_columns, "control", "pd", "holdout_pd_minus_control"])
    summary = (
        frame.groupby(["source_id", *group_columns, "disease_group"], observed=False)[value_column]
        .mean()
        .unstack(fill_value=np.nan)
        .reset_index()
        .rename(columns={"source_id": "holdout_source_id"})
    )
    if {"control", "pd"}.issubset(set(summary.columns)):
        summary["holdout_pd_minus_control"] = summary["pd"] - summary["control"]
    else:
        summary["holdout_pd_minus_control"] = np.nan
    return summary


def aggregate_holdout_effects(
    by_source: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    if by_source.empty:
        return pd.DataFrame(columns=[*group_columns, "control", "pd", "holdout_pd_minus_control", "n_holdout_sources", "holdout_sources"])
    rows: list[dict[str, Any]] = []
    for group_key, local in by_source.groupby(group_columns, observed=False):
        record = {}
        if isinstance(group_key, tuple):
            record.update(dict(zip(group_columns, group_key, strict=False)))
        else:
            record[group_columns[0]] = group_key
        weights = local["holdout_source_id"].map(HOLDOUT_SOURCE_WEIGHTS).fillna(1.0).astype(float)
        for column in ["control", "pd", "holdout_pd_minus_control"]:
            valid = local[column].notna()
            if valid.any():
                record[column] = float(np.average(local.loc[valid, column], weights=weights.loc[valid]))
            else:
                record[column] = float("nan")
        record["n_holdout_sources"] = int(local["holdout_source_id"].nunique())
        record["holdout_sources"] = "|".join(sorted(local["holdout_source_id"].dropna().astype(str).unique().tolist()))
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    interp_config = load_yaml(args.interpretation_config)
    cohort_config = load_yaml(args.cohort_config)
    graph_config = load_yaml(args.graph_config)
    age_config = load_yaml(args.age_config)
    pd_config = load_yaml(args.pd_config)

    output_dir = project_path("data/processed/step2/07_robustness")
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_manifest = read_table(project_path(interp_config["paths"]["graph_manifest_parquet"]))
    compact_universe = read_table(project_path(interp_config["paths"]["lr_universe_compact"])).reset_index(drop=True)
    lr_universe_full = read_table(project_path(interp_config["paths"]["lr_universe_full"]))
    pathway_scores = read_table(project_path("data/processed/step2/04_baselines/step2_pathway_scores.parquet"))
    comm_obs = read_table(project_path("data/processed/step2/00_backbone/step2_backbone_manifest.parquet"))
    comm_genes = (
        pd.read_csv(project_path("data/processed/step2/01_comm_panel/step2_comm_gene_panel.csv"))["gene_symbol"]
        .astype(str)
        .tolist()
    )

    age_predictions = pd.read_csv(project_path("data/processed/step2/06_models/age_model/age_model_predictions.csv"))
    pd_predictions = pd.read_csv(project_path("data/processed/step2/06_models/pd_model/pd_model_predictions.csv"))
    bridge_scores = pd.read_csv(project_path("data/processed/step2/06_models/pd_model/pd_model_lbd_bridge_scores.csv"))

    label_sensitivity = compute_label_confidence_sensitivity(comm_obs, cohort_config)
    label_sensitivity.to_csv(output_dir / "step2_label_confidence_sensitivity.csv", index=False)

    source_coverage = compute_source_task_coverage(graph_manifest)
    source_coverage.to_csv(output_dir / "step2_source_task_coverage.csv", index=False)

    receiver_specificity = compute_receiver_specificity(pathway_scores)
    receiver_specificity.to_csv(output_dir / "step2_receiver_robustness.csv", index=False)

    cell_count_robustness = compute_cell_count_robustness(graph_manifest, age_predictions, pd_predictions)
    cell_count_robustness.to_csv(output_dir / "step2_cell_count_robustness.csv", index=False)

    lbd_bridge = compute_lbd_bridge_summary(pd_predictions, bridge_scores)
    lbd_bridge.to_csv(output_dir / "step2_lbd_bridge_summary.csv", index=False)

    holdout = build_gse157783_holdout(
        output_dir=output_dir,
        cohort_config=cohort_config,
        graph_config=graph_config,
        age_config=age_config,
        pd_config=pd_config,
        graph_manifest=graph_manifest,
        compact_universe=compact_universe,
        lr_universe_full=lr_universe_full,
        comm_genes=comm_genes,
    )
    nigra_holdout = build_gse253462_holdout(
        output_dir=output_dir,
        cohort_config=cohort_config,
        lr_universe_full=lr_universe_full,
        comm_genes=comm_genes,
    )

    holdout_pathway_effects = pd.DataFrame()
    holdout_lr_effects = pd.DataFrame()
    holdout_pathway_effects_by_source = pd.DataFrame()
    holdout_lr_effects_by_source = pd.DataFrame()
    holdout_gnn_summary = pd.DataFrame()
    combined_holdout_pathways = pd.concat(
        [frame for frame in [holdout["pathway_scores"], nigra_holdout["pathway_scores"]] if not frame.empty],
        ignore_index=True,
    ) if (not holdout["pathway_scores"].empty or not nigra_holdout["pathway_scores"].empty) else pd.DataFrame()
    combined_holdout_lr = pd.concat(
        [frame for frame in [holdout["lr_scores"], nigra_holdout["lr_scores"]] if not frame.empty],
        ignore_index=True,
    ) if (not holdout["lr_scores"].empty or not nigra_holdout["lr_scores"].empty) else pd.DataFrame()
    if not combined_holdout_pathways.empty:
        holdout_pathway_effects_by_source = summarize_holdout_effects_by_source(
            combined_holdout_pathways,
            group_columns=["pathway_family"],
            value_column="pathway_score",
        )
        holdout_pathway_effects = aggregate_holdout_effects(
            holdout_pathway_effects_by_source,
            group_columns=["pathway_family"],
        )
        holdout_pathway_effects_by_source.to_csv(output_dir / "step2_holdout_pathway_effects_by_source.csv", index=False)
        holdout_pathway_effects.to_csv(output_dir / "step2_holdout_pathway_effects.csv", index=False)
    if not combined_holdout_lr.empty:
        holdout_lr_effects_by_source = summarize_holdout_effects_by_source(
            combined_holdout_lr,
            group_columns=["pair_key", "pathway_family"],
            value_column="score",
        )
        holdout_lr_effects = aggregate_holdout_effects(
            holdout_lr_effects_by_source,
            group_columns=["pair_key", "pathway_family"],
        )
        holdout_lr_effects_by_source.to_csv(output_dir / "step2_holdout_lr_effects_by_source.csv", index=False)
        holdout_lr_effects.to_csv(output_dir / "step2_holdout_lr_effects.csv", index=False)
    if not holdout["gnn_scores"].empty:
        holdout_gnn_summary = (
            holdout["gnn_scores"]
            .groupby(["patient", "graph_unit_id", "disease_group"], observed=False)
            .agg(
                predicted_age_years=("predicted_age_years", "mean"),
                pd_probability=("pd_probability", "mean"),
            )
            .reset_index()
            .sort_values(["disease_group", "patient"])
        )
        holdout_gnn_summary.to_csv(output_dir / "step2_holdout_gnn_summary.csv", index=False)

    notes = [
        "- All reported model metrics remain donor-grouped.",
        "- `microglia_to_dopaminergic_neuron` PD donors remain highly source-skewed toward the CELLxGENE SNpc collection, so true leave-one-source-out PD evaluation is not estimable yet.",
        "- Region sensitivity is partially implicit in the current backbone because the DA-focused task is already substantia nigra dominant.",
        "- `GSE157783` remains the midbrain external holdout with baseline and approximate GNN scoring on the eligible microglia-to-DA patients.",
        "- `GSE253462` is now staged as a substantia nigra holdout using strict marker-based microglia and dopaminergic-cell assignment, and its effects are written separately before weighted aggregation.",
    ]
    report = "\n".join(
        [
            "# Step 2 Robustness Report",
            "",
            "## Label Confidence Sensitivity",
            markdown_table(label_sensitivity.head(24)),
            "",
            "## Source Coverage",
            markdown_table(source_coverage.head(20)),
            "",
            "## Receiver Robustness",
            markdown_table(receiver_specificity.head(20).fillna("NA")),
            "",
            "## Cell Count Robustness",
            markdown_table(cell_count_robustness.fillna("NA")),
            "",
            "## LBD Bridge Summary",
            markdown_table(lbd_bridge.fillna("NA")),
            "",
            "## Holdout Sources",
            markdown_table(
                pd.concat(
                    [
                        holdout_pathway_effects_by_source[["holdout_source_id"]].drop_duplicates(),
                        holdout_lr_effects_by_source[["holdout_source_id"]].drop_duplicates(),
                    ],
                    ignore_index=True,
                ).drop_duplicates().rename(columns={"holdout_source_id": "available_holdout_source"})
                if not holdout_pathway_effects_by_source.empty or not holdout_lr_effects_by_source.empty
                else pd.DataFrame()
            ),
            "",
            "## GSE157783 Holdout GNN Summary",
            markdown_table(holdout_gnn_summary.fillna("NA") if not holdout_gnn_summary.empty else pd.DataFrame()),
            "",
            "## Notes",
            *notes,
        ]
    )
    write_markdown(output_dir / "step2_robustness_report.md", report)


if __name__ == "__main__":
    main()
