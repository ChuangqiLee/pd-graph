#!/usr/bin/env python3.12
"""Shared helpers for the Step 3 biological interpretation pipeline."""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

if TYPE_CHECKING:
    import anndata as ad

from scripts.step2.common import (
    PROJECT_ROOT,
    ensure_dir,
    load_pickle_gz,
    markdown_table,
    normalize_text,
    project_path,
    safe_slug,
    stable_hash,
    write_markdown,
)


STEP3_ROOT = PROJECT_ROOT / "data/processed/step3"
STEP2_ROOT = PROJECT_ROOT / "data/processed/step2"
COMM_PANEL_PATH = STEP2_ROOT / "01_comm_panel/pd_definitive_step2_comm_panel.h5ad"
PRIMARY_TASK = "microglia_to_dopaminergic_neuron"
TASK_ORDER = [
    "microglia_to_dopaminergic_neuron",
    "microglia_to_neuron",
    "astrocyte_to_dopaminergic_neuron",
    "endothelial_to_dopaminergic_neuron",
]
TASK_TO_CELL_TYPES = {
    "microglia_to_dopaminergic_neuron": ("microglia", "dopaminergic neuron"),
    "microglia_to_neuron": ("microglia", "neuron"),
    "astrocyte_to_dopaminergic_neuron": ("astrocyte", "dopaminergic neuron"),
    "endothelial_to_dopaminergic_neuron": ("endothelial cell", "dopaminergic neuron"),
}
TASK_DISPLAY = {
    "microglia_to_dopaminergic_neuron": "microglia -> dopaminergic neuron",
    "microglia_to_neuron": "microglia -> neuron",
    "astrocyte_to_dopaminergic_neuron": "astrocyte -> dopaminergic neuron",
    "endothelial_to_dopaminergic_neuron": "endothelial -> dopaminergic neuron",
}
COHORT_DISPLAY = {
    "control_known_age": "control",
    "pd_known_age": "pd",
    "lbd_bridge": "lbd",
    "unknown_age_auxiliary": "auxiliary",
}
INTERPRETATION_DISPLAY = {
    "pd_amplified_aging": "PD amplified aging",
    "pd_specific": "PD specific",
    "normal_aging_associated": "Normal aging associated",
    "unstable_or_source_sensitive": "Unstable or source sensitive",
}
AMBIGUOUS_PATHWAY_FAMILIES = {
    "unclassified",
    "extracellular matrix and adhesion",
    "growth factor signaling",
}
KNOWN_Y_GENES = {
    "DDX3Y",
    "EIF1AY",
    "KDM5D",
    "NLGN4Y",
    "RPS4Y1",
    "RPS4Y2",
    "TMSB4Y",
    "UTY",
    "ZFY",
}

PROGRAM_METADATA: dict[str, dict[str, Any]] = {
    "app_centered_stress_signaling": {
        "name": "APP centered stress signaling",
        "priority": 1,
        "summary": "APP-centered neuronal stress and vulnerability with APOE as the dominant microglial ligand context.",
    },
    "apoe_lipoprotein_receptor_signaling": {
        "name": "APOE lipoprotein receptor signaling",
        "priority": 2,
        "summary": "A complementary APOE/LRP branch that likely reflects lipid handling and receptor-mediated uptake.",
    },
    "slit_robo_guidance": {
        "name": "SLIT and ROBO guidance",
        "priority": 3,
        "summary": "Guidance signaling centered on SLIT ligands and ROBO receptors, with PD-enriched pathway behavior.",
    },
    "glutamatergic_ionotropic_receptor_stress": {
        "name": "Glutamatergic ionotropic receptor stress",
        "priority": 4,
        "summary": "Glutamatergic signaling focused on ionotropic receptors such as GRIK and GRIA family members.",
    },
    "glutamatergic_metabotropic_receptor_stress": {
        "name": "Glutamatergic metabotropic receptor stress",
        "priority": 5,
        "summary": "Glutamatergic signaling focused on metabotropic GRM receptors.",
    },
    "dscam_dcc_contact_recognition": {
        "name": "DSCAM and DCC contact recognition",
        "priority": 6,
        "summary": "A contact-recognition branch split from the raw unclassified bucket.",
    },
    "nlgn_nrxn_synaptic_organizer": {
        "name": "NLGN and NRXN synaptic organizer signaling",
        "priority": 7,
        "summary": "Synaptic organizer signaling including NLGN and NRXN family anchors.",
    },
    "adhesion_organizer_network": {
        "name": "Adhesion organizer network",
        "priority": 8,
        "summary": "Adhesion and contact organizer pairs that are more structured than a generic unclassified label.",
    },
    "erbb4_trophic_maintenance": {
        "name": "ERBB4 trophic maintenance",
        "priority": 9,
        "summary": "NRG and ERBB4-centered trophic or maintenance signaling.",
    },
    "extracellular_matrix_integrin_adhesion": {
        "name": "Extracellular matrix and integrin adhesion",
        "priority": 10,
        "summary": "Structural matrix and integrin-style adhesion signaling.",
    },
    "immune_neuronal_il18_interface": {
        "name": "Immune neuronal IL18 interface",
        "priority": 11,
        "summary": "A secondary inflammatory interface retained to capture the original cytokine hypothesis.",
    },
    "exploratory_residual_program": {
        "name": "Exploratory residual program",
        "priority": 12,
        "summary": "Residual low-confidence or biologically mixed candidates that remain narrative only.",
    },
}
PROGRAM_ORDER = sorted(PROGRAM_METADATA, key=lambda key: PROGRAM_METADATA[key]["priority"])

LINEAGE_MODULES: dict[str, dict[str, list[str]]] = {
    "microglia": {
        "homeostatic": ["P2RY12", "CX3CR1", "CSF1R", "C1QA", "C1QB", "C1QC", "TMEM119", "HEXB"],
        "reactive_lipid": ["APOE", "LPL", "SPP1", "GPNMB", "CTSB", "CTSD", "C3", "HLA-DRA"],
        "antigen_presentation": ["HLA-DRA", "HLA-DRB1", "CD74", "B2M", "TAP1", "TAP2"],
        "inflammatory": ["IL1B", "TNF", "CCL2", "CCL3", "CXCL8", "NFKBIA"],
        "stress_response": ["APOE", "C3", "LGALS3", "IFITM3", "CTSB", "APP"],
    },
    "dopaminergic neuron": {
        "dopaminergic_core": ["TH", "SLC6A3", "DDC", "ALDH1A1", "GCH1", "DRD2"],
        "guidance_receiver": ["ROBO1", "ROBO2", "DCC", "DSCAM", "SLITRK1", "SLITRK5"],
        "glutamatergic_receiver": ["GRIK2", "GRIA1", "GRIA4", "GRM7", "GRM8"],
        "trophic_receiver": ["APP", "ERBB4", "LRP1", "LRP8", "IL1RAPL1"],
        "synaptic_organizer": ["NRXN1", "NRXN3", "NLGN1", "LRRTM4", "CADM1", "ADGRL2"],
    },
    "astrocyte": {
        "homeostatic_astrocyte": ["AQP4", "SLC1A3", "SLC1A2", "ALDH1L1", "GJA1", "FGFR3"],
        "reactive_astrocyte": ["GFAP", "VIM", "APOE", "C3", "SERPINA3", "LGALS3"],
        "glutamate_handling": ["SLC1A3", "SLC1A2", "GLS", "GLUL", "GJA1"],
    },
    "endothelial cell": {
        "barrier_endothelial": ["CLDN5", "KDR", "FLT1", "ESAM", "EMCN", "PECAM1"],
        "angiogenic_endothelial": ["KDR", "FLT1", "VWF", "ANGPT2", "PLVAP"],
        "immune_interface_endothelial": ["SELE", "ICAM1", "VCAM1", "CXCL12", "TNFRSF1A"],
    },
    "neuron": {
        "pan_neuron": ["SNAP25", "SYT1", "RBFOX3", "STMN2", "NEFL", "NRXN1"],
        "synaptic_remodeling": ["NRXN1", "NRXN3", "NLGN1", "LRRTM4", "CADM1"],
        "guidance_receiver": ["ROBO1", "ROBO2", "DCC", "DSCAM"],
        "glutamatergic_receiver": ["GRIK2", "GRIA1", "GRIA4", "GRM7", "GRM8"],
    },
}

PROGRAM_VALIDATION_MODULES: dict[str, list[tuple[str, str]]] = {
    "app_centered_stress_signaling": [
        ("microglia", "reactive_lipid"),
        ("microglia", "stress_response"),
        ("dopaminergic neuron", "trophic_receiver"),
    ],
    "apoe_lipoprotein_receptor_signaling": [
        ("microglia", "reactive_lipid"),
        ("microglia", "stress_response"),
        ("dopaminergic neuron", "trophic_receiver"),
    ],
    "immune_neuronal_il18_interface": [
        ("microglia", "reactive_lipid"),
        ("microglia", "stress_response"),
        ("microglia", "antigen_presentation"),
        ("microglia", "inflammatory"),
        ("dopaminergic neuron", "trophic_receiver"),
    ],
    "slit_robo_guidance": [
        ("microglia", "stress_response"),
        ("dopaminergic neuron", "guidance_receiver"),
    ],
    "glutamatergic_ionotropic_receptor_stress": [
        ("microglia", "stress_response"),
        ("dopaminergic neuron", "glutamatergic_receiver"),
    ],
    "glutamatergic_metabotropic_receptor_stress": [
        ("microglia", "stress_response"),
        ("dopaminergic neuron", "glutamatergic_receiver"),
    ],
    "adhesion_organizer_network": [
        ("microglia", "stress_response"),
        ("dopaminergic neuron", "synaptic_organizer"),
        ("dopaminergic neuron", "guidance_receiver"),
    ],
    "dscam_dcc_contact_recognition": [
        ("microglia", "homeostatic"),
        ("microglia", "stress_response"),
        ("dopaminergic neuron", "guidance_receiver"),
        ("dopaminergic neuron", "synaptic_organizer"),
    ],
    "erbb4_trophic_maintenance": [
        ("microglia", "stress_response"),
        ("dopaminergic neuron", "trophic_receiver"),
    ],
    "extracellular_matrix_integrin_adhesion": [
        ("microglia", "stress_response"),
        ("dopaminergic neuron", "guidance_receiver"),
        ("dopaminergic neuron", "synaptic_organizer"),
    ],
    "nlgn_nrxn_synaptic_organizer": [
        ("microglia", "homeostatic"),
        ("dopaminergic neuron", "synaptic_organizer"),
    ],
    "exploratory_residual_program": [
        ("microglia", "stress_response"),
        ("dopaminergic neuron", "synaptic_organizer"),
    ],
}

VALIDATION_GENE_PRIORITY = [
    "GPNMB",
    "HLA-DRA",
    "C3",
    "LGALS3",
    "IFITM3",
    "CTSB",
    "EDIL3",
    "SCARB1",
    "BACE2",
    "LRP1",
    "LRP8",
    "GRM8",
    "GRIA4",
    "GRIA2",
    "OLFM2",
    "NCAM1",
    "ADAM17",
    "NRG1",
    "ERBB4",
    "FGF20",
]


def load_step2_tables() -> dict[str, pd.DataFrame]:
    """Load the Step 2 tables that Step 3 depends on."""

    return {
        "graph_manifest": pd.read_parquet(project_path("data/processed/step2/05_graphs/step2_graph_manifest.parquet")),
        "backbone": pd.read_parquet(project_path("data/processed/step2/00_backbone/step2_backbone_manifest.parquet")),
        "lr_scores": pd.read_parquet(project_path("data/processed/step2/04_baselines/step2_lr_scores.parquet")),
        "pathway_scores": pd.read_parquet(project_path("data/processed/step2/04_baselines/step2_pathway_scores.parquet")),
        "ranked_pairs": pd.read_csv(project_path("data/processed/step2/08_interpretation/step2_ranked_lr_pairs.csv")),
        "ranked_pathways": pd.read_csv(project_path("data/processed/step2/08_interpretation/step2_ranked_pathways.csv")),
        "edge_contexts": pd.read_csv(project_path("data/processed/step2/08_interpretation/step2_top_edge_contexts.csv")),
        "top_step3": pd.read_csv(project_path("data/processed/step2/09_exports/step2_top_candidates_for_step3.csv")),
        "top_step4": pd.read_csv(project_path("data/processed/step2/09_exports/step2_top_candidates_for_step4.csv")),
        "holdout_lr": pd.read_csv(project_path("data/processed/step2/07_robustness/step2_holdout_lr_effects.csv")),
        "holdout_pathways": pd.read_csv(project_path("data/processed/step2/07_robustness/step2_holdout_pathway_effects.csv")),
        "receiver_robustness": pd.read_csv(project_path("data/processed/step2/07_robustness/step2_receiver_robustness.csv")),
        "label_confidence": pd.read_csv(project_path("data/processed/step2/07_robustness/step2_label_confidence_sensitivity.csv")),
        "cell_count_robustness": pd.read_csv(project_path("data/processed/step2/07_robustness/step2_cell_count_robustness.csv")),
        "lbd_bridge": pd.read_csv(project_path("data/processed/step2/07_robustness/step2_lbd_bridge_summary.csv")),
        "holdout_gnn": pd.read_csv(project_path("data/processed/step2/07_robustness/step2_holdout_gnn_summary.csv")),
        "source_task_coverage": pd.read_csv(project_path("data/processed/step2/07_robustness/step2_source_task_coverage.csv")),
        "pair_universe": pd.read_parquet(project_path("data/processed/step2/03_lr_universe/step2_lr_universe_compact.parquet")),
        "age_attention": pd.read_parquet(project_path("data/processed/step2/06_models/age_model/age_model_attention_summary.parquet")),
        "pd_attention": pd.read_parquet(project_path("data/processed/step2/06_models/pd_model/pd_model_attention_summary.parquet")),
    }


def load_comm_adata(*, backed: str | None = None) -> "ad.AnnData":
    import anndata as ad

    return ad.read_h5ad(COMM_PANEL_PATH, backed=backed)


def compute_sha256(path_like: str | Path, *, chunk_size: int = 1 << 20) -> str:
    path = project_path(path_like)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_file(source_path: str | Path, destination_path: str | Path, *, mode: str = "copy") -> dict[str, Any]:
    """Stage a file into the Step 3 frozen input layer."""

    source = project_path(source_path)
    destination = project_path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
    used_mode = mode
    if mode == "hardlink_or_copy":
        try:
            os.link(source, destination)
            used_mode = "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            used_mode = "copy"
    elif mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source)
    else:
        raise ValueError(f"Unsupported stage mode: {mode}")
    return {
        "source_path": str(source),
        "frozen_path": str(destination),
        "stage_mode": used_mode,
        "size_bytes": destination.stat().st_size,
        "sha256": compute_sha256(destination),
    }


def split_pair_key(pair_key: str) -> tuple[str, str]:
    pair_text = normalize_text(pair_key)
    ligand, _, receptor = pair_text.partition("->")
    return ligand.strip(), receptor.strip()


def parse_gene_string(value: Any) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    return [gene.strip().upper() for gene in text.split("|") if gene.strip()]


def ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = normalize_text(value).upper()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def gene_family_token(gene: str) -> str:
    text = normalize_text(gene).upper()
    for prefix in [
        "NLGN",
        "NRXN",
        "DSCAM",
        "LRRTM",
        "LRRC",
        "ROBO",
        "SLIT",
        "GRIA",
        "GRIK",
        "GRM",
        "ERBB",
        "ITGA",
        "ITGB",
        "SLC1A",
        "IL1RAPL",
        "ADGRL",
        "LSAMP",
        "OPCML",
        "NEGR",
        "TENM",
        "CADM",
        "CNTN",
        "RTN4",
        "SEMA",
        "PLXN",
        "CXCL",
        "CXCR",
        "APOE",
        "APP",
        "NRG",
        "HLA",
        "TNFRSF",
        "IL",
    ]:
        if text.startswith(prefix):
            return prefix
    match = re.match(r"([A-Z]+)", text)
    return match.group(1) if match else text


def pair_candidate_id(pair_key: str) -> str:
    return f"lr_pair::{safe_slug(pair_key.replace('->', '_to_'))}"


def pathway_candidate_id(pathway_family: str) -> str:
    return f"pathway::{safe_slug(pathway_family)}"


def program_candidate_id(program_id: str) -> str:
    return f"program::{safe_slug(program_id)}"


def is_y_linked_gene(gene: str) -> bool:
    text = normalize_text(gene).upper()
    return text in KNOWN_Y_GENES or bool(re.search(r"\dY$", text))


def is_y_linked_pair(pair_key: str) -> bool:
    ligand_text, receptor_text = split_pair_key(pair_key)
    genes = parse_gene_string(ligand_text) + parse_gene_string(receptor_text)
    return any(is_y_linked_gene(gene) for gene in genes)


def parse_task_supports(value: Any) -> dict[str, float]:
    text = normalize_text(value)
    if not text:
        return {}
    supports: dict[str, float] = {}
    for item in text.split("|"):
        key, _, raw_value = item.partition(":")
        try:
            supports[key.strip()] = float(raw_value)
        except ValueError:
            continue
    return supports


def task_support_vector(value: Any) -> np.ndarray:
    parsed = parse_task_supports(value)
    return np.asarray([float(parsed.get(task, 0.0)) for task in TASK_ORDER], dtype=float)


def zscore_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.mean(skipna=True)
    std = values.std(skipna=True, ddof=0)
    if pd.isna(std) or float(std) == 0.0:
        return pd.Series(np.zeros(len(values), dtype=float), index=series.index)
    return (values - float(mean)) / float(std)


def minmax_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    min_value = values.min(skipna=True)
    max_value = values.max(skipna=True)
    if pd.isna(min_value) or pd.isna(max_value) or float(max_value) == float(min_value):
        return pd.Series(np.zeros(len(values), dtype=float), index=series.index)
    return (values - float(min_value)) / (float(max_value) - float(min_value))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator in (0, 0.0) or pd.isna(denominator):
        return float(default)
    return float(numerator) / float(denominator)


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    clean = [(float(value), float(weight)) for value, weight in zip(values, weights) if not pd.isna(value) and weight > 0]
    if not clean:
        return float("nan")
    vals, wts = zip(*clean)
    return float(np.average(np.asarray(vals, dtype=float), weights=np.asarray(wts, dtype=float)))


def linear_beta(frame: pd.DataFrame, x_col: str, y_col: str) -> float:
    local = frame[[x_col, y_col]].dropna().copy()
    if len(local) < 3 or local[x_col].nunique() < 2:
        return float("nan")
    design = np.column_stack([np.ones(len(local), dtype=float), local[x_col].astype(float).to_numpy()])
    target = local[y_col].astype(float).to_numpy()
    coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return float(coeffs[1])


def disease_effects(frame: pd.DataFrame, value_col: str) -> dict[str, float]:
    local = frame[[value_col, "age_years", "disease_group"]].dropna().copy()
    local = local.loc[local["disease_group"].isin(["control", "pd"])].copy()
    if len(local) < 4 or local["disease_group"].nunique() < 2:
        return {
            "pd_shift_size": float("nan"),
            "pd_age_interaction_beta": float("nan"),
        }
    local["disease_binary"] = local["disease_group"].map({"control": 0.0, "pd": 1.0}).astype(float)
    design = np.column_stack(
        [
            np.ones(len(local), dtype=float),
            local["age_years"].astype(float).to_numpy(),
            local["disease_binary"].to_numpy(),
            local["age_years"].astype(float).to_numpy() * local["disease_binary"].to_numpy(),
        ]
    )
    target = local[value_col].astype(float).to_numpy()
    coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return {
        "pd_shift_size": float(coeffs[2]),
        "pd_age_interaction_beta": float(coeffs[3]),
    }


def direction_label(value: float, *, tol: float = 0.05) -> str:
    if pd.isna(value):
        return "uncertain"
    if value > tol:
        return "increase"
    if value < -tol:
        return "decrease"
    return "flat"


def classify_bridge_pattern(control_mean: float, pd_mean: float, lbd_mean: float) -> str:
    if any(pd.isna(value) for value in [control_mean, pd_mean, lbd_mean]):
        return "uncertain"
    lower = min(control_mean, pd_mean)
    upper = max(control_mean, pd_mean)
    if lower <= lbd_mean <= upper:
        return "bridge_intermediate"
    return "bridge_divergent"


def choose_primary_task(row: pd.Series) -> str:
    values = {task: float(row.get(task, 0.0) or 0.0) for task in TASK_ORDER}
    best_task = max(values, key=values.get)
    return best_task


def infer_program_from_pair(pair_key: str, pathway_family: str) -> tuple[str, str, str]:
    """Return refined program id, rule note, and coarse confidence."""

    ligand_text, receptor_text = split_pair_key(pair_key)
    ligand_genes = parse_gene_string(ligand_text)
    receptor_genes = parse_gene_string(receptor_text)
    ligand_families = {gene_family_token(gene) for gene in ligand_genes}
    receptor_families = {gene_family_token(gene) for gene in receptor_genes}
    pair_text = normalize_text(pair_key).upper()
    pathway_text = normalize_text(pathway_family).lower()

    if pair_text in {"APOE->APP", "ADAM10->APP"} or "APP" in receptor_families:
        return "app_centered_stress_signaling", "explicit APP-centered anchor pair", "high"
    if "APOE" in ligand_families and receptor_families.intersection({"LRP", "LRP1", "LRP8"}):
        return "apoe_lipoprotein_receptor_signaling", "APOE paired with lipoprotein receptor", "high"
    if pathway_text == "stress response module":
        if receptor_families.intersection({"LRP", "LRP1", "LRP8"}):
            return "apoe_lipoprotein_receptor_signaling", "stress module split into receptor-adjunct branch", "medium"
        return "app_centered_stress_signaling", "stress module fallback", "medium"

    if pathway_text == "slit" or ligand_families.intersection({"SLIT", "OLFM"}) or receptor_families.intersection({"ROBO"}):
        return "slit_robo_guidance", "guidance receptor/ligand heuristic", "high"

    if pathway_text == "glutamate":
        if receptor_families.intersection({"GRM"}):
            return "glutamatergic_metabotropic_receptor_stress", "metabotropic glutamate receptor branch", "high"
        return "glutamatergic_ionotropic_receptor_stress", "ionotropic glutamate receptor branch", "high"

    if pathway_text in {"nrg", "growth factor signaling"} or receptor_families.intersection({"ERBB"}) or ligand_families.intersection({"NRG"}):
        return "erbb4_trophic_maintenance", "ERBB/NRG receptor-centered branch", "high"

    if pathway_text == "cytokine signaling" or ligand_families.intersection({"IL", "TNF", "CXCL"}) or receptor_families.intersection({"IL", "IL1RAPL", "TNFRSF", "CXCR"}):
        return "immune_neuronal_il18_interface", "immune-interface cytokine heuristic", "medium"

    if pathway_text == "extracellular matrix and adhesion":
        if ligand_families.intersection({"SPP", "COL", "FN", "LAM", "ITGA", "ITGB"}) or receptor_families.intersection({"ITGA", "ITGB"}):
            return "extracellular_matrix_integrin_adhesion", "matrix/integrin branch", "high"
        if ligand_families.intersection({"NRXN", "CADM", "LSAMP", "TENM"}) or receptor_families.intersection({"LRRTM", "NLGN", "ADGRL", "OPCML", "NEGR"}):
            return "adhesion_organizer_network", "adhesion organizer split from overlapping family", "medium"
        return "extracellular_matrix_integrin_adhesion", "pathway family fallback", "medium"

    if pathway_text == "unclassified":
        if ligand_families.intersection({"DSCAM", "DCC"}) or receptor_families.intersection({"DSCAM", "DCC"}):
            return "dscam_dcc_contact_recognition", "DSCAM/DCC contact-recognition split", "high"
        if ligand_families.intersection({"NLGN", "NRXN"}) or receptor_families.intersection({"NLGN", "NRXN"}):
            return "nlgn_nrxn_synaptic_organizer", "NLGN/NRXN synaptic-organizer split", "high"
        if ligand_families.intersection({"LRRTM", "LRRC", "CADM", "TENM", "LSAMP", "CNTN", "RTN4"}) or receptor_families.intersection({"LRRTM", "LRRC", "CADM", "TENM", "ADGRL", "OPCML", "NEGR", "CNTN"}):
            return "adhesion_organizer_network", "adhesion organizer split from unclassified family", "medium"
        return "exploratory_residual_program", "unclassified residual", "low"

    return "exploratory_residual_program", "fallback residual assignment", "low"


def program_name(program_id: str) -> str:
    return str(PROGRAM_METADATA.get(program_id, {}).get("name", program_id))


def anchor_genes_from_pairs(anchor_pairs: str | Sequence[str]) -> list[str]:
    if isinstance(anchor_pairs, str):
        pairs = [pair for pair in normalize_text(anchor_pairs).split("|") if pair]
    else:
        pairs = [normalize_text(pair) for pair in anchor_pairs if normalize_text(pair)]
    genes: list[str] = []
    for pair in pairs:
        ligand_text, receptor_text = split_pair_key(pair)
        genes.extend(parse_gene_string(ligand_text))
        genes.extend(parse_gene_string(receptor_text))
    return ordered_unique(genes)


def validation_module_genes(program_id: str) -> list[str]:
    genes: list[str] = []
    for lineage, module_name in PROGRAM_VALIDATION_MODULES.get(program_id, []):
        genes.extend(LINEAGE_MODULES.get(lineage, {}).get(module_name, []))
    return ordered_unique(genes)


def program_validation_gene_panel(
    program_id: str,
    anchor_pairs: str | Sequence[str],
    *,
    sender_state_markers: str = "",
    receiver_state_markers: str = "",
    max_genes: int = 10,
) -> list[str]:
    anchor_genes = anchor_genes_from_pairs(anchor_pairs)
    priority_lookup = {gene: index for index, gene in enumerate(VALIDATION_GENE_PRIORITY)}
    module_genes = validation_module_genes(program_id)
    module_gene_set = set(module_genes)
    state_marker_genes = ordered_unique(parse_gene_string(sender_state_markers) + parse_gene_string(receiver_state_markers))
    state_priority = sorted(
        [gene for gene in state_marker_genes if gene in module_gene_set or gene in set(anchor_genes)],
        key=lambda gene: (priority_lookup.get(gene, len(priority_lookup)), gene),
    )
    module_priority = sorted(
        module_genes,
        key=lambda gene: (priority_lookup.get(gene, len(priority_lookup)), gene),
    )
    panel = ordered_unique(anchor_genes + state_priority + module_priority)
    if len(panel) <= max_genes:
        return panel
    required = ordered_unique(anchor_genes)
    remaining = [gene for gene in panel if gene not in set(required)]
    return required + remaining[: max(0, max_genes - len(required))]


def donor_level_cell_counts(backbone: pd.DataFrame) -> pd.DataFrame:
    counts = (
        backbone.groupby(
            [
                "split_group_id",
                "source_id",
                "cohort_name" if "cohort_name" in backbone.columns else "disease_group",
                "disease_group",
                "age_years",
                "sex_standardized",
                "brain_region_standardized",
                "cell_type_standardized_final",
            ],
            observed=False,
        )["cell_id"]
        .count()
        .rename("cell_count")
        .reset_index()
    )
    return counts


def cell_count_table(backbone: pd.DataFrame) -> pd.DataFrame:
    local = backbone.copy()
    local["cohort_name"] = local.get("cohort_name", pd.Series(["unknown"] * len(local), index=local.index))
    counts = (
        local.groupby(
            [
                "split_group_id",
                "source_id",
                "cohort_name",
                "disease_group",
                "age_years",
                "sex_standardized",
                "brain_region_standardized",
                "cell_type_standardized_final",
            ],
            observed=False,
        )["cell_id"]
        .count()
        .rename("cell_count")
        .reset_index()
        .pivot_table(
            index=[
                "split_group_id",
                "source_id",
                "cohort_name",
                "disease_group",
                "age_years",
                "sex_standardized",
                "brain_region_standardized",
            ],
            columns="cell_type_standardized_final",
            values="cell_count",
            fill_value=0,
        )
        .reset_index()
    )
    counts.columns.name = None
    return counts


def major_source_share(frame: pd.DataFrame, group_col: str = "source_id") -> float:
    if frame.empty or group_col not in frame.columns:
        return float("nan")
    counts = frame[group_col].astype(str).value_counts(dropna=False)
    return safe_div(float(counts.max()), float(counts.sum()), default=float("nan"))


def top_program_anchor_pairs(refined_map: pd.DataFrame, ranked_pairs: pd.DataFrame, *, top_n: int = 5) -> pd.DataFrame:
    merged = refined_map.copy()
    for column in ["ranking_score", "pd_score", "aging_score"]:
        if column not in merged.columns:
            merged = merged.merge(ranked_pairs[["pair_key", column]], on="pair_key", how="left")
    ranking_column = "ranking_score"
    if ranking_column not in merged.columns:
        for fallback in ["ranking_score_x", "ranking_score_y"]:
            if fallback in merged.columns:
                ranking_column = fallback
                break
    ordered = (
        merged.sort_values(["refined_program_id", ranking_column], ascending=[True, False])
        .groupby("refined_program_id", observed=False)
        .head(top_n)
        .copy()
    )
    return ordered


def sparse_to_dense(matrix: sp.spmatrix | np.ndarray) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def extract_layer_matrix(
    adata: ad.AnnData,
    *,
    obs_mask: np.ndarray,
    gene_names: Sequence[str],
    layer: str = "lognorm",
) -> tuple[pd.DataFrame, np.ndarray]:
    genes = [gene for gene in gene_names if gene in adata.var_names]
    gene_index = [adata.var_names.get_loc(gene) for gene in genes]
    row_index = np.where(obs_mask)[0]
    matrix = adata.layers[layer][row_index, :][:, gene_index]
    dense = sparse_to_dense(matrix)
    obs = adata.obs.iloc[row_index].copy()
    return obs, dense.astype(np.float32)


def module_score_frame(expression: pd.DataFrame, modules: dict[str, list[str]]) -> pd.DataFrame:
    scores: dict[str, pd.Series] = {}
    for module_name, genes in modules.items():
        present = [gene for gene in genes if gene in expression.columns]
        if not present:
            scores[module_name] = pd.Series(0.0, index=expression.index)
            continue
        scores[module_name] = expression[present].mean(axis=1)
    return pd.DataFrame(scores, index=expression.index)


def write_simple_report(path_like: str | Path, title: str, sections: list[tuple[str, str]]) -> Path:
    lines = [f"# {title}", ""]
    for header, body in sections:
        lines.extend([f"## {header}", body.rstrip(), ""])
    return write_markdown(path_like, "\n".join(lines).rstrip())


def infer_top_state_label(score_row: pd.Series) -> str:
    if score_row.empty:
        return "unassigned"
    return str(score_row.sort_values(ascending=False).index[0])


def build_similarity_graph_from_edges(nodes: Iterable[dict[str, Any]], edges: Iterable[tuple[str, str, float]]) -> nx.Graph:
    graph = nx.Graph()
    for node in nodes:
        node_id = str(node["id"])
        attrs = {key: value for key, value in node.items() if key != "id"}
        graph.add_node(node_id, **attrs)
    for left, right, weight in edges:
        graph.add_edge(str(left), str(right), weight=float(weight))
    return graph


def markdown_program_summary(refined_map: pd.DataFrame) -> str:
    summary = (
        refined_map.groupby(["refined_program_id", "refined_program_name"], observed=False)["pair_key"]
        .count()
        .rename("n_pairs")
        .reset_index()
        .sort_values("n_pairs", ascending=False)
    )
    return markdown_table(summary)
