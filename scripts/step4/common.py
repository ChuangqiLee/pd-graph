#!/usr/bin/env python3.12
"""Shared helpers for the Step 4 external validation pipeline."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

from scripts.step2.common import (
    PROJECT_ROOT,
    ensure_dir,
    markdown_table,
    normalize_text,
    project_path,
    write_markdown,
)
from scripts.step3.common import compute_sha256, gene_family_token, parse_gene_string, program_validation_gene_panel


STEP4_ROOT = PROJECT_ROOT / "data/processed/step4"
STEP4_INPUTS = STEP4_ROOT / "00_inputs"
STEP4_REGISTRY = STEP4_ROOT / "01_registry"
STEP4_GENETICS = STEP4_ROOT / "02_genetics"
STEP4_SPATIAL = STEP4_ROOT / "03_spatial"
STEP4_INTERNAL = STEP4_ROOT / "04_internal_robustness"
STEP4_INTEGRATION = STEP4_ROOT / "05_integration"
STEP4_REPORTS = STEP4_ROOT / "06_reports"

STEP3_HANDOFF_DIR = PROJECT_ROOT / "data/processed/step3/11_handoff"
STEP3_STATE_DIR = PROJECT_ROOT / "data/processed/step3/05_states"
STEP3_PROGRAM_MAP = PROJECT_ROOT / "data/processed/step3/02_programs/step3_refined_program_map.csv"
STEP3_TOP10 = PROJECT_ROOT / "data/processed/step3/09_top10/step3_top10_refined_programs.csv"
STEP3_TASK_SUPPORT = PROJECT_ROOT / "data/processed/step3/04_task_support/step3_task_support_matrix.csv"
STEP3_STATE_SUPPORT = PROJECT_ROOT / "data/processed/step3/05_states/step3_candidate_state_support.csv"
STEP3_EXEMPLARS = PROJECT_ROOT / "data/processed/step3/06_exemplars/step3_exemplar_edges.csv"
STEP2_COMM_PANEL = PROJECT_ROOT / "data/processed/step2/01_comm_panel/pd_definitive_step2_comm_panel.h5ad"
STEP2_PAIR_UNIVERSE = PROJECT_ROOT / "data/processed/step2/03_lr_universe/step2_lr_universe_compact.parquet"
STEP2_SUMMARY = PROJECT_ROOT / "step2_summary_dossier.md"

PROGRAM_BATCH = {
    "glutamatergic_ionotropic_receptor_stress": "batch_1_full",
    "glutamatergic_metabotropic_receptor_stress": "batch_1_full",
    "apoe_lipoprotein_receptor_signaling": "batch_1_full",
    "app_centered_stress_signaling": "batch_1_full",
    "adhesion_organizer_network": "batch_2_strict",
    "immune_neuronal_il18_interface": "batch_1_full",
    "slit_robo_guidance": "batch_2_strict",
    "slit_robo_guidance_canonical": "batch_2_strict",
    "slit_robo_guidance_context": "batch_2_strict",
    "dscam_dcc_contact_recognition": "resolve_first",
}

SHARED_GENE_CLUSTER = {
    "glutamatergic_ionotropic_receptor_stress": "glutamate_superfamily",
    "glutamatergic_metabotropic_receptor_stress": "glutamate_superfamily",
    "apoe_lipoprotein_receptor_signaling": "apoe_app_superfamily",
    "app_centered_stress_signaling": "apoe_app_superfamily",
    "immune_neuronal_il18_interface": "immune_interface_overlap",
    "adhesion_organizer_network": "adhesion_guidance_superfamily",
    "slit_robo_guidance": "adhesion_guidance_superfamily",
    "slit_robo_guidance_canonical": "adhesion_guidance_superfamily",
    "slit_robo_guidance_context": "adhesion_guidance_superfamily",
    "dscam_dcc_contact_recognition": "adhesion_guidance_superfamily",
}

PROGRAM_NOTES = {
    "glutamatergic_ionotropic_receptor_stress": "Needs explicit count-linked sensitivity because the Step 3 handoff carries cell_count_link.",
    "glutamatergic_metabotropic_receptor_stress": "Must remain separable from the ionotropic program at receptor and direction level.",
    "apoe_lipoprotein_receptor_signaling": "Track overlap with APP-centered signaling to avoid double-counting shared loci.",
    "app_centered_stress_signaling": "Requires explicit source and region sensitivity reruns because Step 3 flagged source_flip and region_flip.",
    "adhesion_organizer_network": "Promotion threshold is strict because Step 3 negative_holdout remains unresolved internally.",
    "immune_neuronal_il18_interface": "Use module-level readouts because IL18-like ligands can be sparsely detected in spatial data.",
    "slit_robo_guidance": "Promotion threshold is strict because Step 3 negative_holdout remains unresolved internally.",
    "slit_robo_guidance_canonical": "Canonical SLIT-ROBO direct-pair branch split from the parent guidance program for holdout-specific evaluation.",
    "slit_robo_guidance_context": "Exploratory guidance-context branch separated from canonical SLIT-ROBO direct pairs; interpret conservatively.",
    "dscam_dcc_contact_recognition": "Proceed only after the task-axis inconsistency is resolved in Phase 0.",
}

STEP4_DIRECTION_OVERRIDES: dict[str, str] = {}

STEP4_PROGRAM_SPLITS = {
    "slit_robo_guidance": [
        {
            "program_id": "slit_robo_guidance_canonical",
            "program_name": "SLIT-ROBO direct guidance",
            "anchor_pairs": ["SLIT2->ROBO2", "SLIT2->ROBO1"],
            "expected_disease_direction": "pd_specific_loss",
            "remove_caveats": ["negative_holdout"],
            "add_caveats": ["split_canonical_branch"],
            "validation_program_id": "slit_robo_guidance",
        },
        {
            "program_id": "slit_robo_guidance_context",
            "program_name": "ROBO guidance context organizers",
            "anchor_pairs": ["OLFM2->ROBO2", "NCAM1->ROBO1"],
            "expected_disease_direction": "pd_specific_gain",
            "remove_caveats": ["negative_holdout"],
            "add_caveats": ["exploratory_context_branch"],
            "validation_program_id": "slit_robo_guidance",
        },
    ],
}

STEP4_INTERNAL_DIRECTION_OVERRIDES = {
    "apoe_lipoprotein_receptor_signaling": "pd_specific_gain",
}

ANCHOR_SEMANTICS: dict[str, dict[str, str]] = {
    "SLC1A3|GLS->GRIA4": {
        "anchor_class": "sender_module_to_receptor",
        "claim_language_level": "anchor_module_only",
        "rationale": "A sender glutamate-handling module projected onto an ionotropic receptor target.",
    },
    "SLC1A3|GLS->GRIA1": {
        "anchor_class": "sender_module_to_receptor",
        "claim_language_level": "anchor_module_only",
        "rationale": "A sender glutamate-handling module projected onto an ionotropic receptor target.",
    },
    "SLC1A3|GLS->GRIK2": {
        "anchor_class": "sender_module_to_receptor",
        "claim_language_level": "anchor_module_only",
        "rationale": "A sender glutamate-handling module projected onto an ionotropic receptor target.",
    },
    "SLC1A3|GLS->GRIA2": {
        "anchor_class": "sender_module_to_receptor",
        "claim_language_level": "anchor_module_only",
        "rationale": "A sender glutamate-handling module projected onto an ionotropic receptor target.",
    },
    "SLC1A3|GLS->GRIA3": {
        "anchor_class": "sender_module_to_receptor",
        "claim_language_level": "anchor_module_only",
        "rationale": "A sender glutamate-handling module projected onto an ionotropic receptor target.",
    },
    "SLC1A3|GLS->GRM7": {
        "anchor_class": "sender_module_to_receptor",
        "claim_language_level": "anchor_module_only",
        "rationale": "A sender glutamate-handling module projected onto a metabotropic receptor target.",
    },
    "SLC1A3|GLS->GRM8": {
        "anchor_class": "sender_module_to_receptor",
        "claim_language_level": "anchor_module_only",
        "rationale": "A sender glutamate-handling module projected onto a metabotropic receptor target.",
    },
    "SLC1A3|GLS->GRM5": {
        "anchor_class": "sender_module_to_receptor",
        "claim_language_level": "anchor_module_only",
        "rationale": "A sender glutamate-handling module projected onto a metabotropic receptor target.",
    },
    "SLC1A3|GLS->GRM3": {
        "anchor_class": "sender_module_to_receptor",
        "claim_language_level": "anchor_module_only",
        "rationale": "A sender glutamate-handling module projected onto a metabotropic receptor target.",
    },
    "SLC1A3|GLS->GRM1": {
        "anchor_class": "sender_module_to_receptor",
        "claim_language_level": "anchor_module_only",
        "rationale": "A sender glutamate-handling module projected onto a metabotropic receptor target.",
    },
    "APOE->LRP8": {
        "anchor_class": "direct_pair",
        "claim_language_level": "direct_pair_allowed",
        "rationale": "Canonical APOE-lipoprotein receptor interaction.",
    },
    "APP->LRP1": {
        "anchor_class": "network_processing_anchor",
        "claim_language_level": "anchor_module_only",
        "rationale": "Biologically linked, but more consistent with receptor-processing or trafficking language than a clean extracellular ligand claim.",
    },
    "APOE->LRP1": {
        "anchor_class": "direct_pair",
        "claim_language_level": "direct_pair_allowed",
        "rationale": "Canonical APOE-lipoprotein receptor interaction.",
    },
    "APOE->LRP2": {
        "anchor_class": "direct_pair",
        "claim_language_level": "direct_pair_allowed",
        "rationale": "Canonical APOE-lipoprotein receptor interaction.",
    },
    "APOE->APP": {
        "anchor_class": "network_processing_anchor",
        "claim_language_level": "program_only",
        "rationale": "Represents pathway coupling around APP rather than a direct extracellular ligand-receptor claim.",
    },
    "APOE->SCARB1": {
        "anchor_class": "direct_pair",
        "claim_language_level": "direct_pair_allowed",
        "rationale": "APOE and scavenger receptor biology supports a direct receptor-style interpretation.",
    },
    "APOE->VLDLR": {
        "anchor_class": "direct_pair",
        "claim_language_level": "direct_pair_allowed",
        "rationale": "Canonical APOE-lipoprotein receptor interaction.",
    },
    "APOE->SORL1": {
        "anchor_class": "network_processing_anchor",
        "claim_language_level": "anchor_module_only",
        "rationale": "SORL1 is best treated as a trafficking and receptor-processing context rather than a clean surface receptor endpoint.",
    },
    "BACE2->APP": {
        "anchor_class": "network_processing_anchor",
        "claim_language_level": "program_only",
        "rationale": "Represents APP processing biology rather than a direct cell-to-cell ligand-receptor claim.",
    },
    "CADM1->CADM1": {
        "anchor_class": "direct_pair",
        "claim_language_level": "direct_pair_allowed",
        "rationale": "A homophilic adhesion interaction with defensible direct-pair wording.",
    },
    "TENM4->ADGRL2": {
        "anchor_class": "direct_pair",
        "claim_language_level": "direct_pair_allowed",
        "rationale": "Latrophilin-teneurin biology supports direct contact interpretation.",
    },
    "CHL1->CNTN1": {
        "anchor_class": "uncertain_anchor",
        "claim_language_level": "anchor_module_only",
        "rationale": "Adhesion-family coupling is plausible, but Step 4 should stay at module level for wording.",
    },
    "RTN4->CNTN1": {
        "anchor_class": "uncertain_anchor",
        "claim_language_level": "anchor_module_only",
        "rationale": "The contact-guidance interpretation is plausible but not strong enough for a direct pair claim here.",
    },
    "ATP1B1->NEGR1": {
        "anchor_class": "uncertain_anchor",
        "claim_language_level": "anchor_module_only",
        "rationale": "Treat as a contact-organizer module rather than a definitive biochemical pair.",
    },
    "IL18->IL1RAPL1": {
        "anchor_class": "uncertain_anchor",
        "claim_language_level": "program_only",
        "rationale": "This does not map onto a canonical IL18 receptor axis, so the final claim should remain at program level.",
    },
    "EDIL3->ITGB1": {
        "anchor_class": "uncertain_anchor",
        "claim_language_level": "anchor_module_only",
        "rationale": "Integrin-related adhesion biology is plausible, but the pair should stay at anchor-module wording.",
    },
    "TGFB1->ITGB1": {
        "anchor_class": "uncertain_anchor",
        "claim_language_level": "anchor_module_only",
        "rationale": "Represents a signaling-adhesion interface rather than a clean receptor-ligand pair in this pipeline.",
    },
    "APP->TNFRSF21": {
        "anchor_class": "network_processing_anchor",
        "claim_language_level": "program_only",
        "rationale": "Treat as network coupling only, not a direct cell-contact claim.",
    },
    "RTN4->TNFRSF19": {
        "anchor_class": "uncertain_anchor",
        "claim_language_level": "anchor_module_only",
        "rationale": "Plausible guidance-related coupling, but keep wording at anchor-module level.",
    },
    "SLIT2->ROBO2": {
        "anchor_class": "direct_pair",
        "claim_language_level": "direct_pair_allowed",
        "rationale": "Canonical guidance ligand-receptor pair.",
    },
    "OLFM2->ROBO2": {
        "anchor_class": "uncertain_anchor",
        "claim_language_level": "anchor_module_only",
        "rationale": "Associated with the same guidance neighborhood, but the pair claim is less canonical than SLIT-ROBO.",
    },
    "SLIT2->ROBO1": {
        "anchor_class": "direct_pair",
        "claim_language_level": "direct_pair_allowed",
        "rationale": "Canonical guidance ligand-receptor pair.",
    },
    "NCAM1->ROBO1": {
        "anchor_class": "uncertain_anchor",
        "claim_language_level": "anchor_module_only",
        "rationale": "Best treated as a guidance-organizer context rather than a clean direct ligand-receptor claim.",
    },
    "DSCAM->DCC": {
        "anchor_class": "uncertain_anchor",
        "claim_language_level": "anchor_module_only",
        "rationale": "Cell-contact biology is plausible, but the direct biochemical pair wording should remain conservative.",
    },
    "DSCAM->PTPRM": {
        "anchor_class": "uncertain_anchor",
        "claim_language_level": "anchor_module_only",
        "rationale": "Cell-contact biology is plausible, but the direct biochemical pair wording should remain conservative.",
    },
    "DSCAM->PTPRK": {
        "anchor_class": "uncertain_anchor",
        "claim_language_level": "anchor_module_only",
        "rationale": "Cell-contact biology is plausible, but the direct biochemical pair wording should remain conservative.",
    },
    "DSCAM->DSCAM": {
        "anchor_class": "direct_pair",
        "claim_language_level": "direct_pair_allowed",
        "rationale": "A homophilic contact-recognition pair with defensible direct-pair wording.",
    },
    "DSCAM->PTPRT": {
        "anchor_class": "uncertain_anchor",
        "claim_language_level": "anchor_module_only",
        "rationale": "Cell-contact biology is plausible, but the direct biochemical pair wording should remain conservative.",
    },
}

PRIMARY_EQTL_TISSUES = [
    "Brain_Substantia_nigra",
    "Brain_Caudate_basal_ganglia",
    "Brain_Putamen_basal_ganglia",
    "Brain_Frontal_Cortex_BA9",
    "Brain_Cortex",
]

PRIMARY_STATE_IDS = [
    "microglia_state_03_stress_response",
    "microglia_state_02_homeostatic",
    "dopaminergic_neuron_state_01_synaptic_organizer",
    "dopaminergic_neuron_state_02_synaptic_organizer",
]

STEP3_TO_STEP4_SPATIAL_COLUMNS = (
    "refined_program_id",
    "refined_program_name",
    "anchor_genes",
    "sender_state_id",
    "sender_state_markers",
    "receiver_state_id",
    "receiver_state_markers",
    "expected_disease_direction",
    "expected_spatial_relationship",
    "caveat_flags",
    "validation_tier",
)

STEP3_TO_STEP4_GENE_COLUMNS = (
    "refined_program_id",
    "refined_program_name",
    "anchor_pairs",
    "anchor_genes",
    "gene_set_members",
    "expected_disease_direction",
    "expected_mr_gwas_members",
    "sender_state_markers",
    "receiver_state_markers",
    "caveat_flags",
    "validation_tier",
    "step4_validation_priority",
)

STEP4_NORMALIZED_HANDOFF_KEYS = (
    "program_id",
    "program_name",
    "candidate_id",
    "program_batch",
    "gene_set_members_list",
    "sender_state_id",
    "receiver_state_id",
    "sender_state_marker_list",
    "receiver_state_marker_list",
    "caveat_flag_list",
    "expected_pd_direction",
    "expected_internal_direction",
    "expected_spatial_relationship",
    "expected_spatial_task",
    "dominant_task",
    "validation_tier",
    "anchor_pairs_json",
    "sender_state_frequency",
    "receiver_state_frequency",
    "task_conflict_status",
    "dscam_resolution_status",
    "notes",
)

STEP4_REGISTRY_COLUMNS = (
    "program_id",
    "program_name",
    "program_batch",
    "anchor_pairs_json",
    "gene_members_json",
    "sender_state_id",
    "receiver_state_id",
    "expected_pd_direction",
    "anchor_semantics_profile",
    "caveat_flags_json",
    "gene_count",
    "anchor_pair_count",
    "shared_gene_cluster_id",
    "requires_counts_adjustment",
    "requires_source_sensitivity",
    "requires_region_sensitivity",
    "requires_strict_replication",
    "blocking_issue_flag",
    "dscam_resolution_status",
    "notes",
)

STEP4_INTERNAL_ANNEX_COLUMNS = (
    "program_id",
    "program_name",
    "n_anchor_pairs_found_in_holdout",
    "mean_holdout_pd_minus_control",
    "holdout_direction_match",
    "holdout_support_tier",
    "holdout_weighted_match_fraction",
    "n_holdout_sources_available",
    "n_holdout_sources_direction_match",
    "gse157783_mean_holdout_pd_minus_control",
    "gse157783_direction_match",
    "gse253462_mean_holdout_pd_minus_control",
    "gse253462_direction_match",
    "expected_pd_direction",
    "expected_internal_direction",
)

STEP4_INTERNAL_SOURCE_ANNEX_COLUMNS = (
    "program_id",
    "program_name",
    "expected_pd_direction",
    "expected_internal_direction",
    "holdout_source_id",
    "mean_holdout_pd_minus_control",
    "n_anchor_pairs_found_in_holdout",
    "direction_match",
)

STEP4_FINAL_MATRIX_COLUMNS = (
    "program_id",
    "program_name",
    "primary_spatial_support_bin",
    "secondary_spatial_support_bin",
    "program_genetic_support_bin",
    "claim_tier",
    "promotion_decision",
    "spatial_primary_gate",
    "spatial_replication_gate",
    "genetic_consensus_gate",
    "internal_stability_gate",
    "final_step4_score",
    "final_evidence_class",
    "decision_rationale",
)

STEP4_DECISION_LOG_COLUMNS = (
    "program_id",
    "program_name",
    "spatial_primary_gate",
    "spatial_replication_gate",
    "genetic_gene_level_gate",
    "genetic_program_level_gate",
    "internal_stability_gate",
    "claim_tier",
    "promotion_decision",
    "final_step4_score",
    "final_evidence_class",
    "decision_rationale",
)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=False)


def write_json_lines(path_like: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    path = project_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(list(rows), indent=2, ensure_ascii=False) + "\n")
    return path


def read_json(path_like: str | Path) -> Any:
    path = project_path(path_like)
    return json.loads(path.read_text(encoding="utf-8"))


def require_columns(frame: pd.DataFrame, required_columns: Iterable[str], *, context: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        available = ", ".join(str(column) for column in frame.columns.tolist())
        raise ValueError(f"{context} is missing required columns: {', '.join(missing)}. Available columns: {available}")


def read_csv_required_columns(path_like: str | Path, required_columns: Iterable[str], *, context: str) -> pd.DataFrame:
    frame = pd.read_csv(project_path(path_like))
    require_columns(frame, required_columns, context=context)
    return frame


def require_record_keys(record: dict[str, Any], required_keys: Iterable[str], *, context: str) -> None:
    missing = [key for key in required_keys if key not in record]
    if missing:
        available = ", ".join(str(key) for key in sorted(record))
        raise ValueError(f"{context} is missing required keys: {', '.join(missing)}. Available keys: {available}")


def require_record_list(records: Iterable[dict[str, Any]], required_keys: Iterable[str], *, context: str) -> None:
    for index, record in enumerate(records):
        require_record_keys(record, required_keys, context=f"{context}[{index}]")


def step4_path(*parts: str) -> Path:
    return STEP4_ROOT.joinpath(*parts)


def ensure_step4_dir(*parts: str) -> Path:
    path = step4_path(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def log(message: str) -> None:
    print(message, flush=True)


def list_from_pipe(value: Any) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    return [item.strip() for item in text.split("|") if item.strip()]


def pipe_join(values: Iterable[str]) -> str:
    return "|".join([normalize_text(value) for value in values if normalize_text(value)])


def apply_step4_program_derivations(merged: pd.DataFrame) -> pd.DataFrame:
    if merged.empty:
        return merged

    derived_rows: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        parent_program_id = str(row["refined_program_id"])
        if parent_program_id in STEP4_PROGRAM_SPLITS:
            sender_markers = str(row.get("sender_state_markers_gene", ""))
            receiver_markers = str(row.get("receiver_state_markers_gene", ""))
            parent_caveats = list_from_pipe(row.get("caveat_flags", ""))
            for spec in STEP4_PROGRAM_SPLITS[parent_program_id]:
                child = dict(row)
                child["refined_program_id"] = spec["program_id"]
                child["refined_program_name"] = spec["program_name"]
                child["anchor_pairs"] = pipe_join(spec["anchor_pairs"])
                child["anchor_genes"] = pipe_join(parse_gene_string(pipe_join(spec["anchor_pairs"]).replace("->", "|")))
                child["expected_disease_direction"] = spec["expected_disease_direction"]
                child["expected_disease_direction_spatial"] = spec["expected_disease_direction"]
                child["expected_internal_direction"] = spec.get("expected_internal_direction", spec["expected_disease_direction"])
                child_caveats = [flag for flag in parent_caveats if flag not in set(spec.get("remove_caveats", []))]
                child_caveats.extend(spec.get("add_caveats", []))
                child["caveat_flags"] = pipe_join(child_caveats)
                child["gene_set_members"] = pipe_join(
                    program_validation_gene_panel(
                        str(spec.get("validation_program_id", parent_program_id)),
                        spec["anchor_pairs"],
                        sender_state_markers=sender_markers,
                        receiver_state_markers=receiver_markers,
                        max_genes=10,
                    )
                )
                child["expected_mr_gwas_members"] = child["gene_set_members"]
                derived_rows.append(child)
            continue

        override = STEP4_DIRECTION_OVERRIDES.get(parent_program_id)
        if override:
            row["expected_disease_direction"] = override
            row["expected_disease_direction_spatial"] = override
        row["expected_internal_direction"] = STEP4_INTERNAL_DIRECTION_OVERRIDES.get(parent_program_id, row["expected_disease_direction"])
        derived_rows.append(row)

    derived = pd.DataFrame(derived_rows)
    return derived.sort_values("step3_program_score", ascending=False).reset_index(drop=True)


def lineage_from_state_id(state_id: str) -> str:
    text = normalize_text(state_id)
    if text.startswith("microglia_"):
        return "microglia"
    if text.startswith("dopaminergic_neuron_"):
        return "dopaminergic neuron"
    if text.startswith("astrocyte_"):
        return "astrocyte"
    if text.startswith("endothelial_cell_"):
        return "endothelial cell"
    if text.startswith("neuron_"):
        return "neuron"
    return "unknown"


def anchor_count_from_raw(raw_anchor_pairs: Any) -> int:
    return max(str(raw_anchor_pairs).count("->"), 0)


def load_step4_handoff_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gene = read_csv_required_columns(
        STEP3_HANDOFF_DIR / "step3_to_step4_gene_set_targets.csv",
        STEP3_TO_STEP4_GENE_COLUMNS,
        context="Step 3 to Step 4 gene handoff",
    )
    spatial = read_csv_required_columns(
        STEP3_HANDOFF_DIR / "step3_to_step4_spatial_targets.csv",
        STEP3_TO_STEP4_SPATIAL_COLUMNS,
        context="Step 3 to Step 4 spatial handoff",
    )
    top10 = read_csv_required_columns(
        STEP3_TOP10,
        (
            "refined_program_id",
            "candidate_id",
            "step3_program_score",
            "dominant_task",
            "anchor_pairs",
            "dominant_sender_state",
            "dominant_receiver_state",
            "step4_validation_priority",
        ),
        context="Step 3 top10 programs",
    )
    top10 = top10.loc[top10["refined_program_id"].isin(gene["refined_program_id"])].copy()
    return gene, spatial, top10


def load_step4_program_frame() -> pd.DataFrame:
    gene, spatial, top10 = load_step4_handoff_tables()
    merged = gene.merge(
        spatial.drop(columns=["refined_program_name", "caveat_flags", "validation_tier"]),
        on="refined_program_id",
        how="inner",
    ).merge(
        top10[
            [
                "refined_program_id",
                "candidate_id",
                "step3_program_score",
                "dominant_task",
                "anchor_pairs",
                "dominant_sender_state",
                "dominant_receiver_state",
                "step4_validation_priority",
            ]
        ],
        on="refined_program_id",
        how="left",
        suffixes=("", "_top10"),
    )
    merged = merged.rename(
        columns={
            "expected_disease_direction_x": "expected_disease_direction",
            "expected_disease_direction_y": "expected_disease_direction_spatial",
            "sender_state_markers_x": "sender_state_markers_gene",
            "sender_state_markers_y": "sender_state_markers_spatial",
            "receiver_state_markers_x": "receiver_state_markers_gene",
            "receiver_state_markers_y": "receiver_state_markers_spatial",
        }
    )
    return apply_step4_program_derivations(merged)


def load_top_anchor_pairs() -> dict[str, list[str]]:
    merged = load_step4_program_frame()
    pair_lookup: dict[str, list[str]] = {}
    for row in merged.itertuples(index=False):
        pair_lookup[str(row.refined_program_id)] = list_from_pipe(getattr(row, "anchor_pairs", ""))
    return pair_lookup


def build_anchor_records() -> list[dict[str, Any]]:
    merged = load_step4_program_frame()
    anchor_lookup = load_top_anchor_pairs()
    records: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        for order_index, pair_key in enumerate(anchor_lookup.get(str(row.refined_program_id), []), start=1):
            ligand_text, receptor_text = pair_key.split("->", maxsplit=1)
            semantics = ANCHOR_SEMANTICS.get(pair_key)
            if semantics is None:
                sender_families = {gene_family_token(gene) for gene in parse_gene_string(ligand_text)}
                receiver_families = {gene_family_token(gene) for gene in parse_gene_string(receptor_text)}
                if receiver_families.intersection({"GRIA", "GRIK", "GRIN", "GRM", "ROBO", "ERBB", "LRP", "IL1RAPL", "TNFRSF"}):
                    semantics = {
                        "anchor_class": "sender_module_to_receptor",
                        "claim_language_level": "anchor_module_only",
                        "rationale": "Automatically inferred from the ligand-module to receptor structure of the updated Step 3 handoff.",
                    }
                elif receiver_families.intersection({"ITGA", "ITGB", "DCC", "DSCAM", "PTPR", "CADM", "CNTN"}):
                    semantics = {
                        "anchor_class": "adhesion_or_contact_anchor",
                        "claim_language_level": "anchor_pair_supported",
                        "rationale": "Automatically inferred for an adhesion or contact-style anchor pair promoted by the updated Step 3 handoff.",
                    }
                else:
                    semantics = {
                        "anchor_class": "pair_level_candidate",
                        "claim_language_level": "anchor_pair_supported",
                        "rationale": "Automatically inferred fallback semantics for an anchor pair that is not yet listed in the explicit Step 4 semantics table.",
                    }
            records.append(
                {
                    "program_id": row.refined_program_id,
                    "program_name": row.refined_program_name,
                    "pair_key": pair_key,
                    "anchor_order": order_index,
                    "sender_genes": parse_gene_string(ligand_text),
                    "receiver_genes": parse_gene_string(receptor_text),
                    "anchor_class": semantics["anchor_class"],
                    "claim_language_level": semantics["claim_language_level"],
                    "rationale": semantics["rationale"],
                }
            )
    return records


def resolve_dscam_status() -> tuple[str, str]:
    task_support = pd.read_csv(STEP3_TASK_SUPPORT)
    state_support = pd.read_csv(STEP3_STATE_SUPPORT)
    exemplars = pd.read_csv(STEP3_EXEMPLARS) if STEP3_EXEMPLARS.exists() else pd.DataFrame()
    task_row = task_support.loc[task_support["candidate_id"].eq("program::dscam_dcc_contact_recognition")].iloc[0]
    state_row = (
        state_support.loc[
            state_support["candidate_id"].eq("program::dscam_dcc_contact_recognition")
            & state_support["state_pair_rank"].eq(1)
        ]
        .iloc[0]
    )
    exemplar_tasks = sorted(
        exemplars.loc[exemplars["refined_program_id"].eq("dscam_dcc_contact_recognition"), "task_name"]
        .astype(str)
        .unique()
        .tolist()
    )
    microglia_support = float(task_row["microglia_to_dopaminergic_neuron"])
    endothelial_support = float(task_row["endothelial_to_dopaminergic_neuron"])
    if (
        str(task_row["primary_task"]) == "microglia_to_dopaminergic_neuron"
        and str(state_row["sender_state_id"]).startswith("microglia_")
        and microglia_support > endothelial_support
    ):
        explanation = (
            "Resolved to microglia_to_dopaminergic_neuron_confirmed because the Step 3 program-level task matrix "
            f"favours the microglial axis ({microglia_support:.3f} vs {endothelial_support:.3f}), the top state pair "
            f"is {state_row['sender_state_id']} -> {state_row['receiver_state_id']}, and exemplar edges map to "
            f"{', '.join(exemplar_tasks) if exemplar_tasks else 'the same microglial task context'}."
        )
        return "microglia_to_dopaminergic_neuron_confirmed", explanation
    if (
        str(task_row["primary_task"]) == "endothelial_to_dopaminergic_neuron"
        and str(state_row["sender_state_id"]).startswith("endothelial_")
    ):
        explanation = (
            "Resolved to endothelial_to_dopaminergic_neuron_confirmed because both the Step 3 task matrix and the "
            "state-pair provenance point to the endothelial sender axis."
        )
        return "endothelial_to_dopaminergic_neuron_confirmed", explanation
    explanation = (
        "Kept unresolved because the Step 3 dominant task and the dominant sender-state lineage do not align cleanly."
    )
    return "unresolved_keep_program_level_only", explanation


def expected_spatial_task(expected_relationship: str) -> str:
    text = normalize_text(expected_relationship)
    if " in " not in text:
        return ""
    return text.rsplit(" in ", maxsplit=1)[-1]


def bh_fdr(values: Iterable[float]) -> np.ndarray:
    pvals = pd.to_numeric(pd.Series(list(values), dtype=float), errors="coerce").fillna(1.0).clip(lower=0.0, upper=1.0)
    array = pvals.to_numpy(dtype=float)
    if array.size == 0:
        return np.asarray([], dtype=float)

    order = np.argsort(array, kind="mergesort")
    ranked = array[order]
    scale = float(array.size) / np.arange(1, array.size + 1, dtype=float)
    adjusted = np.minimum.accumulate((ranked * scale)[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    return restored


def safe_log2fc(mean_in: float, mean_out: float, pseudocount: float = 1e-6) -> float:
    return float(np.log2((float(mean_in) + pseudocount) / (float(mean_out) + pseudocount)))


def maybe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def zscore_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(skipna=True, ddof=0)
    if pd.isna(std) or float(std) == 0.0:
        return pd.Series(np.zeros(len(values), dtype=float), index=series.index)
    return (values - float(values.mean(skipna=True))) / float(std)


def geometric_mean(values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.maximum(arr, eps)
    return np.exp(np.mean(np.log(arr), axis=0))


def program_gene_memberships() -> pd.DataFrame:
    frame = load_step4_program_frame()
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        for gene in list_from_pipe(row.gene_set_members):
            rows.append(
                {
                    "program_id": row.refined_program_id,
                    "program_name": row.refined_program_name,
                    "gene_symbol": gene,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["program_id", "program_name", "gene_symbol"])
    return pd.DataFrame(rows)


def retry_get(url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None, timeout: int = 60, max_attempts: int = 4) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as exc:  # pragma: no cover - defensive retry wrapper
            last_error = exc
            if attempt == max_attempts:
                raise
            time.sleep(1.5 * attempt)
    if last_error is not None:  # pragma: no cover
        raise last_error
    raise RuntimeError(f"Failed GET {url}")


def download_file(url: str, destination: str | Path, *, chunk_size: int = 1 << 20, overwrite: bool = False) -> Path:
    path = project_path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0 and not overwrite:
        return path
    response = retry_get(url, timeout=120)
    with path.open("wb") as handle:
        handle.write(response.content)
    return path


def stream_download_file(url: str, destination: str | Path, *, chunk_size: int = 1 << 20, overwrite: bool = False) -> Path:
    path = project_path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0 and not overwrite:
        return path
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                handle.write(chunk)
    return path


def file_record(path_like: str | Path, *, source_phase: str, input_class: str) -> dict[str, Any]:
    path = project_path(path_like)
    return {
        "relative_path": str(path.relative_to(PROJECT_ROOT)),
        "absolute_path": str(path),
        "source_phase": source_phase,
        "input_class": input_class,
        "size_bytes": path.stat().st_size,
        "sha256": compute_sha256(path),
    }


def render_markdown_summary(title: str, frame: pd.DataFrame, *, intro: str | None = None) -> str:
    lines = [f"# {title}", ""]
    if intro:
        lines.extend([intro.rstrip(), ""])
    lines.append(markdown_table(frame.fillna("NA")))
    return "\n".join(lines).rstrip() + "\n"
