#!/usr/bin/env python3.12
"""Integrate the Step 4 genetic and spatial evidence into one program matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import markdown_table, write_markdown
from scripts.step4.common import (
    STEP4_DECISION_LOG_COLUMNS,
    STEP4_FINAL_MATRIX_COLUMNS,
    STEP4_INTERNAL_ANNEX_COLUMNS,
    STEP4_REGISTRY_COLUMNS,
    ensure_step4_dir,
    read_csv_required_columns,
    require_columns,
)


DEFAULT_OUTPUT_DIR = Path("data/processed/step4/05_integration")

PRIMARY_COMPONENT = {
    "strong_primary_spatial_support": 1.0,
    "moderate_primary_spatial_support": 0.7,
    "weak_primary_spatial_support": 0.45,
    "no_primary_spatial_support": 0.0,
}
SECONDARY_COMPONENT = {
    "moderate_secondary_spatial_support": 0.7,
    "weak_secondary_spatial_support": 0.35,
    "no_secondary_spatial_support": 0.0,
}
GENE_COMPONENT = {
    "strong_genetic_support": 1.0,
    "moderate_genetic_support": 0.6,
    "weak_or_inconclusive": 0.15,
    "non_instrumentable": 0.0,
}
PROGRAM_GENETIC_COMPONENT = {
    "strong_program_genetic_support": 1.0,
    "moderate_program_genetic_support": 0.6,
    "weak_program_genetic_support": 0.15,
}
PRIMARY_GATE = {
    "strong_primary_spatial_support": "pass",
    "moderate_primary_spatial_support": "pass",
    "weak_primary_spatial_support": "suggestive",
    "no_primary_spatial_support": "fail",
}
SECONDARY_GATE = {
    "moderate_secondary_spatial_support": "pass",
    "weak_secondary_spatial_support": "suggestive",
    "no_secondary_spatial_support": "fail",
}
PROGRAM_GENETIC_GATE = {
    "strong_program_genetic_support": "pass",
    "moderate_program_genetic_support": "suggestive",
    "weak_program_genetic_support": "fail",
}
PENALTIES = {
    "source_dominant": 0.05,
    "negative_holdout": 0.08,
    "source_flip": 0.08,
    "region_flip": 0.08,
    "cell_count_link": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrate Step 4 evidence.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def gene_level_gate(row: pd.Series) -> str:
    strong_raw = row.get("strong_genetic_support", 0)
    moderate_raw = row.get("moderate_genetic_support", 0)
    strong_count = 0 if pd.isna(strong_raw) else int(strong_raw)
    moderate_count = 0 if pd.isna(moderate_raw) else int(moderate_raw)
    gene_component = float(row.get("gene_level_component", 0.0) or 0.0)
    if strong_count >= 1 or gene_component >= 0.7:
        return "pass"
    if moderate_count >= 1 or gene_component >= 0.25:
        return "suggestive"
    return "fail"


def consensus_gate(row: pd.Series) -> str:
    if row["genetic_gene_level_gate"] == "pass" or row["genetic_program_level_gate"] == "pass":
        return "pass"
    if row["genetic_gene_level_gate"] == "suggestive" or row["genetic_program_level_gate"] == "suggestive":
        return "suggestive"
    return "fail"


def spatial_gate(row: pd.Series) -> str:
    primary_gate = row["spatial_primary_gate"]
    replication_gate = row["spatial_replication_gate"]
    if primary_gate == "pass" and replication_gate == "pass":
        return "pass"
    if primary_gate in {"pass", "suggestive"} and replication_gate in {"pass", "suggestive"}:
        return "suggestive"
    if primary_gate == "pass":
        return "suggestive"
    return "fail"


def classify_claim(row: pd.Series) -> tuple[str, str, str, str]:
    if bool(row["blocking_issue_flag"]):
        return (
            "blocked",
            "do_not_promote",
            "unresolved",
            "Blocking issue remains unresolved in the Step 4 registry.",
        )
    if row["internal_stability_gate"] == "fail":
        return (
            "needs_internal_resolution",
            "hold_for_rerun",
            "unresolved",
            "External evidence is not interpretable until the internal stability failures are resolved.",
        )
    if row["spatial_overall_gate"] == "pass" and row["genetic_consensus_gate"] == "pass":
        return (
            "external_mechanistic_candidate",
            "main_figure_candidate",
            "externally_supported",
            "Primary and replication spatial gates both pass, and the genetic layer also passes.",
        )
    if row["spatial_overall_gate"] in {"pass", "suggestive"} and row["genetic_consensus_gate"] in {"pass", "suggestive"}:
        return (
            "externally_supported_program",
            "priority_replication_candidate",
            "externally_supported",
            "Spatial evidence is at least suggestive across datasets and the genetic layer is also supportive.",
        )
    if row["spatial_overall_gate"] in {"pass", "suggestive"}:
        return (
            "spatial_support_only",
            "supplement_with_caveats",
            "spatially_supported_genetically_partial",
            "Spatial support is present, but the genetic layer remains partial.",
        )
    if row["genetic_consensus_gate"] in {"pass", "suggestive"}:
        return (
            "genetic_support_only",
            "rerun_spatial_validation",
            "genetically_supported_spatially_partial",
            "The genetic layer is supportive, but the spatial validation remains incomplete.",
        )
    return (
        "narrative_only",
        "do_not_promote",
        "narrative_only",
        "Neither the spatial nor the genetic gates are strong enough for external promotion.",
    )


def build_rationale(row: pd.Series, rationale: str) -> str:
    summary = (
        f"primary={row['spatial_primary_gate']}; replication={row['spatial_replication_gate']}; "
        f"gene={row['genetic_gene_level_gate']}; program_genetics={row['genetic_program_level_gate']}; "
        f"internal={row['internal_stability_gate']}"
    )
    return f"{rationale} Gate summary: {summary}."


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_step4_dir("05_integration")
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = read_csv_required_columns(
        PROJECT_ROOT / "data/processed/step4/01_registry/step4_validation_registry.csv",
        STEP4_REGISTRY_COLUMNS,
        context="Step 4 validation registry",
    )
    if registry.empty:
        empty = pd.DataFrame(
            columns=STEP4_FINAL_MATRIX_COLUMNS,
        )
        empty.to_csv(output_dir / "step4_final_evidence_matrix.csv", index=False)
        empty.to_csv(output_dir / "step4_candidate_decision_log.csv", index=False)
        lines = [
            "# Step 4 Validated Programs",
            "",
            "_No Step 4 programs were available after strict gating in this run._",
            "",
        ]
        write_markdown(output_dir / "step4_validated_programs.md", "\n".join(lines))
        return
    gene_summary = read_csv_required_columns(
        PROJECT_ROOT / "data/processed/step4/02_genetics/01_gene_level/gene_level_support_summary.csv",
        ("gene_symbol", "program_ids_json", "gene_support_bin"),
        context="Step 4 gene-level support summary",
    )
    program_genetic = read_csv_required_columns(
        PROJECT_ROOT / "data/processed/step4/02_genetics/02_program_level/program_genetic_evidence_matrix.csv",
        ("program_id", "program_genetic_support_bin", "n_strong_gene_support", "n_moderate_gene_support", "empirical_p_value", "empirical_fdr"),
        context="Step 4 program-level genetic evidence",
    )
    primary_spatial = read_csv_required_columns(
        PROJECT_ROOT / "data/processed/step4/03_spatial/04_primary_tests/program_primary_spatial_results.csv",
        (
            "program_id",
            "primary_spatial_support_bin",
            "direction_match",
            "snpc_adjacency_beta",
            "snpc_adjacency_p_value",
            "snpc_adjacency_fdr",
            "contrast_beta",
            "contrast_p_value",
            "contrast_fdr",
        ),
        context="Step 4 primary spatial results",
    )
    secondary_spatial = read_csv_required_columns(
        PROJECT_ROOT / "data/processed/step4/03_spatial/05_secondary_replication/secondary_spatial_support_summary.csv",
        ("program_id", "secondary_spatial_support_bin", "total_program_score", "mean_quadrant_score", "n_quadrants_direction_consistent"),
        context="Step 4 secondary spatial summary",
    )
    holdout_annex = read_csv_required_columns(
        PROJECT_ROOT / "data/processed/step4/04_internal_robustness/holdout_annex.csv",
        STEP4_INTERNAL_ANNEX_COLUMNS,
        context="Step 4 internal robustness annex",
    )

    gene_members = []
    for row in gene_summary.itertuples(index=False):
        for program_id in json.loads(row.program_ids_json):
            gene_members.append({"program_id": program_id, "gene_support_bin": row.gene_support_bin})
    gene_members = pd.DataFrame(gene_members)
    gene_program = (
        gene_members.groupby("program_id", observed=False)["gene_support_bin"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column in ["strong_genetic_support", "moderate_genetic_support", "weak_or_inconclusive", "non_instrumentable"]:
        if column not in gene_program.columns:
            gene_program[column] = 0
    gene_program["gene_level_component"] = (
        gene_program["strong_genetic_support"] * GENE_COMPONENT["strong_genetic_support"]
        + gene_program["moderate_genetic_support"] * GENE_COMPONENT["moderate_genetic_support"]
        + gene_program["weak_or_inconclusive"] * GENE_COMPONENT["weak_or_inconclusive"]
    ) / gene_program[
        ["strong_genetic_support", "moderate_genetic_support", "weak_or_inconclusive", "non_instrumentable"]
    ].sum(axis=1).replace(0, 1)

    merged = (
        registry.merge(
            primary_spatial[
                [
                    "program_id",
                    "primary_spatial_support_bin",
                    "direction_match",
                    "snpc_adjacency_beta",
                    "snpc_adjacency_p_value",
                    "snpc_adjacency_fdr",
                    "contrast_beta",
                    "contrast_p_value",
                    "contrast_fdr",
                ]
            ],
            on="program_id",
            how="left",
        )
        .merge(
            secondary_spatial[
                [
                    "program_id",
                    "secondary_spatial_support_bin",
                    "total_program_score",
                    "mean_quadrant_score",
                    "n_quadrants_direction_consistent",
                ]
            ],
            on="program_id",
            how="left",
        )
        .merge(
            program_genetic[
                [
                    "program_id",
                    "program_genetic_support_bin",
                    "n_strong_gene_support",
                    "n_moderate_gene_support",
                    "empirical_p_value",
                    "empirical_fdr",
                ]
            ],
            on="program_id",
            how="left",
        )
        .merge(
            gene_program[
                [
                    "program_id",
                    "gene_level_component",
                    "strong_genetic_support",
                    "moderate_genetic_support",
                    "weak_or_inconclusive",
                    "non_instrumentable",
                ]
            ],
            on="program_id",
            how="left",
        )
        .merge(
            holdout_annex[
                [
                    "program_id",
                    "mean_holdout_pd_minus_control",
                    "holdout_direction_match",
                    "holdout_support_tier",
                    "n_holdout_sources_available",
                    "n_holdout_sources_direction_match",
                    "gse157783_direction_match",
                    "gse253462_direction_match",
                ]
            ],
            on="program_id",
            how="left",
        )
    )

    merged["primary_spatial_component"] = merged["primary_spatial_support_bin"].map(PRIMARY_COMPONENT).fillna(0.0)
    merged["secondary_spatial_component"] = merged["secondary_spatial_support_bin"].map(SECONDARY_COMPONENT).fillna(0.0)
    merged["program_level_genetic_component"] = merged["program_genetic_support_bin"].map(PROGRAM_GENETIC_COMPONENT).fillna(0.0)
    merged["gene_level_component"] = merged["gene_level_component"].fillna(0.0)
    merged["cross_dataset_consistency_component"] = (
        merged["direction_match"].fillna(False).astype(float) * 0.6
        + merged["n_quadrants_direction_consistent"].fillna(0).clip(upper=4).astype(float) / 4.0 * 0.4
    )

    merged["source_dominant_flag"] = merged["caveat_flags_json"].astype(str).str.contains("source_dominant")
    merged["negative_holdout_flag"] = merged["caveat_flags_json"].astype(str).str.contains("negative_holdout")
    merged["source_flip_flag"] = merged["caveat_flags_json"].astype(str).str.contains("source_flip")
    merged["region_flip_flag"] = merged["caveat_flags_json"].astype(str).str.contains("region_flip")
    merged["cell_count_link_flag"] = merged["caveat_flags_json"].astype(str).str.contains("cell_count_link")
    merged["source_dominant_penalty"] = merged["source_dominant_flag"].map({True: PENALTIES["source_dominant"], False: 0.0})
    merged["negative_holdout_penalty"] = merged["negative_holdout_flag"].map({True: PENALTIES["negative_holdout"], False: 0.0})
    merged["source_flip_penalty"] = merged["source_flip_flag"].map({True: PENALTIES["source_flip"], False: 0.0})
    merged["region_flip_penalty"] = merged["region_flip_flag"].map({True: PENALTIES["region_flip"], False: 0.0})
    merged["cell_count_link_penalty"] = merged["cell_count_link_flag"].map({True: PENALTIES["cell_count_link"], False: 0.0})
    merged["blocking_issue_penalty"] = merged["blocking_issue_flag"].astype(bool).map({True: 0.15, False: 0.0})
    merged["total_penalty"] = merged[
        [
            "source_dominant_penalty",
            "negative_holdout_penalty",
            "source_flip_penalty",
            "region_flip_penalty",
            "cell_count_link_penalty",
            "blocking_issue_penalty",
        ]
    ].sum(axis=1)

    merged["final_step4_score"] = (
        0.35 * merged["primary_spatial_component"]
        + 0.15 * merged["secondary_spatial_component"]
        + 0.25 * merged["gene_level_component"]
        + 0.15 * merged["program_level_genetic_component"]
        + 0.10 * merged["cross_dataset_consistency_component"]
        - merged["total_penalty"]
    )

    merged["spatial_primary_gate"] = merged["primary_spatial_support_bin"].map(PRIMARY_GATE).fillna("fail")
    merged.loc[
        merged["spatial_primary_gate"].eq("pass") & ~merged["direction_match"].fillna(False),
        "spatial_primary_gate",
    ] = "suggestive"
    merged["spatial_replication_gate"] = merged["secondary_spatial_support_bin"].map(SECONDARY_GATE).fillna("fail")
    merged.loc[
        merged["spatial_replication_gate"].eq("suggestive")
        & merged["n_quadrants_direction_consistent"].fillna(0).lt(1),
        "spatial_replication_gate",
    ] = "fail"
    merged["genetic_gene_level_gate"] = merged.apply(gene_level_gate, axis=1)
    merged["genetic_program_level_gate"] = merged["program_genetic_support_bin"].map(PROGRAM_GENETIC_GATE).fillna("fail")
    merged["genetic_consensus_gate"] = merged.apply(consensus_gate, axis=1)
    merged["spatial_overall_gate"] = merged.apply(spatial_gate, axis=1)

    merged["hard_internal_issue_flag"] = (
        merged["blocking_issue_flag"].astype(bool)
        | merged["negative_holdout_flag"]
        | merged["source_flip_flag"]
        | merged["region_flip_flag"]
    )
    merged["holdout_support_tier"] = merged["holdout_support_tier"].fillna("fail")
    merged["internal_stability_gate"] = "pass"
    merged.loc[
        ~merged["hard_internal_issue_flag"] & merged["holdout_support_tier"].eq("fail"),
        "internal_stability_gate",
    ] = "fail"
    merged.loc[
        ~merged["hard_internal_issue_flag"]
        & (
            merged["holdout_support_tier"].eq("suggestive")
            | merged["cell_count_link_flag"]
        ),
        "internal_stability_gate",
    ] = "suggestive"
    merged.loc[merged["hard_internal_issue_flag"], "internal_stability_gate"] = "fail"

    claim_rows = merged.apply(classify_claim, axis=1, result_type="expand")
    claim_rows.columns = ["claim_tier", "promotion_decision", "final_evidence_class", "decision_rationale_base"]
    merged = pd.concat([merged, claim_rows], axis=1)
    merged["decision_rationale"] = merged.apply(
        lambda row: build_rationale(row, str(row["decision_rationale_base"])),
        axis=1,
    )
    merged = merged.drop(columns=["decision_rationale_base"])

    require_columns(merged, STEP4_FINAL_MATRIX_COLUMNS, context="Step 4 final evidence matrix")
    merged.to_csv(output_dir / "step4_final_evidence_matrix.csv", index=False)

    decision_log = merged[
        [
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
        ]
    ].copy()
    require_columns(decision_log, STEP4_DECISION_LOG_COLUMNS, context="Step 4 candidate decision log")
    decision_log.to_csv(output_dir / "step4_candidate_decision_log.csv", index=False)

    summary = merged[
        [
            "program_id",
            "program_name",
            "claim_tier",
            "promotion_decision",
            "spatial_primary_gate",
            "spatial_replication_gate",
            "genetic_consensus_gate",
            "internal_stability_gate",
            "final_step4_score",
        ]
    ].copy()
    lines = [
        "# Step 4 Validated Programs",
        "",
        markdown_table(summary.astype(object).fillna("NA")),
        "",
    ]
    write_markdown(output_dir / "step4_validated_programs.md", "\n".join(lines))


if __name__ == "__main__":
    main()
