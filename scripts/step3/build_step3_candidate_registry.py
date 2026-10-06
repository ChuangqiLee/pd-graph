#!/usr/bin/env python3.12
"""Build the Step 3 candidate registry from Step 2 exports."""

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
from scripts.step3.common import (
    AMBIGUOUS_PATHWAY_FAMILIES,
    is_y_linked_pair,
    load_step2_tables,
    pair_candidate_id,
    pathway_candidate_id,
)


DEFAULT_CONFIG = Path("configs/step3/step3_candidate_registry.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Step 3 candidate registry.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def annotate_registry_rows(
    frame: pd.DataFrame,
    *,
    holdout_warning: float,
    fold_warning: float,
    fold_hard: float,
    source_warning: float,
    graph_warning: float,
    graph_hard: float,
) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["fold_stability"] = annotated["fold_stability"].fillna(0.0)
    annotated["source_stability"] = annotated["source_stability"].fillna(0.0)
    annotated["holdout_support"] = annotated["holdout_support"].fillna(0.0)
    annotated["graph_support_units"] = (
        annotated["graph_units_supported"]
        .where(annotated["graph_units_supported"].notna(), annotated["provisional_n_graph_units_supported"])
        .fillna(0.0)
    )
    annotated["holdout_warning_flag"] = annotated["holdout_support"].abs().lt(holdout_warning)
    annotated["negative_holdout_flag"] = annotated["holdout_support"].lt(0.0)
    annotated["low_fold_stability_flag"] = annotated["fold_stability"].lt(fold_warning)
    annotated["hard_fold_instability_flag"] = annotated["fold_stability"].lt(fold_hard)
    annotated["low_source_stability_flag"] = annotated["source_stability"].lt(source_warning)
    annotated["limited_graph_support_flag"] = annotated["graph_support_units"].lt(graph_warning)
    annotated["hard_graph_support_flag"] = annotated["graph_support_units"].lt(graph_hard)
    annotated["step2_instability_flag"] = annotated["interpretation_bin"].eq("unstable_or_source_sensitive")
    annotated["source_sensitivity_pending_flag"] = annotated["low_source_stability_flag"] | annotated["source_stability"].eq(0.0)

    hard_failure = (
        annotated["negative_holdout_flag"]
        | annotated["hard_fold_instability_flag"]
        | annotated["hard_graph_support_flag"]
        | annotated["step2_instability_flag"]
    )
    review_needed = (
        annotated["holdout_warning_flag"]
        | annotated["low_fold_stability_flag"]
        | annotated["low_source_stability_flag"]
        | annotated["limited_graph_support_flag"]
        | annotated["annotation_ambiguity_flag"]
        | annotated["sex_restriction_flag"]
    )

    annotated["registry_gate"] = "promote"
    annotated.loc[review_needed, "registry_gate"] = "review"
    annotated.loc[hard_failure, "registry_gate"] = "hold"
    annotated["stability_bin"] = "stable"
    annotated.loc[review_needed, "stability_bin"] = "watch"
    annotated.loc[hard_failure, "stability_bin"] = "unstable"
    return annotated


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    output_dir = project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = config.get("thresholds", {})
    pair_holdout_warning = float(thresholds.get("pair_holdout_warning", 0.25))
    pathway_holdout_warning = float(thresholds.get("pathway_holdout_warning", 0.05))
    pair_fold_stability_warning = float(thresholds.get("pair_fold_stability_warning", 0.67))
    pair_fold_stability_hard = float(thresholds.get("pair_fold_stability_hard", 0.45))
    pathway_fold_stability_warning = float(thresholds.get("pathway_fold_stability_warning", 0.60))
    pathway_fold_stability_hard = float(thresholds.get("pathway_fold_stability_hard", 0.40))
    pair_source_stability_warning = float(thresholds.get("pair_source_stability_warning", 0.80))
    pathway_source_stability_warning = float(thresholds.get("pathway_source_stability_warning", 0.80))
    pair_graph_units_warning = float(thresholds.get("pair_graph_units_warning", 60))
    pair_graph_units_hard = float(thresholds.get("pair_graph_units_hard", 40))
    pathway_graph_units_warning = float(thresholds.get("pathway_graph_units_warning", 5))
    pathway_graph_units_hard = float(thresholds.get("pathway_graph_units_hard", 3))

    tables = load_step2_tables()
    top_candidates = pd.concat([tables["top_step3"], tables["top_step4"]], ignore_index=True).drop_duplicates(
        subset=["candidate_type", "candidate_id"]
    )
    step4_pair_ids = set(tables["top_step4"].loc[tables["top_step4"]["candidate_type"].eq("lr_pair"), "candidate_id"].astype(str))
    step4_pathway_ids = set(tables["top_step4"].loc[tables["top_step4"]["candidate_type"].eq("pathway"), "candidate_id"].astype(str))

    ranked_pairs = tables["ranked_pairs"].copy()
    ranked_pathways = tables["ranked_pathways"].copy()
    graph_manifest = tables["graph_manifest"][
        ["graph_id", "task_name", "graph_unit_id", "source_id", "cohort_name", "disease_group"]
    ].drop_duplicates()
    edge_contexts = tables["edge_contexts"].merge(graph_manifest, on="graph_id", how="left")

    pair_edge_support = (
        edge_contexts.groupby("pair_key", observed=False)
        .agg(
            provisional_n_graphs_supported=("graph_id", "nunique"),
            provisional_n_graph_units_supported=("graph_unit_id", "nunique"),
            provisional_n_sources_supported=("source_id", "nunique"),
            provisional_primary_task=("task_name", lambda values: pd.Series(values).mode().iloc[0]),
        )
        .reset_index()
    )
    pathway_edge_support = (
        edge_contexts.groupby("pathway_family", observed=False)
        .agg(
            provisional_n_graphs_supported=("graph_id", "nunique"),
            provisional_n_graph_units_supported=("graph_unit_id", "nunique"),
            provisional_n_sources_supported=("source_id", "nunique"),
            provisional_primary_task=("task_name", lambda values: pd.Series(values).mode().iloc[0]),
        )
        .reset_index()
    )

    pair_rows = top_candidates.loc[top_candidates["candidate_type"].eq("lr_pair")].merge(
        ranked_pairs,
        left_on="candidate_id",
        right_on="pair_key",
        how="left",
        suffixes=("", "_rank"),
    )
    pair_rows = pair_rows.merge(pair_edge_support, left_on="candidate_id", right_on="pair_key", how="left")
    pair_rows["entity_level"] = "pair"
    pair_rows["candidate_key"] = pair_rows["candidate_id"].map(pair_candidate_id)
    pair_rows["candidate_label"] = pair_rows["candidate_id"]
    pair_rows["ligand_gene_string"] = pair_rows["ligand_genes"]
    pair_rows["receptor_gene_string"] = pair_rows["receptor_genes"]
    pair_rows["raw_pathway_family"] = pair_rows["pathway_family"]
    pair_rows["step4_shortlist_flag"] = pair_rows["candidate_id"].astype(str).isin(step4_pair_ids)
    pair_rows["sex_restriction_flag"] = pair_rows["candidate_id"].map(is_y_linked_pair)
    pair_rows["annotation_ambiguity_flag"] = pair_rows["raw_pathway_family"].isin(AMBIGUOUS_PATHWAY_FAMILIES)
    pair_rows = annotate_registry_rows(
        pair_rows,
        holdout_warning=pair_holdout_warning,
        fold_warning=pair_fold_stability_warning,
        fold_hard=pair_fold_stability_hard,
        source_warning=pair_source_stability_warning,
        graph_warning=pair_graph_units_warning,
        graph_hard=pair_graph_units_hard,
    )

    pathway_rows = top_candidates.loc[top_candidates["candidate_type"].eq("pathway")].merge(
        ranked_pathways,
        left_on="candidate_id",
        right_on="pathway_family",
        how="left",
        suffixes=("", "_rank"),
    )
    pathway_rows = pathway_rows.merge(pathway_edge_support, left_on="candidate_id", right_on="pathway_family", how="left")
    pathway_rows["entity_level"] = "pathway"
    pathway_rows["candidate_key"] = pathway_rows["candidate_id"].map(pathway_candidate_id)
    pathway_rows["candidate_label"] = pathway_rows["candidate_id"]
    pathway_rows["raw_pathway_family"] = pathway_rows["candidate_id"]
    pathway_rows["step4_shortlist_flag"] = pathway_rows["candidate_id"].astype(str).isin(step4_pathway_ids)
    pathway_rows["sex_restriction_flag"] = False
    pathway_rows["annotation_ambiguity_flag"] = pathway_rows["candidate_id"].isin(AMBIGUOUS_PATHWAY_FAMILIES)
    pathway_rows["ligand_gene_string"] = pd.NA
    pathway_rows["receptor_gene_string"] = pd.NA
    pathway_rows["graph_units_supported"] = pathway_rows["provisional_n_graph_units_supported"]
    pathway_rows = annotate_registry_rows(
        pathway_rows,
        holdout_warning=pathway_holdout_warning,
        fold_warning=pathway_fold_stability_warning,
        fold_hard=pathway_fold_stability_hard,
        source_warning=pathway_source_stability_warning,
        graph_warning=pathway_graph_units_warning,
        graph_hard=pathway_graph_units_hard,
    )

    registry = pd.concat([pathway_rows, pair_rows], ignore_index=True, sort=False)
    registry["refined_program_id"] = pd.NA
    registry["refined_program_name"] = pd.NA
    registry["anchor_pair_ids"] = pd.NA
    registry["directionality_check_status"] = "pending"
    registry["final_step3_tier"] = "pending"
    registry = registry[
        [
            "entity_level",
            "candidate_type",
            "candidate_key",
            "candidate_id",
            "candidate_label",
            "raw_pathway_family",
            "ligand_gene_string",
            "receptor_gene_string",
            "ranking_score",
            "aging_score",
            "pd_score",
            "delta_score",
            "receiver_specificity",
            "holdout_support",
            "fold_stability",
            "source_stability",
            "source_coverage_ratio",
            "interpretation_bin",
            "graph_support_units",
            "step4_shortlist_flag",
            "sex_restriction_flag",
            "source_sensitivity_pending_flag",
            "annotation_ambiguity_flag",
            "holdout_warning_flag",
            "negative_holdout_flag",
            "low_fold_stability_flag",
            "low_source_stability_flag",
            "limited_graph_support_flag",
            "step2_instability_flag",
            "registry_gate",
            "stability_bin",
            "provisional_n_graphs_supported",
            "provisional_n_graph_units_supported",
            "provisional_n_sources_supported",
            "provisional_primary_task",
            "refined_program_id",
            "refined_program_name",
            "anchor_pair_ids",
            "directionality_check_status",
            "final_step3_tier",
        ]
    ].sort_values(["entity_level", "ranking_score"], ascending=[True, False])
    registry.to_csv(output_dir / "step3_candidate_registry.csv", index=False)

    report = "\n".join(
        [
            "# Step 3 Candidate Registry",
            "",
            f"Registry size: **{len(registry)}** candidates.",
            "",
            "## Candidate Preview",
            markdown_table(registry.head(12).fillna("NA")),
            "",
            "## Flag Summary",
            markdown_table(
                pd.DataFrame(
                    [
                        {"flag": "step4_shortlist_flag", "count": int(registry["step4_shortlist_flag"].fillna(False).sum())},
                        {"flag": "sex_restriction_flag", "count": int(registry["sex_restriction_flag"].fillna(False).sum())},
                        {"flag": "annotation_ambiguity_flag", "count": int(registry["annotation_ambiguity_flag"].fillna(False).sum())},
                        {"flag": "holdout_warning_flag", "count": int(registry["holdout_warning_flag"].fillna(False).sum())},
                        {"flag": "low_fold_stability_flag", "count": int(registry["low_fold_stability_flag"].fillna(False).sum())},
                        {"flag": "limited_graph_support_flag", "count": int(registry["limited_graph_support_flag"].fillna(False).sum())},
                    ]
                )
            ),
            "",
            "## Registry Gates",
            markdown_table(registry["registry_gate"].value_counts().rename_axis("registry_gate").reset_index(name="count")),
        ]
    )
    write_markdown(output_dir / "step3_candidate_registry.md", report)


if __name__ == "__main__":
    main()
