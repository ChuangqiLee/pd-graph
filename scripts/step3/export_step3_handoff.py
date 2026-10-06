#!/usr/bin/env python3.12
"""Export the Step 4 handoff tables from Step 3 outputs."""

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
from scripts.step3.common import PRIMARY_TASK, program_validation_gene_panel
from scripts.step4.common import (
    STEP3_TO_STEP4_GENE_COLUMNS,
    STEP3_TO_STEP4_SPATIAL_COLUMNS,
    require_columns,
)


DEFAULT_CONFIG = Path("configs/step3/step3_handoff.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Step 3 -> Step 4 handoff.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    output_dir = project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    include_tiers = set(config["thresholds"]["include_tiers"])
    include_priorities = set(config.get("thresholds", {}).get("include_priorities", ["high", "medium"]))
    include_candidate_ids = set(config.get("thresholds", {}).get("include_candidate_ids", []))

    top10 = pd.read_csv(project_path("data/processed/step3/09_top10/step3_top10_refined_programs.csv"))
    state_support = pd.read_csv(project_path("data/processed/step3/05_states/step3_candidate_state_support.csv"))
    state_markers = pd.read_csv(project_path("data/processed/step3/05_states/step3_state_marker_summary.csv"))
    require_columns(
        top10,
        (
            "candidate_id",
            "refined_program_id",
            "refined_program_name",
            "anchor_pairs",
            "dominant_task",
            "step3_program_score",
            "step4_validation_priority",
            "final_step3_tier",
            "caveat_summary",
            "interpretation_class",
            "dominant_sender_state",
            "dominant_receiver_state",
        ),
        context="Step 3 top10 programs",
    )
    require_columns(
        state_support,
        (
            "candidate_id",
            "candidate_level",
            "state_pair_rank",
            "sender_state_id",
            "receiver_state_id",
        ),
        context="Step 3 candidate state support",
    )
    require_columns(
        state_markers,
        (
            "state_id",
            "top_genes",
        ),
        context="Step 3 state marker summary",
    )
    eligible = top10.loc[
        top10["final_step3_tier"].isin(include_tiers)
        & top10["dominant_task"].eq(PRIMARY_TASK)
        & top10["step4_validation_priority"].isin(include_priorities)
        & ~top10["caveat_summary"].astype(str).str.contains("annotation_ambiguity", na=False)
    ].copy()
    if include_candidate_ids:
        eligible = pd.concat(
            [
                eligible,
                top10.loc[
                    top10["candidate_id"].isin(include_candidate_ids)
                    & top10["final_step3_tier"].isin(include_tiers)
                    & top10["dominant_task"].eq(PRIMARY_TASK)
                    & ~top10["caveat_summary"].astype(str).str.contains("annotation_ambiguity", na=False)
                ].copy(),
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["candidate_id"], keep="first")
    priority_order = {"high": 0, "medium": 1, "low": 2}
    eligible["step4_priority_order"] = eligible["step4_validation_priority"].map(priority_order).fillna(99)
    eligible = eligible.sort_values(["step4_priority_order", "step3_program_score"], ascending=[True, False]).drop(columns="step4_priority_order")

    top_state = state_support.loc[(state_support["candidate_level"].eq("program")) & (state_support["state_pair_rank"].eq(1))].set_index("candidate_id")
    marker_lookup = state_markers.set_index("state_id")

    spatial_rows = []
    gene_rows = []
    for row in eligible.itertuples(index=False):
        candidate_id = row.candidate_id
        state_row = top_state.loc[candidate_id] if candidate_id in top_state.index else None
        sender_markers = marker_lookup.loc[state_row["sender_state_id"], "top_genes"] if state_row is not None and state_row["sender_state_id"] in marker_lookup.index else ""
        receiver_markers = marker_lookup.loc[state_row["receiver_state_id"], "top_genes"] if state_row is not None and state_row["receiver_state_id"] in marker_lookup.index else ""
        anchor_pairs = str(row.anchor_pairs).split("|")
        anchor_genes = sorted({gene for pair in anchor_pairs for gene in pair.replace("->", "|").split("|")})
        validation_gene_panel = program_validation_gene_panel(
            row.refined_program_id,
            anchor_pairs,
            sender_state_markers=sender_markers,
            receiver_state_markers=receiver_markers,
            max_genes=10,
        )
        spatial_rows.append(
            {
                "refined_program_id": row.refined_program_id,
                "refined_program_name": row.refined_program_name,
                "anchor_genes": "|".join(anchor_genes),
                "sender_state_id": state_row["sender_state_id"] if state_row is not None else "NA",
                "sender_state_markers": sender_markers or "NA",
                "receiver_state_id": state_row["receiver_state_id"] if state_row is not None else "NA",
                "receiver_state_markers": receiver_markers or "NA",
                "expected_disease_direction": row.interpretation_class,
                "expected_spatial_relationship": f"{row.dominant_sender_state} proximal to {row.dominant_receiver_state} in {row.dominant_task}",
                "caveat_flags": row.caveat_summary,
                "validation_tier": row.final_step3_tier,
            }
        )
        gene_rows.append(
            {
                "refined_program_id": row.refined_program_id,
                "refined_program_name": row.refined_program_name,
                "anchor_pairs": row.anchor_pairs,
                "anchor_genes": "|".join(anchor_genes),
                "gene_set_members": "|".join(validation_gene_panel),
                "expected_disease_direction": row.interpretation_class,
                "expected_mr_gwas_members": "|".join(validation_gene_panel),
                "sender_state_markers": sender_markers or "NA",
                "receiver_state_markers": receiver_markers or "NA",
                "caveat_flags": row.caveat_summary,
                "validation_tier": row.final_step3_tier,
                "step4_validation_priority": row.step4_validation_priority,
            }
        )

    spatial = pd.DataFrame(spatial_rows, columns=STEP3_TO_STEP4_SPATIAL_COLUMNS)
    gene_set = pd.DataFrame(gene_rows, columns=STEP3_TO_STEP4_GENE_COLUMNS)
    require_columns(spatial, STEP3_TO_STEP4_SPATIAL_COLUMNS, context="exported Step 3 to Step 4 spatial handoff")
    require_columns(gene_set, STEP3_TO_STEP4_GENE_COLUMNS, context="exported Step 3 to Step 4 gene handoff")
    spatial.to_csv(output_dir / "step3_to_step4_spatial_targets.csv", index=False)
    gene_set.to_csv(output_dir / "step3_to_step4_gene_set_targets.csv", index=False)

    summary = "\n".join(
        [
            "# Step 3 to Step 4 Priority Summary",
            "",
            "_No eligible Step 4 programs passed the Step 3 gates in this export._" if spatial.empty else "",
            "",
            "## Spatial Targets",
            markdown_table(spatial.fillna("NA")) if not spatial.empty else "_None_",
            "",
            "## Gene Set Targets",
            markdown_table(gene_set.fillna("NA")) if not gene_set.empty else "_None_",
        ]
    )
    write_markdown(output_dir / "step3_to_step4_priority_summary.md", summary)


if __name__ == "__main__":
    main()
