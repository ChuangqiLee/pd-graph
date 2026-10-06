#!/usr/bin/env python3.12
"""Compute donor-level pair and program scores for Step 3."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import project_path
from scripts.step3.common import (
    TASK_TO_CELL_TYPES,
    load_step2_tables,
    minmax_series,
    pair_candidate_id,
    program_candidate_id,
    top_program_anchor_pairs,
    zscore_series,
)


DEFAULT_CONFIG = Path("configs/step3/step3_donor_models.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Step 3 donor-level pair scores.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def attention_support_table(
    attention: pd.DataFrame,
    graph_manifest: pd.DataFrame,
    ranked_pairs: pd.DataFrame,
    *,
    prefix: str,
) -> pd.DataFrame:
    lookup = ranked_pairs[["compact_pair_index", "pair_key", "pathway_family"]].rename(columns={"compact_pair_index": "top_pair_index"})
    meta_cols = [
        "graph_id",
        "task_name",
        "graph_unit_id",
        "split_group_id",
        "cohort_name",
        "source_id",
        "donor_id",
        "disease_group",
        "age_years",
        "brain_region_standardized",
        "sex_standardized",
        "n_edges",
    ]
    merged = (
        attention.loc[attention["split"].eq("val")]
        .merge(lookup, on="top_pair_index", how="left")
        .merge(graph_manifest[meta_cols], on="graph_id", how="left")
        .copy()
    )
    merged["contribution"] = merged["attention"].astype(float) * merged["top_pair_score"].astype(float)
    aggregated = (
        merged.groupby(
            [
                "task_name",
                "graph_unit_id",
                "split_group_id",
                "cohort_name",
                "source_id",
                "donor_id",
                "disease_group",
                "age_years",
                "brain_region_standardized",
                "sex_standardized",
                "pair_key",
                "pathway_family",
            ],
            observed=False,
        )
        .agg(
            raw_contribution=("contribution", "sum"),
            mean_attention=("attention", "mean"),
            max_attention=("attention", "max"),
            n_supporting_edges=("graph_id", "count"),
            n_supporting_subgraphs=("graph_id", "nunique"),
        )
        .reset_index()
    )
    rename_map = {
        "raw_contribution": f"{prefix}_raw_contribution",
        "mean_attention": f"{prefix}_mean_attention",
        "max_attention": f"{prefix}_max_attention",
        "n_supporting_edges": f"{prefix}_n_supporting_edges",
        "n_supporting_subgraphs": f"{prefix}_n_supporting_subgraphs",
    }
    return aggregated.rename(columns=rename_map)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    output_dir = project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = load_step2_tables()
    refined_map = pd.read_csv(project_path("data/processed/step3/02_programs/step3_refined_program_map.csv"))
    graph_manifest = tables["graph_manifest"].copy()
    ranked_pairs = tables["ranked_pairs"].copy()
    lr_scores = tables["lr_scores"].copy()
    backbone = tables["backbone"].copy()
    graph_meta = graph_manifest[
        [
            "task_name",
            "graph_unit_id",
            "split_group_id",
            "donor_id",
            "brain_region_standardized",
            "sex_standardized",
        ]
    ].drop_duplicates()

    age_support = attention_support_table(tables["age_attention"], graph_manifest, ranked_pairs, prefix="age")
    pd_support = attention_support_table(tables["pd_attention"], graph_manifest, ranked_pairs, prefix="pd")

    pair_donor = lr_scores.merge(
        graph_meta,
        on=["task_name", "graph_unit_id", "split_group_id"],
        how="left",
    ).merge(
        age_support,
        on=[
            "task_name",
            "graph_unit_id",
            "split_group_id",
            "cohort_name",
            "source_id",
            "donor_id",
            "disease_group",
            "age_years",
            "brain_region_standardized",
            "sex_standardized",
            "pair_key",
            "pathway_family",
        ],
        how="left",
    ).merge(
        pd_support,
        on=[
            "task_name",
            "graph_unit_id",
            "split_group_id",
            "cohort_name",
            "source_id",
            "donor_id",
            "disease_group",
            "age_years",
            "pair_key",
            "pathway_family",
            "brain_region_standardized",
            "sex_standardized",
        ],
        how="left",
    )

    graph_totals = (
        graph_manifest.groupby(["task_name", "graph_unit_id", "split_group_id"], observed=False)
        .agg(total_graphs=("graph_id", "nunique"), total_edges=("n_edges", "sum"))
        .reset_index()
    )
    pair_donor = pair_donor.merge(graph_totals, on=["task_name", "graph_unit_id", "split_group_id"], how="left")

    counts = (
        backbone.groupby(["split_group_id", "cell_type_standardized_final"], observed=False)["cell_id"]
        .count()
        .unstack(fill_value=0)
        .reset_index()
    )
    pair_donor = pair_donor.merge(counts, on="split_group_id", how="left")
    pair_donor["sender_cell_count"] = [
        int(pair_donor.at[index, TASK_TO_CELL_TYPES.get(task, ("", ""))[0]]) if TASK_TO_CELL_TYPES.get(task, ("", ""))[0] in pair_donor.columns else 0
        for index, task in enumerate(pair_donor["task_name"].tolist())
    ]
    pair_donor["receiver_cell_count"] = [
        int(pair_donor.at[index, TASK_TO_CELL_TYPES.get(task, ("", ""))[1]]) if TASK_TO_CELL_TYPES.get(task, ("", ""))[1] in pair_donor.columns else 0
        for index, task in enumerate(pair_donor["task_name"].tolist())
    ]
    pair_donor = pair_donor.merge(
        refined_map[["pair_key", "refined_program_id", "refined_program_name"]],
        on="pair_key",
        how="left",
    )
    pair_donor["candidate_id"] = pair_donor["pair_key"].map(pair_candidate_id)
    pair_donor["candidate_level"] = "pair"
    pair_donor["age_raw_contribution"] = pair_donor["age_raw_contribution"].fillna(0.0)
    pair_donor["pd_raw_contribution"] = pair_donor["pd_raw_contribution"].fillna(0.0)
    pair_donor["score_raw"] = pair_donor["score"].fillna(0.0)
    pair_donor["attention_raw_sum"] = pair_donor["age_raw_contribution"] + pair_donor["pd_raw_contribution"]
    pair_donor["contribution_per_sampled_subgraph"] = pair_donor["attention_raw_sum"] / pair_donor["total_graphs"].clip(lower=1)
    pair_donor["contribution_per_edge"] = pair_donor["attention_raw_sum"] / pair_donor["total_edges"].clip(lower=1)

    group_keys = ["pair_key", "task_name"]
    pair_donor["baseline_component"] = pair_donor.groupby(group_keys, observed=False)["score"].transform(lambda values: zscore_series(values).fillna(0.0))
    pair_donor["graph_component"] = pair_donor.groupby(group_keys, observed=False)["contribution_per_sampled_subgraph"].transform(
        lambda values: zscore_series(values).fillna(0.0)
    )
    pair_donor["edge_component"] = pair_donor.groupby(group_keys, observed=False)["contribution_per_edge"].transform(
        lambda values: zscore_series(values).fillna(0.0)
    )
    pair_donor["score_normalized"] = pair_donor[["baseline_component", "graph_component", "edge_component"]].mean(axis=1)
    pair_donor.to_csv(output_dir / "step3_pair_scores_by_donor.csv", index=False)

    anchor_pairs = top_program_anchor_pairs(refined_map, ranked_pairs, top_n=5)
    anchor_pairs["anchor_weight"] = anchor_pairs.groupby("refined_program_id", observed=False)["ranking_score"].transform(
        lambda values: minmax_series(values).fillna(0.0) + 0.1
    )
    program_donor = pair_donor.merge(
        anchor_pairs[["pair_key", "refined_program_id", "refined_program_name", "anchor_weight"]],
        on=["pair_key", "refined_program_id", "refined_program_name"],
        how="inner",
    )
    program_donor["weighted_score_raw"] = program_donor["score_raw"] * program_donor["anchor_weight"]
    program_donor["weighted_score_normalized"] = program_donor["score_normalized"] * program_donor["anchor_weight"]
    aggregated_programs = (
        program_donor.groupby(
            [
                "task_name",
                "graph_unit_id",
                "split_group_id",
                "cohort_name",
                "source_id",
                "donor_id",
                "disease_group",
                "age_years",
                "brain_region_standardized",
                "sex_standardized",
                "refined_program_id",
                "refined_program_name",
                "sender_cell_count",
                "receiver_cell_count",
            ],
            observed=False,
        )
        .agg(
            score_raw=("weighted_score_raw", "sum"),
            score_normalized=("weighted_score_normalized", "sum"),
            anchor_pair_count=("pair_key", "nunique"),
            anchor_pairs=("pair_key", lambda values: "|".join(sorted(dict.fromkeys(str(value) for value in values)))),
        )
        .reset_index()
    )
    aggregated_programs["candidate_id"] = aggregated_programs["refined_program_id"].map(program_candidate_id)
    aggregated_programs["candidate_level"] = "program"
    aggregated_programs.to_csv(output_dir / "step3_program_scores_by_donor.csv", index=False)


if __name__ == "__main__":
    main()
