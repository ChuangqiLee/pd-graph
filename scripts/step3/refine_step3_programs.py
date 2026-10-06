#!/usr/bin/env python3.12
"""Refine raw Step 2 pathway families into Step 3 biological programs."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import markdown_table, project_path, write_markdown
from scripts.step3.common import (
    PROGRAM_METADATA,
    gene_family_token,
    infer_program_from_pair,
    is_y_linked_pair,
    load_step2_tables,
    pair_candidate_id,
    program_name,
    task_support_vector,
)


DEFAULT_CONFIG = Path("configs/step3/step3_program_refinement.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine Step 3 programs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return float(len(left & right)) / float(len(left | right))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    output_dir = project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_cfg = config["similarity"]

    tables = load_step2_tables()
    pair_universe = tables["pair_universe"].copy()
    ranked_pairs = tables["ranked_pairs"].copy()
    pair_catalog = pair_universe.merge(
        ranked_pairs[["pair_key", "ranking_score", "aging_score", "pd_score", "delta_score"]],
        on="pair_key",
        how="left",
    )
    lr_scores = tables["lr_scores"].copy()
    registry_path = project_path("data/processed/step3/01_registry/step3_candidate_registry.csv")
    registry = pd.read_csv(registry_path)

    assignments = []
    for row in pair_catalog.itertuples(index=False):
        program_id, rule_note, confidence = infer_program_from_pair(str(row.pair_key), str(row.pathway_family))
        assignments.append(
            {
                "pair_candidate_key": pair_candidate_id(str(row.pair_key)),
                "pair_key": str(row.pair_key),
                "raw_pathway_family": str(row.pathway_family),
                "refined_program_id": program_id,
                "refined_program_name": program_name(program_id),
                "assignment_rule": rule_note,
                "assignment_confidence": confidence,
                "ligand_family_tokens": "|".join(sorted({gene_family_token(gene) for gene in str(row.ligand_genes).split("|")})),
                "receptor_family_tokens": "|".join(sorted({gene_family_token(gene) for gene in str(row.receptor_genes).split("|")})),
                "ranking_score": row.ranking_score,
                "graph_units_supported": row.graph_units_supported,
                "task_supports": row.task_supports,
                "sex_restriction_flag": is_y_linked_pair(str(row.pair_key)),
            }
        )
    refined_map = pd.DataFrame(assignments)
    refined_map.to_csv(output_dir / "step3_refined_program_map.csv", index=False)

    pair_support = (
        lr_scores.pivot_table(
            index=["task_name", "graph_unit_id"],
            columns="pair_key",
            values="score",
            fill_value=0.0,
        )
        .sort_index()
    )
    ranked_subset = pair_catalog.set_index("pair_key")
    edges: list[tuple[str, str, float]] = []
    nodes: list[dict[str, object]] = []
    pair_keys = refined_map["pair_key"].tolist()
    for row in refined_map.itertuples(index=False):
        nodes.append(
            {
                "id": row.pair_key,
                "raw_pathway_family": row.raw_pathway_family,
                "refined_program_id": row.refined_program_id,
                "refined_program_name": row.refined_program_name,
                "ranking_score": float(row.ranking_score) if pd.notna(row.ranking_score) else 0.0,
            }
        )
    for left_key, right_key in itertools.combinations(pair_keys, 2):
        left = ranked_subset.loc[left_key]
        right = ranked_subset.loc[right_key]
        pathway_match = 1.0 if str(left.pathway_family) == str(right.pathway_family) else 0.0
        ligand_overlap = jaccard(
            {gene_family_token(gene) for gene in str(left.ligand_genes).split("|")},
            {gene_family_token(gene) for gene in str(right.ligand_genes).split("|")},
        )
        receptor_overlap = jaccard(
            {gene_family_token(gene) for gene in str(left.receptor_genes).split("|")},
            {gene_family_token(gene) for gene in str(right.receptor_genes).split("|")},
        )
        task_similarity = cosine_similarity(task_support_vector(left.task_supports), task_support_vector(right.task_supports))
        donor_similarity = 0.0
        if left_key in pair_support.columns and right_key in pair_support.columns:
            left_values = pair_support[left_key].to_numpy(dtype=float)
            right_values = pair_support[right_key].to_numpy(dtype=float)
            if np.std(left_values) > 0 and np.std(right_values) > 0:
                donor_similarity = float(np.corrcoef(left_values, right_values)[0, 1])
                donor_similarity = float(np.nan_to_num(donor_similarity, nan=0.0))
        weight = (
            pathway_match * float(sim_cfg["pathway_weight"])
            + ligand_overlap * float(sim_cfg["ligand_family_weight"])
            + receptor_overlap * float(sim_cfg["receptor_family_weight"])
            + task_similarity * float(sim_cfg["task_similarity_weight"])
            + max(donor_similarity, 0.0) * float(sim_cfg["donor_correlation_weight"])
        )
        if weight >= float(sim_cfg["edge_threshold"]):
            edges.append((left_key, right_key, weight))

    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node["id"], **{key: value for key, value in node.items() if key != "id"})
    for left_key, right_key, weight in edges:
        graph.add_edge(left_key, right_key, weight=weight)
    nx.write_graphml(graph, output_dir / "step3_pair_similarity_network.graphml")

    family_program_map = (
        refined_map.groupby("raw_pathway_family", observed=False)["refined_program_id"]
        .agg(lambda values: "|".join(dict.fromkeys(str(value) for value in values)))
        .reset_index()
    )
    family_program_name_map = (
        refined_map.groupby("raw_pathway_family", observed=False)["refined_program_name"]
        .agg(lambda values: "|".join(dict.fromkeys(str(value) for value in values)))
        .reset_index()
    )
    registry.loc[registry["entity_level"].eq("pair"), "refined_program_id"] = registry.loc[
        registry["entity_level"].eq("pair"), "candidate_id"
    ].map(refined_map.set_index("pair_key")["refined_program_id"])
    registry.loc[registry["entity_level"].eq("pair"), "refined_program_name"] = registry.loc[
        registry["entity_level"].eq("pair"), "candidate_id"
    ].map(refined_map.set_index("pair_key")["refined_program_name"])
    pathway_program_map = family_program_map.set_index("raw_pathway_family")["refined_program_id"]
    pathway_program_names = family_program_name_map.set_index("raw_pathway_family")["refined_program_name"]
    registry.loc[registry["entity_level"].eq("pathway"), "refined_program_id"] = registry.loc[
        registry["entity_level"].eq("pathway"), "candidate_id"
    ].map(pathway_program_map)
    registry.loc[registry["entity_level"].eq("pathway"), "refined_program_name"] = registry.loc[
        registry["entity_level"].eq("pathway"), "candidate_id"
    ].map(pathway_program_names)
    registry.to_csv(registry_path, index=False)

    curation_notes = [
        "- Split the raw `unclassified` family into DSCAM/DCC contact recognition, NLGN/NRXN synaptic organizer, adhesion organizer, and residual exploratory branches.",
        "- Split the raw `Glutamate` family into ionotropic and metabotropic receptor programs.",
        "- Merged `NRG` and `growth factor signaling` receptor-centered ERBB4 pairs into one trophic maintenance program when the receptor context agreed.",
        "- Preserved a separate APOE/LRP receptor adjunct branch so the APP-centered story does not absorb all APOE biology.",
    ]
    report = "\n".join(
        [
            "# Step 3 Program Refinement Log",
            "",
            "## Refined Program Sizes",
            markdown_table(
                refined_map.groupby(["refined_program_id", "refined_program_name"], observed=False)["pair_key"]
                .count()
                .rename("n_pairs")
                .reset_index()
                .sort_values("n_pairs", ascending=False)
            ),
            "",
            "## Program Metadata",
            markdown_table(
                pd.DataFrame(
                    [
                        {
                            "refined_program_id": program_id,
                            "refined_program_name": meta["name"],
                            "priority": meta["priority"],
                            "summary": meta["summary"],
                        }
                        for program_id, meta in PROGRAM_METADATA.items()
                    ]
                ).sort_values("priority")
            ),
            "",
            "## Curation Decisions",
            "\n".join(curation_notes),
        ]
    )
    write_markdown(output_dir / "step3_program_refinement_log.md", report)


if __name__ == "__main__":
    main()
