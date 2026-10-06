#!/usr/bin/env python3
"""Build Step 2 graph datasets."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import (
    infer_graph_unit_id,
    load_yaml,
    markdown_table,
    project_path,
    save_pickle_gz,
    safe_slug,
    write_markdown,
)


DEFAULT_CONFIG = Path("configs/step2/step2_graphs.yaml")
COHORT_CONFIG = Path("configs/step2/step2_cohorts.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step 2 graph datasets.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--cohort-config", default=str(COHORT_CONFIG))
    return parser.parse_args()


def component_geometric_expression(matrix: np.ndarray, component_positions: list[int]) -> np.ndarray:
    if not component_positions:
        return np.zeros((matrix.shape[0],), dtype=np.float32)
    values = matrix[:, component_positions]
    zero_mask = (values <= 0).any(axis=1)
    clipped = np.clip(values, 1e-12, None)
    result = np.exp(np.mean(np.log(clipped), axis=1))
    result[zero_mask] = 0.0
    return result.astype(np.float32)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    cohort_config = load_yaml(args.cohort_config)
    output_dir = project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    comm = ad.read_h5ad(project_path(config["paths"]["comm_panel_h5ad"]))
    graph_manifest = pd.read_csv(project_path(config["paths"]["graph_manifest_csv"]))
    compact_universe = pd.read_parquet(project_path(config["paths"]["lr_universe_compact"]))
    compact_universe = compact_universe.reset_index(drop=True).rename(columns={"index": "compact_pair_index"})

    obs = comm.obs.copy()
    obs["graph_unit_id"] = obs.apply(infer_graph_unit_id, axis=1)
    label_source = obs["cell_type_label_source"].astype(str)
    confidence = pd.to_numeric(obs["cell_type_prediction_confidence"], errors="coerce")
    obs["included_primary_default"] = (
        obs["pass_qc"].astype(bool)
        & obs["cell_type_standardized_final"].astype(str).isin(set(cohort_config["primary_cell_types"]))
        & (label_source.ne("transferred") | confidence.ge(float(cohort_config["transferred_label_confidence_default"])))
    )
    obs["obs_position"] = np.arange(comm.n_obs)

    harmony = np.asarray(comm.obsm["X_pca_harmony"], dtype=np.float32)[:, : int(config["node_features"]["harmony_dims"])]
    comm_latent = np.asarray(comm.obsm["X_comm_latent"], dtype=np.float32)[:, : int(config["node_features"]["comm_latent_dims"])]
    norm = comm.layers["norm"]
    primary_order = list(config["node_caps"].keys())
    cell_type_lookup = {cell_type: idx for idx, cell_type in enumerate(primary_order)}

    node_features = []
    for idx, row in obs.iterrows():
        cell_type = str(row["cell_type_standardized_final"])
        one_hot = np.zeros((len(primary_order),), dtype=np.float32)
        if cell_type in cell_type_lookup:
            one_hot[cell_type_lookup[cell_type]] = 1.0
        qc_features = np.array(
            [
                math.log1p(float(row["qc_total_counts"])),
                math.log1p(float(row["qc_n_genes_by_counts"])),
                float(row["qc_pct_mt"]) / 100.0,
                float(row["cell_type_prediction_confidence"]) if not pd.isna(row["cell_type_prediction_confidence"]) else 0.0,
            ],
            dtype=np.float32,
        )
        node_features.append(np.concatenate([harmony[row["obs_position"]], comm_latent[row["obs_position"]], one_hot, qc_features]))
    node_feature_matrix = np.vstack(node_features).astype(np.float32)

    gene_lookup = {gene: idx for idx, gene in enumerate(comm.var_names.astype(str).tolist())}
    graph_rows = []
    task_summaries = []
    rng = np.random.default_rng(int(config["random_seed"]))

    for task_name, task_config in config["tasks"].items():
        task_dir = output_dir / safe_slug(task_name)
        task_dir.mkdir(parents=True, exist_ok=True)
        task_manifest = graph_manifest.loc[(graph_manifest["task_name"] == task_name) & (graph_manifest["included_task_default"])].copy()
        task_pairs = compact_universe.loc[compact_universe["eligible_tasks"].astype(str).str.contains(task_name)].copy()
        task_pairs["compact_pair_index"] = task_pairs.index.astype(int)
        pathway_names = sorted(task_pairs["pathway_family"].astype(str).unique().tolist())
        pathway_lookup = {name: idx for idx, name in enumerate(pathway_names)}
        pair_family_codes = np.array([pathway_lookup[name] for name in task_pairs["pathway_family"].astype(str)], dtype=int)

        task_pair_gene_union = sorted(
            {
                gene
                for value in pd.concat([task_pairs["ligand_genes"], task_pairs["receptor_genes"]]).astype(str)
                for gene in value.split("|")
                if gene
            }
        )
        task_gene_positions = [gene_lookup[gene] for gene in task_pair_gene_union if gene in gene_lookup]
        task_gene_lookup = {gene: index for index, gene in enumerate(task_pair_gene_union)}
        task_pair_indices = []
        for pair in task_pairs.itertuples(index=False):
            ligand_components = [task_gene_lookup[gene] for gene in str(pair.ligand_genes).split("|") if gene in task_gene_lookup]
            receptor_components = [task_gene_lookup[gene] for gene in str(pair.receptor_genes).split("|") if gene in task_gene_lookup]
            task_pair_indices.append((int(pair.compact_pair_index), ligand_components, receptor_components))

        task_graph_count = 0
        task_node_counts = []
        task_edge_counts = []
        for task_row in task_manifest.itertuples(index=False):
            cell_frame = obs.loc[
                obs["included_primary_default"]
                & obs["graph_unit_id"].astype(str).eq(str(task_row.graph_unit_id))
                & obs["cell_type_standardized_final"].astype(str).isin([task_row.sender_cell_type, task_row.receiver_cell_type])
            ].copy()
            sender_positions_all = cell_frame.loc[cell_frame["cell_type_standardized_final"].astype(str).eq(task_row.sender_cell_type), "obs_position"].to_numpy(dtype=int)
            receiver_positions_all = cell_frame.loc[cell_frame["cell_type_standardized_final"].astype(str).eq(task_row.receiver_cell_type), "obs_position"].to_numpy(dtype=int)
            sender_cap = int(config["node_caps"][task_row.sender_cell_type])
            receiver_cap = int(config["node_caps"][task_row.receiver_cell_type])
            repeats = int(config["subgraphs"]["per_graph_unit"]) if (len(sender_positions_all) > sender_cap or len(receiver_positions_all) > receiver_cap) else 1

            for repeat_index in range(repeats):
                sender_positions = sender_positions_all
                receiver_positions = receiver_positions_all
                if len(sender_positions_all) > sender_cap:
                    sender_positions = np.sort(rng.choice(sender_positions_all, size=sender_cap, replace=False))
                if len(receiver_positions_all) > receiver_cap:
                    receiver_positions = np.sort(rng.choice(receiver_positions_all, size=receiver_cap, replace=False))
                if len(sender_positions) == 0 or len(receiver_positions) == 0:
                    continue

                sender_harmony = harmony[sender_positions]
                receiver_harmony = harmony[receiver_positions]
                neighbors = min(int(config["graph_construction"]["k_neighbors"]), len(receiver_positions))
                knn = NearestNeighbors(n_neighbors=neighbors, metric="euclidean")
                knn.fit(receiver_harmony)
                distances, receiver_neighbor_indices = knn.kneighbors(sender_harmony)

                sender_gene_matrix = norm[sender_positions][:, task_gene_positions]
                receiver_gene_matrix = norm[receiver_positions][:, task_gene_positions]
                if sp.issparse(sender_gene_matrix):
                    sender_gene_matrix = sender_gene_matrix.toarray()
                if sp.issparse(receiver_gene_matrix):
                    receiver_gene_matrix = receiver_gene_matrix.toarray()
                sender_gene_matrix = np.asarray(sender_gene_matrix, dtype=np.float32)
                receiver_gene_matrix = np.asarray(receiver_gene_matrix, dtype=np.float32)

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
                family_indicator = np.zeros((len(task_pair_indices), len(pathway_names)), dtype=np.float32)
                for pair_position, family_code in enumerate(pair_family_codes):
                    family_indicator[pair_position, family_code] = 1.0
                family_denominator = np.maximum(family_indicator.sum(axis=0, keepdims=True), 1.0)

                for sender_local_index, receiver_local_choices in enumerate(receiver_neighbor_indices):
                    for neighbor_rank, receiver_local_index in enumerate(receiver_local_choices):
                        pair_scores = np.sqrt(
                            np.maximum(sender_pair_expr[sender_local_index], 0.0)
                            * np.maximum(receiver_pair_expr[receiver_local_index], 0.0)
                        )
                        pathway_features = (pair_scores @ family_indicator / family_denominator.ravel()).astype(np.float32)
                        distance_feature = np.array([distances[sender_local_index, neighbor_rank]], dtype=np.float32)
                        if pathway_features.max(initial=0.0) < float(config["graph_construction"]["min_active_edge_score"]):
                            continue
                        edge_attr = np.concatenate([pathway_features, distance_feature]).astype(np.float32)
                        top_k = min(int(config["graph_construction"]["top_lr_features_per_edge"]), len(pair_scores))
                        top_indices_local = np.argpartition(pair_scores, -top_k)[-top_k:]
                        top_indices_local = top_indices_local[np.argsort(pair_scores[top_indices_local])[::-1]]
                        edge_index_rows.append([sender_local_index, len(sender_positions) + receiver_local_index])
                        edge_attrs.append(edge_attr)
                        edge_top_pair_indices.append(pair_ids[top_indices_local])
                        edge_top_pair_scores.append(pair_scores[top_indices_local].astype(np.float32))

                if not edge_index_rows:
                    continue

                graph_id = safe_slug(f"{task_name}__{task_row.graph_unit_id}__subgraph_{repeat_index:02d}")
                node_positions = np.concatenate([sender_positions, receiver_positions])
                graph = {
                    "graph_id": graph_id,
                    "task_name": task_name,
                    "node_ids": comm.obs_names[node_positions].astype(str).to_numpy(),
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
                        "graph_unit_id": str(task_row.graph_unit_id),
                        "split_group_id": str(task_row.split_group_id),
                        "source_id": str(task_row.source_id),
                        "sample_id": str(task_row.sample_id),
                        "donor_id": str(task_row.donor_id),
                        "cohort_name": str(task_row.cohort_name),
                        "disease_group": str(task_row.disease_group),
                        "age_years": None if pd.isna(task_row.age_years) else float(task_row.age_years),
                        "age_missing_mask": bool(pd.isna(task_row.age_years)),
                        "brain_region_standardized": str(task_row.brain_region_standardized),
                        "sex_standardized": str(task_row.sex_standardized),
                    },
                }
                graph_path = task_dir / f"{graph_id}.pkl.gz"
                save_pickle_gz(graph_path, graph)
                graph_rows.append(
                    {
                        "graph_id": graph_id,
                        "graph_path": str(graph_path),
                        "task_name": task_name,
                        "graph_unit_id": str(task_row.graph_unit_id),
                        "split_group_id": str(task_row.split_group_id),
                        "cohort_name": str(task_row.cohort_name),
                        "source_id": str(task_row.source_id),
                        "sample_id": str(task_row.sample_id),
                        "donor_id": str(task_row.donor_id),
                        "disease_group": str(task_row.disease_group),
                        "age_years": task_row.age_years,
                        "brain_region_standardized": str(task_row.brain_region_standardized),
                        "sex_standardized": str(task_row.sex_standardized),
                        "subgraph_index": int(repeat_index),
                        "n_nodes": int(len(node_positions)),
                        "n_edges": int(len(edge_index_rows)),
                        "sender_cell_type": str(task_row.sender_cell_type),
                        "receiver_cell_type": str(task_row.receiver_cell_type),
                    }
                )
                task_graph_count += 1
                task_node_counts.append(len(node_positions))
                task_edge_counts.append(len(edge_index_rows))

        task_summaries.append(
            {
                "task_name": task_name,
                "n_graphs": task_graph_count,
                "median_nodes": float(np.median(task_node_counts)) if task_node_counts else 0.0,
                "median_edges": float(np.median(task_edge_counts)) if task_edge_counts else 0.0,
            }
        )

    graph_manifest_out = pd.DataFrame(graph_rows).sort_values(["task_name", "graph_unit_id", "subgraph_index"])
    graph_manifest_out.to_parquet(output_dir / "step2_graph_manifest.parquet", index=False)
    graph_manifest_out.to_csv(output_dir / "step2_graph_manifest.csv", index=False)

    summary_df = pd.DataFrame(task_summaries)
    report = "\n".join(
        [
            "# Step 2 Graph QC",
            "",
            "## Task Summary",
            markdown_table(summary_df),
            "",
            "## Notes",
            "- Graphs are built strictly within biological graph units.",
            "- Sender and receiver cells are sampled stochastically only when they exceed task-specific caps.",
            "- Edge features combine pathway family aggregates with Harmony neighborhood distance.",
        ]
    )
    write_markdown(output_dir / "step2_graph_qc_report.md", report)


if __name__ == "__main__":
    main()
