#!/usr/bin/env python3.12
"""Extract exemplar edges and representative subgraphs for Step 3 programs."""

from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import project_path
from scripts.step3.common import TASK_TO_CELL_TYPES, load_comm_adata, top_program_anchor_pairs


DEFAULT_OUTPUT_DIR = Path("data/processed/step3/06_exemplars")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Step 3 exemplars.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def load_graph(path: Path) -> dict[str, object]:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def main() -> None:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    refined_map = pd.read_csv(project_path("data/processed/step3/02_programs/step3_refined_program_map.csv"))
    ranked_pairs = pd.read_csv(project_path("data/processed/step2/08_interpretation/step2_ranked_lr_pairs.csv"))
    task_support = pd.read_csv(project_path("data/processed/step3/04_task_support/step3_task_support_matrix.csv"))
    graph_manifest = pd.read_parquet(project_path("data/processed/step2/05_graphs/step2_graph_manifest.parquet"))
    state_support = pd.read_csv(project_path("data/processed/step3/05_states/step3_candidate_state_support.csv"))
    age_attention = pd.read_parquet(project_path("data/processed/step2/06_models/age_model/age_model_attention_summary.parquet"))
    pd_attention = pd.read_parquet(project_path("data/processed/step2/06_models/pd_model/pd_model_attention_summary.parquet"))

    task_index = task_support.set_index("candidate_id")
    anchor_pairs = top_program_anchor_pairs(refined_map, ranked_pairs, top_n=3)
    pair_lookup = ranked_pairs[["compact_pair_index", "pair_key"]].rename(columns={"compact_pair_index": "top_pair_index"})

    def attention_frame(att: pd.DataFrame, model_name: str) -> pd.DataFrame:
        local = (
            att.loc[att["split"].eq("val")]
            .merge(pair_lookup, on="top_pair_index", how="left")
            .merge(
                graph_manifest[
                    [
                        "graph_id",
                        "graph_path",
                        "task_name",
                        "graph_unit_id",
                        "cohort_name",
                        "source_id",
                        "donor_id",
                        "disease_group",
                    ]
                ],
                on="graph_id",
                how="left",
            )
            .copy()
        )
        local["model_name"] = model_name
        local["contribution"] = local["attention"].astype(float) * local["top_pair_score"].astype(float)
        return local

    attention = pd.concat(
        [attention_frame(age_attention, "age"), attention_frame(pd_attention, "pd")],
        ignore_index=True,
    )
    adata = load_comm_adata(backed="r")
    cell_index = pd.Series(np.arange(adata.n_obs), index=adata.obs["cell_id"].astype(str))
    gene_index = {gene: idx for idx, gene in enumerate(adata.var_names.astype(str))}
    state_lookup = (
        state_support.loc[state_support["state_pair_rank"].eq(1)]
        .set_index("candidate_id")[["sender_state_id", "receiver_state_id"]]
        .to_dict(orient="index")
    )

    exemplar_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for program_id, frame in anchor_pairs.groupby("refined_program_id", observed=False):
        candidate_id = f"program::{program_id}"
        if candidate_id not in task_index.index:
            continue
        primary_task = str(task_index.loc[candidate_id, "primary_task"])
        local = attention.loc[
            attention["task_name"].eq(primary_task) & attention["pair_key"].isin(frame["pair_key"].astype(str))
        ].copy()
        if local.empty:
            continue
        cohort_priority = ["control_known_age", "pd_known_age", "lbd_bridge"]
        for cohort_name in cohort_priority:
            cohort_rows = local.loc[local["cohort_name"].eq(cohort_name)].copy()
            if cohort_rows.empty:
                continue
            exemplar = cohort_rows.sort_values("contribution", ascending=False).iloc[0]
            graph = load_graph(Path(str(exemplar["graph_path"])))
            node_ids = list(graph["node_ids"])
            positions = np.asarray(graph["node_positions"], dtype=float)
            if positions.ndim == 1:
                sender_mask = np.asarray(graph.get("sender_mask"), dtype=bool)
                receiver_mask = np.asarray(graph.get("receiver_mask"), dtype=bool)
                positions = np.zeros((len(node_ids), 2), dtype=float)
                sender_idx = np.where(sender_mask)[0]
                receiver_idx = np.where(receiver_mask)[0]
                positions[sender_idx, 0] = 0.0
                positions[receiver_idx, 0] = 1.0
                if len(sender_idx):
                    positions[sender_idx, 1] = np.linspace(0.0, 1.0, len(sender_idx))
                if len(receiver_idx):
                    positions[receiver_idx, 1] = np.linspace(0.0, 1.0, len(receiver_idx))
            if positions.shape[1] > 2:
                positions = positions[:, :2]
            node_lookup = {str(cell_id): idx for idx, cell_id in enumerate(node_ids)}
            current_edges = local.loc[
                local["graph_id"].eq(exemplar["graph_id"]) & local["pair_key"].eq(exemplar["pair_key"])
            ].sort_values("contribution", ascending=False).head(5)
            fig, ax = plt.subplots(figsize=(6, 5))
            sender_color = "#2a9d8f"
            receiver_color = "#e76f51"
            ax.scatter(positions[:, 0], positions[:, 1], s=10, alpha=0.25, color="#bdbdbd")
            for edge_row in current_edges.itertuples(index=False):
                if edge_row.src_cell_id not in node_lookup or edge_row.dst_cell_id not in node_lookup:
                    continue
                src_idx = node_lookup[str(edge_row.src_cell_id)]
                dst_idx = node_lookup[str(edge_row.dst_cell_id)]
                ax.plot(
                    [positions[src_idx, 0], positions[dst_idx, 0]],
                    [positions[src_idx, 1], positions[dst_idx, 1]],
                    color="#264653",
                    alpha=0.8,
                    linewidth=1.6,
                )
                ax.scatter(positions[src_idx, 0], positions[src_idx, 1], s=24, color=sender_color)
                ax.scatter(positions[dst_idx, 0], positions[dst_idx, 1], s=24, color=receiver_color)
            ax.set_title(f"{program_id} | {cohort_name}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            figure_path = figure_dir / f"{program_id}_{cohort_name}_{exemplar['graph_id']}.png"
            fig.savefig(figure_path, dpi=220, bbox_inches="tight")
            plt.close(fig)

            ligand_text, receptor_text = str(exemplar["pair_key"]).split("->")
            ligand_genes = [gene for gene in ligand_text.split("|") if gene in gene_index]
            receptor_genes = [gene for gene in receptor_text.split("|") if gene in gene_index]
            src_expr = float("nan")
            dst_expr = float("nan")
            if str(exemplar["src_cell_id"]) in cell_index.index and ligand_genes:
                row_idx = int(cell_index[str(exemplar["src_cell_id"])])
                cols = [gene_index[gene] for gene in ligand_genes]
                src_slice = adata.layers["lognorm"][row_idx, cols]
                src_array = src_slice.toarray() if hasattr(src_slice, "toarray") else np.asarray(src_slice)
                src_expr = float(np.asarray(src_array).mean())
            if str(exemplar["dst_cell_id"]) in cell_index.index and receptor_genes:
                row_idx = int(cell_index[str(exemplar["dst_cell_id"])])
                cols = [gene_index[gene] for gene in receptor_genes]
                dst_slice = adata.layers["lognorm"][row_idx, cols]
                dst_array = dst_slice.toarray() if hasattr(dst_slice, "toarray") else np.asarray(dst_slice)
                dst_expr = float(np.asarray(dst_array).mean())

            exemplar_rows.append(
                {
                    "refined_program_id": program_id,
                    "pair_key": exemplar["pair_key"],
                    "graph_id": exemplar["graph_id"],
                    "task_name": primary_task,
                    "cohort_name": cohort_name,
                    "source_id": exemplar["source_id"],
                    "donor_id": exemplar["donor_id"],
                    "src_cell_id": exemplar["src_cell_id"],
                    "dst_cell_id": exemplar["dst_cell_id"],
                    "attention": exemplar["attention"],
                    "contribution": exemplar["contribution"],
                    "ligand_expression_value": src_expr,
                    "receptor_expression_value": dst_expr,
                    "sender_state_id": state_lookup.get(
                        f"lr_pair::{str(exemplar['pair_key']).lower().replace('|', '_').replace('->', '_to_')}",
                        {},
                    ).get("sender_state_id", "NA"),
                    "receiver_state_id": state_lookup.get(
                        f"lr_pair::{str(exemplar['pair_key']).lower().replace('|', '_').replace('->', '_to_')}",
                        {},
                    ).get("receiver_state_id", "NA"),
                    "exemplar_status": cohort_name.replace("_known_age", "") + "_exemplar",
                    "figure_path": str(figure_path),
                }
            )
            manifest_rows.append(
                {
                    "refined_program_id": program_id,
                    "pair_key": exemplar["pair_key"],
                    "cohort_name": cohort_name,
                    "graph_id": exemplar["graph_id"],
                    "graph_path": exemplar["graph_path"],
                    "figure_path": str(figure_path),
                }
            )

    pd.DataFrame(exemplar_rows).to_csv(output_dir / "step3_exemplar_edges.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(output_dir / "step3_exemplar_subgraph_manifest.csv", index=False)


if __name__ == "__main__":
    main()
