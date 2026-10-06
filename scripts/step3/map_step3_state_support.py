#!/usr/bin/env python3.12
"""Map Step 3 candidates onto sender and receiver states."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import project_path
from scripts.step3.common import TASK_TO_CELL_TYPES, parse_gene_string, split_pair_key, top_program_anchor_pairs


DEFAULT_OUTPUT_DIR = Path("data/processed/step3/05_states")
STATE_FILES = {
    "microglia": "step3_microglia_states.h5ad",
    "dopaminergic neuron": "step3_dopaminergic_states.h5ad",
    "astrocyte": "step3_astrocyte_states.h5ad",
    "endothelial cell": "step3_endothelial_states.h5ad",
    "neuron": "step3_neuron_states.h5ad",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map Step 3 state support.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def build_state_objects(state_root: Path) -> tuple[dict[str, ad.AnnData], dict[str, dict[str, str]], dict[str, pd.DataFrame], dict[str, pd.Series]]:
    datasets: dict[str, ad.AnnData] = {}
    state_lookup: dict[str, dict[str, str]] = {}
    state_means: dict[str, pd.DataFrame] = {}
    state_freq: dict[str, pd.Series] = {}
    for lineage, filename in STATE_FILES.items():
        path = state_root / filename
        if not path.exists():
            continue
        adata = ad.read_h5ad(path)
        datasets[lineage] = adata
        lookup = adata.obs.set_index("cell_id")["state_id"].astype(str).to_dict()
        state_lookup[lineage] = lookup
        expr = pd.DataFrame(np.asarray(adata.X), index=adata.obs["state_id"].astype(str), columns=adata.var_names.astype(str))
        state_means[lineage] = expr.groupby(level=0).mean()
        state_freq[lineage] = adata.obs["state_id"].astype(str).value_counts(normalize=True)
    return datasets, state_lookup, state_means, state_freq


def mean_state_expression(state_mean: pd.DataFrame, genes: list[str]) -> pd.Series:
    present = [gene for gene in genes if gene in state_mean.columns]
    if not present:
        return pd.Series(0.0, index=state_mean.index)
    return state_mean[present].mean(axis=1)


def pair_state_rows(
    registry_pair: pd.Series,
    task_row: pd.Series,
    attention_edges: pd.DataFrame,
    state_lookup: dict[str, dict[str, str]],
    state_means: dict[str, pd.DataFrame],
    state_freq: dict[str, pd.Series],
) -> list[dict[str, object]]:
    pair_key = str(registry_pair.get("candidate_id", getattr(registry_pair, "name", "")))
    candidate_key = str(registry_pair.get("candidate_key", f"lr_pair::{pair_key.lower().replace('|', '_').replace('->', '_to_')}"))
    primary_task = str(task_row["primary_task"])
    sender_type, receiver_type = TASK_TO_CELL_TYPES.get(primary_task, ("", ""))
    if sender_type not in state_means or receiver_type not in state_means:
        return []

    ligand_text, receptor_text = split_pair_key(pair_key)
    ligand_genes = parse_gene_string(ligand_text)
    receptor_genes = parse_gene_string(receptor_text)
    sender_scores = mean_state_expression(state_means[sender_type], ligand_genes)
    receiver_scores = mean_state_expression(state_means[receiver_type], receptor_genes)

    local_edges = attention_edges.loc[
        attention_edges["pair_key"].eq(pair_key) & attention_edges["task_name"].eq(primary_task)
    ].copy()
    local_edges["sender_state_id"] = local_edges["src_cell_id"].map(state_lookup[sender_type])
    local_edges["receiver_state_id"] = local_edges["dst_cell_id"].map(state_lookup[receiver_type])
    local_edges = local_edges.dropna(subset=["sender_state_id", "receiver_state_id"])

    if local_edges.empty:
        top_sender = sender_scores.sort_values(ascending=False).head(1)
        top_receiver = receiver_scores.sort_values(ascending=False).head(1)
        return [
            {
                "candidate_id": candidate_key,
                "candidate_label": pair_key,
                "candidate_level": "pair",
                "sender_cell_type": sender_type,
                "sender_state_id": top_sender.index[0] if len(top_sender) else "unknown",
                "receiver_cell_type": receiver_type,
                "receiver_state_id": top_receiver.index[0] if len(top_receiver) else "unknown",
                "ligand_expression_support": float(top_sender.iloc[0]) if len(top_sender) else 0.0,
                "receptor_expression_support": float(top_receiver.iloc[0]) if len(top_receiver) else 0.0,
                "high_contribution_edge_enrichment": 0.0,
                "state_pair_rank": 1,
                "state_support_status": "expression_only",
            }
        ]

    grouped = (
        local_edges.groupby(["sender_state_id", "receiver_state_id"], observed=False)
        .agg(
            contribution_sum=("contribution", "sum"),
            contribution_mean=("contribution", "mean"),
            edge_hits=("graph_id", "count"),
        )
        .reset_index()
        .sort_values(["contribution_sum", "edge_hits"], ascending=False)
    )
    total_hits = max(float(grouped["edge_hits"].sum()), 1.0)
    rows: list[dict[str, object]] = []
    for rank, row in enumerate(grouped.head(3).itertuples(index=False), start=1):
        expected = (
            float(state_freq[sender_type].get(str(row.sender_state_id), 0.0))
            * float(state_freq[receiver_type].get(str(row.receiver_state_id), 0.0))
            * total_hits
        )
        enrichment = float(row.edge_hits) / expected if expected > 0 else float(row.edge_hits)
        ligand_support = float(sender_scores.get(str(row.sender_state_id), 0.0))
        receptor_support = float(receiver_scores.get(str(row.receiver_state_id), 0.0))
        rows.append(
            {
                "candidate_id": candidate_key,
                "candidate_label": pair_key,
                "candidate_level": "pair",
                "sender_cell_type": sender_type,
                "sender_state_id": str(row.sender_state_id),
                "receiver_cell_type": receiver_type,
                "receiver_state_id": str(row.receiver_state_id),
                "ligand_expression_support": ligand_support,
                "receptor_expression_support": receptor_support,
                "high_contribution_edge_enrichment": enrichment,
                "state_pair_rank": rank,
                "state_support_status": "supported" if ligand_support > 0.1 and receptor_support > 0.1 else "weak_directionality",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "step3_state_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    state_root = project_path("data/processed/step3/05_states")
    datasets, state_lookup, state_means, state_freq = build_state_objects(state_root)
    registry = pd.read_csv(project_path("data/processed/step3/01_registry/step3_candidate_registry.csv"))
    task_support = pd.read_csv(project_path("data/processed/step3/04_task_support/step3_task_support_matrix.csv"))
    refined_map = pd.read_csv(project_path("data/processed/step3/02_programs/step3_refined_program_map.csv"))
    ranked_pairs = pd.read_csv(project_path("data/processed/step2/08_interpretation/step2_ranked_lr_pairs.csv"))
    graph_manifest = pd.read_parquet(project_path("data/processed/step2/05_graphs/step2_graph_manifest.parquet"))
    age_attention = pd.read_parquet(project_path("data/processed/step2/06_models/age_model/age_model_attention_summary.parquet"))
    pd_attention = pd.read_parquet(project_path("data/processed/step2/06_models/pd_model/pd_model_attention_summary.parquet"))

    pair_lookup = refined_map[["pair_key"]].drop_duplicates().merge(
        ranked_pairs[["compact_pair_index", "pair_key"]],
        on="pair_key",
        how="left",
    ).rename(columns={"compact_pair_index": "top_pair_index"})
    graph_meta = graph_manifest[["graph_id", "task_name"]].drop_duplicates()

    def attention_frame(att: pd.DataFrame, model_name: str) -> pd.DataFrame:
        local = (
            att.loc[att["split"].eq("val")]
            .merge(pair_lookup, on="top_pair_index", how="left")
            .merge(graph_meta, on="graph_id", how="left")
            .copy()
        )
        local["model_name"] = model_name
        local["contribution"] = local["attention"].astype(float) * local["top_pair_score"].astype(float)
        return local

    attention_edges = pd.concat(
        [attention_frame(age_attention, "age"), attention_frame(pd_attention, "pd")],
        ignore_index=True,
    )

    pair_candidates = registry.loc[registry["entity_level"].eq("pair")].copy()
    task_index = task_support.set_index("candidate_id")
    pair_rows: list[dict[str, object]] = []
    anchor_pairs = top_program_anchor_pairs(refined_map, ranked_pairs, top_n=5)

    pair_interest = set(pair_candidates["candidate_id"].astype(str)) | set(anchor_pairs["pair_key"].astype(str))
    pair_interest_df = refined_map.loc[refined_map["pair_key"].isin(pair_interest)].copy()
    pair_interest_df["candidate_id"] = pair_interest_df["pair_key"].map(
        registry.set_index("candidate_id")["candidate_key"].to_dict()
    )
    pair_interest_df["candidate_id"] = pair_interest_df["candidate_id"].fillna(pair_interest_df["pair_key"].map(lambda value: f"lr_pair::{value.lower().replace('|', '_').replace('->', '_to_')}"))

    registry_by_pair = registry.loc[registry["entity_level"].eq("pair")].set_index("candidate_id")
    for pair_label, row in pair_interest_df.set_index("pair_key").iterrows():
        candidate_key = str(row["candidate_id"])
        if candidate_key not in task_index.index:
            continue
        registry_row = registry_by_pair.loc[pair_label] if pair_label in registry_by_pair.index else pd.Series({"candidate_id": pair_label, "candidate_key": candidate_key})
        pair_rows.extend(
            pair_state_rows(
                registry_row,
                task_index.loc[candidate_key],
                attention_edges,
                state_lookup,
                state_means,
                state_freq,
            )
        )
    pair_state_support = pd.DataFrame(pair_rows)

    program_rows: list[dict[str, object]] = []
    for program_id, frame in anchor_pairs.groupby("refined_program_id", observed=False):
        program_candidate = f"program::{program_id}"
        if program_candidate not in task_index.index:
            continue
        primary_task = str(task_index.loc[program_candidate, "primary_task"])
        sender_type, receiver_type = TASK_TO_CELL_TYPES.get(primary_task, ("", ""))
        if sender_type not in state_means or receiver_type not in state_means:
            continue
        relevant_pairs = {f"lr_pair::{pair.lower().replace('|', '_').replace('->', '_to_')}" for pair in frame["pair_key"].astype(str)}
        local = pair_state_support.loc[
            pair_state_support["candidate_id"].isin(relevant_pairs)
            & pair_state_support["sender_cell_type"].eq(sender_type)
            & pair_state_support["receiver_cell_type"].eq(receiver_type)
        ].copy()
        if local.empty:
            continue
        grouped = (
            local.groupby(["sender_state_id", "receiver_state_id"], observed=False)
            .agg(
                ligand_expression_support=("ligand_expression_support", "mean"),
                receptor_expression_support=("receptor_expression_support", "mean"),
                high_contribution_edge_enrichment=("high_contribution_edge_enrichment", "mean"),
            )
            .reset_index()
            .sort_values("high_contribution_edge_enrichment", ascending=False)
        )
        for rank, row in enumerate(grouped.head(3).itertuples(index=False), start=1):
            program_rows.append(
                {
                    "candidate_id": program_candidate,
                    "candidate_label": program_id,
                    "candidate_level": "program",
                    "sender_cell_type": sender_type,
                    "sender_state_id": str(row.sender_state_id),
                    "receiver_cell_type": receiver_type,
                    "receiver_state_id": str(row.receiver_state_id),
                    "ligand_expression_support": float(row.ligand_expression_support),
                    "receptor_expression_support": float(row.receptor_expression_support),
                    "high_contribution_edge_enrichment": float(row.high_contribution_edge_enrichment),
                    "state_pair_rank": rank,
                    "state_support_status": "supported"
                    if float(row.ligand_expression_support) > 0.1 and float(row.receptor_expression_support) > 0.1
                    else "weak_directionality",
                }
            )

    candidate_state_support = pd.concat([pair_state_support, pd.DataFrame(program_rows)], ignore_index=True, sort=False)
    candidate_state_support.to_csv(output_dir / "step3_candidate_state_support.csv", index=False)

    if not candidate_state_support.empty:
        heatmap_source = candidate_state_support.loc[candidate_state_support["state_pair_rank"].eq(1)].copy()
        heatmap_source["state_pair"] = heatmap_source["sender_state_id"] + " -> " + heatmap_source["receiver_state_id"]
        top_rows = heatmap_source.groupby("candidate_label", observed=False)["high_contribution_edge_enrichment"].max().sort_values(ascending=False).head(20).index
        heatmap = heatmap_source.loc[heatmap_source["candidate_label"].isin(top_rows)].pivot_table(
            index="candidate_label",
            columns="state_pair",
            values="high_contribution_edge_enrichment",
            fill_value=0.0,
        )
        if not heatmap.empty:
            fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * len(heatmap.index))))
            sns.heatmap(heatmap, cmap="mako", ax=ax)
            ax.set_title("Top candidate state-pair enrichment")
            fig.tight_layout()
            fig.savefig(plot_dir / "step3_candidate_state_support_heatmap.png", dpi=220, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main()
