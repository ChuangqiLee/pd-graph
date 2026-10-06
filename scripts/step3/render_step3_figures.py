#!/usr/bin/env python3.12
"""Render the Step 3 interpretation package figures and dossier."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import markdown_table, project_path, write_markdown


DEFAULT_CONFIG = Path("configs/step3/step3_figures.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Step 3 figures and reports.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    output_dir = project_path(config["paths"]["output_dir"])
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    top10 = pd.read_csv(project_path("data/processed/step3/09_top10/step3_top10_refined_programs.csv"))
    pair_support = pd.read_csv(project_path("data/processed/step3/09_top10/step3_pair_support_table.csv"))
    state_support = pd.read_csv(project_path("data/processed/step3/05_states/step3_candidate_state_support.csv"))
    confounds = pd.read_csv(project_path("data/processed/step3/07_confounds/step3_confound_summary.csv"))
    trajectory = pd.read_csv(project_path("data/processed/step3/03_donor_patterns/step3_trajectory_summary.csv"))
    exemplars = pd.read_csv(project_path("data/processed/step3/06_exemplars/step3_exemplar_edges.csv"))

    evidence = top10[
        [
            "refined_program_name",
            "step3_program_score",
            "anchor_pair_score_mean",
            "anchor_pair_score_max",
            "program_coherence",
            "donor_trajectory_coherence",
            "validation_readiness",
        ]
    ].set_index("refined_program_name")
    fig, ax = plt.subplots(figsize=(9, max(5, 0.45 * len(evidence.index))))
    sns.heatmap(evidence, cmap="viridis", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Step 3 candidate evidence heatmap")
    fig.tight_layout()
    evidence_path = figure_dir / "step3_candidate_evidence_heatmap.png"
    fig.savefig(evidence_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    evidence.reset_index().to_csv(output_dir / "step3_candidate_evidence_matrix.csv", index=False)

    top_state = state_support.loc[
        state_support["candidate_id"].isin(top10["candidate_id"]) & state_support["state_pair_rank"].eq(1)
    ].copy()
    if not top_state.empty:
        top_state["state_pair"] = top_state["sender_state_id"] + " -> " + top_state["receiver_state_id"]
        pivot = top_state.pivot_table(
            index="candidate_label",
            columns="state_pair",
            values="high_contribution_edge_enrichment",
            fill_value=0.0,
        )
        fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(pivot.index))))
        sns.heatmap(pivot, cmap="mako", ax=ax)
        ax.set_title("State-by-candidate enrichment")
        fig.tight_layout()
        fig.savefig(figure_dir / "step3_state_by_candidate_heatmap.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    for row in top10.itertuples(index=False):
        source = project_path(f"data/processed/step3/03_donor_patterns/step3_donor_trajectory_plots/{row.candidate_id.replace('::', '_')}.png")
        if source.exists():
            shutil.copy2(source, figure_dir / f"{row.candidate_id.replace('::', '_')}_trajectory.png")
    for row in exemplars.loc[exemplars["refined_program_id"].isin(top10["refined_program_id"])].itertuples(index=False):
        source = Path(str(row.figure_path))
        if source.exists():
            shutil.copy2(source, figure_dir / source.name)

    graph = nx.DiGraph()
    for row in top10.itertuples(index=False):
        sender = str(row.dominant_sender_state)
        receiver = str(row.dominant_receiver_state)
        graph.add_edge(sender, receiver, label=row.refined_program_name, weight=row.step3_program_score)
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=13)
    nx.draw_networkx_nodes(graph, pos, node_color="#2a9d8f", node_size=900, ax=ax)
    nx.draw_networkx_edges(graph, pos, width=[2.0 + 2.0 * graph[u][v]["weight"] for u, v in graph.edges], alpha=0.7, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
    ax.set_title("Step 3 mechanistic summary schematic")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(figure_dir / "step3_mechanistic_summary_schematic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    promoted_pairs = pair_support.loc[
        pair_support["refined_program_id"].isin(top10["refined_program_id"]) & pair_support["step3_pair_score"].ge(0.45)
    ].copy()
    promoted_pairs.to_csv(output_dir / "step3_pair_appendix.csv", index=False)

    program_sections: list[str] = []
    confound_index = confounds.set_index("candidate_id")
    trajectory_index = trajectory.set_index("candidate_id")
    for row in top10.itertuples(index=False):
        confound_row = confound_index.loc[row.candidate_id] if row.candidate_id in confound_index.index else None
        traj_row = trajectory_index.loc[row.candidate_id] if row.candidate_id in trajectory_index.index else None
        exemplar_subset = exemplars.loc[exemplars["refined_program_id"].eq(row.refined_program_id)].head(3)
        exemplar_lines = "\n".join(f"- `{item.cohort_name}`: `{item.pair_key}` in `{item.graph_id}`" for item in exemplar_subset.itertuples(index=False))
        program_sections.append(
            "\n".join(
                [
                    f"## {row.refined_program_name}",
                    f"- Verdict: `{row.final_step3_tier}`",
                    f"- Anchor pairs: `{row.anchor_pairs}`",
                    f"- Dominant task: `{row.dominant_task}`",
                    f"- Dominant states: `{row.dominant_sender_state}` -> `{row.dominant_receiver_state}`",
                    f"- Interpretation class: `{row.interpretation_class}`",
                    f"- Caveats: `{row.caveat_summary}`",
                    f"- Step 4 priority: `{row.step4_validation_priority}`",
                    f"- Trajectory: `{traj_row['trajectory_class']}`" if traj_row is not None else "- Trajectory: `NA`",
                    f"- Confound status: `{confound_row['promotion_status']}`" if confound_row is not None else "- Confound status: `NA`",
                    "- Exemplars:",
                    exemplar_lines if exemplar_lines else "- None available",
                    "",
                ]
            )
        )

    dossier = "\n".join(
        [
            "# Step 3 Biological Interpretation Dossier",
            "",
            "## Top 10 Programs",
            markdown_table(top10.fillna("NA")),
            "",
            "## Figure Package",
            f"- Evidence heatmap: `{evidence_path}`",
            f"- State heatmap: `{figure_dir / 'step3_state_by_candidate_heatmap.png'}`",
            f"- Mechanistic schematic: `{figure_dir / 'step3_mechanistic_summary_schematic.png'}`",
            "",
            *program_sections,
        ]
    )
    write_markdown(output_dir / "step3_biological_interpretation_dossier.md", dossier)

    onepager = "\n".join(
        [
            "# Step 3 Summary Onepager",
            "",
            "## Key programs",
            markdown_table(top10[["refined_program_name", "final_step3_tier", "dominant_task", "step4_validation_priority"]].fillna("NA")),
            "",
            "## Main message",
            "- The current run is led by APP-centered stress signaling, guidance programs, glutamatergic receptor programs, and synaptic/contact recognition programs rather than by classical cytokines alone.",
        ]
    )
    write_markdown(output_dir / "step3_summary_onepager.md", onepager)


if __name__ == "__main__":
    main()

