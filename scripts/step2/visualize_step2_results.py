#!/usr/bin/env python3
"""Generate a complete static visualization pack for Step 2 outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import fill

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import markdown_table, project_path, write_markdown


DEFAULT_STEP2_ROOT = Path("data/processed/step2")
DEFAULT_OUTPUT_DIR = Path("data/processed/step2/10_visualizations")
RANDOM_SEED = 13

LONG_SOURCE_ID = "b0f0b447-ac37-45b0-b1bf-5c0b7d871120"
TASK_ORDER = [
    "microglia_to_dopaminergic_neuron",
    "astrocyte_to_dopaminergic_neuron",
    "endothelial_to_dopaminergic_neuron",
    "microglia_to_neuron",
]
DISEASE_ORDER = ["control", "lbd", "pd"]
DISEASE_PALETTE = {"control": "#4C78A8", "lbd": "#72B7B2", "pd": "#E45756", "unknown": "#9D9DA1"}
INTERPRETATION_PALETTE = {
    "pd_amplified_aging": "#D95F02",
    "pd_specific": "#1B9E77",
    "normal_aging_associated": "#7570B3",
    "unstable_or_source_sensitive": "#A6761D",
}
SIDE_PALETTE = {"ligand": "#2A9D8F", "receptor": "#E76F51"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Step 2 results.")
    parser.add_argument("--step2-root", default=str(DEFAULT_STEP2_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["figure.titlesize"] = 18
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def pretty_source(source_id: str) -> str:
    mapping = {
        LONG_SOURCE_ID: "CELLxGENE SNpc",
        "GSE126836": "GSE126836",
        "GSE140231": "GSE140231",
        "GSE157783": "GSE157783",
        "GSE193688": "GSE193688",
    }
    return mapping.get(str(source_id), str(source_id))


def pretty_task(task_name: str) -> str:
    mapping = {
        "microglia_to_dopaminergic_neuron": "Microglia -> DA neuron",
        "astrocyte_to_dopaminergic_neuron": "Astrocyte -> DA neuron",
        "endothelial_to_dopaminergic_neuron": "Endothelial -> DA neuron",
        "microglia_to_neuron": "Microglia -> neuron",
    }
    return mapping.get(str(task_name), str(task_name).replace("_", " "))


def pretty_bin(value: str) -> str:
    mapping = {
        "pd_amplified_aging": "PD amplified aging",
        "pd_specific": "PD specific",
        "normal_aging_associated": "Normal aging associated",
        "unstable_or_source_sensitive": "Unstable/source-sensitive",
    }
    return mapping.get(str(value), str(value).replace("_", " "))


def pretty_cohort(value: str) -> str:
    mapping = {
        "control_known_age": "Control, known age",
        "pd_known_age": "PD, known age",
        "unknown_age_auxiliary": "Unknown-age auxiliary",
        "lbd_bridge": "LBD bridge",
    }
    return mapping.get(str(value), str(value).replace("_", " "))


def short_group_id(value: str) -> str:
    text = str(value)
    if "::" in text:
        return text.split("::", 1)[1]
    return text


def build_palette(values: list[str], palette_name: str = "tab10") -> dict[str, tuple[float, float, float]]:
    clean = [str(value) for value in values]
    palette = sns.color_palette(palette_name, n_colors=max(3, len(clean)))
    return {value: palette[index] for index, value in enumerate(clean)}


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def zscore_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for index in result.index:
        row = result.loc[index]
        std = float(row.std(ddof=0))
        if std < 1e-9 or np.isnan(std):
            result.loc[index] = 0.0
        else:
            result.loc[index] = (row - row.mean()) / std
    return result


def annotate_barh(ax: plt.Axes, fmt: str = "{:.2f}") -> None:
    for patch in ax.patches:
        width = patch.get_width()
        ax.text(
            width + (0.01 * max(1.0, ax.get_xlim()[1] - ax.get_xlim()[0])),
            patch.get_y() + patch.get_height() / 2,
            fmt.format(width),
            va="center",
            fontsize=10,
        )


def load_tables(step2_root: Path) -> dict[str, object]:
    data: dict[str, object] = {}
    data["backbone"] = pd.read_parquet(step2_root / "00_backbone/step2_backbone_manifest.parquet")
    data["comm_panel"] = pd.read_csv(step2_root / "01_comm_panel/step2_comm_gene_panel.csv")
    data["cohort_manifest"] = pd.read_csv(step2_root / "02_cohorts/step2_graph_manifest.csv")
    data["baseline_metrics"] = pd.read_csv(step2_root / "04_baselines/step2_baseline_metrics.csv")
    data["pathway_scores"] = pd.read_parquet(step2_root / "04_baselines/step2_pathway_scores.parquet")
    data["graph_manifest"] = pd.read_parquet(step2_root / "05_graphs/step2_graph_manifest.parquet")
    data["age_pred"] = pd.read_csv(step2_root / "06_models/age_model/age_model_predictions.csv")
    data["age_fold_metrics"] = pd.read_csv(step2_root / "06_models/age_model/age_model_fold_metrics.csv")
    data["pd_pred"] = pd.read_csv(step2_root / "06_models/pd_model/pd_model_predictions.csv")
    data["pd_fold_metrics"] = pd.read_csv(step2_root / "06_models/pd_model/pd_model_fold_metrics.csv")
    data["pd_lbd_scores"] = pd.read_csv(step2_root / "06_models/pd_model/pd_model_lbd_bridge_scores.csv")
    data["label_confidence"] = pd.read_csv(step2_root / "07_robustness/step2_label_confidence_sensitivity.csv")
    data["source_coverage"] = pd.read_csv(step2_root / "07_robustness/step2_source_task_coverage.csv")
    data["receiver_robustness"] = pd.read_csv(step2_root / "07_robustness/step2_receiver_robustness.csv")
    data["cell_count_robustness"] = pd.read_csv(step2_root / "07_robustness/step2_cell_count_robustness.csv")
    data["lbd_summary"] = pd.read_csv(step2_root / "07_robustness/step2_lbd_bridge_summary.csv")
    data["holdout_summary"] = pd.read_csv(step2_root / "07_robustness/step2_holdout_gnn_summary.csv")
    data["holdout_counts"] = pd.read_csv(step2_root / "07_robustness/gse157783_holdout/gse157783_patient_celltype_counts.csv")
    data["holdout_baseline"] = pd.read_csv(step2_root / "07_robustness/gse157783_holdout/gse157783_mgda_baseline_scores.csv")
    data["ranked_pathways"] = pd.read_csv(step2_root / "08_interpretation/step2_ranked_pathways.csv")
    data["ranked_lr_pairs"] = pd.read_csv(step2_root / "08_interpretation/step2_ranked_lr_pairs.csv")
    data["edge_contexts"] = pd.read_csv(step2_root / "08_interpretation/step2_top_edge_contexts.csv")
    data["step3_candidates"] = pd.read_csv(step2_root / "09_exports/step2_top_candidates_for_step3.csv")
    data["step4_candidates"] = pd.read_csv(step2_root / "09_exports/step2_top_candidates_for_step4.csv")
    data["top_pathway_scores_by_donor"] = pd.read_csv(step2_root / "09_exports/step2_top_pathway_scores_by_donor.csv")
    data["top_edge_importance"] = pd.read_csv(step2_root / "09_exports/step2_top_edge_importance_summary.csv")
    data["embedding_export"] = ad.read_h5ad(step2_root / "09_exports/step2_embedding_exports.h5ad")
    return data


def main_task_metadata(graph_manifest: pd.DataFrame) -> pd.DataFrame:
    subset = graph_manifest.loc[graph_manifest["task_name"].eq("microglia_to_dopaminergic_neuron")].copy()
    columns = [
        "split_group_id",
        "cohort_name",
        "source_id",
        "disease_group",
        "age_years",
        "brain_region_standardized",
        "sex_standardized",
        "n_nodes",
        "n_edges",
    ]
    return subset[columns].drop_duplicates(subset=["split_group_id"]).reset_index(drop=True)


def donor_level_age_validation(age_pred: pd.DataFrame, donor_meta: pd.DataFrame) -> pd.DataFrame:
    val = age_pred.loc[age_pred["split"].eq("val")].copy()
    donor = (
        val.groupby(["fold", "split_group_id", "age_years"], as_index=False)["pred_age_years"]
        .mean()
        .rename(columns={"pred_age_years": "pred_age_years_mean"})
    )
    donor["residual_years"] = donor["pred_age_years_mean"] - donor["age_years"]
    meta = donor_meta.drop(columns=["age_years"], errors="ignore")
    return donor.merge(meta, on="split_group_id", how="left")


def donor_level_pd_validation(pd_pred: pd.DataFrame, donor_meta: pd.DataFrame) -> pd.DataFrame:
    val = pd_pred.loc[pd_pred["split"].eq("val")].copy()
    donor = (
        val.groupby(["fold", "split_group_id", "disease_target"], as_index=False)
        .agg(pd_probability_mean=("pd_probability", "mean"), decision_threshold=("decision_threshold", "mean"))
    )
    donor["predicted_label"] = donor["pd_probability_mean"].ge(donor["decision_threshold"]).astype(int)
    return donor.merge(donor_meta, on="split_group_id", how="left")


def build_embedding_coordinates(embedding_export: ad.AnnData) -> pd.DataFrame:
    coords = embedding_export.obs.copy()
    coords.index.name = "split_group_id"
    coords = coords.reset_index()
    for obsm_key, prefix in [("X_age_embedding", "age"), ("X_pd_embedding", "pd")]:
        matrix = np.asarray(embedding_export.obsm[obsm_key], dtype=np.float32)
        if matrix.shape[0] >= 4:
            n_neighbors = max(2, min(5, matrix.shape[0] - 1))
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.35, random_state=RANDOM_SEED)
            projected = reducer.fit_transform(matrix)
        else:
            projected = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(matrix)
        coords[f"{prefix}_umap_1"] = projected[:, 0]
        coords[f"{prefix}_umap_2"] = projected[:, 1]
    return coords


def plot_backbone_overview(backbone: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    donors = backbone.drop_duplicates(subset=["split_group_id"]).copy()
    top_cell_types = backbone["cell_type_standardized_final"].value_counts().head(12).sort_values()
    source_disease = (
        donors.groupby(["source_id", "disease_group"], observed=False)["split_group_id"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=[column for column in DISEASE_ORDER if column in donors["disease_group"].unique()], fill_value=0)
    )
    label_source = backbone["cell_type_label_source"].astype("string").fillna("unknown").value_counts()
    age_plot = donors.loc[donors["age_years"].notna()].copy()
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))

    ax = axes[0, 0]
    ax.barh(top_cell_types.index.astype(str), top_cell_types.values, color="#4C78A8")
    ax.set_title("Top Step 2 cell types")
    ax.set_xlabel("Cells")

    ax = axes[0, 1]
    source_disease.index = [pretty_source(value) for value in source_disease.index]
    source_disease.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[DISEASE_PALETTE.get(column, "#9D9DA1") for column in source_disease.columns],
        width=0.82,
    )
    ax.set_title("Unique donors by source and disease")
    ax.set_xlabel("")
    ax.set_ylabel("Donors")
    ax.legend(title="Disease", bbox_to_anchor=(1.02, 1), loc="upper left")

    ax = axes[1, 0]
    if not age_plot.empty:
        sns.boxplot(
            data=age_plot,
            x="disease_group",
            y="age_years",
            hue="disease_group",
            order=[value for value in DISEASE_ORDER if value in age_plot["disease_group"].unique()],
            palette=DISEASE_PALETTE,
            dodge=False,
            ax=ax,
        )
        sns.stripplot(
            data=age_plot,
            x="disease_group",
            y="age_years",
            hue="disease_group",
            order=[value for value in DISEASE_ORDER if value in age_plot["disease_group"].unique()],
            palette=DISEASE_PALETTE,
            dodge=False,
            ax=ax,
            size=5,
            alpha=0.8,
            linewidth=0.4,
            edgecolor="white",
        )
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    ax.set_title("Donor age distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Age (years)")

    ax = axes[1, 1]
    ax.bar(label_source.index.astype(str), label_source.values, color="#2A9D8F")
    ax.set_title("Cell type label provenance")
    ax.set_xlabel("")
    ax.set_ylabel("Cells")
    ax.tick_params(axis="x", rotation=20)

    fig.suptitle(
        f"Step 2 backbone overview: {backbone.shape[0]:,} cells across {donors['split_group_id'].nunique()} donor groups",
        y=1.02,
    )
    return save_figure(fig, output_dir / "00_overview/step2_backbone_overview.png", dpi)


def plot_backbone_qc(backbone: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    primary = backbone.loc[backbone["is_primary_cell_type"].fillna(False)].copy()
    top_ct = primary["cell_type_standardized_final"].value_counts().head(6).index.tolist()
    qc = primary.loc[primary["cell_type_standardized_final"].isin(top_ct)].copy()
    if qc.shape[0] > 30000:
        qc = qc.sample(n=30000, random_state=RANDOM_SEED)
    qc["cell_type_standardized_final"] = pd.Categorical(
        qc["cell_type_standardized_final"],
        categories=top_ct,
        ordered=True,
    )
    metrics = [
        ("qc_total_counts", "UMI counts"),
        ("qc_n_genes_by_counts", "Genes detected"),
        ("qc_pct_mt", "Mitochondrial %"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    for ax, (metric, title) in zip(axes, metrics):
        sns.violinplot(
            data=qc,
            x="cell_type_standardized_final",
            y=metric,
            hue="cell_type_standardized_final",
            inner="quartile",
            cut=0,
            linewidth=0.9,
            palette="Set2",
            ax=ax,
        )
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Counts")
    axes[1].set_ylabel("Genes")
    axes[2].set_ylabel("Percent")
    fig.suptitle("Step 2 backbone QC by major cell type", y=1.02)
    return save_figure(fig, output_dir / "00_overview/step2_backbone_qc.png", dpi)


def plot_comm_panel(comm_panel: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    membership = pd.DataFrame(
        {
            "CellChat": [int(comm_panel["in_cellchat"].sum())],
            "NicheNet": [int(comm_panel["in_nichenet"].sum())],
            "Supplement": [int(comm_panel["is_supplement"].sum())],
            "Sentinel": [int(comm_panel["is_sentinel"].sum())],
        }
    ).melt(var_name="membership", value_name="n_genes")

    supplement_counts: dict[str, int] = {}
    for value in comm_panel["supplement_categories"].fillna(""):
        for item in [part.strip() for part in str(value).split("|") if part.strip()]:
            supplement_counts[item] = supplement_counts.get(item, 0) + 1
    supplement_series = pd.Series(supplement_counts).sort_values(ascending=False).head(10)

    source_coverage = (
        comm_panel["present_source_count"]
        .fillna(0)
        .astype(int)
        .value_counts()
        .sort_index()
        .rename_axis("present_source_count")
        .reset_index(name="n_genes")
    )

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    sns.barplot(data=membership, x="membership", y="n_genes", hue="membership", palette="viridis", ax=axes[0])
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()
    axes[0].set_title("Communication panel membership")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Genes")

    axes[1].barh(supplement_series.index.astype(str)[::-1], supplement_series.values[::-1], color="#F4A261")
    axes[1].set_title("Supplement category coverage")
    axes[1].set_xlabel("Genes")
    axes[1].set_ylabel("")

    axes[2].bar(source_coverage["present_source_count"].astype(str), source_coverage["n_genes"], color="#2A9D8F")
    axes[2].set_title("Recovered genes by source count")
    axes[2].set_xlabel("Number of sources recovered")
    axes[2].set_ylabel("Genes")

    fig.suptitle(
        f"Step 2 communication panel: {comm_panel.shape[0]:,} genes, "
        f"{int(comm_panel['is_sentinel'].sum())} sentinel genes recovered",
        y=1.02,
    )
    return save_figure(fig, output_dir / "00_overview/step2_comm_panel_summary.png", dpi)


def plot_cohort_coverage(
    cohort_manifest: pd.DataFrame,
    source_coverage: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path:
    included = cohort_manifest.loc[cohort_manifest["included_task_default"].fillna(False)].copy()
    eligible = (
        included.groupby(["task_name", "cohort_name"], observed=False)["graph_unit_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    eligible = eligible.reindex(index=[task for task in TASK_ORDER if task in eligible.index])
    eligible.index = [pretty_task(value) for value in eligible.index]
    eligible.columns = [pretty_cohort(value) for value in eligible.columns]

    coverage = source_coverage.copy()
    coverage["source_cohort"] = coverage["source_id"].map(pretty_source) + "\n" + coverage["cohort_name"].map(pretty_cohort)
    source_heatmap = coverage.pivot(index="task_name", columns="source_cohort", values="n_donors").fillna(0)
    source_heatmap = source_heatmap.reindex(index=[task for task in TASK_ORDER if task in source_heatmap.index])
    source_heatmap.index = [pretty_task(value) for value in source_heatmap.index]

    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    sns.heatmap(eligible, annot=True, fmt=".0f", cmap="Blues", linewidths=0.5, cbar_kws={"label": "Graph units"}, ax=axes[0])
    axes[0].set_title("Default-included graph units by task and cohort")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")

    sns.heatmap(
        source_heatmap,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Donors"},
        ax=axes[1],
    )
    axes[1].set_title("Source coverage for robustness")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    fig.suptitle("Step 2 cohort and source coverage", y=1.02)
    return save_figure(fig, output_dir / "00_overview/step2_cohort_coverage.png", dpi)


def plot_graph_statistics(graph_manifest: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    task_palette = build_palette([pretty_task(task) for task in TASK_ORDER], "Set2")
    graph_plot = graph_manifest.copy()
    graph_plot["task_label"] = graph_plot["task_name"].map(pretty_task)
    graph_counts = graph_plot["task_label"].value_counts().reindex([pretty_task(task) for task in TASK_ORDER if task in graph_plot["task_name"].unique()])

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    axes[0].bar(graph_counts.index, graph_counts.values, color=[task_palette[label] for label in graph_counts.index])
    axes[0].set_title("Generated subgraphs by task")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Subgraphs")
    axes[0].tick_params(axis="x", rotation=20)

    sns.scatterplot(
        data=graph_plot,
        x="n_nodes",
        y="n_edges",
        hue="task_label",
        palette=task_palette,
        alpha=0.8,
        s=60,
        ax=axes[1],
    )
    axes[1].set_title("Graph size distribution")
    axes[1].set_xlabel("Nodes")
    axes[1].set_ylabel("Edges")
    axes[1].legend(title="Task", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.suptitle(
        f"Step 2 graph construction: {graph_manifest.shape[0]} subgraphs across "
        f"{graph_manifest['graph_unit_id'].nunique()} donor/sample units",
        y=1.02,
    )
    return save_figure(fig, output_dir / "00_overview/step2_graph_statistics.png", dpi)


def plot_baseline_vs_gnn(
    baseline_metrics: pd.DataFrame,
    age_fold_metrics: pd.DataFrame,
    pd_fold_metrics: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path:
    main_age = baseline_metrics.loc[
        baseline_metrics["task_name"].eq("microglia_to_dopaminergic_neuron")
        & baseline_metrics["analysis"].eq("control_aging")
    ].iloc[0]
    main_pd = baseline_metrics.loc[
        baseline_metrics["task_name"].eq("microglia_to_dopaminergic_neuron")
        & baseline_metrics["analysis"].eq("pd_deviation")
    ].iloc[0]

    age_compare = pd.DataFrame(
        {
            "model": ["Elastic net", "Random forest", "GNN"],
            "metric_value": [main_age["mae_elastic_net"], main_age["mae_random_forest"], age_fold_metrics["mae"].mean()],
        }
    )

    pd_compare = pd.DataFrame(
        {
            "model": ["Logistic", "Random forest", "GNN", "Logistic", "Random forest", "GNN"],
            "metric": ["AUROC", "AUROC", "AUROC", "Balanced accuracy", "Balanced accuracy", "Balanced accuracy"],
            "metric_value": [
                main_pd["auroc_logistic"],
                main_pd["auroc_random_forest"],
                pd_fold_metrics["auroc"].mean(),
                main_pd["balanced_accuracy_logistic"],
                main_pd["balanced_accuracy_random_forest"],
                pd_fold_metrics["balanced_accuracy"].mean(),
            ],
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.barplot(data=age_compare, x="metric_value", y="model", hue="model", palette="Blues_r", ax=axes[0])
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()
    axes[0].set_title("Main task age-model benchmark")
    axes[0].set_xlabel("MAE in years (lower is better)")
    axes[0].set_ylabel("")
    annotate_barh(axes[0])

    sns.barplot(data=pd_compare, x="model", y="metric_value", hue="metric", palette=["#4C78A8", "#E45756"], ax=axes[1])
    axes[1].set_title("Main task PD-model benchmark")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Validation score")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(title="")

    fig.suptitle("Donor-grouped baseline comparison for microglia -> DA neuron", y=1.02)
    return save_figure(fig, output_dir / "01_models/step2_baseline_vs_gnn.png", dpi)


def plot_age_validation(age_val: pd.DataFrame, age_fold_metrics: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    source_palette = build_palette(sorted(age_val["source_id"].map(pretty_source).unique().tolist()), "Dark2")
    age_val = age_val.copy()
    age_val["source_label"] = age_val["source_id"].map(pretty_source)
    mae = float(age_fold_metrics["mae"].mean())

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.scatterplot(
        data=age_val,
        x="age_years",
        y="pred_age_years_mean",
        hue="source_label",
        palette=source_palette,
        s=90,
        ax=axes[0],
    )
    min_age = float(age_val["age_years"].min())
    max_age = float(age_val["age_years"].max())
    axes[0].plot([min_age, max_age], [min_age, max_age], linestyle="--", color="black", linewidth=1.2)
    axes[0].set_title(f"Validation donor predictions (mean MAE {mae:.2f} years)")
    axes[0].set_xlabel("Observed age")
    axes[0].set_ylabel("Predicted age")
    axes[0].legend(title="Source", bbox_to_anchor=(1.02, 1), loc="upper left")

    sns.scatterplot(
        data=age_val,
        x="age_years",
        y="residual_years",
        hue="source_label",
        palette=source_palette,
        s=90,
        ax=axes[1],
        legend=False,
    )
    axes[1].axhline(0, linestyle="--", color="black", linewidth=1.2)
    axes[1].set_title("Residuals vs observed age")
    axes[1].set_xlabel("Observed age")
    axes[1].set_ylabel("Prediction residual")

    fig.suptitle("Step 2 control-aging GNN validation", y=1.02)
    return save_figure(fig, output_dir / "01_models/step2_age_model_validation.png", dpi)


def plot_pd_validation(pd_val: pd.DataFrame, pd_fold_metrics: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    pd_val = pd_val.copy()
    pd_val["disease_group"] = pd_val["disease_group"].astype("string")
    pd_val["disease_group"] = pd_val["disease_group"].fillna(pd_val["disease_target"].map({0: "control", 1: "pd"}))
    pd_val["donor_label"] = pd_val["split_group_id"].map(short_group_id)
    ordered = pd_val.sort_values(["pd_probability_mean", "donor_label"]).reset_index(drop=True)
    ordered["rank"] = np.arange(ordered.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    sns.boxplot(
        data=pd_val,
        x="disease_group",
        y="pd_probability_mean",
        hue="disease_group",
        order=[value for value in DISEASE_ORDER if value in pd_val["disease_group"].unique()],
        palette=DISEASE_PALETTE,
        dodge=False,
        ax=axes[0],
    )
    sns.stripplot(
        data=pd_val,
        x="disease_group",
        y="pd_probability_mean",
        hue="disease_group",
        order=[value for value in DISEASE_ORDER if value in pd_val["disease_group"].unique()],
        palette=DISEASE_PALETTE,
        dodge=False,
        ax=axes[0],
        size=8,
        edgecolor="white",
        linewidth=0.5,
    )
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()
    axes[0].set_title(
        f"Validation donor PD probabilities\nmean AUROC {pd_fold_metrics['auroc'].mean():.3f}, "
        f"balanced accuracy {pd_fold_metrics['balanced_accuracy'].mean():.3f}"
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("PD probability")

    for row in ordered.itertuples(index=False):
        color = DISEASE_PALETTE.get(str(row.disease_group), "#9D9DA1")
        axes[1].vlines(row.rank, 0, row.pd_probability_mean, color=color, alpha=0.75, linewidth=2)
        axes[1].scatter(row.rank, row.pd_probability_mean, color=color, s=85, zorder=3)
        axes[1].scatter(row.rank, row.decision_threshold, marker="_", s=260, color="black", zorder=3)
    axes[1].set_xticks(ordered["rank"])
    axes[1].set_xticklabels(ordered["donor_label"], rotation=45, ha="right")
    axes[1].set_title("Per-donor probability with fold threshold marker")
    axes[1].set_xlabel("Validation donors")
    axes[1].set_ylabel("PD probability")
    custom_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=DISEASE_PALETTE["control"], markersize=9, label="Control"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=DISEASE_PALETTE["pd"], markersize=9, label="PD"),
        Line2D([0], [0], marker="_", color="black", markersize=14, label="Fold threshold"),
    ]
    axes[1].legend(handles=custom_handles, loc="upper left")

    fig.suptitle("Step 2 age-conditioned PD deviation GNN validation", y=1.03)
    return save_figure(fig, output_dir / "01_models/step2_pd_model_validation.png", dpi)


def plot_embedding_umaps(embedding_coords: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    coords = embedding_coords.copy()
    coords["source_label"] = coords["source_id"].map(pretty_source)
    source_palette = build_palette(sorted(coords["source_label"].unique().tolist()), "Set2")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    sns.scatterplot(
        data=coords,
        x="age_umap_1",
        y="age_umap_2",
        hue="age_years",
        palette="viridis",
        s=120,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Age embedding UMAP colored by chronological age")
    axes[0, 0].legend(title="Age", bbox_to_anchor=(1.02, 1), loc="upper left")

    sns.scatterplot(
        data=coords,
        x="age_umap_1",
        y="age_umap_2",
        hue="source_label",
        palette=source_palette,
        s=120,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Age embedding UMAP colored by source")
    axes[0, 1].legend(title="Source", bbox_to_anchor=(1.02, 1), loc="upper left")

    sns.scatterplot(
        data=coords,
        x="pd_umap_1",
        y="pd_umap_2",
        hue="disease_group",
        palette=DISEASE_PALETTE,
        s=120,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("PD embedding UMAP colored by disease group")
    axes[1, 0].legend(title="Disease", bbox_to_anchor=(1.02, 1), loc="upper left")

    sns.scatterplot(
        data=coords,
        x="pd_umap_1",
        y="pd_umap_2",
        hue="pd_probability",
        palette="magma",
        s=120,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("PD embedding UMAP colored by PD probability")
    axes[1, 1].legend(title="PD prob.", bbox_to_anchor=(1.02, 1), loc="upper left")

    for ax in axes.ravel():
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

    fig.suptitle("Donor-level embedding projections exported from Step 2", y=1.01)
    return save_figure(fig, output_dir / "01_models/step2_embedding_umaps.png", dpi)


def plot_label_confidence(label_confidence: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    plot_df = label_confidence.copy()
    plot_df["task_label"] = plot_df["task_name"].map(pretty_task)
    plot_df["cohort_label"] = plot_df["cohort_name"].map(pretty_cohort)
    tasks = ordered_unique(plot_df["task_label"].tolist())
    fig, axes = plt.subplots(1, len(tasks), figsize=(10 * len(tasks), 6), sharey=True)
    if len(tasks) == 1:
        axes = np.array([axes])
    cohort_palette = build_palette(sorted(plot_df["cohort_label"].unique().tolist()), "Dark2")

    for ax, task in zip(axes, tasks):
        subset = plot_df.loc[plot_df["task_label"].eq(task)].copy()
        sns.lineplot(
            data=subset,
            x="confidence_threshold",
            y="n_graph_units",
            hue="cohort_label",
            palette=cohort_palette,
            marker="o",
            linewidth=2.5,
            ax=ax,
        )
        ax.set_title(task)
        ax.set_xlabel("Label confidence threshold")
        ax.set_ylabel("Eligible graph units")
        ax.legend(title="Cohort", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.suptitle("Label-confidence sensitivity analysis", y=1.03)
    return save_figure(fig, output_dir / "02_robustness/step2_label_confidence_sensitivity.png", dpi)


def plot_source_receiver_robustness(
    source_coverage: pd.DataFrame,
    receiver_robustness: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path:
    source_plot = source_coverage.copy()
    source_plot["source_label"] = source_plot["source_id"].map(pretty_source)
    source_pivot = source_plot.pivot_table(
        index="task_name",
        columns="source_label",
        values="n_donors",
        aggfunc="sum",
        fill_value=0,
    )
    source_pivot = source_pivot.reindex(index=[task for task in TASK_ORDER if task in source_pivot.index])
    source_pivot.index = [pretty_task(value) for value in source_pivot.index]

    stable = receiver_robustness.sort_values("receiver_specificity_ratio", ascending=False).head(12).copy()
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))

    sns.heatmap(
        source_pivot,
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        cmap="PuBuGn",
        cbar_kws={"label": "Donors"},
        ax=axes[0],
    )
    axes[0].set_title("Per-source donor support by task")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")

    sns.scatterplot(
        data=receiver_robustness,
        x="mg_neuron_effect",
        y="mg_da_effect",
        size="receiver_specificity_ratio",
        hue="receiver_specificity_ratio",
        palette="viridis",
        sizes=(40, 340),
        alpha=0.75,
        ax=axes[1],
    )
    axes[1].axhline(0, linestyle="--", color="black", linewidth=1)
    axes[1].axvline(0, linestyle="--", color="black", linewidth=1)
    axes[1].set_title("Receiver robustness: DA-specific effects vs broad-neuron comparator")
    axes[1].set_xlabel("microglia -> neuron effect")
    axes[1].set_ylabel("microglia -> DA effect")
    for row in stable.itertuples(index=False):
        axes[1].text(row.mg_neuron_effect + 0.04, row.mg_da_effect + 0.04, str(row.pathway_family), fontsize=10)
    axes[1].legend(title="Receiver specificity", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.suptitle("Source support and receiver robustness", y=1.02)
    return save_figure(fig, output_dir / "02_robustness/step2_source_receiver_robustness.png", dpi)


def plot_cellcount_lbd(
    cell_count_robustness: pd.DataFrame,
    embedding_export: ad.AnnData,
    output_dir: Path,
    dpi: int,
) -> Path:
    donor_obs = embedding_export.obs.reset_index().rename(columns={"index": "split_group_id"})
    disease_order = [value for value in DISEASE_ORDER if value in donor_obs["disease_group"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.barplot(
        data=cell_count_robustness,
        x="spearman_rho",
        y="metric",
        hue="analysis",
        palette=["#4C78A8", "#E45756"],
        ax=axes[0],
    )
    axes[0].axvline(0, linestyle="--", color="black", linewidth=1)
    axes[0].set_title("Cell-count robustness")
    axes[0].set_xlabel("Spearman rho")
    axes[0].set_ylabel("")
    axes[0].legend(title="Analysis", loc="lower right")

    sns.boxplot(
        data=donor_obs,
        x="disease_group",
        y="pd_probability",
        hue="disease_group",
        order=disease_order,
        palette=DISEASE_PALETTE,
        dodge=False,
        ax=axes[1],
    )
    sns.stripplot(
        data=donor_obs,
        x="disease_group",
        y="pd_probability",
        hue="disease_group",
        order=disease_order,
        palette=DISEASE_PALETTE,
        dodge=False,
        ax=axes[1],
        size=8,
        edgecolor="white",
        linewidth=0.5,
    )
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()
    axes[1].set_title("LBD bridge relative to control and PD")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("PD probability")

    fig.suptitle("Model robustness against graph size and disease bridge structure", y=1.03)
    return save_figure(fig, output_dir / "02_robustness/step2_cellcount_lbd.png", dpi)


def plot_external_holdout(
    holdout_counts: pd.DataFrame,
    holdout_summary: pd.DataFrame,
    holdout_baseline: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path:
    counts = holdout_counts.copy()
    counts["disease_group"] = np.where(counts["patient"].str.startswith("PD"), "pd", "control")
    counts["eligible_label"] = np.where(counts["eligible_mg_da"], "eligible", "not eligible")

    compare = holdout_summary.merge(
        holdout_baseline[["graph_unit_id", "baseline_pd_probability"]],
        on="graph_unit_id",
        how="left",
    )
    compare["patient"] = compare["patient"].astype(str)
    inferred_group = pd.Series(
        np.where(compare["patient"].str.startswith("PD"), "pd", "control"),
        index=compare.index,
        dtype="string",
    )
    compare["disease_group"] = compare["disease_group"].astype("string").where(compare["disease_group"].notna(), inferred_group)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    x = np.arange(counts.shape[0])
    width = 0.38
    axes[0].bar(x - width / 2, counts["microglia"], width=width, color="#2A9D8F", label="Microglia")
    axes[0].bar(x + width / 2, counts["dopaminergic neuron"], width=width, color="#E76F51", label="DA neuron")
    for idx, row in counts.iterrows():
        if bool(row["eligible_mg_da"]):
            axes[0].text(
                idx,
                max(float(row["microglia"]), float(row["dopaminergic neuron"])) + 14,
                "eligible",
                ha="center",
                fontsize=9,
            )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(counts["patient"], rotation=35, ha="right")
    axes[0].set_title("GSE157783 cell-type availability")
    axes[0].set_ylabel("Cells")
    axes[0].legend(loc="upper right")

    for row in compare.itertuples(index=False):
        color = DISEASE_PALETTE.get(str(row.disease_group), "#9D9DA1")
        axes[1].plot(
            [0, 1],
            [row.baseline_pd_probability, row.pd_probability],
            color=color,
            linewidth=2,
            alpha=0.85,
        )
        axes[1].scatter([0, 1], [row.baseline_pd_probability, row.pd_probability], color=color, s=90)
        axes[1].text(1.03, row.pd_probability, str(row.patient), va="center", fontsize=9)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Baseline", "GNN"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Holdout PD probability: baseline vs GNN")
    axes[1].set_ylabel("PD probability")

    sns.barplot(
        data=compare,
        x="patient",
        y="predicted_age_years",
        hue="disease_group",
        palette=DISEASE_PALETTE,
        ax=axes[2],
    )
    axes[2].set_title("Holdout predicted ages")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Predicted age (years)")
    axes[2].legend(title="Disease", loc="upper right")

    fig.suptitle("External holdout summary for GSE157783", y=1.03)
    return save_figure(fig, output_dir / "02_robustness/step2_external_holdout.png", dpi)


def plot_top_pathways(ranked_pathways: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    stable = ranked_pathways.loc[ranked_pathways["ranking_score"].notna()].copy()
    top = stable.head(12).sort_values("ranking_score", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))

    axes[0].barh(
        top["pathway_family"],
        top["ranking_score"],
        color=[INTERPRETATION_PALETTE.get(value, "#888888") for value in top["interpretation_bin"]],
    )
    axes[0].set_title("Top ranked pathways")
    axes[0].set_xlabel("Ranking score")
    axes[0].set_ylabel("")

    scatter = stable.loc[stable["aging_score"].notna() & stable["pd_score"].notna()].copy()
    sns.scatterplot(
        data=scatter,
        x="aging_score",
        y="pd_score",
        hue="interpretation_bin",
        size="receiver_specificity",
        sizes=(50, 360),
        palette=INTERPRETATION_PALETTE,
        alpha=0.85,
        ax=axes[1],
    )
    axes[1].axhline(0, linestyle="--", color="black", linewidth=1)
    axes[1].axvline(0, linestyle="--", color="black", linewidth=1)
    axes[1].set_title("Pathway aging vs PD-deviation space")
    axes[1].set_xlabel("Aging score")
    axes[1].set_ylabel("PD score")
    for row in stable.head(8).itertuples(index=False):
        axes[1].text(row.aging_score + 0.03, row.pd_score + 0.03, str(row.pathway_family), fontsize=10)
    axes[1].legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.suptitle("Step 2 pathway prioritization", y=1.03)
    return save_figure(fig, output_dir / "03_interpretation/step2_top_pathways.png", dpi)


def plot_top_lr_pairs(ranked_lr_pairs: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    stable = ranked_lr_pairs.loc[ranked_lr_pairs["ranking_score"].notna()].copy()
    top = stable.head(12).sort_values("ranking_score", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(23, 8))

    axes[0].barh(
        top["pair_key"],
        top["ranking_score"],
        color=[INTERPRETATION_PALETTE.get(value, "#888888") for value in top["interpretation_bin"]],
    )
    axes[0].set_title("Top ranked ligand-receptor pairs")
    axes[0].set_xlabel("Ranking score")
    axes[0].set_ylabel("")

    scatter = stable.loc[stable["aging_score"].notna() & stable["pd_score"].notna()].copy()
    sns.scatterplot(
        data=scatter,
        x="aging_score",
        y="pd_score",
        hue="interpretation_bin",
        size="holdout_support",
        sizes=(50, 360),
        palette=INTERPRETATION_PALETTE,
        alpha=0.8,
        ax=axes[1],
    )
    axes[1].axhline(0, linestyle="--", color="black", linewidth=1)
    axes[1].axvline(0, linestyle="--", color="black", linewidth=1)
    axes[1].set_title("Pair-level aging vs PD-deviation space")
    axes[1].set_xlabel("Aging score")
    axes[1].set_ylabel("PD score")
    for row in stable.head(10).itertuples(index=False):
        if pd.notna(row.aging_score) and pd.notna(row.pd_score):
            axes[1].text(row.aging_score + 0.03, row.pd_score + 0.03, str(row.pair_key), fontsize=9)
    axes[1].legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.suptitle("Step 2 ligand-receptor prioritization", y=1.03)
    return save_figure(fig, output_dir / "03_interpretation/step2_top_lr_pairs.png", dpi)


def plot_pathway_donor_heatmap(top_pathway_scores_by_donor: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    meta_cols = ["graph_unit_id", "split_group_id", "cohort_name", "source_id", "disease_group", "age_years"]
    pathway_cols = [column for column in top_pathway_scores_by_donor.columns if column not in meta_cols]
    donor_scores = top_pathway_scores_by_donor.copy()
    donor_scores = donor_scores.sort_values(["disease_group", "age_years", "split_group_id"])
    donor_scores["donor_label"] = donor_scores["split_group_id"].map(short_group_id)
    heat = donor_scores.set_index("donor_label")[pathway_cols].T
    heat = zscore_rows(heat)

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(heat, cmap="RdBu_r", center=0, linewidths=0.3, cbar_kws={"label": "Row z-score"}, ax=ax)
    ax.set_title("Top pathway scores across donor groups")
    ax.set_xlabel("Donor groups")
    ax.set_ylabel("Pathway")
    return save_figure(fig, output_dir / "03_interpretation/step2_pathway_donor_heatmap.png", dpi)


def plot_cross_task_pathways(
    pathway_scores: pd.DataFrame,
    ranked_pathways: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path:
    top_pathways = ranked_pathways.loc[ranked_pathways["ranking_score"].notna(), "pathway_family"].head(8).tolist()
    plot_df = pathway_scores.loc[pathway_scores["pathway_family"].isin(top_pathways)].copy()
    task_means = plot_df.pivot_table(
        index="pathway_family",
        columns="task_name",
        values="pathway_score",
        aggfunc="mean",
        fill_value=0,
    )
    task_means = task_means.reindex(index=top_pathways)
    task_means = task_means[[task for task in TASK_ORDER if task in task_means.columns]]
    task_means.columns = [pretty_task(value) for value in task_means.columns]
    task_means = zscore_rows(task_means)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(task_means, cmap="mako", linewidths=0.4, annot=True, fmt=".2f", cbar_kws={"label": "Row z-score"}, ax=ax)
    ax.set_title("Where the top pathways recur across Step 2 graph tasks")
    ax.set_xlabel("")
    ax.set_ylabel("")
    return save_figure(fig, output_dir / "03_interpretation/step2_cross_task_pathways.png", dpi)


def draw_pair_network(ax: plt.Axes, pair_rows: pd.DataFrame, pathway_row: pd.Series) -> None:
    ligands = ordered_unique(pair_rows["ligand_genes"].astype(str).tolist())
    receptors = ordered_unique(pair_rows["receptor_genes"].astype(str).tolist())
    ligand_y = np.linspace(0.88, 0.12, num=max(1, len(ligands)))
    receptor_y = np.linspace(0.88, 0.12, num=max(1, len(receptors)))
    ligand_pos = {name: (0.12, ligand_y[index]) for index, name in enumerate(ligands)}
    receptor_pos = {name: (0.88, receptor_y[index]) for index, name in enumerate(receptors)}
    max_rank = float(pair_rows["ranking_score"].max()) if pair_rows["ranking_score"].notna().any() else 1.0

    for row in pair_rows.itertuples(index=False):
        start = ligand_pos[str(row.ligand_genes)]
        end = receptor_pos[str(row.receptor_genes)]
        weight = float(row.ranking_score) / max_rank if max_rank > 0 else 0.4
        ax.plot([start[0], end[0]], [start[1], end[1]], linewidth=1.5 + 4.0 * weight, alpha=0.35 + 0.55 * weight, color="#6C757D")
        label_x = 0.5
        label_y = (start[1] + end[1]) / 2
        ax.text(label_x, label_y, str(row.pair_key), fontsize=8.5, ha="center", va="center")

    for name, (x, y) in ligand_pos.items():
        ax.scatter(x, y, s=240, color=SIDE_PALETTE["ligand"], edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(x - 0.03, y, fill(name, 14), ha="right", va="center", fontsize=9)

    for name, (x, y) in receptor_pos.items():
        ax.scatter(x, y, s=240, color=SIDE_PALETTE["receptor"], edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(x + 0.03, y, fill(name, 14), ha="left", va="center", fontsize=9)

    ax.text(0.12, 0.98, "Microglia ligands", color=SIDE_PALETTE["ligand"], fontsize=10, weight="bold", ha="left")
    ax.text(0.88, 0.98, "Receiver receptors", color=SIDE_PALETTE["receptor"], fontsize=10, weight="bold", ha="right")
    subtitle = (
        f"ranking={pathway_row['ranking_score']:.2f}, "
        f"receiver specificity={pathway_row['receiver_specificity']:.2f}, "
        f"holdout={pathway_row['holdout_support']:.2f}"
    )
    ax.set_title(fill(f"{pathway_row['pathway_family']} | {subtitle}", 36), fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def plot_pathway_networks(
    ranked_pathways: pd.DataFrame,
    ranked_lr_pairs: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path:
    pathway_rows = ranked_pathways.loc[ranked_pathways["ranking_score"].notna()].head(4).copy()
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    for ax, pathway_row in zip(axes.ravel(), pathway_rows.to_dict(orient="records")):
        pair_rows = (
            ranked_lr_pairs.loc[ranked_lr_pairs["pathway_family"].eq(pathway_row["pathway_family"]) & ranked_lr_pairs["ranking_score"].notna()]
            .sort_values("ranking_score", ascending=False)
            .head(6)
            .copy()
        )
        draw_pair_network(ax, pair_rows, pd.Series(pathway_row))
    fig.suptitle("Receiver-oriented pathway skeletons for the top Step 2 candidates", y=0.99)
    return save_figure(fig, output_dir / "03_interpretation/step2_top_pathway_networks.png", dpi)


def build_report(
    output_dir: Path,
    manifest_rows: list[dict[str, str]],
    embedding_coords: pd.DataFrame,
    tables: dict[str, object],
) -> None:
    graph_manifest = tables["graph_manifest"]
    ranked_pathways = tables["ranked_pathways"]
    ranked_lr_pairs = tables["ranked_lr_pairs"]
    age_fold_metrics = tables["age_fold_metrics"]
    pd_fold_metrics = tables["pd_fold_metrics"]
    holdout_summary = tables["holdout_summary"]
    lbd_summary = tables["lbd_summary"]

    snapshot = pd.DataFrame(
        [
            {"metric": "backbone_cells", "value": f"{tables['backbone'].shape[0]:,}"},
            {"metric": "communication_genes", "value": f"{tables['comm_panel'].shape[0]:,}"},
            {"metric": "graph_subgraphs", "value": f"{graph_manifest.shape[0]:,}"},
            {"metric": "modeled_tasks", "value": f"{graph_manifest['task_name'].nunique()}"},
            {"metric": "age_model_mean_mae_years", "value": f"{age_fold_metrics['mae'].mean():.3f}"},
            {"metric": "pd_model_mean_auroc", "value": f"{pd_fold_metrics['auroc'].mean():.3f}"},
            {"metric": "pd_model_mean_balanced_accuracy", "value": f"{pd_fold_metrics['balanced_accuracy'].mean():.3f}"},
            {
                "metric": "top_pathway",
                "value": str(ranked_pathways.loc[ranked_pathways['ranking_score'].notna(), 'pathway_family'].iloc[0]),
            },
            {
                "metric": "top_lr_pair",
                "value": str(ranked_lr_pairs.loc[ranked_lr_pairs['ranking_score'].notna(), 'pair_key'].iloc[0]),
            },
            {
                "metric": "lbd_mean_pd_probability",
                "value": f"{lbd_summary.loc[lbd_summary['cohort_name'].eq('lbd_bridge'), 'mean_pd_probability'].iloc[0]:.3f}",
            },
            {"metric": "holdout_patients_scored", "value": f"{holdout_summary.shape[0]}"},
        ]
    )

    figure_manifest = pd.DataFrame(manifest_rows)
    figure_manifest["relative_path"] = figure_manifest["path"].map(lambda value: str(Path(value).relative_to(output_dir)))
    figure_manifest = figure_manifest[["category", "figure_name", "relative_path", "description"]]
    figure_manifest = figure_manifest.sort_values(["category", "figure_name"]).reset_index(drop=True)
    figure_manifest.to_csv(output_dir / "step2_visualization_manifest.csv", index=False)
    embedding_coords.to_csv(output_dir / "step2_embedding_umap_coordinates.csv", index=False)

    lines: list[str] = []
    lines.append("# Step 2 Visualization Pack")
    lines.append("")
    lines.append("## Snapshot")
    lines.append(markdown_table(snapshot))
    lines.append("")
    lines.append("## Figure Inventory")
    lines.append(markdown_table(figure_manifest))
    lines.append("")
    lines.append("## Notes")
    lines.append("- All plots are static PNG outputs regenerated from the finalized Step 2 artifacts.")
    lines.append("- The pathway skeleton figure is receiver oriented and summarizes top ligand-receptor pairs, not physical spatial edges.")
    lines.append("- `step2_embedding_umap_coordinates.csv` exports donor-level UMAP coordinates alongside cohort, disease, age, and predicted scores.")
    lines.append("- External holdout visualization uses `GSE157783` patients that pass the microglia-to-DA eligibility filter.")
    write_markdown(output_dir / "step2_visualization_report.md", "\n".join(lines))


def main() -> None:
    args = parse_args()
    configure_style()

    step2_root = project_path(args.step2_root)
    output_dir = ensure_dir(project_path(args.output_dir))
    tables = load_tables(step2_root)

    donor_meta = main_task_metadata(tables["graph_manifest"])
    age_val = donor_level_age_validation(tables["age_pred"], donor_meta)
    pd_val = donor_level_pd_validation(tables["pd_pred"], donor_meta)
    embedding_coords = build_embedding_coordinates(tables["embedding_export"])

    manifest_rows: list[dict[str, str]] = []

    figure_specs = [
        ("overview", "backbone_overview", plot_backbone_overview, [tables["backbone"]]),
        ("overview", "backbone_qc", plot_backbone_qc, [tables["backbone"]]),
        ("overview", "comm_panel_summary", plot_comm_panel, [tables["comm_panel"]]),
        ("overview", "cohort_coverage", plot_cohort_coverage, [tables["cohort_manifest"], tables["source_coverage"]]),
        ("overview", "graph_statistics", plot_graph_statistics, [tables["graph_manifest"]]),
        ("models", "baseline_vs_gnn", plot_baseline_vs_gnn, [tables["baseline_metrics"], tables["age_fold_metrics"], tables["pd_fold_metrics"]]),
        ("models", "age_model_validation", plot_age_validation, [age_val, tables["age_fold_metrics"]]),
        ("models", "pd_model_validation", plot_pd_validation, [pd_val, tables["pd_fold_metrics"]]),
        ("models", "embedding_umaps", plot_embedding_umaps, [embedding_coords]),
        ("robustness", "label_confidence_sensitivity", plot_label_confidence, [tables["label_confidence"]]),
        ("robustness", "source_receiver_robustness", plot_source_receiver_robustness, [tables["source_coverage"], tables["receiver_robustness"]]),
        ("robustness", "cellcount_lbd", plot_cellcount_lbd, [tables["cell_count_robustness"], tables["embedding_export"]]),
        ("robustness", "external_holdout", plot_external_holdout, [tables["holdout_counts"], tables["holdout_summary"], tables["holdout_baseline"]]),
        ("interpretation", "top_pathways", plot_top_pathways, [tables["ranked_pathways"]]),
        ("interpretation", "top_lr_pairs", plot_top_lr_pairs, [tables["ranked_lr_pairs"]]),
        ("interpretation", "pathway_donor_heatmap", plot_pathway_donor_heatmap, [tables["top_pathway_scores_by_donor"]]),
        ("interpretation", "cross_task_pathways", plot_cross_task_pathways, [tables["pathway_scores"], tables["ranked_pathways"]]),
        ("interpretation", "top_pathway_networks", plot_pathway_networks, [tables["ranked_pathways"], tables["ranked_lr_pairs"]]),
    ]

    descriptions = {
        "backbone_overview": "Cell universe, donor source composition, age structure, and label provenance.",
        "backbone_qc": "QC metric distributions across major Step 2 cell classes.",
        "comm_panel_summary": "Communication panel membership, supplement categories, and source recovery.",
        "cohort_coverage": "Task-by-cohort eligible graph units and source-level donor support.",
        "graph_statistics": "Task-level subgraph counts and node-edge size distribution.",
        "baseline_vs_gnn": "Main-task donor-grouped baseline comparison against the Step 2 GNNs.",
        "age_model_validation": "Observed versus predicted donor ages and residual structure.",
        "pd_model_validation": "Validation donor PD probabilities and fold-specific threshold behavior.",
        "embedding_umaps": "Donor-level age and PD embedding UMAP projections.",
        "label_confidence_sensitivity": "Eligibility stability under stricter transferred-label confidence thresholds.",
        "source_receiver_robustness": "Per-source donor support and DA-specific receiver robustness.",
        "cellcount_lbd": "Correlation of predictions with graph size and LBD bridge behavior.",
        "external_holdout": "GSE157783 eligibility, baseline-vs-GNN PD scoring, and predicted ages.",
        "top_pathways": "Top ranked pathways and their aging-vs-PD geometry.",
        "top_lr_pairs": "Top ranked LR pairs and their aging-vs-PD geometry.",
        "pathway_donor_heatmap": "Per-donor pathway score heatmap for the leading exported pathways.",
        "cross_task_pathways": "Where the leading pathways recur across secondary partner tasks.",
        "top_pathway_networks": "Receiver-oriented LR skeletons for the highest-priority pathways.",
    }

    for category, figure_name, func, func_args in figure_specs:
        path = func(*func_args, output_dir, args.dpi)
        manifest_rows.append(
            {
                "category": category,
                "figure_name": figure_name,
                "path": str(path),
                "description": descriptions[figure_name],
            }
        )

    build_report(output_dir, manifest_rows, embedding_coords, tables)
    print(f"Step 2 visualization pack written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
