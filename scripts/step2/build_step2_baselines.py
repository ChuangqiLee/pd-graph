#!/usr/bin/env python3
"""Build pseudobulk communication baselines for Step 2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import load_yaml, markdown_table, project_path, write_markdown


LR_CONFIG = Path("configs/step2/step2_lr_universe.yaml")
MODEL_PD_CONFIG = Path("configs/step2/step2_model_pd.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step 2 statistical baselines.")
    parser.add_argument("--lr-config", default=str(LR_CONFIG))
    parser.add_argument("--pd-config", default=str(MODEL_PD_CONFIG))
    return parser.parse_args()


def grouped_predictions_regression(features: pd.DataFrame, target: pd.Series, groups: pd.Series) -> dict[str, float]:
    unique_groups = groups.astype(str).nunique()
    if unique_groups < 3 or len(features) < 6:
        return {"mae_elastic_net": float("nan"), "mae_random_forest": float("nan")}
    n_splits = min(3, unique_groups)
    gkf = GroupKFold(n_splits=n_splits)
    pred_en = np.zeros(len(features), dtype=float)
    pred_rf = np.zeros(len(features), dtype=float)
    for train_idx, test_idx in gkf.split(features, target, groups):
        x_train = features.iloc[train_idx]
        x_test = features.iloc[test_idx]
        y_train = target.iloc[train_idx]
        en_model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000, random_state=13)),
            ]
        )
        rf_model = RandomForestRegressor(n_estimators=300, random_state=13, min_samples_leaf=2)
        en_model.fit(x_train, y_train)
        rf_model.fit(x_train, y_train)
        pred_en[test_idx] = en_model.predict(x_test)
        pred_rf[test_idx] = rf_model.predict(x_test)
    return {
        "mae_elastic_net": float(mean_absolute_error(target, pred_en)),
        "mae_random_forest": float(mean_absolute_error(target, pred_rf)),
    }


def grouped_predictions_classification(features: pd.DataFrame, target: pd.Series, groups: pd.Series) -> dict[str, float]:
    unique_groups = groups.astype(str).nunique()
    if unique_groups < 3 or target.nunique() < 2:
        return {"auroc_logistic": float("nan"), "balanced_accuracy_logistic": float("nan"), "auroc_random_forest": float("nan"), "balanced_accuracy_random_forest": float("nan")}
    n_splits = min(3, unique_groups)
    gkf = GroupKFold(n_splits=n_splits)
    prob_lr = np.zeros(len(features), dtype=float)
    prob_rf = np.zeros(len(features), dtype=float)
    for train_idx, test_idx in gkf.split(features, target, groups):
        x_train = features.iloc[train_idx]
        x_test = features.iloc[test_idx]
        y_train = target.iloc[train_idx]
        lr_model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=13)),
            ]
        )
        rf_model = RandomForestClassifier(n_estimators=400, random_state=13, min_samples_leaf=2, class_weight="balanced")
        lr_model.fit(x_train, y_train)
        rf_model.fit(x_train, y_train)
        prob_lr[test_idx] = lr_model.predict_proba(x_test)[:, 1]
        prob_rf[test_idx] = rf_model.predict_proba(x_test)[:, 1]
    pred_lr = (prob_lr >= 0.5).astype(int)
    pred_rf = (prob_rf >= 0.5).astype(int)
    return {
        "auroc_logistic": float(roc_auc_score(target, prob_lr)),
        "balanced_accuracy_logistic": float(balanced_accuracy_score(target, pred_lr)),
        "auroc_random_forest": float(roc_auc_score(target, prob_rf)),
        "balanced_accuracy_random_forest": float(balanced_accuracy_score(target, pred_rf)),
    }


def main() -> None:
    args = parse_args()
    lr_config = load_yaml(args.lr_config)
    output_dir = project_path("data/processed/step2/04_baselines")
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_manifest = pd.read_csv(project_path("data/processed/step2/02_cohorts/step2_graph_manifest.csv"))
    lr_universe = pd.read_parquet(project_path("data/processed/step2/03_lr_universe/step2_lr_universe_full.parquet"))
    gene_stats = pd.read_parquet(project_path("data/processed/step2/03_lr_universe/step2_graph_unit_celltype_gene_stats.parquet"))

    pseudobulk_profiles = gene_stats.copy()
    pseudobulk_profiles.to_parquet(output_dir / "step2_pseudobulk_profiles.parquet", index=False)

    stats_index = gene_stats.set_index(["graph_unit_id", "cell_type", "gene_symbol"]).sort_index()
    included_graphs = graph_manifest.loc[graph_manifest["included_task_default"]].copy()

    lr_rows: list[dict[str, object]] = []
    for graph in included_graphs.itertuples(index=False):
        task_pairs = lr_universe.loc[lr_universe["eligible_tasks"].astype(str).str.contains(str(graph.task_name))]
        for pair in task_pairs.itertuples(index=False):
            ligand_genes = [gene for gene in str(pair.ligand_genes).split("|") if gene]
            receptor_genes = [gene for gene in str(pair.receptor_genes).split("|") if gene]
            try:
                sender_expr = [
                    float(stats_index.loc[(graph.graph_unit_id, graph.sender_cell_type, gene), "mean_norm"])
                    for gene in ligand_genes
                ]
                sender_pct = [
                    float(stats_index.loc[(graph.graph_unit_id, graph.sender_cell_type, gene), "pct_expr"])
                    for gene in ligand_genes
                ]
                receiver_expr = [
                    float(stats_index.loc[(graph.graph_unit_id, graph.receiver_cell_type, gene), "mean_norm"])
                    for gene in receptor_genes
                ]
                receiver_pct = [
                    float(stats_index.loc[(graph.graph_unit_id, graph.receiver_cell_type, gene), "pct_expr"])
                    for gene in receptor_genes
                ]
            except KeyError:
                continue
            sender_value = float(np.prod(np.maximum(sender_expr, 1e-12)) ** (1.0 / len(sender_expr)))
            receiver_value = float(np.prod(np.maximum(receiver_expr, 1e-12)) ** (1.0 / len(receiver_expr)))
            score = float(np.sqrt(sender_value * receiver_value))
            lr_rows.append(
                {
                    "graph_unit_id": graph.graph_unit_id,
                    "split_group_id": graph.split_group_id,
                    "cohort_name": graph.cohort_name,
                    "task_name": graph.task_name,
                    "source_id": graph.source_id,
                    "disease_group": graph.disease_group,
                    "age_years": graph.age_years,
                    "sender_cell_type": graph.sender_cell_type,
                    "receiver_cell_type": graph.receiver_cell_type,
                    "pair_key": pair.pair_key,
                    "pathway_family": pair.pathway_family,
                    "score": score,
                    "sender_mean_norm": sender_value,
                    "receiver_mean_norm": receiver_value,
                    "sender_pct_expr": float(np.min(sender_pct)),
                    "receiver_pct_expr": float(np.min(receiver_pct)),
                }
            )

    lr_scores = pd.DataFrame(lr_rows)
    lr_scores.to_parquet(output_dir / "step2_lr_scores.parquet", index=False)

    pathway_scores = (
        lr_scores.groupby(
            ["graph_unit_id", "split_group_id", "cohort_name", "task_name", "source_id", "disease_group", "age_years", "pathway_family"],
            observed=False,
        )["score"]
        .mean()
        .rename("pathway_score")
        .reset_index()
    )
    pathway_scores.to_parquet(output_dir / "step2_pathway_scores.parquet", index=False)

    metric_rows: list[dict[str, object]] = []
    for task_name, task_frame in pathway_scores.groupby("task_name", observed=False):
        wide = task_frame.pivot_table(
            index=["graph_unit_id", "split_group_id", "cohort_name", "source_id", "disease_group", "age_years"],
            columns="pathway_family",
            values="pathway_score",
            fill_value=0.0,
        ).reset_index()
        feature_columns = [column for column in wide.columns if column not in {"graph_unit_id", "split_group_id", "cohort_name", "source_id", "disease_group", "age_years"}]
        if not feature_columns:
            continue

        aging = wide.loc[wide["cohort_name"].eq("control_known_age") & wide["age_years"].notna()].copy()
        if not aging.empty:
            age_metrics = grouped_predictions_regression(aging[feature_columns], aging["age_years"], aging["split_group_id"])
            metric_rows.append({"task_name": task_name, "analysis": "control_aging", **age_metrics})

        disease = wide.loc[wide["cohort_name"].isin(["control_known_age", "pd_known_age"]) & wide["age_years"].notna()].copy()
        if not disease.empty and disease["disease_group"].nunique() == 2:
            disease_features = disease[feature_columns].copy()
            disease_features["age_years"] = disease["age_years"].astype(float)
            disease_target = disease["disease_group"].map({"control": 0, "pd": 1}).astype(int)
            disease_metrics = grouped_predictions_classification(disease_features, disease_target, disease["split_group_id"])
            metric_rows.append({"task_name": task_name, "analysis": "pd_deviation", **disease_metrics})

    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(output_dir / "step2_baseline_metrics.csv", index=False)

    report_lines = [
        "# Step 2 Baseline Models",
        "",
        "## Metrics",
        markdown_table(metric_df.fillna("NA")),
        "",
        "## Output Tables",
        f"- pseudobulk_profiles: `{output_dir / 'step2_pseudobulk_profiles.parquet'}`",
        f"- lr_scores: `{output_dir / 'step2_lr_scores.parquet'}`",
        f"- pathway_scores: `{output_dir / 'step2_pathway_scores.parquet'}`",
    ]
    write_markdown(output_dir / "step2_baseline_model_report.md", "\n".join(report_lines))


if __name__ == "__main__":
    main()
