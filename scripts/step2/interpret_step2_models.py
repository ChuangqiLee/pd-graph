#!/usr/bin/env python3
"""Interpret Step 2 age and PD graph models into ranked pathway candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import load_yaml, markdown_table, project_path, read_table, write_markdown


DEFAULT_CONFIG = Path("configs/step2/step2_interpretation.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpret Step 2 graph models.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def linear_age_slope(frame: pd.DataFrame, value_col: str) -> float:
    local = frame[[value_col, "age_years"]].dropna().copy()
    if len(local) < 3 or local["age_years"].nunique() < 2:
        return float("nan")
    design = np.column_stack([np.ones(len(local), dtype=float), local["age_years"].astype(float).to_numpy()])
    target = local[value_col].astype(float).to_numpy()
    coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return float(coeffs[1])


def disease_effect_ols(frame: pd.DataFrame, value_col: str) -> float:
    local = frame[[value_col, "age_years", "disease_group"]].dropna().copy()
    if len(local) < 4 or local["disease_group"].nunique() < 2:
        return float("nan")
    local["disease_binary"] = local["disease_group"].map({"control": 0.0, "pd": 1.0}).astype(float)
    design = np.column_stack(
        [
            np.ones(len(local), dtype=float),
            local["age_years"].astype(float).to_numpy(),
            local["disease_binary"].to_numpy(),
        ]
    )
    target = local[value_col].astype(float).to_numpy()
    coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return float(coeffs[-1])


def zscore_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.mean(skipna=True)
    std = values.std(skipna=True, ddof=0)
    if pd.isna(std) or float(std) == 0.0:
        return pd.Series(np.zeros(len(values), dtype=float), index=series.index)
    return (values - float(mean)) / float(std)


def summarize_attention(attention: pd.DataFrame, compact_universe: pd.DataFrame, *, model_prefix: str) -> pd.DataFrame:
    val = attention.loc[attention["split"].eq("val")].copy()
    val["contribution"] = val["attention"].astype(float) * val["top_pair_score"].astype(float)
    summary = (
        val.groupby("top_pair_index", observed=False)
        .agg(
            attention_mean=("attention", "mean"),
            attention_max=("attention", "max"),
            contribution_mean=("contribution", "mean"),
            contribution_sum=("contribution", "sum"),
            edge_hits=("graph_id", "count"),
            fold_count=("fold", "nunique"),
        )
        .reset_index()
    )
    summary["fold_stability"] = summary["fold_count"] / max(val["fold"].nunique(), 1)
    renamed = summary.rename(
        columns={
            "attention_mean": f"{model_prefix}_attention_mean",
            "attention_max": f"{model_prefix}_attention_max",
            "contribution_mean": f"{model_prefix}_contribution_mean",
            "contribution_sum": f"{model_prefix}_contribution_sum",
            "edge_hits": f"{model_prefix}_edge_hits",
            "fold_count": f"{model_prefix}_fold_count",
            "fold_stability": f"{model_prefix}_fold_stability",
            "top_pair_index": "compact_pair_index",
        }
    )
    return compact_universe.reset_index().rename(columns={"index": "compact_pair_index"}).merge(
        renamed, on="compact_pair_index", how="left"
    )


def pair_baseline_effects(lr_scores: pd.DataFrame, *, task_name: str) -> pd.DataFrame:
    task = lr_scores.loc[lr_scores["task_name"].eq(task_name)].copy()
    rows: list[dict[str, Any]] = []
    for pair_key, frame in task.groupby("pair_key", observed=False):
        rows.append(
            {
                "pair_key": str(pair_key),
                "age_baseline_slope": linear_age_slope(
                    frame.loc[frame["cohort_name"].eq("control_known_age") & frame["age_years"].notna()],
                    "score",
                ),
                "pd_baseline_effect": disease_effect_ols(
                    frame.loc[frame["cohort_name"].isin(["control_known_age", "pd_known_age"]) & frame["age_years"].notna()],
                    "score",
                ),
                "source_coverage_ratio": frame["source_id"].astype(str).nunique() / max(task["source_id"].astype(str).nunique(), 1),
            }
        )
    return pd.DataFrame(rows)


def pathway_baseline_effects(pathway_scores: pd.DataFrame, *, task_name: str) -> pd.DataFrame:
    task = pathway_scores.loc[pathway_scores["task_name"].eq(task_name)].copy()
    rows: list[dict[str, Any]] = []
    for pathway_family, frame in task.groupby("pathway_family", observed=False):
        rows.append(
            {
                "pathway_family": str(pathway_family),
                "age_baseline_slope": linear_age_slope(
                    frame.loc[frame["cohort_name"].eq("control_known_age") & frame["age_years"].notna()],
                    "pathway_score",
                ),
                "pd_baseline_effect": disease_effect_ols(
                    frame.loc[frame["cohort_name"].isin(["control_known_age", "pd_known_age"]) & frame["age_years"].notna()],
                    "pathway_score",
                ),
                "source_coverage_ratio": frame["source_id"].astype(str).nunique() / max(task["source_id"].astype(str).nunique(), 1),
            }
        )
    return pd.DataFrame(rows)


def specificity_from_task_effects(df: pd.DataFrame, id_col: str, value_col: str, other_df: pd.DataFrame) -> pd.DataFrame:
    merged = df[[id_col, value_col]].merge(
        other_df[[id_col, value_col]].rename(columns={value_col: "other_effect"}),
        on=id_col,
        how="outer",
    )
    merged["receiver_specificity"] = np.abs(pd.to_numeric(merged[value_col], errors="coerce")) / (
        np.abs(pd.to_numeric(merged["other_effect"], errors="coerce")) + 1e-6
    )
    return merged[[id_col, "receiver_specificity"]]


def interpret_bin(pd_score: float, aging_score: float, fold_stability: float, source_stability: float) -> str:
    if pd.isna(fold_stability) or float(fold_stability) < 0.34 or pd.isna(source_stability) or float(source_stability) < 0.34:
        return "unstable_or_source_sensitive"
    if abs(pd_score) < 0.35 and abs(aging_score) >= 0.35:
        return "normal_aging_associated"
    if pd_score >= 0.5 and aging_score >= 0.2:
        return "pd_amplified_aging"
    if pd_score >= 0.5 and abs(aging_score) < 0.2:
        return "pd_specific"
    if aging_score > pd_score:
        return "normal_aging_associated"
    return "pd_specific"


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_dir = project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    compact_universe = read_table(project_path(config["paths"]["lr_universe_compact"])).reset_index(drop=True)
    lr_scores = read_table(project_path("data/processed/step2/04_baselines/step2_lr_scores.parquet"))
    pathway_scores = read_table(project_path("data/processed/step2/04_baselines/step2_pathway_scores.parquet"))
    age_attention = read_table(project_path("data/processed/step2/06_models/age_model/age_model_attention_summary.parquet"))
    pd_attention = read_table(project_path("data/processed/step2/06_models/pd_model/pd_model_attention_summary.parquet"))
    receiver_robustness = pd.read_csv(project_path("data/processed/step2/07_robustness/step2_receiver_robustness.csv"))
    holdout_pathway = pd.read_csv(project_path("data/processed/step2/07_robustness/step2_holdout_pathway_effects.csv"))
    holdout_lr = pd.read_csv(project_path("data/processed/step2/07_robustness/step2_holdout_lr_effects.csv"))

    task_name = "microglia_to_dopaminergic_neuron"
    comparator_task = "microglia_to_neuron"

    age_summary = summarize_attention(age_attention, compact_universe, model_prefix="age")
    pd_summary = summarize_attention(pd_attention, compact_universe, model_prefix="pd")
    pair_table = age_summary.merge(
        pd_summary[
            [
                "compact_pair_index",
                "pd_attention_mean",
                "pd_attention_max",
                "pd_contribution_mean",
                "pd_contribution_sum",
                "pd_edge_hits",
                "pd_fold_count",
                "pd_fold_stability",
            ]
        ],
        on="compact_pair_index",
        how="outer",
    )

    mgda_pair_effects = pair_baseline_effects(lr_scores, task_name=task_name)
    mgneuron_pair_effects = pair_baseline_effects(lr_scores, task_name=comparator_task).rename(
        columns={"pd_baseline_effect": "pd_baseline_effect_comparator"}
    )
    pair_specificity = specificity_from_task_effects(
        mgda_pair_effects.rename(columns={"pd_baseline_effect": "task_effect"}),
        "pair_key",
        "task_effect",
        mgneuron_pair_effects.rename(columns={"pd_baseline_effect_comparator": "task_effect"}),
    )
    pair_table = (
        pair_table.merge(mgda_pair_effects, on="pair_key", how="left")
        .merge(pair_specificity, on="pair_key", how="left")
        .merge(holdout_lr[["pair_key", "holdout_pd_minus_control"]], on="pair_key", how="left")
    )

    pair_table["aging_score"] = (
        zscore_series(pair_table["age_contribution_sum"]) + zscore_series(pair_table["age_baseline_slope"])
    ) / 2.0
    pair_table["pd_score"] = (
        zscore_series(pair_table["pd_contribution_sum"])
        + zscore_series(pair_table["pd_baseline_effect"])
        + zscore_series(pair_table["holdout_pd_minus_control"])
    ) / 3.0
    pair_table["delta_score"] = pair_table["pd_score"] - pair_table["aging_score"]
    pair_table["fold_stability"] = pair_table[["age_fold_stability", "pd_fold_stability"]].mean(axis=1, skipna=True)
    pair_table["source_stability"] = pair_table["source_coverage_ratio"]
    pair_table["database_support"] = pd.to_numeric(pair_table["database_support_count"], errors="coerce").fillna(0)
    pair_table["holdout_support"] = pair_table["holdout_pd_minus_control"].fillna(0.0)
    pair_table["replication_component"] = zscore_series(pair_table["holdout_support"]).fillna(0.0)
    pair_table["source_component"] = zscore_series(pair_table["source_stability"]).fillna(0.0)
    pair_table["ranking_score"] = (
        pair_table["pd_score"].fillna(0.0) * 1.1
        + pair_table["aging_score"].fillna(0.0) * 0.5
        + zscore_series(pair_table["receiver_specificity"]).fillna(0.0) * 0.25
        + zscore_series(pair_table["database_support"]).fillna(0.0) * 0.15
        + pair_table["replication_component"] * 0.35
        + pair_table["source_component"] * 0.15
    ) * (0.45 + 0.55 * pair_table["fold_stability"].fillna(0.0))
    pair_table["interpretation_bin"] = [
        interpret_bin(pd_score, aging_score, fold_stability, source_stability)
        for pd_score, aging_score, fold_stability, source_stability in zip(
            pair_table["pd_score"].fillna(0.0),
            pair_table["aging_score"].fillna(0.0),
            pair_table["fold_stability"].fillna(0.0),
            pair_table["source_stability"].fillna(0.0),
        )
    ]
    pair_table = pair_table.sort_values("ranking_score", ascending=False)
    pair_top = pair_table.head(int(config["ranking"]["top_lr_pairs"])).copy()
    pair_top.to_csv(output_dir / "step2_ranked_lr_pairs.csv", index=False)

    pair_pathway_rollup = pair_table.groupby("pathway_family", observed=False).agg(
        age_attention_sum=("age_contribution_sum", "sum"),
        pd_attention_sum=("pd_contribution_sum", "sum"),
        avg_pair_ranking=("ranking_score", "mean"),
        fold_stability=("fold_stability", "mean"),
        source_stability=("source_stability", "mean"),
        database_support=("database_support", "mean"),
    ).reset_index()
    mgda_path_effects = pathway_baseline_effects(pathway_scores, task_name=task_name)
    mgneuron_path_effects = pathway_baseline_effects(pathway_scores, task_name=comparator_task).rename(
        columns={"pd_baseline_effect": "pd_baseline_effect_comparator"}
    )
    pathway_table = (
        pair_pathway_rollup.merge(mgda_path_effects, on="pathway_family", how="left")
        .merge(receiver_robustness[["pathway_family", "receiver_specificity_ratio"]], on="pathway_family", how="left")
        .merge(holdout_pathway[["pathway_family", "holdout_pd_minus_control"]], on="pathway_family", how="left")
    )
    pathway_table["aging_score"] = (
        zscore_series(pathway_table["age_attention_sum"]) + zscore_series(pathway_table["age_baseline_slope"])
    ) / 2.0
    pathway_table["pd_score"] = (
        zscore_series(pathway_table["pd_attention_sum"])
        + zscore_series(pathway_table["pd_baseline_effect"])
        + zscore_series(pathway_table["holdout_pd_minus_control"])
    ) / 3.0
    pathway_table["delta_score"] = pathway_table["pd_score"] - pathway_table["aging_score"]
    pathway_table["receiver_specificity"] = pathway_table["receiver_specificity_ratio"]
    pathway_table["holdout_support"] = pathway_table["holdout_pd_minus_control"].fillna(0.0)
    pathway_table["replication_component"] = zscore_series(pathway_table["holdout_support"]).fillna(0.0)
    pathway_table["source_component"] = zscore_series(pathway_table["source_stability"]).fillna(0.0)
    pathway_table["ranking_score"] = (
        pathway_table["pd_score"].fillna(0.0) * 1.1
        + pathway_table["aging_score"].fillna(0.0) * 0.5
        + zscore_series(pathway_table["receiver_specificity"]).fillna(0.0) * 0.25
        + zscore_series(pathway_table["database_support"]).fillna(0.0) * 0.15
        + pathway_table["replication_component"] * 0.35
        + pathway_table["source_component"] * 0.15
    ) * (0.45 + 0.55 * pathway_table["fold_stability"].fillna(0.0))
    pathway_table["interpretation_bin"] = [
        interpret_bin(pd_score, aging_score, fold_stability, source_stability)
        for pd_score, aging_score, fold_stability, source_stability in zip(
            pathway_table["pd_score"].fillna(0.0),
            pathway_table["aging_score"].fillna(0.0),
            pathway_table["fold_stability"].fillna(0.0),
            pathway_table["source_stability"].fillna(0.0),
        )
    ]
    pathway_table = pathway_table.sort_values("ranking_score", ascending=False)
    pathway_top = pathway_table.head(int(config["ranking"]["top_pathways"])).copy()
    pathway_top.to_csv(output_dir / "step2_ranked_pathways.csv", index=False)

    top_edge_contexts = (
        pd_attention.loc[pd_attention["split"].eq("val")]
        .merge(
            compact_universe.reset_index().rename(columns={"index": "compact_pair_index"})[
                ["compact_pair_index", "pair_key", "pathway_family", "ligand_genes", "receptor_genes"]
            ],
            left_on="top_pair_index",
            right_on="compact_pair_index",
            how="left",
        )
        .assign(contribution=lambda frame: frame["attention"].astype(float) * frame["top_pair_score"].astype(float))
        .sort_values("contribution", ascending=False)
        .head(500)
    )
    top_edge_contexts.to_csv(output_dir / "step2_top_edge_contexts.csv", index=False)

    report = "\n".join(
        [
            "# Step 2 Candidate Report",
            "",
            "## Top Pathways",
            markdown_table(
                pathway_top[
                    [
                        "pathway_family",
                        "ranking_score",
                        "aging_score",
                        "pd_score",
                        "delta_score",
                        "receiver_specificity",
                        "interpretation_bin",
                    ]
                ].round(4)
            ),
            "",
            "## Top LR Pairs",
            markdown_table(
                pair_top[
                    [
                        "pair_key",
                        "pathway_family",
                        "ranking_score",
                        "aging_score",
                        "pd_score",
                        "delta_score",
                        "receiver_specificity",
                        "interpretation_bin",
                    ]
                ].head(20).round(4)
            ),
            "",
            "## Interpretation Notes",
            "- Rankings combine donor-held-out attention summaries, baseline age/disease effects, receiver specificity, and external holdout support where available.",
            "- `PD amplified aging` candidates are pathways that score in both the aging model and the PD deviation model but shift more strongly in PD.",
            "- `PD specific` candidates are pathways with strong PD deviation score and comparatively weak aging score.",
        ]
    )
    write_markdown(output_dir / "step2_candidate_report.md", report)


if __name__ == "__main__":
    main()
