#!/usr/bin/env python3
"""Export Step 2 interpretation and model artifacts for downstream steps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import markdown_table, project_path, write_markdown


DEFAULT_OUTPUT_DIR = Path("data/processed/step2/09_exports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Step 2 results for downstream steps.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def donor_level_embeddings(embedding_path: Path, graph_manifest: pd.DataFrame, prefix: str) -> pd.DataFrame:
    embeddings = pd.read_parquet(embedding_path)
    merged = embeddings.merge(
        graph_manifest[
            [
                "graph_id",
                "split_group_id",
                "cohort_name",
                "source_id",
                "disease_group",
                "age_years",
                "brain_region_standardized",
                "sex_standardized",
            ]
        ],
        on="graph_id",
        how="left",
    )
    emb_cols = [column for column in merged.columns if column.startswith("emb_")]
    donor = (
        merged.groupby(
            [
                "split_group_id",
                "cohort_name",
                "source_id",
                "disease_group",
                "age_years",
                "brain_region_standardized",
                "sex_standardized",
            ],
            observed=False,
        )[emb_cols]
        .mean()
        .reset_index()
    )
    rename_map = {column: f"{prefix}_{column}" for column in emb_cols}
    return donor.rename(columns=rename_map)


def write_embedding_exports(
    *,
    output_dir: Path,
    donor_meta: pd.DataFrame,
    age_emb_cols: list[str],
    pd_emb_cols: list[str],
) -> bool:
    combined = pd.concat(
        [
            donor_meta[age_emb_cols].fillna(0.0),
            donor_meta[pd_emb_cols].fillna(0.0),
        ],
        axis=1,
    )
    obs = donor_meta.drop(columns=age_emb_cols + pd_emb_cols).copy()
    obs.to_csv(output_dir / "step2_embedding_exports_obs.csv")
    combined.to_parquet(output_dir / "step2_embedding_exports_matrix.parquet")

    try:
        import anndata as ad
    except ModuleNotFoundError:
        return False

    export_adata = ad.AnnData(X=combined.to_numpy(dtype=np.float32), obs=obs)
    export_adata.obs_names = donor_meta.index.astype(str)
    if age_emb_cols:
        export_adata.obsm["X_age_embedding"] = donor_meta[age_emb_cols].fillna(0.0).to_numpy(dtype=np.float32)
    if pd_emb_cols:
        export_adata.obsm["X_pd_embedding"] = donor_meta[pd_emb_cols].fillna(0.0).to_numpy(dtype=np.float32)
    export_adata.write_h5ad(output_dir / "step2_embedding_exports.h5ad")
    return True


def main() -> None:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pathway_rank = pd.read_csv(project_path("data/processed/step2/08_interpretation/step2_ranked_pathways.csv"))
    pair_rank = pd.read_csv(project_path("data/processed/step2/08_interpretation/step2_ranked_lr_pairs.csv"))
    top_edges = pd.read_csv(project_path("data/processed/step2/08_interpretation/step2_top_edge_contexts.csv"))
    robustness = pd.read_csv(project_path("data/processed/step2/07_robustness/step2_lbd_bridge_summary.csv"))
    graph_manifest = pd.read_parquet(project_path("data/processed/step2/05_graphs/step2_graph_manifest.parquet"))
    pathway_scores = pd.read_parquet(project_path("data/processed/step2/04_baselines/step2_pathway_scores.parquet"))
    age_predictions = pd.read_csv(project_path("data/processed/step2/06_models/age_model/age_model_predictions.csv"))
    pd_predictions = pd.read_csv(project_path("data/processed/step2/06_models/pd_model/pd_model_predictions.csv"))

    stable_pathways = pathway_rank.loc[
        pathway_rank["interpretation_bin"].ne("unstable_or_source_sensitive")
        & pathway_rank["ranking_score"].notna()
    ].copy()
    stable_pairs = pair_rank.loc[
        pair_rank["interpretation_bin"].ne("unstable_or_source_sensitive")
        & pair_rank["ranking_score"].notna()
    ].copy()

    step3_candidates = pd.concat(
        [
            stable_pathways.head(12).assign(candidate_type="pathway").rename(columns={"pathway_family": "candidate_id"}),
            stable_pairs.head(20).assign(candidate_type="lr_pair").rename(columns={"pair_key": "candidate_id"}),
        ],
        ignore_index=True,
        sort=False,
    )
    keep_cols = [
        "candidate_type",
        "candidate_id",
        "ranking_score",
        "aging_score",
        "pd_score",
        "delta_score",
        "receiver_specificity",
        "holdout_support",
        "interpretation_bin",
    ]
    step3_candidates = step3_candidates[[column for column in keep_cols if column in step3_candidates.columns]]
    step3_candidates.to_csv(output_dir / "step2_top_candidates_for_step3.csv", index=False)

    step4_candidates = pd.concat(
        [
            stable_pathways.loc[stable_pathways["holdout_support"].abs().fillna(0).ge(0.15)]
            .head(8)
            .assign(candidate_type="pathway")
            .rename(columns={"pathway_family": "candidate_id"}),
            stable_pairs.loc[stable_pairs["holdout_support"].abs().fillna(0).ge(0.5)]
            .head(12)
            .assign(candidate_type="lr_pair")
            .rename(columns={"pair_key": "candidate_id"}),
        ],
        ignore_index=True,
        sort=False,
    )
    step4_candidates = step4_candidates[[column for column in keep_cols if column in step4_candidates.columns]]
    step4_candidates.to_csv(output_dir / "step2_top_candidates_for_step4.csv", index=False)

    top_pathway_names = stable_pathways.head(10)["pathway_family"].astype(str).tolist()
    donor_top_pathways = (
        pathway_scores.loc[pathway_scores["task_name"].eq("microglia_to_dopaminergic_neuron") & pathway_scores["pathway_family"].isin(top_pathway_names)]
        .pivot_table(
            index=["graph_unit_id", "split_group_id", "cohort_name", "source_id", "disease_group", "age_years"],
            columns="pathway_family",
            values="pathway_score",
            fill_value=0.0,
        )
        .reset_index()
    )
    donor_top_pathways.to_csv(output_dir / "step2_top_pathway_scores_by_donor.csv", index=False)

    top_edge_export = top_edges.head(250).copy()
    top_edge_export.to_csv(output_dir / "step2_top_edge_importance_summary.csv", index=False)

    age_donor_embeddings = donor_level_embeddings(
        project_path("data/processed/step2/06_models/age_model/age_model_embeddings.parquet"),
        graph_manifest,
        "age",
    )
    pd_donor_embeddings = donor_level_embeddings(
        project_path("data/processed/step2/06_models/pd_model/pd_model_embeddings.parquet"),
        graph_manifest,
        "pd",
    )
    donor_meta = pd.merge(
        age_donor_embeddings,
        pd_donor_embeddings,
        on=[
            "split_group_id",
            "cohort_name",
            "source_id",
            "disease_group",
            "age_years",
            "brain_region_standardized",
            "sex_standardized",
        ],
        how="outer",
    )
    age_pred_donor = (
        age_predictions.loc[age_predictions["split"].eq("val")]
        .groupby("split_group_id", observed=False)["pred_age_years"]
        .mean()
        .rename("pred_age_years")
        .reset_index()
    )
    pd_pred_donor = (
        pd_predictions.loc[pd_predictions["split"].eq("val")]
        .groupby("split_group_id", observed=False)["pd_probability"]
        .mean()
        .rename("pd_probability")
        .reset_index()
    )
    donor_meta = donor_meta.merge(age_pred_donor, on="split_group_id", how="left").merge(pd_pred_donor, on="split_group_id", how="left")
    donor_meta = donor_meta.set_index("split_group_id").sort_index()

    age_emb_cols = [column for column in donor_meta.columns if column.startswith("age_emb_")]
    pd_emb_cols = [column for column in donor_meta.columns if column.startswith("pd_emb_")]
    embedding_h5ad_written = write_embedding_exports(
        output_dir=output_dir,
        donor_meta=donor_meta,
        age_emb_cols=age_emb_cols,
        pd_emb_cols=pd_emb_cols,
    )

    lbd_row = robustness.loc[robustness["cohort_name"].eq("lbd_bridge")].head(1)
    summary = "\n".join(
        [
            "# Step 2 Summary Dossier",
            "",
            "## What Was Modeled",
            "- A communication-ready panel with 186,494 Step 1 backbone cells and 2,303 recovered communication genes.",
            "- Four Tier 1 bipartite graph tasks, with `microglia -> dopaminergic neuron` as the primary axis.",
            "- A donor-grouped control aging GNN and an age-conditioned PD deviation GNN.",
            "",
            "## What Held Up",
            markdown_table(step3_candidates.head(12).fillna("NA")),
            "",
            "## Step 4 Priority",
            markdown_table(step4_candidates.head(12).fillna("NA")),
            "",
            "## Bridge And Holdout",
            markdown_table(robustness.fillna("NA")),
            "",
            "## Remaining Caveats",
            "- PD-labeled `microglia -> dopaminergic neuron` donors remain source-skewed toward the CELLxGENE SNpc collection.",
            "- `GSE157783` holdout scoring is informative but still low-sample, with only one eligible control and two eligible PD patients in the microglia-to-DA slice.",
            "- Candidates marked `unstable_or_source_sensitive` should not be promoted to mechanistic claims without additional reruns.",
            (
                "- `step2_embedding_exports.h5ad` was refreshed in this run."
                if embedding_h5ad_written
                else "- `anndata` was unavailable during export, so embeddings were written as `step2_embedding_exports_obs.csv` plus `step2_embedding_exports_matrix.parquet`."
            ),
        ]
    )
    write_markdown(output_dir / "step2_summary_dossier.md", summary)


if __name__ == "__main__":
    main()
