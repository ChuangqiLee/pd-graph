#!/usr/bin/env python3.12
"""Map the Step 3 sender and receiver states onto the primary Visium object."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import markdown_table, project_path, write_markdown
from scripts.step4.common import PRIMARY_STATE_IDS, ensure_step4_dir


DEFAULT_OUTPUT_DIR = Path("data/processed/step4/03_spatial/02_state_mapping")
TOP_MARKERS = 25
SENSITIVITY_MARKERS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map Step 3 states to primary Visium space.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def dense_array(matrix: sp.spmatrix | np.ndarray) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def build_signature_catalog(rankings: pd.DataFrame, spatial_genes: set[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for state_id in PRIMARY_STATE_IDS:
        local = rankings.loc[rankings["state_id"].eq(state_id)].copy()
        local = local.loc[local["gene_symbol"].isin(spatial_genes)].copy()
        local["signature_weight"] = local["log2_fc"].clip(lower=0.0) * local["auc"].clip(lower=0.5) * local["detection_fraction_in"].clip(lower=0.05)
        local = local.sort_values(["signature_weight", "p_adj", "state_rank"], ascending=[False, True, True]).head(TOP_MARKERS)
        local["signature_rank"] = np.arange(1, len(local) + 1)
        rows.append(local)
    catalog = pd.concat(rows, ignore_index=True)
    return catalog[
        [
            "state_id",
            "state_label",
            "gene_symbol",
            "signature_rank",
            "signature_weight",
            "log2_fc",
            "auc",
            "detection_fraction_in",
            "p_adj",
        ]
    ].copy()


def gene_zscore_frame(adata: ad.AnnData, genes: list[str]) -> pd.DataFrame:
    expr = pd.DataFrame(dense_array(adata[:, genes].X), index=adata.obs_names, columns=genes)
    std = expr.std(axis=0, ddof=0).replace(0.0, np.nan)
    z = (expr - expr.mean(axis=0)) / std
    return z.fillna(0.0)


def main() -> None:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    ensure_step4_dir("03_spatial", "02_state_mapping")

    visium = ad.read_h5ad(PROJECT_ROOT / "data/processed/step4/03_spatial/01_primary_visium/primary_visium_filtered.h5ad")
    rankings = pd.read_csv(PROJECT_ROOT / "data/processed/step4/00_inputs/step3_state_marker_rankings.csv")

    spatial_genes = set(visium.var_names.astype(str))
    signature_catalog = build_signature_catalog(rankings, spatial_genes)
    signature_catalog.to_csv(output_dir / "state_signature_catalog.csv", index=False)

    union_genes = sorted(signature_catalog["gene_symbol"].astype(str).unique().tolist())
    gene_z = gene_zscore_frame(visium, union_genes)

    result = visium.obs[
        [
            "spot_id",
            "sample_accession",
            "donor_id",
            "disease_group",
            "sex",
            "age_years",
            "anatomical_compartment",
            "da_combined_score",
        ]
    ].copy()

    for state_id in PRIMARY_STATE_IDS:
        local = signature_catalog.loc[signature_catalog["state_id"].eq(state_id)].copy()
        genes = local["gene_symbol"].astype(str).tolist()
        weights = local["signature_weight"].astype(float).to_numpy()
        local_z = gene_z[genes].to_numpy(dtype=float)
        primary_score = np.dot(local_z, weights) / max(np.sum(np.abs(weights)), 1e-9)
        sensitivity_genes = genes[: min(SENSITIVITY_MARKERS, len(genes))]
        sensitivity_score = gene_z[sensitivity_genes].mean(axis=1).to_numpy(dtype=float) if sensitivity_genes else np.zeros(visium.n_obs, dtype=float)
        result[f"primary_score__{state_id}"] = primary_score
        result[f"sensitivity_score__{state_id}"] = sensitivity_score
        result[f"primary_score_01__{state_id}"] = pd.Series(primary_score, index=result.index).rank(method="average", pct=True).to_numpy(dtype=float)
        result[f"sensitivity_score_01__{state_id}"] = pd.Series(sensitivity_score, index=result.index).rank(method="average", pct=True).to_numpy(dtype=float)

    primary_cols = [f"primary_score__{state_id}" for state_id in PRIMARY_STATE_IDS]
    sensitivity_cols = [f"sensitivity_score__{state_id}" for state_id in PRIMARY_STATE_IDS]
    result["dominant_state_primary"] = np.asarray(PRIMARY_STATE_IDS)[np.argmax(result[primary_cols].to_numpy(dtype=float), axis=1)]
    result["dominant_state_sensitivity"] = np.asarray(PRIMARY_STATE_IDS)[np.argmax(result[sensitivity_cols].to_numpy(dtype=float), axis=1)]
    result.to_parquet(output_dir / "primary_visium_state_scores.parquet", index=False)

    report_rows: list[dict[str, object]] = []
    for state_id in PRIMARY_STATE_IDS:
        corr = float(np.corrcoef(result[f"primary_score__{state_id}"], result[f"sensitivity_score__{state_id}"])[0, 1])
        comp = (
            result.groupby("anatomical_compartment", observed=False)[[f"primary_score__{state_id}", f"sensitivity_score__{state_id}"]]
            .mean()
            .reset_index()
        )
        row = {
            "state_id": state_id,
            "primary_sensitivity_correlation": corr,
            "mean_primary_in_snpc": float(comp.loc[comp["anatomical_compartment"].eq("SNpc"), f"primary_score__{state_id}"].iloc[0]) if (comp["anatomical_compartment"] == "SNpc").any() else np.nan,
            "mean_primary_in_surrounding": float(comp.loc[comp["anatomical_compartment"].eq("surrounding_tissue"), f"primary_score__{state_id}"].iloc[0]) if (comp["anatomical_compartment"] == "surrounding_tissue").any() else np.nan,
        }
        report_rows.append(row)
    report = pd.DataFrame(report_rows)
    lines = [
        "# State Mapping Sensitivity Report",
        "",
        "Primary mapping used weighted marker-signature projection from the rebuilt Step 3 marker rankings. Sensitivity mapping used an unweighted average of the top signature genes per state.",
        "",
        markdown_table(report.astype(object).fillna("NA")),
        "",
    ]
    write_markdown(output_dir / "state_mapping_sensitivity_report.md", "\n".join(lines))


if __name__ == "__main__":
    main()
