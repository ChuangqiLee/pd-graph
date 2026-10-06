#!/usr/bin/env python3.12
"""Run donor-level primary spatial hypothesis tests for Step 4."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import markdown_table, write_markdown
from scripts.step4.common import bh_fdr, ensure_step4_dir


DEFAULT_OUTPUT_DIR = Path("data/processed/step4/03_spatial/04_primary_tests")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the primary Step 4 spatial hypothesis tests.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def disease_direction_matches(expected_direction: str, beta: float) -> bool:
    if pd.isna(beta):
        return False
    if "loss" in str(expected_direction):
        return float(beta) < 0.0
    if "gain" in str(expected_direction):
        return float(beta) > 0.0
    return False


def directional_one_sided_pvalue(expected_direction: str, beta: float, two_sided_pvalue: float) -> float:
    if pd.isna(beta) or pd.isna(two_sided_pvalue):
        return float("nan")
    half_p = max(min(float(two_sided_pvalue) / 2.0, 1.0), 0.0)
    if disease_direction_matches(expected_direction, beta):
        return half_p
    return 1.0 - half_p


def stouffer_pvalue(pvalues: list[float]) -> float:
    finite = [float(value) for value in pvalues if pd.notna(value)]
    if not finite:
        return float("nan")
    clipped = np.clip(np.asarray(finite, dtype=float), 1e-12, 1.0 - 1e-12)
    z_scores = stats.norm.isf(clipped)
    combined_z = float(np.sum(z_scores) / np.sqrt(len(z_scores)))
    return float(stats.norm.sf(combined_z))


def ols_beta_pvalue(frame: pd.DataFrame, response: str) -> tuple[float, float]:
    local = frame[[response, "disease_binary", "log_spot_count"]].dropna().copy()
    if len(local) < 4 or local["disease_binary"].nunique() < 2:
        return float("nan"), float("nan")
    response_vec = local[response].astype(float).to_numpy()
    design = np.column_stack(
        [
            np.ones(len(local), dtype=float),
            local["disease_binary"].astype(float).to_numpy(),
            local["log_spot_count"].astype(float).to_numpy(),
        ]
    )
    try:
        coefficients, _, rank, _ = np.linalg.lstsq(design, response_vec, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")

    n_obs, n_params = design.shape
    disease_beta = float(coefficients[1]) if len(coefficients) > 1 else float("nan")
    if rank < n_params or n_obs <= n_params:
        return disease_beta, float("nan")

    residuals = response_vec - design @ coefficients
    degrees_of_freedom = n_obs - n_params
    sigma_sq = float(np.dot(residuals, residuals) / degrees_of_freedom)
    covariance = sigma_sq * np.linalg.pinv(design.T @ design)
    disease_se = float(np.sqrt(max(covariance[1, 1], 0.0)))
    if not np.isfinite(disease_se) or disease_se == 0.0:
        return disease_beta, float("nan")

    t_stat = disease_beta / disease_se
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=degrees_of_freedom))
    return disease_beta, p_value


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "program_primary_spatial_figures"
    ensure_step4_dir("03_spatial", "04_primary_tests")
    figure_dir.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(PROJECT_ROOT / "data/processed/step4/01_registry/step4_validation_registry.csv")
    if registry.empty:
        pd.DataFrame(
            columns=[
                "program_id",
                "program_name",
                "expected_pd_direction",
                "sample_accession",
                "disease_group",
                "anatomical_compartment",
                "n_spots",
                "adjacency_excess_score",
                "adjacent_program_score",
                "same_spot_program_score",
                "state_coenrichment_score",
                "program_module_context_score",
                "disease_binary",
                "log_spot_count",
            ]
        ).to_csv(output_dir / "program_donor_level_spatial_summary.csv", index=False)
        pd.DataFrame(
            columns=[
                "program_id",
                "program_name",
                "expected_pd_direction",
                "n_pd_donors",
                "n_control_donors",
                "snpc_adjacency_beta",
                "snpc_adjacency_p_value",
                "contrast_beta",
                "contrast_p_value",
                "same_spot_beta",
                "same_spot_p_value",
                "aligned_metric_count",
                "directional_p_snpc",
                "directional_p_contrast",
                "directional_p_same_spot",
                "composite_directional_p_value",
                "direction_match",
                "primary_spatial_support_bin",
                "snpc_adjacency_fdr",
                "contrast_fdr",
            ]
        ).to_csv(output_dir / "program_primary_spatial_results.csv", index=False)
        methods = [
            "# Primary Spatial Methods",
            "",
            "No Step 4 programs were available for primary spatial testing in this run.",
            "",
        ]
        write_markdown(output_dir / "primary_spatial_methods.md", "\n".join(methods))
        return
    program_scores = pd.read_csv(PROJECT_ROOT / "data/processed/step4/03_spatial/03_anchor_scoring/program_spatial_scores.csv")

    donor_summary = (
        program_scores.groupby(["program_id", "program_name", "expected_pd_direction", "sample_accession", "disease_group", "anatomical_compartment"], observed=False)
        .agg(
            n_spots=("spot_id", "count"),
            adjacency_excess_score=("adjacency_excess_score", "mean"),
            adjacent_program_score=("adjacent_program_score", "mean"),
            same_spot_program_score=("same_spot_program_score", "mean"),
            state_coenrichment_score=("state_coenrichment_score", "mean"),
            program_module_context_score=("program_module_context_score", "mean"),
        )
        .reset_index()
    )
    donor_summary["disease_binary"] = donor_summary["disease_group"].map({"control": 0.0, "pd": 1.0}).astype(float)
    donor_summary["log_spot_count"] = donor_summary["n_spots"].astype(float).map(lambda value: np.log1p(value))
    donor_summary.to_csv(output_dir / "program_donor_level_spatial_summary.csv", index=False)

    results: list[dict[str, object]] = []
    for program_id, frame in donor_summary.groupby("program_id", observed=False):
        snpc = frame.loc[frame["anatomical_compartment"].eq("SNpc")].copy()
        surrounding = frame.loc[frame["anatomical_compartment"].eq("surrounding_tissue")].copy()
        snpc = snpc.rename(columns={column: f"{column}_snpc" for column in ["n_spots", "adjacency_excess_score", "adjacent_program_score", "same_spot_program_score", "state_coenrichment_score", "program_module_context_score", "log_spot_count"]})
        surrounding = surrounding.rename(columns={column: f"{column}_surrounding" for column in ["n_spots", "adjacency_excess_score", "adjacent_program_score", "same_spot_program_score", "state_coenrichment_score", "program_module_context_score", "log_spot_count"]})
        merged = snpc.merge(
            surrounding[
                [
                    "sample_accession",
                    "n_spots_surrounding",
                    "adjacency_excess_score_surrounding",
                    "adjacent_program_score_surrounding",
                    "same_spot_program_score_surrounding",
                    "state_coenrichment_score_surrounding",
                    "program_module_context_score_surrounding",
                    "log_spot_count_surrounding",
                ]
            ],
            on="sample_accession",
            how="left",
        )
        merged["contrast_adjacency_excess"] = merged["adjacency_excess_score_snpc"] - merged["adjacency_excess_score_surrounding"]
        merged["log_spot_count"] = merged["log_spot_count_snpc"]

        beta_snpc, p_snpc = ols_beta_pvalue(merged.rename(columns={"adjacency_excess_score_snpc": "response"}), "response")
        beta_contrast, p_contrast = ols_beta_pvalue(merged.rename(columns={"contrast_adjacency_excess": "response"}), "response")
        beta_same, p_same = ols_beta_pvalue(merged.rename(columns={"same_spot_program_score_snpc": "response"}), "response")
        expected_direction = str(merged["expected_pd_direction"].iloc[0])
        aligned_metric_count = int(
            disease_direction_matches(expected_direction, beta_snpc)
            + disease_direction_matches(expected_direction, beta_contrast)
            + disease_direction_matches(expected_direction, beta_same)
        )
        direction_match = aligned_metric_count >= 2
        directional_p_snpc = directional_one_sided_pvalue(expected_direction, beta_snpc, p_snpc)
        directional_p_contrast = directional_one_sided_pvalue(expected_direction, beta_contrast, p_contrast)
        directional_p_same = directional_one_sided_pvalue(expected_direction, beta_same, p_same)
        composite_directional_p = stouffer_pvalue([directional_p_snpc, directional_p_contrast, directional_p_same])
        if direction_match and pd.notna(composite_directional_p) and composite_directional_p <= 0.10 and aligned_metric_count == 3:
            support_bin = "strong_primary_spatial_support"
        elif direction_match and pd.notna(composite_directional_p) and composite_directional_p <= 0.20:
            support_bin = "moderate_primary_spatial_support"
        elif direction_match:
            support_bin = "weak_primary_spatial_support"
        else:
            support_bin = "no_primary_spatial_support"
        results.append(
            {
                "program_id": program_id,
                "program_name": str(merged["program_name"].iloc[0]),
                "expected_pd_direction": expected_direction,
                "n_pd_donors": int((merged["disease_group"] == "pd").sum()),
                "n_control_donors": int((merged["disease_group"] == "control").sum()),
                "snpc_adjacency_beta": beta_snpc,
                "snpc_adjacency_p_value": p_snpc,
                "contrast_beta": beta_contrast,
                "contrast_p_value": p_contrast,
                "same_spot_beta": beta_same,
                "same_spot_p_value": p_same,
                "aligned_metric_count": aligned_metric_count,
                "directional_p_snpc": directional_p_snpc,
                "directional_p_contrast": directional_p_contrast,
                "directional_p_same_spot": directional_p_same,
                "composite_directional_p_value": composite_directional_p,
                "direction_match": direction_match,
                "primary_spatial_support_bin": support_bin,
            }
        )

        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        plot_frame = program_scores.loc[
            program_scores["program_id"].eq(program_id) & program_scores["anatomical_compartment"].eq("SNpc"),
            ["disease_group", "adjacency_excess_score"],
        ].copy()
        positions = {"control": 0, "pd": 1}
        palette = {"control": "#7c8c6b", "pd": "#b24c3d"}
        values_by_group = [
            plot_frame.loc[plot_frame["disease_group"].eq(group), "adjacency_excess_score"].dropna().to_numpy(dtype=float)
            for group in ["control", "pd"]
        ]
        box = ax.boxplot(
            values_by_group,
            positions=[positions["control"], positions["pd"]],
            widths=0.55,
            patch_artist=True,
            medianprops={"color": "white", "linewidth": 1.5},
            whiskerprops={"color": "#444444", "linewidth": 1.0},
            capprops={"color": "#444444", "linewidth": 1.0},
        )
        for patch, group in zip(box["boxes"], ["control", "pd"]):
            patch.set_facecolor(palette[group])
            patch.set_edgecolor("#444444")
            patch.set_alpha(0.85)

        rng = np.random.default_rng(13)
        for group, group_frame in plot_frame.groupby("disease_group", observed=False):
            y_values = group_frame["adjacency_excess_score"].dropna().to_numpy(dtype=float)
            if y_values.size == 0:
                continue
            x_values = rng.normal(loc=positions[str(group)], scale=0.035, size=y_values.size)
            ax.scatter(x_values, y_values, color="black", alpha=0.25, s=10, linewidths=0.0)
        ax.set_xticks([positions["control"], positions["pd"]], ["control", "pd"])
        ax.set_title(str(merged["program_name"].iloc[0]))
        ax.set_xlabel("")
        ax.set_ylabel("SNpc adjacency excess score")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{program_id}_snpc_adjacency_boxplot.png", dpi=220)
        plt.close(fig)

    results_frame = pd.DataFrame(results)
    results_frame["snpc_adjacency_fdr"] = bh_fdr(results_frame["snpc_adjacency_p_value"].fillna(1.0))
    results_frame["contrast_fdr"] = bh_fdr(results_frame["contrast_p_value"].fillna(1.0))
    results_frame.to_csv(output_dir / "program_primary_spatial_results.csv", index=False)

    methods = [
        "# Primary Spatial Methods",
        "",
        "The donor was treated as the replication unit. For every program we aggregated the spot-level spatial scores inside `SNpc` and `surrounding_tissue`, then fit donor-level OLS models of the form:",
        "",
        "`response ~ disease_binary + log_spot_count`",
        "",
        "The primary donor-level responses were `adjacency_excess_score` inside `SNpc`, the `SNpc - surrounding_tissue` contrast, and the `same_spot_program_score` inside `SNpc`.",
        "",
        "For each response we converted the two-sided disease-effect p value into a one-sided directional p value using the expected Step 3 PD direction, then combined the aligned metrics with Stouffer meta-analysis.",
        "",
        "The main response remains `adjacency_excess_score`, defined as the real-neighborhood anchor score minus the matched random-neighborhood score.",
        "",
    ]
    write_markdown(output_dir / "primary_spatial_methods.md", "\n".join(methods))


if __name__ == "__main__":
    main()
