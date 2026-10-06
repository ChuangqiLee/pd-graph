#!/usr/bin/env python3.12
"""Build the optional Step 4 internal robustness annex."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import markdown_table, write_markdown
from scripts.step4.common import (
    STEP4_INTERNAL_ANNEX_COLUMNS,
    STEP4_INTERNAL_SOURCE_ANNEX_COLUMNS,
    ensure_step4_dir,
    read_csv_required_columns,
    read_json,
    require_columns,
    require_record_list,
    STEP4_NORMALIZED_HANDOFF_KEYS,
)


DEFAULT_OUTPUT_DIR = Path("data/processed/step4/04_internal_robustness")
HOLDOUT_SOURCE_WEIGHTS = {
    "GSE157783": 1.0,
    "GSE253462": 2.0,
    "aggregate": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Step 4 internal robustness annex.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def expected_direction_matches(expected_direction: str, value: float) -> bool:
    if "loss" in str(expected_direction):
        return float(value) < 0.0
    if "gain" in str(expected_direction):
        return float(value) > 0.0
    return False


def classify_holdout_support_tier(
    source_summary: pd.DataFrame,
    expected_direction: str,
) -> tuple[str, float]:
    if source_summary.empty:
        return "fail", 0.0
    weights = source_summary["holdout_source_id"].map(HOLDOUT_SOURCE_WEIGHTS).fillna(1.0).astype(float)
    matches = source_summary["direction_match"].fillna(False).astype(float)
    weighted_match_fraction = float(np.average(matches, weights=weights)) if weights.sum() > 0 else 0.0
    nigra_rows = source_summary.loc[source_summary["holdout_source_id"].eq("GSE253462")]
    nigra_match = bool(nigra_rows["direction_match"].fillna(False).any()) if not nigra_rows.empty else False
    if weighted_match_fraction >= 0.75:
        return "pass", weighted_match_fraction
    if weighted_match_fraction >= 0.50 or nigra_match:
        return "suggestive", weighted_match_fraction
    return "fail", weighted_match_fraction


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_step4_dir("04_internal_robustness")
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized = read_json(PROJECT_ROOT / "data/processed/step4/00_inputs/step4_handoff_normalized.json")
    require_record_list(normalized, STEP4_NORMALIZED_HANDOFF_KEYS, context="Normalized Step 4 handoff")
    holdout_lr_source_path = PROJECT_ROOT / "data/processed/step2/07_robustness/step2_holdout_lr_effects_by_source.csv"
    holdout_lr_path = PROJECT_ROOT / "data/processed/step2/07_robustness/step2_holdout_lr_effects.csv"
    if holdout_lr_source_path.exists():
        holdout_lr = read_csv_required_columns(
            holdout_lr_source_path,
            ("pair_key", "holdout_source_id", "holdout_pd_minus_control"),
            context="Step 2 holdout LR effects by source",
        )
    else:
        holdout_lr = read_csv_required_columns(
            holdout_lr_path,
            ("pair_key", "holdout_pd_minus_control"),
            context="Step 2 aggregate holdout LR effects",
        )
        holdout_lr["holdout_source_id"] = "aggregate"
    source_coverage = read_csv_required_columns(
        PROJECT_ROOT / "data/processed/step2/07_robustness/step2_source_task_coverage.csv",
        ("task_name", "cohort_name", "n_donors"),
        context="Step 2 source task coverage",
    )

    annex_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    for record in normalized:
        expected_internal_direction = str(record.get("expected_internal_direction", record["expected_pd_direction"]))
        anchor_keys = [anchor["pair_key"] for anchor in record["anchor_pairs_json"]]
        local = holdout_lr.loc[holdout_lr["pair_key"].isin(anchor_keys)].copy()
        if local.empty:
            per_source = pd.DataFrame(columns=["holdout_source_id", "mean_holdout_pd_minus_control", "n_anchor_pairs_found_in_holdout", "direction_match"])
        else:
            per_source = (
                local.groupby("holdout_source_id", observed=False)
                .agg(
                    mean_holdout_pd_minus_control=("holdout_pd_minus_control", "mean"),
                    n_anchor_pairs_found_in_holdout=("pair_key", "nunique"),
                )
                .reset_index()
            )
            per_source["direction_match"] = per_source["mean_holdout_pd_minus_control"].map(
                lambda value: expected_direction_matches(expected_internal_direction, value) if pd.notna(value) else False
            )
        holdout_support_tier, weighted_match_fraction = classify_holdout_support_tier(per_source, expected_internal_direction)
        weights = per_source["holdout_source_id"].map(HOLDOUT_SOURCE_WEIGHTS).fillna(1.0).astype(float)
        mean_holdout = (
            float(np.average(per_source["mean_holdout_pd_minus_control"], weights=weights))
            if not per_source.empty and per_source["mean_holdout_pd_minus_control"].notna().any()
            else float("nan")
        )
        gse157783_row = per_source.loc[per_source["holdout_source_id"].eq("GSE157783")]
        gse253462_row = per_source.loc[per_source["holdout_source_id"].eq("GSE253462")]
        annex_rows.append(
            {
                "program_id": record["program_id"],
                "program_name": record["program_name"],
                "n_anchor_pairs_found_in_holdout": int(local["pair_key"].nunique()) if not local.empty else 0,
                "mean_holdout_pd_minus_control": mean_holdout,
                "holdout_direction_match": holdout_support_tier != "fail",
                "holdout_support_tier": holdout_support_tier,
                "holdout_weighted_match_fraction": weighted_match_fraction,
                "n_holdout_sources_available": int(per_source["holdout_source_id"].nunique()) if not per_source.empty else 0,
                "n_holdout_sources_direction_match": int(per_source["direction_match"].fillna(False).sum()) if not per_source.empty else 0,
                "gse157783_mean_holdout_pd_minus_control": (
                    float(gse157783_row["mean_holdout_pd_minus_control"].iloc[0]) if not gse157783_row.empty else float("nan")
                ),
                "gse157783_direction_match": bool(gse157783_row["direction_match"].iloc[0]) if not gse157783_row.empty else False,
                "gse253462_mean_holdout_pd_minus_control": (
                    float(gse253462_row["mean_holdout_pd_minus_control"].iloc[0]) if not gse253462_row.empty else float("nan")
                ),
                "gse253462_direction_match": bool(gse253462_row["direction_match"].iloc[0]) if not gse253462_row.empty else False,
                "expected_pd_direction": record["expected_pd_direction"],
                "expected_internal_direction": expected_internal_direction,
            }
        )
        for source_record in per_source.to_dict(orient="records"):
            source_rows.append(
                {
                    "program_id": record["program_id"],
                    "program_name": record["program_name"],
                    "expected_pd_direction": record["expected_pd_direction"],
                    "expected_internal_direction": expected_internal_direction,
                    **source_record,
                }
            )
    if not normalized:
        annex = pd.DataFrame(
            columns=STEP4_INTERNAL_ANNEX_COLUMNS,
        )
        source_annex = pd.DataFrame(
            columns=STEP4_INTERNAL_SOURCE_ANNEX_COLUMNS,
        )
    else:
        annex = pd.DataFrame(annex_rows)
        source_annex = pd.DataFrame(source_rows)
    require_columns(annex, STEP4_INTERNAL_ANNEX_COLUMNS, context="Step 4 internal robustness annex")
    require_columns(source_annex, STEP4_INTERNAL_SOURCE_ANNEX_COLUMNS, context="Step 4 internal robustness source annex")
    annex.to_csv(output_dir / "holdout_annex.csv", index=False)
    source_annex.to_csv(output_dir / "holdout_annex_by_source.csv", index=False)

    coverage_summary = (
        source_coverage.loc[source_coverage["task_name"].eq("microglia_to_dopaminergic_neuron")]
        .sort_values(["cohort_name", "n_donors"], ascending=[True, False])
        .reset_index(drop=True)
    )
    lines = [
        "# Source Sensitivity Annex",
        "",
        "This annex summarizes the internal source coverage context for the primary microglia-to-dopaminergic-neuron task. It is reported separately from the main external validation evidence matrix.",
        "",
        "## Holdout summary",
        markdown_table(annex.astype(object).fillna("NA")),
        "",
        "## Holdout summary by source",
        markdown_table(source_annex.astype(object).fillna("NA")),
        "",
        "## Source coverage for the primary task",
        markdown_table(coverage_summary.astype(object).fillna("NA")),
        "",
    ]
    write_markdown(output_dir / "source_sensitivity_annex.md", "\n".join(lines))


if __name__ == "__main__":
    main()
