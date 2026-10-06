#!/usr/bin/env python3.12
"""Run confound and robustness checks for Step 3 candidates."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import markdown_table, project_path, write_markdown


DEFAULT_CONFIG = Path("configs/step3/step3_confounds.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 3 confound checks.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def sign_label(value: float) -> str:
    if pd.isna(value):
        return "na"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "flat"


def mean_delta(frame: pd.DataFrame) -> float:
    control = frame.loc[frame["cohort_name"].eq("control_known_age"), "score_normalized"].mean()
    pd_mean = frame.loc[frame["cohort_name"].eq("pd_known_age"), "score_normalized"].mean()
    return float(pd_mean - control) if pd.notna(control) and pd.notna(pd_mean) else float("nan")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    output_dir = project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = config["thresholds"]

    registry = pd.read_csv(project_path("data/processed/step3/01_registry/step3_candidate_registry.csv"))
    task_support = pd.read_csv(project_path("data/processed/step3/04_task_support/step3_task_support_matrix.csv"))
    trajectory = pd.read_csv(project_path("data/processed/step3/03_donor_patterns/step3_trajectory_summary.csv"))
    state_support = pd.read_csv(project_path("data/processed/step3/05_states/step3_candidate_state_support.csv"))
    refined_map = pd.read_csv(project_path("data/processed/step3/02_programs/step3_refined_program_map.csv"))
    pair_scores = pd.read_csv(project_path("data/processed/step3/03_donor_patterns/step3_pair_scores_by_donor.csv"), low_memory=False)
    program_scores = pd.read_csv(project_path("data/processed/step3/03_donor_patterns/step3_program_scores_by_donor.csv"), low_memory=False)
    holdout_lr = pd.read_csv(project_path("data/processed/step2/07_robustness/step2_holdout_lr_effects.csv"))
    label_conf = pd.read_csv(project_path("data/processed/step2/07_robustness/step2_label_confidence_sensitivity.csv"))

    top_state = state_support.loc[state_support["state_pair_rank"].eq(1)].set_index("candidate_id")
    task_index = task_support.set_index("candidate_id")
    trajectory_index = trajectory.set_index("candidate_id")
    registry_pair = registry.loc[registry["entity_level"].eq("pair")].set_index("candidate_key")
    program_anchor = (
        refined_map.groupby("refined_program_id", observed=False)["pair_key"]
        .agg(lambda values: "|".join(sorted(dict.fromkeys(str(value) for value in values))))
        .to_dict()
    )

    candidate_frames = []
    for _, row in registry.loc[registry["entity_level"].eq("pair")].iterrows():
        candidate_frames.append(
            {
                "candidate_id": row["candidate_key"],
                "candidate_level": "pair",
                "candidate_label": row["candidate_id"],
                "raw_pathway_family": row.get("raw_pathway_family", ""),
                "holdout_support": row.get("holdout_support"),
                "sex_restriction_flag": bool(row.get("sex_restriction_flag", False)),
                "annotation_ambiguity_flag": bool(row.get("annotation_ambiguity_flag", False)),
            }
        )
    for program_id, anchor_pairs in program_anchor.items():
        candidate_frames.append(
            {
                "candidate_id": f"program::{program_id}",
                "candidate_level": "program",
                "candidate_label": program_id,
                "raw_pathway_family": "",
                "holdout_support": holdout_lr.loc[holdout_lr["pair_key"].isin(anchor_pairs.split("|")), "holdout_pd_minus_control"].mean(),
                "sex_restriction_flag": any("Y->" in pair or "NLGN4Y" in pair for pair in anchor_pairs.split("|")),
                "annotation_ambiguity_flag": program_id == "exploratory_residual_program",
            }
        )
    candidate_table = pd.DataFrame(candidate_frames)

    rows: list[dict[str, object]] = []
    for row in candidate_table.itertuples(index=False):
        if row.candidate_level == "pair":
            local = pair_scores.loc[pair_scores["candidate_id"].eq(row.candidate_id)].copy()
        else:
            local = program_scores.loc[program_scores["candidate_id"].eq(row.candidate_id)].copy()
        if local.empty or row.candidate_id not in task_index.index:
            continue
        primary_task = str(task_index.loc[row.candidate_id, "primary_task"])
        task_local = local.loc[local["task_name"].eq(primary_task)].copy()
        overall_delta = mean_delta(task_local)
        source_deltas = task_local.groupby("source_id", observed=False).apply(mean_delta)
        source_flip = any(sign_label(value) != sign_label(overall_delta) for value in source_deltas.dropna())
        region_deltas = task_local.groupby("brain_region_standardized", observed=False).apply(mean_delta)
        region_flip = any(sign_label(value) != sign_label(overall_delta) for value in region_deltas.dropna())
        source_majority_share = float(task_local["source_id"].astype(str).value_counts(normalize=True).max())
        sender_corr = abs(task_local["score_normalized"].corr(task_local["sender_cell_count"], method="spearman"))
        receiver_corr = abs(task_local["score_normalized"].corr(task_local["receiver_cell_count"], method="spearman"))
        max_cell_corr = max(sender_corr if pd.notna(sender_corr) else 0.0, receiver_corr if pd.notna(receiver_corr) else 0.0)

        if row.sex_restriction_flag:
            male = task_local.loc[task_local["sex_standardized"].eq("male")].copy()
            male_delta = mean_delta(male)
            if male["cohort_name"].eq("control_known_age").sum() < 1 or male["cohort_name"].eq("pd_known_age").sum() < 1:
                sex_status = "limited"
            elif sign_label(male_delta) != sign_label(overall_delta):
                sex_status = "fail"
            else:
                sex_status = "pass"
        else:
            sex_status = "not_applicable"

        directionality_row = top_state.loc[row.candidate_id] if row.candidate_id in top_state.index else None
        if directionality_row is None:
            directionality_status = "missing"
        elif directionality_row["state_support_status"] == "supported":
            directionality_status = "pass"
        else:
            directionality_status = "weak"

        label_slice = label_conf.loc[
            label_conf["task_name"].eq(primary_task) & label_conf["confidence_threshold"].ge(0.75)
        ].copy()
        control_units = label_slice.loc[label_slice["cohort_name"].eq("control_known_age"), "n_graph_units"].sum()
        pd_units = label_slice.loc[label_slice["cohort_name"].eq("pd_known_age"), "n_graph_units"].sum()
        label_confidence_status = "pass" if control_units >= 2 and pd_units >= 2 else "limited"

        caveats: list[str] = []
        if source_majority_share >= float(thresholds["source_majority_share"]):
            caveats.append("source_dominant")
        if source_flip:
            caveats.append("source_flip")
        if region_flip:
            caveats.append("region_flip")
        if max_cell_corr >= float(thresholds["cell_count_correlation_warning"]):
            caveats.append("cell_count_link")
        if sex_status in {"limited", "fail"}:
            caveats.append(f"sex_{sex_status}")
        if directionality_status != "pass":
            caveats.append(f"directionality_{directionality_status}")
        if row.annotation_ambiguity_flag:
            caveats.append("annotation_ambiguity")
        if pd.notna(row.holdout_support) and float(row.holdout_support) < 0:
            caveats.append("negative_holdout")

        if directionality_status == "pass" and not source_flip and sex_status != "fail" and len(caveats) <= 1:
            promotion_status = "validation_ready"
        elif directionality_status in {"pass", "weak"} and sex_status != "fail" and len(caveats) <= 3:
            promotion_status = "strong_narrative"
        elif directionality_status == "missing" or "source_flip" in caveats or sex_status == "fail":
            promotion_status = "hold_back"
        else:
            promotion_status = "exploratory"

        rows.append(
            {
                "candidate_id": row.candidate_id,
                "candidate_level": row.candidate_level,
                "candidate_label": row.candidate_label,
                "primary_task": primary_task,
                "source_majority_share": source_majority_share,
                "source_direction_flip": source_flip,
                "region_direction_flip": region_flip,
                "cell_count_correlation_abs": max_cell_corr,
                "sex_sensitivity_status": sex_status,
                "directionality_status": directionality_status,
                "label_confidence_status": label_confidence_status,
                "holdout_support": row.holdout_support,
                "annotation_ambiguity_flag": row.annotation_ambiguity_flag,
                "promotion_status": promotion_status,
                "caveat_summary": "|".join(caveats) if caveats else "clear",
            }
        )

    confound_summary = pd.DataFrame(rows).sort_values(["candidate_level", "promotion_status", "holdout_support"], ascending=[True, True, False])
    confound_summary.to_csv(output_dir / "step3_confound_summary.csv", index=False)
    confound_summary[["candidate_id", "candidate_level", "candidate_label", "promotion_status", "caveat_summary"]].to_csv(
        output_dir / "step3_promotion_status.csv",
        index=False,
    )

    report = "\n".join(
        [
            "# Step 3 Source Sex Region Sensitivity",
            "",
            "## Confound Summary Preview",
            markdown_table(confound_summary.head(20).fillna("NA")),
            "",
            "## Promotion Counts",
            markdown_table(
                confound_summary["promotion_status"].value_counts(dropna=False).rename_axis("promotion_status").reset_index(name="count")
            ),
        ]
    )
    write_markdown(output_dir / "step3_source_sex_region_sensitivity.md", report)


if __name__ == "__main__":
    main()

