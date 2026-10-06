#!/usr/bin/env python3.12
"""Model donor trajectories for Step 3 pair and program candidates."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import markdown_table, project_path, write_markdown
from scripts.step3.common import (
    classify_bridge_pattern,
    disease_effects,
    direction_label,
    linear_beta,
)


DEFAULT_CONFIG = Path("configs/step3/step3_donor_models.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model donor trajectories for Step 3.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def choose_primary_task(frame: pd.DataFrame) -> str:
    support = (
        frame.groupby("task_name", observed=False)["score_normalized"]
        .apply(lambda values: values.abs().mean())
        .sort_values(ascending=False)
    )
    return str(support.index[0]) if not support.empty else "unknown"


def trajectory_class(age_beta: float, pd_shift: float, bridge_class: str) -> str:
    if pd.notna(pd_shift) and pd_shift > 0.2 and pd.notna(age_beta) and age_beta > 0.1:
        return "pd_amplified_aging"
    if pd.notna(pd_shift) and pd_shift > 0.2:
        return "pd_specific_gain"
    if pd.notna(pd_shift) and pd_shift < -0.2:
        return "pd_specific_loss"
    if pd.notna(age_beta) and age_beta > 0.1:
        return "aging_increase"
    if pd.notna(age_beta) and age_beta < -0.1:
        return "aging_decrease"
    if bridge_class == "bridge_intermediate":
        return "bridge_intermediate"
    if bridge_class == "bridge_divergent":
        return "bridge_divergent"
    return "uncertain"


def plot_candidate(frame: pd.DataFrame, candidate_slug: str, output_path: Path) -> None:
    known = frame.loc[frame["age_years"].notna()].copy()
    if known.empty:
        return
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=known,
        x="age_years",
        y="score_normalized",
        hue="cohort_name",
        style="disease_group",
        ax=ax,
        s=90,
    )
    ax.set_title(candidate_slug.replace("_", " "))
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Normalized Step 3 score")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_candidates(frame: pd.DataFrame, *, level: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate_id, local in frame.groupby("candidate_id", observed=False):
        primary_task = choose_primary_task(local)
        task_local = local.loc[local["task_name"].eq(primary_task)].copy()
        controls = task_local.loc[task_local["cohort_name"].eq("control_known_age") & task_local["age_years"].notna()].copy()
        known = task_local.loc[task_local["cohort_name"].isin(["control_known_age", "pd_known_age"]) & task_local["age_years"].notna()].copy()
        age_beta = linear_beta(controls, "age_years", "score_normalized")
        disease_stats = disease_effects(known, "score_normalized")
        mean_control = controls["score_normalized"].mean()
        mean_pd = task_local.loc[task_local["cohort_name"].eq("pd_known_age"), "score_normalized"].mean()
        mean_lbd = task_local.loc[task_local["cohort_name"].eq("lbd_bridge"), "score_normalized"].mean()
        bridge_class = classify_bridge_pattern(mean_control, mean_pd, mean_lbd)
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_level": level,
                "candidate_label": str(task_local.iloc[0].get("pair_key") or task_local.iloc[0].get("refined_program_name") or candidate_id),
                "primary_task": primary_task,
                "control_age_beta": age_beta,
                "control_age_direction": direction_label(age_beta),
                "pd_shift_size": disease_stats["pd_shift_size"],
                "pd_shift_direction": direction_label(disease_stats["pd_shift_size"]),
                "pd_age_interaction_beta": disease_stats["pd_age_interaction_beta"],
                "pd_age_interaction_direction": direction_label(disease_stats["pd_age_interaction_beta"]),
                "mean_control_score": mean_control,
                "mean_pd_score": mean_pd,
                "mean_lbd_score": mean_lbd,
                "lbd_bridge_class": bridge_class,
                "trajectory_class": trajectory_class(age_beta, disease_stats["pd_shift_size"], bridge_class),
                "n_control": int(task_local["cohort_name"].eq("control_known_age").sum()),
                "n_pd": int(task_local["cohort_name"].eq("pd_known_age").sum()),
                "n_lbd": int(task_local["cohort_name"].eq("lbd_bridge").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["candidate_level", "pd_shift_size"], ascending=[True, False])


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    output_dir = project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "step3_donor_trajectory_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    pair_scores = pd.read_csv(output_dir / "step3_pair_scores_by_donor.csv", low_memory=False)
    program_scores = pd.read_csv(output_dir / "step3_program_scores_by_donor.csv", low_memory=False)
    holdout_lr = pd.read_csv(project_path("data/processed/step2/07_robustness/step2_holdout_lr_effects.csv"))
    anchor_pairs = pd.read_csv(project_path("data/processed/step3/02_programs/step3_refined_program_map.csv"))
    registry = pd.read_csv(project_path("data/processed/step3/01_registry/step3_candidate_registry.csv"))

    pair_summary = summarize_candidates(pair_scores, level="pair")
    pair_summary = pair_summary.merge(
        pair_scores[["candidate_id", "pair_key"]].drop_duplicates(),
        on="candidate_id",
        how="left",
    ).merge(
        holdout_lr[["pair_key", "holdout_pd_minus_control"]].rename(columns={"holdout_pd_minus_control": "holdout_direction_concordance"}),
        on="pair_key",
        how="left",
    )
    program_meta = program_scores[["candidate_id", "refined_program_id", "refined_program_name"]].drop_duplicates()
    anchor_summary = (
        anchor_pairs.groupby("refined_program_id", observed=False)["pair_key"]
        .agg(lambda values: "|".join(sorted(dict.fromkeys(str(value) for value in values))))
        .rename("anchor_pairs")
        .reset_index()
    )
    program_summary = summarize_candidates(program_scores, level="program").merge(
        program_meta,
        on="candidate_id",
        how="left",
    )
    program_holdout = (
        anchor_pairs.merge(holdout_lr[["pair_key", "holdout_pd_minus_control"]], on="pair_key", how="left")
        .groupby("refined_program_id", observed=False)["holdout_pd_minus_control"]
        .mean()
        .rename("holdout_direction_concordance")
        .reset_index()
    )
    program_summary = program_summary.merge(anchor_summary, on="refined_program_id", how="left").merge(
        program_holdout, on="refined_program_id", how="left"
    )

    trajectory_summary = pd.concat([pair_summary, program_summary], ignore_index=True, sort=False)
    trajectory_summary.to_csv(output_dir / "step3_trajectory_summary.csv", index=False)

    for existing in plot_dir.glob("*.png"):
        existing.unlink()
    pair_plot_ids = set(registry.loc[registry["entity_level"].eq("pair"), "candidate_key"].astype(str))
    for candidate_id in sorted(pair_plot_ids):
        local = pair_scores.loc[pair_scores["candidate_id"].eq(candidate_id)].copy()
        if not local.empty:
            plot_candidate(local, candidate_id.replace("::", "_"), plot_dir / f"{candidate_id.replace('::', '_')}.png")
    for candidate_id, local in program_scores.groupby("candidate_id", observed=False):
        plot_candidate(local, candidate_id.replace("::", "_"), plot_dir / f"{candidate_id.replace('::', '_')}.png")

    report = "\n".join(
        [
            "# Step 3 Donor Trajectory Summary",
            "",
            "## Candidate Trajectory Preview",
            markdown_table(trajectory_summary.head(20).fillna("NA")),
            "",
            "## Program Holdout Concordance",
            markdown_table(program_holdout.fillna("NA")),
        ]
    )
    write_markdown(output_dir / "step3_trajectory_report.md", report)


if __name__ == "__main__":
    main()
