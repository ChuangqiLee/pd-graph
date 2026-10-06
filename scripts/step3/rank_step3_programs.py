#!/usr/bin/env python3.12
"""Rank Step 3 refined programs and assign final tiers."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import markdown_table, project_path, write_markdown
from scripts.step3.common import (
    PRIMARY_TASK,
    PROGRAM_METADATA,
    minmax_series,
    pair_candidate_id,
    program_candidate_id,
    program_name,
    top_program_anchor_pairs,
)


DEFAULT_OUTPUT_DIR = Path("data/processed/step3/09_top10")
ARCHIVE_ROOT = Path("data/processed/full_refresh_archives")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank Step 3 programs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_program_experiment_consensus(archive_root: Path) -> pd.DataFrame:
    archive_dir = project_path(archive_root)
    if not archive_dir.exists():
        return pd.DataFrame(
            columns=[
                "refined_program_id",
                "consensus_run_count",
                "consensus_top10_fraction",
                "consensus_medium_priority_fraction",
                "consensus_external_support_fraction",
            ]
        )

    rows: list[dict[str, object]] = []
    run_count = 0
    for experiment_dir in sorted(path for path in archive_dir.iterdir() if path.is_dir()):
        top10_path = experiment_dir / "step3_top10_refined_programs.csv"
        if not top10_path.exists():
            continue
        top10 = safe_read_csv(top10_path)
        if top10.empty or "refined_program_id" not in top10.columns:
            continue
        run_count += 1
        final_path = experiment_dir / "step4_final_evidence_matrix.csv"
        final_matrix = safe_read_csv(final_path) if final_path.exists() else pd.DataFrame()
        final_lookup: dict[str, dict[str, object]] = {}
        if not final_matrix.empty and "program_id" in final_matrix.columns:
            final_lookup = final_matrix.set_index("program_id").to_dict(orient="index")
        for row in top10.itertuples(index=False):
            final_row = final_lookup.get(str(row.refined_program_id), {})
            claim_tier = str(final_row.get("claim_tier", ""))
            rows.append(
                {
                    "refined_program_id": str(row.refined_program_id),
                    "top10_flag": 1.0,
                    "medium_priority_flag": 1.0 if str(getattr(row, "step4_validation_priority", "")) in {"medium", "high"} else 0.0,
                    "external_support_flag": 1.0 if claim_tier in {"externally_supported_program", "external_mechanistic_candidate"} else 0.0,
                }
            )

    if run_count == 0:
        return pd.DataFrame(
            columns=[
                "refined_program_id",
                "consensus_run_count",
                "consensus_top10_fraction",
                "consensus_medium_priority_fraction",
                "consensus_external_support_fraction",
            ]
        )

    consensus = pd.DataFrame(rows)
    consensus = (
        consensus.groupby("refined_program_id", observed=False)
        .agg(
            top10_count=("top10_flag", "sum"),
            medium_priority_count=("medium_priority_flag", "sum"),
            external_support_count=("external_support_flag", "sum"),
        )
        .reset_index()
    )
    consensus["consensus_run_count"] = int(run_count)
    consensus["consensus_top10_fraction"] = consensus["top10_count"] / float(run_count)
    consensus["consensus_medium_priority_fraction"] = consensus["medium_priority_count"] / float(run_count)
    consensus["consensus_external_support_fraction"] = consensus["external_support_count"] / float(run_count)
    return consensus[
        [
            "refined_program_id",
            "consensus_run_count",
            "consensus_top10_fraction",
            "consensus_medium_priority_fraction",
            "consensus_external_support_fraction",
        ]
    ]


def main() -> None:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry_path = project_path("data/processed/step3/01_registry/step3_candidate_registry.csv")
    registry = pd.read_csv(registry_path)
    refined_map = pd.read_csv(project_path("data/processed/step3/02_programs/step3_refined_program_map.csv"))
    pair_universe = pd.read_parquet(project_path("data/processed/step2/03_lr_universe/step2_lr_universe_compact.parquet"))
    ranked_pairs = pd.read_csv(project_path("data/processed/step2/08_interpretation/step2_ranked_lr_pairs.csv"))
    task_support = pd.read_csv(project_path("data/processed/step3/04_task_support/step3_task_support_matrix.csv"))
    trajectory = pd.read_csv(project_path("data/processed/step3/03_donor_patterns/step3_trajectory_summary.csv"))
    state_support = pd.read_csv(project_path("data/processed/step3/05_states/step3_candidate_state_support.csv"))
    confounds = pd.read_csv(project_path("data/processed/step3/07_confounds/step3_confound_summary.csv"))
    pair_scores = pd.read_csv(project_path("data/processed/step3/03_donor_patterns/step3_pair_scores_by_donor.csv"), low_memory=False)
    program_consensus = load_program_experiment_consensus(ARCHIVE_ROOT).set_index("refined_program_id")
    registry_pair = registry.loc[registry["entity_level"].eq("pair")].drop_duplicates("candidate_key").set_index("candidate_key")

    pair_base = refined_map.merge(
        pair_universe[["pair_key", "support_score", "database_support_count", "avg_sender_pct_expr", "avg_receiver_pct_expr"]],
        on="pair_key",
        how="left",
    ).merge(
        ranked_pairs[
            [
                "pair_key",
                "ranking_score",
                "aging_score",
                "pd_score",
                "delta_score",
                "receiver_specificity",
                "holdout_support",
                "fold_stability",
                "source_stability",
                "interpretation_bin",
                "graph_units_supported",
            ]
        ],
        on="pair_key",
        how="left",
        suffixes=("", "_rank"),
    )
    pair_base["candidate_id"] = pair_base["pair_key"].map(pair_candidate_id)
    task_pair = task_support.loc[task_support["candidate_level"].eq("pair")].set_index("candidate_id")
    traj_pair = trajectory.loc[trajectory["candidate_level"].eq("pair")].set_index("candidate_id")
    state_pair = state_support.loc[(state_support["candidate_level"].eq("pair")) & (state_support["state_pair_rank"].eq(1))].set_index("candidate_id")
    confound_pair = confounds.loc[confounds["candidate_level"].eq("pair")].set_index("candidate_id")
    pair_counts = (
        pair_scores.groupby("candidate_id", observed=False)
        .agg(
            n_graph_units_supported=("graph_unit_id", "nunique"),
            n_sources_supported=("source_id", "nunique"),
            n_control_graph_units=("cohort_name", lambda values: int(pd.Series(values).eq("control_known_age").sum())),
            n_pd_graph_units=("cohort_name", lambda values: int(pd.Series(values).eq("pd_known_age").sum())),
            n_lbd_graph_units=("cohort_name", lambda values: int(pd.Series(values).eq("lbd_bridge").sum())),
            mean_contribution=("attention_raw_sum", "mean"),
            max_contribution=("attention_raw_sum", "max"),
        )
        .reset_index()
        .set_index("candidate_id")
    )

    pair_base["model_rank_component"] = minmax_series(pair_base["ranking_score"].fillna(pair_base["support_score"]))
    pair_base["registry_gate"] = pair_base["candidate_id"].map(registry_pair["registry_gate"]).fillna("review")
    pair_base["stability_bin"] = pair_base["candidate_id"].map(registry_pair["stability_bin"]).fillna("watch")
    pair_base["registry_pass_component"] = pair_base["registry_gate"].map({"promote": 1.0, "review": 0.5, "hold": 0.0}).fillna(0.0)
    pair_base["fold_stability_value"] = pair_base["fold_stability"].fillna(pair_base["candidate_id"].map(registry_pair["fold_stability"])).fillna(0.0)
    pair_base["source_stability_value"] = pair_base["source_stability"].fillna(pair_base["candidate_id"].map(registry_pair["source_stability"])).fillna(0.0)
    pair_base["graph_support_component"] = minmax_series(
        pair_base["graph_units_supported"].fillna(pair_base["candidate_id"].map(registry_pair["graph_support_units"])).fillna(0.0)
    )
    pair_base["holdout_component"] = minmax_series(pair_base["holdout_support"].fillna(0.0).clip(lower=0.0))
    pair_base["has_step2_rank_support"] = pair_base["ranking_score"].notna()
    pair_base["stability_component"] = (
        0.40 * pair_base["fold_stability_value"]
        + 0.20 * pair_base["source_stability_value"]
        + 0.20 * pair_base["registry_pass_component"]
        + 0.10 * pair_base["holdout_component"]
        + 0.10 * pair_base["graph_support_component"]
    )
    pair_base["donor_trajectory_component"] = pair_base["candidate_id"].map(
        traj_pair["pd_shift_size"].abs().add(traj_pair["control_age_beta"].abs(), fill_value=0.0).pipe(minmax_series)
    ).fillna(0.0)
    pair_base["task_specificity_component"] = pair_base["candidate_id"].map(task_pair["task_specificity_score"].pipe(minmax_series)).fillna(0.0)
    pair_base["receiver_specificity_component"] = minmax_series(pair_base["receiver_specificity"].fillna(0.0))
    pair_base["state_support_component"] = pair_base["candidate_id"].map(
        (state_pair["ligand_expression_support"] + state_pair["receptor_expression_support"] + state_pair["high_contribution_edge_enrichment"]).pipe(minmax_series)
    ).fillna(0.0)
    penalties = (
        pair_base["candidate_id"].map(confound_pair["source_direction_flip"]).fillna(False).astype(float) * 0.15
        + pair_base["candidate_id"].map(confound_pair["annotation_ambiguity_flag"] if "annotation_ambiguity_flag" in confound_pair.columns else pd.Series(dtype=float)).fillna(False).astype(float) * 0.1
        + pair_base["candidate_id"].map(confound_pair["sex_sensitivity_status"]).fillna("not_applicable").eq("fail").astype(float) * 0.15
        + pair_base["candidate_id"].map(confound_pair["directionality_status"]).fillna("missing").ne("pass").astype(float) * 0.2
        + pair_base["holdout_support"].fillna(0.0).lt(0).astype(float) * 0.1
        + pair_base["registry_gate"].eq("review").astype(float) * 0.05
        + pair_base["registry_gate"].eq("hold").astype(float) * 0.25
        + pair_base["candidate_id"].map(registry_pair["low_fold_stability_flag"]).fillna(False).astype(float) * 0.10
        + pair_base["candidate_id"].map(registry_pair["limited_graph_support_flag"]).fillna(False).astype(float) * 0.05
        + pair_base["candidate_id"].map(registry_pair["step2_instability_flag"]).fillna(False).astype(float) * 0.15
        + pair_base["interpretation_bin"].fillna("stable").eq("unstable_or_source_sensitive").astype(float) * 0.15
    )
    pair_base["step3_pair_score"] = (
        0.25 * pair_base["model_rank_component"]
        + 0.25 * pair_base["stability_component"]
        + 0.15 * pair_base["donor_trajectory_component"]
        + 0.15 * pair_base["task_specificity_component"]
        + 0.10 * pair_base["receiver_specificity_component"]
        + 0.10 * pair_base["state_support_component"]
        - penalties
    ).clip(lower=0.0, upper=1.0)
    pair_base = pair_base.merge(pair_counts, left_on="candidate_id", right_index=True, how="left")
    pair_base["primary_task"] = pair_base["candidate_id"].map(task_pair["primary_task"])
    pair_base["directionality_check_status"] = pair_base["candidate_id"].map(confound_pair["directionality_status"])
    pair_base["promotion_status"] = pair_base["candidate_id"].map(confound_pair["promotion_status"]).fillna("exploratory")
    pair_base.to_csv(output_dir / "step3_pair_support_table.csv", index=False)

    anchor_pairs = top_program_anchor_pairs(refined_map, ranked_pairs, top_n=5).merge(
        pair_base[
            [
                "pair_key",
                "candidate_id",
                "step3_pair_score",
                "primary_task",
                "registry_gate",
                "fold_stability_value",
                "source_stability_value",
                "graph_units_supported",
                "has_step2_rank_support",
                "interpretation_bin",
            ]
        ],
        on="pair_key",
        how="left",
    )
    state_program = state_support.loc[(state_support["candidate_level"].eq("program")) & (state_support["state_pair_rank"].eq(1))].set_index("candidate_id")
    traj_program = trajectory.loc[trajectory["candidate_level"].eq("program")].set_index("candidate_id")
    confound_program = confounds.loc[confounds["candidate_level"].eq("program")].set_index("candidate_id")
    task_program = task_support.loc[task_support["candidate_level"].eq("program")].set_index("candidate_id")
    program_rows: list[dict[str, object]] = []
    for program_id, frame in anchor_pairs.groupby("refined_program_id", observed=False):
        candidate_id = program_candidate_id(program_id)
        pair_scores_local = frame["step3_pair_score"].dropna()
        if pair_scores_local.empty:
            continue
        primary_tasks = frame["primary_task"].dropna().astype(str)
        dominant_task = primary_tasks.mode().iloc[0] if not primary_tasks.empty else task_program.get("primary_task", {}).get(candidate_id, "unknown")
        coherence = float((primary_tasks == dominant_task).mean()) if not primary_tasks.empty else 0.0
        anchor_observed_fraction = float(frame["has_step2_rank_support"].fillna(False).mean()) if "has_step2_rank_support" in frame.columns else 0.0
        observed_frame = frame.loc[frame["has_step2_rank_support"].fillna(False)].copy() if "has_step2_rank_support" in frame.columns else frame.copy()
        observed_fold = observed_frame["fold_stability_value"].dropna()
        if observed_fold.empty:
            anchor_stability_fraction = 0.5
            anchor_min_fold_stability = 0.5
        else:
            anchor_stability_fraction = float(observed_fold.ge(0.67).mean())
            anchor_min_fold_stability = float(observed_fold.min())
        anchor_registry_pass_fraction = float(observed_frame["registry_gate"].eq("promote").mean()) if not observed_frame.empty else 0.0
        anchor_registry_review_fraction = float(observed_frame["registry_gate"].isin(["promote", "review"]).mean()) if not observed_frame.empty else 0.5
        anchor_hold_fraction = float(observed_frame["registry_gate"].eq("hold").mean()) if not observed_frame.empty else 0.0
        anchor_instability_fraction = (
            float(observed_frame["interpretation_bin"].fillna("stable").eq("unstable_or_source_sensitive").mean()) if not observed_frame.empty else 0.0
        )
        consensus_row = program_consensus.loc[program_id] if program_id in program_consensus.index else pd.Series(dtype=float)
        consensus_top10_fraction = float(consensus_row.get("consensus_top10_fraction", 0.0))
        consensus_medium_fraction = float(consensus_row.get("consensus_medium_priority_fraction", 0.0))
        consensus_external_fraction = float(consensus_row.get("consensus_external_support_fraction", 0.0))
        stability_bonus = (
            0.30 * anchor_stability_fraction
            + 0.30 * anchor_registry_pass_fraction
            + 0.20 * anchor_min_fold_stability
            + 0.20 * anchor_observed_fraction
        )
        validation_readiness = 1.0 if confound_program.get("promotion_status", {}).get(candidate_id, "exploratory") == "validation_ready" else 0.6
        donor_coherence = 1.0 if traj_program.get("trajectory_class", {}).get(candidate_id, "uncertain") != "uncertain" else 0.4
        program_stability_gate = "promote"
        if (not observed_frame.empty) and (anchor_hold_fraction >= 0.5 or anchor_min_fold_stability < 0.45 or anchor_instability_fraction >= 0.4):
            program_stability_gate = "hold"
        elif anchor_observed_fraction < 0.4 or anchor_registry_pass_fraction < 0.35 or anchor_stability_fraction < 0.55:
            program_stability_gate = "review"
        caveat_summary = confound_program.get("caveat_summary", {}).get(candidate_id, "clear")
        caveat_flags = {flag for flag in str(caveat_summary).split("|") if flag and flag != "clear"}
        consensus_rescue = (
            dominant_task == PRIMARY_TASK
            and consensus_top10_fraction >= 0.85
            and consensus_external_fraction > 0.0
            and anchor_observed_fraction >= 0.5
            and anchor_min_fold_stability >= 0.30
            and not caveat_flags.intersection({"negative_holdout", "source_flip", "region_flip"})
        )
        if consensus_rescue and program_stability_gate == "hold":
            program_stability_gate = "review"
        if consensus_rescue:
            validation_readiness = max(validation_readiness, 0.8)
        consensus_bonus = (
            0.05 * consensus_top10_fraction
            + 0.03 * consensus_medium_fraction
            + 0.07 * consensus_external_fraction
        )
        program_score = (
            0.35 * float(pair_scores_local.mean())
            + 0.20 * float(pair_scores_local.max())
            + 0.15 * coherence
            + 0.15 * donor_coherence
            + 0.10 * validation_readiness
            + 0.05 * stability_bonus
            + consensus_bonus
        )
        tier = confound_program.get("promotion_status", {}).get(candidate_id, "exploratory")
        if program_stability_gate == "hold":
            tier = "hold_back"
        elif program_stability_gate == "review" and tier == "validation_ready":
            tier = "strong_narrative"
        if tier == "validation_ready" and program_score < 0.60:
            tier = "strong_narrative"
        if tier == "strong_narrative" and anchor_observed_fraction >= 0.6 and anchor_registry_review_fraction < 0.5:
            tier = "hold_back"
        if consensus_rescue and tier == "hold_back":
            tier = "strong_narrative"
        if dominant_task != PRIMARY_TASK or "sex_fail" in caveat_flags or program_stability_gate == "hold":
            step4_priority = "low"
        elif tier == "validation_ready" and program_stability_gate == "promote" and not caveat_flags.intersection({"negative_holdout", "source_flip", "region_flip"}):
            step4_priority = "high"
        elif tier in {"validation_ready", "strong_narrative"} and program_stability_gate in {"promote", "review"}:
            step4_priority = "medium"
        else:
            step4_priority = "low"
        if consensus_rescue and step4_priority == "low":
            step4_priority = "medium"
        state_row = state_program.loc[candidate_id] if candidate_id in state_program.index else None
        program_rows.append(
            {
                "refined_program_id": program_id,
                "candidate_id": candidate_id,
                "refined_program_name": program_name(program_id),
                "step3_program_score": program_score,
                "anchor_pair_score_mean": float(pair_scores_local.mean()),
                "anchor_pair_score_max": float(pair_scores_local.max()),
                "program_coherence": coherence,
                "anchor_stability_fraction": anchor_stability_fraction,
                "anchor_observed_fraction": anchor_observed_fraction,
                "anchor_registry_pass_fraction": anchor_registry_pass_fraction,
                "anchor_registry_review_fraction": anchor_registry_review_fraction,
                "anchor_min_fold_stability": anchor_min_fold_stability,
                "program_stability_gate": program_stability_gate,
                "donor_trajectory_coherence": donor_coherence,
                "validation_readiness": validation_readiness,
                "dominant_task": dominant_task,
                "anchor_pairs": "|".join(frame.sort_values("step3_pair_score", ascending=False)["pair_key"].astype(str).tolist()),
                "dominant_sender_state": state_row["sender_state_id"] if state_row is not None else "NA",
                "dominant_receiver_state": state_row["receiver_state_id"] if state_row is not None else "NA",
                "interpretation_class": traj_program.get("trajectory_class", {}).get(candidate_id, "uncertain"),
                "caveat_summary": caveat_summary,
                "consensus_run_count": int(consensus_row.get("consensus_run_count", 0) or 0),
                "consensus_top10_fraction": consensus_top10_fraction,
                "consensus_medium_priority_fraction": consensus_medium_fraction,
                "consensus_external_support_fraction": consensus_external_fraction,
                "consensus_rescue_applied": bool(consensus_rescue),
                "final_step3_tier": tier,
                "step4_validation_priority": step4_priority,
            }
        )
    program_table = pd.DataFrame(program_rows).sort_values("step3_program_score", ascending=False)
    top10 = program_table.head(10).copy()
    top10.to_csv(output_dir / "step3_top10_refined_programs.csv", index=False)
    program_table.to_csv(output_dir / "step3_program_appendix.csv", index=False)

    pair_tier_map = pair_base.set_index("candidate_id")["registry_gate"].map({"promote": None, "review": "strong_narrative", "hold": "hold_back"})
    base_pair_tier = pd.Series(
        pair_base["candidate_id"].map(confound_pair["promotion_status"]).fillna("hold_back").to_numpy(),
        index=pair_base["candidate_id"],
    )
    adjusted_pair_tier = base_pair_tier.where(pair_tier_map.isna(), pair_tier_map)
    registry["final_step3_tier"] = registry["candidate_key"].map(adjusted_pair_tier).fillna(registry["final_step3_tier"])
    registry["directionality_check_status"] = registry["candidate_key"].map(confound_pair["directionality_status"]).fillna(
        registry["directionality_check_status"]
    )
    program_registry = pd.DataFrame(
        [
            {
                "entity_level": "program",
                "candidate_type": "refined_program",
                "candidate_key": row["candidate_id"],
                "candidate_id": row["refined_program_id"],
                "candidate_label": row["refined_program_name"],
                "raw_pathway_family": "|".join(
                    sorted(dict.fromkeys(refined_map.loc[refined_map["refined_program_id"].eq(row["refined_program_id"]), "raw_pathway_family"].astype(str)))
                ),
                "ligand_gene_string": pd.NA,
                "receptor_gene_string": pd.NA,
                "ranking_score": row["step3_program_score"],
                "aging_score": pd.NA,
                "pd_score": pd.NA,
                "delta_score": pd.NA,
                "receiver_specificity": pd.NA,
                "holdout_support": pd.NA,
                "fold_stability": row["anchor_min_fold_stability"],
                "source_stability": pd.NA,
                "source_coverage_ratio": pd.NA,
                "interpretation_bin": row["interpretation_class"],
                "graph_support_units": pd.NA,
                "step4_shortlist_flag": row["final_step3_tier"] in {"validation_ready", "strong_narrative"},
                "sex_restriction_flag": "Y" in row["anchor_pairs"],
                "source_sensitivity_pending_flag": False,
                "annotation_ambiguity_flag": row["refined_program_id"] == "exploratory_residual_program",
                "holdout_warning_flag": False,
                "negative_holdout_flag": "negative_holdout" in str(row["caveat_summary"]).split("|"),
                "low_fold_stability_flag": row["anchor_min_fold_stability"] < 0.45,
                "low_source_stability_flag": False,
                "limited_graph_support_flag": False,
                "step2_instability_flag": row["program_stability_gate"] == "hold",
                "registry_gate": row["program_stability_gate"],
                "stability_bin": {"promote": "stable", "review": "watch", "hold": "unstable"}[row["program_stability_gate"]],
                "provisional_n_graphs_supported": pd.NA,
                "provisional_n_graph_units_supported": pd.NA,
                "provisional_n_sources_supported": pd.NA,
                "provisional_primary_task": row["dominant_task"],
                "refined_program_id": row["refined_program_id"],
                "refined_program_name": row["refined_program_name"],
                "anchor_pair_ids": row["anchor_pairs"],
                "directionality_check_status": confound_program.get("directionality_status", {}).get(row["candidate_id"], "NA"),
                "final_step3_tier": row["final_step3_tier"],
            }
            for row in program_table.to_dict(orient="records")
        ]
    )
    registry = pd.concat([registry, program_registry], ignore_index=True, sort=False)
    registry.to_csv(registry_path, index=False)
    write_markdown(
        project_path("data/processed/step3/01_registry/step3_candidate_registry.md"),
        "\n".join(
            [
                "# Step 3 Candidate Registry",
                "",
                "## Final Top Programs",
                markdown_table(top10.fillna("NA")),
            ]
        ),
    )


if __name__ == "__main__":
    main()
