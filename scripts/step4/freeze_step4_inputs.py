#!/usr/bin/env python3.12
"""Freeze Step 4 inputs, rebuild missing Step 3 exports, and normalize the handoff."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import markdown_table, project_path, write_markdown
from scripts.step3.common import stage_file
from scripts.step4.common import (
    PROGRAM_BATCH,
    STEP3_EXEMPLARS,
    STEP3_HANDOFF_DIR,
    STEP3_PROGRAM_MAP,
    STEP3_STATE_DIR,
    STEP3_STATE_SUPPORT,
    STEP3_TASK_SUPPORT,
    STEP3_TOP10,
    STEP4_INPUTS,
    STEP4_NORMALIZED_HANDOFF_KEYS,
    anchor_count_from_raw,
    bh_fdr,
    build_anchor_records,
    ensure_step4_dir,
    expected_spatial_task,
    file_record,
    json_dumps,
    lineage_from_state_id,
    list_from_pipe,
    load_step4_program_frame,
    load_top_anchor_pairs,
    log,
    read_csv_required_columns,
    resolve_dscam_status,
    safe_log2fc,
    require_record_list,
)


STATE_FILES = {
    "astrocyte": STEP3_STATE_DIR / "step3_astrocyte_states.h5ad",
    "dopaminergic neuron": STEP3_STATE_DIR / "step3_dopaminergic_states.h5ad",
    "endothelial cell": STEP3_STATE_DIR / "step3_endothelial_states.h5ad",
    "microglia": STEP3_STATE_DIR / "step3_microglia_states.h5ad",
    "neuron": STEP3_STATE_DIR / "step3_neuron_states.h5ad",
}

DEFAULT_OUTPUT_DIR = Path("data/processed/step4/00_inputs")
STATE_SUMMARY_COLUMNS = [
    "program_id",
    "program_name",
    "sender_state_id",
    "receiver_state_id",
    "sender_state_lineage",
    "receiver_state_lineage",
    "sender_state_cell_count",
    "receiver_state_cell_count",
    "sender_state_frequency",
    "receiver_state_frequency",
    "dominant_task",
    "expected_spatial_task",
    "task_conflict_status",
    "program_batch",
    "dscam_resolution_status",
]
ANCHOR_RECORD_COLUMNS = [
    "program_id",
    "program_name",
    "anchor_order",
    "pair_key",
    "sender_genes",
    "receiver_genes",
    "anchor_class",
    "claim_language_level",
    "rationale",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze and normalize Step 4 inputs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def stage_core_inputs(output_dir: Path) -> pd.DataFrame:
    stage_rows: list[dict[str, object]] = []
    sources = [
        ("idea.md", "docs/idea.md", "project_outline", "project"),
        ("step2_summary_dossier.md", "docs/step2_summary_dossier.md", "step2_summary", "step2"),
        ("step3_to_step4_priority_summary.md", "docs/step3_to_step4_priority_summary.md", "step3_priority_summary", "step3"),
        (STEP3_HANDOFF_DIR / "step3_to_step4_priority_summary.md", "step3_handoff/step3_to_step4_priority_summary.md", "step3_handoff", "step3"),
        (STEP3_HANDOFF_DIR / "step3_to_step4_gene_set_targets.csv", "step3_handoff/step3_to_step4_gene_set_targets.csv", "step3_handoff", "step3"),
        (STEP3_HANDOFF_DIR / "step3_to_step4_spatial_targets.csv", "step3_handoff/step3_to_step4_spatial_targets.csv", "step3_handoff", "step3"),
        (STEP3_TASK_SUPPORT, "step3_handoff/step3_task_support_matrix.csv", "step3_task_support", "step3"),
        (STEP3_STATE_SUPPORT, "step3_handoff/step3_candidate_state_support.csv", "step3_state_support", "step3"),
        (STEP3_TOP10, "step3_handoff/step3_top10_refined_programs.csv", "step3_top10", "step3"),
        (STEP3_PROGRAM_MAP, "step3_handoff/step3_refined_program_map.csv", "step3_refined_map", "step3"),
    ]
    if STEP3_EXEMPLARS.exists():
        sources.append((STEP3_EXEMPLARS, "step3_handoff/step3_exemplar_edges.csv", "step3_exemplars", "step3"))
    for lineage, path in STATE_FILES.items():
        sources.append((path, f"step3_states/{path.name}", f"step3_{lineage}_states", "step3"))
    for source, rel_dest, input_class, source_phase in sources:
        staged = stage_file(source, output_dir / rel_dest, mode="hardlink_or_copy")
        staged["input_class"] = input_class
        staged["source_phase"] = source_phase
        stage_rows.append(staged)
    return pd.DataFrame(stage_rows)


def build_state_membership() -> tuple[pd.DataFrame, dict[str, int]]:
    import anndata as ad

    rows: list[pd.DataFrame] = []
    state_sizes: dict[str, int] = {}
    for lineage, path in STATE_FILES.items():
        adata = ad.read_h5ad(path)
        keep_cols = [
            "cell_id",
            "cell_type_standardized_final",
            "state_id",
            "state_label",
            "state_cluster",
            "donor_id",
            "source_id",
            "disease_group",
            "age_years",
            "sex_standardized",
            "brain_region_standardized",
        ]
        local = adata.obs[[column for column in keep_cols if column in adata.obs.columns]].copy()
        local = local.rename(columns={"cell_type_standardized_final": "cell_type"})
        local["lineage"] = lineage
        local["state_cell_count"] = local.groupby("state_id", observed=False)["cell_id"].transform("count")
        rows.append(local.reset_index(drop=True))
        state_sizes.update(local["state_id"].value_counts().astype(int).to_dict())
    membership = pd.concat(rows, ignore_index=True)
    membership = membership.drop_duplicates(subset=["cell_id"]).sort_values(["lineage", "state_id", "cell_id"]).reset_index(drop=True)
    return membership, state_sizes


def compute_marker_rankings() -> pd.DataFrame:
    import anndata as ad

    ranking_rows: list[dict[str, object]] = []
    for lineage, path in STATE_FILES.items():
        log(f"[step4.freeze] marker ranking for {lineage}")
        adata = ad.read_h5ad(path)
        matrix = np.asarray(adata.X, dtype=np.float32)
        genes = adata.var_names.astype(str).tolist()
        obs = adata.obs.copy()
        for state_id, index in obs.groupby("state_id", observed=False).groups.items():
            mask = obs.index.isin(index)
            state_values = matrix[mask]
            other_values = matrix[~mask]
            pvals: list[float] = []
            local_rows: list[dict[str, object]] = []
            for gene_index, gene in enumerate(genes):
                in_vec = state_values[:, gene_index]
                out_vec = other_values[:, gene_index]
                mean_in = float(np.mean(in_vec))
                mean_out = float(np.mean(out_vec)) if len(out_vec) else 0.0
                detect_in = float(np.mean(in_vec > 0.0))
                detect_out = float(np.mean(out_vec > 0.0)) if len(out_vec) else 0.0
                try:
                    stat, p_value = mannwhitneyu(in_vec, out_vec, alternative="two-sided")
                    auc = float(stat) / float(len(in_vec) * max(len(out_vec), 1))
                except ValueError:
                    p_value = 1.0
                    auc = 0.5
                pvals.append(float(p_value))
                local_rows.append(
                    {
                        "lineage": lineage,
                        "state_id": str(state_id),
                        "state_label": str(obs.loc[index[0], "state_label"]),
                        "gene_symbol": gene,
                        "mean_in_state": mean_in,
                        "mean_out_state": mean_out,
                        "log2_fc": safe_log2fc(mean_in, mean_out),
                        "auc": auc,
                        "p_value": float(p_value),
                        "detection_fraction_in": detect_in,
                        "detection_fraction_out": detect_out,
                        "n_in": int(len(in_vec)),
                        "n_out": int(len(out_vec)),
                    }
                )
            fdr = bh_fdr(pvals)
            for row_dict, p_adj in zip(local_rows, fdr):
                row_dict["p_adj"] = float(p_adj)
            ranked = pd.DataFrame(local_rows).sort_values(
                ["state_id", "log2_fc", "auc", "detection_fraction_in"],
                ascending=[True, False, False, False],
            )
            ranked["state_rank"] = ranked.groupby("state_id", observed=False).cumcount() + 1
            ranking_rows.extend(ranked.to_dict(orient="records"))
    return pd.DataFrame(ranking_rows)


def build_program_state_summary(membership: pd.DataFrame, dscam_status: str) -> pd.DataFrame:
    merged = load_step4_program_frame()
    if merged.empty:
        return pd.DataFrame(columns=STATE_SUMMARY_COLUMNS)
    counts = membership["state_id"].value_counts().astype(int).to_dict()
    lineage_counts = membership["lineage"].value_counts().astype(int).to_dict()
    task_support = pd.read_csv(STEP3_TASK_SUPPORT).set_index("candidate_id")
    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        sender_lineage = lineage_from_state_id(str(row.sender_state_id))
        receiver_lineage = lineage_from_state_id(str(row.receiver_state_id))
        sender_n = int(counts.get(str(row.sender_state_id), 0))
        receiver_n = int(counts.get(str(row.receiver_state_id), 0))
        sender_total = max(int(lineage_counts.get(sender_lineage, 0)), 1)
        receiver_total = max(int(lineage_counts.get(receiver_lineage, 0)), 1)
        expected_task = expected_spatial_task(str(row.expected_spatial_relationship))
        candidate_key = str(row.candidate_id)
        task_row = task_support.loc[candidate_key] if candidate_key in task_support.index else pd.Series(dtype=object)
        dominant_task = str(task_row.get("primary_task", row.dominant_task))
        task_conflict_status = "aligned" if dominant_task == expected_task else "task_label_mismatch"
        rows.append(
            {
                "program_id": row.refined_program_id,
                "program_name": row.refined_program_name,
                "sender_state_id": row.sender_state_id,
                "receiver_state_id": row.receiver_state_id,
                "sender_state_lineage": sender_lineage,
                "receiver_state_lineage": receiver_lineage,
                "sender_state_cell_count": sender_n,
                "receiver_state_cell_count": receiver_n,
                "sender_state_frequency": sender_n / sender_total,
                "receiver_state_frequency": receiver_n / receiver_total,
                "dominant_task": dominant_task,
                "expected_spatial_task": expected_task,
                "task_conflict_status": task_conflict_status,
                "program_batch": PROGRAM_BATCH[row.refined_program_id],
                "dscam_resolution_status": dscam_status if row.refined_program_id == "dscam_dcc_contact_recognition" else "not_applicable",
            }
        )
    return pd.DataFrame(rows, columns=STATE_SUMMARY_COLUMNS)


def build_normalized_handoff(anchor_records: pd.DataFrame, state_summary: pd.DataFrame, dscam_status: str, dscam_note: str) -> list[dict[str, object]]:
    merged = load_step4_program_frame()
    state_summary = state_summary.set_index("program_id")
    normalized: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        local_anchors = anchor_records.loc[anchor_records["program_id"].eq(row.refined_program_id)].copy()
        anchor_json = []
        for anchor in local_anchors.itertuples(index=False):
            anchor_json.append(
                {
                    "pair_key": anchor.pair_key,
                    "sender_genes": list(anchor.sender_genes),
                    "receiver_genes": list(anchor.receiver_genes),
                    "anchor_class": anchor.anchor_class,
                    "claim_language_level": anchor.claim_language_level,
                    "rationale": anchor.rationale,
                }
            )
        summary = state_summary.loc[row.refined_program_id]
        record = {
            "program_id": row.refined_program_id,
            "program_name": row.refined_program_name,
            "candidate_id": row.candidate_id,
            "program_batch": PROGRAM_BATCH[row.refined_program_id],
            "gene_set_members_list": list_from_pipe(row.gene_set_members),
            "sender_state_id": row.sender_state_id,
            "receiver_state_id": row.receiver_state_id,
            "sender_state_marker_list": list_from_pipe(row.sender_state_markers_gene),
            "receiver_state_marker_list": list_from_pipe(row.receiver_state_markers_gene),
            "caveat_flag_list": list_from_pipe(row.caveat_flags),
            "expected_pd_direction": row.expected_disease_direction,
            "expected_internal_direction": getattr(row, "expected_internal_direction", row.expected_disease_direction),
            "expected_spatial_relationship": row.expected_spatial_relationship,
            "expected_spatial_task": expected_spatial_task(row.expected_spatial_relationship),
            "dominant_task": summary["dominant_task"],
            "validation_tier": row.validation_tier,
            "anchor_pairs_json": anchor_json,
            "sender_state_frequency": float(summary["sender_state_frequency"]),
            "receiver_state_frequency": float(summary["receiver_state_frequency"]),
            "task_conflict_status": summary["task_conflict_status"],
            "dscam_resolution_status": dscam_status if row.refined_program_id == "dscam_dcc_contact_recognition" else "not_applicable",
            "notes": dscam_note if row.refined_program_id == "dscam_dcc_contact_recognition" else "",
        }
        normalized.append(record)
    return normalized


def write_blocking_report(output_dir: Path, state_summary: pd.DataFrame, dscam_status: str, dscam_note: str, anchor_records: pd.DataFrame) -> None:
    program_frame = load_step4_program_frame()
    caveat_summary = program_frame[["refined_program_id", "refined_program_name", "caveat_flags"]].copy() if not program_frame.empty else pd.DataFrame(
        columns=["refined_program_id", "refined_program_name", "caveat_flags"]
    )
    caveat_summary["caveat_flags"] = caveat_summary["caveat_flags"].fillna("")
    if anchor_records.empty:
        semantics_summary = pd.DataFrame(columns=["anchor_class", "claim_language_level", "n_anchors"])
    else:
        semantics_summary = (
            anchor_records.groupby(["anchor_class", "claim_language_level"], observed=False)["pair_key"]
            .count()
            .rename("n_anchors")
            .reset_index()
            .sort_values(["claim_language_level", "anchor_class"])
        )
    conflict_columns = ["program_id", "dominant_task", "expected_spatial_task", "task_conflict_status", "dscam_resolution_status"]
    conflict_summary = state_summary[conflict_columns].copy() if not state_summary.empty else pd.DataFrame(columns=conflict_columns)
    lines = [
        "# Step 4 Blocking Issues",
        "",
        "## DSCAM and DCC task-axis resolution",
        f"- Resolution status: `{dscam_status}`",
        f"- Note: {dscam_note}",
        "",
        "## Task conflict summary",
        markdown_table(conflict_summary.fillna("NA")),
        "",
        "## Anchor semantics summary",
        markdown_table(semantics_summary.fillna("NA")),
        "",
        "## Program caveat summary",
        markdown_table(caveat_summary.fillna("NA")),
        "",
    ]
    write_markdown(output_dir / "step4_blocking_issues.md", "\n".join(lines))


def main() -> None:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    ensure_step4_dir("00_inputs")
    ensure_step4_dir("00_inputs", "docs")
    ensure_step4_dir("00_inputs", "step3_handoff")
    ensure_step4_dir("00_inputs", "step3_states")

    membership_path = output_dir / "step3_state_cell_membership.parquet"
    ranking_path = output_dir / "step3_state_marker_rankings.csv"
    staged_inputs = stage_core_inputs(output_dir)
    read_csv_required_columns(
        STEP3_TASK_SUPPORT,
        (
            "candidate_id",
            "primary_task",
            "microglia_to_dopaminergic_neuron",
            "endothelial_to_dopaminergic_neuron",
        ),
        context="Step 3 task support matrix",
    )
    read_csv_required_columns(
        STEP3_STATE_SUPPORT,
        (
            "candidate_id",
            "state_pair_rank",
            "sender_state_id",
            "receiver_state_id",
        ),
        context="Step 3 candidate state support matrix",
    )
    if STEP3_EXEMPLARS.exists():
        read_csv_required_columns(
            STEP3_EXEMPLARS,
            (
                "refined_program_id",
                "task_name",
            ),
            context="Step 3 exemplar edges",
        )
    try:
        membership, _ = build_state_membership()
        membership.to_parquet(membership_path, index=False)
        ranking = compute_marker_rankings()
        ranking.to_csv(ranking_path, index=False)
    except ModuleNotFoundError:
        if not membership_path.exists() or not ranking_path.exists():
            raise
        log("[step4.freeze] anndata unavailable; reusing cached state membership and marker rankings")
        membership = pd.read_parquet(membership_path)
        ranking = pd.read_csv(ranking_path)

    dscam_status, dscam_note = resolve_dscam_status()
    state_summary = build_program_state_summary(membership, dscam_status)
    state_summary_path = output_dir / "step3_program_state_summary.csv"
    state_summary.to_csv(state_summary_path, index=False)

    anchor_records = pd.DataFrame(build_anchor_records(), columns=ANCHOR_RECORD_COLUMNS)
    anchor_records["sender_genes_json"] = anchor_records["sender_genes"].map(json_dumps)
    anchor_records["receiver_genes_json"] = anchor_records["receiver_genes"].map(json_dumps)
    anchor_semantics = anchor_records[
        [
            "program_id",
            "program_name",
            "anchor_order",
            "pair_key",
            "sender_genes_json",
            "receiver_genes_json",
            "anchor_class",
            "claim_language_level",
            "rationale",
        ]
    ].copy()
    anchor_semantics.to_csv(output_dir / "step4_anchor_semantics.csv", index=False)

    normalized_anchor_json = []
    for program_id, frame in anchor_records.groupby("program_id", observed=False):
        normalized_anchor_json.append(
            {
                "program_id": program_id,
                "anchors": [
                    {
                        "pair_key": row.pair_key,
                        "sender_genes": list(row.sender_genes),
                        "receiver_genes": list(row.receiver_genes),
                        "anchor_class": row.anchor_class,
                        "claim_language_level": row.claim_language_level,
                    }
                    for row in frame.itertuples(index=False)
                ],
            }
        )
    (output_dir / "step3_program_anchor_pairs_normalized.json").write_text(
        json_dumps(normalized_anchor_json) + "\n",
        encoding="utf-8",
    )

    normalized_handoff = build_normalized_handoff(anchor_records, state_summary, dscam_status, dscam_note)
    require_record_list(normalized_handoff, STEP4_NORMALIZED_HANDOFF_KEYS, context="Normalized Step 4 handoff")
    (output_dir / "step4_handoff_normalized.json").write_text(
        json_dumps(normalized_handoff) + "\n",
        encoding="utf-8",
    )

    write_blocking_report(output_dir, state_summary, dscam_status, dscam_note, anchor_records)

    manifest_rows = staged_inputs.to_dict(orient="records")
    frozen_outputs = [
        membership_path,
        ranking_path,
        state_summary_path,
        output_dir / "step4_anchor_semantics.csv",
        output_dir / "step3_program_anchor_pairs_normalized.json",
        output_dir / "step4_handoff_normalized.json",
        output_dir / "step4_blocking_issues.md",
    ]
    for path in frozen_outputs:
        manifest_rows.append(file_record(path, source_phase="step4_phase0", input_class="generated"))
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "step4_input_manifest.csv", index=False)

    log("[step4.freeze] wrote normalized Step 4 starting layer")


if __name__ == "__main__":
    main()
