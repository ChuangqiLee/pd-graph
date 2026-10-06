#!/usr/bin/env python3.12
"""Build the Step 4 validation registry."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import markdown_table, project_path, write_markdown
from scripts.step4.common import (
    PROGRAM_BATCH,
    PROGRAM_NOTES,
    SHARED_GENE_CLUSTER,
    ensure_step4_dir,
    read_json,
)


DEFAULT_INPUT = Path("data/processed/step4/00_inputs/step4_handoff_normalized.json")
DEFAULT_OUTPUT_DIR = Path("data/processed/step4/01_registry")
REGISTRY_COLUMNS = [
    "program_id",
    "program_name",
    "program_batch",
    "anchor_pairs_json",
    "gene_members_json",
    "sender_state_id",
    "receiver_state_id",
    "expected_pd_direction",
    "anchor_semantics_profile",
    "caveat_flags_json",
    "gene_count",
    "anchor_pair_count",
    "shared_gene_cluster_id",
    "requires_counts_adjustment",
    "requires_source_sensitivity",
    "requires_region_sensitivity",
    "requires_strict_replication",
    "blocking_issue_flag",
    "dscam_resolution_status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Step 4 validation registry.")
    parser.add_argument("--input-json", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def summarize_anchor_semantics(anchors: list[dict[str, object]]) -> str:
    counter = Counter(f"{anchor['anchor_class']}:{anchor['claim_language_level']}" for anchor in anchors)
    return "; ".join(f"{key}={value}" for key, value in sorted(counter.items()))


def main() -> None:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    ensure_step4_dir("01_registry")
    handoff = read_json(args.input_json)

    rows: list[dict[str, object]] = []
    for record in handoff:
        caveats = list(record["caveat_flag_list"])
        rows.append(
            {
                "program_id": record["program_id"],
                "program_name": record["program_name"],
                "program_batch": PROGRAM_BATCH[record["program_id"]],
                "anchor_pairs_json": json.dumps(record["anchor_pairs_json"], ensure_ascii=False),
                "gene_members_json": json.dumps(record["gene_set_members_list"], ensure_ascii=False),
                "sender_state_id": record["sender_state_id"],
                "receiver_state_id": record["receiver_state_id"],
                "expected_pd_direction": record["expected_pd_direction"],
                "anchor_semantics_profile": summarize_anchor_semantics(record["anchor_pairs_json"]),
                "caveat_flags_json": json.dumps(caveats, ensure_ascii=False),
                "gene_count": len(record["gene_set_members_list"]),
                "anchor_pair_count": len(record["anchor_pairs_json"]),
                "shared_gene_cluster_id": SHARED_GENE_CLUSTER[record["program_id"]],
                "requires_counts_adjustment": "cell_count_link" in caveats,
                "requires_source_sensitivity": "source_flip" in caveats or "source_dominant" in caveats,
                "requires_region_sensitivity": "region_flip" in caveats,
                "requires_strict_replication": "negative_holdout" in caveats or PROGRAM_BATCH[record["program_id"]] == "batch_2_strict",
                "blocking_issue_flag": str(record["dscam_resolution_status"]).startswith("unresolved"),
                "dscam_resolution_status": record["dscam_resolution_status"],
                "notes": PROGRAM_NOTES[record["program_id"]] + (f" {record['notes']}".strip() if record.get("notes") else ""),
            }
        )
    registry = pd.DataFrame(rows, columns=REGISTRY_COLUMNS)
    if not registry.empty:
        registry = registry.sort_values(["program_batch", "program_name"]).reset_index(drop=True)
    registry.to_csv(output_dir / "step4_validation_registry.csv", index=False)

    md_table = registry[
        [
            "program_id",
            "program_batch",
            "gene_count",
            "anchor_pair_count",
            "sender_state_id",
            "receiver_state_id",
            "expected_pd_direction",
            "shared_gene_cluster_id",
            "blocking_issue_flag",
        ]
    ].copy()
    lines = [
        "# Step 4 Validation Registry",
        "",
        "One row per refined program, carrying the normalized anchor list, the expected state pair, and the caveat-derived handling flags used by all later Step 4 analyses.",
        "",
        markdown_table(md_table.fillna("NA")) if not md_table.empty else "_No Step 4 programs were exported from Step 3._",
        "",
    ]
    write_markdown(output_dir / "step4_validation_registry.md", "\n".join(lines))


if __name__ == "__main__":
    main()
