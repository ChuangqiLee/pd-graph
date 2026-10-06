#!/usr/bin/env python3.12
"""Run the secondary human SNpc replication analysis from the Shin et al. GeoMx release."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import markdown_table, write_markdown
from scripts.step4.common import ensure_step4_dir, read_json


DEFAULT_OUTPUT_DIR = Path("data/processed/step4/03_spatial/05_secondary_replication")
FIGSHARE_ARTICLE = "https://api.figshare.com/v2/articles/29323688"
TARGET_WORKBOOK = "PD_HC_total and quadrant.xlsx"
XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the secondary Step 4 spatial replication.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def download_workbook(output_dir: Path) -> Path:
    response = requests.get(FIGSHARE_ARTICLE, timeout=60)
    response.raise_for_status()
    article = response.json()
    file_meta = next(file for file in article["files"] if file["name"] == TARGET_WORKBOOK)
    path = output_dir / "PD_HC_total_and_quadrant.xlsx"
    if not path.exists():
        payload = requests.get(file_meta["download_url"], timeout=120)
        payload.raise_for_status()
        path.write_bytes(payload.content)
    return path


@lru_cache(maxsize=4)
def workbook_sheet_targets(path_str: str) -> tuple[tuple[str, str], ...]:
    with ZipFile(path_str) as workbook:
        workbook_xml = ET.fromstring(workbook.read("xl/workbook.xml"))
        rels_xml = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels_xml}
        return tuple(
            (
                sheet.attrib["name"],
                "xl/" + rel_map[sheet.attrib[f"{REL_NS}id"]].lstrip("/"),
            )
            for sheet in workbook_xml.find(f"{XLSX_NS}sheets")
        )


@lru_cache(maxsize=4)
def workbook_shared_strings(path_str: str) -> tuple[str, ...]:
    with ZipFile(path_str) as workbook:
        if "xl/sharedStrings.xml" not in workbook.namelist():
            return tuple()
        shared_xml = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for string_item in shared_xml.findall(f"{XLSX_NS}si"):
        texts = [text_node.text or "" for text_node in string_item.iterfind(f".//{XLSX_NS}t")]
        values.append("".join(texts))
    return tuple(values)


def load_sheet_rows(path: Path, sheet_name: str) -> list[dict[str, str]]:
    target_lookup = dict(workbook_sheet_targets(str(path)))
    shared = workbook_shared_strings(str(path))
    target = target_lookup[sheet_name]
    with ZipFile(path) as workbook:
        sheet_xml = ET.fromstring(workbook.read(target))
    rows: list[dict[str, str]] = []
    for row in sheet_xml.find(f"{XLSX_NS}sheetData").findall(f"{XLSX_NS}row"):
        row_values: dict[str, str] = {}
        for cell in row.findall(f"{XLSX_NS}c"):
            cell_ref = cell.attrib.get("r", "")
            col_ref = "".join(char for char in cell_ref if char.isalpha())
            value_node = cell.find(f"{XLSX_NS}v")
            if value_node is None:
                value = ""
            else:
                value = value_node.text or ""
                if cell.attrib.get("t") == "s":
                    value = shared[int(value)]
            row_values[col_ref] = value
        if row_values:
            rows.append(row_values)
    return rows


def standard_deg_table(path: Path, sheet_name: str) -> pd.DataFrame:
    rows = load_sheet_rows(path, sheet_name)
    records: list[dict[str, object]] = []
    for row in rows[1:]:
        gene_symbol = str(row.get("A", "")).strip()
        if not gene_symbol:
            continue
        records.append(
            {
                "sheet_name": sheet_name,
                "block_name": sheet_name,
                "gene_symbol": gene_symbol,
                "log2FoldChange": row.get("C", ""),
                "pvalue": row.get("F", ""),
                "padj": row.get("G", ""),
            }
        )
    return pd.DataFrame(records, columns=["sheet_name", "block_name", "gene_symbol", "log2FoldChange", "pvalue", "padj"])


def total_sheet_table(path: Path) -> pd.DataFrame:
    rows = load_sheet_rows(path, "Total")
    block_columns = {
        "total_down": ("A", "C", "F", "G"),
        "total_up": ("S", "U", "X", "Y"),
    }
    records: list[dict[str, object]] = []
    for row in rows[1:]:
        for block_name, (gene_col, logfc_col, pvalue_col, padj_col) in block_columns.items():
            gene_symbol = str(row.get(gene_col, "")).strip()
            if not gene_symbol:
                continue
            records.append(
                {
                    "sheet_name": "Total",
                    "block_name": block_name,
                    "gene_symbol": gene_symbol,
                    "log2FoldChange": row.get(logfc_col, ""),
                    "pvalue": row.get(pvalue_col, ""),
                    "padj": row.get(padj_col, ""),
                }
            )
    return pd.DataFrame(records, columns=["sheet_name", "block_name", "gene_symbol", "log2FoldChange", "pvalue", "padj"])


def program_score(frame: pd.DataFrame, genes: list[str]) -> tuple[float, int]:
    local = frame.loc[frame["gene_symbol"].isin(genes)].copy()
    if local.empty:
        return 0.0, 0
    local["signed_score"] = -pd.to_numeric(local["padj"], errors="coerce").clip(lower=1e-300).map(lambda value: np.log10(value)) * pd.to_numeric(local["log2FoldChange"], errors="coerce").map(lambda value: 1.0 if value > 0 else -1.0 if value < 0 else 0.0)
    return float(local["signed_score"].mean()), int(len(local))


def expected_direction_matches(expected_direction: str, score: float) -> bool:
    if "loss" in str(expected_direction):
        return float(score) < 0.0
    if "gain" in str(expected_direction):
        return float(score) > 0.0
    return False


def directionally_consistent_quadrants(frame: pd.DataFrame, expected_direction: str, *, min_abs_score: float = 0.15) -> int:
    if frame.empty:
        return 0
    mask = frame["program_score"].abs().ge(min_abs_score) & frame["direction_match"].astype(bool)
    return int(mask.sum())


def scoring_gene_set(record: dict[str, object], reference_frame: pd.DataFrame) -> tuple[list[str], str]:
    anchor_genes: list[str] = []
    for anchor in record.get("anchor_pairs_json", []):
        anchor_genes.extend(str(gene) for gene in anchor.get("sender_genes", []) if str(gene))
        anchor_genes.extend(str(gene) for gene in anchor.get("receiver_genes", []) if str(gene))
    anchor_genes = sorted(dict.fromkeys(anchor_genes))
    if anchor_genes:
        _, n_present = program_score(reference_frame, anchor_genes)
        if n_present > 0:
            return anchor_genes, "anchor_core"
        return list(record["gene_set_members_list"]), "program_full_fallback"
    return list(record["gene_set_members_list"]), "program_full"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_step4_dir("03_spatial", "05_secondary_replication")
    output_dir.mkdir(parents=True, exist_ok=True)

    workbook_path = download_workbook(output_dir)
    normalized = read_json(PROJECT_ROOT / "data/processed/step4/00_inputs/step4_handoff_normalized.json")
    normalized_lookup = {str(record["program_id"]): record for record in normalized}
    scores_path = output_dir / "geomx_roi_program_scores.csv"

    if not normalized:
        pd.DataFrame(
            columns=[
                "program_id",
                "program_name",
                "sheet_name",
                "block_name",
                "program_score",
                "n_genes_present",
                "direction_match",
                "expected_pd_direction",
                "scoring_gene_source",
            ]
        ).to_csv(scores_path, index=False)
        summary = pd.DataFrame(
            columns=[
                "program_id",
                "program_name",
                "scoring_gene_source",
                "total_program_score",
                "total_program_n_genes",
                "mean_quadrant_score",
                "n_quadrants_direction_consistent",
                "secondary_spatial_support_bin",
            ]
        )
        summary.to_csv(output_dir / "secondary_spatial_support_summary.csv", index=False)
        lines = [
            "# Secondary Spatial Replication Report",
            "",
            "No Step 4 programs were available for secondary spatial replication in this run.",
            "",
            "Score source for this run: `no_programs_available`.",
            "",
            markdown_table(summary.astype(object).fillna("NA")),
            "",
        ]
        write_markdown(output_dir / "secondary_spatial_report.md", "\n".join(lines))
        return

    try:
        tables = [total_sheet_table(workbook_path)]
        for sheet_name in ["Dorsolateral", "Dorsomedial", "Ventrolateral", "Ventromedial"]:
            tables.append(standard_deg_table(workbook_path, sheet_name))
        deg = pd.concat(tables, ignore_index=True)
        total_deg = deg.loc[deg["sheet_name"].eq("Total")].copy()

        score_rows: list[dict[str, object]] = []
        for record in normalized:
            program_genes, gene_source = scoring_gene_set(record, total_deg)
            for (sheet_name, block_name), frame in deg.groupby(["sheet_name", "block_name"], observed=False):
                score, n_present = program_score(frame, program_genes)
                score_rows.append(
                    {
                        "program_id": record["program_id"],
                        "program_name": record["program_name"],
                        "sheet_name": sheet_name,
                        "block_name": block_name,
                        "program_score": score,
                        "n_genes_present": n_present,
                        "direction_match": expected_direction_matches(record["expected_pd_direction"], score),
                        "expected_pd_direction": record["expected_pd_direction"],
                        "scoring_gene_source": gene_source,
                    }
                )
        scores = pd.DataFrame(score_rows)
        scores.to_csv(scores_path, index=False)
        cache_mode = "recomputed_from_workbook"
    except ImportError:
        if not scores_path.exists():
            raise
        scores = pd.read_csv(scores_path)
        scores = scores.loc[scores["program_id"].astype(str).isin(normalized_lookup)].copy()
        if "direction_match" not in scores.columns and "expected_pd_direction" in scores.columns:
            scores["direction_match"] = [
                expected_direction_matches(expected_direction, score)
                for expected_direction, score in zip(scores["expected_pd_direction"], scores["program_score"])
            ]
        if "scoring_gene_source" not in scores.columns:
            total_deg = deg.loc[deg["sheet_name"].eq("Total")].copy()
            source_lookup = {
                program_id: scoring_gene_set(record, total_deg)[1]
                for program_id, record in normalized_lookup.items()
            }
            scores["scoring_gene_source"] = scores["program_id"].astype(str).map(source_lookup).fillna("program_full")
        cache_mode = "reused_cached_scores"

    summary_rows: list[dict[str, object]] = []
    total_deg = deg.loc[deg["sheet_name"].eq("Total")].copy()
    for program_id, record in normalized_lookup.items():
        frame = scores.loc[scores["program_id"].astype(str).eq(program_id)].copy()
        program_genes, gene_source = scoring_gene_set(record, total_deg)
        total_score, total_n_present = program_score(deg.loc[deg["sheet_name"].eq("Total")].copy(), program_genes)
        quadrant_frame = frame.loc[frame["sheet_name"].isin(["Dorsolateral", "Dorsomedial", "Ventrolateral", "Ventromedial"])].copy()
        n_quadrants_direction_consistent = directionally_consistent_quadrants(
            quadrant_frame,
            str(record["expected_pd_direction"]),
        )
        mean_quadrant_score = float(quadrant_frame["program_score"].mean()) if not quadrant_frame.empty else 0.0
        total_direction_match = expected_direction_matches(str(record["expected_pd_direction"]), total_score)
        if total_direction_match and (
            (abs(total_score) >= 1.0 and n_quadrants_direction_consistent >= 1)
            or (abs(total_score) >= 0.50 and n_quadrants_direction_consistent >= 2)
        ):
            support_bin = "moderate_secondary_spatial_support"
        elif (total_direction_match and abs(total_score) >= 0.25) or n_quadrants_direction_consistent >= 1:
            support_bin = "weak_secondary_spatial_support"
        else:
            support_bin = "no_secondary_spatial_support"
        summary_rows.append(
            {
                "program_id": program_id,
                "program_name": str(record["program_name"]),
                "scoring_gene_source": gene_source,
                "total_program_score": total_score,
                "total_program_n_genes": total_n_present,
                "mean_quadrant_score": mean_quadrant_score,
                "n_quadrants_direction_consistent": n_quadrants_direction_consistent,
                "secondary_spatial_support_bin": support_bin,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "secondary_spatial_support_summary.csv", index=False)

    lines = [
        "# Secondary Spatial Replication Report",
        "",
        "The secondary replication layer used the published PD-versus-control GeoMx differential tables from the Shin et al. SNpc figshare release. The available public workbook provides processed DEG tables rather than raw ROI counts, so the replication statistic is a regional signed enrichment score rather than a donor-level mixed model.",
        "",
        "The total-sheet statistic is computed on the combined PD-vs-control DEG table for each program. This avoids canceling directionally consistent downregulated anchors against separately tabulated upregulated members when a program mixes loss- and stress-related genes.",
        "",
        "When explicit anchor pairs are available, the secondary replication score prioritizes the direct anchor-core genes. If the public GeoMx workbook does not cover any of those anchor-core genes, the score falls back to the broader state-extended program panel.",
        "",
        f"Score source for this run: `{cache_mode}`.",
        "",
        markdown_table(summary.astype(object).fillna("NA")),
        "",
    ]
    write_markdown(output_dir / "secondary_spatial_report.md", "\n".join(lines))


if __name__ == "__main__":
    main()
