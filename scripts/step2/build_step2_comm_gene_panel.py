#!/usr/bin/env python3
"""Build the Step 2 communication gene panel and mapping diagnostics."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import anndata as ad
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import (
    canonical_gene_symbol,
    iter_source_ids_from_atlas,
    load_yaml,
    markdown_table,
    project_path,
    resolve_source_snapshot_path,
    write_json,
    write_markdown,
)
from scripts.step2.resources import (
    build_human_gene_alias_table,
    ensure_step2_resources,
    load_cellchat_database,
    load_nichenet_gene_aliases,
    load_nichenet_geneinfo,
    load_nichenet_lr_network,
    parse_cellchat_lr_pairs,
    parse_nichenet_lr_pairs,
)


DEFAULT_CONFIG = Path("configs/step2/step2_comm_panel.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step 2 communication gene panel.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def flatten_pair_gene_strings(frame: pd.DataFrame, column: str) -> list[str]:
    genes: list[str] = []
    for value in frame[column].astype(str).tolist():
        genes.extend([canonical_gene_symbol(part) for part in value.split("|") if canonical_gene_symbol(part)])
    return genes


def resolve_gene_symbol(symbol: str, alias_lookup: dict[str, str], source_gene_universe: set[str]) -> tuple[str, str]:
    canonical = canonical_gene_symbol(symbol)
    if not canonical:
        return "", "empty"
    if canonical in alias_lookup:
        mapped = alias_lookup[canonical]
        return mapped, ("exact" if mapped == canonical else "alias")
    if canonical in source_gene_universe:
        return canonical, "source_identity"
    return canonical, "unmapped"


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_dir = project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    resource_paths = ensure_step2_resources(config)
    cellchat_db = load_cellchat_database(resource_paths["cellchatdb_human_rda"])
    cellchat_pairs = parse_cellchat_lr_pairs(cellchat_db)
    nichenet_lr = load_nichenet_lr_network(resource_paths["nichenet_lr_network_human_rds"])
    nichenet_pairs = parse_nichenet_lr_pairs(nichenet_lr)
    geneinfo_human = load_nichenet_geneinfo(resource_paths["nichenet_geneinfo_human_rda"])
    alias_human = load_nichenet_gene_aliases(resource_paths["nichenet_geneinfo_alias_human_rda"])

    alias_table = build_human_gene_alias_table(
        geneinfo_human=geneinfo_human,
        geneinfo_alias_human=alias_human,
        manual_aliases=config.get("manual_aliases", {}),
    )
    alias_lookup = dict(zip(alias_table["alias"], alias_table["gene_symbol"]))

    resource_dir = project_path(config["paths"]["resource_dir"])
    cellchat_pairs.to_parquet(resource_dir / "cellchat_lr_pairs.parquet", index=False)
    nichenet_pairs.to_parquet(resource_dir / "nichenet_lr_pairs.parquet", index=False)
    alias_table.to_parquet(resource_dir / "human_gene_alias_table.parquet", index=False)

    source_gene_coverage: dict[str, set[str]] = {}
    source_gene_universe: set[str] = set()
    for source_id in iter_source_ids_from_atlas():
        adata = ad.read_h5ad(resolve_source_snapshot_path(source_id), backed="r")
        source_genes = {
            alias_lookup.get(canonical_gene_symbol(gene), canonical_gene_symbol(gene))
            for gene in adata.var_names.astype(str)
            if canonical_gene_symbol(gene)
        }
        source_gene_coverage[source_id] = source_genes
        source_gene_universe |= source_genes
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()

    gene_origins: dict[str, set[str]] = defaultdict(set)
    original_gene_rows: list[dict[str, str]] = []

    for gene in flatten_pair_gene_strings(cellchat_pairs, "ligand_genes") + flatten_pair_gene_strings(cellchat_pairs, "receptor_genes"):
        original_gene_rows.append({"original_gene": gene, "origin": "cellchat"})
    for gene in flatten_pair_gene_strings(nichenet_pairs, "ligand_genes") + flatten_pair_gene_strings(nichenet_pairs, "receptor_genes"):
        original_gene_rows.append({"original_gene": gene, "origin": "nichenet"})
    for category, genes in config["supplement_gene_sets"].items():
        for gene in genes:
            original_gene_rows.append({"original_gene": gene, "origin": f"supplement:{category}"})
    for gene in config["sentinel_genes"]:
        original_gene_rows.append({"original_gene": gene, "origin": "sentinel"})

    mapping_rows: list[dict[str, str | int]] = []
    kept_gene_rows: list[dict[str, object]] = []
    panel_seen: set[str] = set()
    for row in original_gene_rows:
        original_gene = canonical_gene_symbol(row["original_gene"])
        mapped_gene, mapping_status = resolve_gene_symbol(original_gene, alias_lookup, source_gene_universe)
        present_sources = sorted(source_id for source_id, genes in source_gene_coverage.items() if mapped_gene in genes)
        kept = bool(mapped_gene and present_sources)
        mapping_rows.append(
            {
                "original_gene": original_gene,
                "mapped_gene": mapped_gene,
                "mapping_status": mapping_status,
                "origin": row["origin"],
                "present_source_count": len(present_sources),
                "present_sources": "|".join(present_sources),
                "kept_in_panel": kept,
            }
        )
        if kept:
            gene_origins[mapped_gene].add(str(row["origin"]))

    for gene_symbol, origins in sorted(gene_origins.items()):
        panel_seen.add(gene_symbol)
        origin_list = sorted(origins)
        supplement_categories = sorted(origin.split(":", 1)[1] for origin in origin_list if origin.startswith("supplement:"))
        kept_gene_rows.append(
            {
                "gene_symbol": gene_symbol,
                "panel_origin_count": len(origin_list),
                "panel_origins": "|".join(origin_list),
                "in_cellchat": "cellchat" in origin_list,
                "in_nichenet": "nichenet" in origin_list,
                "is_supplement": any(origin.startswith("supplement:") for origin in origin_list),
                "supplement_categories": "|".join(supplement_categories),
                "is_sentinel": "sentinel" in origin_list,
                "present_source_count": sum(gene_symbol in genes for genes in source_gene_coverage.values()),
                "present_sources": "|".join(sorted(source_id for source_id, genes in source_gene_coverage.items() if gene_symbol in genes)),
            }
        )

    panel_df = pd.DataFrame(kept_gene_rows).sort_values(["is_sentinel", "panel_origin_count", "gene_symbol"], ascending=[False, False, True])
    panel_df.to_csv(output_dir / "step2_comm_gene_panel.csv", index=False)

    mapping_df = pd.DataFrame(mapping_rows).sort_values(["kept_in_panel", "mapping_status", "original_gene"], ascending=[False, True, True])
    mapping_df.to_csv(output_dir / "step2_gene_mapping_table.csv", index=False)

    summary = {
        "panel_genes": int(panel_df.shape[0]),
        "panel_genes_from_cellchat": int(panel_df["in_cellchat"].sum()),
        "panel_genes_from_nichenet": int(panel_df["in_nichenet"].sum()),
        "panel_supplement_genes": int(panel_df["is_supplement"].sum()),
        "panel_sentinel_genes": int(panel_df["is_sentinel"].sum()),
        "mapping_status_counts": mapping_df["mapping_status"].value_counts().sort_index().to_dict(),
        "dropped_genes": int((~mapping_df["kept_in_panel"]).sum()),
    }
    write_json(output_dir / "step2_comm_gene_panel_summary.json", summary)

    top_missing = (
        mapping_df.loc[~mapping_df["kept_in_panel"], ["original_gene", "origin", "mapping_status"]]
        .drop_duplicates()
        .head(25)
    )
    top_panel = panel_df[["gene_symbol", "panel_origins", "present_sources"]].head(25)
    report_lines = [
        "# Step 2 Communication Gene Panel",
        "",
        "## Summary",
        f"- panel_genes: {summary['panel_genes']}",
        f"- genes_from_cellchat: {summary['panel_genes_from_cellchat']}",
        f"- genes_from_nichenet: {summary['panel_genes_from_nichenet']}",
        f"- supplement_genes: {summary['panel_supplement_genes']}",
        f"- sentinel_genes: {summary['panel_sentinel_genes']}",
        f"- dropped_genes: {summary['dropped_genes']}",
        "",
        "## Mapping Status Counts",
        markdown_table(pd.DataFrame(summary["mapping_status_counts"].items(), columns=["mapping_status", "n_genes"])),
        "",
        "## Example Panel Genes",
        markdown_table(top_panel),
        "",
        "## Example Dropped Genes",
        markdown_table(top_missing),
        "",
        "## Resource Outputs",
        f"- CellChat parsed pairs: `{resource_dir / 'cellchat_lr_pairs.parquet'}`",
        f"- NicheNet parsed pairs: `{resource_dir / 'nichenet_lr_pairs.parquet'}`",
        f"- Alias table: `{resource_dir / 'human_gene_alias_table.parquet'}`",
    ]
    write_markdown(output_dir / "step2_gene_mapping_report.md", "\n".join(report_lines))


if __name__ == "__main__":
    main()
