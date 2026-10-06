#!/usr/bin/env python3.12
"""Download and harmonize the Step 4 genetic resource layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import project_path
from scripts.step4.common import (
    PRIMARY_EQTL_TISSUES,
    STEP2_COMM_PANEL,
    download_file,
    ensure_step4_dir,
    log,
    program_gene_memberships,
    retry_get,
    stream_download_file,
)


DEFAULT_OUTPUT_DIR = Path("data/processed/step4/02_genetics/00_resources")
FINNGEN_MANIFEST_URL = "https://storage.googleapis.com/finngen-public-data-r10/summary_stats/R10_manifest.tsv"
GCST_META_URL = "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90275001-GCST90276000/GCST90275127/GCST90275127.tsv-meta.yaml"
GCST_TSV_URL = "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90275001-GCST90276000/GCST90275127/GCST90275127.tsv"
GTEX_GENE_URL = "https://gtexportal.org/api/v2/reference/gene"
GTEX_EQTL_URL = "https://gtexportal.org/api/v2/association/singleTissueEqtl"
GTEX_TISSUE_URL = "https://gtexportal.org/api/v2/dataset/tissueSiteDetail"
ENSEMBL_LOOKUP_URL = "https://rest.ensembl.org/lookup/symbol/homo_sapiens/{symbol}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step 4 genetic resource metadata.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def finngen_parkinson_row(manifest: pd.DataFrame) -> pd.Series:
    mask = manifest["phenocode"].astype(str).eq("G6_PARKINSON")
    if not mask.any():
        raise KeyError("Could not find G6_PARKINSON in the FinnGen R10 manifest")
    return manifest.loc[mask].iloc[0]


def fetch_gtex_gene(symbol: str) -> dict[str, object]:
    response = retry_get(GTEX_GENE_URL, params={"geneId": symbol})
    payload = response.json()
    data = payload.get("data", [])
    if data:
        return dict(data[0])
    response = retry_get(
        ENSEMBL_LOOKUP_URL.format(symbol=symbol),
        headers={"Content-Type": "application/json"},
    )
    payload = response.json()
    return {
        "geneSymbol": symbol,
        "gencodeId": payload.get("id"),
        "chromosome": payload.get("seq_region_name"),
        "start": payload.get("start"),
        "end": payload.get("end"),
        "strand": payload.get("strand"),
        "geneType": payload.get("biotype"),
        "resource": "ensembl_fallback",
    }


def collect_gene_universe() -> pd.DataFrame:
    memberships = program_gene_memberships()
    if memberships.empty:
        return pd.DataFrame(
            columns=[
                "gene_symbol",
                "gencode_id",
                "chromosome",
                "start",
                "end",
                "strand",
                "gene_type",
                "program_ids_json",
                "program_count",
                "resource_hint",
            ]
        )
    grouped = (
        memberships.groupby("gene_symbol", observed=False)["program_id"]
        .agg(lambda values: sorted(set(values)))
        .reset_index()
        .rename(columns={"gene_symbol": "gene_symbol", "program_id": "program_ids"})
    )
    rows: list[dict[str, object]] = []
    for record in grouped.itertuples(index=False):
        gene_meta = fetch_gtex_gene(record.gene_symbol)
        rows.append(
            {
                "gene_symbol": record.gene_symbol,
                "gencode_id": gene_meta.get("gencodeId"),
                "chromosome": gene_meta.get("chromosome"),
                "start": gene_meta.get("start"),
                "end": gene_meta.get("end"),
                "strand": gene_meta.get("strand"),
                "gene_type": gene_meta.get("geneType"),
                "program_ids_json": json.dumps(record.program_ids),
                "program_count": len(record.program_ids),
                "resource_hint": gene_meta.get("resource", "gtex"),
            }
        )
    return pd.DataFrame(rows).sort_values(["chromosome", "start", "gene_symbol"]).reset_index(drop=True)


def fetch_gtex_eqtls(gencode_id: str, tissue: str) -> pd.DataFrame:
    response = retry_get(
        GTEX_EQTL_URL,
        params={
            "datasetId": "gtex_v8",
            "gencodeId": gencode_id,
            "tissueSiteDetailId": tissue,
            "itemsPerPage": 250,
            "page": 0,
        },
        timeout=90,
    )
    payload = response.json()
    data = payload.get("data", [])
    return pd.DataFrame(data)


def build_instrumentability(gene_universe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gene_row in gene_universe.itertuples(index=False):
        gencode_id = gene_row.gencode_id
        if not gencode_id or pd.isna(gencode_id):
            rows.append(
                {
                    "gene_symbol": gene_row.gene_symbol,
                    "gencode_id": gencode_id,
                    "instrumentability_status": "non_instrumentable",
                    "instrumentable_in_primary_brain": False,
                    "best_tissue": "",
                    "n_eqtls": 0,
                    "best_snp_id": "",
                    "best_variant_id": "",
                    "best_p_value": float("nan"),
                    "best_nes": float("nan"),
                    "instrument_tissue_panel_json": json.dumps([]),
                }
            )
            continue
        tissue_hits: list[dict[str, object]] = []
        for tissue in PRIMARY_EQTL_TISSUES:
            eqtls = fetch_gtex_eqtls(gencode_id, tissue)
            if eqtls.empty:
                continue
            eqtls = eqtls.sort_values("pValue", ascending=True)
            top = eqtls.iloc[0]
            tissue_hits.append(
                {
                    "tissue": tissue,
                    "n_eqtls": int(len(eqtls)),
                    "best_snp_id": str(top.get("snpId", "")),
                    "best_variant_id": str(top.get("variantId", "")),
                    "best_p_value": float(top.get("pValue", float("nan"))),
                    "best_nes": float(top.get("nes", float("nan"))),
                }
            )
        if tissue_hits:
            top_hit = sorted(tissue_hits, key=lambda item: (item["best_p_value"], -item["n_eqtls"]))[0]
            rows.append(
                {
                    "gene_symbol": gene_row.gene_symbol,
                    "gencode_id": gencode_id,
                    "instrumentability_status": "instrumentable",
                    "instrumentable_in_primary_brain": True,
                    "best_tissue": top_hit["tissue"],
                    "n_eqtls": sum(item["n_eqtls"] for item in tissue_hits),
                    "best_snp_id": top_hit["best_snp_id"],
                    "best_variant_id": top_hit["best_variant_id"],
                    "best_p_value": top_hit["best_p_value"],
                    "best_nes": top_hit["best_nes"],
                    "instrument_tissue_panel_json": json.dumps(tissue_hits),
                }
            )
        else:
            rows.append(
                {
                    "gene_symbol": gene_row.gene_symbol,
                    "gencode_id": gencode_id,
                    "instrumentability_status": "non_instrumentable",
                    "instrumentable_in_primary_brain": False,
                    "best_tissue": "",
                    "n_eqtls": 0,
                    "best_snp_id": "",
                    "best_variant_id": "",
                    "best_p_value": float("nan"),
                    "best_nes": float("nan"),
                    "instrument_tissue_panel_json": json.dumps([]),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "gene_symbol",
                "gencode_id",
                "instrumentability_status",
                "instrumentable_in_primary_brain",
                "best_tissue",
                "n_eqtls",
                "best_snp_id",
                "best_variant_id",
                "best_p_value",
                "best_nes",
                "instrument_tissue_panel_json",
            ]
        )
    return pd.DataFrame(rows).sort_values(["instrumentability_status", "best_p_value", "gene_symbol"], ascending=[True, True, True]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    downloads_dir = output_dir / "downloads"
    ensure_step4_dir("02_genetics", "00_resources")
    downloads_dir.mkdir(parents=True, exist_ok=True)

    log("[step4.genetics.resources] downloading manifests")
    finngen_manifest_path = download_file(FINNGEN_MANIFEST_URL, downloads_dir / "R10_manifest.tsv")
    finngen_manifest = pd.read_csv(finngen_manifest_path, sep="\t")
    finngen_pd = finngen_parkinson_row(finngen_manifest)
    finngen_pd_path = stream_download_file(
        str(finngen_pd["path_https"]),
        downloads_dir / "finngen_R10_G6_PARKINSON.gz",
    )

    gcst_meta_path = download_file(GCST_META_URL, downloads_dir / "GCST90275127.tsv-meta.yaml")
    gcst_path = stream_download_file(GCST_TSV_URL, downloads_dir / "GCST90275127.tsv")

    tissue_path = download_file(GTEX_TISSUE_URL, downloads_dir / "gtex_tissueSiteDetail.json")

    gene_universe = collect_gene_universe()
    gene_universe.to_csv(output_dir / "step4_gene_universe.csv", index=False)

    instrumentability = build_instrumentability(gene_universe)
    instrumentability.to_csv(output_dir / "step4_instrumentability_table.csv", index=False)

    resource_manifest = pd.DataFrame(
        [
            {
                "resource_id": "finngen_r10_g6_parkinson",
                "resource_type": "gwas_outcome",
                "trait_or_molecule": "Parkinson's disease",
                "study_label": "FinnGen R10 G6_PARKINSON",
                "ancestry": "European",
                "n_cases": int(finngen_pd["num_cases"]),
                "n_controls": int(finngen_pd["num_controls"]),
                "n_samples_total": int(finngen_pd["num_cases"]) + int(finngen_pd["num_controls"]),
                "genome_build": "GRCh38",
                "effect_allele_field": "alt",
                "other_allele_field": "ref",
                "beta_field": "beta",
                "se_field": "sebeta",
                "p_field": "pval",
                "eaf_field": "af_alt",
                "download_status": "downloaded",
                "access_class": "public_download",
                "local_path": str(finngen_pd_path.relative_to(PROJECT_ROOT)),
                "source_url": str(finngen_pd["path_https"]),
            },
            {
                "resource_id": "gcst90275127_multi_ancestry_pd",
                "resource_type": "gwas_outcome",
                "trait_or_molecule": "Parkinson's disease",
                "study_label": "Kim et al. 2024 multi-ancestry PD meta-analysis",
                "ancestry": "Multi-ancestry",
                "n_cases": "",
                "n_controls": "",
                "n_samples_total": "",
                "genome_build": "GRCh37",
                "effect_allele_field": "effect_allele",
                "other_allele_field": "other_allele",
                "beta_field": "beta",
                "se_field": "standard_error",
                "p_field": "p_value",
                "eaf_field": "effect_allele_frequency",
                "download_status": "downloaded",
                "access_class": "public_download",
                "local_path": str(gcst_path.relative_to(PROJECT_ROOT)),
                "source_url": GCST_TSV_URL,
            },
            {
                "resource_id": "gtex_v8_brain_eqtl_api",
                "resource_type": "eqtl_exposure",
                "trait_or_molecule": "cis-eQTL",
                "study_label": "GTEx v8 brain tissues",
                "ancestry": "Predominantly European / mixed donor panel",
                "n_cases": "",
                "n_controls": "",
                "n_samples_total": "",
                "genome_build": "GRCh38",
                "effect_allele_field": "variantId",
                "other_allele_field": "",
                "beta_field": "nes",
                "se_field": "derived_from_beta_and_p",
                "p_field": "pValue",
                "eaf_field": "",
                "download_status": "api_accessed",
                "access_class": "public_api",
                "local_path": str((downloads_dir / "gtex_tissueSiteDetail.json").relative_to(PROJECT_ROOT)),
                "source_url": GTEX_EQTL_URL,
            },
        ]
    )
    resource_manifest.to_csv(output_dir / "resource_manifest.csv", index=False)
    log("[step4.genetics.resources] resource layer ready")


if __name__ == "__main__":
    main()
