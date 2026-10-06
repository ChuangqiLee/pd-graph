#!/usr/bin/env python3.12
"""Download, ingest, and QC the primary Step 4 human Visium dataset."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import tarfile
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pyreadr
import scanpy as sc
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import markdown_table, project_path, write_markdown
from scripts.step4.common import ensure_step4_dir, download_file, log, stream_download_file


DEFAULT_OUTPUT_DIR = Path("data/processed/step4/03_spatial/01_primary_visium")
SOFT_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE253nnn/GSE253975/soft/GSE253975_family.soft.gz"
RAW_TAR_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE253nnn/GSE253975/suppl/GSE253975_RAW.tar"
VISIUM_BARCODE_MAP_URL = "https://raw.githubusercontent.com/pachterlab/SpatialFeatureExperiment/main/data/visium_row_col.rda"
DA_CORE_GENES = ["TH", "SLC6A3", "DDC", "ALDH1A1", "GCH1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest the primary Step 4 Visium dataset.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def parse_soft_manifest(path: Path) -> pd.DataFrame:
    samples: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.rstrip()
            if line.startswith("^SAMPLE = "):
                if current is not None:
                    samples.append(current)
                current = {"geo_accession": line.split(" = ", maxsplit=1)[1]}
                continue
            if current is None:
                continue
            if line.startswith("!Sample_title = "):
                current["sample_title"] = line.split(" = ", maxsplit=1)[1]
            elif line.startswith("!Sample_characteristics_ch1 = "):
                payload = line.split(" = ", maxsplit=1)[1]
                key, _, value = payload.partition(":")
                current[key.strip().lower()] = value.strip()
            elif line.startswith("!Sample_supplementary_file_1 = "):
                current["counts_url"] = line.split(" = ", maxsplit=1)[1]
            elif line.startswith("!Sample_supplementary_file_2 = "):
                current["json_url"] = line.split(" = ", maxsplit=1)[1]
    if current is not None:
        samples.append(current)
    manifest = pd.DataFrame(samples)
    manifest["sample_code"] = (
        manifest["sample_title"]
        .astype(str)
        .str.split(",", n=1)
        .str[0]
        .str.replace("-", "", regex=False)
        .str.strip()
    )
    manifest["counts_filename"] = manifest["counts_url"].astype(str).map(lambda value: Path(value).name)
    manifest["json_filename"] = manifest["json_url"].astype(str).map(lambda value: Path(value).name)
    manifest["donor_id"] = manifest["sample_code"]
    manifest["section_id"] = manifest["json_filename"].astype(str).str.replace(".json.gz", "", regex=False)
    manifest["disease_group"] = (
        manifest["condition"].astype(str).str.strip().str.lower().map({"pd": "pd", "control": "control", "hc": "control"}).fillna("unknown")
    )
    manifest["sex"] = manifest["sex"].astype(str).str.strip().str.lower().map({"m": "male", "f": "female"}).fillna(manifest["sex"].astype(str))
    manifest["age_years"] = pd.to_numeric(manifest["age"], errors="coerce")
    return manifest.sort_values("geo_accession").reset_index(drop=True)


def ensure_tar_members(raw_tar_path: Path, extraction_dir: Path, filenames: list[str]) -> None:
    extraction_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in filenames if not (extraction_dir / name).exists()]
    if not missing:
        return
    with tarfile.open(raw_tar_path, "r") as tar:
        members = {member.name: member for member in tar.getmembers()}
        for name in missing:
            member = members.get(name)
            if member is None:
                raise FileNotFoundError(f"Missing {name} inside {raw_tar_path}")
            tar.extract(member, extraction_dir)


def load_barcode_map(path: Path) -> pd.DataFrame:
    payload = pyreadr.read_r(str(path))
    frame = next(iter(payload.values())).copy()
    frame["barcode_core"] = frame["barcode"].astype(str)
    return frame[["barcode_core", "row", "col"]].drop_duplicates().reset_index(drop=True)


def read_sparse_count_matrix(path: Path) -> tuple[sp.csr_matrix, list[str], list[str]]:
    data: list[float] = []
    row_index: list[int] = []
    col_index: list[int] = []
    gene_names: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
        reader = csv.reader(handle, delimiter=" ", quotechar='"', skipinitialspace=True)
        header = next(reader)
        barcodes = [token for token in header if token]
        n_spots = len(barcodes)
        for gene_idx, row in enumerate(reader):
            tokens = [token for token in row if token]
            if not tokens:
                continue
            gene_names.append(tokens[0])
            values = np.fromiter((float(token) for token in tokens[1:]), dtype=np.float32, count=n_spots)
            nonzero = np.flatnonzero(values)
            if len(nonzero) == 0:
                continue
            row_index.extend(nonzero.tolist())
            col_index.extend([gene_idx] * len(nonzero))
            data.extend(values[nonzero].tolist())
    matrix = sp.csr_matrix((data, (row_index, col_index)), shape=(len(barcodes), len(gene_names)), dtype=np.float32)
    return matrix, gene_names, barcodes


def donor_minmax(values: pd.Series) -> pd.Series:
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value <= min_value:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - min_value) / (max_value - min_value)


def dense_array(matrix: sp.spmatrix | np.ndarray) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def assign_compartments(visium: ad.AnnData) -> ad.AnnData:
    if "da_combined_score" not in visium.obs.columns:
        return visium
    rank_pct = visium.obs.groupby("sample_accession", observed=False)["da_combined_score"].rank(method="first", pct=True)
    visium.obs["anatomical_compartment"] = np.where(rank_pct > 0.65, "SNpc", "surrounding_tissue")
    visium.obs["spot_id"] = visium.obs_names.astype(str)
    return visium


def build_sample_adata(record: pd.Series, extraction_dir: Path, barcode_map: pd.DataFrame) -> tuple[ad.AnnData, dict[str, object]]:
    counts_path = extraction_dir / str(record["counts_filename"])
    json_path = extraction_dir / str(record["json_filename"])
    matrix, genes, barcodes = read_sparse_count_matrix(counts_path)
    with gzip.open(json_path, "rt", encoding="utf-8", errors="ignore") as handle:
        payload = json.load(handle)
    alignment = pd.DataFrame(payload["oligo"])[["row", "col", "imageX", "imageY", "x", "y", "dia"]].copy()
    spot_meta = pd.DataFrame({"barcode_full": barcodes})
    spot_meta["barcode_core"] = spot_meta["barcode_full"].astype(str).str.split("_").str[-1].str.split(".").str[0]
    spot_meta = spot_meta.merge(barcode_map, on="barcode_core", how="left").merge(alignment, on=["row", "col"], how="left")
    if spot_meta[["row", "col", "imageX", "imageY"]].isna().any(axis=None):
        raise ValueError(f"Failed to recover coordinates for {record['geo_accession']}")
    spot_meta["sample_accession"] = record["geo_accession"]
    spot_meta["sample_code"] = record["sample_code"]
    spot_meta["donor_id"] = record["donor_id"]
    spot_meta["section_id"] = record["section_id"]
    spot_meta["sample_title"] = record["sample_title"]
    spot_meta["disease_group"] = record["disease_group"]
    spot_meta["sex"] = record["sex"]
    spot_meta["age_years"] = record["age_years"]
    spot_meta["tissue"] = record.get("tissue", "Substantia Nigra")
    spot_meta["spot_id"] = spot_meta["sample_accession"].astype(str) + "__" + spot_meta["barcode_full"].astype(str)
    obs = spot_meta.set_index("spot_id", drop=True)
    sample_adata = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=pd.Index(genes, name="gene_symbol")))
    sample_adata.layers["counts"] = sample_adata.X.copy()
    sample_adata.obsm["spatial"] = obs[["imageX", "imageY"]].to_numpy(dtype=np.float32)
    return sample_adata, {
        "sample_accession": record["geo_accession"],
        "donor_id": record["donor_id"],
        "sample_code": record["sample_code"],
        "disease_group": record["disease_group"],
        "sex": record["sex"],
        "age_years": record["age_years"],
        "counts_path": str(counts_path),
        "json_path": str(json_path),
        "raw_spot_count": int(sample_adata.n_obs),
    }


def main() -> None:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    downloads_dir = output_dir / "downloads"
    extraction_dir = downloads_dir / "raw_members"
    ensure_step4_dir("03_spatial", "01_primary_visium")
    downloads_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "primary_visium_manifest.csv"
    filtered_path = output_dir / "primary_visium_filtered.h5ad"

    if filtered_path.exists() and manifest_path.exists():
        visium = ad.read_h5ad(filtered_path)
        manifest_base = pd.read_csv(manifest_path)
    else:
        soft_path = download_file(SOFT_URL, downloads_dir / "GSE253975_family.soft.gz")
        raw_tar_path = stream_download_file(RAW_TAR_URL, downloads_dir / "GSE253975_RAW.tar")
        barcode_map_path = download_file(VISIUM_BARCODE_MAP_URL, downloads_dir / "visium_row_col.rda")

        manifest = parse_soft_manifest(soft_path)
        ensure_tar_members(raw_tar_path, extraction_dir, manifest["counts_filename"].tolist() + manifest["json_filename"].tolist())
        barcode_map = load_barcode_map(barcode_map_path)

        sample_adatas: list[ad.AnnData] = []
        manifest_rows: list[dict[str, object]] = []
        for sample in manifest.to_dict(orient="records"):
            log(f"[step4.spatial.primary] ingesting {sample['geo_accession']}")
            adata, row = build_sample_adata(pd.Series(sample), extraction_dir, barcode_map)
            sample_adatas.append(adata)
            manifest_rows.append(row)

        visium = ad.concat(sample_adatas, join="outer", merge="same")
        visium.layers["counts"] = visium.X.copy()
        total_counts = np.asarray(visium.layers["counts"].sum(axis=1)).ravel()
        n_genes_by_counts = np.asarray((visium.layers["counts"] > 0).sum(axis=1)).ravel()
        mito_mask = pd.Index(visium.var_names.astype(str)).str.upper().str.startswith("MT-")
        mito_counts = np.asarray(visium.layers["counts"][:, mito_mask].sum(axis=1)).ravel() if mito_mask.any() else np.zeros(visium.n_obs, dtype=float)
        visium.obs["total_counts"] = total_counts
        visium.obs["n_genes_by_counts"] = n_genes_by_counts
        visium.obs["pct_mito"] = np.divide(mito_counts, np.maximum(total_counts, 1.0)) * 100.0

        keep_mask = (visium.obs["total_counts"] >= 200.0) & (visium.obs["n_genes_by_counts"] >= 150.0)
        visium = visium[keep_mask.to_numpy()].copy()
        visium.obs["in_tissue"] = True

        visium.X = visium.layers["counts"].copy()
        sc.pp.normalize_total(visium, target_sum=1e4)
        sc.pp.log1p(visium)

        da_present = [gene for gene in DA_CORE_GENES if gene in visium.var_names]
        da_expr = pd.DataFrame(dense_array(visium[:, da_present].X), index=visium.obs_names, columns=da_present) if da_present else pd.DataFrame(index=visium.obs_names)
        visium.obs["da_core_score"] = da_expr.mean(axis=1) if not da_expr.empty else 0.0
        th_expr = dense_array(visium[:, ["TH"]].X).ravel() if "TH" in visium.var_names else np.zeros(visium.n_obs, dtype=float)
        visium.obs["th_expression"] = th_expr
        visium.obs["da_combined_score"] = (
            visium.obs.groupby("sample_accession", observed=False)["da_core_score"].transform(donor_minmax)
            + 0.5 * visium.obs.groupby("sample_accession", observed=False)["th_expression"].transform(donor_minmax)
        )
        visium = assign_compartments(visium)
        visium.write_h5ad(filtered_path)

        manifest_frame = pd.DataFrame(manifest_rows)
        kept_counts = visium.obs.groupby("sample_accession", observed=False).size().rename("kept_spot_count").reset_index()
        snpc_counts = (
            visium.obs.groupby(["sample_accession", "anatomical_compartment"], observed=False)
            .size()
            .rename("n_spots")
            .reset_index()
            .pivot(index="sample_accession", columns="anatomical_compartment", values="n_spots")
            .fillna(0)
            .reset_index()
        )
        snpc_counts.columns.name = None
        manifest_base = manifest.rename(columns={"geo_accession": "sample_accession"}).copy()
        manifest_base = manifest_base.merge(manifest_frame, on=["sample_accession", "donor_id", "sample_code", "disease_group", "sex", "age_years"], how="left")
        manifest_base = manifest_base.merge(kept_counts, on="sample_accession", how="left")
        manifest_base = manifest_base.merge(snpc_counts, on="sample_accession", how="left")
        manifest_base.to_csv(manifest_path, index=False)

    visium = assign_compartments(visium)
    visium.write_h5ad(filtered_path)
    kept_counts = visium.obs.groupby("sample_accession", observed=False).size().rename("kept_spot_count").reset_index()
    snpc_counts = (
        visium.obs.groupby(["sample_accession", "anatomical_compartment"], observed=False)
        .size()
        .rename("n_spots")
        .reset_index()
        .pivot(index="sample_accession", columns="anatomical_compartment", values="n_spots")
        .fillna(0)
        .reset_index()
    )
    snpc_counts.columns.name = None
    manifest_base = manifest_base.drop(columns=[column for column in ["SNpc", "surrounding_tissue", "kept_spot_count"] if column in manifest_base.columns])
    manifest_base = manifest_base.merge(kept_counts, on="sample_accession", how="left")
    manifest_base = manifest_base.merge(snpc_counts, on="sample_accession", how="left")
    manifest_base.to_csv(manifest_path, index=False)

    donor_summary = (
        visium.obs.groupby(["sample_accession", "disease_group", "anatomical_compartment"], observed=False)
        .agg(
            n_spots=("spot_id", "count"),
            mean_total_counts=("total_counts", "mean"),
            mean_n_genes=("n_genes_by_counts", "mean"),
            mean_pct_mito=("pct_mito", "mean"),
            mean_da_score=("da_combined_score", "mean"),
        )
        .reset_index()
    )
    report_lines = [
        "# Primary Visium QC Report",
        "",
        "The primary Step 4 spatial object was rebuilt from the GEO supplementary count matrices and alignment JSON files for `GSE253975`.",
        "",
        "## QC thresholds",
        "- Keep spots with `total_counts >= 200` and `n_genes_by_counts >= 150`.",
        "- Treat all matrix barcodes as in-tissue, then split `SNpc` versus `surrounding_tissue` by a donor-specific dopaminergic score threshold at the 65th percentile.",
        "",
        "## Donor-compartment summary",
        markdown_table(donor_summary.astype(object).fillna("NA")),
        "",
    ]
    write_markdown(output_dir / "primary_visium_qc_report.md", "\n".join(report_lines))


if __name__ == "__main__":
    main()
