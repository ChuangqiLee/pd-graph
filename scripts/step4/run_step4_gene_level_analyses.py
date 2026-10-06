#!/usr/bin/env python3.12
"""Run gene-level MR and coloc-style support analyses for Step 4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step4.common import bh_fdr, ensure_step4_dir, retry_get


DEFAULT_INPUT_DIR = Path("data/processed/step4/02_genetics/00_resources")
DEFAULT_OUTPUT_DIR = Path("data/processed/step4/02_genetics/01_gene_level")
GTEX_EQTL_URL = "https://gtexportal.org/api/v2/association/singleTissueEqtl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 4 gene-level genetic support analyses.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def parse_variant_id(variant_id: str) -> tuple[str, int, str, str]:
    parts = str(variant_id).split("_")
    if len(parts) < 4:
        return "", -1, "", ""
    return str(parts[0]), int(parts[1]), str(parts[2]), str(parts[3])


def derive_se_from_beta_p(beta: float, p_value: float) -> float:
    beta = float(beta)
    p_value = float(p_value)
    if not np.isfinite(beta) or beta == 0.0 or not np.isfinite(p_value) or p_value <= 0.0 or p_value >= 1.0:
        return float("nan")
    z_abs = float(norm.isf(p_value / 2.0))
    if not np.isfinite(z_abs) or z_abs == 0.0:
        return float("nan")
    return abs(beta) / z_abs


def fetch_gene_eqtls(gencode_id: str, tissue: str) -> pd.DataFrame:
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
    return pd.DataFrame(payload.get("data", []))


def clump_variants(eqtl: pd.DataFrame, *, min_distance_bp: int = 750_000, max_instruments: int = 3) -> pd.DataFrame:
    if eqtl.empty:
        return eqtl.copy()
    eqtl = eqtl.sort_values("pValue", ascending=True).copy()
    chosen: list[pd.Series] = []
    chosen_positions: list[tuple[str, int]] = []
    for row in eqtl.itertuples(index=False):
        chrom, pos, ref, alt = parse_variant_id(str(row.variantId))
        if pos < 0:
            continue
        if any(chrom == chosen_chrom and abs(pos - chosen_pos) < min_distance_bp for chosen_chrom, chosen_pos in chosen_positions):
            continue
        chosen.append(pd.Series(row._asdict()))
        chosen_positions.append((chrom, pos))
        if len(chosen) >= max_instruments:
            break
    if not chosen:
        return eqtl.head(1).copy()
    out = pd.DataFrame(chosen)
    out["instrument_rank"] = np.arange(1, len(out) + 1)
    return out


def collect_exposures(gene_universe: pd.DataFrame, instrumentability: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    instrument_lookup = instrumentability.set_index("gene_symbol")
    rows: list[pd.DataFrame] = []
    all_rsids: set[str] = set()
    for gene_row in gene_universe.itertuples(index=False):
        if gene_row.gene_symbol not in instrument_lookup.index:
            continue
        inst_row = instrument_lookup.loc[gene_row.gene_symbol]
        if str(inst_row["instrumentability_status"]) != "instrumentable":
            continue
        tissue = str(inst_row["best_tissue"])
        if not tissue:
            continue
        eqtl = fetch_gene_eqtls(str(gene_row.gencode_id), tissue)
        if eqtl.empty:
            continue
        eqtl["gene_symbol"] = gene_row.gene_symbol
        eqtl["gencode_id"] = gene_row.gencode_id
        eqtl["tissue"] = tissue
        eqtl["beta_exposure"] = pd.to_numeric(eqtl["nes"], errors="coerce")
        eqtl["p_exposure"] = pd.to_numeric(eqtl["pValue"], errors="coerce")
        eqtl["se_exposure"] = [derive_se_from_beta_p(beta, p_value) for beta, p_value in zip(eqtl["beta_exposure"], eqtl["p_exposure"])]
        eqtl["F_stat"] = (eqtl["beta_exposure"].astype(float) / eqtl["se_exposure"].astype(float)) ** 2
        eqtl["chromosome"], eqtl["position"], eqtl["ref_allele"], eqtl["alt_allele"] = zip(*eqtl["variantId"].map(parse_variant_id))
        eqtl["snp_id"] = eqtl["snpId"].astype(str)
        clumped = clump_variants(eqtl)
        rank_lookup = dict(zip(clumped["snp_id"].astype(str), clumped["instrument_rank"].astype(int)))
        eqtl["is_selected_instrument"] = eqtl["snp_id"].astype(str).isin(rank_lookup)
        eqtl["instrument_rank"] = eqtl["snp_id"].astype(str).map(rank_lookup).fillna(0).astype(int)
        rows.append(eqtl)
        all_rsids.update(eqtl["snp_id"].dropna().astype(str).tolist())
    exposure = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return exposure, {rsid for rsid in all_rsids if rsid and rsid.lower() != "nan"}


def subset_cache_meta(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.meta.json")


def subset_cache_state(rsids: set[str], source_path: Path) -> dict[str, object]:
    stat = source_path.stat()
    digest = hashlib.sha256("\n".join(sorted(rsids)).encode("utf-8")).hexdigest()
    return {
        "rsid_sha256": digest,
        "n_rsids": len(rsids),
        "source_path": str(source_path.resolve()),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
    }


def load_cached_subset(output_path: Path, rsids: set[str], source_path: Path) -> pd.DataFrame | None:
    meta_path = subset_cache_meta(output_path)
    if not output_path.exists() or not meta_path.exists():
        return None
    try:
        cached_state = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if cached_state != subset_cache_state(rsids, source_path):
        return None
    return pd.read_csv(output_path, sep="\t")


def write_subset_cache(frame: pd.DataFrame, output_path: Path, rsids: set[str], source_path: Path) -> pd.DataFrame:
    frame.to_csv(output_path, sep="\t", index=False)
    subset_cache_meta(output_path).write_text(json.dumps(subset_cache_state(rsids, source_path), indent=2))
    return frame


def extract_outcome_subset_finngen(path: Path, rsids: set[str], output_path: Path) -> pd.DataFrame:
    cached = load_cached_subset(output_path, rsids, path)
    if cached is not None:
        return cached
    rows: list[pd.DataFrame] = []
    usecols = ["#chrom", "pos", "ref", "alt", "rsids", "nearest_genes", "pval", "beta", "sebeta", "af_alt"]
    for chunk in pd.read_csv(path, sep="\t", compression="gzip", usecols=usecols, chunksize=250_000):
        chunk["snp_id"] = chunk["rsids"].astype(str).str.split(r"[;,]", regex=True).str[0]
        local = chunk.loc[chunk["snp_id"].isin(rsids)].copy()
        if local.empty:
            continue
        local["outcome_resource"] = "finngen_r10_g6_parkinson"
        local = local.rename(
            columns={
                "#chrom": "chromosome",
                "pos": "position",
                "ref": "other_allele",
                "alt": "effect_allele",
                "pval": "p_outcome",
                "beta": "beta_outcome",
                "sebeta": "se_outcome",
                "af_alt": "eaf_outcome",
            }
        )
        rows.append(local)
    frame = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=[
                "chromosome",
                "position",
                "other_allele",
                "effect_allele",
                "rsids",
                "nearest_genes",
                "p_outcome",
                "beta_outcome",
                "se_outcome",
                "eaf_outcome",
                "snp_id",
                "outcome_resource",
            ]
        )
    )
    return write_subset_cache(frame, output_path, rsids, path)


def extract_outcome_subset_gcst(path: Path, rsids: set[str], output_path: Path) -> pd.DataFrame:
    cached = load_cached_subset(output_path, rsids, path)
    if cached is not None:
        return cached
    rows: list[pd.DataFrame] = []
    usecols = [
        "chromosome",
        "base_pair_location",
        "effect_allele",
        "other_allele",
        "beta",
        "standard_error",
        "effect_allele_frequency",
        "p_value",
        "rs_id",
    ]
    for chunk in pd.read_csv(path, sep="\t", usecols=usecols, chunksize=250_000):
        chunk["snp_id"] = chunk["rs_id"].astype(str)
        local = chunk.loc[chunk["snp_id"].isin(rsids)].copy()
        if local.empty:
            continue
        local["outcome_resource"] = "gcst90275127_multi_ancestry_pd"
        local = local.rename(
            columns={
                "base_pair_location": "position",
                "p_value": "p_outcome",
                "beta": "beta_outcome",
                "standard_error": "se_outcome",
                "effect_allele_frequency": "eaf_outcome",
            }
        )
        rows.append(local)
    frame = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=[
                "chromosome",
                "position",
                "effect_allele",
                "other_allele",
                "beta_outcome",
                "se_outcome",
                "eaf_outcome",
                "p_outcome",
                "rs_id",
                "snp_id",
                "outcome_resource",
            ]
        )
    )
    return write_subset_cache(frame, output_path, rsids, path)


def harmonize_exposure_outcome(exposure: pd.DataFrame, outcome: pd.DataFrame) -> pd.DataFrame:
    if exposure.empty or outcome.empty:
        return pd.DataFrame()
    merged = exposure.merge(outcome, on="snp_id", how="inner", suffixes=("", "_out"))
    if merged.empty:
        return merged
    aligned_rows: list[pd.Series] = []
    for row in merged.itertuples(index=False):
        exp_alt = str(row.alt_allele)
        exp_ref = str(row.ref_allele)
        out_eff = str(row.effect_allele)
        out_oth = str(row.other_allele)
        sign = None
        if out_eff == exp_alt and out_oth == exp_ref:
            sign = 1.0
        elif out_eff == exp_ref and out_oth == exp_alt:
            sign = -1.0
        elif row.outcome_resource == "finngen_r10_g6_parkinson" and out_eff == exp_alt:
            sign = 1.0
        if sign is None:
            continue
        series = pd.Series(row._asdict())
        series["beta_outcome_aligned"] = float(series["beta_outcome"]) * sign
        series["alignment_sign"] = sign
        aligned_rows.append(series)
    return pd.DataFrame(aligned_rows)


def run_gene_mr(harmonized: pd.DataFrame) -> pd.DataFrame:
    if harmonized.empty:
        return pd.DataFrame()
    harmonized = harmonized.loc[harmonized["is_selected_instrument"].astype(bool)].copy()
    rows: list[dict[str, object]] = []
    for (gene_symbol, tissue, outcome_resource), frame in harmonized.groupby(["gene_symbol", "tissue", "outcome_resource"], observed=False):
        frame = frame.sort_values("p_exposure", ascending=True).copy()
        frame = frame.loc[np.isfinite(frame["beta_exposure"]) & np.isfinite(frame["beta_outcome_aligned"]) & np.isfinite(frame["se_exposure"]) & np.isfinite(frame["se_outcome"])].copy()
        if frame.empty:
            continue
        ratios = frame["beta_outcome_aligned"].to_numpy(dtype=float) / frame["beta_exposure"].to_numpy(dtype=float)
        se_ratio = np.sqrt(
            (frame["se_outcome"].to_numpy(dtype=float) ** 2 / frame["beta_exposure"].to_numpy(dtype=float) ** 2)
            + (
                (frame["beta_outcome_aligned"].to_numpy(dtype=float) ** 2)
                * (frame["se_exposure"].to_numpy(dtype=float) ** 2)
                / frame["beta_exposure"].to_numpy(dtype=float) ** 4
            )
        )
        finite = np.isfinite(ratios) & np.isfinite(se_ratio) & (se_ratio > 0)
        frame = frame.loc[finite].copy()
        ratios = ratios[finite]
        se_ratio = se_ratio[finite]
        if len(frame) == 0:
            continue
        if len(frame) == 1:
            method = "wald_ratio"
            mr_beta = float(ratios[0])
            mr_se = float(se_ratio[0])
        else:
            weights = 1.0 / np.square(se_ratio)
            method = "ivw_approx"
            mr_beta = float(np.sum(weights * ratios) / np.sum(weights))
            mr_se = float(np.sqrt(1.0 / np.sum(weights)))
        mr_z = mr_beta / mr_se if mr_se > 0 else float("nan")
        mr_p = float(2.0 * norm.sf(abs(mr_z))) if np.isfinite(mr_z) else float("nan")
        rows.append(
            {
                "gene_symbol": gene_symbol,
                "tissue": tissue,
                "outcome_resource": outcome_resource,
                "n_instruments": int(len(frame)),
                "selected_instruments_json": json.dumps(frame["snp_id"].astype(str).tolist()),
                "top_instrument": str(frame.iloc[0]["snp_id"]),
                "method": method,
                "mr_beta": mr_beta,
                "mr_se": mr_se,
                "mr_z": mr_z,
                "mr_p_value": mr_p,
                "mean_F_stat": float(frame["F_stat"].mean()),
                "min_F_stat": float(frame["F_stat"].min()),
                "weak_instrument_flag": bool((frame["F_stat"] <= 10.0).all()),
                "exposure_direction_mean": float(frame["beta_exposure"].mean()),
            }
        )
    return pd.DataFrame(rows)


def approximate_bf(beta: np.ndarray, varbeta: np.ndarray, prior_var: float = 0.15**2) -> np.ndarray:
    varbeta = np.maximum(np.asarray(varbeta, dtype=float), 1e-12)
    beta = np.asarray(beta, dtype=float)
    z = beta / np.sqrt(varbeta)
    r = prior_var / (prior_var + varbeta)
    labf = 0.5 * (np.log(np.maximum(1.0 - r, 1e-12)) + r * np.square(z))
    return np.exp(np.clip(labf, a_min=-700, a_max=700))


def run_coloc_proxy(harmonized: pd.DataFrame) -> pd.DataFrame:
    if harmonized.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (gene_symbol, tissue, outcome_resource), frame in harmonized.groupby(["gene_symbol", "tissue", "outcome_resource"], observed=False):
        frame = frame.loc[
            np.isfinite(frame["beta_exposure"])
            & np.isfinite(frame["se_exposure"])
            & np.isfinite(frame["beta_outcome_aligned"])
            & np.isfinite(frame["se_outcome"])
        ].copy()
        if len(frame) < 2:
            pp_h4 = float("nan")
            top_variant = str(frame.iloc[0]["snp_id"]) if len(frame) == 1 else ""
            rows.append(
                {
                    "gene_symbol": gene_symbol,
                    "tissue": tissue,
                    "outcome_resource": outcome_resource,
                    "n_overlap_variants": int(len(frame)),
                    "top_shared_variant": top_variant,
                    "PP_H4_proxy": pp_h4,
                    "PP_H3_proxy": float("nan"),
                    "PP_H1_proxy": float("nan"),
                    "PP_H2_proxy": float("nan"),
                    "PP_H0_proxy": float("nan"),
                    "coloc_support_label": "low_coverage",
                    "method_note": "Less than two overlapping variants were available from the significant cis-eQTL set.",
                }
            )
            continue
        abf1 = approximate_bf(frame["beta_exposure"].to_numpy(dtype=float), np.square(frame["se_exposure"].to_numpy(dtype=float)))
        abf2 = approximate_bf(frame["beta_outcome_aligned"].to_numpy(dtype=float), np.square(frame["se_outcome"].to_numpy(dtype=float)))
        sum_abf1 = float(np.sum(abf1))
        sum_abf2 = float(np.sum(abf2))
        sum_abf12 = float(np.sum(abf1 * abf2))
        p1, p2, p12 = 1e-4, 1e-4, 1e-5
        h0 = 1.0
        h1 = p1 * sum_abf1
        h2 = p2 * sum_abf2
        h3 = p1 * p2 * max(sum_abf1 * sum_abf2 - sum_abf12, 0.0)
        h4 = p12 * sum_abf12
        denom = h0 + h1 + h2 + h3 + h4
        pp = np.asarray([h0, h1, h2, h3, h4], dtype=float) / denom if denom > 0 else np.asarray([np.nan] * 5)
        top_variant = str(frame.iloc[int(np.argmax(abf1 * abf2))]["snp_id"])
        pp_h4 = float(pp[4])
        label = "strong_proxy_shared_signal" if pp_h4 >= 0.80 else "moderate_proxy_shared_signal" if pp_h4 >= 0.60 else "weak_or_no_proxy_shared_signal"
        rows.append(
            {
                "gene_symbol": gene_symbol,
                "tissue": tissue,
                "outcome_resource": outcome_resource,
                "n_overlap_variants": int(len(frame)),
                "top_shared_variant": top_variant,
                "PP_H4_proxy": pp_h4,
                "PP_H3_proxy": float(pp[3]),
                "PP_H1_proxy": float(pp[1]),
                "PP_H2_proxy": float(pp[2]),
                "PP_H0_proxy": float(pp[0]),
                "coloc_support_label": label,
                "method_note": "Approximate coloc ABF using the overlapping significant GTEx cis-eQTL set rather than a full all-variants locus.",
            }
        )
    return pd.DataFrame(rows)


def build_support_summary(gene_universe: pd.DataFrame, instrumentability: pd.DataFrame, mr: pd.DataFrame, coloc: pd.DataFrame) -> pd.DataFrame:
    if gene_universe.empty:
        return pd.DataFrame(
            columns=[
                "gene_symbol",
                "gencode_id",
                "program_ids_json",
                "instrumentability_status",
                "best_tissue",
                "best_snp_id",
                "best_primary_mr_method",
                "best_primary_mr_p_value",
                "best_primary_mr_p_adj_fdr",
                "best_primary_mr_beta",
                "best_primary_mean_F",
                "best_primary_coloc_h4_proxy",
                "best_primary_top_shared_variant",
                "best_secondary_mr_p_value",
                "best_secondary_mr_p_adj_fdr",
                "best_secondary_mr_beta",
                "best_secondary_coloc_h4_proxy",
                "cross_outcome_direction_consistent",
                "gene_support_bin",
            ]
        )
    inst = instrumentability.set_index("gene_symbol")
    primary_mr = mr.loc[mr["outcome_resource"].eq("finngen_r10_g6_parkinson")].copy() if "outcome_resource" in mr.columns else pd.DataFrame()
    if not primary_mr.empty:
        primary_best_idx = primary_mr.groupby("gene_symbol", observed=False)["mr_p_value"].idxmin()
        primary_best = primary_mr.loc[primary_best_idx].copy()
        primary_best["mr_p_adj_fdr"] = bh_fdr(primary_best["mr_p_value"].fillna(1.0))
    else:
        primary_best = pd.DataFrame(columns=["gene_symbol", "mr_p_adj_fdr"])
    primary_best = primary_best.set_index("gene_symbol", drop=False)

    secondary_mr = mr.loc[mr["outcome_resource"].eq("gcst90275127_multi_ancestry_pd")].copy() if "outcome_resource" in mr.columns else pd.DataFrame()
    if not secondary_mr.empty:
        secondary_best_idx = secondary_mr.groupby("gene_symbol", observed=False)["mr_p_value"].idxmin()
        secondary_best = secondary_mr.loc[secondary_best_idx].copy()
        secondary_best["mr_p_adj_fdr"] = bh_fdr(secondary_best["mr_p_value"].fillna(1.0))
    else:
        secondary_best = pd.DataFrame(columns=["gene_symbol", "mr_p_adj_fdr"])
    secondary_best = secondary_best.set_index("gene_symbol", drop=False)

    primary_coloc = coloc.loc[coloc["outcome_resource"].eq("finngen_r10_g6_parkinson")].copy() if "outcome_resource" in coloc.columns else pd.DataFrame()
    if not primary_coloc.empty:
        primary_coloc = primary_coloc.loc[np.isfinite(primary_coloc["PP_H4_proxy"])].copy()
        primary_coloc_idx = primary_coloc.groupby("gene_symbol", observed=False)["PP_H4_proxy"].idxmax() if not primary_coloc.empty else pd.Index([])
        primary_coloc_idx = pd.Index(primary_coloc_idx).dropna()
        primary_coloc_best = primary_coloc.loc[primary_coloc_idx].set_index("gene_symbol", drop=False) if len(primary_coloc_idx) else pd.DataFrame()
    else:
        primary_coloc_best = pd.DataFrame()

    secondary_coloc = coloc.loc[coloc["outcome_resource"].eq("gcst90275127_multi_ancestry_pd")].copy() if "outcome_resource" in coloc.columns else pd.DataFrame()
    if not secondary_coloc.empty:
        secondary_coloc = secondary_coloc.loc[np.isfinite(secondary_coloc["PP_H4_proxy"])].copy()
        secondary_coloc_idx = secondary_coloc.groupby("gene_symbol", observed=False)["PP_H4_proxy"].idxmax() if not secondary_coloc.empty else pd.Index([])
        secondary_coloc_idx = pd.Index(secondary_coloc_idx).dropna()
        secondary_coloc_best = secondary_coloc.loc[secondary_coloc_idx].set_index("gene_symbol", drop=False) if len(secondary_coloc_idx) else pd.DataFrame()
    else:
        secondary_coloc_best = pd.DataFrame()

    rows: list[dict[str, object]] = []
    for gene_row in gene_universe.itertuples(index=False):
        inst_row = inst.loc[gene_row.gene_symbol] if gene_row.gene_symbol in inst.index else pd.Series(dtype=object)
        primary_row = primary_best.loc[gene_row.gene_symbol] if gene_row.gene_symbol in primary_best.index else pd.Series(dtype=object)
        coloc_row = primary_coloc_best.loc[gene_row.gene_symbol] if gene_row.gene_symbol in primary_coloc_best.index else pd.Series(dtype=object)
        secondary_row = secondary_best.loc[gene_row.gene_symbol] if gene_row.gene_symbol in secondary_best.index else pd.Series(dtype=object)
        secondary_coloc_row = secondary_coloc_best.loc[gene_row.gene_symbol] if gene_row.gene_symbol in secondary_coloc_best.index else pd.Series(dtype=object)
        status = str(inst_row.get("instrumentability_status", "non_instrumentable"))
        mr_p_adj = float(primary_row.get("mr_p_adj_fdr", np.nan))
        mr_p_value = float(primary_row.get("mr_p_value", np.nan))
        mr_beta = float(primary_row.get("mr_beta", np.nan))
        mean_F = float(primary_row.get("mean_F_stat", np.nan))
        pp_h4 = float(coloc_row.get("PP_H4_proxy", np.nan))
        secondary_mr_p_adj = float(secondary_row.get("mr_p_adj_fdr", np.nan))
        secondary_mr_p_value = float(secondary_row.get("mr_p_value", np.nan))
        secondary_mr_beta = float(secondary_row.get("mr_beta", np.nan))
        secondary_pp_h4 = float(secondary_coloc_row.get("PP_H4_proxy", np.nan))
        primary_strong = np.isfinite(mr_p_adj) and mr_p_adj <= 0.10 and np.isfinite(pp_h4) and pp_h4 >= 0.80 and np.isfinite(mean_F) and mean_F > 10.0
        primary_supportive = (np.isfinite(mr_p_value) and mr_p_value <= 0.05) or (np.isfinite(pp_h4) and pp_h4 >= 0.60)
        secondary_transfer_strong = (
            np.isfinite(secondary_mr_p_adj)
            and secondary_mr_p_adj <= 0.10
            and np.isfinite(secondary_pp_h4)
            and secondary_pp_h4 >= 0.80
        )
        cross_outcome_direction_consistent = bool(
            np.isfinite(mr_beta)
            and np.isfinite(secondary_mr_beta)
            and np.sign(mr_beta) == np.sign(secondary_mr_beta)
            and np.sign(mr_beta) != 0.0
        )
        if status != "instrumentable":
            support_bin = "non_instrumentable"
        elif primary_strong:
            support_bin = "strong_genetic_support"
        elif primary_supportive and secondary_transfer_strong and cross_outcome_direction_consistent and np.isfinite(mean_F) and mean_F > 10.0:
            support_bin = "strong_genetic_support"
        elif (np.isfinite(mr_p_value) and mr_p_value <= 0.05) or (np.isfinite(pp_h4) and pp_h4 >= 0.60):
            support_bin = "moderate_genetic_support"
        else:
            support_bin = "weak_or_inconclusive"
        rows.append(
            {
                "gene_symbol": gene_row.gene_symbol,
                "gencode_id": gene_row.gencode_id,
                "program_ids_json": gene_row.program_ids_json,
                "instrumentability_status": status,
                "best_tissue": inst_row.get("best_tissue", ""),
                "best_snp_id": inst_row.get("best_snp_id", ""),
                "best_primary_mr_method": primary_row.get("method", ""),
                "best_primary_mr_p_value": mr_p_value,
                "best_primary_mr_p_adj_fdr": mr_p_adj,
                "best_primary_mr_beta": mr_beta,
                "best_primary_mean_F": mean_F,
                "best_primary_coloc_h4_proxy": pp_h4,
                "best_primary_top_shared_variant": coloc_row.get("top_shared_variant", ""),
                "best_secondary_mr_p_value": secondary_mr_p_value,
                "best_secondary_mr_p_adj_fdr": secondary_mr_p_adj,
                "best_secondary_mr_beta": secondary_mr_beta,
                "best_secondary_coloc_h4_proxy": secondary_pp_h4,
                "cross_outcome_direction_consistent": cross_outcome_direction_consistent,
                "gene_support_bin": support_bin,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "gene_symbol",
                "gencode_id",
                "program_ids_json",
                "instrumentability_status",
                "best_tissue",
                "best_snp_id",
                "best_primary_mr_method",
                "best_primary_mr_p_value",
                "best_primary_mr_p_adj_fdr",
                "best_primary_mr_beta",
                "best_primary_mean_F",
                "best_primary_coloc_h4_proxy",
                "best_primary_top_shared_variant",
                "best_secondary_mr_p_value",
                "best_secondary_mr_p_adj_fdr",
                "best_secondary_mr_beta",
                "best_secondary_coloc_h4_proxy",
                "cross_outcome_direction_consistent",
                "gene_support_bin",
            ]
        )
    return pd.DataFrame(rows).sort_values(["gene_support_bin", "best_primary_mr_p_value", "gene_symbol"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_step4_dir("02_genetics", "01_gene_level")
    output_dir.mkdir(parents=True, exist_ok=True)

    gene_universe = pd.read_csv(input_dir / "step4_gene_universe.csv")
    instrumentability = pd.read_csv(input_dir / "step4_instrumentability_table.csv")

    if gene_universe.empty:
        pd.DataFrame(
            columns=[
                "gene_symbol",
                "gencode_id",
                "tissue",
                "snp_id",
                "variantId",
                "beta_exposure",
                "p_exposure",
                "se_exposure",
                "F_stat",
                "is_selected_instrument",
                "instrument_rank",
                "chromosome",
                "position",
                "ref_allele",
                "alt_allele",
            ]
        ).to_csv(output_dir / "gtex_selected_exposure_variants.tsv", sep="\t", index=False)
        pd.DataFrame(columns=["snp_id"]).to_csv(output_dir / "finngen_step4_instrument_subset.tsv", sep="\t", index=False)
        pd.DataFrame(columns=["snp_id"]).to_csv(output_dir / "gcst90275127_step4_instrument_subset.tsv", sep="\t", index=False)
        pd.DataFrame(columns=["snp_id"]).to_csv(output_dir / "harmonized_exposure_outcome_matches.tsv", sep="\t", index=False)
        pd.DataFrame(
            columns=["gene_symbol", "tissue", "outcome_resource", "mr_beta", "mr_p_value", "mean_F_stat", "method"]
        ).to_csv(output_dir / "gene_level_mr_results.tsv", sep="\t", index=False)
        pd.DataFrame(
            columns=["gene_symbol", "tissue", "outcome_resource", "PP_H4_proxy", "top_shared_variant", "coloc_support_label"]
        ).to_csv(output_dir / "gene_level_coloc_results.tsv", sep="\t", index=False)
        build_support_summary(gene_universe, instrumentability, pd.DataFrame(), pd.DataFrame()).to_csv(
            output_dir / "gene_level_support_summary.csv",
            index=False,
        )
        return

    exposure, rsids = collect_exposures(gene_universe, instrumentability)
    exposure.to_csv(output_dir / "gtex_selected_exposure_variants.tsv", sep="\t", index=False)

    finngen_path = PROJECT_ROOT / "data/processed/step4/02_genetics/00_resources/downloads/finngen_R10_G6_PARKINSON.gz"
    gcst_path = PROJECT_ROOT / "data/processed/step4/02_genetics/00_resources/downloads/GCST90275127.tsv"
    finngen_subset = extract_outcome_subset_finngen(finngen_path, rsids, output_dir / "finngen_step4_instrument_subset.tsv")
    gcst_subset = extract_outcome_subset_gcst(gcst_path, rsids, output_dir / "gcst90275127_step4_instrument_subset.tsv")

    harmonized = pd.concat(
        [
            harmonize_exposure_outcome(exposure, finngen_subset),
            harmonize_exposure_outcome(exposure, gcst_subset),
        ],
        ignore_index=True,
    )
    harmonized.to_csv(output_dir / "harmonized_exposure_outcome_matches.tsv", sep="\t", index=False)

    mr_results = run_gene_mr(harmonized)
    if not mr_results.empty:
        mr_results["mr_p_adj_within_results"] = bh_fdr(mr_results["mr_p_value"].fillna(1.0))
    mr_results.to_csv(output_dir / "gene_level_mr_results.tsv", sep="\t", index=False)

    coloc_results = run_coloc_proxy(harmonized)
    coloc_results.to_csv(output_dir / "gene_level_coloc_results.tsv", sep="\t", index=False)

    support_summary = build_support_summary(gene_universe, instrumentability, mr_results, coloc_results)
    support_summary.to_csv(output_dir / "gene_level_support_summary.csv", index=False)


if __name__ == "__main__":
    main()
