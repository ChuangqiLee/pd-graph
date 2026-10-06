#!/usr/bin/env python3.12
"""Run program-level genetic enrichment for Step 4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step3.common import parse_gene_string
from scripts.step4.common import SHARED_GENE_CLUSTER, bh_fdr, ensure_step4_dir, read_json


DEFAULT_OUTPUT_DIR = Path("data/processed/step4/02_genetics/02_program_level")
N_PERMUTATIONS = 5000
STEP2_COMM_GENE_PANEL = Path("data/processed/step2/01_comm_panel/step2_comm_gene_panel.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 4 program-level genetic enrichment.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def load_background_genes() -> list[str]:
    panel_path = PROJECT_ROOT / STEP2_COMM_GENE_PANEL
    if panel_path.exists():
        panel = pd.read_csv(panel_path, usecols=["gene_symbol"])
        return sorted(panel["gene_symbol"].dropna().astype(str).unique().tolist())

    import anndata as ad

    adata = ad.read_h5ad(PROJECT_ROOT / "data/processed/step2/01_comm_panel/pd_definitive_step2_comm_panel.h5ad", backed="r")
    genes = adata.var_names.astype(str).tolist()
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    return genes


def collect_pair_features() -> pd.DataFrame:
    pair_universe = pd.read_parquet(PROJECT_ROOT / "data/processed/step2/03_lr_universe/step2_lr_universe_compact.parquet")
    rows: list[dict[str, object]] = []
    for row in pair_universe.itertuples(index=False):
        for gene in parse_gene_string(row.ligand_genes):
            rows.append(
                {
                    "gene_symbol": gene,
                    "sender_detectability": float(row.avg_sender_pct_expr),
                    "receiver_detectability": 0.0,
                    "support_score": float(row.support_score),
                    "degree_increment": 1,
                }
            )
        for gene in parse_gene_string(row.receptor_genes):
            rows.append(
                {
                    "gene_symbol": gene,
                    "sender_detectability": 0.0,
                    "receiver_detectability": float(row.avg_receiver_pct_expr),
                    "support_score": float(row.support_score),
                    "degree_increment": 1,
                }
            )
    feature_frame = pd.DataFrame(rows)
    feature_frame = (
        feature_frame.groupby("gene_symbol", observed=False)
        .agg(
            network_degree=("degree_increment", "sum"),
            mean_sender_detectability=("sender_detectability", "mean"),
            mean_receiver_detectability=("receiver_detectability", "mean"),
            mean_support_score=("support_score", "mean"),
        )
        .reset_index()
    )
    feature_frame["expression_detectability"] = (
        feature_frame["mean_sender_detectability"].fillna(0.0) + feature_frame["mean_receiver_detectability"].fillna(0.0)
    ) / 2.0
    return feature_frame


def scan_finngen_gene_scores(background_genes: set[str], output_dir: Path) -> pd.DataFrame:
    cache_path = output_dir / "finngen_background_gene_scores.tsv"
    if cache_path.exists():
        return pd.read_csv(cache_path, sep="\t")
    finngen_path = PROJECT_ROOT / "data/processed/step4/02_genetics/00_resources/downloads/finngen_R10_G6_PARKINSON.gz"
    aggregates: dict[str, dict[str, float]] = {
        gene: {"min_p": 1.0, "top_beta": np.nan, "n_variants": 0, "n_locus_hits": 0}
        for gene in background_genes
    }
    usecols = ["nearest_genes", "pval", "beta"]
    for chunk in pd.read_csv(finngen_path, sep="\t", compression="gzip", usecols=usecols, chunksize=250_000):
        chunk = chunk.dropna(subset=["nearest_genes"]).copy()
        chunk["nearest_genes"] = chunk["nearest_genes"].astype(str).str.split(r"[;,]")
        exploded = chunk.explode("nearest_genes")
        exploded["gene_symbol"] = exploded["nearest_genes"].astype(str).str.strip()
        exploded = exploded.loc[exploded["gene_symbol"].isin(background_genes), ["gene_symbol", "pval", "beta"]].copy()
        if exploded.empty:
            continue
        for gene_symbol, frame in exploded.groupby("gene_symbol", observed=False):
            stats = aggregates[gene_symbol]
            local_min_idx = frame["pval"].idxmin()
            local_min_p = float(frame.loc[local_min_idx, "pval"])
            if local_min_p < stats["min_p"]:
                stats["min_p"] = local_min_p
                stats["top_beta"] = float(frame.loc[local_min_idx, "beta"])
            stats["n_variants"] += int(len(frame))
            stats["n_locus_hits"] += int((pd.to_numeric(frame["pval"], errors="coerce") < 5e-8).sum())
    rows = []
    for gene_symbol, stats in aggregates.items():
        min_p = float(stats["min_p"])
        rows.append(
            {
                "gene_symbol": gene_symbol,
                "min_p": min_p,
                "gene_score_unsigned_z": float(norm.isf(max(min_p, 1e-300) / 2.0)) if np.isfinite(min_p) and min_p > 0 else float("nan"),
                "n_variants": int(stats["n_variants"]),
                "n_locus_hits": int(stats["n_locus_hits"]),
                "top_beta": stats["top_beta"],
            }
        )
    gene_scores = pd.DataFrame(rows)
    gene_scores.to_csv(cache_path, sep="\t", index=False)
    return gene_scores


def build_candidate_pools(target_genes: list[str], background: pd.DataFrame, *, pool_size: int = 60) -> dict[str, np.ndarray]:
    bg = background.set_index("gene_symbol")
    pools: dict[str, np.ndarray] = {}
    for gene in target_genes:
        row = bg.loc[gene]
        distance = (
            np.abs(bg["n_variants"] - row["n_variants"]) / max(float(row["n_variants"]), 1.0)
            + np.abs(bg["expression_detectability"] - row["expression_detectability"])
            + np.abs(bg["network_degree"] - row["network_degree"]) / max(float(row["network_degree"]), 1.0)
        )
        pools[gene] = distance.sort_values(ascending=True).head(pool_size).index.to_numpy(dtype=object)
    return pools


def matched_null_sampler(
    observed_genes: list[str],
    background: pd.DataFrame,
    candidate_pools: dict[str, np.ndarray],
    rng: np.random.Generator,
    n_permutations: int = N_PERMUTATIONS,
) -> np.ndarray:
    bg = background.set_index("gene_symbol")
    samples = np.empty(n_permutations, dtype=float)
    for perm_index in range(n_permutations):
        chosen = [str(rng.choice(candidate_pools[gene])) for gene in observed_genes]
        samples[perm_index] = float(bg.loc[chosen, "gene_score_unsigned_z"].mean())
    return samples


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_step4_dir("02_genetics", "02_program_level")
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(PROJECT_ROOT / "data/processed/step4/01_registry/step4_validation_registry.csv")
    normalized = read_json(PROJECT_ROOT / "data/processed/step4/00_inputs/step4_handoff_normalized.json")
    gene_support = pd.read_csv(PROJECT_ROOT / "data/processed/step4/02_genetics/01_gene_level/gene_level_support_summary.csv")

    if registry.empty or not normalized:
        pd.DataFrame(
            columns=[
                "program_id",
                "program_name",
                "n_genes",
                "observed_mean_gene_z",
                "background_mean_gene_z",
                "background_sd_gene_z",
                "magma_like_z",
                "magma_like_p_value",
                "locus_overlap_count",
                "member_genes_json",
                "magma_like_fdr",
            ]
        ).to_csv(output_dir / "program_magma_results.tsv", sep="\t", index=False)
        pd.DataFrame(
            columns=[
                "program_id",
                "program_name",
                "n_genes",
                "observed_mean_gene_z",
                "null_mean_gene_z",
                "null_sd_gene_z",
                "empirical_p_value",
                "empirical_enrichment_delta",
                "locus_overlap_count",
                "shared_gene_cluster_id",
                "empirical_fdr",
            ]
        ).to_csv(output_dir / "program_permutation_results.tsv", sep="\t", index=False)
        pd.DataFrame(
            columns=[
                "program_id",
                "program_name",
                "shared_gene_cluster_id",
                "n_genes",
                "n_strong_gene_support",
                "n_moderate_gene_support",
                "locus_overlap_count",
                "magma_like_p_value",
                "empirical_p_value",
                "program_genetic_support_bin",
                "magma_like_fdr",
                "empirical_fdr",
            ]
        ).to_csv(output_dir / "program_genetic_evidence_matrix.csv", index=False)
        return

    background_genes = set(load_background_genes())
    pair_features = collect_pair_features()
    gene_scores = scan_finngen_gene_scores(background_genes, output_dir)
    background = pair_features.merge(gene_scores, on="gene_symbol", how="outer").fillna(
        {
            "network_degree": 0,
            "expression_detectability": 0.0,
            "mean_support_score": 0.0,
            "min_p": 1.0,
            "gene_score_unsigned_z": 0.0,
            "n_variants": 0,
            "n_locus_hits": 0,
            "top_beta": 0.0,
        }
    )
    background = background.loc[background["gene_symbol"].isin(background_genes)].copy().reset_index(drop=True)

    magma_rows: list[dict[str, object]] = []
    perm_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(13)
    gene_support_lookup = gene_support.set_index("gene_symbol")
    background_lookup = background.set_index("gene_symbol")
    all_target_genes = sorted({gene for record in normalized for gene in record["gene_set_members_list"] if gene in background_lookup.index})
    candidate_pools = build_candidate_pools(all_target_genes, background)
    for record in normalized:
        program_id = record["program_id"]
        genes = [gene for gene in record["gene_set_members_list"] if gene in background_lookup.index]
        local = background_lookup.loc[genes].reset_index()
        observed_mean = float(local["gene_score_unsigned_z"].mean())
        background_mean = float(background["gene_score_unsigned_z"].mean())
        background_sd = float(background["gene_score_unsigned_z"].std(ddof=0))
        magma_like_z = (observed_mean - background_mean) / background_sd if background_sd > 0 else float("nan")
        magma_like_p = float(2.0 * norm.sf(abs(magma_like_z))) if np.isfinite(magma_like_z) else float("nan")
        locus_overlap_count = int((local["min_p"] < 5e-8).sum())
        magma_rows.append(
            {
                "program_id": program_id,
                "program_name": record["program_name"],
                "n_genes": int(len(genes)),
                "observed_mean_gene_z": observed_mean,
                "background_mean_gene_z": background_mean,
                "background_sd_gene_z": background_sd,
                "magma_like_z": magma_like_z,
                "magma_like_p_value": magma_like_p,
                "locus_overlap_count": locus_overlap_count,
                "member_genes_json": json.dumps(genes),
            }
        )

        perm_distribution = matched_null_sampler(genes, background, candidate_pools, rng)
        empirical_p = float((1.0 + np.sum(perm_distribution >= observed_mean)) / (1.0 + len(perm_distribution)))
        perm_rows.append(
            {
                "program_id": program_id,
                "program_name": record["program_name"],
                "n_genes": int(len(genes)),
                "observed_mean_gene_z": observed_mean,
                "null_mean_gene_z": float(np.mean(perm_distribution)),
                "null_sd_gene_z": float(np.std(perm_distribution, ddof=0)),
                "empirical_p_value": empirical_p,
                "empirical_enrichment_delta": observed_mean - float(np.mean(perm_distribution)),
                "locus_overlap_count": locus_overlap_count,
                "shared_gene_cluster_id": SHARED_GENE_CLUSTER[program_id],
            }
        )

        support_bins = local["gene_symbol"].map(lambda gene: gene_support_lookup.loc[gene, "gene_support_bin"] if gene in gene_support_lookup.index else "weak_or_inconclusive")
        n_strong = int((support_bins == "strong_genetic_support").sum())
        n_moderate = int((support_bins == "moderate_genetic_support").sum())
        if empirical_p <= 0.05 and (n_strong + n_moderate) >= 1:
            program_support = "strong_program_genetic_support"
        elif empirical_p <= 0.10 or n_moderate >= 2 or n_strong >= 1:
            program_support = "moderate_program_genetic_support"
        else:
            program_support = "weak_program_genetic_support"
        matrix_rows.append(
            {
                "program_id": program_id,
                "program_name": record["program_name"],
                "shared_gene_cluster_id": SHARED_GENE_CLUSTER[program_id],
                "n_genes": int(len(genes)),
                "n_strong_gene_support": n_strong,
                "n_moderate_gene_support": n_moderate,
                "locus_overlap_count": locus_overlap_count,
                "magma_like_p_value": magma_like_p,
                "empirical_p_value": empirical_p,
                "program_genetic_support_bin": program_support,
            }
        )

    magma = pd.DataFrame(magma_rows)
    if not magma.empty:
        magma["magma_like_fdr"] = bh_fdr(magma["magma_like_p_value"].fillna(1.0))
    magma.to_csv(output_dir / "program_magma_results.tsv", sep="\t", index=False)

    permutation = pd.DataFrame(perm_rows)
    if not permutation.empty:
        permutation["empirical_fdr"] = bh_fdr(permutation["empirical_p_value"].fillna(1.0))
    permutation.to_csv(output_dir / "program_permutation_results.tsv", sep="\t", index=False)

    matrix = pd.DataFrame(matrix_rows).merge(
        magma[["program_id", "magma_like_fdr"]],
        on="program_id",
        how="left",
    ).merge(
        permutation[["program_id", "empirical_fdr"]],
        on="program_id",
        how="left",
    )
    matrix.to_csv(output_dir / "program_genetic_evidence_matrix.csv", index=False)


if __name__ == "__main__":
    main()
