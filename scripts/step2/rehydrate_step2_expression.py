#!/usr/bin/env python3
"""Rehydrate a communication-ready expression matrix for Step 2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata.utils import make_index_unique
from sklearn.decomposition import TruncatedSVD

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import canonical_gene_symbol, load_yaml, project_path, write_json


DEFAULT_CONFIG = Path("configs/step2/step2_comm_panel.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rehydrate Step 2 communication expression matrix.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def load_alias_lookup(resource_dir: Path) -> dict[str, str]:
    alias_table = pd.read_parquet(resource_dir / "human_gene_alias_table.parquet")
    return dict(zip(alias_table["alias"], alias_table["gene_symbol"]))


def collapse_and_align_sparse(
    matrix: sp.spmatrix,
    source_gene_labels: list[str],
    global_gene_order: list[str],
) -> sp.csr_matrix:
    codes, unique_genes = pd.factorize(pd.Index(source_gene_labels), sort=False)
    collapse_matrix = sp.csr_matrix(
        (
            np.ones(len(source_gene_labels), dtype=np.float32),
            (np.arange(len(source_gene_labels)), codes),
        ),
        shape=(len(source_gene_labels), len(unique_genes)),
    )
    collapsed = matrix @ collapse_matrix
    global_lookup = {gene: idx for idx, gene in enumerate(global_gene_order)}
    align_indices = [global_lookup[gene] for gene in unique_genes]
    align_matrix = sp.csr_matrix(
        (
            np.ones(len(align_indices), dtype=np.float32),
            (np.arange(len(align_indices)), align_indices),
        ),
        shape=(len(align_indices), len(global_gene_order)),
    )
    return (collapsed @ align_matrix).tocsr()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_dir = project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resource_dir = project_path(config["paths"]["resource_dir"])

    panel_df = pd.read_csv(output_dir / "step2_comm_gene_panel.csv")
    panel_genes = panel_df["gene_symbol"].astype(str).tolist()
    panel_gene_set = set(panel_genes)
    alias_lookup = load_alias_lookup(resource_dir)
    backbone = pd.read_parquet(project_path(config["paths"]["backbone_manifest"]))

    step1_atlas = ad.read_h5ad(project_path(config["paths"]["step1_atlas"]), backed="r")
    step1_obs = step1_atlas.obs.copy()
    step1_obs.insert(0, "obs_index", step1_atlas.obs_names.astype(str))
    harmony = np.asarray(step1_atlas.obsm["X_pca_harmony"], dtype=np.float32)

    source_blocks: list[sp.csr_matrix] = []
    ordered_ids: list[str] = []
    source_recovery: dict[str, dict[str, object]] = {}

    for source_id in backbone["source_id"].astype(str).drop_duplicates().tolist():
        snapshot_path = project_path(config["paths"]["source_snapshots_dir"]) / f"{source_id}_sampled.h5ad"
        if not snapshot_path.exists():
            matches = sorted(project_path(config["paths"]["source_snapshots_dir"]).glob(f"{source_id}*_sampled.h5ad"))
            if not matches:
                raise FileNotFoundError(f"Missing Step 1 source snapshot for {source_id}")
            snapshot_path = matches[0]
        adata = ad.read_h5ad(snapshot_path, backed="r")
        snapshot_obs_names = make_index_unique(pd.Index(adata.obs_names.astype(str)))
        obs_position_lookup = pd.Series(np.arange(len(snapshot_obs_names)), index=snapshot_obs_names)

        source_backbone = backbone.loc[backbone["source_id"].astype(str) == source_id].copy()
        source_target_ids = source_backbone["cell_id"].astype(str).tolist()
        missing_ids = sorted(set(source_target_ids) - set(obs_position_lookup.index))
        if missing_ids:
            raise RuntimeError(f"Snapshot {source_id} missing {len(missing_ids)} Step 1 cells")
        row_positions = obs_position_lookup.loc[source_target_ids].to_numpy(dtype=int)

        source_var_names = pd.Index(adata.var_names.astype(str))
        source_gene_labels = [alias_lookup.get(canonical_gene_symbol(gene), canonical_gene_symbol(gene)) for gene in source_var_names]
        selected_positions = [index for index, gene in enumerate(source_gene_labels) if gene in panel_gene_set]
        selected_gene_labels = [source_gene_labels[index] for index in selected_positions]

        subset = adata[row_positions, selected_positions].to_memory()
        subset.obs_names = pd.Index(source_target_ids, name="obs_index")
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()

        X = subset.X.tocsr() if sp.issparse(subset.X) else sp.csr_matrix(subset.X)
        aligned = collapse_and_align_sparse(X, selected_gene_labels, panel_genes)
        source_blocks.append(aligned.astype(np.float32))
        ordered_ids.extend(source_target_ids)

        found_genes = sorted(set(selected_gene_labels))
        source_recovery[source_id] = {
            "n_cells": int(len(source_target_ids)),
            "n_panel_genes_found": int(len(found_genes)),
            "n_panel_genes_missing": int(len(panel_genes) - len(found_genes)),
            "missing_panel_genes_preview": [gene for gene in panel_genes if gene not in found_genes][:25],
        }

    if ordered_ids != backbone["cell_id"].astype(str).tolist():
        raise RuntimeError("Source block concatenation did not preserve Step 1 cell order")

    counts = sp.vstack(source_blocks, format="csr")
    comm_obs = step1_obs.set_index("obs_index").loc[ordered_ids].copy()
    comm = ad.AnnData(counts.copy(), obs=comm_obs, var=pd.DataFrame(index=pd.Index(panel_genes, name="gene_symbol")))
    comm.obs_names = pd.Index(ordered_ids, name="obs_index")
    comm.layers["counts"] = comm.X.copy()

    sc.pp.normalize_total(comm, target_sum=float(config["normalization"]["target_sum"]))
    comm.layers["norm"] = comm.X.copy().astype(np.float32)
    sc.pp.log1p(comm)
    comm.layers["lognorm"] = comm.X.copy().astype(np.float32)

    latent_components = min(int(config["normalization"]["comm_latent_components"]), max(2, comm.n_vars - 1))
    svd = TruncatedSVD(n_components=latent_components, random_state=int(config["random_seed"]))
    comm.obsm["X_comm_latent"] = svd.fit_transform(comm.layers["norm"]).astype(np.float32)
    comm.obsm["X_pca_harmony"] = harmony
    comm.uns["step2_comm_latent_explained_variance_ratio"] = svd.explained_variance_ratio_.astype(float).tolist()

    sentinel_status = []
    sentinel_genes = [canonical_gene_symbol(gene) for gene in config["sentinel_genes"]]
    for gene in sentinel_genes:
        sentinel_status.append(
            {
                "gene_symbol": gene,
                "present_in_panel": bool(gene in panel_gene_set),
                "present_sources": [
                    source_id
                    for source_id, payload in source_recovery.items()
                    if gene not in payload["missing_panel_genes_preview"] or gene in panel_genes
                ],
            }
        )

    qc_payload = {
        "n_cells": int(comm.n_obs),
        "n_genes": int(comm.n_vars),
        "layers": list(comm.layers.keys()),
        "obsm_keys": list(comm.obsm.keys()),
        "source_recovery": source_recovery,
        "sentinel_status": sentinel_status,
    }
    write_json(output_dir / "step2_comm_panel_qc.json", qc_payload)

    comm.write_h5ad(output_dir / "pd_definitive_step2_comm_panel.h5ad", compression="gzip")

    if hasattr(step1_atlas, "file") and step1_atlas.file is not None:
        step1_atlas.file.close()


if __name__ == "__main__":
    main()
