#!/usr/bin/env python3.12
"""Build lineage-specific state models for Step 3 interpretation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
import yaml
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import project_path, safe_slug
from scripts.step3.common import LINEAGE_MODULES, extract_layer_matrix, load_comm_adata, parse_gene_string, split_pair_key


DEFAULT_CONFIG = Path("configs/step3/step3_states.yaml")
LINEAGE_OUTPUTS = {
    "microglia": "step3_microglia_states.h5ad",
    "dopaminergic neuron": "step3_dopaminergic_states.h5ad",
    "astrocyte": "step3_astrocyte_states.h5ad",
    "endothelial cell": "step3_endothelial_states.h5ad",
    "neuron": "step3_neuron_states.h5ad",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step 3 lineage states.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def choose_k(features: np.ndarray, *, min_clusters: int, max_clusters: int) -> int:
    n_obs = features.shape[0]
    if n_obs <= min_clusters:
        return max(2, min_clusters)
    sample_size = min(5000, n_obs)
    rng = np.random.default_rng(13)
    sample_idx = rng.choice(n_obs, size=sample_size, replace=False)
    sample = features[sample_idx]
    best_k = min_clusters
    best_score = -1.0
    for k in range(min_clusters, min(max_clusters, n_obs - 1) + 1):
        model = MiniBatchKMeans(n_clusters=k, random_state=13, batch_size=min(2048, n_obs))
        labels = model.fit_predict(sample)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(sample, labels, metric="euclidean")
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def lineage_gene_panel(refined_map: pd.DataFrame, lineage: str) -> list[str]:
    modules = LINEAGE_MODULES.get(lineage, {})
    panel = {gene for genes in modules.values() for gene in genes}
    for pair_key in refined_map["pair_key"].astype(str):
        ligand_text, receptor_text = split_pair_key(pair_key)
        panel.update(parse_gene_string(ligand_text))
        panel.update(parse_gene_string(receptor_text))
    return sorted(panel)


def state_marker_summary(expression: pd.DataFrame, obs: pd.DataFrame, lineage: str) -> pd.DataFrame:
    modules = LINEAGE_MODULES.get(lineage, {})
    module_means = {}
    for module_name, genes in modules.items():
        present = [gene for gene in genes if gene in expression.columns]
        if not present:
            module_means[module_name] = pd.Series(0.0, index=obs.index)
        else:
            module_means[module_name] = expression[present].mean(axis=1)
    module_frame = pd.DataFrame(module_means, index=obs.index)
    rows: list[dict[str, object]] = []
    for cluster_id, index in obs.groupby("state_cluster", observed=False).groups.items():
        cluster_expr = expression.loc[index]
        cluster_modules = module_frame.loc[index].mean(axis=0).sort_values(ascending=False)
        top_label = str(cluster_modules.index[0]) if len(cluster_modules.index) else "unassigned"
        top_genes = cluster_expr.mean(axis=0).sort_values(ascending=False).head(8).index.tolist()
        rows.append(
            {
                "lineage": lineage,
                "state_cluster": int(cluster_id),
                "state_label": top_label,
                "n_cells": len(index),
                "top_module_score": float(cluster_modules.iloc[0]) if len(cluster_modules) else 0.0,
                "top_genes": "|".join(top_genes),
            }
        )
    return pd.DataFrame(rows)


def plot_umap(obs: pd.DataFrame, lineage: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for state_id, local in obs.groupby("state_id", observed=False):
        ax.scatter(local["umap_1"], local["umap_2"], s=6, alpha=0.6, label=str(state_id))
    ax.set_title(f"{lineage} states")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    output_dir = project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "step3_state_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    refined_map = pd.read_csv(project_path("data/processed/step3/02_programs/step3_refined_program_map.csv"))
    adata = load_comm_adata(backed="r")
    all_marker_rows: list[pd.DataFrame] = []

    lineage_list = ["microglia", "dopaminergic neuron", "astrocyte", "endothelial cell", "neuron"]
    for lineage in lineage_list:
        mask = adata.obs["cell_type_standardized_final"].astype(str).eq(lineage).to_numpy()
        if int(mask.sum()) == 0:
            continue
        gene_panel = [gene for gene in lineage_gene_panel(refined_map, lineage) if gene in adata.var_names]
        obs, dense = extract_layer_matrix(adata, obs_mask=mask, gene_names=gene_panel, layer="lognorm")
        latent = np.asarray(adata.obsm["X_comm_latent"][mask], dtype=np.float32)
        expr_df = pd.DataFrame(dense, index=obs.index, columns=gene_panel)

        scaler = StandardScaler()
        latent_scaled = scaler.fit_transform(latent)
        expr_scaled = scaler.fit_transform(dense) if dense.shape[1] else np.empty((dense.shape[0], 0), dtype=np.float32)
        features = np.concatenate([latent_scaled, expr_scaled], axis=1)
        cluster_cfg = config["clustering"].get(lineage, {"min_clusters": 4, "max_clusters": 6})
        k = choose_k(features, min_clusters=int(cluster_cfg["min_clusters"]), max_clusters=int(cluster_cfg["max_clusters"]))
        model = MiniBatchKMeans(n_clusters=k, random_state=13, batch_size=min(2048, len(obs)))
        obs["state_cluster"] = model.fit_predict(features)

        marker_df = state_marker_summary(expr_df, obs, lineage)
        cluster_to_label = marker_df.set_index("state_cluster")["state_label"].to_dict()
        obs["state_label"] = obs["state_cluster"].map(cluster_to_label).astype(str)
        obs["state_id"] = [
            f"{safe_slug(lineage)}_state_{int(cluster):02d}_{safe_slug(label)}"
            for cluster, label in zip(obs["state_cluster"], obs["state_label"])
        ]

        reducer = umap.UMAP(
            n_neighbors=int(config["umap"]["n_neighbors"]),
            min_dist=float(config["umap"]["min_dist"]),
            random_state=int(config["umap"]["random_state"]),
        )
        coords = reducer.fit_transform(latent_scaled)
        obs["umap_1"] = coords[:, 0]
        obs["umap_2"] = coords[:, 1]

        out = ad.AnnData(X=dense.astype(np.float32), obs=obs.copy(), var=pd.DataFrame(index=gene_panel))
        out.obs_names = obs["cell_id"].astype(str)
        out.obsm["X_comm_latent"] = latent
        out.obsm["X_umap"] = coords.astype(np.float32)
        out.write_h5ad(output_dir / LINEAGE_OUTPUTS[lineage])

        marker_df["state_id"] = marker_df["state_cluster"].map(obs.groupby("state_cluster", observed=False)["state_id"].first().to_dict())
        all_marker_rows.append(marker_df)
        plot_umap(obs, lineage, plot_dir / f"{safe_slug(lineage)}_states_umap.png")

    if all_marker_rows:
        pd.concat(all_marker_rows, ignore_index=True).to_csv(output_dir / "step3_state_marker_summary.csv", index=False)


if __name__ == "__main__":
    main()
