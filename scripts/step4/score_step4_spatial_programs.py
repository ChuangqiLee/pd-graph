#!/usr/bin/env python3.12
"""Build anchor-level and program-level spatial communication scores for Step 4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import write_markdown
from scripts.step4.common import ensure_step4_dir, geometric_mean, read_json


DEFAULT_OUTPUT_DIR = Path("data/processed/step4/03_spatial/03_anchor_scoring")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Step 4 programs in primary spatial data.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def dense_array(matrix: sp.spmatrix | np.ndarray) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def minmax_scale(series: pd.Series) -> pd.Series:
    min_value = float(series.min())
    max_value = float(series.max())
    if max_value <= min_value:
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    return (series - min_value) / (max_value - min_value)


def module_score(expr_scaled: pd.DataFrame, genes: list[str], *, method: str = "geometric") -> pd.Series:
    present = [gene for gene in genes if gene in expr_scaled.columns]
    if not present:
        return pd.Series(np.zeros(len(expr_scaled), dtype=float), index=expr_scaled.index)
    matrix = expr_scaled[present].to_numpy(dtype=float).T
    if matrix.shape[0] == 1 or method == "mean":
        values = matrix.mean(axis=0)
    else:
        values = geometric_mean(matrix + 1e-6)
    return pd.Series(values, index=expr_scaled.index)


def build_neighbor_maps(obs: pd.DataFrame, rng: np.random.Generator) -> dict[str, dict[str, np.ndarray]]:
    neighbors: dict[str, dict[str, np.ndarray]] = {}
    for sample_accession, index in obs.groupby("sample_accession", observed=False).groups.items():
        local = obs.loc[index].copy()
        coords = local[["imageX", "imageY"]].to_numpy(dtype=float)
        n_neighbors = min(7, len(local))
        model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        model.fit(coords)
        distances, nn_index = model.kneighbors(coords)
        if n_neighbors > 1:
            distances = distances[:, 1:]
            nn_index = nn_index[:, 1:]
        weights = 1.0 / np.maximum(distances, 1.0)
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
        random_index = rng.integers(0, len(local), size=nn_index.shape)
        neighbors[str(sample_accession)] = {
            "spot_ids": local.index.to_numpy(dtype=object),
            "neighbor_index": nn_index,
            "neighbor_weights": weights,
            "random_neighbor_index": random_index,
        }
    return neighbors


def zscore(values: pd.Series) -> pd.Series:
    std = float(values.std(ddof=0))
    if std == 0.0 or np.isnan(std):
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - float(values.mean())) / std


def main() -> None:
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - runtime environment guard
        raise RuntimeError(
            "score_step4_spatial_programs.py must run in the host python3.12 environment "
            "with anndata installed; do not execute it inside the container."
        ) from exc

    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_step4_dir("03_spatial", "03_anchor_scoring")
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(PROJECT_ROOT / "data/processed/step4/01_registry/step4_validation_registry.csv")
    normalized = read_json(PROJECT_ROOT / "data/processed/step4/00_inputs/step4_handoff_normalized.json")
    visium = ad.read_h5ad(PROJECT_ROOT / "data/processed/step4/03_spatial/01_primary_visium/primary_visium_filtered.h5ad")
    state_scores = pd.read_parquet(PROJECT_ROOT / "data/processed/step4/03_spatial/02_state_mapping/primary_visium_state_scores.parquet")
    state_scores = state_scores.set_index("spot_id")

    step4_genes = sorted({gene for record in normalized for gene in record["gene_set_members_list"]})
    present_genes = [gene for gene in step4_genes if gene in visium.var_names]
    expr = pd.DataFrame(dense_array(visium[:, present_genes].X), index=visium.obs_names, columns=present_genes)
    expr_scaled = expr.apply(minmax_scale, axis=0)

    obs = visium.obs[
        [
            "spot_id",
            "sample_accession",
            "donor_id",
            "disease_group",
            "sex",
            "age_years",
            "anatomical_compartment",
            "imageX",
            "imageY",
            "da_combined_score",
        ]
    ].copy()
    obs.index = obs["spot_id"].astype(str)
    rng = np.random.default_rng(13)
    neighbor_maps = build_neighbor_maps(obs, rng)

    anchor_rows: list[pd.DataFrame] = []
    program_rows: list[pd.DataFrame] = []
    for record in normalized:
        sender_state = state_scores[f"primary_score_01__{record['sender_state_id']}"].astype(float)
        receiver_state = state_scores[f"primary_score_01__{record['receiver_state_id']}"].astype(float)
        valid_anchor_frames: list[pd.DataFrame] = []
        for anchor in record["anchor_pairs_json"]:
            sender_anchor = module_score(expr_scaled, list(anchor["sender_genes"]), method="geometric")
            receiver_anchor = module_score(expr_scaled, list(anchor["receiver_genes"]), method="geometric")
            same_spot = sender_state * receiver_state * sender_anchor * receiver_anchor
            adjacent = pd.Series(np.zeros(len(obs), dtype=float), index=obs.index)
            adjacent_random = pd.Series(np.zeros(len(obs), dtype=float), index=obs.index)
            receiver_combo = receiver_state * receiver_anchor
            for sample_accession, info in neighbor_maps.items():
                local_spot_ids = pd.Index(info["spot_ids"])
                nn_index = info["neighbor_index"]
                nn_weights = info["neighbor_weights"]
                rand_index = info["random_neighbor_index"]
                receiver_values = receiver_combo.loc[local_spot_ids].to_numpy(dtype=float)
                sender_values = (sender_state * sender_anchor).loc[local_spot_ids].to_numpy(dtype=float)
                neighbor_receiver = (receiver_values[nn_index] * nn_weights).sum(axis=1)
                random_receiver = (receiver_values[rand_index] * nn_weights).sum(axis=1)
                adjacent.loc[local_spot_ids] = sender_values * neighbor_receiver
                adjacent_random.loc[local_spot_ids] = sender_values * random_receiver
            anchor_frame = obs.copy()
            anchor_frame["program_id"] = record["program_id"]
            anchor_frame["program_name"] = record["program_name"]
            anchor_frame["pair_key"] = anchor["pair_key"]
            anchor_frame["anchor_class"] = anchor["anchor_class"]
            anchor_frame["claim_language_level"] = anchor["claim_language_level"]
            anchor_frame["sender_state_score"] = sender_state.loc[anchor_frame.index].to_numpy(dtype=float)
            anchor_frame["receiver_state_score"] = receiver_state.loc[anchor_frame.index].to_numpy(dtype=float)
            anchor_frame["sender_anchor_score"] = sender_anchor.loc[anchor_frame.index].to_numpy(dtype=float)
            anchor_frame["receiver_anchor_score"] = receiver_anchor.loc[anchor_frame.index].to_numpy(dtype=float)
            anchor_frame["same_spot_pair_score"] = same_spot.loc[anchor_frame.index].to_numpy(dtype=float)
            anchor_frame["adjacent_pair_score"] = adjacent.loc[anchor_frame.index].to_numpy(dtype=float)
            anchor_frame["random_adjacent_pair_score"] = adjacent_random.loc[anchor_frame.index].to_numpy(dtype=float)
            anchor_frame["adjacency_excess_score"] = anchor_frame["adjacent_pair_score"] - anchor_frame["random_adjacent_pair_score"]
            anchor_frame["same_spot_pair_score_z"] = zscore(anchor_frame["same_spot_pair_score"]).to_numpy(dtype=float)
            anchor_frame["adjacent_pair_score_z"] = zscore(anchor_frame["adjacent_pair_score"]).to_numpy(dtype=float)
            anchor_frame["adjacency_excess_score_z"] = zscore(anchor_frame["adjacency_excess_score"]).to_numpy(dtype=float)
            anchor_rows.append(anchor_frame)
            if anchor["claim_language_level"] != "program_only":
                valid_anchor_frames.append(anchor_frame)

        if valid_anchor_frames:
            stacked = pd.concat(valid_anchor_frames, ignore_index=True)
            grouped = (
                stacked.groupby("spot_id", observed=False)
                .agg(
                    valid_anchor_count=("pair_key", "nunique"),
                    same_spot_program_score=("same_spot_pair_score_z", "mean"),
                    adjacent_program_score=("adjacent_pair_score_z", "mean"),
                    adjacency_excess_score=("adjacency_excess_score_z", "mean"),
                    random_adjacent_program_score=("random_adjacent_pair_score", "mean"),
                )
                .reset_index()
                .set_index("spot_id")
            )
        else:
            grouped = pd.DataFrame(index=obs.index)
            grouped["valid_anchor_count"] = 0
            grouped["same_spot_program_score"] = 0.0
            grouped["adjacent_program_score"] = 0.0
            grouped["adjacency_excess_score"] = 0.0
            grouped["random_adjacent_program_score"] = 0.0
        program_frame = obs.copy()
        program_frame["program_id"] = record["program_id"]
        program_frame["program_name"] = record["program_name"]
        program_frame["expected_pd_direction"] = record["expected_pd_direction"]
        program_frame["sender_state_id"] = record["sender_state_id"]
        program_frame["receiver_state_id"] = record["receiver_state_id"]
        program_frame["valid_anchor_count"] = grouped.loc[program_frame.index, "valid_anchor_count"].to_numpy(dtype=int)
        program_frame["same_spot_program_score"] = grouped.loc[program_frame.index, "same_spot_program_score"].to_numpy(dtype=float)
        program_frame["adjacent_program_score"] = grouped.loc[program_frame.index, "adjacent_program_score"].to_numpy(dtype=float)
        program_frame["random_adjacent_program_score"] = grouped.loc[program_frame.index, "random_adjacent_program_score"].to_numpy(dtype=float)
        program_frame["adjacency_excess_score"] = grouped.loc[program_frame.index, "adjacency_excess_score"].to_numpy(dtype=float)
        program_frame["state_coenrichment_score"] = (sender_state * receiver_state).loc[program_frame.index].to_numpy(dtype=float)
        program_frame["program_module_score"] = module_score(expr_scaled, record["gene_set_members_list"], method="mean").loc[program_frame.index].to_numpy(dtype=float)
        program_frame["program_module_context_score"] = program_frame["state_coenrichment_score"] * program_frame["program_module_score"]
        program_rows.append(program_frame)

    anchor_scores = pd.concat(anchor_rows, ignore_index=True)
    anchor_scores.to_parquet(output_dir / "anchor_pair_spatial_scores.parquet", index=False)

    program_scores = pd.concat(program_rows, ignore_index=True)
    program_scores.to_csv(output_dir / "program_spatial_scores.csv", index=False)

    notes = [
        "# Spatial Score Formula Notes",
        "",
        "The primary Step 4 spatial score uses two related quantities for every anchor and program:",
        "",
        "1. `same_spot_pair_score = sender_state * receiver_state * sender_anchor * receiver_anchor`",
        "2. `adjacent_pair_score = sender_state * sender_anchor * weighted_neighbor_mean(receiver_state * receiver_anchor)`",
        "3. `adjacency_excess_score = adjacent_pair_score - random_adjacent_pair_score`",
        "",
        "Anchor rows marked `program_only` are retained in the audit table but excluded from the anchor-level aggregation that produces the per-program spatial score.",
        "",
    ]
    write_markdown(output_dir / "spatial_score_formula_notes.md", "\n".join(notes))


if __name__ == "__main__":
    main()
