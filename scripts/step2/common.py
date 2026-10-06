#!/usr/bin/env python3
"""Shared helpers for Step 2 pipeline scripts."""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

if TYPE_CHECKING:
    import anndata as ad


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEP1_ATLAS_PATH = PROJECT_ROOT / "data/processed/step1b/pd_definitive_first_pass_atlas.h5ad"
STEP1_SOURCE_SNAPSHOT_DIR = PROJECT_ROOT / "data/processed/step1b/tmp_step1b_sources"
STEP2_ROOT = PROJECT_ROOT / "data/processed/step2"

MISSING_ID_VALUES = {"", "na", "nan", "none", "null", "unknown", "unassigned"}


def project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path_like: str | Path) -> dict[str, Any]:
    path = project_path(path_like)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path_like: str | Path) -> Path:
    path = project_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def safe_slug(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def stable_hash(value: str, *, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def is_informative_id(value: Any) -> bool:
    text = normalize_text(value).lower()
    return text not in MISSING_ID_VALUES


def infer_graph_unit_id(row: pd.Series) -> str:
    source_id = normalize_text(row.get("source_id"))
    sample_id = normalize_text(row.get("sample_id"))
    donor_id = normalize_text(row.get("donor_id"))
    if is_informative_id(sample_id):
        chosen = sample_id
    elif is_informative_id(donor_id):
        chosen = donor_id
    else:
        chosen = normalize_text(row.get("cell_id")) or "unknown_cell"
    return f"{source_id}::{chosen}"


def infer_graph_unit_level(row: pd.Series) -> str:
    sample_id = normalize_text(row.get("sample_id"))
    donor_id = normalize_text(row.get("donor_id"))
    if is_informative_id(sample_id):
        return "sample"
    if is_informative_id(donor_id):
        return "donor_fallback"
    return "cell_fallback"


def infer_split_group_id(row: pd.Series) -> str:
    source_id = normalize_text(row.get("source_id"))
    donor_id = normalize_text(row.get("donor_id"))
    sample_id = normalize_text(row.get("sample_id"))
    if is_informative_id(donor_id):
        chosen = donor_id
    elif is_informative_id(sample_id):
        chosen = sample_id
    else:
        chosen = normalize_text(row.get("cell_id")) or "unknown_cell"
    return f"{source_id}::{chosen}"


def infer_split_group_level(row: pd.Series) -> str:
    donor_id = normalize_text(row.get("donor_id"))
    sample_id = normalize_text(row.get("sample_id"))
    if is_informative_id(donor_id):
        return "donor"
    if is_informative_id(sample_id):
        return "sample_fallback"
    return "cell_fallback"


def log(message: str) -> None:
    print(message, flush=True)


def write_json(path_like: str | Path, payload: Any) -> Path:
    path = project_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return path


def write_markdown(path_like: str | Path, text: str) -> Path:
    path = project_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def read_table(path_like: str | Path) -> pd.DataFrame:
    path = project_path(path_like)
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            fallback = path.with_suffix(".csv")
            if fallback.exists():
                return pd.read_csv(fallback)
            raise
    if path.suffix in {".csv", ".tsv"}:
        sep = "\t" if path.suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(df: pd.DataFrame, path_like: str | Path, *, index: bool = False) -> Path:
    path = project_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=index)
            return path
        except Exception:
            fallback = path.with_suffix(".csv")
            df.to_csv(fallback, index=index)
            return fallback
    if path.suffix in {".csv", ".tsv"}:
        sep = "\t" if path.suffix == ".tsv" else ","
        df.to_csv(path, index=index, sep=sep)
        return path
    raise ValueError(f"Unsupported table format: {path}")


def save_pickle_gz(path_like: str | Path, payload: Any) -> Path:
    path = project_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


class CompatUnpickler(pickle.Unpickler):
    MODULE_REMAP = {
        "numpy._core": "numpy.core",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.umath": "numpy.core.umath",
    }

    def find_class(self, module: str, name: str) -> Any:
        mapped = self.MODULE_REMAP.get(module, module)
        if mapped == module and module.startswith("numpy._core."):
            mapped = module.replace("numpy._core.", "numpy.core.", 1)
        return super().find_class(mapped, name)


def load_pickle_gz(path_like: str | Path) -> Any:
    path = project_path(path_like)
    with gzip.open(path, "rb") as handle:
        return CompatUnpickler(handle).load()


def canonical_gene_symbol(symbol: Any) -> str:
    text = normalize_text(symbol)
    if not text:
        return ""
    return text.upper()


def stacked_sparse_geometric_mean(left: np.ndarray, right: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return np.sqrt(np.maximum(left, eps) * np.maximum(right, eps))


def ensure_obs_names_unique(adata: "ad.AnnData") -> "ad.AnnData":
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()
        if "cell_id" in adata.obs.columns:
            adata.obs["cell_id"] = adata.obs_names.astype(str)
    return adata


def read_step1_atlas(*, backed: str | None = None) -> "ad.AnnData":
    import anndata as ad

    return ad.read_h5ad(STEP1_ATLAS_PATH, backed=backed)


def step1_obs_and_harmony() -> tuple[pd.DataFrame, np.ndarray]:
    import anndata as ad

    atlas = ad.read_h5ad(STEP1_ATLAS_PATH)
    obs = atlas.obs.copy()
    harmony = np.asarray(atlas.obsm["X_pca_harmony"], dtype=np.float32)
    return obs, harmony


def resolve_source_snapshot_path(source_id: str) -> Path:
    candidates = sorted(STEP1_SOURCE_SNAPSHOT_DIR.glob(f"{source_id}*_sampled.h5ad"))
    if not candidates:
        raise FileNotFoundError(f"Missing Step 1 source snapshot for {source_id}")
    if len(candidates) > 1:
        raise RuntimeError(f"Ambiguous Step 1 source snapshots for {source_id}: {candidates}")
    return candidates[0]


def read_source_snapshot(source_id: str, *, backed: str | None = None) -> "ad.AnnData":
    import anndata as ad

    adata = ad.read_h5ad(resolve_source_snapshot_path(source_id), backed=backed)
    if backed is None:
        return ensure_obs_names_unique(adata)
    if not adata.obs_names.is_unique:
        # The b0f0 snapshot needs the same repair logic as the final atlas.
        repaired = adata.to_memory()
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()
        return ensure_obs_names_unique(repaired)
    return adata


def iter_source_ids_from_atlas() -> list[str]:
    import anndata as ad

    atlas = ad.read_h5ad(STEP1_ATLAS_PATH, backed="r")
    source_ids = sorted(atlas.obs["source_id"].astype(str).unique().tolist())
    if hasattr(atlas, "file") and atlas.file is not None:
        atlas.file.close()
    return source_ids


def dense_vector_from_sparse_column(matrix: sp.spmatrix | np.ndarray, column_index: int) -> np.ndarray:
    if sp.issparse(matrix):
        return np.asarray(matrix[:, column_index].toarray()).ravel()
    return np.asarray(matrix[:, column_index]).ravel()


def dense_rows(matrix: sp.spmatrix | np.ndarray, rows: np.ndarray, cols: np.ndarray | None = None) -> np.ndarray:
    if cols is None:
        subset = matrix[rows]
    else:
        subset = matrix[rows][:, cols]
    if sp.issparse(subset):
        return subset.toarray()
    return np.asarray(subset)


def categorical_counts(values: Iterable[Any]) -> dict[str, int]:
    series = pd.Series(list(values), dtype="string").fillna("")
    counts = series.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    table = df.copy()
    table.columns = [str(column) for column in table.columns]
    header = "| " + " | ".join(table.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = []
    for row in table.astype(object).itertuples(index=False):
        rows.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join([header, sep, *rows])
