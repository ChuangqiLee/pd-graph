#!/usr/bin/env python3
"""External resource loading utilities for Step 2."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyreadr
import requests
from rdata import parser as rparser

from .common import canonical_gene_symbol, ensure_dir, normalize_text, safe_slug, stable_hash


def download_file(url: str, destination: str | Path) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        return destination_path
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    destination_path.write_bytes(response.content)
    return destination_path


def _read_pyreadr_object(path: Path) -> OrderedDict[Any, pd.DataFrame]:
    result = pyreadr.read_r(path)
    return result


def load_nichenet_lr_network(path: str | Path) -> pd.DataFrame:
    result = _read_pyreadr_object(Path(path))
    frame = result[next(iter(result.keys()))].copy()
    frame.columns = [str(column) for column in frame.columns]
    return frame


def load_nichenet_geneinfo(path: str | Path) -> pd.DataFrame:
    result = _read_pyreadr_object(Path(path))
    key = next(iter(result.keys()))
    frame = result[key].copy()
    frame.columns = [str(column) for column in frame.columns]
    return frame


def load_nichenet_gene_aliases(path: str | Path) -> pd.DataFrame:
    result = _read_pyreadr_object(Path(path))
    key = next(iter(result.keys()))
    frame = result[key].copy()
    frame.columns = [str(column) for column in frame.columns]
    return frame


def _decode_r_char(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_r_pairlist(obj: Any) -> dict[str, Any]:
    items: dict[str, Any] = {}
    current = obj
    while current is not None and current.info.type != rparser.RObjectType.NILVALUE:
        key = _decode_r_object(current.tag) if current.tag is not None else None
        value = _decode_r_object(current.value[0])
        if key is None:
            key = f"item_{len(items)}"
        items[str(key)] = value
        current = current.value[1]
    return items


def _is_special_row_names(values: list[Any]) -> bool:
    if len(values) != 2:
        return False
    first, second = values
    try:
        second_int = int(second)
    except (TypeError, ValueError):
        return False
    if second_int >= 0:
        return False
    if first is None:
        return True
    try:
        first_int = int(first)
    except (TypeError, ValueError):
        return False
    return first_int < 0


def _decode_r_dataframe(obj: Any) -> pd.DataFrame:
    attrs = _decode_r_attributes(obj.attributes)
    columns = [str(column) for column in attrs.get("names", [])]
    values = [_decode_r_object(value) for value in obj.value]
    data: dict[str, list[Any]] = {}
    target_length = None
    for column_name, column_value in zip(columns, values):
        if isinstance(column_value, list):
            data[column_name] = column_value
            if target_length is None:
                target_length = len(column_value)
        else:
            data[column_name] = [column_value]
            if target_length is None:
                target_length = 1
    frame = pd.DataFrame(data)
    row_names = attrs.get("row.names")
    if isinstance(row_names, list) and not _is_special_row_names(row_names) and len(row_names) == len(frame):
        frame.index = pd.Index([normalize_text(value) for value in row_names], name=None)
    return frame


def _decode_r_attributes(obj: Any) -> dict[str, Any]:
    if obj is None or obj.info.type == rparser.RObjectType.NILVALUE:
        return {}
    return _decode_r_pairlist(obj)


def _decode_r_object(obj: Any) -> Any:
    if obj is None:
        return None
    obj_type = obj.info.type
    if obj_type == rparser.RObjectType.REF:
        return _decode_r_object(obj.referenced_object)
    if obj_type == rparser.RObjectType.CHAR:
        return _decode_r_char(obj.value)
    if obj_type == rparser.RObjectType.SYM:
        return _decode_r_object(obj.value)
    if obj_type == rparser.RObjectType.NILVALUE:
        return None
    if obj_type == rparser.RObjectType.STR:
        return [_decode_r_object(value) for value in obj.value]
    if obj_type == rparser.RObjectType.INT:
        values = np.asarray(obj.value).tolist()
        return [None if value is None else int(value) for value in values]
    if obj_type == rparser.RObjectType.REAL:
        values = np.asarray(obj.value).tolist()
        return [None if value is None or (isinstance(value, float) and np.isnan(value)) else float(value) for value in values]
    if obj_type == rparser.RObjectType.LGL:
        values = np.asarray(obj.value).tolist()
        return [None if value is None else bool(value) for value in values]
    if obj_type == rparser.RObjectType.LIST:
        return _decode_r_pairlist(obj)
    if obj_type == rparser.RObjectType.VEC:
        attrs = _decode_r_attributes(obj.attributes)
        class_attr = attrs.get("class", [])
        if isinstance(class_attr, str):
            class_attr = [class_attr]
        if "data.frame" in class_attr:
            return _decode_r_dataframe(obj)
        values = [_decode_r_object(value) for value in obj.value]
        names = attrs.get("names")
        if isinstance(names, list) and len(names) == len(values):
            return {str(name): value for name, value in zip(names, values)}
        return values
    return obj.value


def load_cellchat_database(path: str | Path) -> dict[str, pd.DataFrame]:
    parsed = rparser.parse_file(Path(path))
    decoded = _decode_r_object(parsed.object)
    if isinstance(decoded, dict) and "CellChatDB.human" in decoded:
        decoded = decoded["CellChatDB.human"]
    if not isinstance(decoded, dict):
        raise RuntimeError(f"Unexpected CellChat database structure in {path}")
    required = {"interaction", "complex", "cofactor", "geneInfo"}
    if not required.issubset(decoded.keys()):
        raise RuntimeError(f"CellChat database missing expected tables: {required - set(decoded.keys())}")
    return {
        "interaction": decoded["interaction"].copy(),
        "complex": decoded["complex"].copy(),
        "cofactor": decoded["cofactor"].copy(),
        "geneInfo": decoded["geneInfo"].copy(),
    }


def build_human_gene_alias_table(
    geneinfo_human: pd.DataFrame,
    geneinfo_alias_human: pd.DataFrame,
    manual_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    manual_aliases = manual_aliases or {}
    rows: list[dict[str, str]] = []

    for symbol in geneinfo_human["symbol"].astype(str).dropna().unique().tolist():
        symbol_upper = canonical_gene_symbol(symbol)
        if symbol_upper:
            rows.append({"alias": symbol_upper, "gene_symbol": symbol_upper, "mapping_source": "identity"})

    for row in geneinfo_alias_human[["alias", "symbol"]].dropna().itertuples(index=False):
        alias_upper = canonical_gene_symbol(row.alias)
        symbol_upper = canonical_gene_symbol(row.symbol)
        if alias_upper and symbol_upper:
            rows.append({"alias": alias_upper, "gene_symbol": symbol_upper, "mapping_source": "nichenet_alias"})

    for alias, gene_symbol in manual_aliases.items():
        alias_upper = canonical_gene_symbol(alias)
        symbol_upper = canonical_gene_symbol(gene_symbol)
        if alias_upper and symbol_upper:
            rows.append({"alias": alias_upper, "gene_symbol": symbol_upper, "mapping_source": "manual_alias"})

    alias_table = pd.DataFrame(rows)
    alias_table = alias_table.drop_duplicates(subset=["alias"], keep="first").sort_values(["alias", "gene_symbol"])
    return alias_table.reset_index(drop=True)


def _complex_lookup(complex_df: pd.DataFrame) -> dict[str, list[str]]:
    if complex_df.empty:
        return {}
    subunit_columns = [column for column in complex_df.columns if str(column).startswith("subunit")]
    lookup: dict[str, list[str]] = {}
    for complex_name, row in complex_df.iterrows():
        genes = [canonical_gene_symbol(row[column]) for column in subunit_columns if normalize_text(row[column])]
        genes = [gene for gene in genes if gene]
        if genes:
            lookup[normalize_text(complex_name)] = genes
    return lookup


def _resolve_cellchat_components(value: Any, complex_lookup: dict[str, list[str]]) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    if text in complex_lookup:
        return complex_lookup[text]
    genes = [canonical_gene_symbol(part) for part in text.replace(";", "|").replace(",", "|").split("|")]
    genes = [gene for gene in genes if gene]
    return genes or [canonical_gene_symbol(text)]


def parse_cellchat_lr_pairs(cellchat_db: dict[str, pd.DataFrame]) -> pd.DataFrame:
    interaction = cellchat_db["interaction"].copy()
    interaction.columns = [str(column) for column in interaction.columns]
    complex_lookup = _complex_lookup(cellchat_db["complex"])

    rows: list[dict[str, Any]] = []
    for row in interaction.itertuples(index=False):
        record = row._asdict()
        ligand_name = normalize_text(record.get("ligand"))
        receptor_name = normalize_text(record.get("receptor"))
        ligand_genes = _resolve_cellchat_components(ligand_name, complex_lookup)
        receptor_genes = _resolve_cellchat_components(receptor_name, complex_lookup)
        if not ligand_genes or not receptor_genes:
            continue
        rows.append(
            {
                "resource_pair_id": f"cellchat::{stable_hash(f'{ligand_name}::{receptor_name}::{record.get('pathway_name', '')}')}",
                "database": "cellchat",
                "source_database": "CellChatDB.human",
                "source": "CellChat",
                "pathway_name": normalize_text(record.get("pathway_name")),
                "annotation": normalize_text(record.get("annotation")),
                "evidence": normalize_text(record.get("evidence")),
                "original_ligand": ligand_name,
                "original_receptor": receptor_name,
                "ligand_genes": "|".join(ligand_genes),
                "receptor_genes": "|".join(receptor_genes),
                "ligand_multimeric": len(ligand_genes) > 1,
                "receptor_multimeric": len(receptor_genes) > 1,
                "agonist": normalize_text(record.get("agonist")),
                "antagonist": normalize_text(record.get("antagonist")),
                "co_a_receptor": normalize_text(record.get("co_A_receptor")),
                "co_i_receptor": normalize_text(record.get("co_I_receptor")),
            }
        )

    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["original_ligand", "original_receptor", "pathway_name"])
    return frame.reset_index(drop=True)


def parse_nichenet_lr_pairs(lr_network: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in lr_network.to_dict(orient="records"):
        ligand = canonical_gene_symbol(record.get("from"))
        receptor = canonical_gene_symbol(record.get("to"))
        if not ligand or not receptor:
            continue
        source_database = normalize_text(record.get("database", "")) or "nichenet"
        source_name = normalize_text(record.get("source", "")) or "NicheNet"
        rows.append(
            {
                "resource_pair_id": f"nichenet::{stable_hash(f'{ligand}::{receptor}::{source_database}::{source_name}')}",
                "database": "nichenet",
                "source_database": source_database,
                "source": source_name,
                "pathway_name": "",
                "annotation": "",
                "evidence": source_database,
                "original_ligand": ligand,
                "original_receptor": receptor,
                "ligand_genes": ligand,
                "receptor_genes": receptor,
                "ligand_multimeric": False,
                "receptor_multimeric": False,
                "agonist": "",
                "antagonist": "",
                "co_a_receptor": "",
                "co_i_receptor": "",
            }
        )
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["original_ligand", "original_receptor", "source_database", "source"])
    return frame.reset_index(drop=True)


def ensure_step2_resources(config: dict[str, Any]) -> dict[str, Path]:
    resource_dir = ensure_dir(config["paths"]["resource_dir"])
    paths = {
        name: download_file(url, resource_dir / Path(url).name)
        for name, url in config["resources"].items()
    }
    return paths
