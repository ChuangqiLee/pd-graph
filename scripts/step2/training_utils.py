#!/usr/bin/env python3
"""Shared training utilities for Step 2 model fitting and calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train_groups: tuple[str, ...]
    val_groups: tuple[str, ...]


@dataclass(frozen=True)
class InnerSplit:
    fit_groups: tuple[str, ...]
    calibration_groups: tuple[str, ...]


def donor_table(
    frame: pd.DataFrame,
    *,
    group_col: str = "split_group_id",
    source_col: str = "source_id",
    target_col: str | None = None,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    keep_cols = [group_col]
    if source_col in frame.columns:
        keep_cols.append(source_col)
    if target_col and target_col in frame.columns:
        keep_cols.append(target_col)
    if extra_cols:
        keep_cols.extend([column for column in extra_cols if column in frame.columns])
    donor_df = frame[keep_cols].drop_duplicates(subset=[group_col]).copy()
    donor_df[group_col] = donor_df[group_col].astype(str)
    if source_col in donor_df.columns:
        donor_df[source_col] = donor_df[source_col].astype(str)
    if target_col and target_col in donor_df.columns:
        donor_df[target_col] = pd.to_numeric(donor_df[target_col], errors="coerce")
    return donor_df.sort_values(group_col).reset_index(drop=True)


def resolve_group_stratification(
    donors: pd.DataFrame,
    *,
    min_groups: int,
    stratify_col: str | None = None,
    stratify_cols: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.Series | None, tuple[str, ...]]:
    ordered: list[str] = []
    if stratify_cols:
        ordered.extend([str(column) for column in stratify_cols if str(column) not in ordered])
    if stratify_col:
        column = str(stratify_col)
        if column not in ordered:
            ordered.append(column)
    if not ordered:
        return None, tuple()

    candidate_sets: list[tuple[str, ...]] = []

    def append_candidate(columns: tuple[str, ...]) -> None:
        if columns and columns not in candidate_sets:
            candidate_sets.append(columns)

    append_candidate(tuple(ordered))
    for width in range(1, len(ordered)):
        append_candidate(tuple(ordered[:width]))
    for column in ordered:
        append_candidate((column,))

    for columns in candidate_sets:
        if any(column not in donors.columns for column in columns):
            continue
        label_frame = donors.loc[:, list(columns)].copy()
        if label_frame.empty:
            continue
        for column in label_frame.columns:
            label_frame[column] = label_frame[column].astype("string").fillna("__na__").astype(str)
        labels = label_frame.agg("::".join, axis=1)
        if labels.nunique() <= 1:
            continue
        if labels.value_counts().min() >= int(min_groups):
            return labels.astype(str), tuple(columns)
    return None, tuple()


def make_outer_group_splits(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    random_seed: int,
    group_col: str = "split_group_id",
    stratify_col: str | None = None,
    stratify_cols: list[str] | tuple[str, ...] | None = None,
    age_col: str | None = None,
    balance_age: bool = False,
) -> list[FoldSplit]:
    extra_cols = [column for column in [age_col, *(stratify_cols or []), stratify_col] if column]
    donors = donor_table(frame, group_col=group_col, target_col=None, extra_cols=extra_cols)
    if donors.empty:
        raise RuntimeError("Cannot create cross-validation splits from an empty donor table.")

    effective_splits = max(2, min(int(n_splits), len(donors)))
    stratify_labels, _ = resolve_group_stratification(
        donors,
        min_groups=effective_splits,
        stratify_col=stratify_col,
        stratify_cols=stratify_cols,
    )
    if balance_age and age_col and age_col in donors.columns and len(donors) >= effective_splits:
        age_values = pd.to_numeric(donors[age_col], errors="coerce")
        fallback_age = float(age_values.dropna().median()) if age_values.notna().any() else 0.0
        donors["_age_value"] = age_values.fillna(fallback_age).astype(float)
        donors["_jitter"] = np.random.default_rng(int(random_seed)).normal(loc=0.0, scale=1e-3, size=len(donors))
        fold_groups: list[list[str]] = [[] for _ in range(effective_splits)]
        fold_cycle = list(range(effective_splits)) + list(range(max(effective_splits - 2, 0), 0, -1))
        if not fold_cycle:
            fold_cycle = [0]
        strata = [(None, donors)]
        if stratify_labels is not None:
            strata = list(donors.assign(_stratum=stratify_labels).groupby("_stratum", observed=False))
        for _, local in strata:
            ordered = local.sort_values(["_age_value", "_jitter"], ascending=[False, False]).reset_index(drop=True)
            for idx, row in enumerate(ordered.itertuples(index=False)):
                target_fold = int(fold_cycle[idx % len(fold_cycle)])
                fold_groups[target_fold].append(str(getattr(row, group_col)))
        return [
            FoldSplit(
                fold=fold_index + 1,
                train_groups=tuple(sorted(group for idx, groups in enumerate(fold_groups) if idx != fold_index for group in groups)),
                val_groups=tuple(sorted(fold_groups[fold_index])),
            )
            for fold_index in range(effective_splits)
        ]

    split_args: list[tuple[np.ndarray, np.ndarray]] = []
    if stratify_labels is not None:
        splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=int(random_seed))
        split_args = list(splitter.split(donors[[group_col]], stratify_labels.astype(str)))
    else:
        splitter = KFold(n_splits=effective_splits, shuffle=True, random_state=int(random_seed))
        split_args = list(splitter.split(donors[[group_col]]))

    folds: list[FoldSplit] = []
    for fold_index, (train_idx, val_idx) in enumerate(split_args, start=1):
        train_groups = tuple(sorted(donors.iloc[train_idx][group_col].astype(str).tolist()))
        val_groups = tuple(sorted(donors.iloc[val_idx][group_col].astype(str).tolist()))
        folds.append(FoldSplit(fold=fold_index, train_groups=train_groups, val_groups=val_groups))
    return folds


def make_inner_group_split(
    frame: pd.DataFrame,
    *,
    random_seed: int,
    calibration_fraction: float,
    min_calibration_groups: int = 2,
    group_col: str = "split_group_id",
    stratify_col: str | None = None,
    stratify_cols: list[str] | tuple[str, ...] | None = None,
    age_col: str | None = None,
    balance_age: bool = False,
) -> InnerSplit:
    extra_cols = [column for column in [age_col, *(stratify_cols or []), stratify_col] if column]
    donors = donor_table(frame, group_col=group_col, target_col=None, extra_cols=extra_cols)
    if len(donors) < 3:
        return InnerSplit(
            fit_groups=tuple(sorted(donors[group_col].astype(str).tolist())),
            calibration_groups=tuple(),
        )

    rng = np.random.default_rng(int(random_seed))
    calibration_groups: list[str] = []
    stratify_labels, _ = resolve_group_stratification(
        donors,
        min_groups=max(2, int(min_calibration_groups)),
        stratify_col=stratify_col,
        stratify_cols=stratify_cols,
    )

    if stratify_labels is not None:
        grouped_donors = donors.assign(_stratum=stratify_labels)
        for label, label_df in grouped_donors.groupby("_stratum", observed=False):
            local_groups = label_df[group_col].astype(str).tolist()
            if len(local_groups) <= 1:
                continue
            n_calibration = int(round(len(local_groups) * float(calibration_fraction)))
            n_calibration = max(1, n_calibration)
            n_calibration = min(n_calibration, len(local_groups) - 1)
            if balance_age and age_col and age_col in label_df.columns:
                local = label_df.copy()
                local["_age_value"] = pd.to_numeric(local[age_col], errors="coerce")
                fallback_age = float(local["_age_value"].dropna().median()) if local["_age_value"].notna().any() else 0.0
                local["_age_value"] = local["_age_value"].fillna(fallback_age)
                local["_jitter"] = rng.normal(loc=0.0, scale=1e-3, size=len(local))
                ordered = local.sort_values(["_age_value", "_jitter"], ascending=[False, False]).reset_index(drop=True)
                positions = np.linspace(0, len(ordered) - 1, num=n_calibration, dtype=int)
                calibration_groups.extend(ordered.iloc[positions][group_col].astype(str).tolist())
            else:
                shuffled = list(rng.permutation(local_groups))
                calibration_groups.extend(shuffled[:n_calibration])
    else:
        local_groups = donors[group_col].astype(str).tolist()
        n_calibration = int(round(len(local_groups) * float(calibration_fraction)))
        n_calibration = max(int(min_calibration_groups), n_calibration)
        n_calibration = min(n_calibration, len(local_groups) - 1)
        if balance_age and age_col and age_col in donors.columns:
            local = donors.copy()
            local["_age_value"] = pd.to_numeric(local[age_col], errors="coerce")
            fallback_age = float(local["_age_value"].dropna().median()) if local["_age_value"].notna().any() else 0.0
            local["_age_value"] = local["_age_value"].fillna(fallback_age)
            local["_jitter"] = rng.normal(loc=0.0, scale=1e-3, size=len(local))
            ordered = local.sort_values(["_age_value", "_jitter"], ascending=[False, False]).reset_index(drop=True)
            positions = np.linspace(0, len(ordered) - 1, num=n_calibration, dtype=int)
            calibration_groups.extend(ordered.iloc[positions][group_col].astype(str).tolist())
        else:
            shuffled = list(rng.permutation(local_groups))
            calibration_groups.extend(shuffled[:n_calibration])

    calibration_groups = sorted(set(calibration_groups))
    if len(calibration_groups) < int(min_calibration_groups):
        remainder = [group for group in donors[group_col].astype(str).tolist() if group not in calibration_groups]
        for group in remainder:
            calibration_groups.append(group)
            if len(calibration_groups) >= int(min_calibration_groups):
                break

    fit_groups = sorted([group for group in donors[group_col].astype(str).tolist() if group not in calibration_groups])
    if not fit_groups:
        moved = calibration_groups.pop()
        fit_groups.append(moved)

    if stratify_labels is not None:
        donor_label_map = pd.Series(stratify_labels.to_numpy(), index=donors[group_col].astype(str)).to_dict()
        fit_labels = {str(donor_label_map[group]) for group in fit_groups if group in donor_label_map}
        calibration_labels = {str(donor_label_map[group]) for group in calibration_groups if group in donor_label_map}
        missing_labels = sorted(calibration_labels - fit_labels)
        for label in missing_labels:
            for group in list(calibration_groups):
                if str(donor_label_map.get(group, "__missing__")) != str(label):
                    continue
                class_count = sum(str(donor_label_map.get(value, "__missing__")) == str(label) for value in calibration_groups)
                if class_count <= 1:
                    continue
                calibration_groups.remove(group)
                fit_groups.append(group)
                break
        fit_groups = sorted(fit_groups)
        calibration_groups = sorted(calibration_groups)

    return InnerSplit(
        fit_groups=tuple(fit_groups),
        calibration_groups=tuple(calibration_groups),
    )


def add_balanced_sample_weights(
    frame: pd.DataFrame,
    *,
    group_col: str = "split_group_id",
    source_col: str = "source_id",
    output_col: str = "sample_weight",
    donor_balance: bool = True,
    source_balance: bool = True,
) -> pd.DataFrame:
    weighted = frame.copy()
    weights = np.ones(len(weighted), dtype=np.float64)

    if donor_balance and group_col in weighted.columns:
        group_sizes = weighted[group_col].astype(str).value_counts().to_dict()
        weights *= weighted[group_col].astype(str).map(lambda value: 1.0 / max(group_sizes.get(value, 1), 1)).to_numpy(dtype=float)

    if source_balance and source_col in weighted.columns and group_col in weighted.columns:
        donor_sources = weighted[[group_col, source_col]].drop_duplicates(subset=[group_col]).copy()
        donor_sources[group_col] = donor_sources[group_col].astype(str)
        donor_sources[source_col] = donor_sources[source_col].astype(str)
        source_group_counts = donor_sources[source_col].value_counts().to_dict()
        group_to_source = donor_sources.set_index(group_col)[source_col].to_dict()
        source_weights = weighted[group_col].astype(str).map(
            lambda value: 1.0 / max(source_group_counts.get(group_to_source.get(value, "__unk__"), 1), 1)
        )
        weights *= source_weights.to_numpy(dtype=float)

    if not np.isfinite(weights).all() or float(np.mean(weights)) <= 0.0:
        weights = np.ones(len(weighted), dtype=np.float64)
    else:
        weights /= float(np.mean(weights))

    weighted[output_col] = weights.astype(np.float32)
    return weighted


def apply_source_filters(
    frame: pd.DataFrame,
    *,
    source_col: str = "source_id",
    include_sources: list[str] | tuple[str, ...] | None = None,
    exclude_sources: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    filtered = frame.copy()
    include_set = {str(value) for value in (include_sources or []) if str(value).strip()}
    exclude_set = {str(value) for value in (exclude_sources or []) if str(value).strip()}
    if include_set:
        filtered = filtered.loc[filtered[source_col].astype(str).isin(include_set)].copy()
    if exclude_set:
        filtered = filtered.loc[~filtered[source_col].astype(str).isin(exclude_set)].copy()
    return filtered


def apply_source_weight_overrides(
    frame: pd.DataFrame,
    *,
    source_col: str = "source_id",
    output_col: str = "sample_weight",
    source_weight_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    weighted = frame.copy()
    overrides = {
        str(key): float(value)
        for key, value in (source_weight_overrides or {}).items()
        if str(key).strip() and np.isfinite(float(value)) and float(value) >= 0.0
    }
    if not overrides:
        return weighted

    if output_col not in weighted.columns:
        weighted[output_col] = 1.0
    source_scale = weighted[source_col].astype(str).map(lambda value: overrides.get(value, 1.0)).to_numpy(dtype=float)
    weights = pd.to_numeric(weighted[output_col], errors="coerce").fillna(1.0).to_numpy(dtype=float) * source_scale
    if not np.isfinite(weights).all() or float(np.mean(weights)) <= 0.0:
        weights = np.ones(len(weighted), dtype=np.float64)
    else:
        weights /= float(np.mean(weights))
    weighted[output_col] = weights.astype(np.float32)
    return weighted


def grouped_probability_frame(
    frame: pd.DataFrame,
    *,
    target_col: str,
    prob_col: str,
    group_col: str,
) -> pd.DataFrame:
    grouped = frame.groupby(group_col, observed=False).agg(
        y_true=(target_col, "first"),
        y_prob=(prob_col, "mean"),
    )
    grouped = grouped.reset_index()
    grouped["y_prob"] = np.clip(pd.to_numeric(grouped["y_prob"], errors="coerce").fillna(0.5), 1e-6, 1.0 - 1e-6)
    grouped["y_true"] = pd.to_numeric(grouped["y_true"], errors="coerce").fillna(0).astype(int)
    return grouped


def fit_platt_scaler(
    frame: pd.DataFrame,
    *,
    logit_col: str,
    target_col: str,
    group_col: str = "split_group_id",
    l2_strength: float = 1.0,
    min_groups: int = 3,
    random_seed: int = 13,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "calibration_method": "identity",
            "calibration_intercept": 0.0,
            "calibration_slope": 1.0,
            "calibration_n_groups": 0,
        }

    keep_cols = [group_col, target_col, logit_col]
    if "source_id" in frame.columns:
        keep_cols.append("source_id")
    calibration = frame[keep_cols].copy()
    calibration[group_col] = calibration[group_col].astype(str)
    calibration[target_col] = pd.to_numeric(calibration[target_col], errors="coerce").fillna(0).astype(int)
    calibration[logit_col] = pd.to_numeric(calibration[logit_col], errors="coerce").fillna(0.0)
    n_groups = int(calibration[group_col].nunique())

    if n_groups < int(min_groups) or calibration[target_col].nunique() < 2:
        return {
            "calibration_method": "identity",
            "calibration_intercept": 0.0,
            "calibration_slope": 1.0,
            "calibration_n_groups": n_groups,
        }

    graph_weights = add_balanced_sample_weights(
        calibration,
        group_col=group_col,
        source_col="source_id",
        donor_balance=True,
        source_balance="source_id" in calibration.columns,
    )["sample_weight"].to_numpy(dtype=float)

    model = LogisticRegression(
        C=float(1.0 / max(float(l2_strength), 1e-6)),
        class_weight="balanced",
        fit_intercept=True,
        max_iter=5000,
        random_state=int(random_seed),
        solver="lbfgs",
    )
    try:
        model.fit(
            calibration[[logit_col]].to_numpy(dtype=float),
            calibration[target_col].to_numpy(dtype=int),
            sample_weight=graph_weights,
        )
    except Exception:
        return {
            "calibration_method": "identity",
            "calibration_intercept": 0.0,
            "calibration_slope": 1.0,
            "calibration_n_groups": n_groups,
        }

    slope = float(model.coef_.reshape(-1)[0])
    intercept = float(model.intercept_.reshape(-1)[0])
    if not np.isfinite(slope) or not np.isfinite(intercept) or slope <= 0.0:
        return {
            "calibration_method": "identity",
            "calibration_intercept": 0.0,
            "calibration_slope": 1.0,
            "calibration_n_groups": n_groups,
        }

    return {
        "calibration_method": "platt",
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_n_groups": n_groups,
    }


def apply_platt_scaler(logits: pd.Series | np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    intercept = float(calibration.get("calibration_intercept", 0.0))
    slope = float(calibration.get("calibration_slope", 1.0))
    calibrated = intercept + slope * values
    return 1.0 / (1.0 + np.exp(-np.clip(calibrated, -30.0, 30.0)))


def select_regularized_group_threshold(
    frame: pd.DataFrame,
    *,
    target_col: str,
    prob_col: str,
    group_col: str = "split_group_id",
    default_threshold: float = 0.5,
    threshold_penalty: float = 0.15,
    prevalence_penalty: float = 0.10,
    min_tuning_groups: int = 6,
) -> float:
    grouped = grouped_probability_frame(frame, target_col=target_col, prob_col=prob_col, group_col=group_col)
    if grouped.empty or grouped["y_true"].nunique() < 2 or len(grouped) < int(min_tuning_groups):
        return float(default_threshold)

    probabilities = np.sort(grouped["y_prob"].astype(float).unique())
    candidates = {float(default_threshold)}
    candidates.update(probabilities.tolist())
    if len(probabilities) > 1:
        candidates.update(((probabilities[:-1] + probabilities[1:]) / 2.0).tolist())

    prevalence = float(grouped["y_true"].mean())
    best_threshold = float(default_threshold)
    best_score = -float("inf")
    for threshold in sorted(candidates):
        y_pred = (grouped["y_prob"] >= float(threshold)).astype(int)
        score = float(balanced_accuracy_score(grouped["y_true"], y_pred))
        score -= float(threshold_penalty) * abs(float(threshold) - float(default_threshold))
        score -= float(prevalence_penalty) * abs(float(y_pred.mean()) - prevalence)
        if score > best_score or (score == best_score and abs(float(threshold) - float(default_threshold)) < abs(best_threshold - float(default_threshold))):
            best_score = score
            best_threshold = float(threshold)
    return best_threshold
