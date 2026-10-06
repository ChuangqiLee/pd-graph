#!/usr/bin/env python3
"""Train the Step 2 age-conditioned PD deviation graph model."""

from __future__ import annotations

import argparse
import copy
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step2.common import load_yaml, markdown_table, read_table, write_markdown, write_table
from scripts.step2.graph_models import (
    AgeConditionedPDModel,
    build_context_vocabulary,
    context_cardinalities,
    grouped_classification_metrics,
    load_graph,
    tensorize_graph,
    tensorize_graph_context,
)
from scripts.step2.training_utils import (
    add_balanced_sample_weights,
    apply_source_filters,
    apply_source_weight_overrides,
    apply_platt_scaler,
    fit_platt_scaler,
    make_inner_group_split,
    make_outer_group_splits,
    select_regularized_group_threshold,
)


DEFAULT_CONFIG = Path("configs/step2/step2_model_pd.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Step 2 PD graph model.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def resolve_stratify_fields(config_block: dict[str, object], default: list[str]) -> list[str]:
    fields = config_block.get("stratify_fields", default)
    if isinstance(fields, str):
        return [fields]
    if isinstance(fields, (list, tuple)):
        return [str(field) for field in fields if str(field).strip()]
    return list(default)


def donor_consistency_loss(logits: list[torch.Tensor], groups: list[str]) -> torch.Tensor:
    if not logits:
        return torch.tensor(0.0)
    unique_groups = sorted(set(groups))
    losses = []
    stacked = torch.stack(logits)
    for group in unique_groups:
        indices = [idx for idx, value in enumerate(groups) if value == group]
        if len(indices) > 1:
            losses.append(torch.var(stacked[indices], unbiased=False))
    if not losses:
        return torch.tensor(0.0, device=stacked.device)
    return torch.stack(losses).mean()


def smoothed_binary_target(target: float, smoothing: float) -> float:
    base = 1.0 if float(target) >= 0.5 else 0.0
    eps = max(0.0, min(float(smoothing), 0.49))
    return base * (1.0 - eps) + 0.5 * eps


def normalized_age_value(age_years: float | int | None, age_mean: float, age_std: float) -> float:
    if age_years is None:
        return 0.0
    age_value = float(age_years)
    if np.isnan(age_value):
        return 0.0
    return (age_value - float(age_mean)) / float(age_std)


def resolve_device(runtime_config: dict[str, object]) -> torch.device:
    requested = str(runtime_config.get("device", "auto")).strip().lower()
    if requested in {"cuda", "gpu"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_context(device: torch.device, enabled: bool) -> object:
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def source_classification_summary(frame: pd.DataFrame, *, prob_col: str = "pd_probability") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["source_id", "n_graphs", "n_groups", "balanced_accuracy", "auroc", "brier_score", "log_loss"])
    rows = []
    for source_id, local in frame.groupby("source_id", observed=False):
        if local["disease_target"].nunique() < 2:
            continue
        metrics = grouped_classification_metrics(
            local,
            target_col="disease_target",
            prob_col=prob_col,
            group_col="split_group_id",
            threshold=float(local["decision_threshold"].iloc[0]),
        )
        rows.append(
            {
                "source_id": source_id,
                "n_graphs": int(len(local)),
                "n_groups": int(local["split_group_id"].nunique()),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "auroc": metrics["auroc"],
                "brier_score": metrics["brier_score"],
                "log_loss": metrics["log_loss"],
            }
        )
    if not rows:
        return pd.DataFrame(columns=["source_id", "n_graphs", "n_groups", "balanced_accuracy", "auroc", "brier_score", "log_loss"])
    return pd.DataFrame(rows).sort_values(["balanced_accuracy", "source_id"], ascending=[False, True]).reset_index(drop=True)


def load_compatible_encoder_state(model: AgeConditionedPDModel, state_path: Path) -> None:
    if not state_path.exists():
        return
    state = torch.load(state_path, map_location="cpu")
    target_state = model.encoder.state_dict()
    compatible = {}
    for key, value in state.items():
        if not str(key).startswith("encoder."):
            continue
        target_key = str(key).replace("encoder.", "", 1)
        if target_key in target_state and tuple(target_state[target_key].shape) == tuple(value.shape):
            compatible[target_key] = value
    if compatible:
        model.encoder.load_state_dict(compatible, strict=False)


def collect_pd_outputs(
    *,
    model: AgeConditionedPDModel,
    frame: pd.DataFrame,
    fold_index: int,
    split_name: str,
    age_mean: float,
    age_std: float,
    vocab: dict[str, dict[str, int]],
    device: torch.device,
    use_amp: bool,
    calibration: dict[str, object],
    decision_threshold: float,
    include_embeddings: bool = True,
    include_attention: bool = True,
    age_value_override: float | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    embedding_rows: list[dict[str, object]] = []
    attention_rows: list[dict[str, object]] = []

    calibration_intercept = float(calibration.get("calibration_intercept", 0.0))
    calibration_slope = float(calibration.get("calibration_slope", 1.0))
    calibration_method = str(calibration.get("calibration_method", "identity"))

    model.eval()
    with torch.no_grad():
        for row in frame.itertuples(index=False):
            graph_obj = load_graph(row.graph_path)
            graph = tensorize_graph(graph_obj, device)
            graph_context = tensorize_graph_context(row, graph, vocab, device)
            if age_value_override is None:
                age_value_scalar = normalized_age_value(getattr(row, "age_years", None), age_mean, age_std)
            else:
                age_value_scalar = float(age_value_override)
            age_value = torch.tensor(age_value_scalar, dtype=torch.float32, device=device)
            with autocast_context(device, use_amp):
                logit, _, embedding, alpha, _ = model(graph, age_value, graph_context)

            logit_value = float(logit.cpu().item())
            prob_raw = float(torch.sigmoid(logit).cpu().item())
            prob_calibrated = float(apply_platt_scaler(np.asarray([logit_value], dtype=float), calibration)[0])
            split_role = str(getattr(row, "split_role", split_name))

            rows.append(
                {
                    "fold": fold_index,
                    "split": split_name,
                    "split_role": split_role,
                    "graph_id": row.graph_id,
                    "split_group_id": row.split_group_id,
                    "source_id": row.source_id,
                    "disease_target": 1 if getattr(row, "disease_group", "control") == "pd" else 0,
                    "pd_logit": logit_value,
                    "pd_probability_raw": prob_raw,
                    "pd_probability": prob_calibrated,
                    "calibration_method": calibration_method,
                    "calibration_intercept": calibration_intercept,
                    "calibration_slope": calibration_slope,
                    "decision_threshold": float(decision_threshold),
                    "pd_prediction": int(prob_calibrated >= float(decision_threshold)),
                }
            )

            if include_embeddings:
                embedding_rows.append(
                    {
                        "fold": fold_index,
                        "split": split_name,
                        "split_role": split_role,
                        "graph_id": row.graph_id,
                        **{f"emb_{idx}": float(value) for idx, value in enumerate(embedding.detach().cpu().numpy())},
                    }
                )
            if include_attention:
                edge_alpha = alpha.detach().cpu().numpy()
                top_edge_indices = np.argsort(edge_alpha)[::-1][:64]
                for edge_rank, edge_idx in enumerate(top_edge_indices, start=1):
                    attention_rows.append(
                        {
                            "fold": fold_index,
                            "split": split_name,
                            "split_role": split_role,
                            "graph_id": row.graph_id,
                            "edge_rank": edge_rank,
                            "attention": float(edge_alpha[edge_idx]),
                            "src_cell_id": str(graph_obj["node_ids"][graph_obj["edge_index"][0, edge_idx]]),
                            "dst_cell_id": str(graph_obj["node_ids"][graph_obj["edge_index"][1, edge_idx]]),
                            "top_pair_index": int(graph_obj["edge_top_pair_indices"][edge_idx, 0]),
                            "top_pair_score": float(graph_obj["edge_top_pair_scores"][edge_idx, 0]),
                        }
                    )
    return pd.DataFrame(rows), embedding_rows, attention_rows


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    seed = int(config.get("random_seed", 13))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    runtime_config = dict(config.get("runtime", {}))
    device = resolve_device(runtime_config)
    use_amp = bool(runtime_config.get("use_amp", True))
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(runtime_config.get("matmul_precision", "high")))

    output_dir = PROJECT_ROOT / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_manifest = read_table(PROJECT_ROOT / config["paths"]["graph_manifest_parquet"])
    allowed_disease_groups = [str(value) for value in config.get("disease_groups", ["control", "pd"])]
    include_age_missing = bool(config.get("include_age_missing", False))
    data_filters = dict(config.get("data_filters", {}))
    train_manifest = graph_manifest.loc[
        graph_manifest["task_name"].eq(config["task_name"])
        & graph_manifest["cohort_name"].isin(config["cohort_names"])
        & graph_manifest["disease_group"].isin(allowed_disease_groups)
        & (graph_manifest["age_years"].notna() | include_age_missing)
    ].copy()
    train_manifest = apply_source_filters(
        train_manifest,
        include_sources=data_filters.get("include_source_ids"),
        exclude_sources=data_filters.get("exclude_source_ids"),
    )
    bridge_manifest = graph_manifest.loc[
        graph_manifest["task_name"].eq(config["task_name"])
        & graph_manifest["cohort_name"].eq(config["bridge_cohort_name"])
    ].copy()
    if train_manifest.empty:
        raise RuntimeError("No graphs available for PD-model training.")

    train_manifest["disease_target"] = train_manifest["disease_group"].map({"control": 0, "pd": 1}).astype(int)

    group_field = str(config["cv"].get("group_field", "split_group_id"))
    cv_stratify_fields = resolve_stratify_fields(dict(config["cv"]), ["disease_target"])
    outer_splits = make_outer_group_splits(
        train_manifest,
        n_splits=int(config["cv"]["n_splits"]),
        random_seed=seed,
        group_col=group_field,
        stratify_col=None,
        stratify_cols=cv_stratify_fields,
        age_col="age_years",
        balance_age=bool(config["cv"].get("balance_age", False)),
    )

    sample_graph = load_graph(train_manifest.iloc[0]["graph_path"])
    node_dim = sample_graph["node_features"].shape[1]
    edge_dim = sample_graph["edge_attr"].shape[1]
    model_config = dict(config["model"])
    calibration_config = dict(config.get("calibration", {}))
    calibration_stratify_fields = resolve_stratify_fields(calibration_config, cv_stratify_fields)
    weighting_config = dict(config.get("weighting", {}))
    source_weight_overrides = {
        str(key): float(value)
        for key, value in dict(weighting_config.get("source_weight_overrides", {})).items()
    }
    hidden_dim = int(model_config["hidden_dim"])
    num_layers = int(model_config["num_layers"])
    dropout = float(model_config["dropout"])
    age_hidden_dim = int(model_config["age_hidden_dim"])
    model_family = str(model_config.get("family", "legacy_gnn"))
    num_heads = int(model_config.get("num_heads", 4))
    context_hidden_dim = int(model_config.get("context_hidden_dim", 48))
    domain_hidden_dim = int(model_config.get("domain_hidden_dim", 32))
    accumulation_steps = int(config["training"].get("accumulation_steps", config["training"]["batch_size"]))
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))
    use_donor_bags = bool(config["training"].get("use_donor_bags", True))
    donor_loss_weight = float(config["training"].get("donor_loss_weight", 1.0))
    graph_loss_weight = float(config["training"].get("graph_loss_weight", 0.25))
    label_smoothing = float(config["training"].get("label_smoothing", 0.05))
    logit_l2_weight = float(config["training"].get("logit_l2_weight", 0.0))

    relevant_manifest = pd.concat([train_manifest, bridge_manifest], ignore_index=True)
    vocab = build_context_vocabulary(relevant_manifest)
    context_sizes = context_cardinalities(vocab)
    source_invariance = dict(config.get("source_invariance", {}))
    source_loss_enabled = bool(source_invariance.get("enabled", True))
    gradient_reversal_lambda = float(source_invariance.get("gradient_reversal_lambda", 0.25))

    prediction_rows = []
    bridge_rows = []
    attention_rows = []
    embedding_rows = []
    fold_metrics = []
    threshold_rows = []
    fold_metadata_rows = []
    split_manifest_rows = []
    source_metric_rows = []

    for fold_split in outer_splits:
        fold_index = int(fold_split.fold)
        torch.manual_seed(seed + fold_index)

        outer_train_df = train_manifest.loc[train_manifest[group_field].astype(str).isin(set(fold_split.train_groups))].copy().reset_index(drop=True)
        val_df = train_manifest.loc[train_manifest[group_field].astype(str).isin(set(fold_split.val_groups))].copy().reset_index(drop=True)
        inner_split = make_inner_group_split(
            outer_train_df,
            random_seed=seed + fold_index,
            calibration_fraction=float(calibration_config.get("fraction", 0.4)),
            min_calibration_groups=int(calibration_config.get("min_groups", 2)),
            group_col=group_field,
            stratify_col=None,
            stratify_cols=calibration_stratify_fields,
            age_col="age_years",
            balance_age=bool(calibration_config.get("balance_age", config["cv"].get("balance_age", False))),
        )
        fit_groups = set(inner_split.fit_groups)
        calibration_groups = set(inner_split.calibration_groups)
        fit_df = outer_train_df.loc[outer_train_df[group_field].astype(str).isin(fit_groups)].copy().reset_index(drop=True)
        calibration_df = outer_train_df.loc[outer_train_df[group_field].astype(str).isin(calibration_groups)].copy().reset_index(drop=True)
        if calibration_df.empty:
            calibration_df = outer_train_df.copy()
        if fit_df.empty:
            fit_df = outer_train_df.copy()

        fit_df = add_balanced_sample_weights(
            fit_df,
            group_col=group_field,
            source_col="source_id",
            donor_balance=bool(weighting_config.get("donor_balance", True)),
            source_balance=bool(weighting_config.get("source_balance", True)),
        )
        fit_df = apply_source_weight_overrides(
            fit_df,
            source_col="source_id",
            output_col="sample_weight",
            source_weight_overrides=source_weight_overrides,
        )
        known_age_values = pd.to_numeric(fit_df["age_years"], errors="coerce").dropna()
        if known_age_values.empty:
            fallback_age_values = pd.to_numeric(outer_train_df["age_years"], errors="coerce").dropna()
            known_age_values = fallback_age_values if not fallback_age_values.empty else pd.Series([0.0], dtype=float)
        age_mean = float(known_age_values.mean())
        age_std = float(known_age_values.std(ddof=0) or 1.0)
        donor_labels = fit_df[[group_field, "disease_target"]].drop_duplicates(subset=[group_field])
        n_pos = int(donor_labels["disease_target"].eq(1).sum())
        n_neg = int(donor_labels["disease_target"].eq(0).sum())
        pos_weight = torch.tensor(float(n_neg / max(n_pos, 1)), dtype=torch.float32, device=device)

        model = AgeConditionedPDModel(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            age_hidden_dim=age_hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            model_family=model_family,
            num_heads=num_heads,
            context_cardinalities=context_sizes,
            context_hidden_dim=context_hidden_dim,
            domain_num_classes=context_sizes["source_id"] if source_loss_enabled else 0,
            domain_hidden_dim=domain_hidden_dim,
            gradient_reversal_lambda=gradient_reversal_lambda,
        ).to(device)
        age_model_path = PROJECT_ROOT / config["paths"]["output_dir"].replace("pd_model", "age_model") / f"fold_{fold_index:02d}_model.pt"
        load_compatible_encoder_state(model, age_model_path)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and use_amp)
        best_state = None
        best_metric = -float("inf")
        patience_counter = 0

        for epoch in range(int(config["training"]["epochs"])):
            model.train()
            shuffled = fit_df.sample(frac=1.0, random_state=seed + fold_index * 1000 + epoch)
            optimizer.zero_grad(set_to_none=True)
            grouped_fit = list(shuffled.groupby(group_field, observed=False))
            if use_donor_bags:
                random.Random(seed + fold_index * 1000 + epoch).shuffle(grouped_fit)
                batch_losses = []
                for step_index, (_, donor_df) in enumerate(grouped_fit, start=1):
                    donor_logits = []
                    donor_graph_losses = []
                    donor_age_losses = []
                    donor_source_losses = []
                    donor_group_names = []
                    sample_weights = []
                    donor_target_value = float(donor_df["disease_target"].iloc[0])
                    smoothed_target = smoothed_binary_target(donor_target_value, label_smoothing)
                    for row in donor_df.itertuples(index=False):
                        graph = tensorize_graph(load_graph(row.graph_path), device)
                        graph_context = tensorize_graph_context(row, graph, vocab, device)
                        age_missing = bool(pd.isna(getattr(row, "age_years", np.nan)))
                        age_value = torch.tensor(
                            normalized_age_value(getattr(row, "age_years", None), age_mean, age_std),
                            dtype=torch.float32,
                            device=device,
                        )
                        disease_target = torch.tensor(smoothed_target, dtype=torch.float32, device=device)
                        sample_weight = torch.tensor(float(getattr(row, "sample_weight", 1.0)), dtype=torch.float32, device=device)
                        age_weight = torch.tensor(0.0 if age_missing else float(getattr(row, "sample_weight", 1.0)), dtype=torch.float32, device=device)
                        with autocast_context(device, use_amp):
                            logit, age_pred, _, _, source_logits = model(graph, age_value, graph_context)
                            disease_loss = F.binary_cross_entropy_with_logits(
                                logit,
                                disease_target,
                                pos_weight=pos_weight,
                                reduction="none",
                            ) * sample_weight
                            age_loss = F.huber_loss(age_pred, age_value, delta=5.0, reduction="none") * age_weight
                            source_loss = torch.tensor(0.0, device=device)
                            if source_loss_enabled and source_logits is not None:
                                source_loss = F.cross_entropy(
                                    source_logits.view(1, -1),
                                    graph_context["source_id"].view(1),
                                    reduction="none",
                                ).mean() * sample_weight
                        donor_logits.append(logit)
                        donor_graph_losses.append(disease_loss)
                        donor_age_losses.append(age_loss)
                        donor_source_losses.append(source_loss)
                        donor_group_names.append(str(getattr(row, group_field)))
                        sample_weights.append(sample_weight)

                    donor_logits_stack = torch.stack(donor_logits)
                    donor_prob = torch.sigmoid(donor_logits_stack).mean().clamp(1e-5, 1.0 - 1e-5)
                    donor_logit = torch.logit(donor_prob)
                    donor_weight = torch.stack(sample_weights).mean()
                    donor_target = torch.tensor(smoothed_target, dtype=torch.float32, device=device)
                    donor_disease_loss = F.binary_cross_entropy_with_logits(
                        donor_logit,
                        donor_target,
                        pos_weight=pos_weight,
                        reduction="none",
                    ) * donor_weight
                    consistency = donor_consistency_loss(list(donor_logits_stack), donor_group_names).to(device)
                    total_loss = (
                        donor_loss_weight * donor_disease_loss
                        + graph_loss_weight * torch.stack(donor_graph_losses).mean()
                        + float(config["loss"]["lambda_age"]) * torch.stack(donor_age_losses).mean()
                        + float(config["loss"]["lambda_consistency"]) * consistency
                        + float(config["loss"].get("lambda_source", 0.0)) * torch.stack(donor_source_losses).mean()
                        + logit_l2_weight * donor_logits_stack.pow(2).mean()
                    )
                    batch_losses.append(total_loss)

                    if step_index % accumulation_steps == 0 or step_index == len(grouped_fit):
                        with autocast_context(device, False):
                            stacked_loss = torch.stack(batch_losses).mean()
                        scaler.scale(stacked_loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        batch_losses = []
            else:
                batch_logits = []
                batch_groups = []
                batch_disease_losses = []
                batch_age_losses = []
                batch_source_losses = []
                for step_index, row in enumerate(shuffled.itertuples(index=False), start=1):
                    graph = tensorize_graph(load_graph(row.graph_path), device)
                    graph_context = tensorize_graph_context(row, graph, vocab, device)
                    age_missing = bool(pd.isna(getattr(row, "age_years", np.nan)))
                    age_value = torch.tensor(
                        normalized_age_value(getattr(row, "age_years", None), age_mean, age_std),
                        dtype=torch.float32,
                        device=device,
                    )
                    disease_target = torch.tensor(smoothed_binary_target(float(row.disease_target), label_smoothing), dtype=torch.float32, device=device)
                    sample_weight = torch.tensor(float(getattr(row, "sample_weight", 1.0)), dtype=torch.float32, device=device)
                    age_weight = torch.tensor(0.0 if age_missing else float(getattr(row, "sample_weight", 1.0)), dtype=torch.float32, device=device)
                    with autocast_context(device, use_amp):
                        logit, age_pred, _, _, source_logits = model(graph, age_value, graph_context)
                        disease_loss = F.binary_cross_entropy_with_logits(logit, disease_target, pos_weight=pos_weight, reduction="none") * sample_weight
                        age_loss = F.huber_loss(age_pred, age_value, delta=5.0, reduction="none") * age_weight
                        source_loss = torch.tensor(0.0, device=device)
                        if source_loss_enabled and source_logits is not None:
                            source_loss = F.cross_entropy(
                                source_logits.view(1, -1),
                                graph_context["source_id"].view(1),
                                reduction="none",
                            ).mean() * sample_weight
                    batch_logits.append(logit)
                    batch_groups.append(str(getattr(row, group_field)))
                    batch_disease_losses.append(disease_loss)
                    batch_age_losses.append(age_loss)
                    batch_source_losses.append(source_loss)

                    if step_index % accumulation_steps == 0 or step_index == len(shuffled):
                        consistency = donor_consistency_loss(batch_logits, batch_groups).to(device)
                        with autocast_context(device, False):
                            total_loss = (
                                torch.stack(batch_disease_losses).mean()
                                + float(config["loss"]["lambda_age"]) * torch.stack(batch_age_losses).mean()
                                + float(config["loss"]["lambda_consistency"]) * consistency
                                + float(config["loss"].get("lambda_source", 0.0)) * torch.stack(batch_source_losses).mean()
                                + logit_l2_weight * torch.stack(batch_logits).pow(2).mean()
                            )
                        scaler.scale(total_loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        batch_logits = []
                        batch_groups = []
                        batch_disease_losses = []
                        batch_age_losses = []
                        batch_source_losses = []

            calibration_eval = calibration_df.copy()
            calibration_eval["split_role"] = "calibration"
            calibration_predictions, _, _ = collect_pd_outputs(
                model=model,
                frame=calibration_eval,
                fold_index=fold_index,
                split_name="train",
                age_mean=age_mean,
                age_std=age_std,
                vocab=vocab,
                device=device,
                use_amp=use_amp,
                calibration={"calibration_method": "identity", "calibration_intercept": 0.0, "calibration_slope": 1.0},
                decision_threshold=float(calibration_config.get("default_threshold", 0.5)),
                include_embeddings=False,
                include_attention=False,
            )
            calibration_metrics = grouped_classification_metrics(
                calibration_predictions,
                target_col="disease_target",
                prob_col="pd_probability_raw",
                group_col="split_group_id",
                threshold=float(calibration_config.get("default_threshold", 0.5)),
            )
            score = calibration_metrics["auroc"] if not np.isnan(calibration_metrics["auroc"]) else calibration_metrics["balanced_accuracy"]
            if score > best_metric:
                best_metric = score
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= int(config["training"]["patience"]):
                    break

        if best_state is None:
            best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        torch.save(best_state, output_dir / f"fold_{fold_index:02d}_model.pt")

        calibration_eval = calibration_df.copy()
        calibration_eval["split_role"] = "calibration"
        calibration_predictions, _, _ = collect_pd_outputs(
            model=model,
            frame=calibration_eval,
            fold_index=fold_index,
            split_name="train",
            age_mean=age_mean,
            age_std=age_std,
            vocab=vocab,
            device=device,
            use_amp=use_amp,
            calibration={"calibration_method": "identity", "calibration_intercept": 0.0, "calibration_slope": 1.0},
            decision_threshold=float(calibration_config.get("default_threshold", 0.5)),
            include_embeddings=False,
            include_attention=False,
        )
        calibration = fit_platt_scaler(
            calibration_predictions,
            logit_col="pd_logit",
            target_col="disease_target",
            group_col="split_group_id",
            l2_strength=float(calibration_config.get("platt_l2", 2.0)),
            min_groups=int(calibration_config.get("platt_min_groups", 3)),
            random_seed=seed + fold_index,
        )
        calibration_predictions["pd_probability"] = apply_platt_scaler(calibration_predictions["pd_logit"].to_numpy(dtype=float), calibration)
        decision_threshold = select_regularized_group_threshold(
            calibration_predictions,
            target_col="disease_target",
            prob_col="pd_probability",
            group_col="split_group_id",
            default_threshold=float(calibration_config.get("default_threshold", 0.5)),
            threshold_penalty=float(calibration_config.get("threshold_penalty", 0.2)),
            prevalence_penalty=float(calibration_config.get("prevalence_penalty", 0.1)),
            min_tuning_groups=int(calibration_config.get("min_threshold_groups", 6)),
        )

        outer_train_eval = outer_train_df.copy()
        outer_train_eval["split_role"] = np.where(
            outer_train_eval[group_field].astype(str).isin(calibration_groups),
            "calibration",
            "fit",
        )
        val_eval = val_df.copy()
        val_eval["split_role"] = "val"

        train_predictions, train_embeddings, train_attention = collect_pd_outputs(
            model=model,
            frame=outer_train_eval,
            fold_index=fold_index,
            split_name="train",
            age_mean=age_mean,
            age_std=age_std,
            vocab=vocab,
            device=device,
            use_amp=use_amp,
            calibration=calibration,
            decision_threshold=decision_threshold,
        )
        val_predictions, val_embeddings, val_attention = collect_pd_outputs(
            model=model,
            frame=val_eval,
            fold_index=fold_index,
            split_name="val",
            age_mean=age_mean,
            age_std=age_std,
            vocab=vocab,
            device=device,
            use_amp=use_amp,
            calibration=calibration,
            decision_threshold=decision_threshold,
        )

        prediction_rows.extend(train_predictions.to_dict("records"))
        prediction_rows.extend(val_predictions.to_dict("records"))
        embedding_rows.extend(train_embeddings)
        embedding_rows.extend(val_embeddings)
        attention_rows.extend(train_attention)
        attention_rows.extend(val_attention)

        raw_val_metrics = grouped_classification_metrics(
            val_predictions,
            target_col="disease_target",
            prob_col="pd_probability_raw",
            group_col="split_group_id",
            threshold=float(calibration_config.get("default_threshold", 0.5)),
        )
        calibrated_val_metrics = grouped_classification_metrics(
            val_predictions,
            target_col="disease_target",
            prob_col="pd_probability",
            group_col="split_group_id",
            threshold=decision_threshold,
        )
        fold_metrics.append(
            {
                "fold": fold_index,
                "decision_threshold": float(decision_threshold),
                "raw_auroc": raw_val_metrics["auroc"],
                "raw_brier_score": raw_val_metrics["brier_score"],
                "raw_log_loss": raw_val_metrics["log_loss"],
                **calibrated_val_metrics,
            }
        )
        threshold_rows.append({"fold": fold_index, "decision_threshold": float(decision_threshold)})

        source_summary = source_classification_summary(val_predictions, prob_col="pd_probability")
        if not source_summary.empty:
            source_summary.insert(0, "fold", fold_index)
            source_metric_rows.extend(source_summary.to_dict("records"))

        fold_metadata_rows.append(
            {
                "fold": fold_index,
                "age_mean": age_mean,
                "age_std": age_std,
                "n_fit_groups": int(len(fit_groups) or outer_train_df[group_field].nunique()),
                "n_calibration_groups": int(len(calibration_groups)),
                "n_val_groups": int(val_df[group_field].nunique()),
                "best_calibration_score": float(best_metric),
                "calibration_method": calibration["calibration_method"],
                "calibration_intercept": float(calibration["calibration_intercept"]),
                "calibration_slope": float(calibration["calibration_slope"]),
                "calibration_n_groups": int(calibration["calibration_n_groups"]),
                "decision_threshold": float(decision_threshold),
            }
        )
        split_manifest_rows.extend(
            outer_train_eval[["graph_id", "split_group_id", "source_id", "split_role"]]
            .assign(fold=fold_index, split="train")
            .to_dict("records")
        )
        split_manifest_rows.extend(
            val_eval[["graph_id", "split_group_id", "source_id", "split_role"]]
            .assign(fold=fold_index, split="val")
            .to_dict("records")
        )

        if not bridge_manifest.empty:
            bridge_eval = bridge_manifest.copy()
            bridge_eval["split_role"] = "bridge"
            bridge_predictions, _, _ = collect_pd_outputs(
                model=model,
                frame=bridge_eval,
                fold_index=fold_index,
                split_name="bridge",
                age_mean=age_mean,
                age_std=age_std,
                vocab=vocab,
                device=device,
                use_amp=use_amp,
                calibration=calibration,
                decision_threshold=decision_threshold,
                include_embeddings=False,
                include_attention=False,
                age_value_override=0.0,
            )
            bridge_rows.extend(bridge_predictions.to_dict("records"))

    predictions = pd.DataFrame(prediction_rows)
    embeddings = pd.DataFrame(embedding_rows)
    attention = pd.DataFrame(attention_rows)
    bridge = pd.DataFrame(bridge_rows)
    metrics = pd.DataFrame(fold_metrics)
    thresholds = pd.DataFrame(threshold_rows)
    fold_metadata = pd.DataFrame(fold_metadata_rows)
    split_manifest = pd.DataFrame(split_manifest_rows)
    source_metrics = pd.DataFrame(source_metric_rows)

    predictions.to_csv(output_dir / "pd_model_predictions.csv", index=False)
    write_table(embeddings, output_dir / "pd_model_embeddings.parquet", index=False)
    write_table(attention, output_dir / "pd_model_attention_summary.parquet", index=False)
    write_table(bridge, output_dir / "pd_model_lbd_bridge_scores.csv", index=False)
    write_table(metrics, output_dir / "pd_model_fold_metrics.csv", index=False)
    write_table(thresholds, output_dir / "pd_model_thresholds.csv", index=False)
    write_table(fold_metadata, output_dir / "pd_model_fold_metadata.csv", index=False)
    write_table(split_manifest, output_dir / "pd_model_split_manifest.csv", index=False)
    write_table(source_metrics, output_dir / "pd_model_source_metrics.csv", index=False)

    report_lines = [
        "# Step 2 PD Model",
        "",
        f"- Model family: `{model_family}`",
        f"- Device used: `{device.type}`",
        f"- Source invariance enabled: `{source_loss_enabled}`",
        f"- Donor-bag objective enabled: `{use_donor_bags}`",
        f"- Label smoothing: `{label_smoothing:.3f}`",
        "",
        "## Fold Metrics",
        markdown_table(metrics),
        "",
        "## Fold Metadata",
        markdown_table(fold_metadata),
        "",
        f"- Mean validation AUROC: `{metrics['auroc'].mean():.3f}`",
        f"- Mean validation balanced accuracy: `{metrics['balanced_accuracy'].mean():.3f}`",
        f"- Mean validation Brier score: `{metrics['brier_score'].mean():.3f}`",
        f"- Mean validation log-loss: `{metrics['log_loss'].mean():.3f}`",
        f"- Mean fold decision threshold: `{metrics['decision_threshold'].mean():.3f}`",
    ]
    if not source_metrics.empty:
        report_lines.extend(
            [
                "",
                "## Validation Source Summary",
                markdown_table(source_metrics),
            ]
        )
    write_markdown(output_dir / "pd_model_report.md", "\n".join(report_lines))


if __name__ == "__main__":
    main()
