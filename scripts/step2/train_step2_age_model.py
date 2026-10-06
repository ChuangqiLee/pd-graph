#!/usr/bin/env python3
"""Train the Step 2 control aging graph model."""

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
    AgeModel,
    build_context_vocabulary,
    context_cardinalities,
    grouped_regression_metrics,
    load_graph,
    tensorize_graph,
    tensorize_graph_context,
)
from scripts.step2.training_utils import (
    add_balanced_sample_weights,
    apply_source_filters,
    apply_source_weight_overrides,
    make_inner_group_split,
    make_outer_group_splits,
)


DEFAULT_CONFIG = Path("configs/step2/step2_model_age.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Step 2 age regression graph model.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def resolve_stratify_fields(config_block: dict[str, object], default: list[str]) -> list[str]:
    fields = config_block.get("stratify_fields", default)
    if isinstance(fields, str):
        return [fields]
    if isinstance(fields, (list, tuple)):
        return [str(field) for field in fields if str(field).strip()]
    return list(default)


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


def source_mae_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["source_id", "n_graphs", "n_groups", "mae"])
    rows = []
    for source_id, local in frame.groupby("source_id", observed=False):
        metrics = grouped_regression_metrics(local, target_col="age_years", pred_col="pred_age_years", group_col="split_group_id")
        rows.append(
            {
                "source_id": source_id,
                "n_graphs": int(len(local)),
                "n_groups": int(local["split_group_id"].nunique()),
                "mae": metrics["mae"],
            }
        )
    if not rows:
        return pd.DataFrame(columns=["source_id", "n_graphs", "n_groups", "mae"])
    return pd.DataFrame(rows).sort_values(["mae", "source_id"]).reset_index(drop=True)


def collect_age_outputs(
    *,
    model: AgeModel,
    frame: pd.DataFrame,
    fold_index: int,
    split_name: str,
    age_mean: float,
    age_std: float,
    vocab: dict[str, dict[str, int]],
    device: torch.device,
    use_amp: bool,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    embedding_rows: list[dict[str, object]] = []
    attention_rows: list[dict[str, object]] = []

    model.eval()
    with torch.no_grad():
        for row in frame.itertuples(index=False):
            graph_obj = load_graph(row.graph_path)
            graph = tensorize_graph(graph_obj, device)
            graph_context = tensorize_graph_context(row, graph, vocab, device)
            with autocast_context(device, use_amp):
                pred, embedding, alpha = model(graph, graph_context)
            pred_age = float(pred.cpu().item() * age_std + age_mean)
            split_role = str(getattr(row, "split_role", split_name))
            rows.append(
                {
                    "fold": fold_index,
                    "split": split_name,
                    "split_role": split_role,
                    "graph_id": row.graph_id,
                    "split_group_id": row.split_group_id,
                    "source_id": row.source_id,
                    "age_years": float(row.age_years),
                    "pred_age_years": pred_age,
                }
            )
            embedding_rows.append(
                {
                    "fold": fold_index,
                    "split": split_name,
                    "split_role": split_role,
                    "graph_id": row.graph_id,
                    **{f"emb_{idx}": float(value) for idx, value in enumerate(embedding.detach().cpu().numpy())},
                }
            )
            edge_alpha = alpha.detach().cpu().numpy()
            top_edge_indices = np.argsort(edge_alpha)[::-1][:64]
            for edge_rank, edge_idx in enumerate(top_edge_indices, start=1):
                top_pair_idx = int(graph_obj["edge_top_pair_indices"][edge_idx, 0])
                top_pair_score = float(graph_obj["edge_top_pair_scores"][edge_idx, 0])
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
                        "top_pair_index": top_pair_idx,
                        "top_pair_score": top_pair_score,
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
    graph_manifest = graph_manifest.loc[
        graph_manifest["task_name"].eq(config["task_name"])
        & graph_manifest["cohort_name"].eq(config["cohort_name"])
        & graph_manifest["age_years"].notna()
    ].copy()
    data_filters = dict(config.get("data_filters", {}))
    graph_manifest = apply_source_filters(
        graph_manifest,
        include_sources=data_filters.get("include_source_ids"),
        exclude_sources=data_filters.get("exclude_source_ids"),
    )
    if graph_manifest.empty:
        raise RuntimeError("No graphs available for age-model training.")

    group_field = str(config["cv"].get("group_field", "split_group_id"))
    cv_stratify_fields = resolve_stratify_fields(dict(config["cv"]), ["source_id"])
    outer_splits = make_outer_group_splits(
        graph_manifest,
        n_splits=int(config["cv"]["n_splits"]),
        random_seed=seed,
        group_col=group_field,
        stratify_col=None,
        stratify_cols=cv_stratify_fields,
        age_col="age_years",
        balance_age=bool(config["cv"].get("balance_age", False)),
    )

    sample_graph = load_graph(graph_manifest.iloc[0]["graph_path"])
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
    model_family = str(model_config.get("family", "legacy_gnn"))
    num_heads = int(model_config.get("num_heads", 4))
    context_hidden_dim = int(model_config.get("context_hidden_dim", 48))
    accumulation_steps = int(config["training"].get("accumulation_steps", config["training"]["batch_size"]))
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))
    huber_delta = float(config["training"]["huber_delta"])

    vocab = build_context_vocabulary(graph_manifest)
    context_sizes = context_cardinalities(vocab)

    fold_metrics = []
    fold_metadata_rows = []
    split_manifest_rows = []
    prediction_rows = []
    embedding_rows = []
    attention_rows = []
    source_metric_rows = []

    for fold_split in outer_splits:
        fold_index = int(fold_split.fold)
        torch.manual_seed(seed + fold_index)

        outer_train_df = graph_manifest.loc[graph_manifest[group_field].astype(str).isin(set(fold_split.train_groups))].copy().reset_index(drop=True)
        val_df = graph_manifest.loc[graph_manifest[group_field].astype(str).isin(set(fold_split.val_groups))].copy().reset_index(drop=True)
        inner_split = make_inner_group_split(
            outer_train_df,
            random_seed=seed + fold_index,
            calibration_fraction=float(calibration_config.get("fraction", 0.34)),
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
        age_mean = float(fit_df["age_years"].mean())
        age_std = float(fit_df["age_years"].std(ddof=0) or 1.0)

        model = AgeModel(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            model_family=model_family,
            num_heads=num_heads,
            context_cardinalities=context_sizes,
            context_hidden_dim=context_hidden_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and use_amp)
        best_state = None
        best_mae = float("inf")
        patience = int(config["training"]["patience"])
        patience_counter = 0

        for epoch in range(int(config["training"]["epochs"])):
            model.train()
            shuffled = fit_df.sample(frac=1.0, random_state=seed + fold_index * 1000 + epoch)
            optimizer.zero_grad(set_to_none=True)
            for step_index, row in enumerate(shuffled.itertuples(index=False), start=1):
                graph = tensorize_graph(load_graph(row.graph_path), device)
                graph_context = tensorize_graph_context(row, graph, vocab, device)
                target = torch.tensor((float(row.age_years) - age_mean) / age_std, dtype=torch.float32, device=device)
                sample_weight = torch.tensor(float(getattr(row, "sample_weight", 1.0)), dtype=torch.float32, device=device)
                with autocast_context(device, use_amp):
                    pred, _, _ = model(graph, graph_context)
                    loss = F.huber_loss(pred, target, delta=huber_delta, reduction="none") * sample_weight
                scaler.scale(loss).backward()
                if step_index % accumulation_steps == 0 or step_index == len(shuffled):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            calibration_eval = calibration_df.copy()
            calibration_eval["split_role"] = "calibration"
            calibration_predictions, _, _ = collect_age_outputs(
                model=model,
                frame=calibration_eval,
                fold_index=fold_index,
                split_name="train",
                age_mean=age_mean,
                age_std=age_std,
                vocab=vocab,
                device=device,
                use_amp=use_amp,
            )
            calibration_metrics = grouped_regression_metrics(
                calibration_predictions,
                target_col="age_years",
                pred_col="pred_age_years",
                group_col="split_group_id",
            )
            if calibration_metrics["mae"] < best_mae:
                best_mae = calibration_metrics["mae"]
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is None:
            best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        torch.save(best_state, output_dir / f"fold_{fold_index:02d}_model.pt")

        outer_train_eval = outer_train_df.copy()
        outer_train_eval["split_role"] = np.where(
            outer_train_eval[group_field].astype(str).isin(calibration_groups),
            "calibration",
            "fit",
        )
        val_eval = val_df.copy()
        val_eval["split_role"] = "val"

        train_predictions, train_embeddings, train_attention = collect_age_outputs(
            model=model,
            frame=outer_train_eval,
            fold_index=fold_index,
            split_name="train",
            age_mean=age_mean,
            age_std=age_std,
            vocab=vocab,
            device=device,
            use_amp=use_amp,
        )
        val_predictions, val_embeddings, val_attention = collect_age_outputs(
            model=model,
            frame=val_eval,
            fold_index=fold_index,
            split_name="val",
            age_mean=age_mean,
            age_std=age_std,
            vocab=vocab,
            device=device,
            use_amp=use_amp,
        )

        prediction_rows.extend(train_predictions.to_dict("records"))
        prediction_rows.extend(val_predictions.to_dict("records"))
        embedding_rows.extend(train_embeddings)
        embedding_rows.extend(val_embeddings)
        attention_rows.extend(train_attention)
        attention_rows.extend(val_attention)

        val_metrics = grouped_regression_metrics(val_predictions, target_col="age_years", pred_col="pred_age_years", group_col="split_group_id")
        fold_metrics.append({"fold": fold_index, **val_metrics})

        source_summary = source_mae_summary(val_predictions)
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
                "best_calibration_mae": float(best_mae),
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

    predictions = pd.DataFrame(prediction_rows)
    embeddings = pd.DataFrame(embedding_rows)
    attention = pd.DataFrame(attention_rows)
    metrics = pd.DataFrame(fold_metrics)
    fold_metadata = pd.DataFrame(fold_metadata_rows)
    split_manifest = pd.DataFrame(split_manifest_rows)
    source_metrics = pd.DataFrame(source_metric_rows)

    predictions.to_csv(output_dir / "age_model_predictions.csv", index=False)
    write_table(embeddings, output_dir / "age_model_embeddings.parquet", index=False)
    write_table(attention, output_dir / "age_model_attention_summary.parquet", index=False)
    write_table(metrics, output_dir / "age_model_fold_metrics.csv", index=False)
    write_table(fold_metadata, output_dir / "age_model_fold_metadata.csv", index=False)
    write_table(split_manifest, output_dir / "age_model_split_manifest.csv", index=False)
    write_table(source_metrics, output_dir / "age_model_source_metrics.csv", index=False)

    report_lines = [
        "# Step 2 Age Model",
        "",
        f"- Model family: `{model_family}`",
        f"- Device used: `{device.type}`",
        "",
        "## Fold Metrics",
        markdown_table(metrics),
        "",
        "## Fold Metadata",
        markdown_table(fold_metadata),
        "",
        f"- Mean validation MAE: `{metrics['mae'].mean():.3f}` years",
        f"- Mean calibration MAE: `{fold_metadata['best_calibration_mae'].mean():.3f}` years",
    ]
    if not source_metrics.empty:
        report_lines.extend(
            [
                "",
                "## Validation Source Summary",
                markdown_table(source_metrics),
            ]
        )
    write_markdown(output_dir / "age_model_report.md", "\n".join(report_lines))


if __name__ == "__main__":
    main()
