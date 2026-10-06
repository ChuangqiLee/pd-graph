#!/usr/bin/env python3
"""Graph models and utilities for Step 2."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import load_pickle_gz


def load_graph(path_like: str | Path) -> dict[str, Any]:
    return load_pickle_gz(path_like)


def tensorize_graph(graph: dict[str, Any], device: torch.device) -> dict[str, Any]:
    sender_mask = torch.as_tensor(graph["sender_mask"], dtype=torch.bool, device=device)
    receiver_mask = torch.as_tensor(graph["receiver_mask"], dtype=torch.bool, device=device)
    role_index = torch.where(sender_mask, 0, 1).to(torch.long)
    return {
        "graph_id": graph["graph_id"],
        "node_ids": graph["node_ids"],
        "edge_index": torch.as_tensor(graph["edge_index"], dtype=torch.long, device=device),
        "node_features": torch.as_tensor(graph["node_features"], dtype=torch.float32, device=device),
        "edge_attr": torch.as_tensor(graph["edge_attr"], dtype=torch.float32, device=device),
        "sender_mask": sender_mask,
        "receiver_mask": receiver_mask,
        "role_index": role_index,
        "top_pair_indices": torch.as_tensor(graph["edge_top_pair_indices"], dtype=torch.long, device=device),
        "top_pair_scores": torch.as_tensor(graph["edge_top_pair_scores"], dtype=torch.float32, device=device),
        "metadata": graph["metadata"],
        "pathway_names": graph["pathway_names"],
    }


def grouped_regression_metrics(frame: pd.DataFrame, *, target_col: str, pred_col: str, group_col: str) -> dict[str, float]:
    grouped = frame.groupby(group_col, observed=False).agg(
        y_true=(target_col, "first"),
        y_pred=(pred_col, "mean"),
    )
    mae = float(np.mean(np.abs(grouped["y_true"].to_numpy() - grouped["y_pred"].to_numpy())))
    return {"mae": mae}


def grouped_classification_metrics(
    frame: pd.DataFrame,
    *,
    target_col: str,
    prob_col: str,
    group_col: str,
    threshold: float = 0.5,
) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    grouped = frame.groupby(group_col, observed=False).agg(
        y_true=(target_col, "first"),
        y_prob=(prob_col, "mean"),
    )
    if grouped.empty:
        return {
            "balanced_accuracy": float("nan"),
            "auroc": float("nan"),
            "brier_score": float("nan"),
            "log_loss": float("nan"),
            "positive_rate": float("nan"),
            "n_groups": 0.0,
        }
    grouped["y_true"] = pd.to_numeric(grouped["y_true"], errors="coerce").fillna(0).astype(int)
    grouped["y_prob"] = np.clip(pd.to_numeric(grouped["y_prob"], errors="coerce").fillna(0.5), 1e-6, 1.0 - 1e-6)
    grouped["y_pred"] = (grouped["y_prob"] >= float(threshold)).astype(int)
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(grouped["y_true"], grouped["y_pred"])),
        "brier_score": float(np.mean((grouped["y_true"].to_numpy(dtype=float) - grouped["y_prob"].to_numpy(dtype=float)) ** 2)),
        "log_loss": float(
            -np.mean(
                grouped["y_true"].to_numpy(dtype=float) * np.log(grouped["y_prob"].to_numpy(dtype=float))
                + (1.0 - grouped["y_true"].to_numpy(dtype=float)) * np.log(1.0 - grouped["y_prob"].to_numpy(dtype=float))
            )
        ),
        "positive_rate": float(grouped["y_prob"].mean()),
        "n_groups": float(len(grouped)),
    }
    if grouped["y_true"].nunique() >= 2:
        metrics["auroc"] = float(roc_auc_score(grouped["y_true"], grouped["y_prob"]))
    else:
        metrics["auroc"] = float("nan")
    return metrics


def _row_value(row: Any, name: str) -> Any:
    if hasattr(row, name):
        return getattr(row, name)
    if isinstance(row, dict):
        return row.get(name)
    return None


def _normalize_category(value: Any) -> str:
    if value is None:
        return "__unk__"
    if isinstance(value, float) and np.isnan(value):
        return "__unk__"
    text = str(value).strip()
    return text if text else "__unk__"


def build_context_vocabulary(manifest: pd.DataFrame) -> dict[str, dict[str, int]]:
    fields = ["source_id", "cohort_name", "brain_region_standardized", "sex_standardized"]
    vocab: dict[str, dict[str, int]] = {}
    for field in fields:
        values = sorted({_normalize_category(value) for value in manifest[field].tolist()})
        if "__unk__" not in values:
            values = ["__unk__", *values]
        vocab[field] = {value: idx for idx, value in enumerate(values)}
    return vocab


def context_cardinalities(vocab: dict[str, dict[str, int]]) -> dict[str, int]:
    return {name: len(mapping) for name, mapping in vocab.items()}


def tensorize_graph_context(
    row: Any,
    graph: dict[str, Any],
    vocab: dict[str, dict[str, int]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sender_count = int(graph["sender_mask"].sum().item())
    receiver_count = int(graph["receiver_mask"].sum().item())
    edge_count = int(graph["edge_index"].shape[1])
    possible_edges = max(sender_count * max(receiver_count, 1), 1)
    density = edge_count / possible_edges

    encoded = {}
    for field in ["source_id", "cohort_name", "brain_region_standardized", "sex_standardized"]:
        value = _normalize_category(_row_value(row, field))
        encoded[field] = torch.tensor(vocab[field].get(value, 0), dtype=torch.long, device=device)
    encoded["numeric"] = torch.tensor(
        [
            1.0 if bool(graph["metadata"].get("age_missing_mask", False)) else 0.0,
            math.log1p(float(sender_count)),
            math.log1p(float(receiver_count)),
            math.log1p(float(edge_count)),
            float(density),
        ],
        dtype=torch.float32,
        device=device,
    )
    return encoded


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambda_scale: float) -> torch.Tensor:
        ctx.lambda_scale = float(lambda_scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.lambda_scale, None


def reverse_gradient(x: torch.Tensor, lambda_scale: float) -> torch.Tensor:
    return GradientReversalFunction.apply(x, float(lambda_scale))


def scatter_softmax(logits: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("Expected logits to have shape [n_edges, n_heads].")
    n_edges, n_heads = logits.shape
    device = logits.device
    head_ids = torch.arange(n_heads, device=device, dtype=index.dtype).unsqueeze(0).expand(n_edges, -1)
    flat_index = (index.unsqueeze(1) * n_heads + head_ids).reshape(-1)
    flat_logits = logits.reshape(-1)

    max_buf = torch.full((num_segments * n_heads,), -1e9, device=device, dtype=logits.dtype)
    max_buf.scatter_reduce_(0, flat_index, flat_logits, reduce="amax", include_self=True)
    stabilized = flat_logits - max_buf[flat_index]
    exp_logits = torch.exp(stabilized)
    sum_buf = torch.zeros((num_segments * n_heads,), device=device, dtype=logits.dtype)
    sum_buf.scatter_add_(0, flat_index, exp_logits)
    alpha = exp_logits / (sum_buf[flat_index] + 1e-12)
    return alpha.reshape(n_edges, n_heads)


class EdgeAwareLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        self.src_proj = nn.Linear(hidden_dim, hidden_dim)
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim * 2 + edge_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src = edge_index[0]
        dst = edge_index[1]
        src_msg = self.src_proj(x[src]) + self.edge_proj(edge_attr)
        attn_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        attn_logits = self.attn(attn_input).squeeze(-1)

        num_nodes = x.shape[0]
        max_per_dst = torch.full((num_nodes,), -1e9, device=x.device)
        max_per_dst.scatter_reduce_(0, dst, attn_logits, reduce="amax", include_self=True)
        exp_logits = torch.exp(attn_logits - max_per_dst[dst])
        sum_per_dst = torch.zeros((num_nodes,), device=x.device)
        sum_per_dst.scatter_add_(0, dst, exp_logits)
        alpha = exp_logits / (sum_per_dst[dst] + 1e-12)

        agg = torch.zeros((num_nodes, src_msg.shape[1]), device=x.device)
        agg.index_add_(0, dst, src_msg * alpha.unsqueeze(-1))
        out = self.self_proj(x) + F.dropout(agg, p=self.dropout, training=self.training)
        out = self.norm(out)
        out = F.relu(out)
        return out, alpha


class GraphEncoder(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.node_input = nn.Linear(node_dim, hidden_dim)
        self.edge_input = nn.Linear(edge_dim, hidden_dim)
        self.layers = nn.ModuleList([EdgeAwareLayer(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)])
        self.dropout = dropout
        self.output_dim = hidden_dim * 4

    def forward(self, graph: dict[str, Any], graph_context: dict[str, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.relu(self.node_input(graph["node_features"]))
        edge_attr = F.relu(self.edge_input(graph["edge_attr"]))
        last_alpha = torch.zeros((graph["edge_index"].shape[1],), device=x.device)
        for layer in self.layers:
            x, last_alpha = layer(x, graph["edge_index"], edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)

        sender_mask = graph["sender_mask"]
        receiver_mask = graph["receiver_mask"]
        sender_pool = x[sender_mask].mean(dim=0) if sender_mask.any() else torch.zeros((x.shape[1],), device=x.device)
        receiver_pool = x[receiver_mask].mean(dim=0) if receiver_mask.any() else torch.zeros((x.shape[1],), device=x.device)
        global_pool = x.mean(dim=0)
        edge_pool = edge_attr.mean(dim=0) if edge_attr.shape[0] else torch.zeros((edge_attr.shape[1],), device=x.device)
        graph_embedding = torch.cat([sender_pool, receiver_pool, global_pool, edge_pool], dim=-1)
        return graph_embedding, last_alpha, x


class GraphContextEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        context_cardinalities: dict[str, int] | None,
        *,
        numeric_dim: int = 5,
        context_hidden_dim: int = 48,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        context_cardinalities = context_cardinalities or {}
        embed_dim = max(8, min(hidden_dim // 4, 32))
        self.source_embedding = nn.Embedding(max(int(context_cardinalities.get("source_id", 1)), 1), embed_dim)
        self.cohort_embedding = nn.Embedding(max(int(context_cardinalities.get("cohort_name", 1)), 1), embed_dim)
        self.region_embedding = nn.Embedding(max(int(context_cardinalities.get("brain_region_standardized", 1)), 1), embed_dim)
        self.sex_embedding = nn.Embedding(max(int(context_cardinalities.get("sex_standardized", 1)), 1), embed_dim)
        self.numeric_encoder = nn.Sequential(
            nn.Linear(numeric_dim, context_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Sequential(
            nn.Linear(embed_dim * 4 + context_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, graph_context: dict[str, torch.Tensor] | None, device: torch.device) -> torch.Tensor:
        if graph_context is None:
            return torch.zeros((self.output[-1].out_features,), device=device)
        pieces = [
            self.source_embedding(graph_context["source_id"]).view(-1),
            self.cohort_embedding(graph_context["cohort_name"]).view(-1),
            self.region_embedding(graph_context["brain_region_standardized"]).view(-1),
            self.sex_embedding(graph_context["sex_standardized"]).view(-1),
            self.numeric_encoder(graph_context["numeric"].view(1, -1)).view(-1),
        ]
        return self.output(torch.cat(pieces, dim=-1))


class EdgeRoleTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Linear(hidden_dim, num_heads)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.context_affine1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.context_affine2 = nn.Linear(hidden_dim, hidden_dim * 2)

    def _apply_context(self, tensor: torch.Tensor, context_vector: torch.Tensor, affine: nn.Linear) -> torch.Tensor:
        gamma, beta = torch.chunk(affine(context_vector), 2, dim=-1)
        return tensor * (1.0 + gamma.unsqueeze(0)) + beta.unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        context_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x_norm = self.norm1(x)
        src = edge_index[0]
        dst = edge_index[1]

        q = self.query_proj(x_norm[dst]).view(-1, self.num_heads, self.head_dim)
        k = self.key_proj(x_norm[src]).view(-1, self.num_heads, self.head_dim)
        v = self.value_proj(x_norm[src]).view(-1, self.num_heads, self.head_dim)
        edge_mod = self.edge_proj(edge_attr).view(-1, self.num_heads, self.head_dim)
        attn_logits = ((q * (k + edge_mod)).sum(dim=-1) / math.sqrt(float(self.head_dim))) + self.edge_bias(edge_attr)
        alpha = scatter_softmax(attn_logits, dst, num_segments=x.shape[0])

        messages = (v + edge_mod) * alpha.unsqueeze(-1)
        agg = torch.zeros((x.shape[0], self.hidden_dim), device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, messages.reshape(messages.shape[0], self.hidden_dim))
        out = self.out_proj(agg)
        out = self._apply_context(out, context_vector, self.context_affine1)
        x = residual + F.dropout(out, p=self.dropout, training=self.training)

        ff = self.ffn(self.norm2(x))
        ff = self._apply_context(ff, context_vector, self.context_affine2)
        x = x + F.dropout(ff, p=self.dropout, training=self.training)
        return x, alpha.mean(dim=1)


class HeteroGraphTransformerEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        *,
        num_heads: int = 4,
        context_cardinalities: dict[str, int] | None = None,
        context_hidden_dim: int = 48,
    ) -> None:
        super().__init__()
        self.node_input = nn.Linear(node_dim, hidden_dim)
        self.edge_input = nn.Linear(edge_dim, hidden_dim)
        self.role_embedding = nn.Embedding(2, hidden_dim)
        self.context_encoder = GraphContextEncoder(
            hidden_dim,
            context_cardinalities,
            numeric_dim=5,
            context_hidden_dim=context_hidden_dim,
            dropout=dropout,
        )
        self.layers = nn.ModuleList([EdgeRoleTransformerLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.dropout = dropout
        self.output_dim = hidden_dim * 7

    def forward(self, graph: dict[str, Any], graph_context: dict[str, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_vector = self.context_encoder(graph_context, graph["node_features"].device)
        x = self.node_input(graph["node_features"]) + self.role_embedding(graph["role_index"])
        x = F.gelu(x)
        edge_attr = F.gelu(self.edge_input(graph["edge_attr"]))
        last_alpha = torch.zeros((graph["edge_index"].shape[1],), device=x.device)

        for layer in self.layers:
            x, last_alpha = layer(x, graph["edge_index"], edge_attr, context_vector)
            x = F.dropout(x, p=self.dropout, training=self.training)

        sender_mask = graph["sender_mask"]
        receiver_mask = graph["receiver_mask"]
        sender_x = x[sender_mask]
        receiver_x = x[receiver_mask]
        sender_mean = sender_x.mean(dim=0) if sender_x.shape[0] else torch.zeros((x.shape[1],), device=x.device)
        receiver_mean = receiver_x.mean(dim=0) if receiver_x.shape[0] else torch.zeros((x.shape[1],), device=x.device)
        sender_max = sender_x.max(dim=0).values if sender_x.shape[0] else torch.zeros((x.shape[1],), device=x.device)
        receiver_max = receiver_x.max(dim=0).values if receiver_x.shape[0] else torch.zeros((x.shape[1],), device=x.device)
        global_mean = x.mean(dim=0)
        edge_mean = edge_attr.mean(dim=0) if edge_attr.shape[0] else torch.zeros((edge_attr.shape[1],), device=x.device)
        graph_embedding = torch.cat(
            [
                sender_mean,
                receiver_mean,
                global_mean,
                sender_max,
                receiver_max,
                context_vector,
                edge_mean,
            ],
            dim=-1,
        )
        return graph_embedding, last_alpha, x


def build_graph_encoder(
    *,
    model_family: str,
    node_dim: int,
    edge_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    num_heads: int = 4,
    context_cardinalities: dict[str, int] | None = None,
    context_hidden_dim: int = 48,
) -> nn.Module:
    family = str(model_family).strip().lower()
    if family in {"legacy_gnn", "edge_aware_gnn", "gnn"}:
        return GraphEncoder(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    if family in {"hetero_transformer", "heterogeneous_graph_transformer", "graph_transformer"}:
        return HeteroGraphTransformerEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads,
            context_cardinalities=context_cardinalities,
            context_hidden_dim=context_hidden_dim,
        )
    raise ValueError(f"Unsupported model_family: {model_family}")


class AgeModel(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        *,
        model_family: str = "legacy_gnn",
        num_heads: int = 4,
        context_cardinalities: dict[str, int] | None = None,
        context_hidden_dim: int = 48,
    ) -> None:
        super().__init__()
        self.encoder = build_graph_encoder(
            model_family=model_family,
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads,
            context_cardinalities=context_cardinalities,
            context_hidden_dim=context_hidden_dim,
        )
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        graph: dict[str, Any],
        graph_context: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding, alpha, node_embeddings = self.encoder(graph, graph_context)
        pred = self.head(embedding).squeeze(-1)
        return pred, embedding, alpha


class AgeConditionedPDModel(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        age_hidden_dim: int,
        num_layers: int,
        dropout: float,
        *,
        model_family: str = "legacy_gnn",
        num_heads: int = 4,
        context_cardinalities: dict[str, int] | None = None,
        context_hidden_dim: int = 48,
        domain_num_classes: int = 0,
        domain_hidden_dim: int = 32,
        gradient_reversal_lambda: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = build_graph_encoder(
            model_family=model_family,
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads,
            context_cardinalities=context_cardinalities,
            context_hidden_dim=context_hidden_dim,
        )
        self.age_encoder = nn.Sequential(
            nn.Linear(1, age_hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim + age_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.age_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.gradient_reversal_lambda = float(gradient_reversal_lambda)
        self.domain_head: nn.Module | None
        if int(domain_num_classes) > 1:
            self.domain_head = nn.Sequential(
                nn.Linear(self.encoder.output_dim, domain_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(domain_hidden_dim, int(domain_num_classes)),
            )
        else:
            self.domain_head = None

    def forward(
        self,
        graph: dict[str, Any],
        age_value: torch.Tensor,
        graph_context: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        embedding, alpha, _ = self.encoder(graph, graph_context)
        age_condition = self.age_encoder(age_value.view(1, 1)).view(-1)
        disease_logit = self.classifier(torch.cat([embedding, age_condition], dim=-1)).squeeze(-1)
        age_pred = self.age_head(embedding).squeeze(-1)
        source_logits = None
        if self.domain_head is not None:
            domain_input = reverse_gradient(embedding, self.gradient_reversal_lambda)
            source_logits = self.domain_head(domain_input)
        return disease_logit, age_pred, embedding, alpha, source_logits
