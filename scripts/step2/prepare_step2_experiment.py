#!/usr/bin/env python3
"""Create isolated Step 2 experiment configs from the current baseline."""

import argparse
import sys
from pathlib import Path
from typing import Any, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_AGE_CONFIG = PROJECT_ROOT / "configs/step2/step2_model_age.yaml"
DEFAULT_PD_CONFIG = PROJECT_ROOT / "configs/step2/step2_model_pd.yaml"
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "configs/step2/experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Step 2 experiment configs.")
    parser.add_argument("--tag", required=True, help="Experiment tag, used as the config directory name.")
    parser.add_argument("--seed", required=True, type=int, help="Random seed applied to age and PD configs.")
    parser.add_argument("--age-config", default=str(DEFAULT_AGE_CONFIG))
    parser.add_argument("--pd-config", default=str(DEFAULT_PD_CONFIG))
    parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--lambda-source", type=float, default=None, help="Optional override for loss.lambda_source.")
    parser.add_argument(
        "--gradient-reversal-lambda",
        type=float,
        default=None,
        help="Optional override for source_invariance.gradient_reversal_lambda.",
    )
    parser.add_argument("--graph-loss-weight", type=float, default=None, help="Optional override for training.graph_loss_weight.")
    parser.add_argument("--label-smoothing", type=float, default=None, help="Optional override for training.label_smoothing.")
    parser.add_argument("--logit-l2-weight", type=float, default=None, help="Optional override for training.logit_l2_weight.")
    parser.add_argument(
        "--exclude-source-id",
        action="append",
        default=[],
        help="Exclude this source_id from Step 2 age/PD training. May be repeated.",
    )
    parser.add_argument(
        "--include-source-id",
        action="append",
        default=[],
        help="Restrict Step 2 age/PD training to this source_id. May be repeated.",
    )
    parser.add_argument(
        "--source-weight-override",
        action="append",
        default=[],
        help="Override source training weight as SOURCE_ID=WEIGHT. May be repeated.",
    )
    return parser.parse_args()


def format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def dump_yaml(data: Any, indent: int = 0) -> List[str]:
    pad = "  " * indent
    lines = []
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{pad}{key}:")
                lines.extend(dump_yaml(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{pad}{key}:")
                for item in value:
                    if isinstance(item, (dict, list)):
                        lines.append(f"{pad}  -")
                        lines.extend(dump_yaml(item, indent + 2))
                    else:
                        lines.append(f"{pad}  - {format_scalar(item)}")
            else:
                lines.append(f"{pad}{key}: {format_scalar(value)}")
        return lines
    if isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.extend(dump_yaml(item, indent + 1))
            else:
                lines.append(f"{pad}- {format_scalar(item)}")
        return lines
    lines.append(f"{pad}{format_scalar(data)}")
    return lines


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    experiment_dir = experiment_root / args.tag
    experiment_dir.mkdir(parents=True, exist_ok=True)

    with open(args.age_config, "r", encoding="utf-8") as handle:
        age_config = yaml.safe_load(handle)
    with open(args.pd_config, "r", encoding="utf-8") as handle:
        pd_config = yaml.safe_load(handle)

    age_config["random_seed"] = int(args.seed)
    pd_config["random_seed"] = int(args.seed)

    if args.lambda_source is not None:
        pd_config.setdefault("loss", {})["lambda_source"] = float(args.lambda_source)
    if args.gradient_reversal_lambda is not None:
        pd_config.setdefault("source_invariance", {})["gradient_reversal_lambda"] = float(args.gradient_reversal_lambda)
    if args.graph_loss_weight is not None:
        pd_config.setdefault("training", {})["graph_loss_weight"] = float(args.graph_loss_weight)
    if args.label_smoothing is not None:
        pd_config.setdefault("training", {})["label_smoothing"] = float(args.label_smoothing)
    if args.logit_l2_weight is not None:
        pd_config.setdefault("training", {})["logit_l2_weight"] = float(args.logit_l2_weight)

    include_sources = [str(value) for value in args.include_source_id if str(value).strip()]
    exclude_sources = [str(value) for value in args.exclude_source_id if str(value).strip()]
    if include_sources or exclude_sources:
        age_filters = age_config.setdefault("data_filters", {})
        pd_filters = pd_config.setdefault("data_filters", {})
        if include_sources:
            age_filters["include_source_ids"] = include_sources
            pd_filters["include_source_ids"] = include_sources
        if exclude_sources:
            age_filters["exclude_source_ids"] = exclude_sources
            pd_filters["exclude_source_ids"] = exclude_sources

    source_weight_overrides: dict[str, float] = {}
    for item in args.source_weight_override:
        if "=" not in str(item):
            raise ValueError(f"Invalid --source-weight-override value: {item}")
        key, raw_value = str(item).split("=", 1)
        source_id = str(key).strip()
        if not source_id:
            raise ValueError(f"Invalid --source-weight-override source: {item}")
        source_weight_overrides[source_id] = float(raw_value)
    if source_weight_overrides:
        age_config.setdefault("weighting", {})["source_weight_overrides"] = source_weight_overrides
        pd_config.setdefault("weighting", {})["source_weight_overrides"] = source_weight_overrides

    age_path = experiment_dir / "step2_model_age.yaml"
    pd_path = experiment_dir / "step2_model_pd.yaml"
    age_path.write_text("\n".join(dump_yaml(age_config)) + "\n", encoding="utf-8")
    pd_path.write_text("\n".join(dump_yaml(pd_config)) + "\n", encoding="utf-8")

    print(age_path)
    print(pd_path)


if __name__ == "__main__":
    main()
