"""Validate the matched train-z-score/window-return pretraining pilot.

The pilot intentionally changes the encoder/pretraining representation while
holding the downstream target fixed as cutoff-relative return.  This checker
fails closed when any other configuration field changes, when chronological
samples or target tensors differ, or when completed runs report incompatible
target/metric definitions.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Mapping

import torch

from config.config_downstream import config as downstream_defaults
from config.experiment import effective_feature_columns
from config.file_options import read_config_file
from run_top_nasdaq100_stocks import (
    effective_experiment_config,
    parse_args as parse_runner_args,
    resolve_mask_strategies,
    resolve_seeds,
    resolve_stocks,
    validate_runner_mask_geometry,
)
from src.data_loaders.data_class_roll_volume import EvaluationDataLoader


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_ZSCORE_CONFIG = (
    REPO_ROOT / "config/experiments/normalization_pilot_train_zscore.json"
)
DEFAULT_WINDOW_RETURN_CONFIG = (
    REPO_ROOT / "config/experiments/normalization_pilot_window_return.json"
)
EXPECTED_RAW_DIFFERENCE = "runner.preprocessing.custom.normalization.method"
EXPECTED_TARGET = "relative_return"
EXPECTED_TARGET_DEFINITION = "Close[t+h] / Close[t] - 1"
METADATA_COMPARISON_FIELDS = (
    "forecast_target",
    "target_definition",
    "metric_definition",
    "forecast_horizon",
    "test_sample_count",
    "test_target_start",
    "test_target_end",
)


def _nested_differences(
    left: Any,
    right: Any,
    prefix: str = "",
) -> dict[str, dict[str, Any]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: dict[str, dict[str, Any]] = {}
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                differences[path] = {
                    "train_zscore": left.get(key),
                    "window_return": right.get(key),
                }
            else:
                differences.update(_nested_differences(left[key], right[key], path))
        return differences
    if left != right:
        return {
            prefix: {
                "train_zscore": left,
                "window_return": right,
            }
        }
    return {}


def _load_pair_args(train_zscore_config: Path, window_return_config: Path):
    train_path = Path(train_zscore_config).resolve()
    window_path = Path(window_return_config).resolve()
    train_args = parse_runner_args(["--config", str(train_path)])
    window_args = parse_runner_args(["--config", str(window_path)])
    for args in (train_args, window_args):
        validate_runner_mask_geometry(args, resolve_mask_strategies(args))
    return train_path, window_path, train_args, window_args


def validate_config_pair(
    train_zscore_config: Path,
    window_return_config: Path,
) -> dict[str, Any]:
    """Require a relative-return pilot differing only by normalization mode."""
    train_path, window_path, train_args, window_args = _load_pair_args(
        train_zscore_config,
        window_return_config,
    )
    _, train_raw = read_config_file(train_path)
    _, window_raw = read_config_file(window_path)
    raw_differences = _nested_differences(train_raw, window_raw)
    if set(raw_differences) != {EXPECTED_RAW_DIFFERENCE}:
        raise ValueError(
            "Pilot configs must differ only at "
            f"{EXPECTED_RAW_DIFFERENCE}; actual differences="
            f"{sorted(raw_differences)}"
        )

    train_effective = effective_experiment_config(train_args)
    window_effective = effective_experiment_config(window_args)
    semantic_differences = {
        key: {
            "train_zscore": train_effective.get(key),
            "window_return": window_effective.get(key),
        }
        for key in sorted(set(train_effective) | set(window_effective))
        if train_effective.get(key) != window_effective.get(key)
    }
    if set(semantic_differences) != {"normalization"}:
        raise ValueError(
            "Resolved pilot configs must differ only in normalization; "
            f"actual differences={sorted(semantic_differences)}"
        )
    if (
        train_args.normalization != "train_zscore"
        or window_args.normalization != "window_return"
    ):
        raise ValueError(
            "Expected train_zscore and window_return normalization modes, got "
            f"{train_args.normalization!r} and {window_args.normalization!r}"
        )
    if (
        train_args.forecast_target != EXPECTED_TARGET
        or window_args.forecast_target != EXPECTED_TARGET
    ):
        raise ValueError(
            "Both pilot configs must use relative_return to guarantee an "
            "identical target space across input normalizations"
        )

    train_stocks = resolve_stocks(train_args)
    window_stocks = resolve_stocks(window_args)
    train_seeds = resolve_seeds(train_args)
    window_seeds = resolve_seeds(window_args)
    train_strategies = resolve_mask_strategies(train_args)
    window_strategies = resolve_mask_strategies(window_args)
    coverage_pairs = (
        ("stocks", train_stocks, window_stocks),
        ("seeds", train_seeds, window_seeds),
        ("mask strategies", train_strategies, window_strategies),
    )
    for label, train_values, window_values in coverage_pairs:
        if train_values != window_values:
            raise ValueError(
                f"Pilot {label} differ: train_zscore={train_values}, "
                f"window_return={window_values}"
            )

    return {
        "valid": True,
        "train_zscore_config": str(train_path),
        "window_return_config": str(window_path),
        "forecast_target": EXPECTED_TARGET,
        "stocks": train_stocks,
        "seeds": train_seeds,
        "mask_strategies": train_strategies,
        "semantic_differences": semantic_differences,
        "raw_difference": EXPECTED_RAW_DIFFERENCE,
    }


def _feature_columns(args) -> list[str]:
    return effective_feature_columns(
        args.market_features,
        args.sentiment_features or downstream_defaults["sentiment_features"],
        bool(args.use_sentiment),
    )


def _dataset_kwargs(args, stock: str, data_path: Path) -> dict[str, Any]:
    sentiment_path = (
        data_path.parent / f"{stock}_daily_sentiment.csv"
        if args.use_sentiment
        else None
    )
    return {
        "path_data": str(data_path),
        "patch_size": int(args.patch_size),
        "forecast_horizon": args.forecast_horizon,
        "context_size": int(downstream_defaults["context_size"]),
        "stride": int(downstream_defaults["eval_stride"]),
        "sampling_mode": args.sampling_mode,
        "feature_cols": _feature_columns(args),
        "target_col": downstream_defaults["target_col"],
        "forecast_target": args.forecast_target,
        "timestamp_col": downstream_defaults["timestamp_col"],
        "validation_fraction": downstream_defaults["validation_fraction"],
        "sentiment_path": None if sentiment_path is None else str(sentiment_path),
        "feature_transform": args.feature_transform,
        "market_data": args.market_data,
        "sentiment_normalization": args.sentiment_normalization,
        "robust_zscore_clip": args.robust_zscore_clip,
        "train_end_date": downstream_defaults["train_end_date"],
        "test_start_date": downstream_defaults["test_start_date"],
        "data_end_date": downstream_defaults["data_end_date"],
    }


def _build_split_datasets(args, stock: str, data_path: Path):
    common = _dataset_kwargs(args, stock, data_path)
    train = EvaluationDataLoader(
        split="train",
        normalization=args.normalization,
        **common,
    )
    datasets = {"train": train}
    for split in ("val", "test"):
        datasets[split] = EvaluationDataLoader(
            split=split,
            normalization=args.normalization,
            normalization_stats=train.normalization_stats,
            sentiment_normalization_stats=train.sentiment_normalization_stats,
            **common,
        )
    return datasets


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _stack_dataset_items(dataset, position: int) -> torch.Tensor:
    return torch.stack([dataset[index][position] for index in range(len(dataset))])


def compare_target_datasets(
    train_zscore_config: Path,
    window_return_config: Path,
    *,
    repo_root: Path = REPO_ROOT,
    data_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Compare every chronological target tensor for the matched pilot pair."""
    config_report = validate_config_pair(
        train_zscore_config,
        window_return_config,
    )
    _, _, train_args, window_args = _load_pair_args(
        train_zscore_config,
        window_return_config,
    )
    root = Path(repo_root).resolve()
    stock_reports: dict[str, Any] = {}
    for stock in config_report["stocks"]:
        data_path = (
            Path(data_paths[stock])
            if data_paths is not None and stock in data_paths
            else root / "data" / stock / f"{stock}.csv"
        )
        if not data_path.exists():
            raise FileNotFoundError(f"Pilot data file not found: {data_path}")
        # The project loaders emit human-oriented preprocessing diagnostics.
        # Keep this validation command's stdout as one parseable JSON document.
        with redirect_stdout(io.StringIO()):
            train_datasets = _build_split_datasets(train_args, stock, data_path)
            window_datasets = _build_split_datasets(window_args, stock, data_path)
        split_reports = {}
        for split in ("train", "val", "test"):
            train_dataset = train_datasets[split]
            window_dataset = window_datasets[split]
            if len(train_dataset) != len(window_dataset):
                raise ValueError(
                    f"{stock} {split} sample counts differ: "
                    f"train_zscore={len(train_dataset)}, "
                    f"window_return={len(window_dataset)}"
                )
            train_targets = _stack_dataset_items(train_dataset, 1)
            window_targets = _stack_dataset_items(window_dataset, 1)
            train_contexts = _stack_dataset_items(train_dataset, 0)
            window_contexts = _stack_dataset_items(window_dataset, 0)
            dates_equal = train_dataset.dates == window_dataset.dates
            starts_equal = (
                train_dataset.sample_starts == window_dataset.sample_starts
            )
            targets_equal = torch.equal(train_targets, window_targets)
            contexts_differ = not torch.equal(train_contexts, window_contexts)
            if not dates_equal or not starts_equal or not targets_equal:
                raise ValueError(
                    f"{stock} {split} is not target-identical: "
                    f"dates_equal={dates_equal}, "
                    f"sample_starts_equal={starts_equal}, "
                    f"targets_bitwise_equal={targets_equal}"
                )
            if not contexts_differ:
                raise ValueError(
                    f"{stock} {split} contexts are identical; normalization "
                    "did not change the model representation"
                )
            split_reports[split] = {
                "sample_count": len(train_dataset),
                "dates_equal": dates_equal,
                "sample_starts_equal": starts_equal,
                "targets_bitwise_equal": targets_equal,
                "contexts_differ": contexts_differ,
                "train_zscore_target_sha256": _tensor_sha256(train_targets),
                "window_return_target_sha256": _tensor_sha256(window_targets),
            }
        stock_reports[stock] = split_reports
    return {
        "valid": True,
        "forecast_target": EXPECTED_TARGET,
        "stocks": stock_reports,
    }


def _metadata_files(root: Path) -> dict[Path, Path]:
    resolved = Path(root).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Result root not found: {resolved}")
    files = {
        path.relative_to(resolved): path
        for path in resolved.rglob("preprocessing_config.json")
    }
    if not files:
        raise ValueError(f"No preprocessing_config.json files found under {resolved}")
    return files


def validate_result_metadata_pair(
    train_zscore_results: Path,
    window_return_results: Path,
) -> dict[str, Any]:
    """Verify completed paired runs report one target and metric space."""
    train_files = _metadata_files(Path(train_zscore_results))
    window_files = _metadata_files(Path(window_return_results))
    if set(train_files) != set(window_files):
        missing_train = sorted(str(path) for path in set(window_files) - set(train_files))
        missing_window = sorted(str(path) for path in set(train_files) - set(window_files))
        raise ValueError(
            "Paired result metadata coverage differs: "
            f"missing_train_zscore={missing_train}, "
            f"missing_window_return={missing_window}"
        )

    for relative_path in sorted(train_files):
        train_metadata = json.loads(
            train_files[relative_path].read_text(encoding="utf-8")
        )
        window_metadata = json.loads(
            window_files[relative_path].read_text(encoding="utf-8")
        )
        if train_metadata.get("normalization") != "train_zscore":
            raise ValueError(
                f"{relative_path}: expected train_zscore result normalization"
            )
        if window_metadata.get("normalization") != "window_return":
            raise ValueError(
                f"{relative_path}: expected window_return result normalization"
            )
        for field in METADATA_COMPARISON_FIELDS:
            if train_metadata.get(field) != window_metadata.get(field):
                raise ValueError(
                    f"{relative_path}: result metadata field {field!r} differs: "
                    f"train_zscore={train_metadata.get(field)!r}, "
                    f"window_return={window_metadata.get(field)!r}"
                )
        if train_metadata.get("forecast_target") != EXPECTED_TARGET:
            raise ValueError(
                f"{relative_path}: forecast_target must be {EXPECTED_TARGET!r}"
            )
        if train_metadata.get("target_definition") != EXPECTED_TARGET_DEFINITION:
            raise ValueError(
                f"{relative_path}: target_definition must be "
                f"{EXPECTED_TARGET_DEFINITION!r}"
            )

    return {
        "valid": True,
        "matched_runs": len(train_files),
        "forecast_target": EXPECTED_TARGET,
        "target_definition": EXPECTED_TARGET_DEFINITION,
        "metadata_fields_compared": list(METADATA_COMPARISON_FIELDS),
    }


def _parse_cli(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Validate that the normalization pilot changes only model input "
            "normalization while preserving relative-return targets."
        )
    )
    parser.add_argument(
        "--train-zscore-config",
        type=Path,
        default=DEFAULT_TRAIN_ZSCORE_CONFIG,
    )
    parser.add_argument(
        "--window-return-config",
        type=Path,
        default=DEFAULT_WINDOW_RETURN_CONFIG,
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--check-results",
        action="store_true",
        help="Also validate paired preprocessing_config.json run metadata.",
    )
    parser.add_argument("--train-zscore-results", type=Path)
    parser.add_argument("--window-return-results", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_cli(argv)
    config_report = validate_config_pair(
        args.train_zscore_config,
        args.window_return_config,
    )
    target_report = compare_target_datasets(
        args.train_zscore_config,
        args.window_return_config,
        repo_root=args.repo_root,
    )
    report: dict[str, Any] = {
        "valid": True,
        "config": config_report,
        "targets": target_report,
    }
    if args.check_results:
        train_results = args.train_zscore_results or (
            Path(args.repo_root) / "results" / Path(args.train_zscore_config).stem
        )
        window_results = args.window_return_results or (
            Path(args.repo_root) / "results" / Path(args.window_return_config).stem
        )
        report["results"] = validate_result_metadata_pair(
            train_results,
            window_results,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
