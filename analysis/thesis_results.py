"""Audit, normalize, and analyse saved TS-JEPA forecasting experiments.

The module never trains a model.  It treats timestamped comparison/score files
as immutable source data, preserves the stock/seed hierarchy, and refuses to
pool incompatible forecasting procedures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import shlex
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from config.file_options import (
    flatten_runner_options,
    read_config_file,
    results_dir_from_config,
)


METHOD_ORDER = (
    "Shared-target JEPA--MAE",
    "Local-MAE/Long-JEPA",
    "GRU",
    "Naive-last",
    "Drift",
    "Mean-context",
)
LEARNED_METHODS = METHOD_ORDER[:3]
STRATEGY_METHODS = {
    "random": "Shared-target JEPA--MAE",
    "local_long": "Local-MAE/Long-JEPA",
}
RAW_MODEL_METHODS = {
    "GRU": "GRU",
    "naive_last": "Naive-last",
    "drift": "Drift",
    "mean_context": "Mean-context",
}
REFERENCE_METHOD = "Naive-last"
METRICS = ("mse", "mae", "direction_accuracy")
LOWER_IS_BETTER = {"mse": True, "mae": True, "direction_accuracy": False}
DIRECTION_DEFINITION = (
    "project_within_trajectory_v1: sign of consecutive forecast-horizon "
    "differences equals sign of consecutive target differences; relative-return "
    "paths additionally include the known zero origin; cumulative/excess "
    "log-return targets compare the binary indicators (forecast > 0) and "
    "(target > 0) at each horizon"
)
TIDY_COLUMNS = (
    "stock",
    "seed",
    "method",
    "strategy",
    "split",
    "mse",
    "mae",
    "direction_accuracy",
    "forecast_horizon",
    "checkpoint_path",
    "checkpoint_selection",
    "encoder_weight_source",
    "fine_tune_encoder",
    "trend_weight",
    "trend_loss_temperature",
    "trend_loss_threshold",
    "trend_selection_weight",
    "normalization",
    "feature_transform",
    "forecast_target",
    "target_definition",
    "metric_definition",
    "test_start",
    "test_end",
    "test_signature",
    "experiment_id",
    "config_signature",
    "timestamp",
    "git_commit_sha",
    "original_source_file",
    "score_source_file",
)
INVENTORY_COLUMNS = TIDY_COLUMNS + (
    "raw_model",
    "selected_bundle",
    "included_in_canonical",
    "exclusion_reason",
    "metadata_sources",
)
ISSUE_COLUMNS = (
    "severity",
    "status",
    "stock",
    "seed",
    "method",
    "strategy",
    "source_file",
    "details",
)
PREDICTION_COLUMNS = (
    "stock",
    "seed",
    "method",
    "strategy",
    "rolling_step",
    "horizon",
    "target_index",
    "target_date",
    "predicted_value",
    "true_value",
    "source_file",
)
PAIRED_RUN_COLUMNS = (
    "stock",
    "seed",
    "model",
    "strategy",
    "test_signature",
    "model_mse",
    "naive_mse",
    "delta_mse",
    "relative_mse_improvement_pct",
    "model_mae",
    "naive_mae",
    "delta_mae",
    "relative_mae_improvement_pct",
    "model_source_file",
    "naive_source_file",
)
THESIS_TABLE_FILES = (
    "table_main_metrics.csv",
    "table_main_metrics.tex",
    "table_paired_vs_naive.csv",
    "table_paired_vs_naive.tex",
    "table_shared_vs_local.csv",
    "table_shared_vs_local.tex",
    "table_appendix_stock_metrics.csv",
    "table_appendix_stock_metrics.tex",
    "table_reproducibility.tex",
)
THESIS_FIGURE_STEMS = (
    "fig_paired_mse_forest",
    "fig_paired_mae_forest",
    "fig_relative_mse_heatmap",
    "fig_relative_mae_heatmap",
    "fig_direction_accuracy_heatmap",
    "fig_mse_by_horizon",
    "fig_mae_by_horizon",
    "fig_direction_by_horizon",
    "fig_seed_level_delta_mse_distribution",
    "fig_representative_prediction_trajectory",
)
THESIS_DIAGNOSTIC_STEMS = (
    "diagnostic_downstream_training_loss",
    "diagnostic_downstream_validation_mse",
    "diagnostic_pretraining_losses",
)
TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")
SEED_RE = re.compile(r"^seed_(-?\d+)$")


@dataclass
class Bundle:
    strategy: str
    stock: str
    seed: int
    timestamp: str
    comparison_path: Path
    metadata: dict[str, Any]
    rows: list[dict[str, Any]] = field(default_factory=list)
    predictions: dict[str, pd.DataFrame] = field(default_factory=dict)
    issues: list[dict[str, Any]] = field(default_factory=list)
    selected: bool = False

    @property
    def key(self) -> tuple[str, str, int]:
        return self.strategy, self.stock, self.seed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build auditable thesis tables and figures from saved runs only."
    )
    parser.add_argument(
        "--config",
        default="config/experiments/top10_with_sentiment.json",
        help="Experiment JSON used to define expected stocks, seeds, and strategies.",
    )
    parser.add_argument("--stocks", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument(
        "--reference-strategy",
        default="random",
        help="Strategy whose GRU and deterministic baselines define main-table rows.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=20_000,
        help="Number of deterministic stock-level bootstrap resamples.",
    )
    parser.add_argument("--analysis-seed", type=int, default=20260822)
    parser.add_argument(
        "--allow-incomplete",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate explicitly exploratory summaries from compatible available runs.",
    )
    parser.add_argument(
        "--skip-figures",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_scope(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    config_path = Path(args.config)
    config_data = read_config_file(config_path)[1] if config_path.exists() else {}
    common = config_data.get("common", {})
    runner = flatten_runner_options(config_data.get("runner", {}), config_path)
    analysis = config_data.get("analysis", {})
    stocks = args.stocks or common.get("stocks") or []
    seeds = args.seeds or common.get("seeds") or []
    strategies = (
        args.strategies
        or runner.get("mask_strategies")
        or analysis.get("strategies")
        or ["random", "local_long"]
    )
    results_dir = results_dir_from_config(config_path)
    scope = {
        "config_path": str(config_path),
        "config_data": config_data,
        "stocks": [str(stock).upper() for stock in stocks],
        "seeds": [int(seed) for seed in seeds],
        "strategies": [str(strategy) for strategy in strategies],
        "reference_strategy": args.reference_strategy,
    }
    for key in ("stocks", "seeds", "strategies"):
        values = scope[key]
        if not values:
            raise ValueError(f"No expected {key} are configured")
        if len(values) != len(set(values)):
            raise ValueError(f"Expected {key} must be unique: {values}")
    if scope["reference_strategy"] not in scope["strategies"]:
        raise ValueError(
            "--reference-strategy must be one of the analysed strategies: "
            f"{scope['strategies']}"
        )
    return scope, results_dir


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def issue(
    status: str,
    details: str,
    *,
    severity: str = "error",
    stock: str = "",
    seed: int | str = "",
    method: str = "",
    strategy: str = "",
    source_file: str = "",
) -> dict[str, Any]:
    return {
        "severity": severity,
        "status": status,
        "stock": stock,
        "seed": seed,
        "method": method,
        "strategy": strategy,
        "source_file": source_file,
        "details": details,
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    rng: np.random.Generator,
    samples: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan")
    if len(array) == 1:
        return float(array[0]), float(array[0])
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """One-based average ranks with deterministic tie handling."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def exact_wilcoxon(values: Sequence[float]) -> tuple[float, float, float, int]:
    """Exact two-sided signed-rank permutation test and rank-biserial effect.

    Zero differences are removed (Wilcox convention).  The returned effect is
    signed in delta coordinates: negative values favour the model because the
    analysed delta is model minus reference.
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array) & (array != 0.0)]
    n = len(array)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    ranks = average_ranks(np.abs(array))
    positive = float(ranks[array > 0].sum())
    negative = float(ranks[array < 0].sum())
    statistic = min(positive, negative)
    total = float(ranks.sum())
    signed_sums = np.empty(1 << n, dtype=float)
    for mask in range(1 << n):
        signed_sums[mask] = sum(
            ranks[index] for index in range(n) if mask & (1 << index)
        )
    tolerance = 1e-12
    lower = float(np.mean(signed_sums <= positive + tolerance))
    upper = float(np.mean(signed_sums >= positive - tolerance))
    p_value = min(1.0, 2.0 * min(lower, upper))
    effect = (positive - negative) / total
    return statistic, p_value, float(effect), n


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by the incomplete beta function."""
    maximum_iterations = 300
    tolerance = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    value = d
    for iteration in range(1, maximum_iterations + 1):
        m2 = 2 * iteration
        numerator = iteration * (b - iteration) * x / (
            (qam + m2) * (a + m2)
        )
        d = 1.0 + numerator * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + numerator / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        value *= d * c
        numerator = -(a + iteration) * (qab + iteration) * x / (
            (a + m2) * (qap + m2)
        )
        d = 1.0 + numerator * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + numerator / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        value *= delta
        if abs(delta - 1.0) < tolerance:
            return value
    raise RuntimeError("Incomplete-beta continued fraction did not converge")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _student_t_cdf(value: float, degrees_freedom: int) -> float:
    if degrees_freedom <= 0:
        raise ValueError("Student-t degrees of freedom must be positive")
    if math.isinf(value):
        return 1.0 if value > 0 else 0.0
    x = degrees_freedom / (degrees_freedom + value * value)
    tail = 0.5 * _regularized_incomplete_beta(
        degrees_freedom / 2.0,
        0.5,
        x,
    )
    return 1.0 - tail if value >= 0 else tail


def paired_difference_statistics(values: Sequence[float]) -> dict[str, Any]:
    """Return paired t-test and signed Cohen's dz for stock-level differences."""
    differences = np.asarray(values, dtype=float)
    if differences.ndim != 1:
        raise ValueError("Paired differences must be one-dimensional")
    if not np.isfinite(differences).all():
        raise ValueError("Paired differences must be finite")
    count = len(differences)
    mean = float(np.mean(differences)) if count else float("nan")
    median = float(np.median(differences)) if count else float("nan")
    result = {
        "n_stocks": count,
        "mean_delta": mean,
        "median_delta": median,
        "t_statistic": float("nan"),
        "p_value": float("nan"),
        "cohens_dz": float("nan"),
        "status": "insufficient_stock_observations",
    }
    if count < 2:
        return result
    standard_deviation = float(np.std(differences, ddof=1))
    scale = max(1.0, float(np.max(np.abs(differences))))
    roundoff_tolerance = 16.0 * np.finfo(float).eps * scale
    if float(np.ptp(differences)) <= roundoff_tolerance:
        result["status"] = "zero_variance_differences"
        return result
    t_statistic = mean / (standard_deviation / math.sqrt(count))
    p_value = 2.0 * (1.0 - _student_t_cdf(abs(t_statistic), count - 1))
    result.update(
        {
            "t_statistic": t_statistic,
            "p_value": max(0.0, min(1.0, p_value)),
            "cohens_dz": mean / standard_deviation,
            "status": "ok",
        }
    )
    return result


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    finite = sorted(
        ((key, float(value)) for key, value in p_values.items() if math.isfinite(value)),
        key=lambda item: item[1],
    )
    adjusted: dict[str, float] = {key: float("nan") for key in p_values}
    running = 0.0
    count = len(finite)
    for index, (key, value) in enumerate(finite):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[key] = running
    return adjusted


def compute_direction_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    forecast_target: str,
) -> float:
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if predictions.shape != targets.shape or predictions.ndim != 2:
        raise ValueError(
            "Direction accuracy requires equal [rolling_step, horizon] arrays; "
            f"got predictions={predictions.shape}, targets={targets.shape}"
        )
    if forecast_target in ("cumulative_log_return", "excess_log_return"):
        return float(((predictions > 0) == (targets > 0)).mean())
    if forecast_target == "relative_return":
        origin = np.zeros((predictions.shape[0], 1), dtype=float)
        predictions = np.concatenate([origin, predictions], axis=1)
        targets = np.concatenate([origin, targets], axis=1)
    if predictions.shape[1] < 2:
        return float("nan")
    pred_direction = np.sign(np.diff(predictions, axis=1))
    target_direction = np.sign(np.diff(targets, axis=1))
    return float((pred_direction == target_direction).mean())


def direction_by_horizon(frame: pd.DataFrame, forecast_target: str) -> dict[int, float]:
    pivot_pred = frame.pivot(index="rolling_step", columns="horizon", values="predicted_value")
    pivot_true = frame.pivot(index="rolling_step", columns="horizon", values="true_value")
    horizons = sorted(set(pivot_pred.columns) & set(pivot_true.columns))
    output: dict[int, float] = {}
    for horizon in horizons:
        pred = pivot_pred[horizon].to_numpy(float)
        true = pivot_true[horizon].to_numpy(float)
        if forecast_target in ("cumulative_log_return", "excess_log_return"):
            output[int(horizon)] = float(((pred > 0) == (true > 0)).mean())
        elif horizon == 1 and forecast_target == "relative_return":
            output[int(horizon)] = float((np.sign(pred) == np.sign(true)).mean())
        elif horizon == 1:
            output[int(horizon)] = float("nan")
        elif horizon - 1 in pivot_pred and horizon - 1 in pivot_true:
            pred_prev = pivot_pred[horizon - 1].to_numpy(float)
            true_prev = pivot_true[horizon - 1].to_numpy(float)
            output[int(horizon)] = float(
                (np.sign(pred - pred_prev) == np.sign(true - true_prev)).mean()
            )
        else:
            output[int(horizon)] = float("nan")
    return output


def _parse_text_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if not path.exists():
        return metadata
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            normalized = key.strip().lower().replace(" ", "_")
            if normalized in {
                "data_source",
                "evaluation_type",
                "forecast_target",
                "feature_transform",
                "features",
                "normalization",
                "market_data",
                "generated_at",
            }:
                metadata[normalized] = value.strip()
    except OSError:
        return {}
    return metadata


def _parse_cli_options(tokens: Sequence[str]) -> dict[str, Any]:
    """Extract scalar CLI options needed by the reproducibility audit."""
    options: dict[str, Any] = {}
    scalar_names = {
        "--pretrain-checkpoint-path": "checkpoint_path",
        "--pretrain_checkpoint_path": "checkpoint_path",
        "--pretrain-encoder-weights": "encoder_weight_source",
        "--pretrain_encoder_weights": "encoder_weight_source",
        "--checkpoint-selection": "checkpoint_selection",
        "--checkpoint_to_use": "checkpoint_epoch",
        "--checkpoint-to-use": "checkpoint_epoch",
        "--forecast-target": "forecast_target",
        "--forecast_target": "forecast_target",
        "--normalization": "normalization",
        "--feature-transform": "feature_transform",
        "--feature_transform": "feature_transform",
        "--patch-size": "patch_size",
        "--patch_size": "patch_size",
        "--context-size": "context_size",
        "--eval-stride": "eval_stride",
        "--seed": "seed",
        "--num_epochs": "downstream_epochs",
        "--batch_size": "batch_size",
        "--lr": "forecast_head_lr",
        "--lr_pretrain": "pretrain_lr",
        "--ema_pretrain": "ema_momentum",
        "--mask_ratio": "mask_ratio",
        "--lambda_jepa": "lambda_jepa",
        "--lambda-jepa": "lambda_jepa",
        "--lambda_mae": "lambda_mae",
        "--lambda-mae": "lambda_mae",
        "--mae-window-patches": "mae_window_patches",
        "--jepa-gap-patches": "jepa_gap_patches",
        "--jepa-target-patches": "jepa_target_patches",
        "--pretrain_encoder_embed_dim": "encoder_dim",
        "--pretrain_encoder_nhead": "encoder_heads",
        "--pretrain_encoder_num_layers": "encoder_layers",
        "--pretrain_decoder_embed_dim": "predictor_dim",
        "--pretrain_decoder_nhead": "predictor_heads",
        "--pretrain_decoder_num_layers": "predictor_layers",
        "--train_end_date": "train_end",
        "--test_start_date": "test_start",
        "--data_end_date": "test_end",
        "--results_dir": "results_dir",
        "--results-dir": "results_dir",
    }
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in scalar_names and index + 1 < len(tokens):
            options[scalar_names[token]] = tokens[index + 1]
            index += 2
            continue
        if token in ("--fine-tune-encoder", "--fine_tune_encoder"):
            options["fine_tune_encoder"] = True
        elif token in ("--no-fine-tune-encoder", "--no-fine_tune_encoder"):
            options["fine_tune_encoder"] = False
        index += 1
    return options


def _load_runner_commands(results_dir: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    path = results_dir / "top_nasdaq100_stock_runs.txt"
    commands: dict[tuple[str, str, int], dict[str, Any]] = defaultdict(dict)
    if not path.exists():
        return commands
    pattern = re.compile(r"^([^/]+)/([^[]+)\[seed=(-?\d+)\]:\s+(.+)$")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return commands
    for line in lines:
        match = pattern.match(line)
        if not match:
            continue
        strategy, stock, seed_text, command_text = match.groups()
        try:
            tokens = shlex.split(command_text)
        except ValueError:
            continue
        values = _parse_cli_options(tokens)
        values["runner_command"] = command_text
        values["runner_command_source"] = str(path)
        key = strategy, stock.upper(), int(seed_text)
        if any("eval_dual_loss.py" in token for token in tokens):
            commands[key].update(values)
        else:
            for field_name in (
                "lambda_jepa",
                "lambda_mae",
                "mae_window_patches",
                "jepa_gap_patches",
                "jepa_target_patches",
                "mask_ratio",
                "patch_size",
            ):
                if field_name in values:
                    commands[key][field_name] = values[field_name]
    return commands


def _manifest_metadata(scope: Mapping[str, Any], results_dir: Path) -> dict[str, Any]:
    config_data = scope.get("config_data", {})
    common = dict(config_data.get("common", {}))
    config_path = Path(str(scope.get("config_path", "<config>")))
    runner = flatten_runner_options(config_data.get("runner", {}), config_path)
    metadata = {**common, **runner}
    runtime_manifest = results_dir / "experiment_manifest.json"
    if runtime_manifest.exists():
        runtime = _load_json(runtime_manifest)
        arguments = runtime.get("arguments")
        if isinstance(arguments, dict):
            metadata.update(arguments)
        metadata.update({key: value for key, value in runtime.items() if key != "arguments"})
        metadata["runtime_manifest_path"] = str(runtime_manifest)
    metadata["config_path"] = scope.get("config_path")
    return metadata


def _load_checkpoint_metadata(path_value: Any) -> tuple[dict[str, Any], str | None]:
    """Read configuration fields from a trusted, locally generated checkpoint."""
    if not path_value:
        return {}, None
    path = Path(str(path_value)).expanduser()
    if not path.exists():
        return {}, f"checkpoint path does not exist: {path}"
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {}, f"cannot load checkpoint metadata from {path}: {exc}"
    if not isinstance(checkpoint, dict):
        return {}, f"checkpoint is not a mapping: {path}"
    config = checkpoint.get("config", {})
    metadata = dict(config) if isinstance(config, dict) else {}
    metadata.update(
        {
            "checkpoint_path": str(path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_strategy": checkpoint.get("strategy"),
            "checkpoint_has_encoder_ema": "encoder_ema" in checkpoint,
            "checkpoint_metadata_source": str(path),
        }
    )
    return metadata, None


def _run_identity(path: Path, results_dir: Path, strategies: Sequence[str]) -> tuple[str, str, int]:
    relative_parts = path.relative_to(results_dir).parts
    seed_index = None
    seed = None
    for index, part in enumerate(relative_parts):
        match = SEED_RE.match(part)
        if match:
            seed_index = index
            seed = int(match.group(1))
            break
    if seed_index is None or seed_index == 0:
        raise ValueError("comparison file is not nested under STOCK/seed_N")
    stock = relative_parts[seed_index - 1].upper()
    strategy = "legacy"
    for part in relative_parts[:seed_index]:
        if part in strategies:
            strategy = part
            break
    return strategy, stock, int(seed)


def _normalise_metadata(
    metadata: dict[str, Any],
    *,
    strategy: str,
    comparison_text: Mapping[str, str],
) -> dict[str, Any]:
    aliases = {
        "pretrain_checkpoint_path": "checkpoint_path",
        "pretrain_encoder_weights": "encoder_weight_source",
        "encoder_weights": "encoder_weight_source",
        "use_best_checkpoint": "use_best_checkpoint",
        "checkpoint_to_use": "checkpoint_epoch",
        "num_epochs": "downstream_epochs",
        "eval_num_epochs": "downstream_epochs",
        "pretrain_num_epochs": "pretrain_epochs",
        "series_split_size": "pretrain_context_length",
        "pretrain_stride": "pretrain_stride",
        "patch_size": "patch_size",
        "context_size": "context_size",
        "eval_stride": "eval_stride",
        "test_start_date": "test_start",
        "data_end_date": "test_end",
        "train_end_date": "train_end",
        "encoder_embed_dim": "encoder_dim",
        "encoder_nhead": "encoder_heads",
        "encoder_num_layers": "encoder_layers",
        "predictor_embed": "predictor_dim",
        "predictor_nhead": "predictor_heads",
        "predictor_num_layers": "predictor_layers",
        "feature_names": "feature_cols",
    }
    normalized = dict(metadata)
    for old, new in aliases.items():
        if old in metadata and new not in normalized:
            normalized[new] = metadata[old]
    for key in (
        "forecast_target",
        "feature_transform",
        "normalization",
    ):
        if comparison_text.get(key):
            normalized[key] = comparison_text[key]
    if comparison_text.get("features"):
        normalized["feature_cols"] = [
            item.strip() for item in comparison_text["features"].split(",") if item.strip()
        ]
    normalized["split"] = "test"
    normalized["strategy"] = strategy
    if normalized.get("test_target_start"):
        normalized["test_start"] = normalized["test_target_start"]
    if normalized.get("test_target_end"):
        normalized["test_end"] = normalized["test_target_end"]
    patch_size = normalized.get("patch_size")
    try:
        normalized["forecast_horizon"] = int(patch_size)
    except (TypeError, ValueError):
        normalized["forecast_horizon"] = None
    if normalized.get("use_best_checkpoint") is True:
        normalized["checkpoint_selection"] = "best_pretraining_validation"
    elif (
        normalized.get("checkpoint_selection") == "fixed_pretraining_epoch"
        and normalized.get("checkpoint_epoch") is not None
    ):
        normalized["checkpoint_selection"] = (
            f"fixed_pretraining_epoch_{normalized['checkpoint_epoch']}"
        )
    elif normalized.get("checkpoint_selection") is None and normalized.get("checkpoint_epoch") is not None:
        normalized["checkpoint_selection"] = f"fixed_epoch_{normalized['checkpoint_epoch']}"
    normalized.setdefault("encoder_weight_source", None)
    normalized.setdefault("fine_tune_encoder", None)
    normalized.setdefault("trend_weight", None)
    normalized.setdefault("trend_loss_temperature", None)
    normalized.setdefault("trend_loss_threshold", None)
    normalized.setdefault("trend_selection_weight", None)
    normalized.setdefault("forecast_target", None)
    normalized.setdefault("feature_transform", None)
    normalized.setdefault("normalization", None)
    target = normalized.get("forecast_target")
    normalized.setdefault(
        "target_definition",
        {
            "value": "future target values in the configured normalized feature space",
            "relative_return": "Close[t+h] / Close[t] - 1",
            "cumulative_log_return": "log(Close[t+h] / Close[t])",
            "excess_log_return": (
                "log(Close[t+h] / Close[t]) - log(Market[t+h] / Market[t])"
            ),
        }.get(target),
    )
    normalized["metric_definition"] = (
        "MSE and MAE over all saved rolling-step/horizon target values; "
        + DIRECTION_DEFINITION
    )
    normalized.setdefault("experiment_id", normalized.get("config_fingerprint"))
    return normalized


def _configuration_signature(metadata: Mapping[str, Any]) -> str:
    keys = (
        "forecast_horizon",
        "forecast_target",
        "target_definition",
        "feature_transform",
        "feature_cols",
        "normalization",
        "test_start",
        "test_end",
        "sampling_mode",
        "eval_stride",
        "checkpoint_selection",
        "encoder_weight_source",
        "fine_tune_encoder",
        "trend_weight",
        "trend_loss_temperature",
        "trend_loss_threshold",
        "trend_selection_weight",
        "lambda_jepa",
        "lambda_mae",
    )
    return stable_hash({key: metadata.get(key) for key in keys})[:16]


def _score_path_for_model(bundle: Bundle, raw_model: str) -> Path | None:
    directory = bundle.comparison_path.parent
    timestamp = bundle.timestamp
    prefix = bundle.comparison_path.name.split("model_comparison_", 1)[0]
    if raw_model == "TS-JEPA":
        pattern = f"{prefix}scores_after_observation_{timestamp}.csv"
    elif raw_model == "GRU":
        pattern = f"{prefix}gru_scores_after_observation_{timestamp}.csv"
    else:
        pattern = f"{prefix}baseline_scores_after_observation_{timestamp}.csv"
    exact = directory / pattern
    if exact.exists():
        return exact

    # Writers obtain timestamps independently; accept only an unambiguous file
    # within two seconds of the comparison timestamp.
    stem_middle = {
        "TS-JEPA": "scores_after_observation",
        "GRU": "gru_scores_after_observation",
    }.get(raw_model, "baseline_scores_after_observation")
    candidates = list(directory.glob(f"{prefix}{stem_middle}_*.csv"))
    try:
        target_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    nearby = []
    for candidate in candidates:
        match = TIMESTAMP_RE.search(candidate.name)
        if not match:
            continue
        candidate_time = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        distance = abs((candidate_time - target_time).total_seconds())
        if distance <= 2:
            nearby.append((distance, candidate.name, candidate))
    if not nearby:
        return None
    nearby.sort()
    if len(nearby) > 1 and nearby[0][0] == nearby[1][0]:
        return None
    return nearby[0][2]


def _load_score_frame(path: Path, raw_model: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "model" in frame.columns:
        frame = frame[frame["model"].astype(str) == raw_model].copy()
    required = {"rolling_step", "horizon_step", "predicted_value", "true_value"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing score columns {missing}")
    frame = frame.rename(columns={"horizon_step": "horizon"})
    frame["rolling_step"] = pd.to_numeric(frame["rolling_step"], errors="coerce")
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce")
    for column in ("predicted_value", "true_value"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "target_index" not in frame:
        frame["target_index"] = np.nan
    if "target_date" not in frame:
        frame["target_date"] = ""
    post_required = ["rolling_step", "horizon", "predicted_value", "true_value"]
    if frame.empty or frame[post_required].isna().any().any():
        raise ValueError("score rows are empty or contain non-finite required values")
    if frame.duplicated(["rolling_step", "horizon"]).any():
        raise ValueError("duplicate rolling_step/horizon score rows")
    horizon_counts = frame.groupby("rolling_step")["horizon"].nunique()
    if horizon_counts.nunique() != 1:
        raise ValueError("inconsistent forecast horizon across rolling steps")
    return frame.sort_values(["rolling_step", "horizon"]).reset_index(drop=True)


def _arrays_from_scores(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pred = frame.pivot(index="rolling_step", columns="horizon", values="predicted_value")
    true = frame.pivot(index="rolling_step", columns="horizon", values="true_value")
    if pred.isna().any().any() or true.isna().any().any() or pred.shape != true.shape:
        raise ValueError("score rows do not form a complete rolling-step/horizon matrix")
    return pred.to_numpy(float), true.to_numpy(float)


def _test_signature(frame: pd.DataFrame) -> str:
    columns = ["rolling_step", "horizon", "target_index", "target_date", "true_value"]
    records = frame[columns].fillna("").to_dict(orient="records")
    return stable_hash(records)[:20]


def discover_bundles(
    scope: Mapping[str, Any], results_dir: Path
) -> tuple[list[Bundle], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    if not results_dir.exists():
        issues.append(
            issue(
                "results_directory_missing",
                f"Configured result root does not exist: {results_dir}",
                source_file=str(results_dir),
            )
        )
        return [], issues
    manifest = _manifest_metadata(scope, results_dir)
    runner_commands = _load_runner_commands(results_dir)
    checkpoint_cache: dict[str, tuple[dict[str, Any], str | None]] = {}
    comparison_paths = sorted(results_dir.rglob("*_model_comparison_*.csv"))
    bundles: list[Bundle] = []
    for comparison_path in comparison_paths:
        match = TIMESTAMP_RE.search(comparison_path.name)
        if not match:
            issues.append(
                issue(
                    "invalid_result_filename",
                    "Comparison filename has no YYYYMMDD_HHMMSS timestamp",
                    source_file=str(comparison_path),
                )
            )
            continue
        try:
            strategy, stock, seed = _run_identity(
                comparison_path, results_dir, scope["strategies"]
            )
        except ValueError as exc:
            issues.append(
                issue(
                    "invalid_result_path",
                    str(exc),
                    source_file=str(comparison_path),
                )
            )
            continue
        comparison_text = _parse_text_metadata(comparison_path.with_suffix(".txt"))
        metadata = dict(manifest)
        metadata_sources = [scope.get("config_path")]
        command_metadata = runner_commands.get((strategy, stock, seed), {})
        if command_metadata:
            metadata.update(command_metadata)
            metadata_sources.append(command_metadata.get("runner_command_source"))
        checkpoint_path = metadata.get("checkpoint_path") or metadata.get(
            "pretrain_checkpoint_path"
        )
        if checkpoint_path:
            checkpoint_key = str(checkpoint_path)
            if checkpoint_key not in checkpoint_cache:
                checkpoint_cache[checkpoint_key] = _load_checkpoint_metadata(
                    checkpoint_path
                )
            checkpoint_metadata, checkpoint_error = checkpoint_cache[checkpoint_key]
            if checkpoint_metadata:
                metadata.update(checkpoint_metadata)
                metadata_sources.append(
                    checkpoint_metadata.get("checkpoint_metadata_source")
                )
            elif checkpoint_error:
                issues.append(
                    issue(
                        "checkpoint_metadata_unavailable",
                        checkpoint_error,
                        severity="warning",
                        stock=stock,
                        seed=seed,
                        strategy=strategy,
                        source_file=str(comparison_path),
                    )
                )
        preprocessing_path = comparison_path.parent / "preprocessing_config.json"
        if preprocessing_path.exists():
            try:
                preprocessing = _load_json(preprocessing_path)
                metadata.update(preprocessing)
                metadata_sources.append(str(preprocessing_path))
            except ValueError as exc:
                issues.append(
                    issue(
                        "invalid_preprocessing_metadata",
                        str(exc),
                        stock=stock,
                        seed=seed,
                        strategy=strategy,
                        source_file=str(preprocessing_path),
                    )
                )
        metadata["direction_definition_explicit"] = bool(
            metadata.get("direction_accuracy_definition")
        )
        metadata = _normalise_metadata(
            metadata, strategy=strategy, comparison_text=comparison_text
        )
        metadata["metadata_sources"] = ";".join(
            str(item) for item in metadata_sources if item
        )
        metadata["config_signature"] = _configuration_signature(metadata)
        bundle = Bundle(
            strategy=strategy,
            stock=stock,
            seed=seed,
            timestamp=match.group(1),
            comparison_path=comparison_path,
            metadata=metadata,
        )
        try:
            comparison = pd.read_csv(comparison_path)
        except Exception as exc:
            bundle.issues.append(
                issue(
                    "invalid_comparison_file",
                    str(exc),
                    stock=stock,
                    seed=seed,
                    strategy=strategy,
                    source_file=str(comparison_path),
                )
            )
            bundles.append(bundle)
            continue
        required_columns = {"model", "mse", "mae", "trend_accuracy"}
        if not required_columns.issubset(comparison.columns):
            bundle.issues.append(
                issue(
                    "invalid_comparison_schema",
                    f"Missing columns: {sorted(required_columns - set(comparison.columns))}",
                    stock=stock,
                    seed=seed,
                    strategy=strategy,
                    source_file=str(comparison_path),
                )
            )
            bundles.append(bundle)
            continue
        if comparison["model"].duplicated().any():
            bundle.issues.append(
                issue(
                    "duplicate_model_rows",
                    "A comparison file contains duplicate model names",
                    stock=stock,
                    seed=seed,
                    strategy=strategy,
                    source_file=str(comparison_path),
                )
            )
        for comparison_row in comparison.to_dict(orient="records"):
            raw_model = str(comparison_row["model"])
            method = (
                STRATEGY_METHODS.get(strategy)
                if raw_model == "TS-JEPA"
                else RAW_MODEL_METHODS.get(raw_model)
            )
            if method is None:
                bundle.issues.append(
                    issue(
                        "unsupported_model_row",
                        f"Model {raw_model!r} is not part of the thesis method set",
                        severity="warning",
                        stock=stock,
                        seed=seed,
                        strategy=strategy,
                        source_file=str(comparison_path),
                    )
                )
                continue
            summary_mse = finite_or_nan(comparison_row.get("mse"))
            summary_mae = finite_or_nan(comparison_row.get("mae"))
            summary_direction = finite_or_nan(comparison_row.get("trend_accuracy"))
            score_path = _score_path_for_model(bundle, raw_model)
            score_frame = None
            test_signature = None
            if score_path is not None:
                try:
                    score_frame = _load_score_frame(score_path, raw_model)
                    pred, true = _arrays_from_scores(score_frame)
                    recomputed_mse = float(np.mean((pred - true) ** 2))
                    recomputed_mae = float(np.mean(np.abs(pred - true)))
                    if "forecast_target" in score_frame and not score_frame.empty:
                        forecast_target_value = score_frame["forecast_target"].iloc[0]
                    else:
                        forecast_target_value = metadata.get("forecast_target")
                    if forecast_target_value in (None, "") or pd.isna(
                        forecast_target_value
                    ):
                        forecast_target = None
                        recomputed_direction = float("nan")
                        bundle.issues.append(
                            issue(
                                "forecast_target_unrecoverable",
                                "Saved score rows and metadata do not identify the "
                                "forecast target, so direction accuracy cannot be "
                                "reconstructed without assuming semantics",
                                stock=stock,
                                seed=seed,
                                method=method,
                                strategy=strategy,
                                source_file=str(score_path),
                            )
                        )
                    else:
                        forecast_target = str(forecast_target_value)
                        recomputed_direction = compute_direction_accuracy(
                            pred, true, forecast_target
                        )
                    for metric_name, stored, recomputed in (
                        ("mse", summary_mse, recomputed_mse),
                        ("mae", summary_mae, recomputed_mae),
                        ("direction_accuracy", summary_direction, recomputed_direction),
                    ):
                        if (
                            math.isfinite(stored)
                            and math.isfinite(recomputed)
                            and not math.isclose(
                            stored, recomputed, rel_tol=1e-6, abs_tol=1e-9
                            )
                        ):
                            bundle.issues.append(
                                issue(
                                    "summary_reconstruction_mismatch",
                                    f"{metric_name}: stored={stored}, recomputed={recomputed}",
                                    stock=stock,
                                    seed=seed,
                                    method=method,
                                    strategy=strategy,
                                    source_file=str(score_path),
                                )
                            )
                    summary_mse = recomputed_mse
                    summary_mae = recomputed_mae
                    summary_direction = recomputed_direction
                    metadata["forecast_horizon"] = int(score_frame["horizon"].max())
                    if forecast_target is not None:
                        metadata["forecast_target"] = forecast_target
                    test_signature = _test_signature(score_frame)
                    bundle.predictions[raw_model] = score_frame
                except Exception as exc:
                    bundle.issues.append(
                        issue(
                            "invalid_score_file",
                            str(exc),
                            stock=stock,
                            seed=seed,
                            method=method,
                            strategy=strategy,
                            source_file=str(score_path),
                        )
                    )
            elif not metadata.get("direction_definition_explicit"):
                summary_direction = float("nan")
                bundle.issues.append(
                    issue(
                        "direction_accuracy_definition_unverified",
                        "A summary direction value exists, but no saved predictions "
                        "or explicit metric-definition metadata can verify it",
                        stock=stock,
                        seed=seed,
                        method=method,
                        strategy=strategy,
                        source_file=str(comparison_path),
                    )
                )
            elif not math.isfinite(summary_direction):
                bundle.issues.append(
                    issue(
                        "direction_accuracy_unavailable",
                        "No valid saved predictions or finite summary value are available",
                        stock=stock,
                        seed=seed,
                        method=method,
                        strategy=strategy,
                        source_file=str(comparison_path),
                    )
                )
            row = {
                "stock": stock,
                "seed": seed,
                "method": method,
                "strategy": strategy,
                "split": metadata.get("split", "test"),
                "mse": summary_mse,
                "mae": summary_mae,
                "direction_accuracy": summary_direction,
                "forecast_horizon": metadata.get("forecast_horizon"),
                "checkpoint_path": metadata.get("checkpoint_path"),
                "checkpoint_selection": metadata.get("checkpoint_selection"),
                "encoder_weight_source": metadata.get("encoder_weight_source"),
                "fine_tune_encoder": metadata.get("fine_tune_encoder"),
                "trend_weight": metadata.get("trend_weight"),
                "trend_loss_temperature": metadata.get("trend_loss_temperature"),
                "trend_loss_threshold": metadata.get("trend_loss_threshold"),
                "trend_selection_weight": metadata.get("trend_selection_weight"),
                "normalization": metadata.get("normalization"),
                "feature_transform": metadata.get("feature_transform"),
                "forecast_target": metadata.get("forecast_target"),
                "target_definition": metadata.get("target_definition"),
                "metric_definition": metadata.get("metric_definition"),
                "test_start": metadata.get("test_start"),
                "test_end": metadata.get("test_end"),
                "test_signature": test_signature,
                "experiment_id": metadata.get("experiment_id")
                or _configuration_signature(metadata),
                "config_signature": _configuration_signature(metadata),
                "timestamp": bundle.timestamp,
                "git_commit_sha": metadata.get("git_commit_sha"),
                "original_source_file": str(comparison_path),
                "score_source_file": str(score_path) if score_path else "",
                "raw_model": raw_model,
                "metadata_sources": metadata.get("metadata_sources", ""),
            }
            if not all(math.isfinite(float(row[name])) for name in ("mse", "mae")):
                bundle.issues.append(
                    issue(
                        "non_finite_metric",
                        "MSE or MAE is missing/non-finite",
                        stock=stock,
                        seed=seed,
                        method=method,
                        strategy=strategy,
                        source_file=str(comparison_path),
                    )
                )
            bundle.rows.append(row)
        bundle.metadata["config_signature"] = _configuration_signature(
            bundle.metadata
        )
        bundles.append(bundle)
        issues.extend(bundle.issues)
    return bundles, issues


def select_bundles(
    bundles: Sequence[Bundle], issues: list[dict[str, Any]]
) -> dict[tuple[str, str, int], Bundle]:
    grouped: dict[tuple[str, str, int], list[Bundle]] = defaultdict(list)
    for bundle in bundles:
        grouped[bundle.key].append(bundle)
    selected: dict[tuple[str, str, int], Bundle] = {}
    for key, candidates in grouped.items():
        candidates = sorted(candidates, key=lambda item: (item.timestamp, str(item.comparison_path)))
        strategy, stock, seed = key
        usable = [candidate for candidate in candidates if candidate.rows]
        if len(candidates) > 1:
            signatures = sorted(
                {str(candidate.metadata.get("config_signature")) for candidate in usable}
            )
            severity = "error" if len(signatures) > 1 else "warning"
            status = (
                "conflicting_duplicate_configurations"
                if len(signatures) > 1
                else "duplicate_experiments"
            )
            issues.append(
                issue(
                    status,
                    f"Found {len(candidates)} timestamped comparison bundles; "
                    f"selected latest={candidates[-1].comparison_path.name}; "
                    f"config_signatures={signatures}",
                    severity=severity,
                    stock=stock,
                    seed=seed,
                    strategy=strategy,
                    source_file=";".join(str(item.comparison_path) for item in candidates),
                )
            )
        if usable:
            usable[-1].selected = True
            selected[key] = usable[-1]
    return selected


def _method_strategy(method: str, reference_strategy: str) -> str:
    if method == "Shared-target JEPA--MAE":
        return "random"
    if method == "Local-MAE/Long-JEPA":
        return "local_long"
    return reference_strategy


def _row_for_method(bundle: Bundle | None, method: str) -> dict[str, Any] | None:
    if bundle is None:
        return None
    matches = [row for row in bundle.rows if row["method"] == method]
    if len(matches) != 1:
        return None
    return matches[0]


def _pair_compatible(model: Mapping[str, Any], reference: Mapping[str, Any]) -> tuple[bool, str]:
    keys = (
        "stock",
        "seed",
        "split",
        "forecast_horizon",
        "forecast_target",
        "target_definition",
        "normalization",
        "metric_definition",
        "test_start",
        "test_end",
    )
    mismatches = []
    for key in keys:
        left, right = model.get(key), reference.get(key)
        if left in (None, "") or right in (None, ""):
            continue
        if str(left) != str(right):
            mismatches.append(f"{key}={left!r}/{right!r}")
    left_signature, right_signature = model.get("test_signature"), reference.get("test_signature")
    if left_signature and right_signature and left_signature != right_signature:
        mismatches.append("saved target rows differ")
    return not mismatches, "; ".join(mismatches)


def _shared_local_pair_compatible(
    shared: Mapping[str, Any], local: Mapping[str, Any]
) -> tuple[bool, str]:
    compatible, details = _pair_compatible(shared, local)
    mismatches = [details] if not compatible else []
    for key in ("experiment_id", "config_signature"):
        left, right = shared.get(key), local.get(key)
        left_present = left not in (None, "") and not pd.isna(left)
        right_present = right not in (None, "") and not pd.isna(right)
        if left_present and right_present and str(left) != str(right):
            mismatches.append(f"{key}={left!r}/{right!r}")
    return not mismatches, "; ".join(mismatches)


def build_canonical_data(
    bundles: Sequence[Bundle],
    selected: Mapping[tuple[str, str, int], Bundle],
    scope: Mapping[str, Any],
    issues: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    canonical_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    included_ids: set[tuple[str, str, int, str, str]] = set()

    # Select one coherent configuration per method. A minority configuration is
    # excluded instead of being silently pooled with the dominant procedure.
    candidate_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for method in METHOD_ORDER:
        strategy = _method_strategy(method, scope["reference_strategy"])
        for stock in scope["stocks"]:
            for seed in scope["seeds"]:
                row = _row_for_method(selected.get((strategy, stock, seed)), method)
                if row is not None:
                    candidate_by_method[method].append(row)
    accepted_signatures: dict[str, str] = {}
    accepted_test_signatures: dict[tuple[str, str], str] = {}
    for method, candidates in candidate_by_method.items():
        counts = Counter(str(row.get("config_signature")) for row in candidates)
        accepted = sorted(counts, key=lambda value: (-counts[value], value))[0]
        accepted_signatures[method] = accepted
        if len(counts) > 1:
            issues.append(
                issue(
                    "conflicting_configurations",
                    f"Method has incompatible run configurations {dict(counts)}; "
                    f"only dominant signature {accepted} is eligible",
                    method=method,
                )
            )
        candidate_frame = pd.DataFrame(candidates)
        for stock, stock_rows in candidate_frame.groupby("stock"):
            test_values = [
                str(value)
                for value in stock_rows["test_signature"]
                if value not in (None, "") and not pd.isna(value)
            ]
            test_counts = Counter(test_values)
            if not test_counts:
                continue
            accepted_test = sorted(
                test_counts,
                key=lambda value: (-test_counts[value], value),
            )[0]
            accepted_test_signatures[(method, stock)] = accepted_test
            if len(test_counts) > 1:
                issues.append(
                    issue(
                        "inconsistent_test_periods_across_seeds",
                        f"Saved target signatures vary across seeds: {dict(test_counts)}; "
                        f"only dominant signature {accepted_test} is eligible",
                        stock=stock,
                        method=method,
                    )
                )

    for method in METHOD_ORDER:
        strategy = _method_strategy(method, scope["reference_strategy"])
        for stock in scope["stocks"]:
            for seed in scope["seeds"]:
                bundle = selected.get((strategy, stock, seed))
                row = _row_for_method(bundle, method)
                if row is None:
                    issues.append(
                        issue(
                            "missing_run",
                            "No selected, valid result row is available",
                            stock=stock,
                            seed=seed,
                            method=method,
                            strategy=strategy,
                        )
                    )
                    continue
                if str(row.get("config_signature")) != accepted_signatures.get(method):
                    issues.append(
                        issue(
                            "excluded_conflicting_run",
                            "Run does not match the dominant method configuration",
                            stock=stock,
                            seed=seed,
                            method=method,
                            strategy=strategy,
                            source_file=row["original_source_file"],
                        )
                    )
                    continue
                accepted_test = accepted_test_signatures.get((method, stock))
                if (
                    accepted_test is not None
                    and row.get("test_signature") not in (None, "")
                    and str(row.get("test_signature")) != accepted_test
                ):
                    issues.append(
                        issue(
                            "excluded_inconsistent_test_period",
                            "Run target signature differs from the dominant "
                            "stock/method test signature",
                            stock=stock,
                            seed=seed,
                            method=method,
                            strategy=strategy,
                            source_file=row["original_source_file"],
                        )
                    )
                    continue
                if not all(math.isfinite(float(row[name])) for name in ("mse", "mae")):
                    continue
                canonical_rows.append({key: row.get(key) for key in TIDY_COLUMNS})
                included_ids.add(
                    (strategy, stock, seed, row["timestamp"], row["raw_model"])
                )

    # Cross-method summaries require the same held-out targets. Use the
    # canonical naive-last row as the compatibility anchor for each stock/seed.
    invalid_canonical_keys: set[tuple[str, int, str]] = set()
    preliminary = pd.DataFrame(canonical_rows, columns=TIDY_COLUMNS)
    if not preliminary.empty:
        for (stock, seed), group in preliminary.groupby(["stock", "seed"]):
            references = group[group["method"] == REFERENCE_METHOD]
            if references.empty:
                continue
            reference = references.iloc[0].to_dict()
            for candidate in group.to_dict(orient="records"):
                compatible, details = _pair_compatible(candidate, reference)
                if compatible:
                    continue
                invalid_canonical_keys.add((stock, int(seed), candidate["method"]))
                issues.append(
                    issue(
                        "inconsistent_test_period_or_target",
                        details,
                        stock=stock,
                        seed=int(seed),
                        method=candidate["method"],
                        strategy=candidate["strategy"],
                        source_file=candidate["original_source_file"],
                    )
                )
    if invalid_canonical_keys:
        canonical_rows = [
            row
            for row in canonical_rows
            if (row["stock"], int(row["seed"]), row["method"])
            not in invalid_canonical_keys
        ]
        for bundle in bundles:
            for row in bundle.rows:
                if (
                    row["stock"],
                    int(row["seed"]),
                    row["method"],
                ) in invalid_canonical_keys:
                    included_ids.discard(
                        (
                            bundle.strategy,
                            bundle.stock,
                            bundle.seed,
                            bundle.timestamp,
                            row["raw_model"],
                        )
                    )

    # Check deterministic baseline invariance across seeds and strategies.
    deterministic = {"Naive-last", "Drift", "Mean-context"}
    selected_rows = [row for bundle in selected.values() for row in bundle.rows]
    baseline_frame = pd.DataFrame(selected_rows)
    if not baseline_frame.empty:
        baseline_frame = baseline_frame[baseline_frame["method"].isin(deterministic)]
        for (stock, method), group in baseline_frame.groupby(["stock", "method"]):
            for metric in ("mse", "mae", "direction_accuracy"):
                values = group[metric].dropna().to_numpy(float)
                if len(values) > 1 and not np.allclose(values, values[0], rtol=1e-8, atol=1e-10):
                    issues.append(
                        issue(
                            "inconsistent_deterministic_baseline",
                            f"{metric} varies across strategy/seed reruns: "
                            f"min={values.min()}, max={values.max()}",
                            stock=stock,
                            method=method,
                        )
                    )

    for bundle in bundles:
        for row in bundle.rows:
            identifier = (
                bundle.strategy,
                bundle.stock,
                bundle.seed,
                bundle.timestamp,
                row["raw_model"],
            )
            included = identifier in included_ids
            if included:
                reason = ""
            elif not bundle.selected:
                reason = "older duplicate bundle or invalid bundle"
            elif row["method"] == "GRU" and bundle.strategy != scope["reference_strategy"]:
                reason = "alternate strategy-specific GRU; reference strategy selected"
            elif row["method"] in deterministic and bundle.strategy != scope["reference_strategy"]:
                reason = "duplicate deterministic baseline; reference strategy selected"
            elif row["method"] not in METHOD_ORDER:
                reason = "outside thesis method set"
            else:
                reason = "not selected because of scope, missing coverage, or configuration conflict"
            inventory_rows.append(
                {
                    **{key: row.get(key) for key in TIDY_COLUMNS},
                    "raw_model": row.get("raw_model"),
                    "selected_bundle": bundle.selected,
                    "included_in_canonical": included,
                    "exclusion_reason": reason,
                    "metadata_sources": row.get("metadata_sources", ""),
                }
            )

    tidy = pd.DataFrame(canonical_rows, columns=TIDY_COLUMNS)
    if not tidy.empty:
        method_index = {method: index for index, method in enumerate(METHOD_ORDER)}
        tidy["_method_order"] = tidy["method"].map(method_index)
        tidy = (
            tidy.sort_values(["stock", "seed", "_method_order"])
            .drop(columns="_method_order")
            .reset_index(drop=True)
        )
        duplicates = tidy.duplicated(["stock", "seed", "method"], keep=False)
        for row in tidy[duplicates].itertuples():
            issues.append(
                issue(
                    "duplicate_canonical_run",
                    "Canonical stock/seed/method key is duplicated",
                    stock=row.stock,
                    seed=row.seed,
                    method=row.method,
                    strategy=row.strategy,
                    source_file=row.original_source_file,
                )
            )

    inventory = pd.DataFrame(inventory_rows, columns=INVENTORY_COLUMNS)
    prediction_rows: list[dict[str, Any]] = []
    for bundle in selected.values():
        for row in bundle.rows:
            identifier = (
                bundle.strategy,
                bundle.stock,
                bundle.seed,
                bundle.timestamp,
                row["raw_model"],
            )
            if identifier not in included_ids:
                continue
            frame = bundle.predictions.get(row["raw_model"])
            if frame is None:
                continue
            for prediction in frame.to_dict(orient="records"):
                prediction_rows.append(
                    {
                        "stock": bundle.stock,
                        "seed": bundle.seed,
                        "method": row["method"],
                        "strategy": bundle.strategy,
                        "rolling_step": int(prediction["rolling_step"]),
                        "horizon": int(prediction["horizon"]),
                        "target_index": prediction.get("target_index"),
                        "target_date": prediction.get("target_date", ""),
                        "predicted_value": float(prediction["predicted_value"]),
                        "true_value": float(prediction["true_value"]),
                        "source_file": row["score_source_file"],
                    }
                )
    predictions = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS)

    paired_rows: list[dict[str, Any]] = []
    pair_methods = [method for method in METHOD_ORDER if method != REFERENCE_METHOD]
    for method in pair_methods:
        strategy = _method_strategy(method, scope["reference_strategy"])
        for stock in scope["stocks"]:
            for seed in scope["seeds"]:
                bundle = selected.get((strategy, stock, seed))
                model = _row_for_method(bundle, method)
                reference = _row_for_method(bundle, REFERENCE_METHOD)
                if model is None or reference is None:
                    continue
                if (stock, int(seed), method) in invalid_canonical_keys:
                    continue
                if str(model.get("config_signature")) != accepted_signatures.get(method):
                    continue
                accepted_test = accepted_test_signatures.get((method, stock))
                if (
                    accepted_test is not None
                    and model.get("test_signature") not in (None, "")
                    and str(model.get("test_signature")) != accepted_test
                ):
                    continue
                compatible, details = _pair_compatible(model, reference)
                if not compatible:
                    issues.append(
                        issue(
                            "incompatible_pair",
                            details,
                            stock=stock,
                            seed=seed,
                            method=method,
                            strategy=strategy,
                            source_file=model["original_source_file"],
                        )
                    )
                    continue
                paired_row = {
                    "stock": stock,
                    "seed": seed,
                    "model": method,
                    "strategy": strategy,
                    "test_signature": model.get("test_signature") or reference.get("test_signature"),
                    "model_source_file": model["original_source_file"],
                    "naive_source_file": reference["original_source_file"],
                }
                for metric in ("mse", "mae"):
                    model_value = float(model[metric])
                    naive_value = float(reference[metric])
                    paired_row[f"model_{metric}"] = model_value
                    paired_row[f"naive_{metric}"] = naive_value
                    paired_row[f"delta_{metric}"] = model_value - naive_value
                    paired_row[f"relative_{metric}_improvement_pct"] = (
                        100.0 * (naive_value - model_value) / naive_value
                        if naive_value != 0.0
                        else float("nan")
                    )
                paired_rows.append(paired_row)
    paired_runs = pd.DataFrame(paired_rows, columns=PAIRED_RUN_COLUMNS)
    return tidy, inventory, predictions, paired_runs


def build_stock_summary(tidy: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "stock",
        "method",
        "mse_mean",
        "mse_std",
        "mae_mean",
        "mae_std",
        "direction_accuracy_mean",
        "direction_accuracy_std",
        "n_valid_seeds",
        "n_direction_seeds",
    )
    if tidy.empty:
        return pd.DataFrame(columns=columns)
    summary = (
        tidy.groupby(["stock", "method"], as_index=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            direction_accuracy_mean=("direction_accuracy", "mean"),
            direction_accuracy_std=("direction_accuracy", "std"),
            n_valid_seeds=("seed", "nunique"),
            n_direction_seeds=("direction_accuracy", "count"),
        )
        .reset_index(drop=True)
    )
    for column in ("mse_std", "mae_std", "direction_accuracy_std"):
        summary[column] = summary[column].fillna(0.0)
    method_index = {method: index for index, method in enumerate(METHOD_ORDER)}
    summary["_method_order"] = summary["method"].map(method_index)
    return (
        summary.sort_values(["stock", "_method_order"])
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )


def build_overall_summary(stock_summary: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "method",
        "mse",
        "mse_std_across_stocks",
        "mae",
        "mae_std_across_stocks",
        "direction_accuracy",
        "direction_accuracy_std_across_stocks",
        "mse_average_rank",
        "mae_average_rank",
        "direction_accuracy_average_rank",
        "mse_stock_wins",
        "mse_stock_second_places",
        "mae_stock_wins",
        "mae_stock_second_places",
        "direction_accuracy_stock_wins",
        "direction_accuracy_stock_second_places",
        "n_stocks",
        "n_direction_stocks",
    )
    if stock_summary.empty:
        return pd.DataFrame(columns=columns)
    rank_rows = []
    for stock, group in stock_summary.groupby("stock"):
        for metric in METRICS:
            value_column = f"{metric}_mean"
            finite = group[np.isfinite(group[value_column])].copy()
            if finite.empty:
                continue
            ascending = LOWER_IS_BETTER[metric]
            finite["rank"] = finite[value_column].rank(method="average", ascending=ascending)
            finite["competition_rank"] = finite[value_column].rank(
                method="min", ascending=ascending
            )
            for row in finite.itertuples():
                rank_rows.append(
                    {
                        "stock": stock,
                        "method": row.method,
                        "metric": metric,
                        "rank": float(row.rank),
                        "win": int(row.competition_rank == 1),
                        "second": int(row.competition_rank == 2),
                    }
                )
    ranks = pd.DataFrame(rank_rows)
    rows = []
    for method in METHOD_ORDER:
        group = stock_summary[stock_summary["method"] == method]
        if group.empty:
            continue
        row = {
            "method": method,
            "mse": float(group["mse_mean"].mean()),
            "mse_std_across_stocks": float(group["mse_mean"].std(ddof=1)) if len(group) > 1 else 0.0,
            "mae": float(group["mae_mean"].mean()),
            "mae_std_across_stocks": float(group["mae_mean"].std(ddof=1)) if len(group) > 1 else 0.0,
            "direction_accuracy": float(group["direction_accuracy_mean"].mean()),
            "direction_accuracy_std_across_stocks": (
                float(group["direction_accuracy_mean"].std(ddof=1))
                if group["direction_accuracy_mean"].count() > 1
                else 0.0
            ),
            "n_stocks": int(group["stock"].nunique()),
            "n_direction_stocks": int(group["direction_accuracy_mean"].count()),
        }
        for metric in METRICS:
            if ranks.empty:
                metric_ranks = pd.DataFrame()
            else:
                metric_ranks = ranks[
                    (ranks["method"] == method) & (ranks["metric"] == metric)
                ]
            row[f"{metric}_average_rank"] = (
                float(metric_ranks["rank"].mean()) if not metric_ranks.empty else float("nan")
            )
            row[f"{metric}_stock_wins"] = (
                int(metric_ranks["win"].sum()) if not metric_ranks.empty else 0
            )
            row[f"{metric}_stock_second_places"] = (
                int(metric_ranks["second"].sum()) if not metric_ranks.empty else 0
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def build_paired_summary(
    paired_runs: pd.DataFrame,
    *,
    analysis_seed: int,
    bootstrap_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = (
        "model",
        "n_stocks",
        "mean_delta_mse",
        "median_delta_mse",
        "relative_mse_improvement_pct",
        "mse_ci_low",
        "mse_ci_high",
        "mse_wilcoxon_statistic",
        "mse_p_value",
        "mse_holm_p_value",
        "mse_rank_biserial",
        "mse_wilcoxon_n",
        "mse_stock_wins",
        "mse_stock_losses",
        "mse_stock_ties",
        "mse_run_wins",
        "mse_run_total",
        "mean_delta_mae",
        "median_delta_mae",
        "relative_mae_improvement_pct",
        "mae_ci_low",
        "mae_ci_high",
        "mae_wilcoxon_statistic",
        "mae_p_value",
        "mae_holm_p_value",
        "mae_rank_biserial",
        "mae_wilcoxon_n",
        "mae_stock_wins",
        "mae_stock_losses",
        "mae_stock_ties",
        "mae_run_wins",
        "mae_run_total",
    )
    stock_columns = (
        "stock",
        "model",
        "n_paired_seeds",
        "delta_mse",
        "relative_mse_improvement_pct",
        "delta_mae",
        "relative_mae_improvement_pct",
    )
    if paired_runs.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=stock_columns)
    learned = paired_runs[paired_runs["model"].isin(LEARNED_METHODS)].copy()
    stock_pairs = (
        learned.groupby(["stock", "model"], as_index=False)
        .agg(
            n_paired_seeds=("seed", "nunique"),
            delta_mse=("delta_mse", "mean"),
            relative_mse_improvement_pct=("relative_mse_improvement_pct", "mean"),
            delta_mae=("delta_mae", "mean"),
            relative_mae_improvement_pct=("relative_mae_improvement_pct", "mean"),
        )
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    for model in LEARNED_METHODS:
        stock_group = stock_pairs[stock_pairs["model"] == model]
        run_group = learned[learned["model"] == model]
        if stock_group.empty:
            continue
        row: dict[str, Any] = {"model": model, "n_stocks": stock_group["stock"].nunique()}
        for metric in ("mse", "mae"):
            deltas = stock_group[f"delta_{metric}"].to_numpy(float)
            relative = stock_group[f"relative_{metric}_improvement_pct"].to_numpy(float)
            seed_offset = int(stable_hash([model, metric])[:8], 16)
            rng = np.random.default_rng(analysis_seed + seed_offset)
            ci_low, ci_high = bootstrap_mean_ci(
                deltas, rng=rng, samples=bootstrap_samples
            )
            statistic, p_value, effect, wilcoxon_n = exact_wilcoxon(deltas)
            run_deltas = run_group[f"delta_{metric}"].to_numpy(float)
            row.update(
                {
                    f"mean_delta_{metric}": float(np.mean(deltas)),
                    f"median_delta_{metric}": float(np.median(deltas)),
                    f"relative_{metric}_improvement_pct": float(np.nanmean(relative)),
                    f"{metric}_ci_low": ci_low,
                    f"{metric}_ci_high": ci_high,
                    f"{metric}_wilcoxon_statistic": statistic,
                    f"{metric}_p_value": p_value,
                    f"{metric}_holm_p_value": float("nan"),
                    f"{metric}_rank_biserial": effect,
                    f"{metric}_wilcoxon_n": wilcoxon_n,
                    f"{metric}_stock_wins": int(np.sum(deltas < 0)),
                    f"{metric}_stock_losses": int(np.sum(deltas > 0)),
                    f"{metric}_stock_ties": int(np.sum(deltas == 0)),
                    f"{metric}_run_wins": int(np.sum(run_deltas < 0)),
                    f"{metric}_run_total": int(np.isfinite(run_deltas).sum()),
                }
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    for metric in ("mse", "mae"):
        adjusted = holm_adjust(
            dict(zip(summary["model"], summary[f"{metric}_p_value"]))
            if not summary.empty
            else {}
        )
        if not summary.empty:
            summary[f"{metric}_holm_p_value"] = summary["model"].map(adjusted)
    return summary.reindex(columns=columns), stock_pairs.reindex(columns=stock_columns)


def build_shared_vs_local_comparison(
    tidy: pd.DataFrame,
    scope: Mapping[str, Any],
    issues: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair Shared and Local runs by configured stock/seed, then infer by stock."""
    shared_method, local_method = METHOD_ORDER[:2]
    stock_columns = (
        "stock",
        "n_matched_seeds",
        "matched_seeds",
        "shared_mse",
        "local_mse",
        "delta_mse",
        "shared_mae",
        "local_mae",
        "delta_mae",
    )
    summary_columns = (
        "metric",
        "n_stocks",
        "mean_delta",
        "median_delta",
        "t_statistic",
        "p_value",
        "cohens_dz",
        "status",
    )
    required = {"stock", "seed", "method", "mse", "mae"}
    missing = sorted(required - set(tidy.columns))
    if missing:
        raise ValueError(f"Canonical data are missing Shared-vs-Local columns: {missing}")
    expected_stocks = [str(stock) for stock in scope["stocks"]]
    expected_seeds = [int(seed) for seed in scope["seeds"]]
    relevant = tidy[
        tidy["stock"].astype(str).isin(expected_stocks)
        & pd.to_numeric(tidy["seed"], errors="coerce").isin(expected_seeds)
        & tidy["method"].isin((shared_method, local_method))
    ].copy()
    duplicates = relevant.duplicated(["stock", "seed", "method"], keep=False)
    if duplicates.any():
        keys = relevant.loc[duplicates, ["stock", "seed", "method"]]
        raise ValueError(
            "Canonical Shared-vs-Local keys are duplicated: "
            + repr(keys.head().to_dict(orient="records"))
        )

    stock_rows: list[dict[str, Any]] = []
    for stock in expected_stocks:
        matched: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        for seed in expected_seeds:
            seed_rows = relevant[
                (relevant["stock"].astype(str) == stock)
                & (pd.to_numeric(relevant["seed"], errors="coerce") == seed)
            ]
            shared_rows = seed_rows[seed_rows["method"] == shared_method]
            local_rows = seed_rows[seed_rows["method"] == local_method]
            if shared_rows.empty or local_rows.empty:
                if not shared_rows.empty or not local_rows.empty:
                    available = shared_rows if not shared_rows.empty else local_rows
                    missing_method = local_method if local_rows.empty else shared_method
                    issues.append(
                        issue(
                            "unmatched_shared_local_seed",
                            f"No {missing_method} row is available for this configured seed",
                            severity="warning",
                            stock=stock,
                            seed=seed,
                            method=missing_method,
                            source_file=str(
                                available.iloc[0].get("original_source_file", "")
                            ),
                        )
                    )
                continue
            shared = shared_rows.iloc[0].to_dict()
            local = local_rows.iloc[0].to_dict()
            compatible, details = _shared_local_pair_compatible(shared, local)
            if not compatible:
                issues.append(
                    issue(
                        "incompatible_shared_local_pair",
                        details,
                        severity="warning",
                        stock=stock,
                        seed=seed,
                        method=f"{shared_method} vs {local_method}",
                        source_file=";".join(
                            str(row.get("original_source_file", ""))
                            for row in (shared, local)
                        ),
                    )
                )
                continue
            matched.append((seed, shared, local))
        if not matched:
            issues.append(
                issue(
                    "shared_local_stock_excluded",
                    "No compatible matched Shared-vs-Local seeds are available",
                    severity="warning",
                    stock=stock,
                    method=f"{shared_method} vs {local_method}",
                )
            )
            continue
        stock_row: dict[str, Any] = {
            "stock": stock,
            "n_matched_seeds": len(matched),
            "matched_seeds": ";".join(str(seed) for seed, _, _ in matched),
        }
        for metric in ("mse", "mae"):
            shared_value = float(np.mean([float(row[metric]) for _, row, _ in matched]))
            local_value = float(np.mean([float(row[metric]) for _, _, row in matched]))
            stock_row[f"shared_{metric}"] = shared_value
            stock_row[f"local_{metric}"] = local_value
            stock_row[f"delta_{metric}"] = shared_value - local_value
        stock_rows.append(stock_row)

    stock_pairs = pd.DataFrame(stock_rows, columns=stock_columns)
    summary_rows = []
    for metric in ("mse", "mae"):
        values = (
            stock_pairs[f"delta_{metric}"].to_numpy(float)
            if not stock_pairs.empty
            else np.asarray([], dtype=float)
        )
        summary_rows.append(
            {"metric": metric.upper(), **paired_difference_statistics(values)}
        )
    summary = pd.DataFrame(summary_rows, columns=summary_columns)
    return stock_pairs, summary


def build_relative_stock_data(paired_runs: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "stock",
        "method",
        "relative_mse_improvement_pct",
        "relative_mae_improvement_pct",
        "n_paired_seeds",
    )
    if paired_runs.empty:
        return pd.DataFrame(columns=columns)
    summary = (
        paired_runs.groupby(["stock", "model"], as_index=False)
        .agg(
            relative_mse_improvement_pct=("relative_mse_improvement_pct", "mean"),
            relative_mae_improvement_pct=("relative_mae_improvement_pct", "mean"),
            n_paired_seeds=("seed", "nunique"),
        )
        .rename(columns={"model": "method"})
    )
    baseline_rows = []
    for stock in sorted(summary["stock"].unique()):
        baseline_rows.append(
            {
                "stock": stock,
                "method": REFERENCE_METHOD,
                "relative_mse_improvement_pct": 0.0,
                "relative_mae_improvement_pct": 0.0,
                "n_paired_seeds": int(
                    summary.loc[summary["stock"] == stock, "n_paired_seeds"].max()
                ),
            }
        )
    return pd.concat([summary, pd.DataFrame(baseline_rows)], ignore_index=True).reindex(
        columns=columns
    )


def build_horizon_metrics(
    predictions: pd.DataFrame,
    tidy: pd.DataFrame,
    *,
    analysis_seed: int,
    bootstrap_samples: int,
) -> pd.DataFrame:
    columns = (
        "method",
        "horizon",
        "mse",
        "mse_ci_low",
        "mse_ci_high",
        "mae",
        "mae_ci_low",
        "mae_ci_high",
        "direction_accuracy",
        "direction_ci_low",
        "direction_ci_high",
        "n_stocks",
        "n_runs",
    )
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    target_lookup = {
        (row.stock, int(row.seed), row.method): str(row.forecast_target or "value")
        for row in tidy.itertuples()
    }
    run_rows = []
    for (stock, seed, method), group in predictions.groupby(["stock", "seed", "method"]):
        forecast_target = target_lookup.get((stock, int(seed), method), "value")
        direction = direction_by_horizon(group, forecast_target)
        for horizon, horizon_group in group.groupby("horizon"):
            error = horizon_group["predicted_value"].to_numpy(float) - horizon_group[
                "true_value"
            ].to_numpy(float)
            run_rows.append(
                {
                    "stock": stock,
                    "seed": int(seed),
                    "method": method,
                    "horizon": int(horizon),
                    "mse": float(np.mean(error**2)),
                    "mae": float(np.mean(np.abs(error))),
                    "direction_accuracy": direction.get(int(horizon), float("nan")),
                }
            )
    runs = pd.DataFrame(run_rows)
    stock_level = (
        runs.groupby(["stock", "method", "horizon"], as_index=False)
        .agg(
            mse=("mse", "mean"),
            mae=("mae", "mean"),
            direction_accuracy=("direction_accuracy", "mean"),
            n_runs=("seed", "nunique"),
        )
        .reset_index(drop=True)
    )
    rows = []
    for (method, horizon), group in stock_level.groupby(["method", "horizon"]):
        row = {
            "method": method,
            "horizon": int(horizon),
            "n_stocks": int(group["stock"].nunique()),
            "n_runs": int(group["n_runs"].sum()),
        }
        for metric, prefix in (
            ("mse", "mse"),
            ("mae", "mae"),
            ("direction_accuracy", "direction"),
        ):
            values = group[metric].dropna().to_numpy(float)
            row[metric] = float(np.mean(values)) if len(values) else float("nan")
            rng = np.random.default_rng(
                analysis_seed + int(stable_hash([method, horizon, metric])[:8], 16)
            )
            low, high = bootstrap_mean_ci(
                values, rng=rng, samples=bootstrap_samples
            )
            row[f"{prefix}_ci_low"] = low
            row[f"{prefix}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["method", "horizon"]
    ).reset_index(drop=True)


def validate_summaries(
    tidy: pd.DataFrame,
    stock_summary: pd.DataFrame,
    predictions: pd.DataFrame,
    issues: list[dict[str, Any]],
) -> None:
    if tidy.empty:
        return
    reconstructed = build_stock_summary(tidy)
    comparable_columns = [
        "stock",
        "method",
        "mse_mean",
        "mae_mean",
        "direction_accuracy_mean",
        "n_valid_seeds",
    ]
    left = stock_summary[comparable_columns].sort_values(["stock", "method"]).reset_index(drop=True)
    right = reconstructed[comparable_columns].sort_values(["stock", "method"]).reset_index(drop=True)
    if not left.equals(right):
        issues.append(
            issue(
                "summary_not_reconstructable",
                "stock_summary.csv cannot be reconstructed exactly from all_runs_tidy.csv",
            )
        )
    if not predictions.empty:
        trace_keys = set(
            zip(tidy["stock"], tidy["seed"].astype(int), tidy["method"])
        )
        for key in predictions[["stock", "seed", "method"]].drop_duplicates().itertuples(index=False, name=None):
            if (key[0], int(key[1]), key[2]) not in trace_keys:
                issues.append(
                    issue(
                        "untraceable_prediction",
                        "Prediction rows have no canonical run",
                        stock=key[0],
                        seed=int(key[1]),
                        method=key[2],
                    )
                )


MODEL_COLORS = {
    "Shared-target JEPA--MAE": "#1f77b4",
    "Local-MAE/Long-JEPA": "#d95f02",
    "GRU": "#2ca02c",
    "Naive-last": "#4d4d4d",
    "Drift": "#9467bd",
    "Mean-context": "#8c564b",
}


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _save_figure(fig: Any, figures_dir: Path, stem: str) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = [figures_dir / f"{stem}.pdf", figures_dir / f"{stem}.png"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", dpi=300 if path.suffix == ".png" else None)
    return paths


def plot_paired_forest(
    paired_runs: pd.DataFrame,
    paired_summary: pd.DataFrame,
    figures_dir: Path,
    *,
    metric: str,
    analysis_seed: int,
    bootstrap_samples: int,
) -> list[Path]:
    learned = paired_runs[paired_runs["model"].isin(LEARNED_METHODS)]
    if learned.empty:
        return []
    plt = _configure_matplotlib()
    available = [method for method in LEARNED_METHODS if method in set(learned["model"])]
    fig, axes = plt.subplots(1, len(available), figsize=(4.0 * len(available), 5.4), sharex=False)
    axes = np.atleast_1d(axes)
    delta_column = f"delta_{metric}"
    for axis, method in zip(axes, available):
        method_runs = learned[learned["model"] == method]
        stocks = sorted(method_runs["stock"].unique())
        points, low_errors, high_errors = [], [], []
        for stock in stocks:
            values = method_runs.loc[method_runs["stock"] == stock, delta_column].to_numpy(float)
            point = float(values.mean())
            rng = np.random.default_rng(
                analysis_seed + int(stable_hash([method, stock, metric])[:8], 16)
            )
            low, high = bootstrap_mean_ci(values, rng=rng, samples=bootstrap_samples)
            points.append(point)
            low_errors.append(point - low)
            high_errors.append(high - point)
        overall = paired_summary[paired_summary["model"] == method]
        if not overall.empty:
            overall_point = float(overall.iloc[0][f"mean_delta_{metric}"])
            overall_low = float(overall.iloc[0][f"{metric}_ci_low"])
            overall_high = float(overall.iloc[0][f"{metric}_ci_high"])
            stocks.append("Overall")
            points.append(overall_point)
            low_errors.append(overall_point - overall_low)
            high_errors.append(overall_high - overall_point)
        y = np.arange(len(stocks))
        axis.errorbar(
            points,
            y,
            xerr=np.vstack([low_errors, high_errors]),
            fmt="o",
            color=MODEL_COLORS[method],
            ecolor="#777777",
            elinewidth=1,
            capsize=2,
            markersize=4,
        )
        if stocks and stocks[-1] == "Overall":
            axis.scatter(points[-1], y[-1], marker="D", s=34, color=MODEL_COLORS[method], zorder=3)
            axis.axhline(y[-1] - 0.5, color="#bdbdbd", linewidth=0.8)
        axis.axvline(0.0, color="black", linewidth=0.9, linestyle="--")
        axis.set_yticks(y)
        axis.set_yticklabels(stocks)
        axis.invert_yaxis()
        axis.set_title(method)
        axis.set_xlabel(f"$\\Delta${metric.upper()} (model $-$ naive-last)")
        axis.grid(axis="x", color="#e0e0e0", linewidth=0.6)
    fig.suptitle(
        f"Stock-level paired {metric.upper()} differences\nLeft of zero favours the model",
        y=1.01,
    )
    fig.tight_layout()
    return _save_figure(fig, figures_dir, f"fig_paired_{metric}_forest")


def _heatmap(
    frame: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    colorbar_label: str,
    figures_dir: Path,
    stem: str,
    centered: bool,
) -> list[Path]:
    if frame.empty or frame[value_column].dropna().empty:
        return []
    plt = _configure_matplotlib()
    from matplotlib.colors import TwoSlopeNorm

    available_methods = [method for method in METHOD_ORDER if method in set(frame["method"])]
    stocks = sorted(frame["stock"].unique())
    pivot = frame.pivot(index="stock", columns="method", values=value_column).reindex(
        index=stocks, columns=available_methods
    )
    values = pivot.to_numpy(float)
    masked = np.ma.masked_invalid(values)
    fig_width = max(6.5, 1.25 * len(available_methods))
    fig_height = max(3.6, 0.38 * len(stocks) + 1.4)
    fig, axis = plt.subplots(figsize=(fig_width, fig_height))
    if centered:
        finite = np.abs(values[np.isfinite(values)])
        bound = max(float(finite.max()) if finite.size else 1.0, 1e-12)
        image = axis.imshow(
            masked,
            aspect="auto",
            cmap="RdBu",
            norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
        )
    else:
        image = axis.imshow(masked, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(available_methods)))
    axis.set_xticklabels(available_methods, rotation=30, ha="right")
    axis.set_yticks(np.arange(len(stocks)))
    axis.set_yticklabels(stocks)
    axis.set_title(title)
    for row_index in range(len(stocks)):
        for column_index in range(len(available_methods)):
            value = values[row_index, column_index]
            if not math.isfinite(value):
                continue
            label = f"{value:.1f}" if centered else f"{value:.3f}"
            if centered:
                contrast = abs(value) / bound
            else:
                contrast = abs(value - 0.5) / 0.5
            text_color = "white" if contrast > 0.62 else "black"
            axis.text(column_index, row_index, label, ha="center", va="center", color=text_color, fontsize=7)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    return _save_figure(fig, figures_dir, stem)


def plot_relative_heatmaps(relative: pd.DataFrame, figures_dir: Path) -> list[Path]:
    paths = []
    paths.extend(
        _heatmap(
            relative,
            value_column="relative_mse_improvement_pct",
            title="Relative MSE performance versus naive-last",
            colorbar_label="Improvement (%) — positive favours method",
            figures_dir=figures_dir,
            stem="fig_relative_mse_heatmap",
            centered=True,
        )
    )
    paths.extend(
        _heatmap(
            relative,
            value_column="relative_mae_improvement_pct",
            title="Relative MAE performance versus naive-last",
            colorbar_label="Improvement (%) — positive favours method",
            figures_dir=figures_dir,
            stem="fig_relative_mae_heatmap",
            centered=True,
        )
    )
    return paths


def plot_direction_heatmap(stock_summary: pd.DataFrame, figures_dir: Path) -> list[Path]:
    frame = stock_summary.rename(columns={"direction_accuracy_mean": "value"})
    return _heatmap(
        frame,
        value_column="value",
        title="Mean direction accuracy across seeds",
        colorbar_label="Direction accuracy",
        figures_dir=figures_dir,
        stem="fig_direction_accuracy_heatmap",
        centered=False,
    )


def plot_horizon_metrics(horizon_metrics: pd.DataFrame, figures_dir: Path) -> list[Path]:
    if horizon_metrics.empty:
        return []
    plt = _configure_matplotlib()
    paths: list[Path] = []
    specifications = (
        ("mse", "mse_ci_low", "mse_ci_high", "MSE", "fig_mse_by_horizon"),
        ("mae", "mae_ci_low", "mae_ci_high", "MAE", "fig_mae_by_horizon"),
        (
            "direction_accuracy",
            "direction_ci_low",
            "direction_ci_high",
            "Direction accuracy",
            "fig_direction_by_horizon",
        ),
    )
    for metric, low_column, high_column, ylabel, stem in specifications:
        if horizon_metrics[metric].dropna().empty:
            continue
        fig, axis = plt.subplots(figsize=(6.5, 4.0))
        for method in METHOD_ORDER:
            rows = horizon_metrics[horizon_metrics["method"] == method].sort_values("horizon")
            rows = rows[np.isfinite(rows[metric])]
            if rows.empty:
                continue
            x = rows["horizon"].to_numpy(float)
            y = rows[metric].to_numpy(float)
            axis.plot(
                x,
                y,
                marker="o",
                markersize=3.5,
                linewidth=1.4,
                label=method,
                color=MODEL_COLORS[method],
            )
            low = rows[low_column].to_numpy(float)
            high = rows[high_column].to_numpy(float)
            if np.isfinite(low).all() and np.isfinite(high).all():
                axis.fill_between(x, low, high, color=MODEL_COLORS[method], alpha=0.12, linewidth=0)
        axis.set_xlabel("Forecast horizon (trading steps ahead)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(sorted(horizon_metrics["horizon"].unique()))
        if metric == "direction_accuracy":
            axis.set_ylim(0.0, 1.0)
            axis.axhline(0.5, color="#777777", linestyle=":", linewidth=0.8)
        else:
            axis.set_ylim(bottom=0.0)
        axis.grid(axis="y", color="#e0e0e0", linewidth=0.6)
        axis.legend(ncol=2, frameon=False)
        axis.set_title(f"{ylabel} by forecast horizon")
        fig.tight_layout()
        paths.extend(_save_figure(fig, figures_dir, stem))
    return paths


def plot_seed_difference_distribution(
    paired_runs: pd.DataFrame, figures_dir: Path, *, analysis_seed: int
) -> list[Path]:
    frame = paired_runs[paired_runs["model"].isin(LEARNED_METHODS)]
    if frame.empty:
        return []
    plt = _configure_matplotlib()
    methods = [method for method in LEARNED_METHODS if method in set(frame["model"])]
    data = [frame.loc[frame["model"] == method, "delta_mse"].to_numpy(float) for method in methods]
    fig, axis = plt.subplots(figsize=(7.2, 4.3))
    box = axis.boxplot(
        data,
        tick_labels=methods,
        showfliers=False,
        patch_artist=True,
        widths=0.55,
    )
    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(MODEL_COLORS[method])
        patch.set_alpha(0.28)
    rng = np.random.default_rng(analysis_seed)
    stocks = sorted(frame["stock"].unique())
    stock_colors = {stock: plt.cm.tab10(index % 10) for index, stock in enumerate(stocks)}
    for method_index, method in enumerate(methods, start=1):
        rows = frame[frame["model"] == method]
        jitter = rng.uniform(-0.15, 0.15, size=len(rows))
        axis.scatter(
            method_index + jitter,
            rows["delta_mse"],
            c=[stock_colors[stock] for stock in rows["stock"]],
            s=12,
            alpha=0.65,
            linewidths=0,
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    axis.set_ylabel("$\\Delta$MSE (model $-$ naive-last)")
    axis.set_title("Seed-level paired MSE differences (descriptive)")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", color="#e0e0e0", linewidth=0.6)
    fig.tight_layout()
    return _save_figure(fig, figures_dir, "fig_seed_level_delta_mse_distribution")


def plot_representative_trajectory(
    predictions: pd.DataFrame,
    tidy: pd.DataFrame,
    figures_dir: Path,
) -> tuple[list[Path], dict[str, Any] | None]:
    if predictions.empty:
        return [], None
    available = predictions[["stock", "seed", "method"]].drop_duplicates()
    complete_keys = []
    required = set(METHOD_ORDER[:5])
    for (stock, seed), group in available.groupby(["stock", "seed"]):
        if required.issubset(set(group["method"])):
            complete_keys.append((stock, int(seed)))
    if not complete_keys:
        return [], None
    eligible_stocks = sorted({stock for stock, _ in complete_keys})
    stock = "NVDA" if "NVDA" in eligible_stocks else eligible_stocks[0]
    stock_shared = tidy[
        (tidy["stock"] == stock) & (tidy["method"] == "Shared-target JEPA--MAE")
    ].copy()
    valid_seeds = {seed for candidate_stock, seed in complete_keys if candidate_stock == stock}
    stock_shared = stock_shared[stock_shared["seed"].astype(int).isin(valid_seeds)]
    median_mse = float(stock_shared["mse"].median())
    stock_shared["distance_to_median"] = (stock_shared["mse"] - median_mse).abs()
    selected_row = stock_shared.sort_values(["distance_to_median", "seed"]).iloc[0]
    seed = int(selected_row["seed"])
    rolling_step = int(
        predictions[(predictions["stock"] == stock) & (predictions["seed"] == seed)][
            "rolling_step"
        ].min()
    )
    frame = predictions[
        (predictions["stock"] == stock)
        & (predictions["seed"] == seed)
        & (predictions["rolling_step"] == rolling_step)
    ]
    actual_rows = frame[frame["method"] == "Shared-target JEPA--MAE"].sort_values("horizon")
    if actual_rows.empty:
        return [], None
    plt = _configure_matplotlib()
    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    x = actual_rows["horizon"].to_numpy(int)
    axis.plot(x, actual_rows["true_value"], marker="o", color="black", linewidth=2.0, label="Actual")
    for method in METHOD_ORDER:
        method_rows = frame[frame["method"] == method].sort_values("horizon")
        if method_rows.empty:
            continue
        if not np.allclose(
            method_rows["true_value"].to_numpy(float),
            actual_rows["true_value"].to_numpy(float),
            rtol=1e-8,
            atol=1e-10,
        ):
            continue
        axis.plot(
            method_rows["horizon"],
            method_rows["predicted_value"],
            marker="o",
            markersize=3.5,
            linewidth=1.3,
            color=MODEL_COLORS[method],
            label=method,
        )
    axis.set_xlabel("Forecast horizon (trading steps ahead)")
    axis.set_ylabel("Saved forecast target value")
    axis.set_xticks(x)
    axis.set_title(f"Representative forecast trajectory: {stock}, seed {seed}")
    axis.grid(axis="y", color="#e0e0e0", linewidth=0.6)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    paths = _save_figure(fig, figures_dir, "fig_representative_prediction_trajectory")
    target_dates = [str(value) for value in actual_rows["target_date"] if str(value)]
    metadata = {
        "stock": stock,
        "seed": seed,
        "rolling_step": rolling_step,
        "target_dates": target_dates,
        "selection_rule": (
            "Use NVDA when it has complete saved predictions (otherwise the "
            "alphabetically first complete stock); select the Shared-target seed "
            "whose overall test MSE is closest to that stock's median, breaking "
            "ties by smaller seed; plot the first saved rolling step."
        ),
    }
    return paths, metadata


def plot_downstream_diagnostics(tidy: pd.DataFrame, diagnostics_dir: Path) -> list[Path]:
    history_rows = []
    for row in tidy[tidy["method"].isin(LEARNED_METHODS[:2])].itertuples():
        path = Path(row.original_source_file).parent / "loss.txt"
        if not path.exists():
            continue
        try:
            history = pd.read_csv(path)
        except Exception:
            continue
        required = {"epoch", "train_loss", "val_mse"}
        if not required.issubset(history.columns):
            continue
        history = history.copy()
        history["stock"] = row.stock
        history["seed"] = int(row.seed)
        history["method"] = row.method
        history["source_file"] = str(path)
        history_rows.append(history)
    if not history_rows:
        return []
    histories = pd.concat(history_rows, ignore_index=True)
    per_stock = (
        histories.groupby(["method", "stock", "epoch"], as_index=False)
        .agg(train_loss=("train_loss", "mean"), val_mse=("val_mse", "mean"))
    )
    overall = (
        per_stock.groupby(["method", "epoch"], as_index=False)
        .agg(
            train_loss=("train_loss", "mean"),
            train_loss_std=("train_loss", "std"),
            val_mse=("val_mse", "mean"),
            val_mse_std=("val_mse", "std"),
        )
        .fillna(0.0)
    )
    plt = _configure_matplotlib()
    paths = []
    for metric, std_column, ylabel, stem in (
        ("train_loss", "train_loss_std", "Downstream training loss", "diagnostic_downstream_training_loss"),
        ("val_mse", "val_mse_std", "Validation MSE", "diagnostic_downstream_validation_mse"),
    ):
        fig, axis = plt.subplots(figsize=(6.5, 3.8))
        for method in METHOD_ORDER[:2]:
            rows = overall[overall["method"] == method].sort_values("epoch")
            if rows.empty:
                continue
            x = rows["epoch"].to_numpy(float)
            y = rows[metric].to_numpy(float)
            std = rows[std_column].to_numpy(float)
            axis.plot(x, y, label=method, color=MODEL_COLORS[method], linewidth=1.2)
            axis.fill_between(x, np.maximum(0.0, y - std), y + std, color=MODEL_COLORS[method], alpha=0.12)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{ylabel} (mean across stocks after seed averaging)")
        axis.grid(axis="y", color="#e0e0e0", linewidth=0.6)
        axis.legend(frameon=False)
        fig.tight_layout()
        paths.extend(_save_figure(fig, diagnostics_dir, stem))
    return paths


def plot_pretraining_diagnostics(
    bundles: Sequence[Bundle], diagnostics_dir: Path
) -> list[Path]:
    rows = []
    for bundle in bundles:
        if not bundle.selected or bundle.strategy not in STRATEGY_METHODS:
            continue
        history = bundle.metadata.get("validation_history")
        if not isinstance(history, list):
            continue
        method = STRATEGY_METHODS[bundle.strategy]
        for entry in history:
            if not isinstance(entry, dict):
                continue
            if not {"epoch", "total_loss", "jepa_loss", "mae_loss"}.issubset(entry):
                continue
            rows.append(
                {
                    "method": method,
                    "stock": bundle.stock,
                    "seed": bundle.seed,
                    "epoch": int(entry["epoch"]),
                    "total_loss": float(entry["total_loss"]),
                    "jepa_loss": float(entry["jepa_loss"]),
                    "mae_loss": float(entry["mae_loss"]),
                }
            )
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    stock_level = (
        frame.groupby(["method", "stock", "epoch"], as_index=False)
        .agg(
            total_loss=("total_loss", "mean"),
            jepa_loss=("jepa_loss", "mean"),
            mae_loss=("mae_loss", "mean"),
        )
    )
    overall = (
        stock_level.groupby(["method", "epoch"], as_index=False)
        .agg(
            total_loss=("total_loss", "mean"),
            jepa_loss=("jepa_loss", "mean"),
            mae_loss=("mae_loss", "mean"),
        )
    )
    plt = _configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), sharex=False)
    for axis, metric, title in zip(
        axes,
        ("total_loss", "jepa_loss", "mae_loss"),
        ("Weighted total", "JEPA latent", "MAE reconstruction"),
    ):
        for method in METHOD_ORDER[:2]:
            method_rows = overall[overall["method"] == method].sort_values("epoch")
            if method_rows.empty:
                continue
            axis.plot(
                method_rows["epoch"],
                method_rows[metric],
                color=MODEL_COLORS[method],
                linewidth=1.25,
                label=method,
            )
        axis.set_xlabel("Pre-training epoch")
        axis.set_ylabel("Validation loss")
        axis.set_title(title)
        axis.grid(axis="y", color="#e0e0e0", linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Pre-training validation diagnostics (optimization only)", y=1.04)
    fig.tight_layout()
    return _save_figure(fig, diagnostics_dir, "diagnostic_pretraining_losses")


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _best_second_methods(
    frame: pd.DataFrame, column: str, *, lower: bool
) -> tuple[set[str], set[str]]:
    finite = frame[np.isfinite(frame[column])]
    if finite.empty:
        return set(), set()
    values = sorted(finite[column].unique(), reverse=not lower)
    best_value = values[0]
    second_value = values[1] if len(values) > 1 else None
    best = set(finite.loc[finite[column] == best_value, "method"])
    second = (
        set(finite.loc[finite[column] == second_value, "method"])
        if second_value is not None
        else set()
    )
    return best, second


def _decorate(value: str, method: str, best: set[str], second: set[str]) -> str:
    if method in best:
        return rf"\textbf{{{value}}}"
    if method in second:
        return rf"\underline{{{value}}}"
    return value


def write_main_table(overall: pd.DataFrame, tables_dir: Path) -> list[Path]:
    if overall.empty:
        return []
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "table_main_metrics.csv"
    overall.to_csv(csv_path, index=False)
    best_second = {
        column: _best_second_methods(overall, column, lower=lower)
        for column, lower in (
            ("mse", True),
            ("mae", True),
            ("direction_accuracy", False),
            ("mse_average_rank", True),
            ("mae_average_rank", True),
            ("direction_accuracy_average_rank", True),
        )
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Overall forecasting performance. Each metric is first averaged across seeds within an equity and then across equities; $\pm$ denotes the standard deviation across equity-level means. Best values are bold and second-best values are underlined.}",
        r"\label{tab:main_metrics}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & MSE $\downarrow$ & MAE $\downarrow$ & Direction $\uparrow$ \\",
        r"\midrule",
    ]
    indexed = overall.set_index("method")
    for method in METHOD_ORDER:
        if method not in indexed.index:
            continue
        row = indexed.loc[method]
        formatted: dict[str, str] = {}
        for metric, std_column in (
            ("mse", "mse_std_across_stocks"),
            ("mae", "mae_std_across_stocks"),
            ("direction_accuracy", "direction_accuracy_std_across_stocks"),
        ):
            if math.isfinite(float(row[metric])):
                precision = 5 if metric in ("mse", "mae") else 3
                value = rf"${float(row[metric]):.{precision}f} \pm {float(row[std_column]):.{precision}f}$"
                formatted[metric] = _decorate(
                    value, method, *best_second[metric]
                )
            else:
                formatted[metric] = "--"
        lines.append(
            " & ".join(
                [
                    latex_escape(method),
                    formatted["mse"],
                    formatted["mae"],
                    formatted["direction_accuracy"],
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\medskip",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Method & MSE rank $\downarrow$ & MAE rank $\downarrow$ & Direction rank $\downarrow$ \\",
            r"\midrule",
        ]
    )
    for method in METHOD_ORDER:
        if method not in indexed.index:
            continue
        row = indexed.loc[method]
        formatted_ranks = []
        for metric in METRICS:
            column = f"{metric}_average_rank"
            value = (
                f"{float(row[column]):.2f}"
                if math.isfinite(float(row[column]))
                else "--"
            )
            formatted_ranks.append(
                _decorate(value, method, *best_second[column])
            )
        lines.append(
            " & ".join([latex_escape(method), *formatted_ranks]) + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    tex_path = tables_dir / "table_main_metrics.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [csv_path, tex_path]


def _format_p(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{value:.3f}" if value >= 0.001 else r"$<0.001$"


def write_paired_table(summary: pd.DataFrame, tables_dir: Path) -> list[Path]:
    if summary.empty:
        return []
    csv_path = tables_dir / "table_paired_vs_naive.csv"
    summary.to_csv(csv_path, index=False)
    lines: list[str] = []
    for metric in ("mse", "mae"):
        metric_upper = metric.upper()
        lines.extend(
            [
                r"\begin{table}[t]",
                r"\centering",
                r"\small",
                rf"\caption{{Stock-level paired {metric_upper} comparison against naive-last. Negative $\Delta$ and positive relative improvement favour the model. Confidence intervals bootstrap equity-level seed means; adjusted $p$ values use Holm correction across the three learned procedures.}}",
                rf"\label{{tab:paired_{metric}_naive}}",
                r"\begin{tabular}{lrrrrrrr}",
                r"\toprule",
                rf"Model & $\Delta${metric_upper} & 95\% CI & Rel. impr. & Adj. $p$ & $r_{{rb}}$ & Stocks & Runs \\",
                r"\midrule",
            ]
        )
        for row in summary.itertuples():
            delta = float(getattr(row, f"mean_delta_{metric}"))
            low = float(getattr(row, f"{metric}_ci_low"))
            high = float(getattr(row, f"{metric}_ci_high"))
            relative = float(getattr(row, f"relative_{metric}_improvement_pct"))
            adjusted = float(getattr(row, f"{metric}_holm_p_value"))
            effect = float(getattr(row, f"{metric}_rank_biserial"))
            stock_wins = int(getattr(row, f"{metric}_stock_wins"))
            stock_losses = int(getattr(row, f"{metric}_stock_losses"))
            run_wins = int(getattr(row, f"{metric}_run_wins"))
            run_total = int(getattr(row, f"{metric}_run_total"))
            lines.append(
                " & ".join(
                    [
                        latex_escape(row.model),
                        f"{delta:.5f}",
                        rf"[{low:.5f}, {high:.5f}]",
                        rf"{relative:.1f}\%",
                        _format_p(adjusted),
                        f"{effect:.2f}",
                        f"{stock_wins}/{stock_losses}",
                        f"{run_wins}/{run_total}",
                    ]
                )
                + r" \\"
            )
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        )
    tex_path = tables_dir / "table_paired_vs_naive.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return [csv_path, tex_path]


def write_shared_vs_local_table(
    summary: pd.DataFrame, tables_dir: Path
) -> list[Path]:
    if summary.empty:
        return []
    csv_path = tables_dir / "table_shared_vs_local.csv"
    summary.to_csv(csv_path, index=False)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Stock-level paired comparison of Shared-target JEPA--MAE and Local-MAE/Long-JEPA. Differences are Shared minus Local, so negative values favour Shared for both error metrics.}",
        r"\label{tab:shared_vs_local}",
        r"\begin{tabular}{lrrrrrrl}",
        r"\toprule",
        r"Metric & Stocks & Mean $\Delta$ & Median $\Delta$ & $t$ & $p$ & Cohen's $d_z$ & Status \\",
        r"\midrule",
    ]
    for row in summary.itertuples():
        t_statistic = (
            f"{float(row.t_statistic):.3f}"
            if math.isfinite(float(row.t_statistic))
            else "--"
        )
        effect = (
            f"{float(row.cohens_dz):.3f}"
            if math.isfinite(float(row.cohens_dz))
            else "--"
        )
        mean_delta = (
            f"{float(row.mean_delta):.5f}"
            if math.isfinite(float(row.mean_delta))
            else "--"
        )
        median_delta = (
            f"{float(row.median_delta):.5f}"
            if math.isfinite(float(row.median_delta))
            else "--"
        )
        status = {
            "ok": "ok",
            "insufficient_stock_observations": "insufficient stocks",
            "zero_variance_differences": "zero variance",
        }.get(str(row.status), str(row.status).replace("_", " "))
        lines.append(
            " & ".join(
                [
                    latex_escape(row.metric),
                    str(int(row.n_stocks)),
                    mean_delta,
                    median_delta,
                    t_statistic,
                    _format_p(float(row.p_value)),
                    effect,
                    latex_escape(status),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    tex_path = tables_dir / "table_shared_vs_local.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [csv_path, tex_path]


def write_appendix_table(stock_summary: pd.DataFrame, tables_dir: Path) -> list[Path]:
    if stock_summary.empty:
        return []
    csv_path = tables_dir / "table_appendix_stock_metrics.csv"
    stock_summary.to_csv(csv_path, index=False)
    lines = [
        r"\begin{longtable}{llrrrr}",
        r"\caption{Complete stock-level forecasting results. Standard deviations are across valid seeds.}\label{tab:appendix_stock_metrics}\\",
        r"\toprule",
        r"Equity & Method & MSE & MAE & Direction & Seeds \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Equity & Method & MSE & MAE & Direction & Seeds \\",
        r"\midrule",
        r"\endhead",
    ]
    method_index = {method: index for index, method in enumerate(METHOD_ORDER)}
    frame = stock_summary.assign(_order=stock_summary["method"].map(method_index)).sort_values(
        ["stock", "_order"]
    )
    for row in frame.itertuples():
        direction = (
            rf"${row.direction_accuracy_mean:.3f} \pm {row.direction_accuracy_std:.3f}$"
            if math.isfinite(float(row.direction_accuracy_mean))
            else "--"
        )
        lines.append(
            " & ".join(
                [
                    latex_escape(row.stock),
                    latex_escape(row.method),
                    rf"${row.mse_mean:.5f} \pm {row.mse_std:.5f}$",
                    rf"${row.mae_mean:.5f} \pm {row.mae_std:.5f}$",
                    direction,
                    str(int(row.n_valid_seeds)),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    tex_path = tables_dir / "table_appendix_stock_metrics.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [csv_path, tex_path]


def _display_unique(values: Iterable[Any]) -> str:
    cleaned = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if isinstance(value, (list, dict)):
            text = json.dumps(_jsonable(value), sort_keys=True)
        else:
            text = str(value)
        if text and text.lower() not in ("nan", "none"):
            cleaned.append(text)
    unique = sorted(set(cleaned))
    return "; ".join(unique) if unique else "Not recoverable from saved run artifacts"


def write_reproducibility_table(
    tidy: pd.DataFrame,
    bundles: Sequence[Bundle],
    scope: Mapping[str, Any],
    tables_dir: Path,
) -> list[Path]:
    if tidy.empty:
        return []
    selected_metadata = [bundle.metadata for bundle in bundles if bundle.selected]
    fields = [
        ("Equities", ", ".join(sorted(tidy["stock"].unique()))),
        ("Seeds", ", ".join(map(str, sorted(tidy["seed"].astype(int).unique())))),
        ("Train period end", _display_unique(metadata.get("train_end") for metadata in selected_metadata)),
        ("Test period", f"{_display_unique(tidy['test_start'])} to {_display_unique(tidy['test_end'])}"),
        ("Historical context length (rows)", _display_unique(metadata.get("window_length") for metadata in selected_metadata)),
        ("Forecast horizon", _display_unique(tidy["forecast_horizon"])),
        ("Patch size", _display_unique(metadata.get("patch_size") for metadata in selected_metadata)),
        ("Pre-training stride", _display_unique(metadata.get("pretrain_stride") for metadata in selected_metadata)),
        ("Downstream stride", _display_unique(metadata.get("eval_stride") for metadata in selected_metadata)),
        ("Input features", _display_unique(metadata.get("feature_cols") for metadata in selected_metadata)),
        ("Normalization", _display_unique(tidy["normalization"])),
        ("Batch size", _display_unique(metadata.get("batch_size") for metadata in selected_metadata)),
        ("Encoder dimension", _display_unique(metadata.get("encoder_dim") for metadata in selected_metadata)),
        ("Encoder layers", _display_unique(metadata.get("encoder_layers") for metadata in selected_metadata)),
        ("Attention heads", _display_unique(metadata.get("encoder_heads") for metadata in selected_metadata)),
        ("Predictor configuration", _display_unique({key: metadata.get(key) for key in ("predictor_dim", "predictor_layers", "predictor_heads")} for metadata in selected_metadata)),
        ("Mask ratio", _display_unique(metadata.get("mask_ratio") for metadata in selected_metadata)),
        ("Local/long target settings", _display_unique({key: metadata.get(key) for key in ("mae_window_patches", "jepa_gap_patches", "jepa_target_patches")} for metadata in selected_metadata)),
        ("JEPA loss weight", _display_unique(metadata.get("lambda_jepa") for metadata in selected_metadata)),
        ("MAE loss weight", _display_unique(metadata.get("lambda_mae") for metadata in selected_metadata)),
        ("EMA momentum", _display_unique(metadata.get("ema_momentum") for metadata in selected_metadata)),
        ("Transferred encoder weights", _display_unique(tidy["encoder_weight_source"])),
        ("Fine-tune encoder", _display_unique(tidy["fine_tune_encoder"])),
        ("Encoder learning rate", _display_unique(metadata.get("encoder_finetune_lr") for metadata in selected_metadata)),
        ("Forecast-head learning rate", _display_unique(metadata.get("forecast_head_lr") for metadata in selected_metadata)),
        ("Trend-loss settings", _display_unique({key: metadata.get(key) for key in ("trend_weight", "trend_loss_temperature", "trend_loss_threshold", "trend_selection_weight")} for metadata in selected_metadata)),
        ("Checkpoint selection", _display_unique(tidy["checkpoint_selection"])),
        ("Pre-training epochs", _display_unique(metadata.get("pretrain_epochs") for metadata in selected_metadata)),
        ("Downstream epochs", _display_unique(metadata.get("downstream_epochs") for metadata in selected_metadata)),
        ("Run Python version", _display_unique(metadata.get("python_version") for metadata in selected_metadata)),
        ("Run PyTorch version", _display_unique(metadata.get("pytorch_version") for metadata in selected_metadata)),
        ("Run CUDA version", _display_unique(metadata.get("cuda_version") for metadata in selected_metadata)),
        ("Run hardware", _display_unique(metadata.get("hardware") for metadata in selected_metadata)),
        ("Run Git commit", _display_unique(tidy["git_commit_sha"])),
    ]
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\small",
        r"\caption{Reproducibility information recovered from saved run metadata, runner commands, and checkpoint references. Unrecoverable fields are stated explicitly rather than inferred from current defaults.}",
        r"\label{tab:reproducibility}",
        r"\begin{tabular}{p{0.32\linewidth}p{0.60\linewidth}}",
        r"\toprule",
        r"Parameter & Recovered value \\",
        r"\midrule",
    ]
    for name, value in fields:
        lines.append(f"{latex_escape(name)} & {latex_escape(value)} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    tex_path = tables_dir / "table_reproducibility.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [tex_path]


def build_coverage_summary(tidy: pd.DataFrame, scope: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    expected_runs = len(scope["stocks"]) * len(scope["seeds"])
    for method in METHOD_ORDER:
        group = tidy[tidy["method"] == method] if not tidy.empty else tidy
        rows.append(
            {
                "method": method,
                "strategy": _method_strategy(method, scope["reference_strategy"]),
                "expected_stocks": len(scope["stocks"]),
                "available_stocks": int(group["stock"].nunique()) if not group.empty else 0,
                "expected_seeds_per_stock": len(scope["seeds"]),
                "available_unique_seeds": int(group["seed"].nunique()) if not group.empty else 0,
                "expected_runs": expected_runs,
                "available_runs": len(group),
                "coverage_pct": 100.0 * len(group) / expected_runs if expected_runs else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_configuration_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "config_signature",
        "strategy",
        "forecast_target",
        "normalization",
        "feature_transform",
        "forecast_horizon",
        "checkpoint_selection",
        "encoder_weight_source",
        "fine_tune_encoder",
        "trend_weight",
        "test_start",
        "test_end",
        "run_rows",
        "source_files",
    )
    if inventory.empty:
        return pd.DataFrame(columns=columns)
    frame = (
        inventory.groupby(
            [
                "config_signature",
                "strategy",
                "forecast_target",
                "normalization",
                "feature_transform",
                "forecast_horizon",
                "checkpoint_selection",
                "encoder_weight_source",
                "fine_tune_encoder",
                "trend_weight",
                "test_start",
                "test_end",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            run_rows=("original_source_file", "size"),
            source_files=(
                "original_source_file",
                lambda values: ";".join(sorted(set(map(str, values)))),
            ),
        )
        .reset_index(drop=True)
    )
    return frame.reindex(columns=columns)


def _git_value(arguments: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_None._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for row in frame.reindex(columns=columns).itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4g}" if math.isfinite(value) else "--")
            else:
                values.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def write_analysis_readme(
    output_dir: Path,
    *,
    scope: Mapping[str, Any],
    results_dir: Path,
    tidy: pd.DataFrame,
    inventory: pd.DataFrame,
    issues: pd.DataFrame,
    coverage: pd.DataFrame,
    paired_summary: pd.DataFrame,
    shared_local_summary: pd.DataFrame,
    representative_metadata: Mapping[str, Any] | None,
    artifact_records: Sequence[Mapping[str, Any]],
    command: str,
) -> Path:
    sources = (
        sorted(set(inventory["original_source_file"].dropna().astype(str)))
        if not inventory.empty
        else []
    )
    excluded = (
        inventory[~inventory["included_in_canonical"].astype(bool)]
        if not inventory.empty
        else inventory
    )
    error_count = int((issues["severity"] == "error").sum()) if not issues.empty else 0
    warning_count = int((issues["severity"] == "warning").sum()) if not issues.empty else 0
    lines = [
        "# TS-JEPA thesis result manifest",
        "",
        "This directory is generated exclusively from saved experiment artifacts; no model training is performed.",
        "",
        "## Analysis scope and coverage",
        "",
        f"- Config: `{scope['config_path']}`",
        f"- Result root: `{results_dir}`",
        f"- Expected equities: {', '.join(scope['stocks'])}",
        f"- Expected seeds: {', '.join(map(str, scope['seeds']))}",
        f"- Strategies: {', '.join(scope['strategies'])}",
        f"- Canonical baseline/GRU strategy: `{scope['reference_strategy']}`",
        f"- Canonical rows: {len(tidy)}",
        f"- Audit issues: {error_count} errors and {warning_count} warnings",
        "",
        _markdown_table(
            coverage,
            ["method", "available_runs", "expected_runs", "coverage_pct"],
        ),
        "",
        "`missing_runs.csv` is authoritative for missing stocks, seeds, methods, duplicate reruns, conflicting configurations, non-finite values, test-target mismatches, and deterministic-baseline inconsistencies. Incomplete coverage is never silently imputed.",
        "",
        "## Source data and exclusions",
        "",
    ]
    if sources:
        lines.extend([f"- `{source}`" for source in sources])
    else:
        lines.append("No compatible comparison source file was found in the configured result root.")
    lines.extend(["", f"Excluded inventory rows: {len(excluded)}."])
    if not excluded.empty:
        reason_counts = excluded["exclusion_reason"].value_counts()
        lines.extend([f"- {reason}: {count}" for reason, count in reason_counts.items()])
    lines.extend(
        [
            "",
            "The inventory selects the latest timestamp only for duplicate bundles. A duplicate with multiple recoverable configuration signatures is an error. Strategy-specific GRU rows outside the configured reference strategy and duplicate deterministic baselines are retained in `run_inventory.csv` but excluded from the canonical dataset.",
            "",
            "## Metrics and direction-accuracy audit",
            "",
            "MSE and MAE are recomputed over all saved rolling-step × horizon values whenever score files exist. Stored summary values are checked against those reconstructions.",
            "",
            f"Direction accuracy uses `{DIRECTION_DEFINITION}`. The identical implementation is applied to learned models and baselines. Naive-last and mean-context therefore have valid direction scores when trajectories are saved; a constant predicted value path generally produces zero-valued predicted differences for a value target, which only count as correct when the corresponding true difference is also zero. Unsupported values remain missing.",
            "",
            "All reported values remain in the saved target space (`normalization`, `forecast_target`, and `target_definition` are preserved in `all_runs_tidy.csv`). No conversion to absolute prices is inferred.",
            "",
            "## Aggregation and statistical procedure",
            "",
            "For each method, metrics are first averaged over seeds within an equity. Overall metrics and ranks are then averaged over equities. Variability in the main table is the standard deviation across equity-level seed means.",
            "",
            "Paired differences use Δ = method − naive-last, so negative values favour the method. Relative improvement is `100 × (naive − method) / naive`, so positive values favour the method. Pairing requires the same equity, seed, strategy-specific run bundle, target definition, normalization, metric definition, horizon, and saved target signature whenever available.",
            "",
            "Primary inference averages seed-level paired differences within each equity and uses equities as the statistical units. The 95% interval is a percentile bootstrap that resamples equity-level means. The Wilcoxon result is an exact two-sided signed-rank sign-permutation test (zero differences removed); rank-biserial correlation is signed in Δ coordinates, so a negative effect favours the model. P-values are Holm-adjusted separately for the three learned-model MSE and MAE comparisons. Seed-level win counts and distribution figures are descriptive only.",
            "",
            "The separate Shared-vs-Local comparison first retains compatible seeds matched by equity, seed, target definition, normalization, metric definition, forecast horizon, test period, and saved target signature. Shared and Local MSE/MAE are averaged over the same matched seeds within each equity. A two-sided paired Student t-test and signed Cohen's dz then use the resulting equity-level values only, with Δ = Shared − Local; negative values favour Shared for these error metrics. No Direction Accuracy test or multiple-comparison correction is applied to this separate comparison.",
            "",
            "## Paired results snapshot",
            "",
            _markdown_table(
                paired_summary,
                [
                    "model",
                    "mean_delta_mse",
                    "mse_ci_low",
                    "mse_ci_high",
                    "mse_holm_p_value",
                    "mse_stock_wins",
                    "mse_run_wins",
                    "mse_run_total",
                ],
            ),
            "",
            "## Shared-target vs Local-MAE/Long-JEPA snapshot",
            "",
            _markdown_table(
                shared_local_summary,
                [
                    "metric",
                    "n_stocks",
                    "mean_delta",
                    "median_delta",
                    "t_statistic",
                    "p_value",
                    "cohens_dz",
                    "status",
                ],
            ),
            "",
            "## Representative trajectory",
            "",
        ]
    )
    if representative_metadata:
        lines.extend(
            [
                f"- Stock: {representative_metadata['stock']}",
                f"- Seed: {representative_metadata['seed']}",
                f"- Rolling step: {representative_metadata['rolling_step']}",
                f"- Target dates: {', '.join(representative_metadata.get('target_dates', [])) or 'not saved'}",
                f"- Selection rule: {representative_metadata['selection_rule']}",
            ]
        )
    else:
        lines.append("Not generated: complete aligned raw predictions were unavailable.")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "The Shared-target and Local-MAE/Long-JEPA rows are complete forecasting procedures, not causal ablations of JEPA, MAE, EMA, or pre-training. Unless separately controlled rows appear in the inventory, random-initialized Transformer, JEPA-only, MAE-only, and online-vs-EMA causal analyses cannot be produced. Decreasing pre-training or downstream loss demonstrates optimization behaviour only.",
            "",
            "Forecast-horizon and qualitative figures are omitted when score-level predictions are absent. Pre-training JEPA/MAE/total-loss diagnostics are omitted when checkpoint validation histories or training histories are absent. These omissions are recorded in `artifact_manifest.csv`.",
            "",
            "## Generated artifact manifest",
            "",
            _markdown_table(pd.DataFrame(artifact_records), ["artifact", "status", "description"]),
            "",
            "## Reproduction command",
            "",
            "```bash",
            command,
            "```",
            "",
            "The analysis bootstrap seed and sample count are recorded in the command and output metadata. PDF figures are vector outputs; matching PNG files are previews.",
        ]
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


def _record_artifact(
    records: list[dict[str, Any]],
    path: Path,
    *,
    status: str,
    source_data: str,
    description: str,
    output_dir: Path,
) -> None:
    records.append(
        {
            "artifact": str(path.relative_to(output_dir)),
            "status": status,
            "source_data": source_data,
            "description": description,
        }
    )


def _clear_stale_thesis_artifacts(output_dir: Path) -> None:
    """Remove only outputs owned by this pipeline before a fresh audit."""
    relative_paths = [Path("tables") / name for name in THESIS_TABLE_FILES]
    relative_paths.extend(
        Path("figures") / f"{stem}.{suffix}"
        for stem in THESIS_FIGURE_STEMS
        for suffix in ("pdf", "png")
    )
    relative_paths.extend(
        Path("figures") / "diagnostics" / f"{stem}.{suffix}"
        for stem in THESIS_DIAGNOSTIC_STEMS
        for suffix in ("pdf", "png")
    )
    relative_paths.append(Path("data") / "representative_example.json")
    for relative_path in relative_paths:
        path = output_dir / relative_path
        if path.is_file():
            path.unlink()


def _console_report(
    *,
    configurations: pd.DataFrame,
    coverage: pd.DataFrame,
    issues: pd.DataFrame,
    overall: pd.DataFrame,
    paired: pd.DataFrame,
    shared_local: pd.DataFrame,
    artifacts: Sequence[Mapping[str, Any]],
) -> None:
    print("\n=== Thesis results analysis report ===")
    print(f"Discovered experiment configurations: {len(configurations)}")
    if not configurations.empty:
        for row in configurations.itertuples():
            print(
                f"  {row.config_signature}: strategy={row.strategy}, "
                f"target={row.forecast_target}, normalization={row.normalization}, "
                f"rows={row.run_rows}"
            )
    print("Valid canonical runs per method:")
    for row in coverage.itertuples():
        print(f"  {row.method}: {row.available_runs}/{row.expected_runs}")
    if not issues.empty:
        counts = issues.groupby(["severity", "status"]).size().sort_values(ascending=False)
        print("Missing/incompatible findings:")
        for (severity, status), count in counts.items():
            print(f"  {severity}/{status}: {count}")
    print("Overall stock-aware results:")
    if overall.empty:
        print("  unavailable")
    else:
        for row in overall.itertuples():
            print(
                f"  {row.method}: MSE={row.mse:.6g}, MAE={row.mae:.6g}, "
                f"direction={row.direction_accuracy:.4g}"
            )
    print("Paired learned models vs naive-last:")
    if paired.empty:
        print("  unavailable")
    else:
        for row in paired.itertuples():
            print(
                f"  {row.model}: delta_MSE={row.mean_delta_mse:.6g}, "
                f"CI=[{row.mse_ci_low:.6g}, {row.mse_ci_high:.6g}], "
                f"Holm p={row.mse_holm_p_value:.4g}, "
                f"stock wins={row.mse_stock_wins}/{row.n_stocks}, "
                f"run wins={row.mse_run_wins}/{row.mse_run_total}"
            )
    print("Paired Shared-target vs Local-MAE/Long-JEPA:")
    for row in shared_local.itertuples():
        print(
            f"  {row.metric}: n_stocks={row.n_stocks}, "
            f"delta={row.mean_delta:.6g}, t={row.t_statistic:.6g}, "
            f"p={row.p_value:.4g}, dz={row.cohens_dz:.4g}, status={row.status}"
        )
    generated = [record["artifact"] for record in artifacts if record["status"] == "generated"]
    print("Generated artifacts:")
    for artifact in generated:
        print(f"  {artifact}")


def run_analysis(args: argparse.Namespace) -> int:
    scope, results_dir = load_scope(args)
    output_dir = (
        results_dir.parent.parent / "analysis_artifacts" / Path(args.config).stem
    )
    data_dir = output_dir / "data"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    diagnostics_dir = figures_dir / "diagnostics"
    for directory in (data_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _clear_stale_thesis_artifacts(output_dir)

    print(f"Result root: {results_dir}")
    print(f"Analysis output: {output_dir}")
    print(f"Expected stocks ({len(scope['stocks'])}): {', '.join(scope['stocks'])}")
    print(f"Expected seeds ({len(scope['seeds'])}): {', '.join(map(str, scope['seeds']))}")
    print(f"Strategies: {', '.join(scope['strategies'])}")

    bundles, issue_rows = discover_bundles(scope, results_dir)
    selected = select_bundles(bundles, issue_rows)
    tidy, inventory, predictions, paired_runs = build_canonical_data(
        bundles, selected, scope, issue_rows
    )
    stock_summary = build_stock_summary(tidy)
    overall_summary = build_overall_summary(stock_summary)
    paired_summary, paired_stocks = build_paired_summary(
        paired_runs,
        analysis_seed=args.analysis_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    shared_local_stocks, shared_local_summary = build_shared_vs_local_comparison(
        tidy, scope, issue_rows
    )
    relative_stock = build_relative_stock_data(paired_runs)
    horizon_metrics = build_horizon_metrics(
        predictions,
        tidy,
        analysis_seed=args.analysis_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    validate_summaries(tidy, stock_summary, predictions, issue_rows)
    issues = pd.DataFrame(issue_rows, columns=ISSUE_COLUMNS)
    if not issues.empty:
        issues = issues.sort_values(
            ["severity", "status", "stock", "seed", "method", "strategy"],
            na_position="last",
        ).reset_index(drop=True)
    coverage = build_coverage_summary(tidy, scope)
    configurations = build_configuration_inventory(inventory)

    datasets = {
        "all_runs_tidy.csv": tidy,
        "run_inventory.csv": inventory,
        "missing_runs.csv": issues,
        "coverage_summary.csv": coverage,
        "configuration_inventory.csv": configurations,
        "predictions_tidy.csv": predictions,
        "paired_run_differences.csv": paired_runs,
        "paired_stock_differences.csv": paired_stocks,
        "stock_summary.csv": stock_summary,
        "overall_summary.csv": overall_summary,
        "paired_vs_naive.csv": paired_summary,
        "paired_shared_vs_local.csv": shared_local_stocks,
        "relative_performance_by_stock.csv": relative_stock,
        "horizon_metrics.csv": horizon_metrics,
    }
    artifact_records: list[dict[str, Any]] = []
    for filename, frame in datasets.items():
        path = data_dir / filename
        frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
        _record_artifact(
            artifact_records,
            path,
            status="generated",
            source_data=(
                "saved experiment CSV/JSON/TXT artifacts"
                if filename in ("all_runs_tidy.csv", "run_inventory.csv", "missing_runs.csv", "predictions_tidy.csv")
                else "data/all_runs_tidy.csv and/or data/predictions_tidy.csv"
            ),
            description=f"Canonical analysis dataset: {filename}",
            output_dir=output_dir,
        )

    print("\nCoverage summary (before thesis outputs):")
    for row in coverage.itertuples():
        print(f"  {row.method}: {row.available_runs}/{row.expected_runs} runs ({row.coverage_pct:.1f}%)")

    generated_tables: list[Path] = []
    generated_figures: list[Path] = []
    representative_metadata = None
    validity_errors_present = (
        not issues.empty and bool((issues["severity"] == "error").any())
    )
    generate_thesis_outputs = not tidy.empty and (
        args.allow_incomplete or not validity_errors_present
    )
    if generate_thesis_outputs:
        generated_tables.extend(write_main_table(overall_summary, tables_dir))
        generated_tables.extend(write_paired_table(paired_summary, tables_dir))
        generated_tables.extend(
            write_shared_vs_local_table(shared_local_summary, tables_dir)
        )
        generated_tables.extend(write_appendix_table(stock_summary, tables_dir))
        generated_tables.extend(
            write_reproducibility_table(tidy, bundles, scope, tables_dir)
        )
        if not args.skip_figures:
            generated_figures.extend(
                plot_paired_forest(
                    paired_runs,
                    paired_summary,
                    figures_dir,
                    metric="mse",
                    analysis_seed=args.analysis_seed,
                    bootstrap_samples=args.bootstrap_samples,
                )
            )
            generated_figures.extend(
                plot_paired_forest(
                    paired_runs,
                    paired_summary,
                    figures_dir,
                    metric="mae",
                    analysis_seed=args.analysis_seed,
                    bootstrap_samples=args.bootstrap_samples,
                )
            )
            generated_figures.extend(plot_relative_heatmaps(relative_stock, figures_dir))
            generated_figures.extend(plot_direction_heatmap(stock_summary, figures_dir))
            generated_figures.extend(plot_horizon_metrics(horizon_metrics, figures_dir))
            generated_figures.extend(
                plot_seed_difference_distribution(
                    paired_runs, figures_dir, analysis_seed=args.analysis_seed
                )
            )
            trajectory_paths, representative_metadata = plot_representative_trajectory(
                predictions, tidy, figures_dir
            )
            generated_figures.extend(trajectory_paths)
            generated_figures.extend(
                plot_downstream_diagnostics(tidy, diagnostics_dir)
            )
            generated_figures.extend(
                plot_pretraining_diagnostics(bundles, diagnostics_dir)
            )
    for path in generated_tables:
        source_data = (
            "data/paired_shared_vs_local.csv"
            if "shared_vs_local" in path.name
            else "data/stock_summary.csv, data/overall_summary.csv, and/or data/paired_vs_naive.csv"
        )
        _record_artifact(
            artifact_records,
            path,
            status="generated",
            source_data=source_data,
            description="Thesis or appendix table",
            output_dir=output_dir,
        )
    for path in generated_figures:
        _record_artifact(
            artifact_records,
            path,
            status="generated",
            source_data="data/paired_run_differences.csv, data/stock_summary.csv, data/horizon_metrics.csv, and/or data/predictions_tidy.csv",
            description="Publication-quality thesis figure",
            output_dir=output_dir,
        )
    expected_optional = {
        "tables/table_main_metrics.tex": "Complete compatible model coverage was unavailable.",
        "tables/table_paired_vs_naive.tex": "Paired learned-model and naive-last runs were unavailable.",
        "tables/table_shared_vs_local.tex": "Compatible Shared-vs-Local stock pairs were unavailable.",
        "tables/table_appendix_stock_metrics.tex": "No compatible stock-level summaries were available.",
        "tables/table_reproducibility.tex": "No included run metadata were available.",
        "figures/fig_paired_mse_forest.pdf": "Paired stock-level MSE observations were unavailable.",
        "figures/fig_paired_mae_forest.pdf": "Paired stock-level MAE observations were unavailable.",
        "figures/fig_relative_mse_heatmap.pdf": "Paired stock-level relative MSE observations were unavailable.",
        "figures/fig_relative_mae_heatmap.pdf": "Paired stock-level relative MAE observations were unavailable.",
        "figures/fig_direction_accuracy_heatmap.pdf": "Direction-accuracy observations were unavailable.",
        "figures/fig_mse_by_horizon.pdf": "Raw horizon predictions were unavailable.",
        "figures/fig_mae_by_horizon.pdf": "Raw horizon predictions were unavailable.",
        "figures/fig_direction_by_horizon.pdf": "Raw horizon predictions or supported horizon directions were unavailable.",
        "figures/fig_seed_level_delta_mse_distribution.pdf": "Seed-level paired observations were unavailable.",
        "figures/fig_representative_prediction_trajectory.pdf": "Complete aligned raw predictions were unavailable.",
        "figures/diagnostics/diagnostic_downstream_training_loss.pdf": "Timestamp-aligned downstream histories were unavailable.",
        "figures/diagnostics/diagnostic_downstream_validation_mse.pdf": "Timestamp-aligned downstream validation histories were unavailable.",
        "figures/diagnostics/diagnostic_pretraining_losses.pdf": "Pre-training JEPA/MAE/total histories were unavailable.",
    }
    recorded_names = {record["artifact"] for record in artifact_records}
    for relative, reason in expected_optional.items():
        if relative not in recorded_names:
            artifact_records.append(
                {
                    "artifact": relative,
                    "status": "omitted",
                    "source_data": "none",
                    "description": reason,
                }
            )
    if representative_metadata:
        metadata_path = data_dir / "representative_example.json"
        metadata_path.write_text(
            json.dumps(_jsonable(representative_metadata), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _record_artifact(
            artifact_records,
            metadata_path,
            status="generated",
            source_data="data/predictions_tidy.csv and data/all_runs_tidy.csv",
            description="Deterministic representative-example selection metadata",
            output_dir=output_dir,
        )

    command_parts = [
        "conda run --no-capture-output -n ts-jepa python analyze_thesis_results.py",
        f"--config {shlex.quote(str(args.config))}",
        f"--reference-strategy {shlex.quote(scope['reference_strategy'])}",
        f"--bootstrap-samples {args.bootstrap_samples}",
        f"--analysis-seed {args.analysis_seed}",
    ]
    if args.allow_incomplete:
        command_parts.append("--allow-incomplete")
    if args.skip_figures:
        command_parts.append("--skip-figures")
    command = (" " + "\\" + "\n  ").join(command_parts)
    readme_path = output_dir / "README.md"
    manifest_path = output_dir / "artifact_manifest.csv"
    metadata_path = output_dir / "analysis_metadata.json"
    _record_artifact(
        artifact_records,
        readme_path,
        status="generated",
        source_data="all generated data and artifact status records",
        description="Methodology, coverage, exclusions, interpretation limits, and reproduction command",
        output_dir=output_dir,
    )
    _record_artifact(
        artifact_records,
        manifest_path,
        status="generated",
        source_data="artifact generation records",
        description="Mapping from every planned output artifact to its source data and status",
        output_dir=output_dir,
    )
    _record_artifact(
        artifact_records,
        metadata_path,
        status="generated",
        source_data="analysis command, environment, Git checkout, and audit counts",
        description="Machine-readable analysis provenance",
        output_dir=output_dir,
    )
    readme_path = write_analysis_readme(
        output_dir,
        scope=scope,
        results_dir=results_dir,
        tidy=tidy,
        inventory=inventory,
        issues=issues,
        coverage=coverage,
        paired_summary=paired_summary,
        shared_local_summary=shared_local_summary,
        representative_metadata=representative_metadata,
        artifact_records=artifact_records,
        command=command,
    )
    artifact_manifest = pd.DataFrame(artifact_records)
    artifact_manifest.to_csv(manifest_path, index=False)
    analysis_metadata = {
        "analysis_git_commit": _git_value(["rev-parse", "HEAD"]),
        "analysis_git_branch": _git_value(["branch", "--show-current"]),
        "analysis_python": platform.python_version(),
        "analysis_numpy": np.__version__,
        "analysis_pandas": pd.__version__,
        "analysis_seed": args.analysis_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "scope": {key: value for key, value in scope.items() if key != "config_data"},
        "results_dir": str(results_dir),
        "canonical_rows": len(tidy),
        "error_issues": int((issues["severity"] == "error").sum()) if not issues.empty else 0,
        "warning_issues": int((issues["severity"] == "warning").sum()) if not issues.empty else 0,
    }
    metadata_path.write_text(
        json.dumps(_jsonable(analysis_metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _console_report(
        configurations=configurations,
        coverage=coverage,
        issues=issues,
        overall=overall_summary,
        paired=paired_summary,
        shared_local=shared_local_summary,
        artifacts=artifact_records,
    )
    error_count = analysis_metadata["error_issues"]
    if error_count and not args.allow_incomplete:
        raise RuntimeError(
            f"Analysis found {error_count} validity error(s). Inventory outputs were "
            f"written to {output_dir}; rerun with --allow-incomplete only for "
            "explicitly exploratory partial summaries."
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_analysis(args)
