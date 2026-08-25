"""Configuration and result analysis for the controlled sentiment ablations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from config.experiment import effective_feature_columns, resolve_forecast_horizon
from run_top_nasdaq100_stocks import (
    effective_experiment_config,
    experiment_config_signature,
    parse_args as parse_runner_args,
    resolve_preprocessing_settings,
)


EXPECTED_STOCKS = [
    "NVDA",
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "AVGO",
    "META",
    "TSLA",
    "COST",
    "WMT",
]
EXPECTED_SEEDS = list(range(42, 52))
EXPECTED_CONFIGS = {
    "top10_h1_without_sentiment.json": {
        "forecast_horizon": 1,
        "feature_cols": ["Close", "Volume", "MA10", "MA50"],
        "input_dimension": 20,
        "sentiment_normalization": "none",
    },
    "top10_h1_with_sentiment.json": {
        "forecast_horizon": 1,
        "feature_cols": [
            "Close",
            "Volume",
            "MA10",
            "MA50",
            "sentiment_mean",
        ],
        "input_dimension": 25,
        "sentiment_normalization": "none",
    },
    "top10_sentiment_has_news.json": {
        "forecast_horizon": 5,
        "feature_cols": [
            "Close",
            "Volume",
            "MA10",
            "MA50",
            "sentiment_mean",
            "has_news",
        ],
        "input_dimension": 30,
        "sentiment_normalization": "none",
    },
    "top10_sentiment_zscore.json": {
        "forecast_horizon": 5,
        "feature_cols": [
            "Close",
            "Volume",
            "MA10",
            "MA50",
            "sentiment_mean_z",
        ],
        "input_dimension": 25,
        "sentiment_normalization": "train_zscore",
    },
}

CANONICAL_COLUMNS = [
    "condition",
    "stock",
    "seed",
    "model",
    "metric",
    "value",
    "forecast_horizon",
    "source_file",
]
CANONICAL_MODELS = (
    "TS-JEPA/random",
    "TS-JEPA/local_long",
    "GRU/random",
)
CANONICAL_METRICS = ("mse", "mae", "direction_accuracy")


def _semantic_options(args) -> dict[str, Any]:
    effective = effective_experiment_config(args)
    preprocessing = resolve_preprocessing_settings(args)
    horizon = resolve_forecast_horizon(args.forecast_horizon, args.patch_size)
    preprocessing_keys = {
        "feature_transform",
        "normalization",
        "market_data",
        "market_features",
        "sentiment_features",
        "sentiment_normalization",
        "use_sentiment",
    }
    downstream_keys = {"forecast_target", "forecast_horizon", "eval_num_epochs"}
    normalized_preprocessing = {
        key: effective.get(key)
        for key in sorted(preprocessing_keys)
        if key in effective
    }
    normalized_preprocessing["sentiment_features"] = list(
        preprocessing.get("sentiment_features") or []
    )
    normalized_preprocessing["sentiment_normalization"] = preprocessing[
        "sentiment_normalization"
    ]
    normalized_downstream = {
        key: effective.get(key)
        for key in sorted(downstream_keys)
        if key in effective
    }
    normalized_downstream["forecast_horizon"] = horizon
    other = {
        key: value
        for key, value in effective.items()
        if key not in preprocessing_keys | downstream_keys
    }
    return {
        "runner": {
            "preprocessing": normalized_preprocessing,
            "downstream": normalized_downstream,
            "other": other,
        }
    }


def semantic_experiment_config(config_path: Path) -> dict[str, Any]:
    """Resolve a config through the production parser into canonical semantics."""
    config_path = Path(config_path).resolve()
    args = parse_runner_args(["--config", str(config_path)])
    preprocessing = resolve_preprocessing_settings(args)
    market_features = list(
        preprocessing.get("market_features")
        or ["Close", "Volume", "MA10", "MA50"]
    )
    sentiment_features = list(preprocessing.get("sentiment_features") or [])
    feature_cols = effective_feature_columns(
        market_features,
        sentiment_features or ["sentiment_mean"],
        preprocessing["use_sentiment"],
    )
    forecast_horizon = resolve_forecast_horizon(
        args.forecast_horizon,
        args.patch_size,
    )
    return {
        "config_name": config_path.stem,
        "config_path": str(config_path),
        "stocks": list(args.stocks),
        "seeds": list(args.seeds),
        "patch_size": int(args.patch_size),
        "forecast_horizon": forecast_horizon,
        "feature_cols": feature_cols,
        "feature_count": len(feature_cols),
        "input_dimension": int(args.patch_size) * len(feature_cols),
        "mask_strategies": list(args.mask_strategies),
        "sentiment_normalization": preprocessing["sentiment_normalization"],
        "results_dir": str(Path(args.results_dir).resolve()),
        "effective_config": effective_experiment_config(args),
        "semantic_options": _semantic_options(args),
    }


def nested_config_diff(
    control: Mapping[str, Any],
    intervention: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return deterministic dotted-path differences between nested mappings."""
    differences: dict[str, dict[str, Any]] = {}

    def visit(left: Any, right: Any, prefix: str) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for key in sorted(set(left) | set(right)):
                path = f"{prefix}.{key}" if prefix else str(key)
                visit(left.get(key), right.get(key), path)
            return
        if left != right:
            differences[prefix] = {
                "control": left,
                "intervention": right,
            }

    visit(control, intervention, "")
    return differences


def _published_control_status(repo_root: Path) -> dict[str, Any]:
    controls = {
        "top10_with_sentiment.json": repo_root
        / "thesis_results/top10_with_sentiment/5b8f3897bf23-02add88f32d5/provenance/experiment_manifest.json",
        "top10_without_sentiment.json": repo_root
        / "thesis_results/top10_without_sentiment/2fab810c1e1d-d0fb2944255b/provenance/experiment_manifest.json",
    }
    report = {}
    for filename, manifest_path in controls.items():
        config_path = repo_root / "config" / "experiments" / filename
        args = parse_runner_args(["--config", str(config_path)])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        effective_match = (
            effective_experiment_config(args) == manifest["effective_config"]
        )
        signature_match = (
            experiment_config_signature(args) == manifest["config_signature"]
        )
        report[filename] = {
            "manifest": str(manifest_path),
            "effective_config_match": effective_match,
            "signature_match": signature_match,
            "verified": effective_match and signature_match,
        }
    return report


def validate_ablation_configs(repo_root: Path) -> dict[str, Any]:
    """Validate coverage, dimensions, immutable controls, and isolation rules."""
    repo_root = Path(repo_root).resolve()
    config_dir = repo_root / "config" / "experiments"
    snapshots = {
        filename: semantic_experiment_config(config_dir / filename)
        for filename in EXPECTED_CONFIGS
    }
    controls = {
        "with": semantic_experiment_config(config_dir / "top10_with_sentiment.json"),
        "without": semantic_experiment_config(
            config_dir / "top10_without_sentiment.json"
        ),
    }

    comparisons = {
        "h1_without_vs_control": (
            controls["without"],
            snapshots["top10_h1_without_sentiment.json"],
            {"runner.downstream.forecast_horizon"},
        ),
        "h1_with_vs_control": (
            controls["with"],
            snapshots["top10_h1_with_sentiment.json"],
            {"runner.downstream.forecast_horizon"},
        ),
        "h1_with_vs_without": (
            snapshots["top10_h1_without_sentiment.json"],
            snapshots["top10_h1_with_sentiment.json"],
            {
                "runner.preprocessing.sentiment_features",
                "runner.preprocessing.use_sentiment",
            },
        ),
        "has_news_vs_control": (
            controls["with"],
            snapshots["top10_sentiment_has_news.json"],
            {"runner.preprocessing.sentiment_features"},
        ),
        "zscore_vs_control": (
            controls["with"],
            snapshots["top10_sentiment_zscore.json"],
            {
                "runner.preprocessing.sentiment_features",
                "runner.preprocessing.sentiment_normalization",
            },
        ),
    }
    comparison_report = {}
    isolation_valid = True
    for name, (control, intervention, allowed) in comparisons.items():
        differences = nested_config_diff(
            control["semantic_options"],
            intervention["semantic_options"],
        )
        actual = set(differences)
        valid = actual == allowed
        isolation_valid = isolation_valid and valid
        comparison_report[name] = {
            "valid": valid,
            "allowed_paths": sorted(allowed),
            "actual_paths": sorted(actual),
            "differences": differences,
        }

    config_report = {}
    matrix_valid = True
    for filename, expected in EXPECTED_CONFIGS.items():
        snapshot = snapshots[filename]
        checks = {
            "stocks": snapshot["stocks"] == EXPECTED_STOCKS,
            "seeds": snapshot["seeds"] == EXPECTED_SEEDS,
            "patch_size": snapshot["patch_size"] == 5,
            "forecast_horizon": snapshot["forecast_horizon"]
            == expected["forecast_horizon"],
            "feature_cols": snapshot["feature_cols"] == expected["feature_cols"],
            "input_dimension": snapshot["input_dimension"]
            == expected["input_dimension"],
            "mask_strategies": snapshot["mask_strategies"]
            == ["random", "local_long"],
            "sentiment_normalization": snapshot["sentiment_normalization"]
            == expected["sentiment_normalization"],
        }
        valid = all(checks.values())
        matrix_valid = matrix_valid and valid
        config_report[filename] = {
            "valid": valid,
            "checks": checks,
            "snapshot": snapshot,
        }

    published = _published_control_status(repo_root)
    published_verified = all(item["verified"] for item in published.values())
    return {
        "valid": matrix_valid and isolation_valid and published_verified,
        "published_controls_verified": published_verified,
        "published_controls": published,
        "configs": config_report,
        "comparisons": comparison_report,
    }


def _canonical_model(method: str, strategy: str) -> str | None:
    normalized = str(method).strip().lower()
    strategy = str(strategy).strip().lower()
    if normalized in ("shared-target jepa--mae", "ts-jepa") and strategy == "random":
        return "TS-JEPA/random"
    if normalized in ("local-mae/long-jepa", "ts-jepa") and strategy == "local_long":
        return "TS-JEPA/local_long"
    if normalized == "gru" and strategy == "random":
        return "GRU/random"
    return None


def _validate_canonical_results(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = [column for column in CANONICAL_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing canonical columns: {missing}")
    result = frame[CANONICAL_COLUMNS].copy()
    result["stock"] = result["stock"].astype(str).str.upper()
    result["seed"] = pd.to_numeric(result["seed"], errors="raise").astype(int)
    result["forecast_horizon"] = pd.to_numeric(
        result["forecast_horizon"], errors="raise"
    ).astype(int)
    result["value"] = pd.to_numeric(result["value"], errors="raise")
    if not np.isfinite(result["value"].to_numpy(dtype=float)).all():
        raise ValueError(f"{label} metric values must be finite")
    key = ["stock", "seed", "model", "metric"]
    if result.duplicated(key).any():
        duplicates = result.loc[result.duplicated(key, keep=False), key]
        raise ValueError(
            f"{label} contains duplicate metric keys: "
            + duplicates.head().to_dict(orient="records").__repr__()
        )
    unknown_models = sorted(set(result["model"]) - set(CANONICAL_MODELS))
    unknown_metrics = sorted(set(result["metric"]) - set(CANONICAL_METRICS))
    if unknown_models or unknown_metrics:
        raise ValueError(
            f"{label} contains unsupported models/metrics: "
            f"models={unknown_models}, metrics={unknown_metrics}"
        )
    return result.sort_values(key, kind="mergesort").reset_index(drop=True)


def load_published_results(path: Path, condition: str) -> pd.DataFrame:
    """Load one immutable publication package into strict long-form rows."""
    path = Path(path)
    source = path / "data" / "all_runs_tidy.csv" if path.is_dir() else path
    if not source.is_file():
        raise FileNotFoundError(f"Published result table not found: {source}")
    table = pd.read_csv(source)
    required = {
        "stock",
        "seed",
        "method",
        "strategy",
        "mse",
        "mae",
        "direction_accuracy",
        "forecast_horizon",
    }
    missing = sorted(required - set(table))
    if missing:
        raise ValueError(f"Published result table is missing columns: {missing}")
    table["model"] = [
        _canonical_model(method, strategy)
        for method, strategy in zip(table["method"], table["strategy"])
    ]
    table = table[table["model"].notna()].copy()
    long = table.melt(
        id_vars=["stock", "seed", "model", "forecast_horizon"],
        value_vars=list(CANONICAL_METRICS),
        var_name="metric",
        value_name="value",
    )
    long.insert(0, "condition", condition)
    long["source_file"] = str(source)
    return _validate_canonical_results(long, f"published results {source}")


def load_raw_experiment_results(
    results_dir: Path,
    condition: str,
    stocks,
    seeds,
) -> pd.DataFrame:
    """Load completed runner outputs with exact stock/seed/strategy coverage."""
    results_dir = Path(results_dir)
    rows = []
    missing_runs = []
    for stock in stocks:
        for seed in seeds:
            for strategy in ("random", "local_long"):
                run_dir = results_dir / strategy / stock / f"seed_{int(seed)}"
                manifest_path = run_dir / "run_manifest.json"
                metadata_path = run_dir / "preprocessing_config.json"
                if not manifest_path.is_file() or not metadata_path.is_file():
                    missing_runs.append(str(run_dir))
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                comparison_names = [
                    name
                    for name in manifest.get("comparison_files", [])
                    if str(name).endswith(".csv")
                ]
                if manifest.get("status") != "complete" or len(comparison_names) != 1:
                    missing_runs.append(str(run_dir))
                    continue
                comparison_path = run_dir / comparison_names[0]
                if not comparison_path.is_file():
                    missing_runs.append(str(comparison_path))
                    continue
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                horizon = int(metadata["forecast_horizon"])
                table = pd.read_csv(comparison_path)
                direction_column = (
                    "direction_accuracy"
                    if "direction_accuracy" in table
                    else "trend_accuracy"
                )
                required = {"model", "mse", "mae", direction_column}
                missing = sorted(required - set(table))
                if missing:
                    raise ValueError(
                        f"Raw result table {comparison_path} is missing {missing}"
                    )
                for record in table.to_dict(orient="records"):
                    model = _canonical_model(record["model"], strategy)
                    if model is None:
                        continue
                    for metric, column in (
                        ("mse", "mse"),
                        ("mae", "mae"),
                        ("direction_accuracy", direction_column),
                    ):
                        rows.append(
                            {
                                "condition": condition,
                                "stock": stock,
                                "seed": int(seed),
                                "model": model,
                                "metric": metric,
                                "value": record[column],
                                "forecast_horizon": horizon,
                                "source_file": str(comparison_path),
                            }
                        )
    if missing_runs:
        raise FileNotFoundError(
            "Missing completed experiment results:\n  "
            + "\n  ".join(sorted(missing_runs))
        )
    result = _validate_canonical_results(
        pd.DataFrame(rows, columns=CANONICAL_COLUMNS),
        f"raw results {results_dir}",
    )
    expected = len(stocks) * len(seeds) * len(CANONICAL_MODELS) * len(
        CANONICAL_METRICS
    )
    if len(result) != expected:
        raise ValueError(
            f"Raw results have wrong coverage: expected {expected}, got {len(result)}"
        )
    return result


def pair_condition_results(
    control: pd.DataFrame,
    intervention: pd.DataFrame,
    hypothesis: str,
) -> pd.DataFrame:
    """Outer-pair deterministic stock/seed/model/metric results."""
    control = _validate_canonical_results(control, "control")
    intervention = _validate_canonical_results(intervention, "intervention")
    expected_horizon = {"H1": 1, "H2": 5, "H3": 5}.get(hypothesis)
    control_horizons = set(control["forecast_horizon"])
    intervention_horizons = set(intervention["forecast_horizon"])
    if (
        len(control_horizons) != 1
        or len(intervention_horizons) != 1
        or control_horizons != intervention_horizons
        or (
            expected_horizon is not None
            and control_horizons != {expected_horizon}
        )
    ):
        raise ValueError(
            "Paired conditions have an invalid forecast horizon: "
            f"control={sorted(control_horizons)}, "
            f"intervention={sorted(intervention_horizons)}, "
            f"expected={expected_horizon}"
        )

    key = ["stock", "seed", "model", "metric"]
    paired = control.merge(
        intervention,
        on=key,
        how="outer",
        suffixes=("_control", "_intervention"),
        indicator=True,
        validate="one_to_one",
    )
    if not (paired["_merge"] == "both").all():
        missing = paired.loc[paired["_merge"] != "both", [*key, "_merge"]]
        raise ValueError(
            "Conditions contain missing paired results: "
            + missing.head().to_dict(orient="records").__repr__()
        )
    if (paired["value_control"] == 0).any():
        raise ValueError("Cannot compute percent delta from a zero control value")
    paired.insert(0, "hypothesis", hypothesis)
    paired["control"] = paired["value_control"]
    paired["intervention"] = paired["value_intervention"]
    paired["delta"] = paired["intervention"] - paired["control"]
    paired["percent_delta"] = 100.0 * paired["delta"] / paired["control"].abs()
    paired["improved"] = np.where(
        paired["metric"].isin(("mse", "mae")),
        paired["delta"] < 0,
        paired["delta"] > 0,
    )
    keep = [
        "hypothesis",
        *key,
        "control",
        "intervention",
        "delta",
        "percent_delta",
        "improved",
        "forecast_horizon_control",
        "source_file_control",
        "source_file_intervention",
    ]
    return paired[keep].sort_values(key, kind="mergesort").reset_index(drop=True)


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
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


def _student_t_critical_975(degrees_freedom: int) -> float:
    lower, upper = 0.0, 64.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if _student_t_cdf(midpoint, degrees_freedom) < 0.975:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def holm_adjust(p_values) -> list[float]:
    """Return Holm family-wise adjusted p-values in original order."""
    values = [float(value) for value in p_values]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Holm adjustment requires finite p-values in [0, 1]")
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    adjusted = [0.0] * len(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_stock_statistics(
    pairs: pd.DataFrame,
    expected_stock_count: int = 10,
) -> pd.DataFrame:
    """Perform paired Student-t inference on stock-level mean seed deltas."""
    required = {"hypothesis", "stock", "seed", "model", "metric", "delta"}
    missing = sorted(required - set(pairs))
    if missing:
        raise ValueError(f"Paired table is missing columns: {missing}")
    if not np.isfinite(pd.to_numeric(pairs["delta"], errors="raise")).all():
        raise ValueError("Paired deltas must be finite")
    stock_means = (
        pairs.groupby(
            ["hypothesis", "model", "metric", "stock"],
            sort=True,
            as_index=False,
        )["delta"]
        .mean()
        .rename(columns={"delta": "stock_mean_delta"})
    )
    rows = []
    for identifiers, group in stock_means.groupby(
        ["hypothesis", "model", "metric"], sort=True
    ):
        values = [float(value) for value in group["stock_mean_delta"]]
        count = len(values)
        if count != int(expected_stock_count):
            raise ValueError(
                f"Stock-level inference requires n={expected_stock_count}, got {count} "
                f"for {identifiers}"
            )
        mean = sum(values) / count
        variance = sum((value - mean) ** 2 for value in values) / (count - 1)
        std = math.sqrt(max(0.0, variance))
        if std == 0.0:
            if mean == 0.0:
                t_stat, p_value, dz = 0.0, 1.0, 0.0
            else:
                t_stat = math.copysign(math.inf, mean)
                p_value = 0.0
                dz = math.copysign(math.inf, mean)
            half_width = 0.0
        else:
            standard_error = std / math.sqrt(count)
            t_stat = mean / standard_error
            p_value = 2.0 * (1.0 - _student_t_cdf(abs(t_stat), count - 1))
            dz = mean / std
            half_width = _student_t_critical_975(count - 1) * standard_error
        rows.append(
            {
                "hypothesis": identifiers[0],
                "model": identifiers[1],
                "metric": identifiers[2],
                "n_stocks": count,
                "mean_delta": mean,
                "std_delta": std,
                "t_stat": t_stat,
                "p_value": max(0.0, min(1.0, p_value)),
                "dz": dz,
                "ci_low": mean - half_width,
                "ci_high": mean + half_width,
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["hypothesis", "model", "metric"], kind="mergesort"
    ).reset_index(drop=True)
    result["p_holm"] = holm_adjust(result["p_value"].tolist())
    return result
