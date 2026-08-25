"""Configuration and result analysis for the controlled sentiment ablations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

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
