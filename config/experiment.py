"""Central experiment defaults, feature selection, and config validation.

The training code still consumes flat dictionaries for checkpoint and CLI
compatibility.  Those dictionaries are generated from the sectioned defaults
in this module so each user-facing default has one authoritative definition.
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Mapping, Sequence


DEFAULT_MARKET_FEATURES = ("Close", "Volume", "MA10", "MA50")
DEFAULT_SENTIMENT_FEATURES = ("sentiment_mean",)
KNOWN_SENTIMENT_FEATURES = (
    "sentiment_mean",
    "sentiment_sum",
    "sentiment_max",
    "sentiment_min",
    "sentiment_std",
    "news_count",
    "has_news",
    "sentiment_mean_z",
)


COMMON_DATA_DEFAULTS = {
    "data": "NVDA",
    "timestamp_col": "Date",
    "patch_size": 5,
    "market_features": list(DEFAULT_MARKET_FEATURES),
    "sentiment_features": list(DEFAULT_SENTIMENT_FEATURES),
    "use_sentiment": True,
    "sentiment_path": "./NVDA_daily_sentiment.csv",
    "train_end_date": "2024-12-31",
    "test_start_date": "2025-01-01",
    "data_end_date": "2026-01-01",
    "validation_fraction": 0.05,
    "target_feature_index": 0,
    "target_col": "Close",
}

COMMON_PREPROCESSING_DEFAULTS = {
    "feature_transform": "raw",
    "robust_zscore_clip": None,
    "market_data": None,
    "sentiment_normalization": "none",
    "sentiment_normalization_stats": None,
}

COMMON_DECODER_DEFAULTS = {
    "decoder_type": "residual_mlp",
    "decoder_hidden_dim": 128,
    "decoder_num_layers": 2,
    "decoder_dropout": 0.1,
}


PRETRAIN_CONFIG_SECTIONS = {
    "data": {
        **COMMON_DATA_DEFAULTS,
        "input_mode": "timeseries",
        "series_split_size": 20,
        "pretrain_stride": 5,
        "sampling_mode": "sliding_window",
        "mnist_root": "./data/MNIST",
        "mnist_train_samples": 512,
        "mnist_val_samples": 128,
        "download_mnist": False,
    },
    "preprocessing": {
        **COMMON_PREPROCESSING_DEFAULTS,
        "normalization": "train_zscore",
    },
    "masking": {
        "mask_strategy": "random",
        "mask_ratio": 0.7,
        # Kept only because it is part of legacy checkpoint names.
        "ratio_patches": 10,
        "mae_window_patches": 1,
        "jepa_gap_patches": 4,
        "jepa_target_patches": 4,
        "anchor_strategy": "random",
        "fixed_anchor": 0,
        "future_target_patches": 4,
        "causal_num_blocks": 2,
        "causal_block_patches": 2,
        "causal_block_gap_patches": 1,
    },
    "model": {
        "encoder_embed_dim": 256,
        "encoder_nhead": 2,
        "encoder_num_layers": 1,
        "encoder_kernel_size": 3,
        "encoder_embed_bias": True,
        "predictor_embed": 128,
        "predictor_nhead": 2,
        "predictor_num_layers": 1,
        **COMMON_DECODER_DEFAULTS,
    },
    "pretraining": {
        "batch_size": 32,
        "checkpoint_save": 500,
        "checkpoint_print": 30,
        "validation_interval": 10,
        "validation_max_batches": None,
        "clip_grad": 1,
        "ipe_scale": 1.25,
        "lambda_jepa": 1.0,
        "lambda_mae": 0.5,
        "jepa_loss": "rmse",
        "mae_loss": "rmse",
        "lr": 1e-5,
        "end_lr": 1e-6,
        "num_epochs": 2001,
        "ema_momentum": 0.998,
    },
    "evaluation": {
        "run_eval": True,
        "eval_use_best": True,
        "eval_checkpoint_to_use": None,
        "eval_encoder_weights": "ema",
        "eval_forecast_target": "relative_return",
        "eval_forecast_horizon": None,
        "eval_num_epochs": 501,
        "eval_results_dir": "./results/NVDA/relative_return/seed_42",
    },
    "runtime": {
        "seed": 42,
        "deterministic": True,
        "resume_from": None,
        "max_batches_per_epoch": None,
        "save_final": True,
        "path_suffix": None,
        "compatible_save_name": False,
        "notes": "",
    },
}


DOWNSTREAM_CONFIG_SECTIONS = {
    "data": {
        **COMMON_DATA_DEFAULTS,
        "sampling_mode": "sliding_window",
        "context_size": 12,
        "eval_stride": 5,
    },
    "preprocessing": {
        **COMMON_PREPROCESSING_DEFAULTS,
        "normalization": "window_return",
        "normalization_stats": None,
    },
    "checkpoint": {
        "checkpoint_selection": "last",
        "pretrain_checkpoint_path": None,
        "pretrain_encoder_weights": "ema",
        "checkpoint_to_use": 2000,
        "lr_pretrain": 1e-5,
        "ema_pretrain": 0.998,
        # Legacy checkpoint identity fields.
        "ratio_patches": 10,
        "mask_ratio": 0.7,
        "mask_strategy": "random",
        "lambda_jepa": 1.0,
        "lambda_mae": 1.0,
        "mae_window_patches": 1,
        "jepa_gap_patches": 4,
        "jepa_target_patches": 4,
        "future_target_patches": 4,
        "causal_num_blocks": 2,
        "causal_block_patches": 2,
        "causal_block_gap_patches": 1,
    },
    "model": {
        "cnn_out_channels": [32, 64, 128],
        "cnn_kernel_size": 3,
        "cnn_dense_dim": 32,
        "embed_dim": 128,
        "nhead": 2,
        "num_layers": 1,
        "kernel_size": 3,
        "transformer_dense_dim": 64,
        "pooling": "Mean",
        "pretrain_encoder_embed_dim": 256,
        "pretrain_encoder_nhead": 2,
        "pretrain_encoder_num_layers": 1,
        "pretrain_encoder_kernel_size": 3,
        "pretrain_encoder_embed_bias": True,
        "pretrain_transformer_dense_dim": 128,
        "pretrain_decoder_embed_dim": 128,
        "pretrain_decoder_nhead": 2,
        "pretrain_decoder_num_layers": 1,
        **COMMON_DECODER_DEFAULTS,
    },
    "downstream": {
        "num_epochs": 501,
        "batch_size": 32,
        "lr": 1e-3,
        "ratio_supervision": 1.0,
        "fine_tune_encoder": True,
        "encoder_finetune_lr": 1e-5,
        "trend_weight": 0.001,
        "trend_loss_temperature": 0.01,
        "trend_loss_threshold": 1e-5,
        "trend_selection_weight": 0.0005,
    },
    "evaluation": {
        "forecast_target": "value",
        "forecast_horizon": None,
        "eval_type": "last",
    },
    "runtime": {
        "seed": 42,
        "path_save": "./logs/output_model/",
        "results_dir": "./results",
    },
}


def none_if_requested(value: Any) -> Any:
    """Convert the CLI spellings for an absent optional value to ``None``."""
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in ("", "none", "null"):
        return None
    return value


def resolve_forecast_horizon(
    forecast_horizon: int | None,
    patch_size: int,
) -> int:
    """Resolve the downstream target width independently of input patches."""
    resolved = int(patch_size if forecast_horizon is None else forecast_horizon)
    if resolved <= 0:
        raise ValueError(f"forecast_horizon must be positive, got {resolved}")
    return resolved


def _deduplicate(names: Sequence[str], label: str) -> list[str]:
    result = list(dict.fromkeys(names))
    if not result:
        raise ValueError(f"{label} must contain at least one feature")
    return result


def effective_feature_columns(
    market_features: Sequence[str],
    sentiment_features: Sequence[str],
    use_sentiment: bool,
) -> list[str]:
    """Return the one effective feature order consumed by the data pipeline."""
    market = _deduplicate(market_features, "market_features")
    sentiment = _deduplicate(sentiment_features, "sentiment_features")
    overlap = sorted(set(market) & set(sentiment))
    if overlap:
        raise ValueError(
            "market_features and sentiment_features must be disjoint; "
            f"overlap={overlap}"
        )
    return market + sentiment if use_sentiment else market


def resolve_feature_selection(
    config: Mapping[str, Any],
    *,
    feature_cols: Sequence[str] | None = None,
    market_features: Sequence[str] | None = None,
    sentiment_features: Sequence[str] | None = None,
    use_sentiment: bool | None = None,
) -> dict[str, Any]:
    """Resolve modern feature sections and the legacy ``feature_cols`` override.

    ``feature_cols`` remains accepted for existing commands.  With no explicit
    sentiment switch it is used exactly and the toggle is inferred.  An
    explicit switch is authoritative: ``--no-sentiment`` removes all known
    sentiment columns, while ``--use-sentiment`` adds the configured sentiment
    columns.
    """
    configured_market = _deduplicate(
        market_features or config.get("market_features", DEFAULT_MARKET_FEATURES),
        "market_features",
    )
    configured_sentiment = _deduplicate(
        sentiment_features
        or config.get("sentiment_features", DEFAULT_SENTIMENT_FEATURES),
        "sentiment_features",
    )
    known_sentiment = set(KNOWN_SENTIMENT_FEATURES) | set(configured_sentiment)

    if feature_cols is None:
        resolved_use_sentiment = (
            bool(config.get("use_sentiment", True))
            if use_sentiment is None
            else bool(use_sentiment)
        )
        resolved_features = effective_feature_columns(
            configured_market,
            configured_sentiment,
            resolved_use_sentiment,
        )
    else:
        requested = _deduplicate(feature_cols, "feature_cols")
        if use_sentiment is None:
            resolved_use_sentiment = any(
                name in known_sentiment for name in requested
            )
            resolved_features = requested
        elif use_sentiment:
            resolved_use_sentiment = True
            resolved_features = list(
                dict.fromkeys([*requested, *configured_sentiment])
            )
        else:
            resolved_use_sentiment = False
            resolved_features = [
                name for name in requested if name not in known_sentiment
            ]
            if not resolved_features:
                raise ValueError(
                    "--no-sentiment removed every requested feature; configure "
                    "at least one market feature"
                )

    return {
        "market_features": configured_market,
        "sentiment_features": configured_sentiment,
        "use_sentiment": resolved_use_sentiment,
        "feature_cols": resolved_features,
    }


def flatten_config_sections(sections: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Flatten sectioned defaults, rejecting duplicate parameter definitions."""
    flattened: dict[str, Any] = {}
    owners: dict[str, str] = {}
    for section, values in sections.items():
        for key, value in values.items():
            if key in flattened:
                raise ValueError(
                    f"Configuration parameter {key!r} is defined in both "
                    f"{owners[key]!r} and {section!r}"
                )
            flattened[key] = copy.deepcopy(value)
            owners[key] = section
    flattened.update(resolve_feature_selection(flattened))
    return flattened


def build_pretrain_config() -> dict[str, Any]:
    return flatten_config_sections(PRETRAIN_CONFIG_SECTIONS)


def build_downstream_config() -> dict[str, Any]:
    return flatten_config_sections(DOWNSTREAM_CONFIG_SECTIONS)


def _parse_date(value: Any, name: str):
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO date, got {value!r}") from error


def validate_data_config(config: Mapping[str, Any], *, stage: str) -> None:
    """Validate shared data invariants after all CLI overrides are resolved."""
    patch_size = int(config["patch_size"])
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")

    feature_cols = list(config.get("feature_cols", ()))
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one feature")
    sentiment_features = set(config.get("sentiment_features", ()))
    selected_sentiment = sentiment_features.intersection(feature_cols)
    if bool(config.get("use_sentiment")) and not selected_sentiment:
        raise ValueError(
            "use_sentiment=True requires at least one configured sentiment feature "
            "in feature_cols"
        )
    if not bool(config.get("use_sentiment")) and selected_sentiment:
        raise ValueError(
            "use_sentiment=False is inconsistent with sentiment columns in "
            f"feature_cols: {sorted(selected_sentiment)}"
        )
    sentiment_normalization = config.get("sentiment_normalization", "none")
    if sentiment_normalization not in ("none", "train_zscore"):
        raise ValueError(
            "sentiment_normalization must be 'none' or 'train_zscore', got "
            f"{sentiment_normalization!r}"
        )
    has_zscore_feature = "sentiment_mean_z" in feature_cols
    if sentiment_normalization == "train_zscore" and not has_zscore_feature:
        raise ValueError(
            "sentiment_normalization='train_zscore' requires sentiment_mean_z"
        )
    if sentiment_normalization == "none" and has_zscore_feature:
        raise ValueError(
            "sentiment_mean_z requires sentiment_normalization='train_zscore'"
        )

    validation_fraction = float(config["validation_fraction"])
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be in [0, 1), got "
            f"{validation_fraction}"
        )

    if config.get("input_mode", "timeseries") == "timeseries":
        train_end = _parse_date(config.get("train_end_date"), "train_end_date")
        test_start = _parse_date(config.get("test_start_date"), "test_start_date")
        data_end = _parse_date(config.get("data_end_date"), "data_end_date")
        if test_start is None:
            raise ValueError("test_start_date must be defined for timeseries input")
        if train_end is not None and train_end >= test_start:
            raise ValueError(
                "train_end_date must be earlier than test_start_date: "
                f"train_end_date={train_end}, test_start_date={test_start}"
            )
        if data_end is not None and data_end < test_start:
            raise ValueError(
                "data_end_date must not be earlier than test_start_date: "
                f"data_end_date={data_end}, test_start_date={test_start}"
            )

    if stage == "pretrain" and config.get("input_mode") == "timeseries":
        context_length = int(config["series_split_size"])
        if context_length <= 0:
            raise ValueError(
                f"series_split_size must be positive, got {context_length}"
            )
        if context_length % patch_size != 0:
            raise ValueError(
                "series_split_size must be divisible by patch_size: "
                f"series_split_size={context_length}, patch_size={patch_size}"
            )
    elif stage == "downstream":
        resolve_forecast_horizon(config.get("forecast_horizon"), patch_size)
        context_size = int(config["context_size"])
        if context_size <= 0:
            raise ValueError(f"context_size must be positive, got {context_size}")
        eval_stride = int(config["eval_stride"])
        if eval_stride <= 0:
            raise ValueError(f"eval_stride must be positive, got {eval_stride}")
    else:
        if stage not in ("pretrain", "downstream"):
            raise ValueError(f"Unknown configuration stage: {stage!r}")

    clip = config.get("robust_zscore_clip")
    if clip is not None and float(clip) <= 0:
        raise ValueError("robust_zscore_clip must be positive when supplied")
