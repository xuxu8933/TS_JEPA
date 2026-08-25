"""Leakage-safe, reusable preprocessing for financial time series.

All rolling/lagged quantities in this module are causal.  Statistics that
need fitting deliberately live in ``data_class_roll_volume.py`` and are fit
only after the chronological split.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


RETURN_FEATURE_COLS = (
    "log_return_1",
    "log_return_5",
    "close_ma10_gap",
    "close_ma50_gap",
    "ma10_ma50_gap",
    "log_volume_change",
    "rolling_vol_10",
    "rolling_vol_20",
)

MARKET_FEATURE_COLS = (
    "market_log_return_1",
    "market_log_return_5",
)

FEATURE_TRANSFORMS = ("raw", "return")

DERIVED_SENTIMENT_FEATURE_SOURCES = {
    "has_news": "news_count",
    "sentiment_mean_z": "sentiment_mean",
}

SENTIMENT_NORMALIZATION_MODES = ("none", "train_zscore")


@dataclass
class PreparedFinancialFrame:
    frame: pd.DataFrame
    feature_cols: list[str]
    warmup_report: dict[str, int]
    market_data: str | None
    market_alignment_report: dict[str, int] | None


def fit_transform_sentiment_features(
    frame_splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    feature_cols: Sequence[str],
    mode: str,
    state: Mapping[str, Any] | None = None,
    eps: float = 1e-6,
) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], dict[str, Any] | None]:
    """Fit selected sentiment transforms on the chronological train split."""
    if mode not in SENTIMENT_NORMALIZATION_MODES:
        raise ValueError(
            f"Unknown sentiment_normalization={mode!r}; expected one of "
            f"{SENTIMENT_NORMALIZATION_MODES}"
        )
    z_features = [
        feature
        for feature in feature_cols
        if feature in DERIVED_SENTIMENT_FEATURE_SOURCES and feature.endswith("_z")
    ]
    if mode == "none":
        if z_features:
            raise ValueError(
                "Derived z-score sentiment features require "
                "sentiment_normalization='train_zscore'"
            )
        if state is not None:
            raise ValueError(
                "sentiment_normalization_stats must be omitted when mode='none'"
            )
        return tuple(frame.copy() for frame in frame_splits), None
    if not z_features:
        raise ValueError(
            "sentiment_normalization='train_zscore' requires a configured "
            "derived *_z sentiment feature"
        )

    train = frame_splits[0]
    if train.empty:
        raise ValueError("Cannot fit sentiment normalization on an empty train split")
    if state is None:
        features = {}
        for feature in z_features:
            source = DERIVED_SENTIMENT_FEATURE_SOURCES[feature]
            values = pd.to_numeric(train[feature], errors="raise").astype("float64")
            mean = float(values.mean())
            fitted_std = float(values.std(ddof=0))
            scale = fitted_std if fitted_std >= eps else 1.0
            features[feature] = {
                "source": source,
                "mean": mean,
                "std": scale,
                "eps": float(eps),
            }
        resolved_state = {
            "mode": "train_zscore",
            "fit_split": "train",
            "features": features,
        }
    else:
        resolved_state = dict(state)
        if resolved_state.get("mode") != "train_zscore":
            raise ValueError("Sentiment normalization state mode mismatch")
        if resolved_state.get("fit_split") != "train":
            raise ValueError("Sentiment normalization state must be fit on train")
        state_features = resolved_state.get("features", {})
        if set(state_features) != set(z_features):
            raise ValueError(
                "Sentiment normalization feature order/state mismatch: "
                f"state={sorted(state_features)}, loader={sorted(z_features)}"
            )
        for feature in z_features:
            expected_source = DERIVED_SENTIMENT_FEATURE_SOURCES[feature]
            if state_features[feature].get("source") != expected_source:
                raise ValueError(
                    f"Sentiment source mismatch for {feature!r}: expected "
                    f"{expected_source!r}"
                )

    transformed = []
    for frame in frame_splits:
        output = frame.copy()
        for feature in z_features:
            feature_state = resolved_state["features"][feature]
            output[feature] = (
                pd.to_numeric(output[feature], errors="raise").astype("float64")
                - float(feature_state["mean"])
            ) / float(feature_state["std"])
        transformed.append(output)
    return tuple(transformed), resolved_state


def _numeric_column(frame, column, source):
    if column not in frame.columns:
        raise ValueError(
            f"Missing required column {column!r} in {source}. "
            f"Available columns: {list(frame.columns)}"
        )
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        bad = int(values.isna().sum())
        raise ValueError(f"{source} contains {bad} non-numeric/NaN {column!r} values")
    return values.astype("float64")


def construct_return_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct the canonical return representation without future values."""
    result = frame.copy()
    close = _numeric_column(result, "Close", "stock data")
    volume = _numeric_column(result, "Volume", "stock data")
    if (close <= 0).any():
        raise ValueError("Close must be strictly positive for log-return features")
    if (volume < 0).any():
        raise ValueError("Volume must be non-negative for log1p volume changes")

    # Moving averages include time t and preceding observations only.
    result["MA10"] = close.rolling(window=10, min_periods=10).mean()
    result["MA50"] = close.rolling(window=50, min_periods=50).mean()
    log_close = np.log(close)
    log_volume = np.log1p(volume)
    result["log_return_1"] = log_close - log_close.shift(1)
    result["log_return_5"] = log_close - log_close.shift(5)
    result["close_ma10_gap"] = close / result["MA10"] - 1.0
    result["close_ma50_gap"] = close / result["MA50"] - 1.0
    result["ma10_ma50_gap"] = result["MA10"] / result["MA50"] - 1.0
    result["log_volume_change"] = log_volume - log_volume.shift(1)
    result["rolling_vol_10"] = result["log_return_1"].rolling(
        window=10, min_periods=10
    ).std()
    result["rolling_vol_20"] = result["log_return_1"].rolling(
        window=20, min_periods=20
    ).std()
    return result


def construct_raw_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve the historical raw representation and causal MA semantics."""
    result = frame.copy()
    close = _numeric_column(result, "Close", "stock data")
    result["Volume"] = _numeric_column(result, "Volume", "stock data")
    result["MA10"] = close.rolling(window=10, min_periods=10).mean()
    result["MA50"] = close.rolling(window=50, min_periods=50).mean()
    return result


def resolve_return_feature_cols(feature_cols, sentiment_cols, market_data=None):
    """Return canonical feature order, treating configured sentiment as optional."""
    requested = set(feature_cols or ())
    ordered_sentiment = [name for name in sentiment_cols if name in requested]
    resolved = list(RETURN_FEATURE_COLS) + ordered_sentiment
    if market_data is not None:
        resolved.extend(MARKET_FEATURE_COLS)
    return resolved


def _resolve_market_path(market_data, stock_path):
    if market_data is None:
        return None
    requested = os.fspath(market_data)
    candidates = [requested]
    if not requested.lower().endswith(".csv"):
        candidates.extend(
            [
                os.path.join("data", requested, f"{requested}.csv"),
                os.path.join(
                    os.path.dirname(os.path.dirname(stock_path)),
                    requested,
                    f"{requested}.csv",
                ),
            ]
        )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not resolve market_data={market_data!r}; checked {candidates}"
    )


def align_market_data(
    stock_frame: pd.DataFrame,
    market_data,
    stock_path,
    timestamp_col="Date",
):
    """Inner-align precomputed causal market returns to stock trading dates."""
    market_path = _resolve_market_path(market_data, stock_path)
    market = pd.read_csv(market_path, parse_dates=[timestamp_col], low_memory=False)
    market = market.sort_values(timestamp_col).drop_duplicates(
        timestamp_col, keep="last"
    )
    market_close = _numeric_column(market, "Close", f"market data {market_path}")
    if (market_close <= 0).any():
        raise ValueError("Market Close must be strictly positive")
    market_log_close = np.log(market_close)
    market["market_log_return_1"] = market_log_close - market_log_close.shift(1)
    market["market_log_return_5"] = market_log_close - market_log_close.shift(5)
    market["_market_close"] = market_close
    market = market[
        [timestamp_col, "_market_close", *MARKET_FEATURE_COLS]
    ]

    before = len(stock_frame)
    stock_dates = set(stock_frame[timestamp_col])
    market_dates = set(market[timestamp_col])
    aligned = stock_frame.merge(
        market,
        on=timestamp_col,
        how="inner",
        sort=True,
        validate="one_to_one",
    )
    report = {
        "stock_rows_before_alignment": int(before),
        "market_rows_before_alignment": int(len(market)),
        "stock_dates_without_market": int(len(stock_dates - market_dates)),
        "market_dates_without_stock": int(len(market_dates - stock_dates)),
        "aligned_rows": int(len(aligned)),
        "stock_rows_dropped": int(before - len(aligned)),
    }
    print(
        "[market_alignment] "
        + ", ".join(f"{key}={value}" for key, value in report.items())
    )
    return aligned, report, market_path


def _warmup_report(frame, required_cols, feature_transform):
    if feature_transform == "return":
        ma50 = frame["MA50"].isna()
        return5 = frame["log_return_5"].isna()
        vol20 = frame["rolling_vol_20"].isna()
    else:
        ma50 = frame["MA50"].isna()
        return5 = pd.Series(False, index=frame.index)
        vol20 = pd.Series(False, index=frame.index)

    required_invalid = ~np.isfinite(frame[required_cols].astype("float64")).all(axis=1)
    named_invalid = ma50 | return5 | vol20
    other = required_invalid & ~named_invalid
    report = {
        "ma50_warmup": int(ma50.sum()),
        "return_5_warmup": int(return5.sum()),
        "volatility_20_warmup": int(vol20.sum()),
        "other_required_transformations": int(other.sum()),
        "total_rows_removed": int(required_invalid.sum()),
    }
    print(
        "[financial_features] removed rows: "
        + ", ".join(f"{key}={value}" for key, value in report.items())
    )
    return report, required_invalid


def prepare_financial_frame(
    path_data,
    feature_cols,
    sentiment_cols,
    merge_sentiment,
    timestamp_col="Date",
    feature_transform="raw",
    market_data=None,
    data_end_date=None,
    log_volume=True,
):
    """Read, causally transform, align, and warm-up-filter one full series."""
    if feature_transform not in FEATURE_TRANSFORMS:
        raise ValueError(
            f"Unknown feature_transform={feature_transform!r}; "
            f"expected one of {FEATURE_TRANSFORMS}"
        )
    frame = pd.read_csv(
        path_data, parse_dates=[timestamp_col], low_memory=False, sep=","
    )
    frame = frame.sort_values(timestamp_col).drop_duplicates(
        timestamp_col, keep="last"
    ).reset_index(drop=True)
    if data_end_date is not None:
        frame = frame[frame[timestamp_col] <= pd.Timestamp(data_end_date)].copy()
    if frame.empty:
        raise ValueError(f"No rows available in {path_data!r} after date filtering")

    frame = merge_sentiment(frame)
    if feature_transform == "return":
        frame = construct_return_features(frame)
        resolved_features = resolve_return_feature_cols(
            feature_cols, sentiment_cols, market_data=market_data
        )
    else:
        frame = construct_raw_features(frame)
        resolved_features = list(feature_cols)

    alignment_report = None
    resolved_market = None
    if market_data is not None:
        frame, alignment_report, resolved_market = align_market_data(
            frame,
            market_data=market_data,
            stock_path=path_data,
            timestamp_col=timestamp_col,
        )

    missing = [name for name in resolved_features if name not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing requested features after {feature_transform!r} transform: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )

    # Raw mode historically discarded the MA50 warm-up even when MA50 was not
    # selected. Keep that behavior for checkpoint/data compatibility.
    required = list(dict.fromkeys([*resolved_features, "Close", "MA50"]))
    if market_data is not None:
        required.append("_market_close")
    report, invalid = _warmup_report(frame, required, feature_transform)
    frame = frame.loc[~invalid].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No rows remain after required feature warm-up filtering")

    if feature_transform == "raw" and log_volume and "Volume" in resolved_features:
        volume = _numeric_column(frame, "Volume", "stock data")
        if (volume < 0).any():
            raise ValueError("Volume must be non-negative for log1p transformation")
        frame["Volume"] = np.log1p(volume)

    return PreparedFinancialFrame(
        frame=frame,
        feature_cols=resolved_features,
        warmup_report=report,
        market_data=None if market_data is None else str(market_data),
        market_alignment_report=alignment_report,
    )


def cumulative_log_return_target(future_prices, origin_price):
    """Return log(P[t+h] / P[t]) for h=1..H."""
    if np.any(np.asarray(future_prices) <= 0) or float(origin_price) <= 0:
        raise ValueError("Prices must be strictly positive for log-return targets")
    return np.log(np.asarray(future_prices) / float(origin_price))
