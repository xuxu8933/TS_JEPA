import os
import pandas as pd
import torch
import random
from torch.utils.data import Dataset

from config.experiment import resolve_forecast_horizon
from .financial_preprocessing import (
    FEATURE_TRANSFORMS,
    prepare_financial_frame,
)


DEFAULT_SENTIMENT_COLS = (
    "sentiment_mean",
    "sentiment_sum",
    "sentiment_max",
    "sentiment_min",
    "sentiment_std",
    "news_count",
)

NORMALIZATION_MODES = (
    "window_return",
    "train_zscore",
    "train_robust_zscore",
    "none",
)
SAMPLING_MODES = ("sliding_window", "temporal_segments")
FORECAST_TARGETS = (
    "value",
    "relative_return",
    "cumulative_log_return",
    "excess_log_return",
)


def cumulative_return_normalize_with_base(series, base, eps=1e-8, passthrough_indices=None):
    """
    Works for both:
        series: [T]
        series: [T, C]

    For multi-feature data, base is the first observable row [C].
    """
    base = torch.where(torch.abs(base) < eps, base + eps, base)
    normalized = series / base - 1.0

    if passthrough_indices:
        normalized = normalized.clone()
        normalized[..., passthrough_indices] = series[..., passthrough_indices]

    return normalized


def _resolve_normalization_mode(normalization, normalize=True):
    if normalization is None:
        normalization = "window_return" if normalize else "none"
    if normalization not in NORMALIZATION_MODES:
        raise ValueError(
            f"Unknown normalization={normalization!r}; expected one of "
            f"{NORMALIZATION_MODES}"
        )
    return normalization


def _resolve_sampling_mode(sampling_mode):
    if sampling_mode not in SAMPLING_MODES:
        raise ValueError(
            f"Unknown sampling_mode={sampling_mode!r}; expected one of "
            f"{SAMPLING_MODES}"
        )
    return sampling_mode


def _resolve_forecast_target(forecast_target):
    if forecast_target not in FORECAST_TARGETS:
        raise ValueError(
            f"Unknown forecast_target={forecast_target!r}; expected one of "
            f"{FORECAST_TARGETS}"
        )
    return forecast_target


def _sample_starts(series_length, sample_length, stride, sampling_mode):
    """Return chronological starts for sliding or non-overlapping samples."""
    if sample_length <= 0:
        raise ValueError(f"sample_length must be positive, got {sample_length}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    effective_stride = sample_length if sampling_mode == "temporal_segments" else stride
    return (
        list(range(0, series_length - sample_length + 1, effective_stride)),
        effective_stride,
    )


def train_zscore_state(train_series, feature_cols, eps=1e-6):
    """Fit reusable feature statistics from the chronological train split only."""
    if len(train_series) == 0:
        raise ValueError("Cannot fit train_zscore normalization on an empty train split")
    mean = train_series.mean(dim=0)
    std = train_series.std(dim=0, unbiased=False).clamp_min(eps)
    return {
        "mode": "train_zscore",
        "feature_cols": list(feature_cols),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def train_robust_zscore_state(train_series, feature_cols, eps=1e-6, clip=None):
    """Fit median/MAD statistics on the chronological training split only."""
    if len(train_series) == 0:
        raise ValueError(
            "Cannot fit train_robust_zscore normalization on an empty train split"
        )
    median = torch.quantile(train_series, 0.5, dim=0)
    mad = torch.quantile(torch.abs(train_series - median), 0.5, dim=0)
    raw_scale = 1.4826 * mad
    scale = torch.where(raw_scale > eps, raw_scale, torch.ones_like(raw_scale))
    return {
        "mode": "train_robust_zscore",
        "feature_cols": list(feature_cols),
        "median": median.tolist(),
        "mad": mad.tolist(),
        "scale": scale.tolist(),
        "clip": None if clip is None else float(clip),
        "eps": float(eps),
    }


def normalization_tensors(state, feature_cols):
    if state is None:
        return None, None
    if state.get("mode") not in ("train_zscore", "train_robust_zscore"):
        raise ValueError(
            "Fitted normalization requires train_zscore or train_robust_zscore "
            "state, "
            f"got {state.get('mode')!r}"
        )
    if list(state.get("feature_cols", [])) != list(feature_cols):
        raise ValueError(
            "Normalization feature order does not match loader feature order: "
            f"state={state.get('feature_cols')}, loader={list(feature_cols)}"
        )
    if state["mode"] == "train_zscore":
        center = torch.tensor(state["mean"], dtype=torch.float32)
        scale = torch.tensor(state["std"], dtype=torch.float32)
    else:
        center = torch.tensor(state["median"], dtype=torch.float32)
        scale = torch.tensor(state["scale"], dtype=torch.float32)
    return center, scale


def _as_list(x):
    if isinstance(x, str):
        return [x]
    return list(x)


def _chronological_split_frames(
    df,
    timestamp_col="Date",
    validation_fraction=0.05,
    train_end_date=None,
    test_start_date=None,
    data_end_date=None,
):
    """Split an already-prepared frame without fitting or recomputing features."""
    df = df.copy().sort_values(by=[timestamp_col]).reset_index(drop=True)
    if data_end_date is not None:
        df = df[df[timestamp_col] <= pd.Timestamp(data_end_date)].copy()
        if df.empty:
            raise ValueError(
                f"No rows found on or before data_end_date={data_end_date!r}."
            )
    if test_start_date is None:
        raise ValueError("test_start_date must be defined for chronological splitting.")
    if not 0 <= float(validation_fraction) < 1:
        raise ValueError("validation_fraction must be in [0, 1)")

    train_end_ts = pd.Timestamp(train_end_date) if train_end_date is not None else None
    test_start_ts = pd.Timestamp(test_start_date)
    if train_end_ts is not None and train_end_ts >= test_start_ts:
        raise ValueError(
            "train_end_date must be earlier than test_start_date: "
            f"train_end_date={train_end_date!r}, test_start_date={test_start_date!r}"
        )
    if train_end_ts is None:
        train_val = df[df[timestamp_col] < test_start_ts].copy()
    else:
        train_val = df[df[timestamp_col] <= train_end_ts].copy()
    test = df[df[timestamp_col] >= test_start_ts].copy()
    if train_val.empty:
        raise ValueError(
            f"No train/validation rows found for train_end_date={train_end_date!r} "
            f"and test_start_date={test_start_date!r}."
        )
    if test.empty:
        raise ValueError(f"No test rows found for test_start_date={test_start_date!r}.")

    val_len = int(len(train_val) * validation_fraction)
    train_len = len(train_val) - val_len
    if train_len <= 0:
        raise ValueError(
            f"Date split left no train rows. train_val_len={len(train_val)}, "
            f"validation_fraction={validation_fraction}"
        )
    train = train_val.iloc[:train_len].copy()
    val = train_val.iloc[train_len:].copy()
    boundary = (
        f"<= {train_end_date}" if train_end_date is not None else f"< {test_start_date}"
    )
    print(
        "[chronological_split] date split: "
        f"train_val {boundary}, test>= {test_start_date}, data<= {data_end_date}, "
        f"train_len={len(train)}, val_len={len(val)}, test_len={len(test)}"
    )
    return train, val, test


def chronological_split(
    df,
    feature_cols=("Close", "Volume"),
    timestamp_col="Date",
    validation_fraction=0.05,
    train_end_date=None,
    test_start_date=None,
    data_end_date=None,
):
    """
    Date-based chronological split.

    test_start_date is required. Rows on or after it form the held-out test
    period. Earlier rows (or rows through train_end_date, when provided) are
    split into train and validation periods using validation_fraction.
    """
    feature_cols = _as_list(feature_cols)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Available columns: {list(df.columns)}")
    train, val, test = _chronological_split_frames(
        df,
        timestamp_col=timestamp_col,
        validation_fraction=validation_fraction,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        data_end_date=data_end_date,
    )
    return tuple(
        torch.tensor(part[feature_cols].values, dtype=torch.float32)
        for part in (train, val, test)
    )


def _infer_sentiment_path(path_data):
    data_dir = os.path.dirname(path_data)
    ticker = os.path.splitext(os.path.basename(path_data))[0]

    candidates = [
        os.path.join(data_dir, f"{ticker}_daily_sentiment.csv"),
        os.path.join(os.getcwd(), f"{ticker}_daily_sentiment.csv"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def _merge_daily_sentiment(
    df,
    path_data,
    timestamp_col,
    feature_cols,
    sentiment_path=None,
    sentiment_cols=DEFAULT_SENTIMENT_COLS,
):
    feature_cols = _as_list(feature_cols)
    sentiment_cols = _as_list(sentiment_cols)
    requested_sentiment_cols = [c for c in feature_cols if c in sentiment_cols]
    existing_sentiment_cols = [c for c in requested_sentiment_cols if c in df.columns]
    missing_requested_sentiment_cols = [
        c for c in requested_sentiment_cols
        if c not in df.columns
    ]

    if existing_sentiment_cols:
        df = df.copy()
        for col in existing_sentiment_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if not missing_requested_sentiment_cols:
        return df

    sentiment_path = sentiment_path or _infer_sentiment_path(path_data)

    if sentiment_path is None or not os.path.exists(sentiment_path):
        raise FileNotFoundError(
            "Sentiment features were requested, but no daily sentiment CSV was found. "
            f"Looked for a ticker-level file next to {path_data!r} and in the project root. "
            "Pass sentiment_path explicitly or remove sentiment columns from feature_cols."
        )

    sentiment_df = pd.read_csv(sentiment_path, parse_dates=["date"], low_memory=False)
    missing = [
        c for c in missing_requested_sentiment_cols
        if c not in sentiment_df.columns
    ]
    if missing:
        raise ValueError(
            f"Missing sentiment columns in {sentiment_path}: {missing}. "
            f"Available columns: {list(sentiment_df.columns)}"
        )

    keep_cols = ["date"] + [c for c in sentiment_cols if c in sentiment_df.columns]
    sentiment_df = sentiment_df[keep_cols].copy()
    sentiment_df["date"] = sentiment_df["date"].dt.normalize()

    df = df.copy()
    df["_sentiment_date"] = df[timestamp_col].dt.normalize()
    df = df.merge(
        sentiment_df,
        how="left",
        left_on="_sentiment_date",
        right_on="date",
    )
    df.drop(columns=["_sentiment_date", "date"], inplace=True)

    for col in sentiment_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    print(
        f"[load_price_series] merged sentiment from {sentiment_path}; "
        f"requested={missing_requested_sentiment_cols}"
    )

    return df


def load_price_series(
    path_data,
    feature_cols=("Close", "Volume"),
    timestamp_col="Date",
    validation_fraction=0.05,
    log_volume=True,
    sentiment_path=None,
    sentiment_cols=DEFAULT_SENTIMENT_COLS,
    train_end_date=None,
    test_start_date=None,
    data_end_date=None,
    feature_transform="raw",
    market_data=None,
    return_metadata=False,
):
    """
    Load CSV and return chronological train / val / test tensors.

    feature_cols can be:
        ("Close", "Volume")
        ("Open", "High", "Low", "Close", "Volume")
    """
    feature_cols = _as_list(feature_cols)
    prepared = prepare_financial_frame(
        path_data=path_data,
        feature_cols=feature_cols,
        sentiment_cols=sentiment_cols,
        merge_sentiment=lambda frame: _merge_daily_sentiment(
            df=frame,
            path_data=path_data,
            timestamp_col=timestamp_col,
            feature_cols=feature_cols,
            sentiment_path=sentiment_path,
            sentiment_cols=sentiment_cols,
        ),
        timestamp_col=timestamp_col,
        feature_transform=feature_transform,
        market_data=market_data,
        data_end_date=data_end_date,
        log_volume=log_volume,
    )
    frame_splits = _chronological_split_frames(
        prepared.frame,
        timestamp_col=timestamp_col,
        validation_fraction=validation_fraction,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        data_end_date=data_end_date,
    )
    feature_splits = tuple(
        torch.tensor(
            split_frame[prepared.feature_cols].values,
            dtype=torch.float32,
        )
        for split_frame in frame_splits
    )
    if not return_metadata:
        return feature_splits

    close_splits = tuple(
        torch.tensor(split_frame["Close"].values, dtype=torch.float32)
        for split_frame in frame_splits
    )
    market_close_splits = (
        tuple(
            torch.tensor(split_frame["_market_close"].values, dtype=torch.float32)
            for split_frame in frame_splits
        )
        if market_data is not None
        else (None, None, None)
    )
    date_splits = tuple(
        list(pd.to_datetime(split_frame[timestamp_col]))
        for split_frame in frame_splits
    )
    return {
        "features": feature_splits,
        "close": close_splits,
        "market_close": market_close_splits,
        "dates": date_splits,
        "feature_cols": list(prepared.feature_cols),
        "feature_transform": feature_transform,
        "warmup_report": dict(prepared.warmup_report),
        "market_data": prepared.market_data,
        "market_alignment_report": prepared.market_alignment_report,
    }


class CSVDataLoader(Dataset):
    """
    Loader for TS-JEPA pretraining.

    Output shape per sample:
        patches_tensor: [num_patches, patch_size * feature_dim]

    Example:
        feature_cols=("Close", "Volume"), patch_size=5
        patches_tensor.shape = [num_patches, 10]
    """

    def __init__(
        self,
        path_data,
        series_split_size=120,
        patch_size=5,
        mask_ratio=0.15,
        stride=None,
        sampling_mode="sliding_window",
        normalize=True,
        normalization=None,
        normalization_stats=None,
        split="train",
        mask_seed=None,
        feature_cols=("Close", "Volume"),
        timestamp_col="Date",
        validation_fraction=0.05,
        log_volume=True,
        sentiment_path=None,
        sentiment_cols=DEFAULT_SENTIMENT_COLS,
        feature_transform="raw",
        market_data=None,
        robust_zscore_clip=None,
        train_end_date=None,
        test_start_date=None,
        data_end_date=None,
    ):
        self.series_split_size = series_split_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.sampling_mode = _resolve_sampling_mode(sampling_mode)
        self.normalization = _resolve_normalization_mode(normalization, normalize)
        self.normalize = self.normalization != "none"
        self.split = split
        self.mask_seed = mask_seed
        self.feature_transform = feature_transform
        if feature_transform not in FEATURE_TRANSFORMS:
            raise ValueError(
                f"Unknown feature_transform={feature_transform!r}; "
                f"expected one of {FEATURE_TRANSFORMS}"
            )
        self.market_data = market_data
        if feature_transform == "return" and self.normalization == "window_return":
            raise ValueError(
                "window_return is not defined for already-return-based features; "
                "use train_zscore, train_robust_zscore, or none"
            )
        self.robust_zscore_clip = robust_zscore_clip
        requested_feature_cols = _as_list(feature_cols)
        self.passthrough_indices = [
            idx
            for idx, col in enumerate(requested_feature_cols)
            if col in set(sentiment_cols)
        ]

        prepared = load_price_series(
            path_data=path_data,
            feature_cols=requested_feature_cols,
            timestamp_col=timestamp_col,
            validation_fraction=validation_fraction,
            log_volume=log_volume,
            sentiment_path=sentiment_path,
            sentiment_cols=sentiment_cols,
            feature_transform=feature_transform,
            market_data=market_data,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            data_end_date=data_end_date,
            return_metadata=True,
        )
        self.train_df, self.val_df, self.test_df = prepared["features"]
        self.feature_cols = prepared["feature_cols"]
        self.feature_names = list(self.feature_cols)
        self.feature_dim = len(self.feature_cols)
        self.warmup_report = prepared["warmup_report"]
        self.market_alignment_report = prepared["market_alignment_report"]
        self.passthrough_indices = [
            idx for idx, col in enumerate(self.feature_cols) if col in set(sentiment_cols)
        ]

        if split == "train":
            self.time_series = self.train_df
        elif split == "val":
            self.time_series = self.val_df
        elif split == "test":
            self.time_series = self.test_df
        else:
            raise ValueError("Pretraining split must be 'train', 'val', or 'test'")

        requested_stride = stride if stride is not None else patch_size
        self.sample_starts, self.stride = _sample_starts(
            series_length=len(self.time_series),
            sample_length=self.series_split_size,
            stride=requested_stride,
            sampling_mode=self.sampling_mode,
        )

        if self.normalization in ("train_zscore", "train_robust_zscore"):
            if normalization_stats is not None and normalization_stats.get("mode") != self.normalization:
                raise ValueError(
                    "Normalization state mode does not match loader mode: "
                    f"state={normalization_stats.get('mode')!r}, "
                    f"loader={self.normalization!r}"
                )
            if normalization_stats is None:
                if self.normalization == "train_zscore":
                    normalization_stats = train_zscore_state(
                        self.train_df, self.feature_cols
                    )
                else:
                    normalization_stats = train_robust_zscore_state(
                        self.train_df,
                        self.feature_cols,
                        clip=robust_zscore_clip,
                    )
            self.normalization_stats = normalization_stats
            self.normalization_mean, self.normalization_std = normalization_tensors(
                self.normalization_stats,
                self.feature_cols,
            )
        else:
            self.normalization_stats = {
                "mode": self.normalization,
                "feature_cols": list(self.feature_cols),
            }
            self.normalization_mean = None
            self.normalization_std = None

        self.split_series = self._make_training_samples(
            series=self.time_series,
            window_size=self.series_split_size,
        )

    def _make_training_samples(self, series, window_size):
        windows = [
            series[start:start + window_size]
            for start in self.sample_starts
        ]

        if len(windows) == 0:
            raise ValueError(
                f"No training sample can be created. "
                f"len(series)={len(series)}, "
                f"window_size={window_size}, stride={self.stride}, "
                f"sampling_mode={self.sampling_mode!r}"
            )

        return windows

    def __len__(self):
        return len(self.split_series)

    def __getitem__(self, idx):
        selected_series = self.split_series[idx]  # [T, C]

        if self.normalization == "window_return":
            base = selected_series[0]             # [C]
            selected_series = cumulative_return_normalize_with_base(
                selected_series,
                base=base,
                passthrough_indices=self.passthrough_indices,
            )
        elif self.normalization in ("train_zscore", "train_robust_zscore"):
            selected_series = (
                selected_series - self.normalization_mean
            ) / self.normalization_std
            clip = self.normalization_stats.get("clip")
            if clip is not None:
                selected_series = selected_series.clamp(-float(clip), float(clip))

        num_patches = len(selected_series) // self.patch_size

        patches = [
            selected_series[i * self.patch_size:(i + 1) * self.patch_size]
            for i in range(num_patches)
        ]

        patches_tensor = torch.stack(patches)     # [num_patches, patch_size, C]
        patches_tensor = patches_tensor.reshape(num_patches, -1)  # [num_patches, patch_size*C]

        num_masked_patches = int(num_patches * self.mask_ratio)
        num_masked_patches = max(1, num_masked_patches)

        if self.mask_seed is None:
            mask_indices = random.sample(range(num_patches), num_masked_patches)
        else:
            mask_indices = random.Random(self.mask_seed + idx).sample(
                range(num_patches),
                num_masked_patches,
            )
        non_mask_indices = [i for i in range(num_patches) if i not in mask_indices]

        mask_indices = torch.tensor(mask_indices, dtype=torch.long)
        non_mask_indices = torch.tensor(non_mask_indices, dtype=torch.long)

        return patches_tensor, mask_indices, non_mask_indices


class EvaluationDataLoader(Dataset):
    """
    Loader for downstream forecasting / evaluation.

    Context uses all feature_cols, flattened by patch:
        context_patches: [context_size, patch_size * feature_dim]

    Target can be one column, e.g. target_col="Close":
        target_patch: [patch_size]

    forecast_target="relative_return" produces the cumulative simple-return
    path from the last observed target value at the forecast cutoff:
        target[h] / context[-1] - 1

    forecast_target="cumulative_log_return" produces:
        log(Close[t+h] / Close[t]), h=1,...,H
    """

    def __init__(
        self,
        path_data,
        patch_size=5,
        forecast_horizon=None,
        context_size=10,
        stride=1,
        sampling_mode="sliding_window",
        split="test",
        normalize=True,
        normalization=None,
        normalization_stats=None,
        feature_cols=("Close", "Volume"),
        target_col="Close",
        forecast_target="value",
        timestamp_col="Date",
        validation_fraction=0.05,
        log_volume=True,
        sentiment_path=None,
        sentiment_cols=DEFAULT_SENTIMENT_COLS,
        feature_transform="raw",
        market_data=None,
        robust_zscore_clip=None,
        train_end_date=None,
        test_start_date=None,
        data_end_date=None,
    ):
        self.patch_size = patch_size
        self.forecast_horizon = resolve_forecast_horizon(
            forecast_horizon,
            patch_size,
        )
        self.context_size = context_size
        self.sampling_mode = _resolve_sampling_mode(sampling_mode)
        self.split = split
        self.normalization = _resolve_normalization_mode(normalization, normalize)
        self.normalize = self.normalization != "none"
        self.feature_transform = feature_transform
        self.market_data = market_data
        if feature_transform == "return" and self.normalization == "window_return":
            raise ValueError(
                "window_return is not defined for already-return-based features; "
                "use train_zscore, train_robust_zscore, or none"
            )
        self.robust_zscore_clip = robust_zscore_clip
        requested_feature_cols = _as_list(feature_cols)
        self.target_col = target_col
        self.forecast_target = _resolve_forecast_target(forecast_target)

        prepared = load_price_series(
            path_data=path_data,
            feature_cols=requested_feature_cols,
            timestamp_col=timestamp_col,
            validation_fraction=validation_fraction,
            log_volume=log_volume,
            sentiment_path=sentiment_path,
            sentiment_cols=sentiment_cols,
            feature_transform=feature_transform,
            market_data=market_data,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            data_end_date=data_end_date,
            return_metadata=True,
        )
        self.train_df, self.val_df, self.test_df = prepared["features"]
        self.train_close, self.val_close, self.test_close = prepared["close"]
        (
            self.train_market_close,
            self.val_market_close,
            self.test_market_close,
        ) = prepared["market_close"]
        self.train_dates, self.val_dates, self.test_dates = prepared["dates"]
        self.feature_cols = prepared["feature_cols"]
        self.feature_names = list(self.feature_cols)
        self.feature_dim = len(self.feature_cols)
        self.warmup_report = prepared["warmup_report"]
        self.market_alignment_report = prepared["market_alignment_report"]
        self.passthrough_indices = [
            idx for idx, col in enumerate(self.feature_cols) if col in set(sentiment_cols)
        ]

        if target_col in self.feature_cols:
            self.target_idx = self.feature_cols.index(target_col)
        elif self.forecast_target in (
            "relative_return",
            "cumulative_log_return",
            "excess_log_return",
        ) and target_col == "Close":
            self.target_idx = None
        else:
            raise ValueError(
                f"target_col={target_col!r} must be in feature_cols={self.feature_cols} "
                f"for forecast_target={self.forecast_target!r}"
            )
        if self.forecast_target == "excess_log_return" and market_data is None:
            raise ValueError("excess_log_return requires market_data")

        if self.normalization in ("train_zscore", "train_robust_zscore"):
            if normalization_stats is not None and normalization_stats.get("mode") != self.normalization:
                raise ValueError(
                    "Normalization state mode does not match loader mode: "
                    f"state={normalization_stats.get('mode')!r}, "
                    f"loader={self.normalization!r}"
                )
            if normalization_stats is None:
                if self.normalization == "train_zscore":
                    normalization_stats = train_zscore_state(
                        self.train_df, self.feature_cols
                    )
                else:
                    normalization_stats = train_robust_zscore_state(
                        self.train_df,
                        self.feature_cols,
                        clip=robust_zscore_clip,
                    )
            self.normalization_stats = normalization_stats
            self.normalization_mean, self.normalization_std = normalization_tensors(
                self.normalization_stats,
                self.feature_cols,
            )
        else:
            self.normalization_stats = {
                "mode": self.normalization,
                "feature_cols": list(self.feature_cols),
            }
            self.normalization_mean = None
            self.normalization_std = None

        if split == "train":
            self.series = self.train_df
            self.close_series = self.train_close
            self.market_close_series = self.train_market_close
            self.dates = self.train_dates
        elif split == "val":
            self.series = self.val_df
            self.close_series = self.val_close
            self.market_close_series = self.val_market_close
            self.dates = self.val_dates
        elif split == "test":
            self.series = self.test_df
            self.close_series = self.test_close
            self.market_close_series = self.test_market_close
            self.dates = self.test_dates
        elif split == "all":
            self.series = torch.cat([self.train_df, self.val_df, self.test_df], dim=0)
            self.close_series = torch.cat(
                [self.train_close, self.val_close, self.test_close], dim=0
            )
            self.market_close_series = (
                torch.cat(
                    [
                        self.train_market_close,
                        self.val_market_close,
                        self.test_market_close,
                    ],
                    dim=0,
                )
                if self.train_market_close is not None
                else None
            )
            self.dates = self.train_dates + self.val_dates + self.test_dates
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', 'test', or 'all'.")

        context_length = self.context_size * self.patch_size
        sample_length = context_length + self.forecast_horizon
        self.sample_starts, self.stride = _sample_starts(
            series_length=len(self.series),
            sample_length=sample_length,
            stride=stride,
            sampling_mode=self.sampling_mode,
        )
        if self.sampling_mode == "temporal_segments":
            # The evaluator uses the true target offsets for forecast logging.
            self.indices = [start + context_length for start in self.sample_starts]
        self.samples = self._make_samples(self.series)

        print(
            f"[EvaluationDataLoader] split={split}, "
            f"series_len={len(self.series)}, "
            f"num_samples={len(self.samples)}, "
            f"context_size={context_size}, "
            f"patch_size={patch_size}, "
            f"forecast_horizon={self.forecast_horizon}, "
            f"feature_dim={self.feature_dim}, "
            f"target_col={target_col}, "
            f"forecast_target={self.forecast_target}, "
            f"sampling_mode={self.sampling_mode}, "
            f"stride={self.stride}"
        )

    def _make_samples(self, series):
        context_len = self.context_size * self.patch_size
        target_len = self.forecast_horizon
        total_len = context_len + target_len

        samples = []
        for start in self.sample_starts:
            full_window = series[start:start + total_len]
            context_flat = full_window[:context_len]   # [context_len, C]
            target_flat = full_window[context_len:]    # [forecast_horizon, C]
            samples.append((context_flat, target_flat))

        if len(samples) == 0:
            raise ValueError(
                f"No evaluation sample can be created. "
                f"len(series)={len(series)}, "
                f"context_size={self.context_size}, "
                f"patch_size={self.patch_size}, "
                f"forecast_horizon={self.forecast_horizon}, "
                f"stride={self.stride}, "
                f"sampling_mode={self.sampling_mode!r}"
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context_flat, target_flat = self.samples[idx]
        context_len = self.context_size * self.patch_size
        start = self.sample_starts[idx]
        cutoff_index = start + context_len - 1
        future_slice = slice(
            cutoff_index + 1,
            cutoff_index + 1 + self.forecast_horizon,
        )

        if self.forecast_target == "relative_return":
            # Build the label before input normalization. The denominator is
            # the last value available at the forecast cutoff, never a future
            # observation, so the target is leakage-safe and independent of
            # the configured encoder-input normalization.
            if self.target_col == "Close":
                cutoff_value = self.close_series[cutoff_index]
                future_values = self.close_series[future_slice]
            else:
                cutoff_value = context_flat[-1, self.target_idx]
                future_values = target_flat[:, self.target_idx]
            target_patch = cumulative_return_normalize_with_base(
                future_values,
                base=cutoff_value,
            ).reshape(self.forecast_horizon)
        elif self.forecast_target in (
            "cumulative_log_return",
            "excess_log_return",
        ):
            cutoff_close = self.close_series[cutoff_index]
            future_close = self.close_series[future_slice]
            if cutoff_close <= 0 or torch.any(future_close <= 0):
                raise ValueError("Close must be positive for cumulative log-return targets")
            target_patch = torch.log(future_close / cutoff_close)
            if self.forecast_target == "excess_log_return":
                market_cutoff = self.market_close_series[cutoff_index]
                market_future = self.market_close_series[future_slice]
                if market_cutoff <= 0 or torch.any(market_future <= 0):
                    raise ValueError(
                        "Market Close must be positive for excess log-return targets"
                    )
                target_patch = target_patch - torch.log(
                    market_future / market_cutoff
                )

        if self.normalization == "window_return":
            base = context_flat[0]  # [C], only from first context point
            context_flat = cumulative_return_normalize_with_base(
                context_flat,
                base=base,
                passthrough_indices=self.passthrough_indices,
            )
            target_flat = cumulative_return_normalize_with_base(
                target_flat,
                base=base,
                passthrough_indices=self.passthrough_indices,
            )
        elif self.normalization in ("train_zscore", "train_robust_zscore"):
            context_flat = (
                context_flat - self.normalization_mean
            ) / self.normalization_std
            target_flat = (
                target_flat - self.normalization_mean
            ) / self.normalization_std
            clip = self.normalization_stats.get("clip")
            if clip is not None:
                context_flat = context_flat.clamp(-float(clip), float(clip))
                target_flat = target_flat.clamp(-float(clip), float(clip))

        context_patches = context_flat.reshape(
            self.context_size,
            self.patch_size * self.feature_dim,
        )

        # Forecast only target_col, usually Close. Relative-return labels were
        # already computed from raw values above.
        if self.forecast_target == "value":
            target_patch = target_flat[:, self.target_idx].reshape(
                self.forecast_horizon
            )

        return context_patches, target_patch
