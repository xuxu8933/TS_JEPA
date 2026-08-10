import os
import pandas as pd
import torch
import random
from torch.utils.data import Dataset


DEFAULT_SENTIMENT_COLS = (
    "sentiment_mean",
    "sentiment_sum",
    "sentiment_max",
    "sentiment_min",
    "sentiment_std",
    "news_count",
)

NORMALIZATION_MODES = ("window_return", "train_zscore", "none")
SAMPLING_MODES = ("sliding_window", "temporal_segments")
FORECAST_TARGETS = ("value", "relative_return")


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


def normalization_tensors(state, feature_cols):
    if state is None:
        return None, None
    if state.get("mode") != "train_zscore":
        raise ValueError(
            "train_zscore requires normalization state with mode='train_zscore', "
            f"got {state.get('mode')!r}"
        )
    if list(state.get("feature_cols", [])) != list(feature_cols):
        raise ValueError(
            "Normalization feature order does not match loader feature order: "
            f"state={state.get('feature_cols')}, loader={list(feature_cols)}"
        )
    mean = torch.tensor(state["mean"], dtype=torch.float32)
    std = torch.tensor(state["std"], dtype=torch.float32)
    return mean, std


def _as_list(x):
    if isinstance(x, str):
        return [x]
    return list(x)


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
    df = df.copy()
    df.sort_values(by=[timestamp_col], inplace=True)

    if data_end_date is not None:
        data_end_ts = pd.Timestamp(data_end_date)
        df = df[df[timestamp_col] <= data_end_ts].copy()
        if df.empty:
            raise ValueError(
                f"No rows found on or before data_end_date={data_end_date!r}."
            )

    feature_cols = _as_list(feature_cols)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Available columns: {list(df.columns)}")

    if test_start_date is None:
        raise ValueError("test_start_date must be defined for chronological splitting.")

    train_end_ts = pd.Timestamp(train_end_date) if train_end_date is not None else None
    test_start_ts = pd.Timestamp(test_start_date)
    if train_end_ts is not None and train_end_ts >= test_start_ts:
        raise ValueError(
            "train_end_date must be earlier than test_start_date: "
            f"train_end_date={train_end_date!r}, test_start_date={test_start_date!r}"
        )

    if train_end_ts is None:
        train_val_df = df[df[timestamp_col] < test_start_ts].copy()
    else:
        train_val_df = df[df[timestamp_col] <= train_end_ts].copy()
    test_df = df[df[timestamp_col] >= test_start_ts].copy()

    if train_val_df.empty:
        raise ValueError(
            f"No train/validation rows found for train_end_date={train_end_date!r} "
            f"and test_start_date={test_start_date!r}."
        )
    if test_df.empty:
        raise ValueError(
            f"No test rows found for test_start_date={test_start_date!r}."
        )

    train_val_values = torch.tensor(
        train_val_df[feature_cols].values,
        dtype=torch.float32,
    )
    test_values = torch.tensor(
        test_df[feature_cols].values,
        dtype=torch.float32,
    )

    val_len = int(len(train_val_values) * validation_fraction)
    train_len = len(train_val_values) - val_len
    if train_len <= 0:
        raise ValueError(
            f"Date split left no train rows. train_val_len={len(train_val_values)}, "
            f"validation_fraction={validation_fraction}"
        )

    if val_len > 0:
        train_df, val_df = torch.split(train_val_values, [train_len, val_len])
    else:
        train_df = train_val_values
        val_df = train_val_values[:0]

    train_val_boundary = (
        f"<= {train_end_date}" if train_end_date is not None else f"< {test_start_date}"
    )
    print(
        "[chronological_split] date split: "
        f"train_val {train_val_boundary}, test>= {test_start_date}, "
        f"data<= {data_end_date}, "
        f"train_len={len(train_df)}, val_len={len(val_df)}, test_len={len(test_values)}"
    )

    return train_df, val_df, test_values


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
):
    """
    Load CSV and return chronological train / val / test tensors.

    feature_cols can be:
        ("Close", "Volume")
        ("Open", "High", "Low", "Close", "Volume")
    """
    df = pd.read_csv(
        path_data,
        parse_dates=[timestamp_col],
        low_memory=False,
        sep=",",
    )
    feature_cols = _as_list(feature_cols)

    df = _merge_daily_sentiment(
        df=df,
        path_data=path_data,
        timestamp_col=timestamp_col,
        feature_cols=feature_cols,
        sentiment_path=sentiment_path,
        sentiment_cols=sentiment_cols,
    )

    # Add moving average features
    # df["MA5"] = df["Close"].rolling(window=5, min_periods=5).mean()
    df["MA10"] = df["Close"].rolling(window=10, min_periods=10).mean()
    df["MA50"] = df["Close"].rolling(window=50, min_periods=50).mean()

    # Drop rows where MA is NaN
    df = df.dropna().reset_index(drop=True)

    # Downcast float columns
    fcols = df.select_dtypes("float").columns
    df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")

    # Downcast integer columns
    icols = df.select_dtypes("integer").columns
    df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")

    # Volume is usually very large and heavy-tailed, so log1p is safer.
    if log_volume and "Volume" in feature_cols:
        df["Volume"] = torch.log1p(
            torch.tensor(df["Volume"].values, dtype=torch.float32)
        ).numpy()

    train_df, val_df, test_df = chronological_split(
        df=df,
        feature_cols=feature_cols,
        timestamp_col=timestamp_col,
        validation_fraction=validation_fraction,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        data_end_date=data_end_date,
    )

    return train_df, val_df, test_df


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
        batch_size=32,
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
        train_end_date=None,
        test_start_date=None,
        data_end_date=None,
    ):
        self.batch_size = batch_size
        self.series_split_size = series_split_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.sampling_mode = _resolve_sampling_mode(sampling_mode)
        self.normalization = _resolve_normalization_mode(normalization, normalize)
        self.normalize = self.normalization != "none"
        self.split = split
        self.mask_seed = mask_seed
        self.feature_cols = _as_list(feature_cols)
        self.feature_dim = len(self.feature_cols)
        self.passthrough_indices = [
            idx
            for idx, col in enumerate(self.feature_cols)
            if col in set(sentiment_cols)
        ]

        self.train_df, self.val_df, self.test_df = load_price_series(
            path_data=path_data,
            feature_cols=self.feature_cols,
            timestamp_col=timestamp_col,
            validation_fraction=validation_fraction,
            log_volume=log_volume,
            sentiment_path=sentiment_path,
            sentiment_cols=sentiment_cols,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            data_end_date=data_end_date,
        )

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

        if self.normalization == "train_zscore":
            self.normalization_stats = normalization_stats or train_zscore_state(
                self.train_df,
                self.feature_cols,
            )
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
        elif self.normalization == "train_zscore":
            selected_series = (
                selected_series - self.normalization_mean
            ) / self.normalization_std

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
    """

    def __init__(
        self,
        path_data,
        patch_size=5,
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
        train_end_date=None,
        test_start_date=None,
        data_end_date=None,
    ):
        self.patch_size = patch_size
        self.context_size = context_size
        self.sampling_mode = _resolve_sampling_mode(sampling_mode)
        self.split = split
        self.normalization = _resolve_normalization_mode(normalization, normalize)
        self.normalize = self.normalization != "none"
        self.feature_cols = _as_list(feature_cols)
        self.feature_dim = len(self.feature_cols)
        self.passthrough_indices = [
            idx
            for idx, col in enumerate(self.feature_cols)
            if col in set(sentiment_cols)
        ]
        self.target_col = target_col
        self.forecast_target = _resolve_forecast_target(forecast_target)

        if target_col not in self.feature_cols:
            raise ValueError(f"target_col={target_col} must be in feature_cols={self.feature_cols}")
        self.target_idx = self.feature_cols.index(target_col)

        self.train_df, self.val_df, self.test_df = load_price_series(
            path_data=path_data,
            feature_cols=self.feature_cols,
            timestamp_col=timestamp_col,
            validation_fraction=validation_fraction,
            log_volume=log_volume,
            sentiment_path=sentiment_path,
            sentiment_cols=sentiment_cols,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            data_end_date=data_end_date,
        )

        if self.normalization == "train_zscore":
            self.normalization_stats = normalization_stats or train_zscore_state(
                self.train_df,
                self.feature_cols,
            )
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
        elif split == "val":
            self.series = self.val_df
        elif split == "test":
            self.series = self.test_df
        elif split == "all":
            self.series = torch.cat([self.train_df, self.val_df, self.test_df], dim=0)
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', 'test', or 'all'.")

        context_length = self.context_size * self.patch_size
        sample_length = context_length + self.patch_size
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
            f"feature_dim={self.feature_dim}, "
            f"target_col={target_col}, "
            f"forecast_target={self.forecast_target}, "
            f"sampling_mode={self.sampling_mode}, "
            f"stride={self.stride}"
        )

    def _make_samples(self, series):
        context_len = self.context_size * self.patch_size
        target_len = self.patch_size
        total_len = context_len + target_len

        samples = []
        for start in self.sample_starts:
            full_window = series[start:start + total_len]
            context_flat = full_window[:context_len]   # [context_len, C]
            target_flat = full_window[context_len:]    # [patch_size, C]
            samples.append((context_flat, target_flat))

        if len(samples) == 0:
            raise ValueError(
                f"No evaluation sample can be created. "
                f"len(series)={len(series)}, "
                f"context_size={self.context_size}, "
                f"patch_size={self.patch_size}, "
                f"stride={self.stride}, "
                f"sampling_mode={self.sampling_mode!r}"
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context_flat, target_flat = self.samples[idx]

        if self.forecast_target == "relative_return":
            # Build the label before input normalization. The denominator is
            # the last value available at the forecast cutoff, never a future
            # observation, so the target is leakage-safe and independent of
            # the configured encoder-input normalization.
            cutoff_value = context_flat[-1, self.target_idx]
            target_patch = cumulative_return_normalize_with_base(
                target_flat[:, self.target_idx],
                base=cutoff_value,
            ).reshape(self.patch_size)

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
        elif self.normalization == "train_zscore":
            context_flat = (
                context_flat - self.normalization_mean
            ) / self.normalization_std
            target_flat = (
                target_flat - self.normalization_mean
            ) / self.normalization_std

        context_patches = context_flat.reshape(
            self.context_size,
            self.patch_size * self.feature_dim,
        )

        # Forecast only target_col, usually Close. Relative-return labels were
        # already computed from raw values above.
        if self.forecast_target == "value":
            target_patch = target_flat[:, self.target_idx].reshape(self.patch_size)

        return context_patches, target_patch
