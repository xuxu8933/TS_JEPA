import pandas as pd
import torch
import random
from torch.utils.data import Dataset


def cumulative_return_normalize_with_base(series, base, eps=1e-8):
    """
    Bulbea-style cumulative return normalization:

        x_norm[t] = x[t] / base - 1

    Important:
    - base should come only from observable history.
    - For evaluation, base should be the first point of the context window,
      not from target future values.

    series: 1D torch.Tensor
    base: scalar torch.Tensor
    """
    if torch.abs(base) < eps:
        base = base + eps

    return series / base - 1.0


def chronological_split(
    df,
    input_col="Close",
    timestamp_col="Date",
    validation_fraction=0.05,
    test_fraction=0.30,
):
    """
    Chronological split:

        train: earliest period
        val:   middle period
        test:  latest future period

    This prevents future leakage.
    """
    df = df.copy()

    df.sort_values(by=[timestamp_col], inplace=True)

    values = torch.tensor(df[input_col].values).float()

    val_len = int(len(values) * validation_fraction)
    test_len = int(len(values) * test_fraction)
    train_len = len(values) - val_len - test_len

    train_df, val_df, test_df = torch.split(
        values,
        [train_len, val_len, test_len]
    )

    return train_df, val_df, test_df


def load_price_series(
    path_data,
    input_col="Close",
    timestamp_col="Date",
    validation_fraction=0.05,
    test_fraction=0.30,
):
    """
    Load CSV and return chronological train / val / test tensors.
    """
    df = pd.read_csv(
        path_data,
        parse_dates=[timestamp_col],
        low_memory=False,
        sep=",",
    )

    # Downcast float columns
    fcols = df.select_dtypes("float").columns
    df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")

    # Downcast integer columns
    icols = df.select_dtypes("integer").columns
    df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")

    train_df, val_df, test_df = chronological_split(
        df=df,
        input_col=input_col,
        timestamp_col=timestamp_col,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )

    return train_df, val_df, test_df


class CSVDataLoader(Dataset):
    """
    Loader for TS-JEPA pretraining.

    Main idea:
    - Use only train split for pretraining.
    - Split train time series into rolling / sliding windows.
    - Normalize each window with its own first value.
    - Then split each window into patches.
    - Randomly mask patches for JEPA pretraining.

    This avoids using validation/test future data during pretraining.
    """

    def __init__(
        self,
        path_data,
        batch_size=32,
        series_split_size=120,
        patch_size=5,
        mask_ratio=0.15,
        stride=60,
        normalize=True,
        input_col="Close",
        timestamp_col="Date",
        validation_fraction=0.05,
        test_fraction=0.30,
    ):
        self.batch_size = batch_size
        self.series_split_size = series_split_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.normalize = normalize

        # For pretraining, using patch_size as stride gives more samples.
        # You can set stride=60 if you want less overlap.
        self.stride = stride if stride is not None else patch_size

        self.train_df, self.val_df, self.test_df = load_price_series(
            path_data=path_data,
            input_col=input_col,
            timestamp_col=timestamp_col,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )

        # Important:
        # Pretraining only uses train_df.
        self.time_series = self.train_df

        self.split_series = self._make_sliding_windows(
            series=self.time_series,
            window_size=self.series_split_size,
            stride=self.stride,
        )

    def _make_sliding_windows(self, series, window_size, stride):
        windows = [
            series[i:i + window_size]
            for i in range(
                0,
                len(series) - window_size + 1,
                stride
            )
        ]

        if len(windows) == 0:
            raise ValueError(
                f"No window can be created. "
                f"len(series)={len(series)}, "
                f"window_size={window_size}, stride={stride}"
            )

        return windows

    def __len__(self):
        return len(self.split_series)

    def __getitem__(self, idx):
        selected_series = self.split_series[idx]

        # Normalize each pretraining window by its own first observable value.
        if self.normalize:
            base = selected_series[0]
            selected_series = cumulative_return_normalize_with_base(
                selected_series,
                base=base,
            )

        num_patches = len(selected_series) // self.patch_size

        patches = [
            selected_series[i * self.patch_size:(i + 1) * self.patch_size]
            for i in range(num_patches)
        ]

        patches_tensor = torch.stack(patches)

        num_masked_patches = int(num_patches * self.mask_ratio)
        num_masked_patches = max(1, num_masked_patches)

        mask_indices = random.sample(
            range(num_patches),
            num_masked_patches
        )

        non_mask_indices = [
            i for i in range(num_patches)
            if i not in mask_indices
        ]

        mask_indices = torch.tensor(mask_indices, dtype=torch.long)
        non_mask_indices = torch.tensor(non_mask_indices, dtype=torch.long)

        return patches_tensor, mask_indices, non_mask_indices


class EvaluationDataLoader(Dataset):
    """
    Loader for downstream forecasting / linear probing / evaluation.

    Main idea:
    - You choose split="train", "val", or "test".
    - For downstream decoder training, use split="train".
    - For validation, use split="val".
    - For final Impermanent-style future evaluation, use split="test".
    - Each sample is:

        context_patches -> target_patch

    - Normalization uses only the first point of the context window as base.
      The target is normalized using the same context base.
      This avoids future-statistics leakage.
    """

    def __init__(
        self,
        path_data,
        patch_size=5,
        context_size=10,
        stride=1,
        split="test",
        normalize=True,
        input_col="Close",
        timestamp_col="Date",
        validation_fraction=0.05,
        test_fraction=0.30,
    ):
        self.patch_size = patch_size
        self.context_size = context_size
        self.stride = stride
        self.split = split
        self.normalize = normalize

        self.train_df, self.val_df, self.test_df = load_price_series(
            path_data=path_data,
            input_col=input_col,
            timestamp_col=timestamp_col,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )

        if split == "train":
            self.series = self.train_df
        elif split == "val":
            self.series = self.val_df
        elif split == "test":
            self.series = self.test_df
        elif split == "all":
            self.series = torch.cat(
                [self.train_df, self.val_df, self.test_df],
                dim=0,
            )
        else:
            raise ValueError(
                f"Unknown split: {split}. "
                f"Use 'train', 'val', 'test', or 'all'."
            )

        self.samples = self._make_rolling_samples(self.series)

        print(
            f"[EvaluationDataLoader] split={split}, "
            f"series_len={len(self.series)}, "
            f"num_samples={len(self.samples)}, "
            f"context_size={context_size}, "
            f"patch_size={patch_size}, "
            f"stride={stride}"
        )

    def _make_rolling_samples(self, series):
        """
        Create rolling forecasting samples.

        Each sample length:

            context_size * patch_size + patch_size

        That means:

            context patches + next target patch
        """
        context_len = self.context_size * self.patch_size
        target_len = self.patch_size
        total_len = context_len + target_len

        samples = []

        for start in range(
            0,
            len(series) - total_len + 1,
            self.stride,
        ):
            full_window = series[start:start + total_len]

            context_flat = full_window[:context_len]
            target_flat = full_window[context_len:]

            samples.append((context_flat, target_flat))

        if len(samples) == 0:
            raise ValueError(
                f"No evaluation sample can be created. "
                f"len(series)={len(series)}, "
                f"context_size={self.context_size}, "
                f"patch_size={self.patch_size}, "
                f"stride={self.stride}"
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context_flat, target_flat = self.samples[idx]

        if self.normalize:
            # Strict deployment-faithful normalization:
            # base comes only from the first point of the context.
            base = context_flat[0]

            context_flat = cumulative_return_normalize_with_base(
                context_flat,
                base=base,
            )

            target_flat = cumulative_return_normalize_with_base(
                target_flat,
                base=base,
            )

        context_patches = context_flat.reshape(
            self.context_size,
            self.patch_size,
        )

        target_patch = target_flat.reshape(
            self.patch_size,
        )

        return context_patches, target_patch