import pandas as pd
import torch
import random

def cumulative_return_normalize(series, eps=1e-8):
    """
    Bulbea-style normalization:
        x_norm[t] = x[t] / x[0] - 1

    series: 1D torch.Tensor
    """
    base = series[0]

    # 防止第一个值为 0
    if torch.abs(base) < eps:
        base = base + eps

    return series / base - 1.0

class CSVDataLoader():
    def __init__(
        self,
        path_data,
        batch_size=32,
        series_split_size=120,
        patch_size=5,
        mask_ratio=0.15,
        stride=60,
        normalize=True,
    ):
        # 注意：bulbea 风格更适合 price，例如 close，而不是 close_r
        input_variables = "Close"
        timestamp_col = "Date"
        validation_fraction = 0.05
        test_fraction = 0.3

        self.normalize = normalize

        df = pd.read_csv(
            path_data,
            parse_dates=[timestamp_col],
            low_memory=False,
            sep=",",
        )

        fcols = df.select_dtypes("float").columns
        df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")
        # df_mean = df[fcols].mean(0)
        # df_std = df[fcols].std(0)
        # df[fcols] = (df[fcols] - df_mean) / df_std

        icols = df.select_dtypes("integer").columns
        df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")

        # 先按时间排序
        df.sort_values(by=[timestamp_col], inplace=True)

        val_len = int(len(df) * validation_fraction)
        test_len = int(len(df) * test_fraction)
        train_len = len(df) - val_len - test_len

        df = torch.tensor(df[input_variables].values).float()
        train_df, val_df, test_df = torch.split(df, [train_len, val_len, test_len])

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.series_split_size = series_split_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.stride = stride if stride is not None else patch_size

        self.time_series_list = train_df

        # sliding windows
        self.split_series = [
            self.time_series_list[i:i + self.series_split_size]
            for i in range(
                0,
                len(self.time_series_list) - self.series_split_size + 1,
                self.stride
            )
        ]

    def __getitem__(self, idx):
        # Split the original time series into smaller time series chunks
        # ts = self.time_series_list
        # num_splits = len(ts) // self.series_split_size
        # split_series = [ts[i*self.series_split_size:(i+1)*self.series_split_size] for i in range(num_splits)]

        # Select the series based on the index (assuming batched access)
        # selected_series = split_series[idx % len(split_series)]

        selected_series = self.split_series[idx]

        # Bulbea-style normalization
        if self.normalize:
            selected_series = cumulative_return_normalize(selected_series)

        num_patches = len(selected_series) // self.patch_size

        patches = [
            selected_series[i * self.patch_size:(i + 1) * self.patch_size]
            for i in range(num_patches)
        ]

        patches_tensor = torch.stack(patches)

        num_masked_patches = int(num_patches * self.mask_ratio)
        num_masked_patches = max(1, num_masked_patches)

        mask_indices = random.sample(range(num_patches), num_masked_patches)
        non_mask_indices = [
            i for i in range(num_patches)
            if i not in mask_indices
        ]

        mask_indices = torch.tensor(mask_indices)
        non_mask_indices = torch.tensor(non_mask_indices)

        return patches_tensor, mask_indices, non_mask_indices

    def __len__(self):
        return len(self.split_series)
        # Number of smaller time series created from the full series
        # return len(self.time_series_list) // self.series_split_size    


class EvaluationDataLoader():
    """
    Evaluation loader for downstream forecasting.
    Bulbea-style normalization is applied per context-target window.
    """
    def __init__(
        self,
        path_data,
        patch_size=32,
        context_size=10,
        normalize=True,
     ):
        self.patch_size = patch_size
        self.context_size = context_size
        self.normalize = normalize

        input_variables = "Close"
        timestamp_col = "Date"
        validation_fraction = 0.05
        test_fraction = 0.3

        df = pd.read_csv(path_data, parse_dates=[timestamp_col],
                                                    low_memory=False, sep=",")

        fcols = df.select_dtypes("float").columns
        df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")

        # df_mean = df[fcols].mean(0)
        # df_std = df[fcols].std(0)

        # df[fcols] = (df[fcols] - df_mean) / df_std


        icols = df.select_dtypes("integer").columns
        df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")

        df.sort_values(by=[timestamp_col], inplace=True)

        # Split into train, validation, and test sets
        val_len = int(len(df) * validation_fraction)
        test_len = int(len(df) * test_fraction)
        train_len = len(df) - val_len - test_len
        df = torch.tensor(df[input_variables].values).float()
        train_df, val_df, test_df = torch.split(df, [train_len, val_len, test_len])

        # Store data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.series = self.train_df
        print("series len: ", len(self.series))
        # Split the entire time series into patches
        self.patches_tensor = self.split_into_patches(self.series, self.patch_size)

    def split_into_patches(self, series, patch_size):
        num_patches = len(series) // patch_size
        patches = [series[i * patch_size:(i + 1) * patch_size] for i in range(num_patches)]
        return torch.stack(patches)  # Shape will be (num_patches, patch_size)

    def __len__(self):
        # Number of available samples based on the context size
        return len(self.patches_tensor) - self.context_size

    def __getitem__(self, idx):
        # Here we ensure that each time we return a context window of 10 patches
        if idx + self.context_size + 1 > len(self.patches_tensor):
            raise IndexError("Index out of range for context window")

        # Get context patches (previous 10) and the target patch (next one)
        context_patches = self.patches_tensor[idx:idx + self.context_size]
        target_patch = self.patches_tensor[idx + self.context_size]

        if self.normalize:
            # 拼成完整 window
            full_window = torch.cat(
                [
                    context_patches.flatten(),
                    target_patch.flatten()
                ],
                dim=0
            )

            # 用 context 的第一个点作为 base
            full_window = cumulative_return_normalize(full_window)

            context_len = self.context_size * self.patch_size

            context_flat = full_window[:context_len]
            target_flat = full_window[context_len:]

            context_patches = context_flat.reshape(
                self.context_size,
                self.patch_size
            )

            target_patch = target_flat.reshape(
                self.patch_size
            )

        return context_patches, target_patch