"""
Script for the Data Loaders.
"""

from torch.utils.data import DataLoader
from .data_class_roll_volume import CSVDataLoader, EvaluationDataLoader


def get_jepa_loaders(
    path,
    batch_size,
    ratio_patches=10,
    mask_ratio=0.9,
    train_until_index=None,
    series_split_size=60,
    patch_size=5,
    feature_cols=("Close", "Volume"),
    timestamp_col="Date",
    normalize=True,
):
    """
    Load and prepare the data to be used with TS-JEPA pretraining.
    """

    dataset = CSVDataLoader(
        path_data=path,
        series_split_size=series_split_size,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        normalize=normalize,
        feature_cols=feature_cols,
        timestamp_col=timestamp_col,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return loader


def get_evaluation_loaders(
    path_data,
    batch_size,
    ratio_patches,
    mask_ratio=None,
    split="train",
    patch_size=5,
    context_size=10,
    stride=1,
    normalize=True,
    feature_cols=("Close", "Volume"),
    target_col="Close",
    timestamp_col="Date",
):
    """
    Downstream / evaluation loader.
    """

    dataset = EvaluationDataLoader(
        path_data=path_data,
        patch_size=patch_size,
        context_size=context_size,
        stride=stride,
        split=split,
        normalize=normalize,
        feature_cols=feature_cols,
        target_col=target_col,
        timestamp_col=timestamp_col,
    )

    shuffle = True if split == "train" else False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )

    return loader


if __name__ == "__main__":
    pass
