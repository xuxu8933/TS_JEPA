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
    series_split_size=60,
    patch_size=5,
    stride=None,
    feature_cols=("Close", "Volume"),
    timestamp_col="Date",
    normalize=True,
    normalization=None,
    normalization_stats=None,
    split="train",
    mask_seed=None,
    sentiment_path=None,
    validation_fraction=0.05,
    test_fraction=0.30,
    train_end_date=None,
    test_start_date=None,
):
    """
    Load and prepare the data to be used with TS-JEPA pretraining.
    """

    dataset = CSVDataLoader(
        path_data=path,
        series_split_size=series_split_size,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        stride=stride,
        normalize=normalize,
        normalization=normalization,
        normalization_stats=normalization_stats,
        split=split,
        mask_seed=mask_seed,
        feature_cols=feature_cols,
        timestamp_col=timestamp_col,
        sentiment_path=sentiment_path,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
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
    normalization=None,
    normalization_stats=None,
    feature_cols=("Close", "Volume"),
    target_col="Close",
    timestamp_col="Date",
    sentiment_path=None,
    validation_fraction=0.05,
    test_fraction=0.30,
    train_end_date=None,
    test_start_date=None,
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
        normalization=normalization,
        normalization_stats=normalization_stats,
        feature_cols=feature_cols,
        target_col=target_col,
        timestamp_col=timestamp_col,
        sentiment_path=sentiment_path,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
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
