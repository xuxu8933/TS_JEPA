"""Shared CLI, reproducibility, initialization, and metric utilities."""

from collections.abc import Sequence

import torch
import argparse
import json
import math

import random
import numpy as np

from config.experiment import (
    none_if_requested,
    resolve_forecast_horizon,
    resolve_feature_selection,
    validate_data_config,
)


def ordered_scalar_mean(values: Sequence[torch.Tensor]) -> float:
    """Average detached scalars with one device-to-host transfer."""
    if not values:
        raise ValueError("ordered_scalar_mean requires at least one scalar")
    if any(value.numel() != 1 for value in values):
        raise ValueError("ordered_scalar_mean accepts only scalar tensors")
    host_values = torch.stack(
        [value.detach().reshape(()) for value in values]
    ).cpu().tolist()
    return sum(host_values) / len(host_values)


def set_seed(seed, deterministic=None):
    """Seed every random source used by the training pipelines."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic is not None:
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
        torch.use_deterministic_algorithms(deterministic, warn_only=True)


def prepare_args(config):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=config["data"])
    parser.add_argument(
        "--seed",
        type=int,
        default=config.get("seed", 42),
        help="Random seed for downstream model training and evaluation.",
    )
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--name_model", type=str, default="pre_train")
    parser.add_argument(
        "--results_dir",
        "--results-dir",
        dest="results_dir",
        default=config.get("results_dir", "./results"),
        help="Directory for downstream metrics, tables, and figures.",
    )

    parser.add_argument("--lr", type=float, default=config["lr"])
    parser.add_argument("--lr_pretrain", type=float, default=config["lr_pretrain"])
    parser.add_argument("--ratio_patches", type=int, default=config["ratio_patches"])
    parser.add_argument("--mask_ratio", type=float, default=config["mask_ratio"])
    parser.add_argument("--pooling", type=str, default=config["pooling"])
    parser.add_argument("--pre_train_mask", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=config["batch_size"])
    parser.add_argument("--num_epochs", type=int, default=config["num_epochs"])

    parser.add_argument(
        "--checkpoint_to_use", type=int, default=config["checkpoint_to_use"]
    )
    parser.add_argument(
        "--pretrain_checkpoint_path",
        "--pretrain-checkpoint-path",
        dest="pretrain_checkpoint_path",
        type=str,
        default=config.get("pretrain_checkpoint_path", None),
        help="Explicit pretrained encoder checkpoint path. Overrides the default naming scheme.",
    )
    parser.add_argument(
        "--checkpoint-selection",
        default=config.get("checkpoint_selection", "unknown"),
        help="Resolved pre-training checkpoint selection mode for provenance logging.",
    )
    parser.add_argument(
        "--pretrain_encoder_weights",
        "--pretrain-encoder-weights",
        dest="pretrain_encoder_weights",
        choices=("ema", "online"),
        default=config.get("pretrain_encoder_weights", "ema"),
        help="Choose EMA target or online context encoder weights from the checkpoint.",
    )
    parser.add_argument(
        "--ratio_supervision", type=float, default=config["ratio_supervision"]
    )
    parser.add_argument(
        "--patch_size",
        "--patch-size",
        dest="patch_size",
        type=int,
        default=config["patch_size"],
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=config["context_size"],
        help="Number of historical patches used for downstream forecasting.",
    )
    parser.add_argument(
        "--eval-stride",
        type=int,
        default=config["eval_stride"],
        help="Chronological stride between downstream evaluation samples.",
    )
    parser.add_argument(
        "--evaluation-split",
        choices=("validation", "test"),
        default=config.get("evaluation_split", "test"),
        help="Holdout split used only after downstream checkpoint selection.",
    )
    parser.add_argument(
        "--experiment-config-signature",
        default=config.get("experiment_config_signature"),
        help="Runner configuration signature recorded in metrics artifacts.",
    )
    parser.add_argument(
        "--mask-strategy",
        choices=("random", "local_long", "future_block", "causal_multiblock"),
        default=config.get("mask_strategy", "random"),
    )
    parser.add_argument(
        "--target_feature_index",
        "--target-feature-index",
        dest="target_feature_index",
        type=int,
        default=config.get("target_feature_index", 0),
    )
    parser.add_argument(
        "--forecast_target",
        "--forecast-target",
        dest="forecast_target",
        choices=(
            "value",
            "relative_return",
            "cumulative_log_return",
            "excess_log_return",
        ),
        default=config.get("forecast_target", "value"),
        help=(
            "Predict normalized target values or the future cumulative simple-return "
            "path P[t+h] / P[t] - 1."
        ),
    )
    parser.add_argument(
        "--forecast_horizon",
        "--forecast-horizon",
        dest="forecast_horizon",
        type=int,
        default=config.get("forecast_horizon"),
        help=(
            "Number of downstream target steps. Defaults to patch_size for "
            "backward compatibility."
        ),
    )
    parser.add_argument(
        "--normalization",
        choices=("window_return", "train_zscore", "train_robust_zscore", "none"),
        default=config.get("normalization", "window_return"),
    )
    parser.add_argument(
        "--feature_transform",
        "--feature-transform",
        dest="feature_transform",
        choices=("raw", "return"),
        default=config.get("feature_transform", "raw"),
    )
    parser.add_argument(
        "--sentiment-normalization",
        choices=("none", "train_zscore"),
        default=config.get("sentiment_normalization", "none"),
    )
    parser.add_argument(
        "--market_data",
        "--market-data",
        dest="market_data",
        default=config.get("market_data", None),
        help="Optional market ticker or CSV path used for aligned market returns.",
    )
    parser.add_argument(
        "--robust_zscore_clip",
        "--robust-zscore-clip",
        dest="robust_zscore_clip",
        type=float,
        default=config.get("robust_zscore_clip", None),
    )
    parser.add_argument(
        "--sampling_mode",
        "--sampling-mode",
        dest="sampling_mode",
        choices=("sliding_window", "temporal_segments"),
        default=config.get("sampling_mode", "sliding_window"),
    )
    parser.add_argument(
        "--normalization_stats_json",
        "--normalization-stats-json",
        dest="normalization_stats_json",
        default=None,
        help="Serialized train-only normalization state stored by pretraining.",
    )
    parser.add_argument(
        "--sentiment-normalization-stats-json",
        default=None,
        help="Serialized train-only state for derived sentiment channels.",
    )
    parser.add_argument(
        "--feature_cols",
        "--feature-cols",
        dest="feature_cols",
        nargs="+",
        default=None,
        help="Compatibility override for the final effective feature list.",
    )
    parser.add_argument(
        "--market-features",
        nargs="+",
        default=None,
        help="Market feature names used to construct the historical input.",
    )
    parser.add_argument(
        "--sentiment-features",
        nargs="+",
        default=None,
        help="Sentiment/news feature names enabled by --use-sentiment.",
    )
    sentiment_group = parser.add_mutually_exclusive_group()
    sentiment_group.add_argument(
        "--use-sentiment",
        dest="use_sentiment",
        action="store_true",
        help="Include configured sentiment/news features.",
    )
    sentiment_group.add_argument(
        "--no-sentiment",
        dest="use_sentiment",
        action="store_false",
        help="Use market features only; no sentiment file is read.",
    )
    parser.set_defaults(use_sentiment=None)
    parser.add_argument(
        "--timestamp_col",
        "--timestamp-col",
        dest="timestamp_col",
        default=config["timestamp_col"],
    )
    parser.add_argument(
        "--sentiment_path",
        "--sentiment-path",
        dest="sentiment_path",
        default=config.get("sentiment_path", None),
    )
    parser.add_argument(
        "--target-col",
        default=config["target_col"],
        help="Raw target column used to construct downstream labels.",
    )
    parser.add_argument(
        "--train_end_date",
        "--train-end-date",
        dest="train_end_date",
        default=config.get("train_end_date", None),
    )
    parser.add_argument(
        "--test_start_date",
        "--test-start-date",
        dest="test_start_date",
        default=config.get("test_start_date", None),
    )
    parser.add_argument(
        "--data_end_date",
        "--data-end-date",
        dest="data_end_date",
        default=config.get("data_end_date", None),
        help="Inclusive maximum timestamp allowed in training and evaluation data.",
    )
    parser.add_argument(
        "--validation_fraction",
        "--validation-fraction",
        dest="validation_fraction",
        type=float,
        default=config.get("validation_fraction", 0.05),
    )

    # Transformers parameters
    parser.add_argument("--embed_dim", type=int, default=config["embed_dim"])
    parser.add_argument("--nhead", type=int, default=config["nhead"])
    parser.add_argument("--num_layers", type=int, default=config["num_layers"])
    parser.add_argument("--kernel_size", type=int, default=config["kernel_size"])
    parser.add_argument(
        "--transformer_dense_dim", type=int, default=config["transformer_dense_dim"]
    )

    # CNN parameters
    parser.add_argument(
        "--cnn_out_channels", type=int, default=config["cnn_out_channels"]
    )
    parser.add_argument(
        "--cnn_kernel_size", type=int, default=config["cnn_kernel_size"]
    )
    parser.add_argument("--cnn_dense_dim", type=int, default=config["cnn_dense_dim"])

    # Pretrain parameters
    parser.add_argument(
        "--pretrain_encoder_embed_dim",
        type=int,
        default=config["pretrain_encoder_embed_dim"],
    )
    parser.add_argument(
        "--pretrain_encoder_nhead", type=int, default=config["pretrain_encoder_nhead"]
    )
    parser.add_argument(
        "--pretrain_encoder_num_layers",
        type=int,
        default=config["pretrain_encoder_num_layers"],
    )
    parser.add_argument(
        "--pretrain_encoder_kernel_size",
        type=int,
        default=config["pretrain_encoder_kernel_size"],
    )
    parser.add_argument(
        "--pretrain_encoder_embed_bias",
        "--pretrain-encoder-embed-bias",
        dest="pretrain_encoder_embed_bias",
        action="store_true",
        default=config.get("pretrain_encoder_embed_bias", True),
    )
    parser.add_argument(
        "--no-pretrain-encoder-embed-bias",
        dest="pretrain_encoder_embed_bias",
        action="store_false",
    )
    parser.add_argument(
        "--pretrain_transformer_dense_dim",
        type=int,
        default=config["pretrain_transformer_dense_dim"],
    )
    parser.add_argument("--ema_pretrain", type=float, default=config["ema_pretrain"])

    parser.add_argument(
        "--pretrain_decoder_embed_dim",
        type=int,
        default=config["pretrain_decoder_embed_dim"],
    )
    parser.add_argument(
        "--pretrain_decoder_nhead", type=int, default=config["pretrain_decoder_nhead"]
    )
    parser.add_argument(
        "--pretrain_decoder_num_layers",
        type=int,
        default=config["pretrain_decoder_num_layers"],
    )

    args = parser.parse_args()
    seed = int(args.seed)
    set_seed(seed)

    config["model"] = args.name_model.lower()
    config["data"] = args.data
    config["lr"] = args.lr
    config["batch_size"] = args.batch_size
    config["num_epochs"] = args.num_epochs
    config["lr_pretrain"] = args.lr_pretrain
    config["seed"] = seed
    config["ratio_patches"] = args.ratio_patches
    config["checkpoint_to_use"] = args.checkpoint_to_use
    config["pretrain_checkpoint_path"] = args.pretrain_checkpoint_path
    config["checkpoint_selection"] = args.checkpoint_selection
    config["pretrain_encoder_weights"] = args.pretrain_encoder_weights
    config["results_dir"] = args.results_dir

    config["path_data"] = "./data/" + args.data + "/" + args.data + ".csv"

    config["mask_ratio"] = args.mask_ratio
    config["pre_train_mask"] = args.pre_train_mask
    config["ratio_supervision"] = args.ratio_supervision
    config["pooling"] = args.pooling
    config["patch_size"] = args.patch_size
    config["forecast_horizon"] = resolve_forecast_horizon(
        args.forecast_horizon,
        args.patch_size,
    )
    config["context_size"] = args.context_size
    config["eval_stride"] = args.eval_stride
    config["evaluation_split"] = args.evaluation_split
    config["experiment_config_signature"] = args.experiment_config_signature
    config["mask_strategy"] = args.mask_strategy
    config["target_feature_index"] = args.target_feature_index
    config["target_col"] = args.target_col
    config["forecast_target"] = args.forecast_target
    config["normalization"] = args.normalization
    config["feature_transform"] = args.feature_transform
    config["sentiment_normalization"] = args.sentiment_normalization
    config["market_data"] = none_if_requested(args.market_data)
    config["robust_zscore_clip"] = args.robust_zscore_clip
    config["sampling_mode"] = args.sampling_mode
    config["normalization_stats"] = (
        json.loads(args.normalization_stats_json)
        if args.normalization_stats_json is not None
        else config.get("normalization_stats")
    )
    config["sentiment_normalization_stats"] = (
        json.loads(args.sentiment_normalization_stats_json)
        if args.sentiment_normalization_stats_json is not None
        else config.get("sentiment_normalization_stats")
    )
    config.update(
        resolve_feature_selection(
            config,
            feature_cols=args.feature_cols,
            market_features=args.market_features,
            sentiment_features=args.sentiment_features,
            use_sentiment=args.use_sentiment,
        )
    )
    config["timestamp_col"] = args.timestamp_col
    config["sentiment_path"] = none_if_requested(args.sentiment_path)
    config["train_end_date"] = none_if_requested(args.train_end_date)
    config["test_start_date"] = none_if_requested(args.test_start_date)
    config["data_end_date"] = none_if_requested(args.data_end_date)
    config["validation_fraction"] = args.validation_fraction
    validate_data_config(config, stage="downstream")

    config["pretrain_transformer_dense_dim"] = args.transformer_dense_dim

    # Transformer parameters
    config["embed_dim"] = args.embed_dim
    config["nhead"] = args.nhead
    config["num_layers"] = args.num_layers
    config["kernel_size"] = args.kernel_size
    config["transformer_dense_dim"] = args.transformer_dense_dim

    # CNN parameters
    config["cnn_out_channels"] = args.cnn_out_channels
    config["cnn_kernel_size"] = args.cnn_kernel_size
    config["cnn_dense_dim"] = args.cnn_dense_dim

    # Pretrained encoder
    config["pretrain_encoder_embed_dim"] = args.pretrain_encoder_embed_dim
    config["pretrain_encoder_nhead"] = args.pretrain_encoder_nhead
    config["pretrain_encoder_num_layers"] = args.pretrain_encoder_num_layers
    config["pretrain_encoder_kernel_size"] = args.pretrain_encoder_kernel_size
    config["pretrain_encoder_embed_bias"] = args.pretrain_encoder_embed_bias
    config["pretrain_transformer_dense_dim"] = args.pretrain_transformer_dense_dim

    config["ema_pretrain"] = args.ema_pretrain

    # Pretrained decoder
    config["pretrain_decoder_embed_dim"] = args.pretrain_decoder_embed_dim
    config["pretrain_decoder_nhead"] = args.pretrain_decoder_nhead
    config["pretrain_decoder_num_layers"] = args.pretrain_decoder_num_layers

    config["notes"] = args.notes

    config["path_save"] = "./logs/output_model/" + args.data

    return config


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)


def _reduce(metric, reduction="mean", axis=None):
    if reduction == "mean":
        return np.nanmean(metric, axis=axis)
    elif reduction == "sum":
        return np.nansum(metric, axis=axis)
    elif reduction == "none":
        return metric


def mse(
    y: np.ndarray,
    y_hat: np.ndarray,
    reduction: str = "mean",
    axis=None,
):
    delta_y = np.square(y - y_hat)
    return _reduce(delta_y, reduction=reduction, axis=axis)


def mae(y: np.ndarray, y_hat: np.ndarray, reduction: str = "mean", axis=None):
    delta_y = np.abs(y - y_hat)
    return _reduce(delta_y, reduction=reduction, axis=axis)


if __name__ == "__main__":
    pass
