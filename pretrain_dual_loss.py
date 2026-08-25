"""
Pretrain TS-JEPA with unified JEPA + MAE reconstruction objectives.

The predictor produces target embeddings for masked patches:
    z_hat_T = P(E_context(x_C))

The reconstruction decoder maps predicted embeddings back to patch vectors:
    x_hat_T = D(z_hat_T)

Four masking strategies are supported:
    random:     JEPA and MAE use the same randomly masked patches.
    local_long: MAE reconstructs a local window while JEPA predicts a farther
                latent window from causal context.
    future_block: use past context to predict one contiguous future block.
    causal_multiblock: use past context to predict multiple future blocks.
"""

import argparse
import copy
import hashlib
import json
import os
import random
import runpy
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from config.config_pretrain import config as base_config
from config.experiment import (
    none_if_requested,
    resolve_feature_selection,
    validate_data_config,
)
from main.utils import init_weights, ordered_scalar_mean, set_seed
from src.data_loaders.data_loader_mnist_rows import get_mnist_row_loader
from src.data_loaders.data_loader_roll_volume import get_jepa_loaders
from src.models.decoder import ResidualMLPDecoder, build_reconstruction_decoder
from src.models.encoder import Encoder
from src.models.predictor import Predictor
from src.models.utils.mask_utils import apply_mask


def _float_for_path(value):
    return str(value).replace("/", "_")


EXPERIMENT_ID_KEYS = (
    "input_mode",
    "mnist_train_samples",
    "mnist_val_samples",
    "mask_strategy",
    "seed",
    "deterministic",
    "series_split_size",
    "patch_size",
    "pretrain_stride",
    "sampling_mode",
    "feature_transform",
    "normalization",
    "sentiment_normalization",
    "robust_zscore_clip",
    "market_data",
    "feature_cols",
    "timestamp_col",
    "sentiment_path",
    "train_end_date",
    "test_start_date",
    "data_end_date",
    "validation_fraction",
    "batch_size",
    "mask_ratio",
    "end_lr",
    "clip_grad",
    "ipe_scale",
    "lambda_jepa",
    "lambda_mae",
    "jepa_loss",
    "mae_loss",
    "encoder_embed_dim",
    "encoder_nhead",
    "encoder_num_layers",
    "encoder_kernel_size",
    "encoder_embed_bias",
    "predictor_embed",
    "predictor_nhead",
    "predictor_num_layers",
    "decoder_type",
    "decoder_hidden_dim",
    "decoder_num_layers",
    "decoder_dropout",
    "mae_window_patches",
    "jepa_gap_patches",
    "jepa_target_patches",
    "anchor_strategy",
    "fixed_anchor",
    "future_target_patches",
    "causal_num_blocks",
    "causal_block_patches",
    "causal_block_gap_patches",
)


def experiment_fingerprint(config):
    """Stable identifier preventing incompatible runs from sharing a filename."""
    identity = {key: config.get(key) for key in EXPERIMENT_ID_KEYS}
    # Preserve the historical fingerprint for the backward-compatible raw
    # defaults, while distinguishing every newly enabled preprocessing choice.
    if identity.get("feature_transform") in (None, "raw"):
        identity.pop("feature_transform", None)
    if identity.get("market_data") is None:
        identity.pop("market_data", None)
    if identity.get("robust_zscore_clip") is None:
        identity.pop("robust_zscore_clip", None)
    if identity.get("sentiment_normalization") in (None, "none"):
        identity.pop("sentiment_normalization", None)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def build_pretrain_path(config):
    path_save = (
        "./logs/output_model/"
        + config["data"]
        + "/lr_"
        + str(config["lr"])
        + "_ema_momentum_"
        + str(config["ema_momentum"])
        + "_mask_ratio_"
        + str(config["mask_ratio"])
        + "_ratio_patches_"
        + str(config["ratio_patches"])
        + "_encoder_"
        + str(config["encoder_embed_dim"])
        + "_"
        + str(config["encoder_nhead"])
        + "_"
        + str(config["encoder_num_layers"])
        + "_predictor_"
        + str(config["predictor_embed"])
        + "_"
        + str(config["predictor_nhead"])
        + "_"
        + str(config["predictor_num_layers"])
    )

    return path_save + config.get("path_suffix", "")


def parse_args(config, default_mask_strategy=None, argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Unified dual-loss TS-JEPA pretraining: "
            "lambda_jepa * JEPA + lambda_mae * MAE."
        )
    )

    parser.add_argument("--data", type=str, default=config["data"])
    parser.add_argument(
        "--input-mode",
        choices=("timeseries", "mnist_rows"),
        default=config.get("input_mode", "timeseries"),
        help="Use stock-style CSV windows or 28 MNIST image rows as tokens.",
    )
    parser.add_argument(
        "--mnist-root",
        default=config.get("mnist_root", "./data/MNIST"),
        help="MNIST cache directory used with --input-mode mnist_rows.",
    )
    parser.add_argument(
        "--mnist-train-samples",
        type=int,
        default=config.get("mnist_train_samples", 512),
        help="Number of MNIST training images used in row mode.",
    )
    parser.add_argument(
        "--mnist-val-samples",
        type=int,
        default=config.get("mnist_val_samples", 128),
        help="Disjoint MNIST training images reserved for deterministic validation.",
    )
    parser.add_argument(
        "--download-mnist",
        action="store_true",
        default=config.get("download_mnist", False),
        help="Download MNIST if it is absent from --mnist-root.",
    )
    parser.add_argument(
        "--mask-strategy",
        choices=("random", "local_long", "future_block", "causal_multiblock"),
        default=(
            config.get("mask_strategy", "random")
            if default_mask_strategy is None
            else default_mask_strategy
        ),
        help=(
            "Choose random targets, local-MAE plus long-JEPA, one future "
            "block, or multiple causal future blocks."
        ),
    )
    parser.add_argument(
        "--mask_ratio",
        "--mask-ratio",
        dest="mask_ratio",
        type=float,
        default=config["mask_ratio"],
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=config["batch_size"],
    )
    parser.add_argument("--lr", type=float, default=config["lr"])
    parser.add_argument(
        "--end_lr",
        "--end-lr",
        dest="end_lr",
        type=float,
        default=config.get("end_lr", config["lr"] * 0.5),
    )
    parser.add_argument(
        "--num_epochs",
        "--num-epochs",
        dest="num_epochs",
        type=int,
        default=config["num_epochs"],
    )
    parser.add_argument(
        "--ema_momentum",
        "--ema-momentum",
        dest="ema_momentum",
        type=float,
        default=config["ema_momentum"],
    )
    parser.add_argument(
        "--ratio_patches",
        "--ratio-patches",
        dest="ratio_patches",
        type=int,
        default=config["ratio_patches"],
        help=(
            "Legacy checkpoint-naming field retained for compatibility; "
            "--mask-ratio controls the number of random masked patches."
        ),
    )
    parser.add_argument(
        "--checkpoint_save",
        "--checkpoint-save",
        dest="checkpoint_save",
        type=int,
        default=config["checkpoint_save"],
    )
    parser.add_argument(
        "--checkpoint_print",
        "--checkpoint-print",
        dest="checkpoint_print",
        type=int,
        default=config["checkpoint_print"],
    )
    parser.add_argument(
        "--clip_grad",
        "--clip-grad",
        dest="clip_grad",
        type=float,
        default=config.get("clip_grad", 1.0),
    )
    parser.add_argument(
        "--ipe_scale",
        "--ipe-scale",
        dest="ipe_scale",
        type=float,
        default=config.get("ipe_scale", 1.25),
    )
    parser.add_argument("--notes", type=str, default=config.get("notes", ""))
    parser.add_argument(
        "--seed",
        type=int,
        default=config.get("seed", 42),
        help="Random seed. Defaults to the reproducible config value (42).",
    )
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=config.get("deterministic", True),
        help="Request deterministic PyTorch kernels (default).",
    )
    parser.add_argument(
        "--no-deterministic",
        dest="deterministic",
        action="store_false",
        help="Allow nondeterministic kernels for maximum throughput.",
    )

    parser.add_argument(
        "--series_split_size",
        "--series-split-size",
        dest="series_split_size",
        type=int,
        default=config["series_split_size"],
    )
    parser.add_argument(
        "--patch_size",
        "--patch-size",
        dest="patch_size",
        type=int,
        default=config["patch_size"],
    )
    parser.add_argument(
        "--pretrain_stride",
        "--pretrain-stride",
        dest="pretrain_stride",
        type=int,
        default=config.get("pretrain_stride", None),
        help="Sliding-window stride. Defaults to --patch-size.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("sliding_window", "temporal_segments"),
        default=config.get("sampling_mode", "sliding_window"),
        help=(
            "Build overlapping sliding windows or contiguous non-overlapping "
            "temporal segments."
        ),
    )
    parser.add_argument(
        "--normalization",
        choices=("window_return", "train_zscore", "train_robust_zscore", "none"),
        default=config.get("normalization", "window_return"),
    )
    parser.add_argument(
        "--feature-transform",
        choices=("raw", "return"),
        default=config.get("feature_transform", "raw"),
        help="Use backward-compatible raw features or causal return features.",
    )
    parser.add_argument(
        "--market-data",
        default=config.get("market_data", None),
        help="Optional market ticker (for example NASDAQ100) or CSV path.",
    )
    parser.add_argument(
        "--robust-zscore-clip",
        type=float,
        default=config.get("robust_zscore_clip", None),
        help="Optional symmetric clipping after train-only robust scaling.",
    )
    parser.add_argument(
        "--target_feature_index",
        "--target-feature-index",
        dest="target_feature_index",
        type=int,
        default=config.get("target_feature_index", 0),
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
        help="Market feature names used to construct historical inputs.",
    )
    parser.add_argument(
        "--sentiment-features",
        nargs="+",
        default=None,
        help="Sentiment/news features enabled by --use-sentiment.",
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
        type=str,
        default=config["timestamp_col"],
    )
    parser.add_argument(
        "--sentiment_path",
        "--sentiment-path",
        dest="sentiment_path",
        type=str,
        default=config.get("sentiment_path", None),
    )
    parser.add_argument(
        "--sentiment-normalization",
        choices=("none", "train_zscore"),
        default=config.get("sentiment_normalization", "none"),
    )
    parser.add_argument(
        "--train_end_date",
        "--train-end-date",
        dest="train_end_date",
        type=str,
        default=config.get("train_end_date", None),
    )
    parser.add_argument(
        "--test_start_date",
        "--test-start-date",
        dest="test_start_date",
        type=str,
        default=config.get("test_start_date", None),
    )
    parser.add_argument(
        "--data_end_date",
        "--data-end-date",
        dest="data_end_date",
        type=str,
        default=config.get("data_end_date", None),
        help="Inclusive maximum timestamp allowed in pretraining and evaluation data.",
    )
    parser.add_argument(
        "--validation_fraction",
        "--validation-fraction",
        dest="validation_fraction",
        type=float,
        default=config.get("validation_fraction", 0.05),
    )

    parser.add_argument(
        "--encoder_embed_dim",
        "--encoder-embed-dim",
        dest="encoder_embed_dim",
        type=int,
        default=config["encoder_embed_dim"],
    )
    parser.add_argument(
        "--encoder_nhead",
        "--encoder-nhead",
        dest="encoder_nhead",
        type=int,
        default=config["encoder_nhead"],
    )
    parser.add_argument(
        "--encoder_num_layers",
        "--encoder-num-layers",
        dest="encoder_num_layers",
        type=int,
        default=config["encoder_num_layers"],
    )
    parser.add_argument(
        "--encoder_kernel_size",
        "--encoder-kernel-size",
        dest="encoder_kernel_size",
        type=int,
        default=config["encoder_kernel_size"],
    )
    parser.add_argument(
        "--encoder_embed_bias",
        "--encoder-embed-bias",
        dest="encoder_embed_bias",
        action="store_true",
        default=config.get("encoder_embed_bias", True),
    )
    parser.add_argument(
        "--no-encoder-embed-bias",
        dest="encoder_embed_bias",
        action="store_false",
        help="Disable bias in the encoder tokenizer embedding layer.",
    )

    parser.add_argument(
        "--predictor_embed",
        "--predictor-embed",
        dest="predictor_embed",
        type=int,
        default=config["predictor_embed"],
    )
    parser.add_argument(
        "--predictor_nhead",
        "--predictor-nhead",
        dest="predictor_nhead",
        type=int,
        default=config["predictor_nhead"],
    )
    parser.add_argument(
        "--predictor_num_layers",
        "--predictor-num-layers",
        dest="predictor_num_layers",
        type=int,
        default=config["predictor_num_layers"],
    )

    parser.add_argument(
        "--lambda_jepa",
        "--lambda-jepa",
        dest="lambda_jepa",
        type=float,
        default=config.get("lambda_jepa", 1.0),
    )
    parser.add_argument(
        "--lambda_mae",
        "--lambda-mae",
        dest="lambda_mae",
        type=float,
        default=config.get("lambda_mae", 1.0),
    )
    parser.add_argument(
        "--jepa_loss",
        "--jepa-loss",
        dest="jepa_loss",
        choices=("mse", "l1", "smooth_l1"),
        default=config.get("jepa_loss", "mse"),
    )
    parser.add_argument(
        "--mae_loss",
        "--mae-loss",
        dest="mae_loss",
        choices=("mse", "l1", "smooth_l1"),
        default=config.get("mae_loss", "mse"),
    )
    parser.add_argument(
        "--decoder_type",
        "--decoder-type",
        dest="decoder_type",
        choices=("linear", "mlp", "residual_mlp"),
        default=config.get(
            "decoder_type",
            config.get("mae_decoder_type", "residual_mlp"),
        ),
    )
    parser.add_argument(
        "--decoder_hidden_dim",
        "--decoder-hidden-dim",
        dest="decoder_hidden_dim",
        type=int,
        default=config.get(
            "decoder_hidden_dim",
            config.get("mae_decoder_hidden_dim", 256),
        ),
    )
    parser.add_argument(
        "--decoder_num_layers",
        "--decoder-num-layers",
        dest="decoder_num_layers",
        type=int,
        default=config.get(
            "decoder_num_layers",
            config.get("mae_decoder_num_layers", 2),
        ),
    )
    parser.add_argument(
        "--decoder_dropout",
        "--decoder-dropout",
        dest="decoder_dropout",
        type=float,
        default=config.get(
            "decoder_dropout",
            config.get("mae_decoder_dropout", 0.1),
        ),
    )

    parser.add_argument(
        "--mae_window_patches",
        "--mae-window-patches",
        dest="mae_window_patches",
        type=int,
        default=config.get("mae_window_patches", 1),
        help="Number of local patches reconstructed by MAE in local_long mode.",
    )
    parser.add_argument(
        "--jepa_gap_patches",
        "--jepa-gap-patches",
        dest="jepa_gap_patches",
        type=int,
        default=config.get("jepa_gap_patches", 4),
        help="Offset from the local MAE anchor to the JEPA target window.",
    )
    parser.add_argument(
        "--jepa_target_patches",
        "--jepa-target-patches",
        dest="jepa_target_patches",
        type=int,
        default=config.get("jepa_target_patches", 4),
        help="Number of farther latent patches predicted in local_long mode.",
    )
    parser.add_argument(
        "--anchor_strategy",
        "--anchor-strategy",
        dest="anchor_strategy",
        choices=("random", "fixed"),
        default=config.get("anchor_strategy", "random"),
    )
    parser.add_argument(
        "--fixed_anchor",
        "--fixed-anchor",
        dest="fixed_anchor",
        type=int,
        default=config.get("fixed_anchor", 0),
    )
    parser.add_argument(
        "--future-target-patches",
        type=int,
        default=config.get("future_target_patches", 4),
    )
    parser.add_argument(
        "--causal-num-blocks",
        type=int,
        default=config.get("causal_num_blocks", 2),
    )
    parser.add_argument(
        "--causal-block-patches",
        type=int,
        default=config.get("causal_block_patches", 2),
    )
    parser.add_argument(
        "--causal-block-gap-patches",
        type=int,
        default=config.get("causal_block_gap_patches", 1),
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=config.get("validation_interval", 10),
    )
    parser.add_argument(
        "--validation-max-batches",
        type=int,
        default=config.get("validation_max_batches", None),
    )
    parser.add_argument(
        "--resume-from",
        default=config.get("resume_from", None),
        help="Resume complete training state from a unified checkpoint.",
    )

    parser.add_argument(
        "--max_batches_per_epoch",
        "--max-batches-per-epoch",
        dest="max_batches_per_epoch",
        type=int,
        default=config.get("max_batches_per_epoch", None),
    )
    parser.add_argument(
        "--no_save_final",
        "--no-save-final",
        dest="save_final",
        action="store_false",
        default=config.get("save_final", True),
    )
    parser.add_argument(
        "--path_suffix",
        "--path-suffix",
        dest="path_suffix",
        type=str,
        default=config.get("path_suffix", None),
    )
    parser.add_argument(
        "--compatible_save_name",
        "--compatible-save-name",
        dest="compatible_save_name",
        action="store_true",
        default=config.get("compatible_save_name", False),
        help="Save without a strategy suffix for compatibility with legacy tooling.",
    )
    parser.add_argument(
        "--run_eval",
        "--run-eval",
        dest="run_eval",
        action="store_true",
        default=config.get("run_eval", False),
    )
    parser.add_argument(
        "--no-run-eval",
        dest="run_eval",
        action="store_false",
        help="Disable the config-driven downstream evaluation stage.",
    )
    parser.add_argument(
        "--eval_checkpoint_to_use",
        "--eval-checkpoint-to-use",
        dest="eval_checkpoint_to_use",
        type=int,
        default=config.get("eval_checkpoint_to_use", None),
    )
    parser.add_argument(
        "--eval_num_epochs",
        "--eval-num-epochs",
        dest="eval_num_epochs",
        type=int,
        default=config.get("eval_num_epochs", None),
    )
    parser.add_argument(
        "--eval-use-best",
        dest="eval_use_best",
        action="store_true",
        default=config.get("eval_use_best", False),
        help="Run downstream evaluation from the best validation checkpoint.",
    )
    parser.add_argument(
        "--no-eval-use-best",
        dest="eval_use_best",
        action="store_false",
        help="Use --eval-checkpoint-to-use or the last saved checkpoint.",
    )
    parser.add_argument(
        "--eval-encoder-weights",
        choices=("ema", "online"),
        default=config.get("eval_encoder_weights", "ema"),
        help="Encoder weights used by automatic downstream evaluation.",
    )
    parser.add_argument(
        "--eval-forecast-target",
        choices=(
            "value",
            "relative_return",
            "cumulative_log_return",
            "excess_log_return",
        ),
        default=config.get("eval_forecast_target", "value"),
        help=(
            "Downstream target used by --run-eval. relative_return predicts "
            "P[t+h] / P[t] - 1."
        ),
    )
    parser.add_argument(
        "--eval-forecast-horizon",
        type=int,
        default=config.get("eval_forecast_horizon"),
        help="Optional downstream target width used by --run-eval.",
    )
    parser.add_argument(
        "--eval-results-dir",
        default=config.get("eval_results_dir", None),
        help="Optional output directory for automatic downstream evaluation.",
    )

    args = parser.parse_args(argv)
    cfg = copy.deepcopy(config)

    for key, value in vars(args).items():
        cfg[key] = value

    cfg.update(
        resolve_feature_selection(
            config,
            feature_cols=args.feature_cols,
            market_features=args.market_features,
            sentiment_features=args.sentiment_features,
            use_sentiment=args.use_sentiment,
        )
    )
    cfg["sentiment_path"] = none_if_requested(args.sentiment_path)
    cfg["market_data"] = none_if_requested(args.market_data)
    cfg["train_end_date"] = none_if_requested(args.train_end_date)
    cfg["test_start_date"] = none_if_requested(args.test_start_date)
    cfg["data_end_date"] = none_if_requested(args.data_end_date)
    cfg["path_data"] = "./data/" + args.data + "/" + args.data + ".csv"

    if args.sampling_mode == "temporal_segments":
        cfg["pretrain_stride"] = args.series_split_size
    else:
        cfg["pretrain_stride"] = (
            args.patch_size if args.pretrain_stride is None else args.pretrain_stride
        )

    validate_data_config(cfg, stage="pretrain")

    seed = int(args.seed)
    set_seed(seed, deterministic=args.deterministic)
    cfg["seed"] = seed

    positive_args = (
        ("batch-size", args.batch_size),
        ("num-epochs", args.num_epochs),
        ("checkpoint-save", args.checkpoint_save),
        ("checkpoint-print", args.checkpoint_print),
        ("ratio-patches", args.ratio_patches),
        ("pretrain-stride", cfg["pretrain_stride"]),
        ("validation-interval", args.validation_interval),
        ("future-target-patches", args.future_target_patches),
        ("causal-num-blocks", args.causal_num_blocks),
        ("causal-block-patches", args.causal_block_patches),
        ("mnist-val-samples", args.mnist_val_samples),
    )
    for name, value in positive_args:
        if int(value) <= 0:
            raise ValueError(f"--{name} must be positive, got {value}")
    if args.max_batches_per_epoch is not None and args.max_batches_per_epoch <= 0:
        raise ValueError("--max-batches-per-epoch must be positive when set")
    if args.validation_max_batches is not None and args.validation_max_batches <= 0:
        raise ValueError("--validation-max-batches must be positive when set")
    if args.causal_block_gap_patches < 0:
        raise ValueError("--causal-block-gap-patches must be non-negative")
    if not 0.0 < float(args.mask_ratio) < 1.0:
        raise ValueError("--mask-ratio must be strictly between 0 and 1")
    if float(args.lr) <= 0:
        raise ValueError("--lr must be positive")
    if not 0 < float(args.end_lr) <= float(args.lr):
        raise ValueError("--end-lr must be positive and no greater than --lr")
    if not 0.0 <= float(args.ema_momentum) < 1.0:
        raise ValueError("--ema-momentum must be in [0, 1)")
    if float(args.ipe_scale) <= 0:
        raise ValueError("--ipe-scale must be positive")
    if float(args.lambda_jepa) < 0 or float(args.lambda_mae) < 0:
        raise ValueError("--lambda-jepa and --lambda-mae must be non-negative")
    if float(args.lambda_jepa) == 0 and float(args.lambda_mae) == 0:
        raise ValueError("At least one loss weight must be positive")

    if args.compatible_save_name:
        cfg["path_suffix"] = ""
    elif args.path_suffix is not None:
        cfg["path_suffix"] = args.path_suffix
    else:
        if args.mask_strategy == "local_long":
            cfg["path_suffix"] = (
                "_local_mae_long_jepa_ljepa_"
                + _float_for_path(args.lambda_jepa)
                + "_lmae_"
                + _float_for_path(args.lambda_mae)
                + "_mae_"
                + str(args.mae_window_patches)
                + "_gap_"
                + str(args.jepa_gap_patches)
                + "_jepa_"
                + str(args.jepa_target_patches)
            )
        elif args.mask_strategy == "future_block":
            cfg["path_suffix"] = (
                "_future_block_ljepa_"
                + _float_for_path(args.lambda_jepa)
                + "_lmae_"
                + _float_for_path(args.lambda_mae)
                + "_target_"
                + str(args.future_target_patches)
            )
        elif args.mask_strategy == "causal_multiblock":
            cfg["path_suffix"] = (
                "_causal_multiblock_ljepa_"
                + _float_for_path(args.lambda_jepa)
                + "_lmae_"
                + _float_for_path(args.lambda_mae)
                + "_blocks_"
                + str(args.causal_num_blocks)
                + "_size_"
                + str(args.causal_block_patches)
                + "_gap_"
                + str(args.causal_block_gap_patches)
            )
        else:
            cfg["path_suffix"] = (
                "_dual_jepa_mae_ljepa_"
                + _float_for_path(args.lambda_jepa)
                + "_lmae_"
                + _float_for_path(args.lambda_mae)
            )

    if not args.compatible_save_name:
        cfg["config_fingerprint"] = experiment_fingerprint(cfg)
        cfg["path_suffix"] += "_cfg_" + cfg["config_fingerprint"]

    cfg["path_save"] = build_pretrain_path(cfg)
    return cfg


def validate_strategy_config(config, num_patches):
    """Validate the configured structured mask and return its anchor range."""
    strategy = config.get("mask_strategy", "local_long")

    if strategy == "future_block":
        target_len = int(config["future_target_patches"])
        max_anchor = num_patches - target_len
        min_anchor = 1
        if max_anchor < min_anchor:
            raise ValueError(
                "future_block requires at least one context patch before its target: "
                f"num_patches={num_patches}, target_patches={target_len}"
            )
        if config["anchor_strategy"] == "fixed" and not (
            min_anchor <= int(config["fixed_anchor"]) <= max_anchor
        ):
            raise ValueError(
                f"--fixed-anchor must be in [{min_anchor}, {max_anchor}] "
                "for future_block"
            )
        return min_anchor, max_anchor

    if strategy == "causal_multiblock":
        num_blocks = int(config["causal_num_blocks"])
        block_len = int(config["causal_block_patches"])
        gap = int(config["causal_block_gap_patches"])
        target_span = num_blocks * block_len + (num_blocks - 1) * gap
        max_anchor = num_patches - target_span
        min_anchor = 1
        if max_anchor < min_anchor:
            raise ValueError(
                "causal_multiblock target geometry does not fit after a non-empty "
                f"context: num_patches={num_patches}, target_span={target_span}"
            )
        if config["anchor_strategy"] == "fixed" and not (
            min_anchor <= int(config["fixed_anchor"]) <= max_anchor
        ):
            raise ValueError(
                f"--fixed-anchor must be in [{min_anchor}, {max_anchor}] "
                "for causal_multiblock"
            )
        return min_anchor, max_anchor

    if strategy != "local_long":
        raise ValueError(f"No structured-mask validation for strategy={strategy!r}")

    mae_len = int(config["mae_window_patches"])
    gap = int(config["jepa_gap_patches"])
    jepa_len = int(config["jepa_target_patches"])

    if mae_len <= 0:
        raise ValueError("--mae-window-patches must be positive")
    if gap <= 0:
        raise ValueError("--jepa-gap-patches must be positive")
    if jepa_len <= 0:
        raise ValueError("--jepa-target-patches must be positive")
    if mae_len >= gap:
        raise ValueError(
            "--mae-window-patches must be smaller than --jepa-gap-patches "
            "so at least one visible context patch remains before the long target."
        )

    max_anchor = num_patches - gap - jepa_len
    if max_anchor < 0:
        raise ValueError(
            "Need at least gap + jepa_target patches. "
            f"num_patches={num_patches}, gap={gap}, jepa_target={jepa_len}"
        )

    if config["anchor_strategy"] == "fixed":
        fixed_anchor = int(config["fixed_anchor"])
        if fixed_anchor < 0 or fixed_anchor > max_anchor:
            raise ValueError(
                f"--fixed-anchor must be in [0, {max_anchor}], got {fixed_anchor}"
            )

    return 0, max_anchor


def make_strategy_masks(
    config,
    batch_size,
    num_patches,
    device,
    anchor_override=None,
):
    """Build one structured causal mask geometry for a complete batch.

    A common anchor keeps every row's context length equal, which is required by
    the current dense Transformer and mask representation. A new random anchor
    is sampled for every batch.
    """
    strategy = config.get("mask_strategy", "local_long")
    min_anchor, max_anchor = validate_strategy_config(config, num_patches)

    if anchor_override is not None:
        anchor = int(anchor_override)
    elif config["anchor_strategy"] == "random":
        anchor = int(
            torch.randint(min_anchor, max_anchor + 1, size=()).item()
        )
    else:
        anchor = int(config["fixed_anchor"])

    if anchor < min_anchor or anchor > max_anchor:
        raise ValueError(
            f"anchor must be in [{min_anchor}, {max_anchor}], got {anchor}"
        )

    if strategy == "future_block":
        target_len = int(config["future_target_patches"])
        target_indices = torch.arange(anchor, anchor + target_len, device=device)
        context_indices = torch.arange(0, anchor, device=device)
        mae_indices = target_indices
        jepa_indices = target_indices
        predict_indices = target_indices
    elif strategy == "causal_multiblock":
        num_blocks = int(config["causal_num_blocks"])
        block_len = int(config["causal_block_patches"])
        block_gap = int(config["causal_block_gap_patches"])
        blocks = [
            torch.arange(
                anchor + block_idx * (block_len + block_gap),
                anchor + block_idx * (block_len + block_gap) + block_len,
                device=device,
            )
            for block_idx in range(num_blocks)
        ]
        target_indices = torch.cat(blocks, dim=0)
        context_indices = torch.arange(0, anchor, device=device)
        mae_indices = target_indices
        jepa_indices = target_indices
        predict_indices = target_indices
    else:
        mae_len = int(config["mae_window_patches"])
        gap = int(config["jepa_gap_patches"])
        jepa_len = int(config["jepa_target_patches"])

        mae_start = anchor
        jepa_start = anchor + gap
        mae_indices = torch.arange(mae_start, mae_start + mae_len, device=device)
        jepa_indices = torch.arange(jepa_start, jepa_start + jepa_len, device=device)
        predict_indices = torch.cat([mae_indices, jepa_indices], dim=0)

        context_indices = torch.cat(
            [
                torch.arange(0, mae_start, device=device),
                torch.arange(mae_start + mae_len, jepa_start, device=device),
            ]
        )
    if context_indices.numel() == 0:
        raise ValueError("Strategy created an empty context mask")

    def repeat(indices):
        return indices.unsqueeze(0).expand(batch_size, -1)

    return {
        "anchor": anchor,
        "mae": repeat(mae_indices),
        "jepa": repeat(jepa_indices),
        "predict": repeat(predict_indices),
        "context": repeat(context_indices),
    }


def validate_training_config(config, num_patches, patch_dim, loader_length):
    """Fail early with actionable errors instead of failing inside a model."""
    positive_ints = (
        ("batch-size", config["batch_size"]),
        ("num-epochs", config["num_epochs"]),
        ("checkpoint-save", config["checkpoint_save"]),
        ("checkpoint-print", config["checkpoint_print"]),
        ("ratio-patches", config["ratio_patches"]),
        ("encoder-embed-dim", config["encoder_embed_dim"]),
        ("encoder-nhead", config["encoder_nhead"]),
        ("encoder-num-layers", config["encoder_num_layers"]),
        ("encoder-kernel-size", config["encoder_kernel_size"]),
        ("predictor-embed", config["predictor_embed"]),
        ("predictor-nhead", config["predictor_nhead"]),
        ("predictor-num-layers", config["predictor_num_layers"]),
    )
    for name, value in positive_ints:
        if int(value) <= 0:
            raise ValueError(f"--{name} must be positive, got {value}")

    if loader_length <= 0:
        raise ValueError("The training loader contains no batches")
    if config["max_batches_per_epoch"] is not None and int(
        config["max_batches_per_epoch"]
    ) <= 0:
        raise ValueError("--max-batches-per-epoch must be positive when set")
    if not 0.0 < float(config["mask_ratio"]) < 1.0:
        raise ValueError("--mask-ratio must be strictly between 0 and 1")
    if float(config["lr"]) <= 0:
        raise ValueError("--lr must be positive")
    if not 0 < float(config["end_lr"]) <= float(config["lr"]):
        raise ValueError("--end-lr must be positive and no greater than --lr")
    if not 0.0 <= float(config["ema_momentum"]) < 1.0:
        raise ValueError("--ema-momentum must be in [0, 1)")
    if float(config["ipe_scale"]) <= 0:
        raise ValueError("--ipe-scale must be positive")
    if float(config["lambda_jepa"]) < 0 or float(config["lambda_mae"]) < 0:
        raise ValueError("--lambda-jepa and --lambda-mae must be non-negative")
    if float(config["lambda_jepa"]) == 0 and float(config["lambda_mae"]) == 0:
        raise ValueError("At least one of --lambda-jepa or --lambda-mae must be positive")
    if not 0.0 <= float(config["decoder_dropout"]) < 1.0:
        raise ValueError("--decoder-dropout must be in [0, 1)")
    if int(config["decoder_hidden_dim"]) <= 0:
        raise ValueError("--decoder-hidden-dim must be positive")
    if int(config["decoder_num_layers"]) <= 0:
        raise ValueError("--decoder-num-layers must be positive")
    if int(config["encoder_embed_dim"]) % 2 != 0:
        raise ValueError("--encoder-embed-dim must be even for sinusoidal positions")
    if int(config["predictor_embed"]) % 2 != 0:
        raise ValueError("--predictor-embed must be even for sinusoidal positions")
    if int(config["encoder_embed_dim"]) % int(config["encoder_nhead"]) != 0:
        raise ValueError("--encoder-embed-dim must be divisible by --encoder-nhead")
    if int(config["predictor_embed"]) % int(config["predictor_nhead"]) != 0:
        raise ValueError("--predictor-embed must be divisible by --predictor-nhead")
    if int(config["encoder_kernel_size"]) > patch_dim:
        raise ValueError(
            "--encoder-kernel-size cannot exceed the flattened patch dimension "
            f"({patch_dim})"
        )
    if num_patches < 2:
        raise ValueError(f"At least two patches are required, got {num_patches}")

    if config["mask_strategy"] != "random":
        if config["input_mode"] != "timeseries":
            raise ValueError("Structured causal masks require --input-mode timeseries")
        validate_strategy_config(config, num_patches)


def ema_momentum_at_step(base_momentum, step, total_steps):
    """Linearly increase EMA momentum from its base value toward one."""
    progress = min(max(step, 0), total_steps) / max(total_steps, 1)
    return float(base_momentum) + progress * (1.0 - float(base_momentum))


def loss_value(pred, target, kind):
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "l1":
        return F.l1_loss(pred, target)
    if kind == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    raise ValueError(f"Unknown loss kind: {kind}")


@torch.no_grad()
def evaluate_pretraining(
    encoder,
    predictor,
    decoder,
    encoder_ema,
    loader,
    device,
    config,
):
    """Evaluate deterministic masked objectives and simple collapse diagnostics."""
    encoder.eval()
    predictor.eval()
    decoder.eval()
    encoder_ema.eval()

    total_jepa = 0.0
    total_mae = 0.0
    num_batches = 0
    embedding_batches = []

    for batch_idx, (patches, dataset_masks, dataset_non_masks) in enumerate(loader):
        if (
            config.get("validation_max_batches") is not None
            and batch_idx >= int(config["validation_max_batches"])
        ):
            break

        patches = patches.to(device)
        if config["mask_strategy"] != "random":
            min_anchor, max_anchor = validate_strategy_config(
                config,
                patches.shape[1],
            )
            if config.get("anchor_strategy", "random") == "fixed":
                anchor = int(config["fixed_anchor"])
            else:
                anchor = min_anchor + (batch_idx % (max_anchor - min_anchor + 1))
            objective_masks = make_strategy_masks(
                config=config,
                batch_size=patches.size(0),
                num_patches=patches.shape[1],
                device=device,
                anchor_override=anchor,
            )
        else:
            dataset_masks = dataset_masks.to(device)
            dataset_non_masks = dataset_non_masks.to(device)
            objective_masks = {
                "anchor": None,
                "mae": dataset_masks,
                "jepa": dataset_masks,
                "predict": dataset_masks,
                "context": dataset_non_masks,
            }

        full_target_embeddings = encoder_ema(patches)
        full_target_embeddings = F.layer_norm(
            full_target_embeddings,
            (full_target_embeddings.size(-1),),
        )
        target_ema = apply_mask(full_target_embeddings, objective_masks["jepa"])
        target_patches = apply_mask(patches, objective_masks["mae"])
        context_tokens = encoder(patches, mask=objective_masks["context"])
        predicted = predictor(
            context_tokens,
            mask=objective_masks["predict"],
            non_masks=objective_masks["context"],
        )

        if config["mask_strategy"] == "local_long":
            mae_len = int(config["mae_window_patches"])
            pred_mae = predicted[:, :mae_len]
            pred_jepa = predicted[:, mae_len:]
        else:
            pred_mae = predicted
            pred_jepa = predicted

        reconstructed = decoder(pred_mae)
        total_jepa += loss_value(
            pred_jepa,
            target_ema,
            config["jepa_loss"],
        ).item()
        total_mae += loss_value(
            reconstructed,
            target_patches,
            config["mae_loss"],
        ).item()
        num_batches += 1

        if sum(batch.shape[0] for batch in embedding_batches) < 4096:
            embedding_batches.append(
                full_target_embeddings.reshape(-1, full_target_embeddings.shape[-1])
                .detach()
                .cpu()
            )

    if num_batches == 0:
        raise RuntimeError("No validation batches were processed")

    val_jepa = total_jepa / num_batches
    val_mae = total_mae / num_batches
    val_total = config["lambda_jepa"] * val_jepa + config["lambda_mae"] * val_mae

    embeddings = torch.cat(embedding_batches, dim=0)[:4096]
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    embedding_std = embeddings.std(dim=0, unbiased=False).mean().item()
    covariance = centered.T.matmul(centered) / max(centered.shape[0] - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    covariance_offdiag = off_diagonal.abs().mean().item()
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    ).item()

    return {
        "total_loss": float(val_total),
        "jepa_loss": float(val_jepa),
        "mae_loss": float(val_mae),
        "embedding_std": float(embedding_std),
        "covariance_offdiag": float(covariance_offdiag),
        "effective_rank": float(effective_rank),
    }


def initialize_models(encoder, predictor, decoder):
    for model in (encoder, predictor, decoder):
        for module in model.modules():
            init_weights(module)

    # Generic Linear initialization otherwise overwrites the residual decoder's
    # intended zero residual branch.
    if isinstance(decoder, ResidualMLPDecoder):
        torch.nn.init.zeros_(decoder.residual_head[-1].weight)
        torch.nn.init.zeros_(decoder.residual_head[-1].bias)


def save_checkpoint(
    encoder,
    predictor,
    decoder,
    path_save,
    epoch,
    config,
    encoder_ema=None,
    optimizer=None,
    scheduler=None,
    global_step=0,
    ema_schedule_steps=None,
    best_validation_loss=None,
    checkpoint_path=None,
):
    path_name = checkpoint_path or (
        path_save + "_epoch_" + str(epoch) + ".pt"
    )
    os.makedirs(os.path.dirname(path_name), exist_ok=True)
    payload = {
        "strategy": {
            "random": "dual_jepa_mae",
            "local_long": "local_mae_long_jepa",
            "future_block": "future_block_jepa_mae",
            "causal_multiblock": "causal_multiblock_jepa_mae",
        }[config.get("mask_strategy", "random")],
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "decoder": decoder.state_dict(),
        "epoch": epoch,
        "global_step": int(global_step),
        "ema_schedule_steps": ema_schedule_steps,
        "best_validation_loss": best_validation_loss,
        "config": copy.deepcopy(config),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if encoder_ema is not None:
        payload["encoder_ema"] = encoder_ema.state_dict()
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path_name)
    print("Saved checkpoint:", path_name)
    return path_name


def restore_training_state(
    checkpoint_path,
    encoder,
    predictor,
    decoder,
    encoder_ema,
    optimizer,
    scheduler,
    device,
    expected_fingerprint=None,
):
    """Restore a complete unified checkpoint and return loop state."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = {
        "encoder",
        "encoder_ema",
        "predictor",
        "decoder",
        "optimizer",
        "scheduler",
        "epoch",
        "global_step",
        "ema_schedule_steps",
        "rng_state",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(
            "Checkpoint cannot resume training because it lacks: "
            + ", ".join(missing)
        )

    checkpoint_fingerprint = checkpoint.get("config", {}).get("config_fingerprint")
    if (
        expected_fingerprint is not None
        and checkpoint_fingerprint is not None
        and checkpoint_fingerprint != expected_fingerprint
    ):
        raise ValueError(
            "Resume checkpoint configuration does not match this run: "
            f"checkpoint={checkpoint_fingerprint}, current={expected_fingerprint}"
        )

    encoder.load_state_dict(checkpoint["encoder"])
    encoder_ema.load_state_dict(checkpoint["encoder_ema"])
    predictor.load_state_dict(checkpoint["predictor"])
    decoder.load_state_dict(checkpoint["decoder"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    rng_state = checkpoint["rng_state"]
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"].cpu())
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])

    return {
        "start_epoch": int(checkpoint["epoch"]) + 1,
        "global_step": int(checkpoint["global_step"]),
        "ema_schedule_steps": checkpoint.get("ema_schedule_steps"),
        "best_validation_loss": checkpoint.get("best_validation_loss"),
        "best_validation_epoch": checkpoint.get("config", {}).get(
            "best_validation_epoch"
        ),
        "validation_history": checkpoint.get("config", {}).get(
            "validation_history",
            [],
        ),
    }


def last_saved_checkpoint_epoch(config):
    if config.get("last_saved_epoch") is not None:
        return int(config["last_saved_epoch"])
    raise FileNotFoundError(
        "No checkpoint was saved. Remove --no-save-final, lower "
        "--checkpoint-save, or pass --eval-checkpoint-to-use for an existing file."
    )


def run_downstream_evaluation(config):
    if config.get("eval_use_best"):
        checkpoint_path = config["path_save"] + "_best.pt"
        checkpoint_to_use = int(config.get("best_validation_epoch", 0))
    else:
        checkpoint_to_use = config.get("eval_checkpoint_to_use")
        if checkpoint_to_use is None:
            checkpoint_to_use = last_saved_checkpoint_epoch(config)
        checkpoint_path = (
            config["path_save"] + "_epoch_" + str(checkpoint_to_use) + ".pt"
        )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Downstream checkpoint not found: {checkpoint_path}")

    eval_argv = [
        "eval_forecast_prequential_with_baselines_gru_volume.py",
        "--data",
        str(config["data"]),
        "--checkpoint_to_use",
        str(checkpoint_to_use),
        "--pretrain_checkpoint_path",
        checkpoint_path,
        "--pretrain-encoder-weights",
        str(config.get("eval_encoder_weights", "ema")),
        "--forecast-target",
        str(config.get("eval_forecast_target", "value")),
        "--lr_pretrain",
        str(config["lr"]),
        "--ema_pretrain",
        str(config["ema_momentum"]),
        "--mask_ratio",
        str(config["mask_ratio"]),
        "--ratio_patches",
        str(config["ratio_patches"]),
        "--pretrain_encoder_embed_dim",
        str(config["encoder_embed_dim"]),
        "--pretrain_encoder_nhead",
        str(config["encoder_nhead"]),
        "--pretrain_encoder_num_layers",
        str(config["encoder_num_layers"]),
        "--pretrain_encoder_kernel_size",
        str(config["encoder_kernel_size"]),
        "--pretrain_decoder_embed_dim",
        str(config["predictor_embed"]),
        "--pretrain_decoder_nhead",
        str(config["predictor_nhead"]),
        "--pretrain_decoder_num_layers",
        str(config["predictor_num_layers"]),
        "--patch_size",
        str(config["patch_size"]),
        "--target_feature_index",
        str(config.get("target_feature_index", 0)),
        "--normalization",
        str(config.get("normalization", "window_return")),
        "--feature-transform",
        str(config.get("feature_transform", "raw")),
        "--market-data",
        str(config.get("market_data") or "none"),
        "--sampling-mode",
        str(config.get("sampling_mode", "sliding_window")),
        "--normalization_stats_json",
        json.dumps(config.get("normalization_stats")),
        "--sentiment-normalization",
        str(config.get("sentiment_normalization", "none")),
        "--sentiment-normalization-stats-json",
        json.dumps(config.get("sentiment_normalization_stats")),
        "--feature_cols",
        *[str(column) for column in config["feature_cols"]],
        "--market-features",
        *[str(column) for column in config["market_features"]],
        "--sentiment-features",
        *[str(column) for column in config["sentiment_features"]],
        "--timestamp_col",
        str(config["timestamp_col"]),
        "--target-col",
        str(config["target_col"]),
        "--sentiment_path",
        str(config["sentiment_path"] or "none"),
        "--train_end_date",
        str(config["train_end_date"] or "none"),
        "--test_start_date",
        str(config["test_start_date"] or "none"),
        "--data_end_date",
        str(config["data_end_date"] or "none"),
        "--validation_fraction",
        str(config["validation_fraction"]),
    ]
    if config.get("eval_forecast_horizon") is not None:
        eval_argv.extend(
            ["--forecast-horizon", str(config["eval_forecast_horizon"])]
        )
    eval_argv.append(
        "--use-sentiment" if config["use_sentiment"] else "--no-sentiment"
    )
    if config.get("robust_zscore_clip") is not None:
        eval_argv.extend(
            ["--robust-zscore-clip", str(config["robust_zscore_clip"])]
        )

    if not config["encoder_embed_bias"]:
        eval_argv.append("--no-pretrain-encoder-embed-bias")

    if config.get("eval_num_epochs") is not None:
        eval_argv.extend(["--num_epochs", str(config["eval_num_epochs"])])
    if config.get("eval_results_dir") is not None:
        eval_argv.extend(["--results_dir", str(config["eval_results_dir"])])

    print("\n=== Downstream evaluation with GRU baseline ===")
    print("data =", config["data"])
    print("checkpoint_path =", checkpoint_path)

    original_argv = sys.argv[:]
    try:
        sys.argv = eval_argv
        runpy.run_path(
            "eval_forecast_prequential_with_baselines_gru_volume.py",
            run_name="__main__",
        )
    finally:
        sys.argv = original_argv


def main(default_mask_strategy=None, argv=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = parse_args(
        base_config,
        default_mask_strategy=default_mask_strategy,
        argv=argv,
    )
    checkpoint_target = config.get("eval_forecast_target", "value")
    config["target_definition"] = {
        "cumulative_log_return": "log(Close[t+h] / Close[t])",
        "excess_log_return": (
            "log(Close[t+h] / Close[t]) - "
            "log(Market[t+h] / Market[t])"
        ),
        "relative_return": "Close[t+h] / Close[t] - 1",
        "value": "selected normalized feature value",
    }[checkpoint_target]
    print("Device:", device)

    if config["mask_strategy"] != "random" and config["input_mode"] != "timeseries":
        raise ValueError("Structured causal masks require --input-mode timeseries")
    if config["sampling_mode"] == "temporal_segments" and config["input_mode"] != "timeseries":
        raise ValueError("Temporal segments require --input-mode timeseries")
    if config["input_mode"] == "mnist_rows":
        loader = get_mnist_row_loader(
            root=config["mnist_root"],
            batch_size=config["batch_size"],
            mask_ratio=config["mask_ratio"],
            train=True,
            sample_count=config["mnist_train_samples"],
            download=config["download_mnist"],
            seed=config["seed"],
        )
        val_loader = get_mnist_row_loader(
            root=config["mnist_root"],
            batch_size=config["batch_size"],
            mask_ratio=config["mask_ratio"],
            train=True,
            sample_count=config["mnist_val_samples"],
            download=config["download_mnist"],
            seed=config["seed"],
            deterministic_masks=True,
            sample_offset=config["mnist_train_samples"],
            shuffle=False,
        )
        config["normalization_stats"] = None
        config["sentiment_normalization_stats"] = None
    else:
        loader = get_jepa_loaders(
            path=config["path_data"],
            batch_size=config["batch_size"],
            mask_ratio=config["mask_ratio"],
            series_split_size=config["series_split_size"],
            patch_size=config["patch_size"],
            stride=config["pretrain_stride"],
            sampling_mode=config["sampling_mode"],
            normalization=config["normalization"],
            feature_transform=config.get("feature_transform", "raw"),
            market_data=config.get("market_data"),
            sentiment_normalization=config.get("sentiment_normalization", "none"),
            robust_zscore_clip=config.get("robust_zscore_clip"),
            feature_cols=config["feature_cols"],
            timestamp_col=config["timestamp_col"],
            sentiment_path=config["sentiment_path"],
            validation_fraction=config["validation_fraction"],
            train_end_date=config["train_end_date"],
            test_start_date=config["test_start_date"],
            data_end_date=config["data_end_date"],
        )
        config["normalization_stats"] = copy.deepcopy(
            loader.dataset.normalization_stats
        )
        config["sentiment_normalization_stats"] = copy.deepcopy(
            loader.dataset.sentiment_normalization_stats
        )
        config["feature_cols"] = list(loader.dataset.feature_cols)
        config["feature_names"] = list(loader.dataset.feature_names)
        config["feature_dim"] = int(loader.dataset.feature_dim)
        config["warmup_report"] = copy.deepcopy(loader.dataset.warmup_report)
        config["market_alignment_report"] = copy.deepcopy(
            loader.dataset.market_alignment_report
        )
        val_loader = None
        if float(config["validation_fraction"]) > 0:
            try:
                val_loader = get_jepa_loaders(
                    path=config["path_data"],
                    batch_size=config["batch_size"],
                    mask_ratio=config["mask_ratio"],
                    series_split_size=config["series_split_size"],
                    patch_size=config["patch_size"],
                    stride=config["pretrain_stride"],
                    sampling_mode=config["sampling_mode"],
                    normalization=config["normalization"],
                    normalization_stats=config["normalization_stats"],
                    feature_transform=config.get("feature_transform", "raw"),
                    market_data=config.get("market_data"),
                    sentiment_normalization=config.get(
                        "sentiment_normalization", "none"
                    ),
                    sentiment_normalization_stats=config.get(
                        "sentiment_normalization_stats"
                    ),
                    robust_zscore_clip=config.get("robust_zscore_clip"),
                    split="val",
                    mask_seed=config["seed"] + 10_000,
                    feature_cols=config["feature_cols"],
                    timestamp_col=config["timestamp_col"],
                    sentiment_path=config["sentiment_path"],
                    validation_fraction=config["validation_fraction"],
                    train_end_date=config["train_end_date"],
                    test_start_date=config["test_start_date"],
                    data_end_date=config["data_end_date"],
                )
            except ValueError as error:
                print(
                    "Warning: validation split is too short for one pretraining "
                    f"window; validation is disabled for this run: {error}"
                )

    sample_patches, _, _ = loader.dataset[0]
    num_patches = sample_patches.shape[0]
    patch_dim = sample_patches.shape[-1]
    validate_training_config(
        config=config,
        num_patches=num_patches,
        patch_dim=patch_dim,
        loader_length=len(loader),
    )

    print("\n=== Unified JEPA + MAE pretrain config ===")
    print("data =", config["data"])
    print("input_mode =", config["input_mode"])
    print("mask_strategy =", config["mask_strategy"])
    print("seed =", config["seed"])
    if config["input_mode"] == "mnist_rows":
        print("mnist_root =", config["mnist_root"])
        print("mnist_train_samples =", config["mnist_train_samples"])
    else:
        print("path_data =", config["path_data"])
        print("use_sentiment =", config["use_sentiment"])
        print("market_features =", config["market_features"])
        print("sentiment_features =", config["sentiment_features"])
        print("feature_cols =", config["feature_cols"])
        print("sentiment_path =", config["sentiment_path"])
        print("train_end_date =", config["train_end_date"])
        print("test_start_date =", config["test_start_date"])
        print("data_end_date =", config["data_end_date"])
        print("sampling_mode =", config["sampling_mode"])
        print("pretrain_stride =", config["pretrain_stride"])
        print("normalization =", config["normalization"])
        print("feature_transform =", config.get("feature_transform", "raw"))
        print("robust_zscore_clip =", config.get("robust_zscore_clip"))
        print("market_data =", config.get("market_data"))
        print("warmup_report =", config.get("warmup_report"))
        print("train_windows =", len(loader.dataset))
        print("validation_windows =", len(val_loader.dataset) if val_loader else 0)
    print("num_patches =", num_patches)
    print("patch_dim =", patch_dim)
    print("lambda_jepa =", config["lambda_jepa"])
    print("lambda_mae =", config["lambda_mae"])
    print("jepa_loss =", config["jepa_loss"])
    print("mae_loss =", config["mae_loss"])
    print("decoder_type =", config["decoder_type"])
    if config["mask_strategy"] == "local_long":
        print("mae_window_patches =", config["mae_window_patches"])
        print("jepa_gap_patches =", config["jepa_gap_patches"])
        print("jepa_target_patches =", config["jepa_target_patches"])
        print("anchor_strategy =", config["anchor_strategy"])
    elif config["mask_strategy"] == "future_block":
        print("future_target_patches =", config["future_target_patches"])
        print("anchor_strategy =", config["anchor_strategy"])
    elif config["mask_strategy"] == "causal_multiblock":
        print("causal_num_blocks =", config["causal_num_blocks"])
        print("causal_block_patches =", config["causal_block_patches"])
        print("causal_block_gap_patches =", config["causal_block_gap_patches"])
        print("anchor_strategy =", config["anchor_strategy"])
    print("path_save =", config["path_save"])

    encoder = Encoder(
        num_patches=num_patches,
        dim_in=patch_dim,
        kernel_size=config["encoder_kernel_size"],
        embed_dim=config["encoder_embed_dim"],
        embed_bias=config["encoder_embed_bias"],
        nhead=config["encoder_nhead"],
        num_layers=config["encoder_num_layers"],
        jepa=True,
    )

    predictor = Predictor(
        num_patches=num_patches,
        encoder_embed_dim=config["encoder_embed_dim"],
        predictor_embed_dim=config["predictor_embed"],
        nhead=config["predictor_nhead"],
        num_layers=config["predictor_num_layers"],
    )

    decoder = build_reconstruction_decoder(
        decoder_type=config["decoder_type"],
        embedding_dim=config["encoder_embed_dim"],
        output_dim=patch_dim,
        hidden_dim=config["decoder_hidden_dim"],
        num_layers=config["decoder_num_layers"],
        dropout=config["decoder_dropout"],
    )

    initialize_models(encoder, predictor, decoder)

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters()},
            {"params": predictor.parameters()},
            {"params": decoder.parameters()},
        ],
        lr=config["lr"],
    )

    end_factor = config["end_lr"] / config["lr"] if config["lr"] > 0 else 1.0
    scheduler = lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=end_factor,
        total_iters=config["num_epochs"],
    )

    encoder = encoder.to(device)
    predictor = predictor.to(device)
    decoder = decoder.to(device)

    encoder_ema = copy.deepcopy(encoder)
    encoder_ema.eval()
    for p in encoder_ema.parameters():
        p.requires_grad = False

    steps_per_epoch = len(loader)
    if config["max_batches_per_epoch"] is not None:
        steps_per_epoch = min(steps_per_epoch, config["max_batches_per_epoch"])
    ema_schedule_steps = max(
        1,
        int(config["num_epochs"] * steps_per_epoch * config["ipe_scale"]),
    )

    saved_epochs = set()
    global_step = 0
    start_epoch = 0
    best_validation_loss = float("inf")
    best_validation_epoch = None
    validation_history = []

    if config.get("resume_from"):
        restored = restore_training_state(
            checkpoint_path=config["resume_from"],
            encoder=encoder,
            predictor=predictor,
            decoder=decoder,
            encoder_ema=encoder_ema,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            expected_fingerprint=config.get("config_fingerprint"),
        )
        start_epoch = restored["start_epoch"]
        global_step = restored["global_step"]
        if restored["ema_schedule_steps"] is not None:
            ema_schedule_steps = int(restored["ema_schedule_steps"])
        if restored["best_validation_loss"] is not None:
            best_validation_loss = float(restored["best_validation_loss"])
        best_validation_epoch = restored["best_validation_epoch"]
        validation_history = list(restored["validation_history"])
        print(
            f"Resumed {config['resume_from']} at epoch={start_epoch}, "
            f"global_step={global_step}"
        )

    for epoch in range(start_epoch, config["num_epochs"]):
        encoder.train()
        predictor.train()
        decoder.train()

        epoch_losses = []
        epoch_jepa_losses = []
        epoch_mae_losses = []
        total_anchor = 0.0
        num_batches = 0

        for batch_idx, (patches, dataset_masks, dataset_non_masks) in enumerate(loader):
            if (
                config["max_batches_per_epoch"] is not None
                and batch_idx >= config["max_batches_per_epoch"]
            ):
                break

            patches = patches.to(device)
            if config["mask_strategy"] != "random":
                objective_masks = make_strategy_masks(
                    config=config,
                    batch_size=patches.size(0),
                    num_patches=num_patches,
                    device=device,
                )
            else:
                dataset_masks = dataset_masks.to(device)
                dataset_non_masks = dataset_non_masks.to(device)
                objective_masks = {
                    "anchor": None,
                    "mae": dataset_masks,
                    "jepa": dataset_masks,
                    "predict": dataset_masks,
                    "context": dataset_non_masks,
                }

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                target_ema = encoder_ema(patches)
                target_ema = F.layer_norm(target_ema, (target_ema.size(-1),))
                target_ema = apply_mask(target_ema, objective_masks["jepa"])
                target_patches = apply_mask(patches, objective_masks["mae"])

            context_tokens = encoder(patches, mask=objective_masks["context"])
            pred_target_embeddings = predictor(
                context_tokens,
                mask=objective_masks["predict"],
                non_masks=objective_masks["context"],
            )

            if config["mask_strategy"] == "local_long":
                mae_len = int(config["mae_window_patches"])
                pred_mae_embeddings = pred_target_embeddings[:, :mae_len]
                pred_jepa_embeddings = pred_target_embeddings[:, mae_len:]
            else:
                pred_mae_embeddings = pred_target_embeddings
                pred_jepa_embeddings = pred_target_embeddings

            reconstructed_target = decoder(pred_mae_embeddings)

            jepa_loss = loss_value(
                pred_jepa_embeddings,
                target_ema,
                config["jepa_loss"],
            )
            mae_loss = loss_value(
                reconstructed_target,
                target_patches,
                config["mae_loss"],
            )
            loss = (
                config["lambda_jepa"] * jepa_loss
                + config["lambda_mae"] * mae_loss
            )

            loss.backward()

            if config["clip_grad"] is not None and config["clip_grad"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters())
                    + list(predictor.parameters())
                    + list(decoder.parameters()),
                    max_norm=config["clip_grad"],
                )

            optimizer.step()

            m = ema_momentum_at_step(
                config["ema_momentum"],
                global_step,
                ema_schedule_steps,
            )
            with torch.no_grad():
                for param_q, param_k in zip(
                    encoder.parameters(),
                    encoder_ema.parameters(),
                ):
                    param_k.data.mul_(m).add_(
                        (1.0 - m) * param_q.detach().data
                    )
            global_step += 1

            epoch_losses.append(loss.detach())
            epoch_jepa_losses.append(jepa_loss.detach())
            epoch_mae_losses.append(mae_loss.detach())
            if objective_masks["anchor"] is not None:
                total_anchor += float(objective_masks["anchor"])
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError(
                "No training batches were processed; check the loader and "
                "--max-batches-per-epoch."
            )
        scheduler.step()

        total_loss = ordered_scalar_mean(epoch_losses)
        total_jepa_loss = ordered_scalar_mean(epoch_jepa_losses)
        total_mae_loss = ordered_scalar_mean(epoch_mae_losses)

        if epoch % config["checkpoint_print"] == 0:
            message = (
                f"Epoch {epoch}, lr: {optimizer.param_groups[0]['lr']:.3g} "
                f"- Total: {total_loss:.6f} "
                f"- JEPA: {total_jepa_loss:.6f} "
                f"- MAE: {total_mae_loss:.6f} "
                f"- EMA: {m:.6f}"
            )
            if config["mask_strategy"] != "random":
                message += f" - avg_anchor: {total_anchor / num_batches:.2f}"
            print(message)

        should_validate = val_loader is not None and (
            epoch % int(config["validation_interval"]) == 0
            or epoch == config["num_epochs"] - 1
        )
        if should_validate:
            validation_metrics = evaluate_pretraining(
                encoder=encoder,
                predictor=predictor,
                decoder=decoder,
                encoder_ema=encoder_ema,
                loader=val_loader,
                device=device,
                config=config,
            )
            print(
                f"Validation epoch {epoch} - Total: "
                f"{validation_metrics['total_loss']:.6f} - JEPA: "
                f"{validation_metrics['jepa_loss']:.6f} - MAE: "
                f"{validation_metrics['mae_loss']:.6f} - EmbStd: "
                f"{validation_metrics['embedding_std']:.6f} - EffRank: "
                f"{validation_metrics['effective_rank']:.2f} - CovOffDiag: "
                f"{validation_metrics['covariance_offdiag']:.6f}"
            )
            validation_history.append({"epoch": epoch, **validation_metrics})
            config["validation_history"] = copy.deepcopy(validation_history)
            if validation_metrics["total_loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["total_loss"]
                best_validation_epoch = epoch
                config["best_validation_epoch"] = epoch
                config["best_validation_metrics"] = validation_metrics
                save_checkpoint(
                    encoder,
                    predictor,
                    decoder,
                    config["path_save"],
                    epoch,
                    config,
                    encoder_ema=encoder_ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    ema_schedule_steps=ema_schedule_steps,
                    best_validation_loss=best_validation_loss,
                    checkpoint_path=config["path_save"] + "_best.pt",
                )

        if epoch % config["checkpoint_save"] == 0 and epoch != 0:
            save_checkpoint(
                encoder,
                predictor,
                decoder,
                config["path_save"],
                epoch,
                config,
                encoder_ema=encoder_ema,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                ema_schedule_steps=ema_schedule_steps,
                best_validation_loss=(
                    best_validation_loss
                    if best_validation_loss != float("inf")
                    else None
                ),
            )
            saved_epochs.add(epoch)

    final_epoch = config["num_epochs"] - 1
    if config["save_final"] and final_epoch not in saved_epochs:
        save_checkpoint(
            encoder,
            predictor,
            decoder,
            config["path_save"],
            final_epoch,
            config,
            encoder_ema=encoder_ema,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            ema_schedule_steps=ema_schedule_steps,
            best_validation_loss=(
                best_validation_loss
                if best_validation_loss != float("inf")
                else None
            ),
        )
        saved_epochs.add(final_epoch)

    if saved_epochs:
        config["last_saved_epoch"] = max(saved_epochs)
    if best_validation_epoch is not None:
        config["best_validation_epoch"] = best_validation_epoch

    if config.get("run_eval", False):
        run_downstream_evaluation(config)


if __name__ == "__main__":
    main()
