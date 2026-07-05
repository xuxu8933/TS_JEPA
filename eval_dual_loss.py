"""
Evaluate a dual JEPA + MAE pretrained encoder on the downstream forecast task.

This is a thin wrapper around eval_forecast_prequential_with_baselines_gru_volume.py.
It only changes checkpoint resolution so the evaluator can load checkpoints saved
by pretrain_dual_loss.py's default suffixed naming scheme.
"""

import argparse
import os
import runpy
import sys

from config.config_downstream import config as downstream_config


def _float_for_path(value):
    return str(value).replace("/", "_")


def default_dual_checkpoint_path(args):
    base_name = (
        "lr_"
        + str(args.lr_pretrain)
        + "_ema_momentum_"
        + str(args.ema_pretrain)
        + "_mask_ratio_"
        + str(args.mask_ratio)
        + "_ratio_patches_"
        + str(args.ratio_patches)
        + "_encoder_"
        + str(args.pretrain_encoder_embed_dim)
        + "_"
        + str(args.pretrain_encoder_nhead)
        + "_"
        + str(args.pretrain_encoder_num_layers)
        + "_predictor_"
        + str(args.pretrain_decoder_embed_dim)
        + "_"
        + str(args.pretrain_decoder_nhead)
        + "_"
        + str(args.pretrain_decoder_num_layers)
    )

    if args.compatible_save_name:
        suffix = ""
    elif args.pretrain_path_suffix is not None:
        suffix = args.pretrain_path_suffix
    else:
        suffix = (
            "_dual_jepa_mae_ljepa_"
            + _float_for_path(args.lambda_jepa)
            + "_lmae_"
            + _float_for_path(args.lambda_mae)
        )

    filename = base_name + suffix + "_epoch_" + str(args.checkpoint_to_use) + ".pt"
    return os.path.join(args.checkpoint_dir, args.data, filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run downstream evaluation for a dual JEPA+MAE checkpoint."
    )

    parser.add_argument("--data", default=downstream_config["data"])
    parser.add_argument(
        "--checkpoint_to_use",
        "--checkpoint-to-use",
        dest="checkpoint_to_use",
        type=int,
        default=downstream_config["checkpoint_to_use"],
    )
    parser.add_argument(
        "--checkpoint_dir",
        "--checkpoint-dir",
        dest="checkpoint_dir",
        default=downstream_config.get("path_save", "./logs/output_model/"),
    )
    parser.add_argument(
        "--pretrain_checkpoint_path",
        "--pretrain-checkpoint-path",
        dest="pretrain_checkpoint_path",
        default=None,
        help="Use this exact checkpoint path instead of computing the dual-loss path.",
    )
    parser.add_argument(
        "--pretrain_path_suffix",
        "--pretrain-path-suffix",
        dest="pretrain_path_suffix",
        default=None,
        help="Custom suffix used by pretrain_dual_loss.py after the predictor fields.",
    )
    parser.add_argument(
        "--compatible_save_name",
        "--compatible-save-name",
        dest="compatible_save_name",
        action="store_true",
        help="Look for a checkpoint saved without the dual-loss suffix.",
    )

    parser.add_argument(
        "--lambda_jepa",
        "--lambda-jepa",
        dest="lambda_jepa",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lambda_mae",
        "--lambda-mae",
        dest="lambda_mae",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lr_pretrain",
        "--lr-pretrain",
        dest="lr_pretrain",
        type=float,
        default=downstream_config["lr_pretrain"],
    )
    parser.add_argument(
        "--ema_pretrain",
        "--ema-pretrain",
        dest="ema_pretrain",
        type=float,
        default=downstream_config["ema_pretrain"],
    )
    parser.add_argument(
        "--mask_ratio",
        "--mask-ratio",
        dest="mask_ratio",
        type=float,
        default=downstream_config["mask_ratio"],
    )
    parser.add_argument(
        "--ratio_patches",
        "--ratio-patches",
        dest="ratio_patches",
        type=int,
        default=downstream_config["ratio_patches"],
    )
    parser.add_argument(
        "--pretrain_encoder_embed_dim",
        "--pretrain-encoder-embed-dim",
        dest="pretrain_encoder_embed_dim",
        type=int,
        default=downstream_config["pretrain_encoder_embed_dim"],
    )
    parser.add_argument(
        "--pretrain_encoder_nhead",
        "--pretrain-encoder-nhead",
        dest="pretrain_encoder_nhead",
        type=int,
        default=downstream_config["pretrain_encoder_nhead"],
    )
    parser.add_argument(
        "--pretrain_encoder_num_layers",
        "--pretrain-encoder-num-layers",
        dest="pretrain_encoder_num_layers",
        type=int,
        default=downstream_config["pretrain_encoder_num_layers"],
    )
    parser.add_argument(
        "--pretrain_encoder_kernel_size",
        "--pretrain-encoder-kernel-size",
        dest="pretrain_encoder_kernel_size",
        type=int,
        default=downstream_config["pretrain_encoder_kernel_size"],
    )
    parser.add_argument(
        "--pretrain_decoder_embed_dim",
        "--pretrain-decoder-embed-dim",
        dest="pretrain_decoder_embed_dim",
        type=int,
        default=downstream_config["pretrain_decoder_embed_dim"],
    )
    parser.add_argument(
        "--pretrain_decoder_nhead",
        "--pretrain-decoder-nhead",
        dest="pretrain_decoder_nhead",
        type=int,
        default=downstream_config["pretrain_decoder_nhead"],
    )
    parser.add_argument(
        "--pretrain_decoder_num_layers",
        "--pretrain-decoder-num-layers",
        dest="pretrain_decoder_num_layers",
        type=int,
        default=downstream_config["pretrain_decoder_num_layers"],
    )

    parser.add_argument(
        "--batch_size",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=None,
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--num_epochs",
        "--num-epochs",
        dest="num_epochs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--dry_run",
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Print the checkpoint path and delegated command without running evaluation.",
    )

    args, passthrough_args = parser.parse_known_args()
    return args, passthrough_args


def build_eval_argv(args, passthrough_args):
    checkpoint_path = (
        args.pretrain_checkpoint_path
        if args.pretrain_checkpoint_path
        else default_dual_checkpoint_path(args)
    )

    eval_argv = [
        "eval_forecast_prequential_with_baselines_gru_volume.py",
        "--data",
        str(args.data),
        "--checkpoint_to_use",
        str(args.checkpoint_to_use),
        "--pretrain_checkpoint_path",
        checkpoint_path,
        "--lr_pretrain",
        str(args.lr_pretrain),
        "--ema_pretrain",
        str(args.ema_pretrain),
        "--mask_ratio",
        str(args.mask_ratio),
        "--ratio_patches",
        str(args.ratio_patches),
        "--pretrain_encoder_embed_dim",
        str(args.pretrain_encoder_embed_dim),
        "--pretrain_encoder_nhead",
        str(args.pretrain_encoder_nhead),
        "--pretrain_encoder_num_layers",
        str(args.pretrain_encoder_num_layers),
        "--pretrain_encoder_kernel_size",
        str(args.pretrain_encoder_kernel_size),
        "--pretrain_decoder_embed_dim",
        str(args.pretrain_decoder_embed_dim),
        "--pretrain_decoder_nhead",
        str(args.pretrain_decoder_nhead),
        "--pretrain_decoder_num_layers",
        str(args.pretrain_decoder_num_layers),
    ]

    if args.batch_size is not None:
        eval_argv.extend(["--batch_size", str(args.batch_size)])
    if args.lr is not None:
        eval_argv.extend(["--lr", str(args.lr)])
    if args.num_epochs is not None:
        eval_argv.extend(["--num_epochs", str(args.num_epochs)])

    eval_argv.extend(passthrough_args)
    return eval_argv, checkpoint_path


def main():
    args, passthrough_args = parse_args()
    eval_argv, checkpoint_path = build_eval_argv(args, passthrough_args)

    print("Dual-loss checkpoint:", checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "Dual-loss checkpoint not found. "
            "Train it with pretrain_dual_loss.py, pass --pretrain-checkpoint-path, "
            "or adjust --lambda-jepa/--lambda-mae/--pretrain-path-suffix."
        )

    if args.dry_run:
        print("Delegated argv:")
        print(" ".join(eval_argv))
        return

    original_argv = sys.argv[:]
    try:
        sys.argv = eval_argv
        runpy.run_path(
            "eval_forecast_prequential_with_baselines_gru_volume.py",
            run_name="__main__",
        )
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
