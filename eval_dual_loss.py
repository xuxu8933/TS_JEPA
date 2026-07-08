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

import numpy as np
import torch

from config.config_downstream import config as downstream_config
from pretrain_dual_loss import build_decoder
from src.data_loaders.data_loader_mnist_rows import get_mnist_row_loader
from src.models.encoder import Encoder
from src.models.predictor import Predictor
from src.models.utils.mask_utils import apply_mask


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
        "--eval-mode",
        choices=("forecast", "mnist_rows"),
        default="forecast",
        help="Run stock forecasting or masked MNIST-row reconstruction.",
    )
    parser.add_argument("--mnist-root", default="./data/MNIST")
    parser.add_argument("--mnist-test-samples", type=int, default=128)
    parser.add_argument("--download-mnist", action="store_true")
    parser.add_argument(
        "--require-better-than-naive",
        action="store_true",
        help="Fail MNIST evaluation unless reconstruction beats previous-row copying.",
    )
    parser.add_argument(
        "--prediction-output",
        default=None,
        help="Optional .npz path for MNIST inputs, reconstructions, and masks.",
    )
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


def evaluate_mnist_rows(args, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    seed = int(config.get("seed", 0))
    batch_size = args.batch_size or 64

    loader = get_mnist_row_loader(
        root=args.mnist_root,
        batch_size=batch_size,
        mask_ratio=float(config["mask_ratio"]),
        train=False,
        sample_count=args.mnist_test_samples,
        download=args.download_mnist,
        seed=seed + 10_000,
        deterministic_masks=True,
    )

    encoder = Encoder(
        num_patches=28,
        dim_in=28,
        kernel_size=config["encoder_kernel_size"],
        embed_dim=config["encoder_embed_dim"],
        embed_bias=config["encoder_embed_bias"],
        nhead=config["encoder_nhead"],
        num_layers=config["encoder_num_layers"],
        jepa=True,
    )
    predictor = Predictor(
        num_patches=28,
        encoder_embed_dim=config["encoder_embed_dim"],
        predictor_embed_dim=config["predictor_embed"],
        nhead=config["predictor_nhead"],
        num_layers=config["predictor_num_layers"],
    )
    decoder = build_decoder(config, patch_dim=28)

    encoder.load_state_dict(checkpoint["encoder"])
    predictor.load_state_dict(checkpoint["predictor"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    predictor.eval()
    decoder.eval()

    model_squared_error = 0.0
    naive_squared_error = 0.0
    value_count = 0
    examples = []

    with torch.no_grad():
        for image_rows, masks, non_masks in loader:
            context_tokens = encoder(image_rows, mask=non_masks)
            predicted_embeddings = predictor(
                context_tokens,
                mask=masks,
                non_masks=non_masks,
            )
            reconstructed_rows = decoder(predicted_embeddings)
            target_rows = apply_mask(image_rows, masks)

            previous_indices = (masks - 1).clamp_min(0)
            gather_indices = previous_indices.unsqueeze(-1).expand(-1, -1, 28)
            naive_rows = torch.gather(image_rows, dim=1, index=gather_indices)

            model_squared_error += (reconstructed_rows - target_rows).square().sum().item()
            naive_squared_error += (naive_rows - target_rows).square().sum().item()
            value_count += target_rows.numel()

            if len(examples) < 8:
                for sample_idx in range(min(image_rows.shape[0], 8 - len(examples))):
                    reconstruction = image_rows[sample_idx].clone()
                    reconstruction[masks[sample_idx]] = reconstructed_rows[sample_idx]
                    naive = image_rows[sample_idx].clone()
                    naive[masks[sample_idx]] = naive_rows[sample_idx]
                    examples.append(
                        (
                            image_rows[sample_idx].numpy(),
                            reconstruction.numpy(),
                            naive.numpy(),
                            masks[sample_idx].numpy(),
                        )
                    )

    model_mse = model_squared_error / value_count
    naive_mse = naive_squared_error / value_count
    result = {
        "model_mse": model_mse,
        "naive_mse": naive_mse,
        "inputs": np.stack([example[0] for example in examples]),
        "reconstructions": np.stack([example[1] for example in examples]),
        "naive_reconstructions": np.stack([example[2] for example in examples]),
        "masks": np.stack([example[3] for example in examples]),
    }

    if args.prediction_output:
        output_path = os.path.abspath(args.prediction_output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(output_path, **result)
        print("Predictions:", output_path)

    print(
        f"MNIST rows: model_mse={model_mse:.6f}, "
        f"naive_previous_row_mse={naive_mse:.6f}"
    )
    if args.require_better_than_naive and model_mse >= naive_mse:
        raise AssertionError(
            f"MNIST row reconstruction did not beat previous-row copying: "
            f"{model_mse:.6f} >= {naive_mse:.6f}"
        )
    return result


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
        if args.eval_mode == "mnist_rows":
            print("Evaluation mode: mnist_rows")
        else:
            print("Delegated argv:")
            print(" ".join(eval_argv))
        return

    if args.eval_mode == "mnist_rows":
        evaluate_mnist_rows(args, checkpoint_path)
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
