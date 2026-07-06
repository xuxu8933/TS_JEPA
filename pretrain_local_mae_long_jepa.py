"""
Pretrain TS-JEPA with local MAE and long-horizon JEPA objectives.

Strategy C:
    - MAE masks a short local window and reconstructs local patch values.
    - JEPA predicts a farther future latent window with an EMA target encoder.

With the repo default patch_size=5, the defaults map roughly to:
    L_MAE  on x[t:t+5]
    L_JEPA on z[t+20:t+40]
"""

import argparse
import copy
import os
import random
import runpy
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from config.config_pretrain import config as base_config
from main.utils import init_weights
from pretrain_dual_loss import (
    _float_for_path,
    _none_if_requested,
    build_decoder,
    build_pretrain_path,
    last_saved_checkpoint_epoch,
    loss_value,
)
from src.data_loaders.data_loader_roll_volume import get_jepa_loaders
from src.models.encoder import Encoder
from src.models.predictor import Predictor
from src.models.utils.mask_utils import apply_mask


warnings.filterwarnings("ignore")


def parse_args(config):
    parser = argparse.ArgumentParser(
        description="Local-MAE + long-horizon-JEPA time-series pretraining."
    )

    parser.add_argument("--data", type=str, default=config["data"])
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=config["batch_size"])
    parser.add_argument("--lr", type=float, default=config["lr"])
    parser.add_argument("--end_lr", "--end-lr", dest="end_lr", type=float, default=config.get("end_lr", config["lr"] * 0.5))
    parser.add_argument("--num_epochs", "--num-epochs", dest="num_epochs", type=int, default=config["num_epochs"])
    parser.add_argument("--ema_momentum", "--ema-momentum", dest="ema_momentum", type=float, default=config["ema_momentum"])
    parser.add_argument("--ratio_patches", "--ratio-patches", dest="ratio_patches", type=int, default=config["ratio_patches"])
    parser.add_argument("--mask_ratio", "--mask-ratio", dest="mask_ratio", type=float, default=config["mask_ratio"])
    parser.add_argument("--checkpoint_save", "--checkpoint-save", dest="checkpoint_save", type=int, default=config["checkpoint_save"])
    parser.add_argument("--checkpoint_print", "--checkpoint-print", dest="checkpoint_print", type=int, default=config["checkpoint_print"])
    parser.add_argument("--clip_grad", "--clip-grad", dest="clip_grad", type=float, default=config.get("clip_grad", 1.0))
    parser.add_argument("--ipe_scale", "--ipe-scale", dest="ipe_scale", type=float, default=config.get("ipe_scale", 1.25))
    parser.add_argument("--notes", type=str, default="")

    parser.add_argument("--series_split_size", "--series-split-size", dest="series_split_size", type=int, default=config.get("series_split_size", 120))
    parser.add_argument("--patch_size", "--patch-size", dest="patch_size", type=int, default=config.get("patch_size", 5))
    parser.add_argument("--feature_cols", "--feature-cols", dest="feature_cols", nargs="+", default=None)
    parser.add_argument("--timestamp_col", "--timestamp-col", dest="timestamp_col", type=str, default=config.get("timestamp_col", "Date"))
    parser.add_argument("--sentiment_path", "--sentiment-path", dest="sentiment_path", type=str, default=config.get("sentiment_path", None))
    parser.add_argument("--train_end_date", "--train-end-date", dest="train_end_date", type=str, default=config.get("train_end_date", None))
    parser.add_argument("--test_start_date", "--test-start-date", dest="test_start_date", type=str, default=config.get("test_start_date", None))
    parser.add_argument("--validation_fraction", "--validation-fraction", dest="validation_fraction", type=float, default=config.get("validation_fraction", 0.05))
    parser.add_argument("--test_fraction", "--test-fraction", dest="test_fraction", type=float, default=config.get("test_fraction", 0.30))

    parser.add_argument("--encoder_embed_dim", "--encoder-embed-dim", dest="encoder_embed_dim", type=int, default=config["encoder_embed_dim"])
    parser.add_argument("--encoder_nhead", "--encoder-nhead", dest="encoder_nhead", type=int, default=config["encoder_nhead"])
    parser.add_argument("--encoder_num_layers", "--encoder-num-layers", dest="encoder_num_layers", type=int, default=config["encoder_num_layers"])
    parser.add_argument("--encoder_kernel_size", "--encoder-kernel-size", dest="encoder_kernel_size", type=int, default=config["encoder_kernel_size"])
    parser.add_argument("--encoder_embed_bias", "--encoder-embed-bias", dest="encoder_embed_bias", action="store_true", default=config.get("encoder_embed_bias", True))

    parser.add_argument("--predictor_embed", "--predictor-embed", dest="predictor_embed", type=int, default=config["predictor_embed"])
    parser.add_argument("--predictor_nhead", "--predictor-nhead", dest="predictor_nhead", type=int, default=config["predictor_nhead"])
    parser.add_argument("--predictor_num_layers", "--predictor-num-layers", dest="predictor_num_layers", type=int, default=config["predictor_num_layers"])

    parser.add_argument("--lambda_jepa", "--lambda-jepa", dest="lambda_jepa", type=float, default=config.get("lambda_jepa", 1.0))
    parser.add_argument("--lambda_mae", "--lambda-mae", dest="lambda_mae", type=float, default=config.get("lambda_mae", 1.0))
    parser.add_argument("--jepa_loss", "--jepa-loss", dest="jepa_loss", choices=("mse", "l1", "smooth_l1"), default=config.get("jepa_loss", "mse"))
    parser.add_argument("--mae_loss", "--mae-loss", dest="mae_loss", choices=("mse", "l1", "smooth_l1"), default=config.get("mae_loss", "mse"))
    parser.add_argument("--decoder_type", "--decoder-type", dest="decoder_type", choices=("linear", "mlp", "residual_mlp"), default=config.get("mae_decoder_type", "linear"))
    parser.add_argument("--decoder_hidden_dim", "--decoder-hidden-dim", dest="decoder_hidden_dim", type=int, default=config.get("mae_decoder_hidden_dim", 256))
    parser.add_argument("--decoder_num_layers", "--decoder-num-layers", dest="decoder_num_layers", type=int, default=config.get("mae_decoder_num_layers", 2))
    parser.add_argument("--decoder_dropout", "--decoder-dropout", dest="decoder_dropout", type=float, default=config.get("mae_decoder_dropout", 0.1))

    parser.add_argument("--mae_window_patches", "--mae-window-patches", dest="mae_window_patches", type=int, default=1)
    parser.add_argument("--jepa_gap_patches", "--jepa-gap-patches", dest="jepa_gap_patches", type=int, default=4)
    parser.add_argument("--jepa_target_patches", "--jepa-target-patches", dest="jepa_target_patches", type=int, default=4)
    parser.add_argument("--anchor_strategy", "--anchor-strategy", dest="anchor_strategy", choices=("random", "fixed"), default="random")
    parser.add_argument("--fixed_anchor", "--fixed-anchor", dest="fixed_anchor", type=int, default=0)

    parser.add_argument("--max_batches_per_epoch", "--max-batches-per-epoch", dest="max_batches_per_epoch", type=int, default=None)
    parser.add_argument("--no_save_final", "--no-save-final", dest="save_final", action="store_false", default=True)
    parser.add_argument("--path_suffix", "--path-suffix", dest="path_suffix", type=str, default=None)
    parser.add_argument(
        "--compatible_save_name",
        "--compatible-save-name",
        dest="compatible_save_name",
        action="store_true",
        help="Do not suffix the checkpoint path, so the current downstream evaluator can find it.",
    )
    parser.add_argument("--run_eval", "--run-eval", dest="run_eval", action="store_true")
    parser.add_argument("--eval_checkpoint_to_use", "--eval-checkpoint-to-use", dest="eval_checkpoint_to_use", type=int, default=None)
    parser.add_argument("--eval_num_epochs", "--eval-num-epochs", dest="eval_num_epochs", type=int, default=None)

    args = parser.parse_args()
    cfg = copy.deepcopy(config)

    for key, value in vars(args).items():
        cfg[key] = value

    cfg["feature_cols"] = args.feature_cols or config.get("feature_cols", ["Close", "Volume"])
    cfg["sentiment_path"] = _none_if_requested(args.sentiment_path)
    cfg["train_end_date"] = _none_if_requested(args.train_end_date)
    cfg["test_start_date"] = _none_if_requested(args.test_start_date)
    cfg["path_data"] = "./data/" + args.data + "/" + args.data + ".csv"

    seed = random.randint(0, 100)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    cfg["seed"] = seed

    if args.max_batches_per_epoch is not None and args.max_batches_per_epoch <= 0:
        raise ValueError("--max-batches-per-epoch must be positive when set")

    if args.compatible_save_name:
        cfg["path_suffix"] = ""
    elif args.path_suffix is not None:
        cfg["path_suffix"] = args.path_suffix
    else:
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

    cfg["path_save"] = build_pretrain_path(cfg)
    return cfg


def validate_strategy_config(config, num_patches):
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
            f"Need at least gap + jepa_target patches. "
            f"num_patches={num_patches}, gap={gap}, jepa_target={jepa_len}"
        )

    if config["anchor_strategy"] == "fixed":
        fixed_anchor = int(config["fixed_anchor"])
        if fixed_anchor < 0 or fixed_anchor > max_anchor:
            raise ValueError(
                f"--fixed-anchor must be in [0, {max_anchor}], got {fixed_anchor}"
            )

    return max_anchor


def make_strategy_masks(config, batch_size, num_patches, device):
    mae_len = int(config["mae_window_patches"])
    gap = int(config["jepa_gap_patches"])
    jepa_len = int(config["jepa_target_patches"])
    max_anchor = validate_strategy_config(config, num_patches)

    if config["anchor_strategy"] == "random":
        anchor = random.randint(0, max_anchor)
    else:
        anchor = int(config["fixed_anchor"])

    mae_start = anchor
    jepa_start = anchor + gap

    mae_indices = torch.arange(mae_start, mae_start + mae_len, dtype=torch.long)
    jepa_indices = torch.arange(jepa_start, jepa_start + jepa_len, dtype=torch.long)
    pred_indices = torch.cat([mae_indices, jepa_indices], dim=0)

    context_indices = [
        idx
        for idx in range(0, jepa_start)
        if idx < mae_start or idx >= mae_start + mae_len
    ]
    if not context_indices:
        raise ValueError("Strategy created an empty context mask")

    context_indices = torch.tensor(context_indices, dtype=torch.long)

    return {
        "anchor": anchor,
        "mae": mae_indices.unsqueeze(0).repeat(batch_size, 1).to(device),
        "jepa": jepa_indices.unsqueeze(0).repeat(batch_size, 1).to(device),
        "predict": pred_indices.unsqueeze(0).repeat(batch_size, 1).to(device),
        "context": context_indices.unsqueeze(0).repeat(batch_size, 1).to(device),
    }


def save_checkpoint(encoder, predictor, decoder, path_save, epoch, config):
    path_name = path_save + "_epoch_" + str(epoch) + ".pt"
    os.makedirs(os.path.dirname(path_name), exist_ok=True)
    torch.save(
        {
            "strategy": "local_mae_long_jepa",
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "decoder": decoder.state_dict(),
            "epoch": epoch,
            "config": config,
        },
        path_name,
    )
    print("Saved checkpoint:", path_name)
    return path_name


def run_downstream_evaluation(config):
    checkpoint_to_use = config.get("eval_checkpoint_to_use")
    if checkpoint_to_use is None:
        checkpoint_to_use = last_saved_checkpoint_epoch(config)

    checkpoint_path = config["path_save"] + "_epoch_" + str(checkpoint_to_use) + ".pt"
    eval_argv = [
        "eval_forecast_prequential_with_baselines_gru_volume.py",
        "--data",
        str(config["data"]),
        "--checkpoint_to_use",
        str(checkpoint_to_use),
        "--pretrain_checkpoint_path",
        checkpoint_path,
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
    ]

    if config.get("eval_num_epochs") is not None:
        eval_argv.extend(["--num_epochs", str(config["eval_num_epochs"])])

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


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = parse_args(base_config)
    print("Device:", device)

    loader = get_jepa_loaders(
        path=config["path_data"],
        batch_size=config["batch_size"],
        ratio_patches=config["ratio_patches"],
        mask_ratio=config["mask_ratio"],
        series_split_size=config["series_split_size"],
        patch_size=config["patch_size"],
        feature_cols=config["feature_cols"],
        timestamp_col=config["timestamp_col"],
        sentiment_path=config["sentiment_path"],
        validation_fraction=config["validation_fraction"],
        test_fraction=config["test_fraction"],
        train_end_date=config["train_end_date"],
        test_start_date=config["test_start_date"],
    )

    sample_patches, _, _ = loader.dataset[0]
    num_patches = sample_patches.shape[0]
    patch_dim = sample_patches.shape[-1]
    max_anchor = validate_strategy_config(config, num_patches)

    print("\n=== Local-MAE + long-JEPA pretrain config ===")
    print("data =", config["data"])
    print("path_data =", config["path_data"])
    print("feature_cols =", config["feature_cols"])
    print("sentiment_path =", config["sentiment_path"])
    print("train_end_date =", config["train_end_date"])
    print("test_start_date =", config["test_start_date"])
    print("num_patches =", num_patches)
    print("patch_dim =", patch_dim)
    print("mae_window_patches =", config["mae_window_patches"])
    print("jepa_gap_patches =", config["jepa_gap_patches"])
    print("jepa_target_patches =", config["jepa_target_patches"])
    print("anchor_strategy =", config["anchor_strategy"])
    print("max_anchor =", max_anchor)
    print("lambda_jepa =", config["lambda_jepa"])
    print("lambda_mae =", config["lambda_mae"])
    print("decoder_type =", config["decoder_type"])
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

    decoder = build_decoder(config, patch_dim)

    for model in (encoder, predictor, decoder):
        for module in model.modules():
            init_weights(module)

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

    ema_scheduler = (
        config["ema_momentum"]
        + i
        * (1 - config["ema_momentum"])
        / (config["num_epochs"] * config["ipe_scale"])
        for i in range(int(config["num_epochs"] * config["ipe_scale"]) + 1)
    )

    saved_epochs = set()
    mae_len = int(config["mae_window_patches"])

    for epoch in range(config["num_epochs"]):
        encoder.train()
        predictor.train()
        decoder.train()

        total_loss = 0.0
        total_jepa_loss = 0.0
        total_mae_loss = 0.0
        total_anchor = 0.0
        num_batches = 0
        m = next(ema_scheduler)

        for batch_idx, (patches, _, _) in enumerate(loader):
            if (
                config["max_batches_per_epoch"] is not None
                and batch_idx >= config["max_batches_per_epoch"]
            ):
                break

            patches = patches.to(device)
            masks = make_strategy_masks(
                config=config,
                batch_size=patches.size(0),
                num_patches=num_patches,
                device=device,
            )

            optimizer.zero_grad()

            with torch.no_grad():
                target_ema = encoder_ema(patches)
                target_ema = F.layer_norm(target_ema, (target_ema.size(-1),))
                target_jepa = apply_mask(target_ema, masks["jepa"])
                target_local_patches = apply_mask(patches, masks["mae"])

            context_tokens = encoder(patches, mask=masks["context"])
            pred_embeddings = predictor(
                context_tokens,
                mask=masks["predict"],
                non_masks=masks["context"],
            )
            pred_local_embeddings = pred_embeddings[:, :mae_len]
            pred_jepa_embeddings = pred_embeddings[:, mae_len:]
            reconstructed_local = decoder(pred_local_embeddings)

            jepa_loss = loss_value(
                pred_jepa_embeddings,
                target_jepa,
                config["jepa_loss"],
            )
            mae_loss = loss_value(
                reconstructed_local,
                target_local_patches,
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

            with torch.no_grad():
                for param_q, param_k in zip(
                    encoder.parameters(),
                    encoder_ema.parameters(),
                ):
                    param_k.data.mul_(m).add_(
                        (1.0 - m) * param_q.detach().data
                    )

            total_loss += loss.item()
            total_jepa_loss += jepa_loss.item()
            total_mae_loss += mae_loss.item()
            total_anchor += float(masks["anchor"])
            num_batches += 1

        scheduler.step()

        total_loss /= num_batches
        total_jepa_loss /= num_batches
        total_mae_loss /= num_batches
        avg_anchor = total_anchor / num_batches

        if epoch % config["checkpoint_print"] == 0:
            print(
                f"Epoch {epoch}, lr: {optimizer.param_groups[0]['lr']:.3g} "
                f"- Total: {total_loss:.6f} "
                f"- JEPA: {total_jepa_loss:.6f} "
                f"- MAE: {total_mae_loss:.6f} "
                f"- avg_anchor: {avg_anchor:.2f}"
            )

        if epoch % config["checkpoint_save"] == 0 and epoch != 0:
            save_checkpoint(
                encoder,
                predictor,
                decoder,
                config["path_save"],
                epoch,
                config,
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
        )
        saved_epochs.add(final_epoch)

    if saved_epochs:
        config["last_saved_epoch"] = max(saved_epochs)

    if config.get("run_eval", False):
        run_downstream_evaluation(config)


if __name__ == "__main__":
    main()
