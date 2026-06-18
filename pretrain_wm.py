"""
Pretrain TS-JEPA on OHLCV/sentiment market data.
"""

import copy
import os
import runpy
import sys
import warnings

import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from config.config_pretrain import config
from main.utils import init_weights, prepare_args_pretrain
from src.data_loaders.data_loader_roll_volume import get_jepa_loaders
from src.models.encoder import Encoder
from src.models.predictor import Predictor
from src.models.utils.mask_utils import apply_mask


warnings.filterwarnings("ignore")


def loss_pred(pred, target_ema):
    loss = 0.0
    for pred_i, target_ema_i in zip(pred, target_ema):
        loss = loss + torch.mean(torch.abs(pred_i - target_ema_i))
    loss /= len(pred)
    return loss


def save_model(model, path_save, epoch):
    save_dict = {"encoder": model.state_dict(), "epoch": epoch}
    path_name = path_save + "_epoch_" + str(epoch) + ".pt"
    os.makedirs(os.path.dirname(path_name), exist_ok=True)
    torch.save(save_dict, path_name)


def last_saved_checkpoint_epoch(config):
    checkpoint_save = int(config["checkpoint_save"])
    final_epoch = int(config["num_epochs"]) - 1
    if final_epoch <= 0:
        return 0
    return (final_epoch // checkpoint_save) * checkpoint_save


def run_downstream_evaluation(config):
    checkpoint_to_use = config.get("eval_checkpoint_to_use")
    if checkpoint_to_use is None:
        checkpoint_to_use = last_saved_checkpoint_epoch(config)

    eval_argv = [
        "eval_forecast_prequential_with_baselines_gru_volume.py",
        "--data",
        str(config["data"]),
        "--checkpoint_to_use",
        str(checkpoint_to_use),
    ]

    if config.get("eval_num_epochs") is not None:
        eval_argv.extend(["--num_epochs", str(config["eval_num_epochs"])])

    print("\n=== Downstream evaluation with GRU baseline ===")
    print("data =", config["data"])
    print("checkpoint_to_use =", checkpoint_to_use)
    if config.get("eval_num_epochs") is not None:
        print("eval_num_epochs =", config["eval_num_epochs"])

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = prepare_args_pretrain(config)
    print("Device:", device)

    feature_cols = config.get("feature_cols", ["Close", "Volume"])
    sentiment_path = config.get("sentiment_path", None)
    train_end_date = config.get("train_end_date", None)
    test_start_date = config.get("test_start_date", None)
    validation_fraction = config.get("validation_fraction", 0.05)
    test_fraction = config.get("test_fraction", 0.30)

    if not config.get("skip_pretrain", False):
        loader = get_jepa_loaders(
            path=config["path_data"],
            batch_size=config["batch_size"],
            ratio_patches=config["ratio_patches"],
            mask_ratio=config["mask_ratio"],
            series_split_size=config["series_split_size"],
            patch_size=config.get("patch_size", 5),
            feature_cols=feature_cols,
            timestamp_col=config.get("timestamp_col", "Date"),
            sentiment_path=sentiment_path,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
        )

        sample_patches, _, _ = loader.dataset[0]
        input_dim = sample_patches.shape[-1]

        print("\n=== Encoder config ===")
        print("data =", config["data"])
        print("path_data =", config["path_data"])
        print("feature_cols =", feature_cols)
        print("sentiment_path =", sentiment_path)
        print("train_end_date =", train_end_date)
        print("test_start_date =", test_start_date)
        print("validation_fraction =", validation_fraction)
        print("test_fraction =", test_fraction)
        print("num_patches =", sample_patches.shape[0])
        print("dim_in =", input_dim)
        print("kernel_size =", config["encoder_kernel_size"])
        print("embed_dim =", config["encoder_embed_dim"])
        print("nhead =", config["encoder_nhead"])
        print("num_layers =", config["encoder_num_layers"])

        encoder = Encoder(
            num_patches=sample_patches.shape[0],
            dim_in=input_dim,
            kernel_size=config["encoder_kernel_size"],
            embed_dim=config["encoder_embed_dim"],
            embed_bias=config["encoder_embed_bias"],
            nhead=config["encoder_nhead"],
            num_layers=config["encoder_num_layers"],
            jepa=True,
        )

        predictor = Predictor(
            num_patches=sample_patches.shape[0],
            encoder_embed_dim=config["encoder_embed_dim"],
            predictor_embed_dim=config["predictor_embed"],
            nhead=config["predictor_nhead"],
            num_layers=config["predictor_num_layers"],
        )

        for m in encoder.modules():
            init_weights(m)
        for m in predictor.modules():
            init_weights(m)

        optimizer = torch.optim.AdamW(
            [
                {"params": (p for p in encoder.parameters())},
                {"params": (p for p in predictor.parameters())},
            ],
            lr=config["lr"],
        )

        scheduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.5,
            total_iters=config["num_epochs"],
        )

        encoder = encoder.to(device)
        predictor = predictor.to(device)
        encoder_ema = copy.deepcopy(encoder)
        for p in encoder_ema.parameters():
            p.requires_grad = False

        checkpoint_save = config["checkpoint_save"]
        checkpoint_print = config["checkpoint_print"]
        path_save = config["path_save"]
        clip_grad = config["clip_grad"]

        ema_scheduler = (
            config["ema_momentum"]
            + i
            * (1 - config["ema_momentum"])
            / (config["num_epochs"] * config["ipe_scale"])
            for i in range(int(config["num_epochs"] * config["ipe_scale"]) + 1)
        )

        for epoch in range(config["num_epochs"]):
            encoder.train()
            predictor.train()

            total_loss = 0.0
            m = next(ema_scheduler)

            for patches, masks, non_masks in loader:
                patches = patches.to(device)
                masks = masks.to(device)
                non_masks = non_masks.to(device)

                optimizer.zero_grad()

                with torch.no_grad():
                    target_ema = encoder_ema(patches)
                    target_ema = F.layer_norm(target_ema, (target_ema.size(-1),))
                    target_ema = apply_mask(target_ema, masks)

                tokens = encoder(patches, mask=non_masks)
                pred = predictor(tokens, mask=masks, non_masks=non_masks)
                loss = loss_pred(pred, target_ema)
                loss.backward()

                if clip_grad is not None and clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(encoder.parameters()) + list(predictor.parameters()),
                        max_norm=clip_grad,
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

            scheduler.step()
            total_loss /= len(loader)

            if epoch % checkpoint_print == 0:
                print(
                    f"Epoch {epoch}, lr: {optimizer.param_groups[0]['lr']:.3g} "
                    f"- JEPA Loss: {total_loss:.4f}"
                )

            if epoch % checkpoint_save == 0 and epoch != 0:
                save_model(encoder, path_save, epoch)
    else:
        print("Skipping TS-JEPA pretraining")

    if config.get("run_eval", False):
        run_downstream_evaluation(config)
