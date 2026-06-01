"""
    Script to pretrain a TS-JEPA before using it on the forecasting downstream.
"""

import warnings

warnings.filterwarnings("ignore")

import time
import copy
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from main.utils import prepare_args_pretrain, init_weights, grad_logger

from src.models.encoder import Encoder
from src.models.predictor import Predictor

from src.models.utils.mask_utils import apply_mask
from src.data_loaders.data_loader_roll_volume import get_jepa_loaders

from config.config_pretrain import config


def loss_pred(pred, target_ema):
    loss = 0.0
    for pred_i, target_ema_i in zip(pred, target_ema):
        loss = loss + torch.mean(torch.abs(pred_i - target_ema_i))
    loss /= len(pred)
    return loss


def save_model(model, epoch):
    save_dict = {"encoder": model.state_dict(), "epoch": epoch}

    try:
        path_name = path_save + "_epoch_" + str(epoch) + ".pt"
        torch.save(save_dict, path_name)
    except:
        print("Problem saving checkpoint")


# Define the custom learning rate schedule
def lr_lambda(epoch):
    start_lr = config["lr"]
    end_lr = config["end_lr"]
    if epoch < config["num_epochs"]:
        return start_lr + (end_lr - start_lr) * (epoch / (config["num_epochs"] - 1))
    else:
        return end_lr


if __name__ == "__main__":
    # Load device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Init config and args
    config = prepare_args_pretrain(config)

    # Load Data
    feature_cols = config.get("feature_cols", ["Close", "Volume"])

    loader = get_jepa_loaders(
        path=config["path_data"],
        batch_size=config["batch_size"],
        ratio_patches=config["ratio_patches"],
        mask_ratio=config["mask_ratio"],
        series_split_size=config["series_split_size"],
        patch_size=config.get("patch_size", 5),
        feature_cols=feature_cols,
        timestamp_col=config.get("timestamp_col", "Date"),
    )

    sample_patches, _, _ = loader.dataset[0]
    input_dim = sample_patches.shape[-1]
    print("\n=== Encoder config ===")
    print("feature_cols =", feature_cols)
    print("num_patches =", sample_patches.shape[0])
    print("dim_in =", input_dim)
    print("kernel_size =", config["encoder_kernel_size"])
    print("embed_dim =", config["encoder_embed_dim"])
    print("nhead =", config["encoder_nhead"])
    print("num_layers =", config["encoder_num_layers"])
    # Load Encoder
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

    # Load Predictor
    predictor = Predictor(
        num_patches=sample_patches.shape[0],
        encoder_embed_dim=config["encoder_embed_dim"],
        predictor_embed_dim=config["predictor_embed"],
        nhead=config["predictor_nhead"],
        num_layers=config["predictor_num_layers"],
    )

    # Init weights -- Similar to VJEPA
    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    param_groups = [
        {"params": (p for n, p in encoder.named_parameters())},
        {"params": (p for n, p in predictor.named_parameters())},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=config["lr"])

    # Initialize the scheduler
    scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.5, total_iters=config["num_epochs"]
    )

    encoder = encoder.to(device)
    predictor = predictor.to(device)

    # Initialize the EMA-Encoder
    encoder_ema = copy.deepcopy(encoder)

    # Stop-gradient step in the EMA
    for p in encoder_ema.parameters():
        p.requires_grad = False

    checkpoint_save = config["checkpoint_save"]
    checkpoint_print = config["checkpoint_print"]
    path_save = config["path_save"]
    clip_grad = config["clip_grad"]
    warmup = config["warmup_ratio"] * config["num_epochs"]

    # Initialize the EMA Scheduler (parameter m in the paper)
    num_batches = len(loader)
    ema_scheduler = (
        config["ema_momentum"]
        + i
        * (1 - config["ema_momentum"])
        / (config["num_epochs"] * config["ipe_scale"])
        for i in range(int(config["num_epochs"] * config["ipe_scale"]) + 1)
    )


    total_loss, total_var_encoder, total_var_decoder = 0.0, 0.0, 0.0

    # Save Initial Model -- Useful to compare when evaluating
    # save_model(encoder, 0)

    # Training loop
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
                    max_norm=clip_grad
                )

            optimizer.step()

            with torch.no_grad():
                for param_q, param_k in zip(encoder.parameters(), encoder_ema.parameters()):
                    param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

            total_loss += loss.item()

        scheduler.step()

        total_loss /= len(loader)

        if epoch % checkpoint_print == 0:
            print(
                f"Epoch {epoch}, lr: {optimizer.param_groups[0]['lr']:.3g} "
                f"- JEPA Loss: {total_loss:.4f}"
            )

        # Save model's checkpoint
        if epoch % checkpoint_save == 0 and epoch != 0:
            save_model(encoder, epoch)
