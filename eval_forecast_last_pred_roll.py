"""
    Script to run the short-term forecasting task.
    ---
        We consider the horizon and then predict a single value.
"""

from config.config_downstream import config

import torch
import json
import copy
import logging
import argparse
import pickle

from main.utils import prepare_args
from main.utils import mse, mae, _reduce

import numpy as np
import random

from src.data_loaders.data_loader_roll import get_jepa_loaders, get_evaluation_loaders
from src.models.encoder import Encoder
from src.models.decoder import LinearDecoder

import warnings

warnings.filterwarnings("ignore")

def mase_score(y_true, y_pred, insample, seasonality=5, eps=1e-8):
    """
    Mean Absolute Scaled Error.

    y_true: future target, shape [H]
    y_pred: prediction, shape [H]
    insample: historical values before cutoff
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    insample = np.asarray(insample)

    mae_model = np.mean(np.abs(y_true - y_pred))

    if len(insample) <= seasonality:
        return np.nan

    naive_error = np.mean(
        np.abs(insample[seasonality:] - insample[:-seasonality])
    )

    return mae_model / (naive_error + eps)


def zero_model_prediction(horizon):
    """
    For normalized return data, zero is a meaningful baseline.
    """
    return np.zeros(horizon)

def get_context_embedding(encoded_patches, eval_type):
    if eval_type == "last":
        return encoded_patches[:, -1, :]
    elif eval_type == "mean":
        return encoded_patches.mean(dim=1)
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")

if __name__ == "__main__":
    # Parse the args and get the config setup
    config = prepare_args(config)

    # Define some parameters
    # num_epochs = 100
    context = 24
    num_patches = 12
    patch_size = 5

    # Load device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Init Encoder, Decoder, Optimizer

    # Load Data
    print("Load data")
    config["path_data"] = "./data/" + config["data"] + "/" + config["data"] + ".csv"

    loader = get_evaluation_loaders(
        config["path_data"],
        config["batch_size"],
        config["ratio_patches"],
        config["mask_ratio"],
        patch_size=patch_size,
        context_size=num_patches,
    )
    sample_context, sample_target = loader.dataset[0]
    print("sample_context.shape =", sample_context.shape)
    print("sample_target.shape =", sample_target.shape)
    input_dim = len(loader.dataset[0][0][0])#
    # Encoder
    encoder = Encoder(
        num_patches=num_patches,
        dim_in=input_dim,
        kernel_size=config["pretrain_encoder_kernel_size"],
        embed_dim=config["pretrain_encoder_embed_dim"],
        embed_bias=config["pretrain_encoder_embed_bias"],
        nhead=config["pretrain_encoder_nhead"],
        num_layers=config["pretrain_encoder_num_layers"],
        jepa=True,
    )

    decoder = torch.nn.Sequential(
        torch.nn.Linear(config["pretrain_encoder_embed_dim"], 128),
        torch.nn.GELU(),
        torch.nn.Linear(128, 5)
    )
    # decoder = torch.nn.Sequential(
    #     torch.nn.Linear(config["pretrain_encoder_embed_dim"], 256),
    #     torch.nn.GELU(),
    #     torch.nn.Linear(256, 128),
    #     torch.nn.GELU(),
    #     torch.nn.Linear(128, 5)
    # )    
    # Load the pretrained model
    # path_name = "lr_" + str(config["lr_pretrain"]) \
    #         + "_encoder_" + str(config["pretrain_encoder_embed_dim"]) + "_" \
    #         + str(config["pretrain_encoder_nhead"]) + "_" \
    #         + str(config["pretrain_encoder_num_layers"]) \
    #         + "_epoch_" + str(config["checkpoint_to_use"])

    path_name = (
        "/lr_"
        + str(config["lr_pretrain"])
        + "_ema_momentum_"
        + str(config["ema_pretrain"])
        + "_mask_ratio_"
        + str(config["mask_ratio"])
        + "_ratio_patches_"
        + str(config["ratio_patches"])
        + "_encoder_"
        + str(config["pretrain_encoder_embed_dim"])
        + "_"
        + str(config["pretrain_encoder_nhead"])
        + "_"
        + str(config["pretrain_encoder_num_layers"])
        + "_predictor_"
        + str(config["pretrain_decoder_embed_dim"])
        + "_"
        + str(config["pretrain_decoder_nhead"])
        + "_"
        + str(config["pretrain_decoder_num_layers"])
        + "_epoch_"
        + str(config["checkpoint_to_use"])
    )

    name_loader = torch.load(
        config["path_save"] + path_name + ".pt", map_location=torch.device("cpu")
    )["encoder"]
    encoder.load_state_dict(name_loader)
    print("Model loaded")

    # We consider training only the decoder head
    param_groups = [{"params": (p for n, p in decoder.named_parameters())}]

    # optimizer = torch.optim.AdamW(param_groups, lr=config["lr"])
    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": 1e-4},
        {"params": decoder.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)
    # We train the model on the train set
    print("start train")
    loss_history = []
    patch_size = 5

    for epoch in range(config["num_epochs"]):
        encoder.eval()
        decoder.train()

        total_loss = 0.0

        for context_patches, target_patch in loader:
            # context_patches = context_patches.to(device)
            # target_patch = target_patch.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                encoded_patches = encoder(context_patches)

            context_embedding = get_context_embedding(
                encoded_patches,
                config["eval_type"]
            )

            predicted_next_patch = decoder(context_embedding)

            mse_loss = torch.nn.functional.mse_loss(
                predicted_next_patch,
                target_patch,
                reduction="mean",
            )

            pred_diff = predicted_next_patch[:, 1:] - predicted_next_patch[:, :-1]
            true_diff = target_patch[:, 1:] - target_patch[:, :-1]

            trend_loss = torch.nn.functional.l1_loss(
                torch.sign(pred_diff),
                torch.sign(true_diff),
            )

            loss = mse_loss + 0.0 * trend_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} - Total loss: {avg_loss:.6f}")
    save_path = "./results/loss.txt"

    with open(save_path, "w") as f:
        for epoch, loss in enumerate(loss_history):
            f.write(f"{epoch},{loss}\n")

    print(f"Loss saved to {save_path}")

    # =========================
    # Impermanent-style rolling evaluation
    # =========================

    encoder.eval()
    decoder.eval()

    patch_size = 5
    num_patches = 12
    context_len = num_patches * patch_size
    horizon = patch_size

    # full normalized series
    full_series = loader.dataset.full_series

    train_len = loader.dataset.train_len
    val_len = loader.dataset.val_len
    test_start = train_len + val_len
    test_end = len(full_series)

    # first cutoff is right before test period
    # cutoff means: last observed index
    first_cutoff = test_start - 1

    # We predict one patch ahead, so step is patch_size.
    step_size = patch_size

    l_val_mse = []
    l_val_mae = []
    l_val_mase = []
    l_zero_mase = []
    l_scaled_mase = []

    all_preds = []
    all_targets = []
    cutoff_list = []

    with torch.no_grad():
        for cutoff in range(first_cutoff, test_end - horizon, step_size):

            # context must be strictly before or equal to cutoff
            context_start = cutoff - context_len + 1

            if context_start < 0:
                continue

            context_series = full_series[context_start:cutoff + 1]

            target_series = full_series[cutoff + 1:cutoff + 1 + horizon]

            if len(context_series) != context_len:
                continue

            if len(target_series) != horizon:
                continue

            current_context = context_series.reshape(
                num_patches,
                patch_size,
            ).unsqueeze(0)

            target_patch = target_series.to(device)

            encoded_patches = encoder(current_context)

            context_embedding = get_context_embedding(
                encoded_patches,
                config["eval_type"],
            )

            predicted_next_patch = decoder(context_embedding)

            pred_np = predicted_next_patch.flatten().detach().cpu().numpy()
            target_np = target_patch.flatten().detach().cpu().numpy()

            # historical values before cutoff for MASE denominator
            insample_np = full_series[:cutoff + 1].detach().cpu().numpy()

            # ZeroModel baseline
            zero_pred_np = zero_model_prediction(horizon)

            val_mse = np.mean((pred_np - target_np) ** 2)
            val_mae = np.mean(np.abs(pred_np - target_np))

            val_mase = mase_score(
                y_true=target_np,
                y_pred=pred_np,
                insample=insample_np,
                seasonality=5,
            )

            zero_mase = mase_score(
                y_true=target_np,
                y_pred=zero_pred_np,
                insample=insample_np,
                seasonality=5,
            )

            scaled_mase = val_mase / max(zero_mase, 1e-8)

            l_val_mse.append(val_mse)
            l_val_mae.append(val_mae)
            l_val_mase.append(val_mase)
            l_zero_mase.append(zero_mase)
            l_scaled_mase.append(scaled_mase)

            all_preds.append(pred_np)
            all_targets.append(target_np)
            cutoff_list.append(cutoff)

    print("Rolling evaluation finished")
    print("Number of cutoffs:", len(cutoff_list))
    print("MSE:", np.nanmean(l_val_mse))
    print("MAE:", np.nanmean(l_val_mae))
    print("MASE:", np.nanmean(l_val_mase))
    print("ZeroModel MASE:", np.nanmean(l_zero_mase))
    print("Scaled MASE:", np.nanmean(l_scaled_mase))

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    pred_series = np.concatenate(all_preds)
    target_series = np.concatenate(all_targets)

    trend_pred = np.sign(np.diff(pred_series))
    trend_true = np.sign(np.diff(target_series))

    trend_accuracy = (trend_pred == trend_true).mean()

    print("Trend Accuracy is: {:.4f}".format(trend_accuracy))
    import pandas as pd

    results_df = pd.DataFrame({
        "cutoff": cutoff_list,
        "mse": l_val_mse,
        "mae": l_val_mae,
        "mase": l_val_mase,
        "zero_mase": l_zero_mase,
        "scaled_mase": l_scaled_mase,
    })

    result_path = "./results/" + config["eval_type"] + "_rolling_eval.csv"
    results_df.to_csv(result_path, index=False)

    print(f"Rolling eval results saved to {result_path}") 

    import matplotlib.pyplot as plt
    from datetime import datetime
    plt.figure(figsize=(12, 5))
    plt.plot(target_series, label="Ground Truth")
    plt.plot(pred_series, label="Prediction")
    plt.legend()
    plt.title("Rolling Prediction vs Ground Truth")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_png_path = f"./results/"+config["eval_type"]+f"_rolling_prediction_{timestamp}.png"
    plt.savefig(pred_png_path, dpi=300, bbox_inches="tight")    
    # plt.savefig(
    #     "./results/" + config["eval_type"] + "_rolling_prediction.png",
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    plt.show()    