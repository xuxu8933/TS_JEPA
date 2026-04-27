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

from src.data_loaders.data_loader import get_jepa_loaders, get_evaluation_loaders
from src.models.encoder import Encoder
from src.models.decoder import LinearDecoder

import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # Parse the args and get the config setup
    config = prepare_args(config)

    # Define some parameters
    # num_epochs = 100
    context = 12
    num_patches = 12

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
        # encoder.eval()
        encoder.train()
        decoder.train()
        total_loss = 0
        for context_patches, target_patch in loader:
            optimizer.zero_grad()
            # with torch.no_grad():
            encoded_patches = encoder(context_patches)

            # summed_embedding = torch.sum(encoded_patches, dim=1)
            # context_embedding = encoded_patches[:, -1, :]
            context_embedding = encoded_patches.mean(dim=1)
            predicted_next_patch = decoder(context_embedding)
            # predicted_next_patch = decoder(summed_embedding)
            loss = torch.nn.functional.mse_loss(
                predicted_next_patch, target_patch, reduction="mean"
            )

            loss.backward()
            optimizer.step()

            # total_loss += loss / config["batch_size"]
            total_loss += loss.item()

            # total_loss += loss.item()
            # loss_value = total_loss.item() if torch.is_tensor(total_loss) else float(total_loss)
            # loss_history.append(loss_value)
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} - Total loss: {avg_loss}")
    save_path = "./results/loss.txt"

    with open(save_path, "w") as f:
        for epoch, loss in enumerate(loss_history):
            f.write(f"{epoch},{loss}\n")

    print(f"Loss saved to {save_path}")

    # =========================
    # Test on rolling prediction
    # =========================

    encoder.eval()
    decoder.eval()

    num_steps = (
        len(loader.dataset.test_df) - num_patches * patch_size
    ) // patch_size

    l_val_mse = []
    l_val_mae = []
    all_preds = []
    all_targets = []    
    # We test the model on the last prediction
    # We define the number of steps we will have
    # num_steps = (len(loader.dataset.test_df[context * num_patches :])) // context
    with torch.no_grad():
        for step in range(num_steps):
            current_series = loader.dataset.test_df[
                step * patch_size :
                step * patch_size + num_patches * patch_size
            ]

            current_context = current_series.reshape(
                num_patches,
                patch_size
            ).unsqueeze(0)

            target_value = loader.dataset.test_df[
                step * patch_size + num_patches * patch_size :
                step * patch_size + (num_patches + 1) * patch_size
            ]

            encoded_patches = encoder(current_context)

            # Same strategy as training
            # context_embedding = encoded_patches[:, -1, :]
            context_embedding = encoded_patches.mean(dim=1)

            predicted_next_patch = decoder(context_embedding)
            # predicted_next_patch = current_context[:, -1, :]
            all_preds.append(predicted_next_patch.flatten().cpu().numpy())
            all_targets.append(target_value.cpu().numpy())
            val_mse = mse(
                predicted_next_patch.flatten().detach().numpy(),
                target_value.numpy()
            )

            val_mae = mae(
                predicted_next_patch.flatten().detach().numpy(),
                target_value.numpy()
            )

            l_val_mse.append(val_mse)
            l_val_mae.append(val_mae)

    print("MSE Loss is: {}".format(np.mean(l_val_mse)))
    print("MAE Loss is: {}".format(np.mean(l_val_mae)))
    trend_pred = np.sign(np.diff(all_preds))
    trend_true = np.sign(np.diff(all_targets))

    accuracy = (trend_pred == trend_true).mean()

    print("Trend accuracy:", accuracy)
    import numpy as np

    pred_series = np.concatenate(all_preds)
    target_series = np.concatenate(all_targets)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.plot(target_series, label="Ground Truth")
    plt.plot(pred_series, label="Prediction")
    plt.legend()
    plt.title("Prediction vs Ground Truth")
    plt.show()    