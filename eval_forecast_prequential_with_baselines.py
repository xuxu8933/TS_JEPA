"""
    Script to run the short-term forecasting task.
    ---
    Impermanent-style downstream evaluation:
        - pretrain encoder checkpoint is loaded
        - encoder is frozen
        - decoder is trained only on train split
        - validation is done on val split
        - final rolling evaluation is done on test split
"""

from config.config_downstream import config

import torch
import logging
import warnings
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from datetime import datetime
import imageio.v2 as imageio

from main.utils import prepare_args
from main.utils import mse, mae

from src.data_loaders.data_loader_roll import get_evaluation_loaders
from src.models.encoder import Encoder
from src.models.decoder import LinearDecoder

warnings.filterwarnings("ignore")

def visualize_all_rolling_predictions_as_series(
    all_preds,
    all_targets,
    config,
    save_dir="./results",
):
    os.makedirs(save_dir, exist_ok=True)

    pred_series = all_preds.reshape(-1)
    target_series = all_targets.reshape(-1)

    plt.figure(figsize=(14, 5))
    plt.plot(target_series, label="Ground Truth")
    plt.plot(pred_series, label="Prediction")
    plt.legend()
    plt.xlabel("Time index in test rolling prediction")
    plt.ylabel("Normalized price / return")
    plt.title("All Rolling Forecasts: Prediction vs Ground Truth")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_path = os.path.join(
        save_dir,
        config["eval_type"] + f"_all_rolling_predictions_{timestamp}.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"All rolling prediction figure saved to {save_path}")

def visualize_all_rolling_windows(
    encoder,
    decoder,
    dataset,
    device,
    eval_type,
    config,
    save_dir="./results/rolling_windows",
    max_samples=None,
    make_gif=True,
    gif_name="rolling_windows.gif",
    gif_duration=0.5,
):
    os.makedirs(save_dir, exist_ok=True)

    encoder.eval()
    decoder.eval()

    if max_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(len(dataset), max_samples)

    print(f"Visualizing {num_samples} rolling windows...")

    saved_paths = []

    for sample_idx in range(num_samples):
        context_patches, target_patch = dataset[sample_idx]

        context_patches_input = context_patches.unsqueeze(0).to(device)

        with torch.no_grad():
            encoded_patches = encoder(context_patches_input)

            context_embedding = get_context_embedding(
                encoded_patches,
                eval_type,
            )

            pred_patch = decoder(context_embedding)

        context_np = context_patches.flatten().cpu().numpy()
        target_np = target_patch.cpu().numpy()
        pred_np = pred_patch.flatten().detach().cpu().numpy()

        context_x = np.arange(len(context_np))
        target_x = np.arange(
            len(context_np),
            len(context_np) + len(target_np)
        )

        plt.figure(figsize=(12, 5))

        plt.plot(context_x, context_np, label="Context", linewidth=2)
        plt.plot(target_x, target_np, marker="o", label="Ground Truth Future")
        plt.plot(target_x, pred_np, marker="o", label="Predicted Future")

        plt.axvline(
            x=len(context_np) - 1,
            linestyle="--",
            label="Forecast Origin",
        )

        plt.legend()
        plt.xlabel("Time index inside rolling window")
        plt.ylabel("Normalized price / return")
        plt.title(f"Rolling Forecast Window #{sample_idx}")

        save_path = os.path.join(
            save_dir,
            f"{config['eval_type']}_rolling_window_{sample_idx:04d}.png"
        )

        plt.savefig(save_path, dpi=150)
        plt.close()

        saved_paths.append(save_path)

    print(f"Saved {num_samples} rolling window figures to: {save_dir}")

    # 自动生成 GIF
    if make_gif and len(saved_paths) > 0:
        gif_path = os.path.join(save_dir, gif_name)
        frames = [imageio.imread(p) for p in saved_paths]
        imageio.mimsave(gif_path, frames, duration=gif_duration)
        print(f"GIF saved to: {gif_path}")

def visualize_one_rolling_window(
    encoder,
    decoder,
    dataset,
    sample_idx,
    device,
    eval_type,
    config,
):
    encoder.eval()
    decoder.eval()

    context_patches, target_patch = dataset[sample_idx]

    context_patches_input = context_patches.unsqueeze(0).to(device)

    with torch.no_grad():
        encoded_patches = encoder(context_patches_input)

        context_embedding = get_context_embedding(
            encoded_patches,
            eval_type,
        )

        pred_patch = decoder(context_embedding)

    context_np = context_patches.flatten().cpu().numpy()
    target_np = target_patch.cpu().numpy()
    pred_np = pred_patch.flatten().detach().cpu().numpy()

    context_x = np.arange(len(context_np))
    target_x = np.arange(len(context_np), len(context_np) + len(target_np))

    plt.figure(figsize=(12, 5))

    plt.plot(context_x, context_np, label="Context", linewidth=2)
    plt.plot(target_x, target_np, marker="o", label="Ground Truth Future")
    plt.plot(target_x, pred_np, marker="o", label="Predicted Future")

    plt.axvline(
        x=len(context_np) - 1,
        linestyle="--",
        label="Forecast Origin",
    )

    plt.legend()
    plt.xlabel("Time index inside rolling window")
    plt.ylabel("Normalized price / return")
    plt.title(f"Rolling Forecast Window #{sample_idx}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = (
        "./results/"
        + config["eval_type"]
        + f"_rolling_window_{sample_idx}_{timestamp}.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Rolling window figure saved to {save_path}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_context_embedding(encoded_patches, eval_type):
    if eval_type == "last":
        return encoded_patches[:, -1, :]
    elif eval_type == "mean":
        return encoded_patches.mean(dim=1)
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")



def _maybe_get_dataset_index(dataset, sample_idx, eval_stride, horizon_step):
    """
    Best-effort target index for logging.

    If the dataset exposes original indices/cutoffs, use them.
    Otherwise fall back to the index inside the rolling evaluation split.
    """
    # Common possible names. This keeps the evaluator robust to different
    # dataset implementations.
    for attr_name in ["cutoffs", "cutoff_indices", "start_indices", "indices"]:
        if hasattr(dataset, attr_name):
            values = getattr(dataset, attr_name)
            try:
                return int(values[sample_idx]) + int(horizon_step)
            except Exception:
                pass

    return int(sample_idx * eval_stride + horizon_step)



def make_baseline_prediction(context_patches, horizon, baseline_name):
    """
    Produce a baseline forecast from the same historical context window.

    Baselines:
        - naive_last: repeat the last observed value for all future steps
        - previous_patch: reuse the last context patch as the next patch
        - mean_context: repeat the mean of the full context window
        - drift: linear extrapolation from first to last context value
    """
    context_np = context_patches.detach().cpu().numpy()
    context_flat = context_np.reshape(-1)

    if baseline_name == "naive_last":
        return np.repeat(context_flat[-1], horizon)

    if baseline_name == "previous_patch":
        last_patch = context_np[-1].reshape(-1)
        if len(last_patch) >= horizon:
            return last_patch[:horizon]
        return np.resize(last_patch, horizon)

    if baseline_name == "mean_context":
        return np.repeat(context_flat.mean(), horizon)

    if baseline_name == "drift":
        if len(context_flat) <= 1:
            return np.repeat(context_flat[-1], horizon)
        slope = (context_flat[-1] - context_flat[0]) / (len(context_flat) - 1)
        steps = np.arange(1, horizon + 1)
        return context_flat[-1] + slope * steps

    raise ValueError(f"Unknown baseline_name: {baseline_name}")


def prequential_baseline_evaluate(
    dataset,
    config,
    baseline_names=None,
    save_dir="./results",
):
    """
    Evaluate simple baselines in the same prequential rolling setting.

    Each baseline receives exactly the same context window and target patch as
    the TS-JEPA model, so the comparison is fair and deployment-faithful.
    """
    os.makedirs(save_dir, exist_ok=True)

    if baseline_names is None:
        baseline_names = [
            "naive_last",
            "previous_patch",
            "mean_context",
            "drift",
        ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_stride = int(config.get("eval_stride", 1))

    all_forecast_rows = []
    all_score_rows = []
    summary_rows = []
    prediction_store = {}
    target_store = None

    for baseline_name in baseline_names:
        baseline_preds = []
        baseline_targets = []

        for sample_idx in range(len(dataset)):
            context_patches, target_patch = dataset[sample_idx]

            true_np = target_patch.detach().cpu().numpy().reshape(-1)
            horizon = len(true_np)

            # Forecast using only historical context.
            pred_np = make_baseline_prediction(
                context_patches=context_patches,
                horizon=horizon,
                baseline_name=baseline_name,
            ).reshape(-1)

            # Lock forecast before scoring.
            for horizon_step, pred_value in enumerate(pred_np):
                target_index = _maybe_get_dataset_index(
                    dataset=dataset,
                    sample_idx=sample_idx,
                    eval_stride=eval_stride,
                    horizon_step=horizon_step,
                )

                all_forecast_rows.append({
                    "model": baseline_name,
                    "rolling_step": sample_idx,
                    "horizon_step": horizon_step + 1,
                    "target_index": target_index,
                    "predicted_value": float(pred_value),
                    "locked": True,
                })

            # Observation arrives; now compute score.
            for horizon_step, (pred_value, true_value) in enumerate(
                zip(pred_np, true_np)
            ):
                error = float(pred_value - true_value)
                target_index = _maybe_get_dataset_index(
                    dataset=dataset,
                    sample_idx=sample_idx,
                    eval_stride=eval_stride,
                    horizon_step=horizon_step,
                )

                all_score_rows.append({
                    "model": baseline_name,
                    "rolling_step": sample_idx,
                    "horizon_step": horizon_step + 1,
                    "target_index": target_index,
                    "predicted_value": float(pred_value),
                    "true_value": float(true_value),
                    "error": error,
                    "absolute_error": abs(error),
                    "squared_error": error ** 2,
                })

            baseline_preds.append(pred_np[None, :])
            baseline_targets.append(true_np[None, :])

        baseline_preds = np.concatenate(baseline_preds, axis=0)
        baseline_targets = np.concatenate(baseline_targets, axis=0)

        prediction_store[baseline_name] = baseline_preds
        target_store = baseline_targets

        summary_rows.append({
            "model": baseline_name,
            "mse": float(np.mean((baseline_preds - baseline_targets) ** 2)),
            "mae": float(np.mean(np.abs(baseline_preds - baseline_targets))),
            "trend_accuracy": compute_trend_accuracy(
                all_preds=baseline_preds,
                all_targets=baseline_targets,
            ),
        })

    import csv

    baseline_forecast_csv_path = os.path.join(
        save_dir,
        config["eval_type"] + f"_baseline_locked_forecasts_{timestamp}.csv",
    )
    baseline_score_csv_path = os.path.join(
        save_dir,
        config["eval_type"] + f"_baseline_scores_after_observation_{timestamp}.csv",
    )
    baseline_summary_csv_path = os.path.join(
        save_dir,
        config["eval_type"] + f"_baseline_summary_{timestamp}.csv",
    )

    with open(baseline_forecast_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "rolling_step",
                "horizon_step",
                "target_index",
                "predicted_value",
                "locked",
            ],
        )
        writer.writeheader()
        writer.writerows(all_forecast_rows)

    with open(baseline_score_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "rolling_step",
                "horizon_step",
                "target_index",
                "predicted_value",
                "true_value",
                "error",
                "absolute_error",
                "squared_error",
            ],
        )
        writer.writeheader()
        writer.writerows(all_score_rows)

    with open(baseline_summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "mse", "mae", "trend_accuracy"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Baseline locked forecasts saved to: {baseline_forecast_csv_path}")
    print(f"Baseline scores saved to: {baseline_score_csv_path}")
    print(f"Baseline summary saved to: {baseline_summary_csv_path}")

    return (
        summary_rows,
        prediction_store,
        target_store,
        baseline_forecast_csv_path,
        baseline_score_csv_path,
        baseline_summary_csv_path,
    )


def save_model_comparison(
    model_rows,
    config,
    save_dir="./results",
):
    """Save and print a comparison table for TS-JEPA and baselines."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_csv_path = os.path.join(
        save_dir,
        config["eval_type"] + f"_model_comparison_{timestamp}.csv",
    )

    # Lower MSE is better; sort accordingly.
    model_rows = sorted(model_rows, key=lambda row: row["mse"])

    import csv
    with open(comparison_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "mse", "mae", "trend_accuracy"],
        )
        writer.writeheader()
        writer.writerows(model_rows)

    print("========== Model Comparison ==========")
    for row in model_rows:
        print(
            f"{row['model']}: "
            f"MSE={row['mse']:.6f}, "
            f"MAE={row['mae']:.6f}, "
            f"TrendAcc={row['trend_accuracy']:.4f}"
        )
    print(f"Model comparison saved to: {comparison_csv_path}")

    return comparison_csv_path


def visualize_model_comparison(model_rows, config, save_dir="./results"):
    """Bar plots for MSE and MAE of TS-JEPA vs baselines."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_rows = sorted(model_rows, key=lambda row: row["mse"])
    names = [row["model"] for row in model_rows]
    mses = [row["mse"] for row in model_rows]
    maes = [row["mae"] for row in model_rows]

    plt.figure(figsize=(10, 5))
    plt.bar(names, mses)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("MSE")
    plt.title("Prequential Rolling Evaluation: MSE Comparison")
    plt.tight_layout()
    mse_png_path = os.path.join(
        save_dir,
        config["eval_type"] + f"_model_comparison_mse_{timestamp}.png",
    )
    plt.savefig(mse_png_path, dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar(names, maes)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("MAE")
    plt.title("Prequential Rolling Evaluation: MAE Comparison")
    plt.tight_layout()
    mae_png_path = os.path.join(
        save_dir,
        config["eval_type"] + f"_model_comparison_mae_{timestamp}.png",
    )
    plt.savefig(mae_png_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"MSE comparison figure saved to: {mse_png_path}")
    print(f"MAE comparison figure saved to: {mae_png_path}")

    return mse_png_path, mae_png_path

def prequential_rolling_evaluate(
    encoder,
    decoder,
    dataset,
    device,
    eval_type,
    config,
    save_dir="./results",
):
    """
    Deployment-faithful / Impermanent-style rolling evaluation.

    For each cutoff:
        1. read context only
        2. forecast next h points
        3. store forecast rows as locked forecasts
        4. then compare with target_patch and store score rows

    In offline experiments target_patch is already available in memory,
    but the code order keeps the semantics explicit:
        forecast first, score later.
    """
    os.makedirs(save_dir, exist_ok=True)

    encoder.eval()
    decoder.eval()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    forecast_csv_path = os.path.join(
        save_dir,
        config["eval_type"] + f"_locked_forecasts_{timestamp}.csv",
    )
    score_csv_path = os.path.join(
        save_dir,
        config["eval_type"] + f"_scores_after_observation_{timestamp}.csv",
    )

    eval_stride = int(config.get("eval_stride", 1))

    forecast_rows = []
    score_rows = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for sample_idx in range(len(dataset)):
            context_patches, target_patch = dataset[sample_idx]

            # =========================
            # 1. Context before cutoff
            # =========================
            context_patches_input = context_patches.unsqueeze(0).to(device)

            # =========================
            # 2. Forecast before using truth
            # =========================
            encoded_patches = encoder(context_patches_input)
            context_embedding = get_context_embedding(
                encoded_patches,
                eval_type,
            )
            predicted_next_patch = decoder(context_embedding)

            pred_np = predicted_next_patch.squeeze(0).detach().cpu().numpy()

            # =========================
            # 3. Lock forecast
            # =========================
            for horizon_step, pred_value in enumerate(pred_np):
                target_index = _maybe_get_dataset_index(
                    dataset=dataset,
                    sample_idx=sample_idx,
                    eval_stride=eval_stride,
                    horizon_step=horizon_step,
                )

                forecast_rows.append({
                    "rolling_step": sample_idx,
                    "horizon_step": horizon_step + 1,
                    "target_index": target_index,
                    "predicted_value": float(pred_value),
                    "locked": True,
                })

            # =========================
            # 4. Observation arrives
            # =========================
            true_np = target_patch.detach().cpu().numpy()

            all_preds.append(pred_np[None, :])
            all_targets.append(true_np[None, :])

            # =========================
            # 5. Score after observation
            # =========================
            for horizon_step, (pred_value, true_value) in enumerate(
                zip(pred_np, true_np)
            ):
                error = float(pred_value - true_value)
                target_index = _maybe_get_dataset_index(
                    dataset=dataset,
                    sample_idx=sample_idx,
                    eval_stride=eval_stride,
                    horizon_step=horizon_step,
                )

                score_rows.append({
                    "rolling_step": sample_idx,
                    "horizon_step": horizon_step + 1,
                    "target_index": target_index,
                    "predicted_value": float(pred_value),
                    "true_value": float(true_value),
                    "error": error,
                    "absolute_error": abs(error),
                    "squared_error": error ** 2,
                })

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # =========================
    # Save locked forecasts
    # =========================
    import csv

    with open(forecast_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rolling_step",
                "horizon_step",
                "target_index",
                "predicted_value",
                "locked",
            ],
        )
        writer.writeheader()
        writer.writerows(forecast_rows)

    # =========================
    # Save scores after observation
    # =========================
    with open(score_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rolling_step",
                "horizon_step",
                "target_index",
                "predicted_value",
                "true_value",
                "error",
                "absolute_error",
                "squared_error",
            ],
        )
        writer.writeheader()
        writer.writerows(score_rows)

    test_mse = float(np.mean((all_preds - all_targets) ** 2))
    test_mae = float(np.mean(np.abs(all_preds - all_targets)))

    print(f"Locked forecasts saved to: {forecast_csv_path}")
    print(f"Scores saved to: {score_csv_path}")

    return test_mse, test_mae, all_preds, all_targets, forecast_csv_path, score_csv_path


def evaluate_model(
    encoder,
    decoder,
    loader,
    device,
    eval_type,
):
    encoder.eval()
    decoder.eval()

    l_mse = []
    l_mae = []

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for context_patches, target_patch in loader:
            context_patches = context_patches.to(device)
            target_patch = target_patch.to(device)

            encoded_patches = encoder(context_patches)

            context_embedding = get_context_embedding(
                encoded_patches,
                eval_type,
            )

            predicted_next_patch = decoder(context_embedding)

            pred_np = predicted_next_patch.detach().cpu().numpy()
            target_np = target_patch.detach().cpu().numpy()

            all_preds.append(pred_np)
            all_targets.append(target_np)

            batch_mse = mse(
                pred_np.reshape(-1),
                target_np.reshape(-1),
            )

            batch_mae = mae(
                pred_np.reshape(-1),
                target_np.reshape(-1),
            )

            l_mse.append(batch_mse)
            l_mae.append(batch_mae)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mean_mse = float(np.mean(l_mse))
    mean_mae = float(np.mean(l_mae))

    return mean_mse, mean_mae, all_preds, all_targets


def compute_trend_accuracy(all_preds, all_targets):
    """
    Trend accuracy based on within-patch direction.

    For each predicted target patch:

        pred_diff = pred[:, 1:] - pred[:, :-1]
        true_diff = true[:, 1:] - true[:, :-1]

    Accuracy checks whether predicted direction matches true direction.
    """
    pred_diff = all_preds[:, 1:] - all_preds[:, :-1]
    true_diff = all_targets[:, 1:] - all_targets[:, :-1]

    trend_pred = np.sign(pred_diff)
    trend_true = np.sign(true_diff)

    trend_accuracy = (trend_pred == trend_true).mean()

    return float(trend_accuracy)


if __name__ == "__main__":
    set_seed(42)

    # =========================
    # Config
    # =========================

    config = prepare_args(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    config["path_data"] = (
        "./data/" + config["data"] + "/" + config["data"] + ".csv"
    )

    print("Load data from:", config["path_data"])

    # =========================
    # Important parameters
    # =========================
    # These should match your pretraining patch setting.
    patch_size = 5

    # Number of context patches for downstream forecasting.
    # For example:
    # context_size=10 means 10 patches * 5 days = 50 days context.
    context_size = config.get("context_size", 12)

    # For rolling evaluation, stride=1 gives day-by-day rolling.
    # If you want patch-by-patch rolling, use stride=patch_size.
    eval_stride = config.get("eval_stride", 5)

    # =========================
    # Data loaders
    # =========================

    train_loader = get_evaluation_loaders(
        config["path_data"],
        config["batch_size"],
        config["ratio_patches"],
        config["mask_ratio"],
        split="train",
        patch_size=patch_size,
        context_size=context_size,
        stride=eval_stride,
        normalize=True,
    )

    val_loader = get_evaluation_loaders(
        config["path_data"],
        config["batch_size"],
        config["ratio_patches"],
        config["mask_ratio"],
        split="val",
        patch_size=patch_size,
        context_size=context_size,
        stride=eval_stride,
        normalize=True,
    )

    test_loader = get_evaluation_loaders(
        config["path_data"],
        config["batch_size"],
        config["ratio_patches"],
        config["mask_ratio"],
        split="test",
        patch_size=patch_size,
        context_size=context_size,
        stride=eval_stride,
        normalize=True,
    )

    sample_context, sample_target = train_loader.dataset[0]

    print("sample_context.shape =", sample_context.shape)
    print("sample_target.shape =", sample_target.shape)

    num_patches = sample_context.shape[0]
    input_dim = sample_context.shape[1]

    print("num_patches =", num_patches)
    print("input_dim / patch_size =", input_dim)

    # =========================
    # Model
    # =========================

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

    decoder = LinearDecoder(
        emb_dim=config["pretrain_encoder_embed_dim"],
        patch_size=patch_size,
    )

    encoder.to(device)
    decoder.to(device)

    # =========================
    # Load pretrained encoder
    # =========================

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

    checkpoint_path = config["path_save"] + path_name + ".pt"

    print("Load checkpoint:", checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    encoder.load_state_dict(checkpoint["encoder"])

    print("Pretrained encoder loaded")

    # Freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False

    encoder.eval()

    # Train only decoder
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=config["lr"],
    )

    # =========================
    # Train decoder
    # =========================

    print("Start downstream decoder training")

    loss_history = []
    val_mse_history = []
    val_mae_history = []

    best_val_mse = float("inf")
    best_decoder_state = None

    for epoch in range(config["num_epochs"]):
        encoder.eval()
        decoder.train()

        total_loss = 0.0

        for context_patches, target_patch in train_loader:
            optimizer.zero_grad()

            context_patches = context_patches.to(device)
            target_patch = target_patch.to(device)

            with torch.no_grad():
                encoded_patches = encoder(context_patches)

            context_embedding = get_context_embedding(
                encoded_patches,
                config["eval_type"],
            )

            predicted_next_patch = decoder(context_embedding)

            mse_loss = torch.nn.functional.mse_loss(
                predicted_next_patch,
                target_patch,
                reduction="mean",
            )

            pred_diff = predicted_next_patch[:, 1:] - predicted_next_patch[:, :-1]
            true_diff = target_patch[:, 1:] - target_patch[:, :-1]

            trend_loss = torch.relu(
                -pred_diff * true_diff
            ).mean()

            # You can change this weight later.
            trend_weight = config.get("trend_weight", 0.0)

            loss = mse_loss + trend_weight * trend_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        loss_history.append(avg_train_loss)

        # =========================
        # Validation
        # =========================

        val_mse, val_mae, _, _ = evaluate_model(
            encoder=encoder,
            decoder=decoder,
            loader=val_loader,
            device=device,
            eval_type=config["eval_type"],
        )

        val_mse_history.append(val_mse)
        val_mae_history.append(val_mae)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_decoder_state = {
                k: v.detach().cpu().clone()
                for k, v in decoder.state_dict().items()
            }

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: "
                f"train_loss={avg_train_loss:.6f}, "
                f"val_mse={val_mse:.6f}, "
                f"val_mae={val_mae:.6f}"
            )

    # Restore best decoder
    if best_decoder_state is not None:
        decoder.load_state_dict(best_decoder_state)
        decoder.to(device)

    # =========================
    # Save training history
    # =========================

    save_path = "./results/loss.txt"

    with open(save_path, "w") as f:
        f.write("epoch,train_loss,val_mse,val_mae\n")
        for epoch in range(len(loss_history)):
            f.write(
                f"{epoch},"
                f"{loss_history[epoch]},"
                f"{val_mse_history[epoch]},"
                f"{val_mae_history[epoch]}\n"
            )

    print(f"Loss saved to {save_path}")

    # =========================
    # Final test on future test split
    # =========================

    test_mse, test_mae, all_preds, all_targets, forecast_csv_path, score_csv_path = (
        prequential_rolling_evaluate(
            encoder=encoder,
            decoder=decoder,
            dataset=test_loader.dataset,
            device=device,
            eval_type=config["eval_type"],
            config=config,
            save_dir="./results",
        )
    )

    trend_accuracy = compute_trend_accuracy(
        all_preds=all_preds,
        all_targets=all_targets,
    )

    print("========== Final Test ==========")
    print("TS-JEPA Test MSE is: {:.6f}".format(test_mse))
    print("TS-JEPA Test MAE is: {:.6f}".format(test_mae))
    print("TS-JEPA Trend Accuracy is: {:.4f}".format(trend_accuracy))

    # =========================
    # Baseline evaluation
    # =========================

    baseline_summary_rows, baseline_preds, baseline_targets, _, _, _ = (
        prequential_baseline_evaluate(
            dataset=test_loader.dataset,
            config=config,
            baseline_names=config.get(
                "baseline_names",
                [
                    "naive_last",
                    "previous_patch",
                    "mean_context",
                    "drift",
                ],
            ),
            save_dir="./results",
        )
    )

    model_comparison_rows = [
        {
            "model": "TS-JEPA",
            "mse": test_mse,
            "mae": test_mae,
            "trend_accuracy": trend_accuracy,
        }
    ] + baseline_summary_rows

    comparison_csv_path = save_model_comparison(
        model_rows=model_comparison_rows,
        config=config,
        save_dir="./results",
    )

    visualize_model_comparison(
        model_rows=model_comparison_rows,
        config=config,
        save_dir="./results",
    )

    # =========================
    # Visualization 1:
    # Rolling error over time
    # =========================

    rolling_mse = ((all_preds - all_targets) ** 2).mean(axis=1)
    rolling_mae = np.abs(all_preds - all_targets).mean(axis=1)

    plt.figure(figsize=(12, 5))
    plt.plot(rolling_mse, label="Rolling MSE")
    plt.plot(rolling_mae, label="Rolling MAE")
    plt.legend()
    plt.xlabel("Rolling evaluation step")
    plt.ylabel("Error")
    plt.title("Rolling Forecast Error over Time")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metric_png_path = (
        "./results/"
        + config["eval_type"]
        + f"_rolling_metrics_test_{timestamp}.png"
    )

    plt.savefig(metric_png_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Rolling metric figure saved to {metric_png_path}")


    # =========================
    # Visualization 2:
    # Last point of each predicted patch
    # =========================

    pred_last = all_preds[:, -1]
    true_last = all_targets[:, -1]

    plt.figure(figsize=(12, 5))
    plt.plot(true_last, label="Ground Truth")
    plt.plot(pred_last, label="Prediction")
    plt.legend()
    plt.xlabel("Rolling evaluation step")
    plt.ylabel("Normalized price / return")
    plt.title("Rolling Forecast: Last Point of Each Predicted Patch")

    last_point_png_path = (
        "./results/"
        + config["eval_type"]
        + f"_rolling_last_point_test_{timestamp}.png"
    )

    plt.savefig(last_point_png_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Rolling last-point figure saved to {last_point_png_path}")


    # =========================
    # Visualization 3:
    # Example rolling window
    # =========================

    # visualize_one_rolling_window(
    #     encoder=encoder,
    #     decoder=decoder,
    #     dataset=test_loader.dataset,
    #     sample_idx=min(60, len(test_loader.dataset) - 1),
    #     device=device,
    #     eval_type=config["eval_type"],
    #     config=config,
    # )

    # visualize_all_rolling_windows(
    #     encoder=encoder,
    #     decoder=decoder,
    #     dataset=test_loader.dataset,
    #     device=device,
    #     eval_type=config["eval_type"],
    #     config=config,
    # )    
    visualize_all_rolling_windows(
        encoder=encoder,
        decoder=decoder,
        dataset=test_loader.dataset,
        device=device,
        eval_type=config["eval_type"],
        config=config,
        save_dir="./results/rolling_windows",
        max_samples=100,         # 或者 None 表示全部
        make_gif=True,
        gif_name="rolling_windows.gif",
        gif_duration=0.4,
    )
    visualize_all_rolling_predictions_as_series(
        all_preds=all_preds,
        all_targets=all_targets,
        config=config,
    )

    # # =========================
    # # Plot prediction vs ground truth
    # # =========================

    # pred_series = all_preds.reshape(-1)
    # target_series = all_targets.reshape(-1)

    # plt.figure(figsize=(12, 5))
    # plt.plot(target_series, label="Ground Truth")
    # plt.plot(pred_series, label="Prediction")
    # plt.legend()
    # plt.title(
    #     "Prediction vs Ground Truth "
    #     f"({config['eval_type']}, split=test)"
    # )

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # pred_png_path = (
    #     "./results/"
    #     + config["eval_type"]
    #     + f"_prediction_test_{timestamp}.png"
    # )

    # plt.savefig(pred_png_path, dpi=300, bbox_inches="tight")
    # plt.show()

    # print(f"Prediction figure saved to {pred_png_path}")