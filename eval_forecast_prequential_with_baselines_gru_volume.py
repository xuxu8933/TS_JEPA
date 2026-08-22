"""
    Script to run the short-term forecasting task.
    ---
    Impermanent-style downstream evaluation:
        - pretrain encoder checkpoint is loaded
        - encoder is frozen or fine-tuned depending on config
        - decoder is trained on train split
        - validation is done on val split
        - final rolling evaluation is done on test split
"""

from config.config_downstream import config

import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import platform
import subprocess
from datetime import datetime
import imageio.v2 as imageio

from main.utils import prepare_args
from main.utils import mse, mae

from src.data_loaders.data_loader_roll_volume import get_evaluation_loaders
from src.models.encoder import Encoder
from src.models.decoder import build_reconstruction_decoder

import torch.nn as nn


class GRUForecastModel(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=5,
        dropout=0.1,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, context_patches):
        """
        context_patches:
            single feature: [batch, context_size, patch_size]
            multi feature:  [batch, context_size, patch_size * feature_dim]
        """
        x = context_patches.reshape(
            context_patches.size(0),
            -1,
            self.input_size,
        )

        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        pred = self.head(last_hidden)

        return pred

warnings.filterwarnings("ignore")


def data_title(config):
    return str(config.get("data", "unknown")).upper()


def forecast_axis_label(config):
    target = config.get("forecast_target", "value")
    if target == "cumulative_log_return":
        return "Cumulative log return from forecast origin"
    if target == "excess_log_return":
        return "Cumulative excess log return from forecast origin"
    if target == "relative_return":
        return "Relative return from forecast cutoff"
    return "Normalized target value"


def runtime_provenance():
    """Best-effort immutable environment metadata for result auditing."""
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    hardware = platform.platform()
    if torch.cuda.is_available():
        hardware += f"; GPU={torch.cuda.get_device_name(0)}"
    return {
        "git_commit_sha": git_commit,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "hardware": hardware,
    }


# =========================================================
# Multi-feature support, e.g. feature_cols=["Close", "Volume"]
# Loader should return context_patches as [B, context_size, patch_size * feature_dim]
# and target_patch either as [B, patch_size] or [B, patch_size * feature_dim].
# We forecast/score only target_feature_index, usually Close=0.
# =========================================================
def select_target_feature_tensor(patch, patch_size, feature_dim=1, target_feature_index=0):
    """Return target feature with shape [patch_size] or [B, patch_size]."""
    if feature_dim <= 1:
        return patch

    if patch.dim() == 1:
        if patch.numel() == patch_size:
            return patch
        return patch.reshape(patch_size, feature_dim)[:, target_feature_index]

    if patch.dim() == 2:
        if patch.shape[-1] == patch_size:
            return patch
        return patch.reshape(patch.shape[0], patch_size, feature_dim)[:, :, target_feature_index]

    raise ValueError(f"Unexpected target patch shape: {tuple(patch.shape)}")


def select_context_feature_numpy(context_patches, patch_size, feature_dim=1, target_feature_index=0):
    """Return one feature from context as a flat numpy series for baselines/plots."""
    context_np = context_patches.detach().cpu().numpy()

    if feature_dim <= 1:
        return context_np.reshape(-1)

    # [context_size, patch_size * feature_dim]
    if context_np.ndim == 2:
        return context_np.reshape(context_np.shape[0], patch_size, feature_dim)[:, :, target_feature_index].reshape(-1)

    # [batch, context_size, patch_size * feature_dim]
    if context_np.ndim == 3:
        return context_np.reshape(context_np.shape[0], context_np.shape[1], patch_size, feature_dim)[:, :, :, target_feature_index].reshape(context_np.shape[0], -1)

    raise ValueError(f"Unexpected context patch shape: {context_np.shape}")

def evaluate_gru_model(
    model,
    loader,
    device,
):
    model.eval()

    l_mse = []
    l_mae = []

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for context_patches, target_patch in loader:
            context_patches = context_patches.to(device)
            target_patch = target_patch.to(device)

            pred = model(context_patches)

            target_patch = select_target_feature_tensor(
                target_patch,
                patch_size=model.output_size,
                feature_dim=model.input_size,
                target_feature_index=0,
            )

            pred_np = pred.detach().cpu().numpy()
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

def prequential_gru_evaluate(
    model,
    dataset,
    device,
    config,
    save_dir="./results",
):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    forecast_csv_path = os.path.join(
        save_dir,
        config["eval_type"] + f"_gru_locked_forecasts_{timestamp}.csv",
    )

    score_csv_path = os.path.join(
        save_dir,
        config["eval_type"] + f"_gru_scores_after_observation_{timestamp}.csv",
    )

    eval_stride = int(config.get("eval_stride", 1))

    forecast_rows = []
    score_rows = []

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for sample_idx in range(len(dataset)):
            context_patches, target_patch = dataset[sample_idx]

            context_input = context_patches.unsqueeze(0).to(device)

            pred_patch = model(context_input)

            target_patch = select_target_feature_tensor(
                target_patch,
                patch_size=model.output_size,
                feature_dim=model.input_size,
                target_feature_index=0,
            )

            pred_np = pred_patch.squeeze(0).detach().cpu().numpy()
            true_np = target_patch.detach().cpu().numpy()

            horizon = len(true_np)
            pred_np = pred_np[:horizon]

            # lock forecast
            for horizon_step, pred_value in enumerate(pred_np):
                target_index = _maybe_get_dataset_index(
                    dataset=dataset,
                    sample_idx=sample_idx,
                    eval_stride=eval_stride,
                    horizon_step=horizon_step,
                )
                target_date = _maybe_get_dataset_date(
                    dataset, sample_idx, horizon_step
                )

                forecast_rows.append({
                    "forecast_target": config.get("forecast_target", "value"),
                    "rolling_step": sample_idx,
                    "horizon_step": horizon_step + 1,
                    "target_index": target_index,
                    "target_date": target_date,
                    "predicted_value": float(pred_value),
                    "locked": True,
                })

            # score after observation
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
                target_date = _maybe_get_dataset_date(
                    dataset, sample_idx, horizon_step
                )

                score_rows.append({
                    "forecast_target": config.get("forecast_target", "value"),
                    "rolling_step": sample_idx,
                    "horizon_step": horizon_step + 1,
                    "target_index": target_index,
                    "target_date": target_date,
                    "predicted_value": float(pred_value),
                    "true_value": float(true_value),
                    "error": error,
                    "absolute_error": abs(error),
                    "squared_error": error ** 2,
                })

            all_preds.append(pred_np[None, :])
            all_targets.append(true_np[None, :])

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    import csv

    with open(forecast_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "forecast_target",
                "rolling_step",
                "horizon_step",
                "target_index",
                "target_date",
                "predicted_value",
                "locked",
            ],
        )
        writer.writeheader()
        writer.writerows(forecast_rows)

    with open(score_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "forecast_target",
                "rolling_step",
                "horizon_step",
                "target_index",
                "target_date",
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

    print(f"GRU locked forecasts saved to: {forecast_csv_path}")
    print(f"GRU scores saved to: {score_csv_path}")

    return test_mse, test_mae, all_preds, all_targets, forecast_csv_path, score_csv_path

def visualize_all_rolling_predictions_as_series(
    all_preds,
    all_targets,
    config,
    save_dir="./results",
    gru_preds=None,
):
    os.makedirs(save_dir, exist_ok=True)

    tsjepa_series = all_preds.reshape(-1)
    target_series = all_targets.reshape(-1)

    plt.figure(figsize=(14, 5))

    plt.plot(
        target_series,
        label="Ground Truth",
        linewidth=2,
    )

    plt.plot(
        tsjepa_series,
        label="TS-JEPA Prediction",
        linewidth=2,
    )

    if gru_preds is not None:
        gru_series = gru_preds.reshape(-1)

        plt.plot(
            gru_series,
            label="GRU Prediction",
            linewidth=2,
            linestyle="--",
        )

    plt.legend()
    plt.xlabel("Time index in test rolling prediction")
    plt.ylabel(forecast_axis_label(config))

    if gru_preds is not None:
        plt.title(
            f"{data_title(config)} - All Rolling Forecasts: "
            "TS-JEPA vs GRU vs Ground Truth"
        )
    else:
        plt.title(
            f"{data_title(config)} - All Rolling Forecasts: "
            "Prediction vs Ground Truth"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if gru_preds is not None:
        file_name = (
            config["eval_type"]
            + f"_all_rolling_predictions_tsjepa_vs_gru_{timestamp}.png"
        )
    else:
        file_name = (
            config["eval_type"]
            + f"_all_rolling_predictions_{timestamp}.png"
        )

    save_path = os.path.join(save_dir, file_name)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"All rolling prediction figure saved to {save_path}")

def extract_context_target_series(context_patches, config):
    """Extract the downstream target feature from flattened context patches."""
    feature_dim = int(
        config.get("feature_dim", len(config.get("feature_cols", ["Close"])))
    )
    patch_size = int(config.get("patch_size", 5))
    target_feature_index = int(config.get("target_feature_index", 0))
    context_np = context_patches.detach().cpu().numpy()

    if feature_dim <= 1:
        return context_np.reshape(-1).astype(np.float64)
    if context_np.ndim != 2:
        raise ValueError(
            "Expected unbatched context patches with shape "
            f"[context_size, patch_size * feature_dim], got {context_np.shape}"
        )

    return context_np.reshape(
        context_np.shape[0], patch_size, feature_dim
    )[:, :, target_feature_index].reshape(-1).astype(np.float64)


def context_target_levels(context_patches, config):
    """Recover target-feature levels up to a multiplicative constant.

    Relative-return baselines only need ratios. For window-return inputs,
    adding one recovers price/base; for train z-score inputs, checkpoint
    statistics recover the original level.
    """
    context = extract_context_target_series(context_patches, config)
    normalization = config.get("normalization", "window_return")

    if normalization == "window_return":
        return context + 1.0
    if normalization in ("train_zscore", "train_robust_zscore"):
        stats = config.get("normalization_stats")
        if not stats:
            raise ValueError(
                "return baselines with fitted normalization require "
                "normalization_stats"
            )
        target_idx = int(config.get("target_feature_index", 0))
        if normalization == "train_zscore":
            return context * float(stats["std"][target_idx]) + float(
                stats["mean"][target_idx]
            )
        return context * float(stats["scale"][target_idx]) + float(
            stats["median"][target_idx]
        )
    return context


def context_feature_values(context_patches, config, feature_name):
    """Extract and invert one named feature from a flattened context tensor."""
    feature_cols = list(config.get("feature_cols", []))
    if feature_name not in feature_cols:
        raise ValueError(
            f"Feature {feature_name!r} is required by the baseline; "
            f"available={feature_cols}"
        )
    feature_config = dict(config)
    feature_config["target_feature_index"] = feature_cols.index(feature_name)
    values = extract_context_target_series(context_patches, feature_config)
    normalization = config.get("normalization", "none")
    index = feature_cols.index(feature_name)
    stats = config.get("normalization_stats")
    if normalization == "train_zscore":
        return values * float(stats["std"][index]) + float(stats["mean"][index])
    if normalization == "train_robust_zscore":
        return values * float(stats["scale"][index]) + float(stats["median"][index])
    if normalization == "window_return":
        raise ValueError(
            "window_return cannot be inverted for already-return-based features"
        )
    return values


def context_one_day_log_returns(context_patches, config):
    """Recover historical one-day stock or excess log returns for baselines."""
    if config.get("feature_transform", "raw") == "return":
        stock_returns = context_feature_values(
            context_patches, config, "log_return_1"
        )
        if config.get("forecast_target") == "excess_log_return":
            market_returns = context_feature_values(
                context_patches, config, "market_log_return_1"
            )
            return stock_returns - market_returns
        return stock_returns

    levels = context_target_levels(context_patches, config)
    if len(levels) < 2 or np.any(levels <= 0):
        return np.empty(0, dtype=np.float64)
    return np.diff(np.log(levels))


def historical_cumulative_log_return_path(context_patches, config):
    """Historical log-return path anchored at zero at the forecast origin."""
    returns = context_one_day_log_returns(context_patches, config)
    if len(returns) == 0:
        return np.zeros(1, dtype=np.float64)
    log_level = np.concatenate([[0.0], np.cumsum(returns)])
    return log_level - log_level[-1]


def _safe_numpy_base(value, eps=1e-8):
    if abs(float(value)) >= eps:
        return float(value)
    return eps if float(value) >= 0 else -eps


def make_baseline_prediction(
    context_patches,
    horizon,
    baseline_name,
    config=None,
):
    """
    Generate baseline forecast from historical context.

    Parameters
    ----------
    context_patches : torch.Tensor
        Shape: [context_size, patch_size * feature_dim]
        Historical context window before the forecast cutoff.

    horizon : int
        Number of future time steps to forecast.

    baseline_name : str
        Baseline model name:
            - "naive_last"
            - "previous_patch"
            - "mean_context"
            - "drift"

    Returns
    -------
    np.ndarray
        Shape: [horizon]
        Baseline prediction.
    """

    if config is None:
        config = globals().get("config", {})
    forecast_target = config.get("forecast_target", "value")

    if forecast_target in ("cumulative_log_return", "excess_log_return"):
        # h=1..H definitions:
        # naive_last:       0 for every h (no-change return forecast)
        # previous_patch:   cumulative sum of replayed recent 1-day returns
        # mean_context:     h times the mean historical 1-day return
        # drift:            endpoint log drift accumulated for h days
        one_day_returns = context_one_day_log_returns(context_patches, config)
        if baseline_name == "naive_last" or len(one_day_returns) == 0:
            pred = np.zeros(horizon, dtype=np.float64)
        elif baseline_name == "previous_patch":
            replay = np.resize(one_day_returns[-horizon:], horizon)
            pred = np.cumsum(replay)
        elif baseline_name == "mean_context":
            pred = np.arange(1, horizon + 1) * one_day_returns.mean()
        elif baseline_name == "drift":
            drift = one_day_returns.sum() / len(one_day_returns)
            pred = np.arange(1, horizon + 1) * drift
        else:
            raise ValueError(f"Unknown baseline_name: {baseline_name}")
        return pred.astype(np.float32)

    if forecast_target == "relative_return":
        levels = context_target_levels(context_patches, config)
        cutoff = _safe_numpy_base(levels[-1])

        if baseline_name == "naive_last":
            pred = np.zeros(horizon, dtype=np.float64)
        elif baseline_name in ("previous_patch", "mean_context"):
            previous = np.where(
                np.abs(levels[:-1]) < 1e-8,
                np.where(levels[:-1] < 0, -1e-8, 1e-8),
                levels[:-1],
            )
            one_step_returns = levels[1:] / previous - 1.0
            if len(one_step_returns) == 0:
                return np.zeros(horizon, dtype=np.float32)
            if baseline_name == "previous_patch":
                replay_returns = np.resize(one_step_returns[-horizon:], horizon)
            else:
                replay_returns = np.repeat(one_step_returns.mean(), horizon)
            pred = np.cumprod(1.0 + replay_returns) - 1.0
        elif baseline_name == "drift":
            if len(levels) <= 1:
                pred = np.zeros(horizon, dtype=np.float64)
            else:
                slope = (levels[-1] - levels[0]) / (len(levels) - 1)
                future_levels = levels[-1] + slope * np.arange(1, horizon + 1)
                pred = future_levels / cutoff - 1.0
        else:
            raise ValueError(f"Unknown baseline_name: {baseline_name}")

        return pred.astype(np.float32)

    context_flat = extract_context_target_series(context_patches, config)

    if baseline_name == "naive_last":
        # Repeat the last observed value for all future steps
        pred = np.repeat(context_flat[-1], horizon)

    elif baseline_name == "previous_patch":
        # Use the last context patch as the forecast
        last_patch = context_flat[-horizon:]

        if len(last_patch) >= horizon:
            pred = last_patch[:horizon]
        else:
            pred = np.resize(last_patch, horizon)

    elif baseline_name == "mean_context":
        # Repeat the mean value of the whole context window
        pred = np.repeat(context_flat.mean(), horizon)

    elif baseline_name == "drift":
        # Linear extrapolation based on the first and last context values
        if len(context_flat) <= 1:
            pred = np.repeat(context_flat[-1], horizon)
        else:
            slope = (context_flat[-1] - context_flat[0]) / (len(context_flat) - 1)
            steps = np.arange(1, horizon + 1)
            pred = context_flat[-1] + slope * steps

    else:
        raise ValueError(f"Unknown baseline_name: {baseline_name}")

    return pred.astype(np.float32)


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
    sample_indices=None,
    baseline_names=None,
):
    """
    Visualize rolling forecast windows with TS-JEPA prediction and baseline predictions.

    This function supports two modes:

    1. Visualize selected rolling windows:
        sample_indices=[60]

    2. Visualize the first N rolling windows:
        max_samples=100

    If sample_indices is given, max_samples is ignored.

    Each figure contains:
        - Historical context
        - Ground truth future
        - TS-JEPA prediction
        - Baseline predictions

    Parameters
    ----------
    encoder : torch.nn.Module
        Pretrained and frozen TS-JEPA encoder.

    decoder : torch.nn.Module
        Trained downstream forecasting decoder.

    dataset : torch.utils.data.Dataset
        Rolling evaluation dataset.
        Each item should return:
            context_patches, target_patch

    device : torch.device
        Device used for model inference.

    eval_type : str
        Context embedding strategy:
            - "last"
            - "mean"

    config : dict
        Experiment config.

    save_dir : str
        Directory to save generated figures.

    max_samples : int or None
        Number of rolling windows to visualize if sample_indices is None.

    make_gif : bool
        Whether to generate a GIF from saved figures.

    gif_name : str
        GIF filename.

    gif_duration : float
        Duration per frame in the GIF.

    sample_indices : list[int] or None
        Specific rolling window indices to visualize.
        Example:
            sample_indices=[60]
            sample_indices=[0, 20, 40, 60]

    baseline_names : list[str] or None
        Baselines to plot.
        Default:
            [
                "naive_last",
                "previous_patch",
                "mean_context",
                "drift",
            ]
    """

    os.makedirs(save_dir, exist_ok=True)

    encoder.eval()
    decoder.eval()

    if baseline_names is None:
        baseline_names = [
            "naive_last",
            "previous_patch",
            "mean_context",
            "drift",
        ]

    # -------------------------------------------------
    # Decide which rolling window indices to visualize
    # -------------------------------------------------
    if sample_indices is not None:
        indices_to_plot = []

        for idx in sample_indices:
            if 0 <= idx < len(dataset):
                indices_to_plot.append(idx)
            else:
                print(
                    f"Warning: sample_idx={idx} is out of range. "
                    f"Valid range is [0, {len(dataset) - 1}]. "
                    f"This index will be skipped."
                )

    else:
        if max_samples is None:
            num_samples = len(dataset)
        else:
            num_samples = min(len(dataset), max_samples)

        indices_to_plot = list(range(num_samples))

    print(f"Visualizing {len(indices_to_plot)} rolling windows...")

    saved_paths = []

    # -------------------------------------------------
    # Plot each selected rolling window
    # -------------------------------------------------
    for sample_idx in indices_to_plot:
        context_patches, target_patch = dataset[sample_idx]

        context_patches_input = context_patches.unsqueeze(0).to(device)

        # ---------------------------------------------
        # TS-JEPA forecast
        # ---------------------------------------------
        with torch.no_grad():
            encoded_patches = encoder(context_patches_input)

            context_embedding = get_context_embedding(
                encoded_patches,
                eval_type,
            )

            pred_patch = decoder(context_embedding)

        # ---------------------------------------------
        # Convert tensors to numpy
        # ---------------------------------------------
        if config.get("forecast_target") == "relative_return":
            context_levels = context_target_levels(context_patches, config)
            context_np = (
                context_levels / _safe_numpy_base(context_levels[-1]) - 1.0
            )
        elif config.get("forecast_target") in (
            "cumulative_log_return",
            "excess_log_return",
        ):
            context_np = historical_cumulative_log_return_path(
                context_patches, config
            )
        else:
            context_np = extract_context_target_series(context_patches, config)
        target_np = target_patch.detach().cpu().numpy().reshape(-1)
        pred_np = pred_patch.flatten().detach().cpu().numpy()

        horizon = len(target_np)

        # In case decoder output is longer or shorter than target
        pred_np = pred_np[:horizon]

        # ---------------------------------------------
        # X-axis
        # ---------------------------------------------
        context_x = np.arange(len(context_np))

        target_x = np.arange(
            len(context_np),
            len(context_np) + horizon,
        )

        # ---------------------------------------------
        # Plot
        # ---------------------------------------------
        plt.figure(figsize=(14, 6))

        # Historical context
        plt.plot(
            context_x,
            context_np,
            label="Context",
            linewidth=2,
        )

        # Ground truth future
        plt.plot(
            target_x,
            target_np,
            marker="o",
            linewidth=2,
            label="Ground Truth Future",
        )

        # TS-JEPA prediction
        plt.plot(
            target_x,
            pred_np,
            marker="o",
            linewidth=2,
            label="TS-JEPA Prediction",
        )

        # Baseline predictions
        for baseline_name in baseline_names:
            baseline_pred = make_baseline_prediction(
                context_patches=context_patches,
                horizon=horizon,
                baseline_name=baseline_name,
                config=config,
            )

            plt.plot(
                target_x,
                baseline_pred,
                marker="x",
                linestyle="--",
                linewidth=1.5,
                label=f"Baseline: {baseline_name}",
            )

        # Forecast origin
        plt.axvline(
            x=len(context_np) - 1,
            linestyle="--",
            linewidth=1.5,
            label="Forecast Origin",
        )

        plt.legend()
        plt.xlabel("Time index inside rolling window")
        plt.ylabel(forecast_axis_label(config))
        plt.title(
            f"{data_title(config)} - Rolling Forecast Window #{sample_idx} "
            f"with Baselines"
        )

        save_path = os.path.join(
            save_dir,
            f"{config['eval_type']}_rolling_window_{sample_idx:04d}_with_baselines.png",
        )

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        saved_paths.append(save_path)

    print(f"Saved {len(saved_paths)} rolling window figures to: {save_dir}")

    # -------------------------------------------------
    # Generate GIF
    # -------------------------------------------------
    if make_gif and len(saved_paths) > 0:
        gif_path = os.path.join(save_dir, gif_name)

        frames = [
            imageio.imread(path)
            for path in saved_paths
        ]

        imageio.mimsave(
            gif_path,
            frames,
            duration=gif_duration,
        )

        print(f"GIF saved to: {gif_path}")

def visualize_one_rolling_window(
    encoder,
    decoder,
    dataset,
    sample_idx,
    device,
    eval_type,
    config,
    save_dir="./results",
):
    os.makedirs(save_dir, exist_ok=True)

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

    if config.get("forecast_target") == "relative_return":
        context_levels = context_target_levels(context_patches, config)
        context_np = context_levels / _safe_numpy_base(context_levels[-1]) - 1.0
    elif config.get("forecast_target") in (
        "cumulative_log_return",
        "excess_log_return",
    ):
        context_np = historical_cumulative_log_return_path(context_patches, config)
    else:
        context_np = extract_context_target_series(context_patches, config)
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
    plt.ylabel(forecast_axis_label(config))
    plt.title(f"{data_title(config)} - Rolling Forecast Window #{sample_idx}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        save_dir,
        config["eval_type"]
        + f"_rolling_window_{sample_idx}_{timestamp}.png",
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Rolling window figure saved to {save_path}")

def get_context_embedding(encoded_patches, eval_type):
    if eval_type == "last":
        return encoded_patches[:, -1, :]
    elif eval_type == "mean":
        return encoded_patches.mean(dim=1)
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")


def load_pretrained_encoder_state(encoder, state_dict):
    """Load learned weights while allowing a different context token count."""
    encoder_state = dict(state_dict)
    checkpoint_pos = encoder_state.get("pos_embed")
    if checkpoint_pos is not None and checkpoint_pos.shape != encoder.pos_embed.shape:
        print(
            "Regenerating sinusoidal encoder positions for downstream context: "
            f"checkpoint={tuple(checkpoint_pos.shape)}, "
            f"downstream={tuple(encoder.pos_embed.shape)}"
        )
        encoder_state.pop("pos_embed")
        incompatible = encoder.load_state_dict(encoder_state, strict=False)
        if incompatible.missing_keys != ["pos_embed"] or incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected encoder checkpoint mismatch: {incompatible}")
        return

    encoder.load_state_dict(encoder_state)


def load_pretraining_checkpoint(checkpoint_path, map_location):
    """Load a trusted, locally produced full-state pretraining checkpoint.

    Unified checkpoints intentionally contain optimizer and RNG state, including
    NumPy objects that PyTorch 2.6's default weights-only loader rejects.
    Checkpoints from untrusted sources must not be loaded through this helper.
    """
    return torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )


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

    if hasattr(dataset, "sample_starts"):
        try:
            context_length = int(dataset.context_size) * int(dataset.patch_size)
            return (
                int(dataset.sample_starts[sample_idx])
                + context_length
                + int(horizon_step)
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            pass

    return int(sample_idx * eval_stride + horizon_step)


def _maybe_get_dataset_date(dataset, sample_idx, horizon_step):
    """Return the actual target date when the evaluation dataset exposes it."""
    if not hasattr(dataset, "dates") or not hasattr(dataset, "sample_starts"):
        return ""
    try:
        context_length = int(dataset.context_size) * int(dataset.patch_size)
        target_position = (
            int(dataset.sample_starts[sample_idx])
            + context_length
            + int(horizon_step)
        )
        value = dataset.dates[target_position]
        return value.isoformat() if hasattr(value, "isoformat") else str(value)
    except (AttributeError, IndexError, TypeError, ValueError):
        return ""



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

            target_patch = select_target_feature_tensor(
                target_patch,
                patch_size=int(config.get("patch_size", 5)),
                feature_dim=int(config.get("feature_dim", 1)),
                target_feature_index=int(config.get("target_feature_index", 0)),
            )
            true_np = target_patch.detach().cpu().numpy().reshape(-1)
            horizon = len(true_np)

            # Forecast using only historical context.
            pred_np = make_baseline_prediction(
                context_patches=context_patches,
                horizon=horizon,
                baseline_name=baseline_name,
                config=config,
            ).reshape(-1)

            # Lock forecast before scoring.
            for horizon_step, pred_value in enumerate(pred_np):
                target_index = _maybe_get_dataset_index(
                    dataset=dataset,
                    sample_idx=sample_idx,
                    eval_stride=eval_stride,
                    horizon_step=horizon_step,
                )
                target_date = _maybe_get_dataset_date(
                    dataset, sample_idx, horizon_step
                )

                all_forecast_rows.append({
                    "model": baseline_name,
                    "forecast_target": config.get("forecast_target", "value"),
                    "rolling_step": sample_idx,
                    "horizon_step": horizon_step + 1,
                    "target_index": target_index,
                    "target_date": target_date,
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
                target_date = _maybe_get_dataset_date(
                    dataset, sample_idx, horizon_step
                )

                all_score_rows.append({
                    "model": baseline_name,
                    "forecast_target": config.get("forecast_target", "value"),
                    "rolling_step": sample_idx,
                    "horizon_step": horizon_step + 1,
                    "target_index": target_index,
                    "target_date": target_date,
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
            "forecast_target": config.get("forecast_target", "value"),
            "mse": float(np.mean((baseline_preds - baseline_targets) ** 2)),
            "mae": float(np.mean(np.abs(baseline_preds - baseline_targets))),
            "trend_accuracy": compute_trend_accuracy(
                all_preds=baseline_preds,
                all_targets=baseline_targets,
                include_origin=config.get("forecast_target") == "relative_return",
                direct_return=config.get("forecast_target")
                in ("cumulative_log_return", "excess_log_return"),
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
                "forecast_target",
                "rolling_step",
                "horizon_step",
                "target_index",
                "target_date",
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
                "forecast_target",
                "rolling_step",
                "horizon_step",
                "target_index",
                "target_date",
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
            fieldnames=[
                "model",
                "forecast_target",
                "mse",
                "mae",
                "trend_accuracy",
            ],
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
    """Save and print comparison tables for TS-JEPA and baselines."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_csv_path = os.path.join(
        save_dir,
        config["eval_type"] + f"_model_comparison_{timestamp}.csv",
    )
    comparison_txt_path = os.path.join(
        save_dir,
        config["eval_type"] + f"_model_comparison_{timestamp}.txt",
    )

    # Lower MSE is better; sort accordingly.
    model_rows = sorted(model_rows, key=lambda row: row["mse"])

    import csv
    with open(comparison_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "forecast_target",
                "mse",
                "mae",
                "trend_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(model_rows)

    lines = [
        f"Data source: {data_title(config)}",
        f"Evaluation type: {config['eval_type']}",
        f"Forecast target: {config.get('forecast_target', 'value')}",
        f"Feature transform: {config.get('feature_transform', 'raw')}",
        f"Features: {','.join(config.get('feature_cols', []))}",
        f"Normalization: {config.get('normalization', 'none')}",
        f"Market data: {config.get('market_data') or 'disabled'}",
        f"Generated at: {timestamp}",
        "",
        "Model Comparison",
        "model,forecast_target,mse,mae,trend_accuracy",
    ]
    for row in model_rows:
        lines.append(
            f"{row['model']},"
            f"{row.get('forecast_target', config.get('forecast_target', 'value'))},"
            f"{row['mse']:.6f},"
            f"{row['mae']:.6f},"
            f"{row['trend_accuracy']:.4f}"
        )

    with open(comparison_txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"========== Model Comparison ({data_title(config)}) ==========")
    for row in model_rows:
        print(
            f"{row['model']}: "
            f"MSE={row['mse']:.6f}, "
            f"MAE={row['mae']:.6f}, "
            f"TrendAcc={row['trend_accuracy']:.4f}"
        )
    print(f"Model comparison saved to: {comparison_csv_path}")
    print(f"Text comparison saved to: {comparison_txt_path}")

    return comparison_csv_path, comparison_txt_path


def visualize_model_comparison(model_rows, config, save_dir="./results"):
    """Bar plots for MSE, MAE, and trend accuracy of TS-JEPA vs baselines."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_rows = sorted(model_rows, key=lambda row: row["mse"])
    names = [row["model"] for row in model_rows]
    mses = [row["mse"] for row in model_rows]
    maes = [row["mae"] for row in model_rows]
    trend_rows = sorted(
        model_rows,
        key=lambda row: row["trend_accuracy"],
        reverse=True,
    )
    trend_names = [row["model"] for row in trend_rows]
    trend_accs = [row["trend_accuracy"] for row in trend_rows]

    plt.figure(figsize=(10, 5))
    plt.bar(names, mses)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("MSE")
    plt.title(f"{data_title(config)} - Prequential Rolling Evaluation: MSE Comparison")
    plt.tight_layout()
    mse_png_path = os.path.join(
        save_dir,
        config["eval_type"] + f"_model_comparison_mse_{timestamp}.png",
    )
    plt.savefig(mse_png_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(names, maes)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("MAE")
    plt.title(f"{data_title(config)} - Prequential Rolling Evaluation: MAE Comparison")
    plt.tight_layout()
    mae_png_path = os.path.join(
        save_dir,
        config["eval_type"] + f"_model_comparison_mae_{timestamp}.png",
    )
    plt.savefig(mae_png_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(trend_names, trend_accs)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Trend Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title(
        f"{data_title(config)} - Prequential Rolling Evaluation: "
        "Trend Accuracy Comparison"
    )
    plt.tight_layout()
    trend_png_path = os.path.join(
        save_dir,
        config["eval_type"] + f"_model_comparison_trend_accuracy_{timestamp}.png",
    )
    plt.savefig(trend_png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"MSE comparison figure saved to: {mse_png_path}")
    print(f"MAE comparison figure saved to: {mae_png_path}")
    print(f"Trend accuracy comparison figure saved to: {trend_png_path}")

    return mse_png_path, mae_png_path, trend_png_path

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
                target_date = _maybe_get_dataset_date(
                    dataset, sample_idx, horizon_step
                )

                forecast_rows.append({
                    "forecast_target": config.get("forecast_target", "value"),
                    "rolling_step": sample_idx,
                    "horizon_step": horizon_step + 1,
                    "target_index": target_index,
                    "target_date": target_date,
                    "predicted_value": float(pred_value),
                    "locked": True,
                })

            # =========================
            # 4. Observation arrives
            # =========================
            target_patch = select_target_feature_tensor(
                target_patch,
                patch_size=predicted_next_patch.shape[-1],
                feature_dim=int(config.get("feature_dim", 1)),
                target_feature_index=int(config.get("target_feature_index", 0)),
            )
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
                target_date = _maybe_get_dataset_date(
                    dataset, sample_idx, horizon_step
                )

                score_rows.append({
                    "forecast_target": config.get("forecast_target", "value"),
                    "rolling_step": sample_idx,
                    "horizon_step": horizon_step + 1,
                    "target_index": target_index,
                    "target_date": target_date,
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
                "forecast_target",
                "rolling_step",
                "horizon_step",
                "target_index",
                "target_date",
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
                "forecast_target",
                "rolling_step",
                "horizon_step",
                "target_index",
                "target_date",
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

            target_patch = select_target_feature_tensor(
                target_patch,
                patch_size=predicted_next_patch.shape[-1],
                feature_dim=int(config.get("feature_dim", 1)) if "config" in globals() else 1,
                target_feature_index=int(config.get("target_feature_index", 0)) if "config" in globals() else 0,
            )

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


def compute_trend_accuracy(
    all_preds,
    all_targets,
    include_origin=False,
    direct_return=False,
):
    """
    Trend accuracy based on within-patch direction.

    For each predicted target patch:

        pred_diff = pred[:, 1:] - pred[:, :-1]
        true_diff = true[:, 1:] - true[:, :-1]

    Accuracy checks whether predicted direction matches true direction. For a
    cutoff-relative return path, include_origin=True also scores the first move
    from the known zero-return origin.
    """
    if direct_return:
        return float(((all_preds > 0) == (all_targets > 0)).mean())

    if include_origin:
        # A relative-return path is anchored at zero at the forecast cutoff.
        # Include cutoff -> horizon 1 so the first predicted move is scored.
        origin = np.zeros((all_preds.shape[0], 1), dtype=all_preds.dtype)
        all_preds = np.concatenate([origin, all_preds], axis=1)
        all_targets = np.concatenate([origin, all_targets], axis=1)

    pred_diff = all_preds[:, 1:] - all_preds[:, :-1]
    true_diff = all_targets[:, 1:] - all_targets[:, :-1]

    trend_pred = np.sign(pred_diff)
    trend_true = np.sign(true_diff)

    trend_accuracy = (trend_pred == trend_true).mean()

    return float(trend_accuracy)


def directional_auxiliary_loss(
    predicted_patch,
    target_patch,
    temperature=0.01,
    threshold=0.0,
    include_origin=False,
    direct_return=False,
):
    """
    Differentiable direction loss for within-patch trend.

    It encourages predicted consecutive differences to have the same sign as
    the true consecutive differences. Near-zero true moves can be ignored with
    threshold to avoid training hard on noise.
    """
    if direct_return:
        valid_mask = target_patch.abs() > threshold
        if not valid_mask.any():
            return predicted_patch.new_tensor(0.0)
        true_sign = torch.sign(target_patch[valid_mask])
        pred_scaled = predicted_patch[valid_mask] / max(float(temperature), 1e-8)
        return torch.nn.functional.softplus(-true_sign * pred_scaled).mean()

    if include_origin:
        origin = torch.zeros_like(predicted_patch[:, :1])
        predicted_patch = torch.cat([origin, predicted_patch], dim=1)
        target_patch = torch.cat([origin, target_patch], dim=1)

    if predicted_patch.shape[1] < 2:
        return predicted_patch.new_tensor(0.0)

    pred_diff = predicted_patch[:, 1:] - predicted_patch[:, :-1]
    true_diff = target_patch[:, 1:] - target_patch[:, :-1]

    valid_mask = true_diff.abs() > threshold
    if not valid_mask.any():
        return predicted_patch.new_tensor(0.0)

    true_sign = torch.sign(true_diff[valid_mask])
    pred_scaled = pred_diff[valid_mask] / max(float(temperature), 1e-8)

    return torch.nn.functional.softplus(-true_sign * pred_scaled).mean()


if __name__ == "__main__":
    # =========================
    # Config
    # =========================

    config = prepare_args(config)
    results_dir = config.get("results_dir", "./results")
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Results directory:", results_dir)

    config["path_data"] = (
        "./data/" + config["data"] + "/" + config["data"] + ".csv"
    )

    print("Load data from:", config["path_data"])

    # =========================
    # Important parameters
    # =========================
    # These should match your pretraining patch setting.
    patch_size = int(config["patch_size"])
    feature_cols = list(config["feature_cols"])
    use_sentiment = bool(config["use_sentiment"])
    timestamp_col = config["timestamp_col"]
    sentiment_path = config.get("sentiment_path", None)
    train_end_date = config.get("train_end_date", None)
    test_start_date = config.get("test_start_date", None)
    data_end_date = config.get("data_end_date", None)
    validation_fraction = config.get("validation_fraction", 0.05)
    normalization = config.get("normalization", "window_return")
    forecast_target = config.get("forecast_target", "value")
    sampling_mode = config.get("sampling_mode", "sliding_window")
    normalization_stats = config.get("normalization_stats", None)
    feature_transform = config.get("feature_transform", "raw")
    market_data = config.get("market_data", None)
    robust_zscore_clip = config.get("robust_zscore_clip", None)
    target_feature_index = int(config.get("target_feature_index", 0))
    target_col = config["target_col"]
    if forecast_target == "value" and target_col not in feature_cols:
        if not 0 <= target_feature_index < len(feature_cols):
            raise ValueError(
                f"target_feature_index={target_feature_index} is outside "
                f"feature_cols={feature_cols}"
            )
        target_col = feature_cols[target_feature_index]

    config["patch_size"] = patch_size
    config["feature_cols"] = feature_cols
    config["target_feature_index"] = target_feature_index
    config["forecast_target"] = forecast_target

    print("feature_cols =", feature_cols)
    print("use_sentiment =", use_sentiment)
    print("timestamp_col =", timestamp_col)
    print("sentiment_path =", sentiment_path)
    print("train_end_date =", train_end_date)
    print("test_start_date =", test_start_date)
    print("data_end_date =", data_end_date)
    print("validation_fraction =", validation_fraction)
    print("target_feature_index =", target_feature_index)
    print("target_col =", target_col)
    print("forecast_target =", forecast_target)
    print("feature_transform =", feature_transform)
    print("normalization =", normalization)
    print("robust_zscore_clip =", robust_zscore_clip)
    print("market_data =", market_data)
    print("sampling_mode =", sampling_mode)
    print("trend_weight =", config.get("trend_weight", 0.0))
    print("trend_loss_temperature =", config.get("trend_loss_temperature", 0.01))
    print("trend_loss_threshold =", config.get("trend_loss_threshold", 0.0))
    print("trend_selection_weight =", config.get("trend_selection_weight", 0.0))
    print("fine_tune_encoder =", config.get("fine_tune_encoder", False))
    print("encoder_finetune_lr =", config.get("encoder_finetune_lr", 1e-5))

    # Number of context patches for downstream forecasting.
    # For example:
    # context_size=10 means 10 patches * 5 days = 50 days context.
    context_size = config["context_size"]

    # For rolling evaluation, stride=1 gives day-by-day rolling.
    # If you want patch-by-patch rolling, use stride=patch_size.
    eval_stride = config["eval_stride"]

    # =========================
    # Data loaders
    # =========================

    train_loader = get_evaluation_loaders(
        config["path_data"],
        config["batch_size"],
        split="train",
        patch_size=patch_size,
        context_size=context_size,
        stride=eval_stride,
        sampling_mode=sampling_mode,
        normalization=normalization,
        normalization_stats=normalization_stats,
        feature_transform=feature_transform,
        market_data=market_data,
        robust_zscore_clip=robust_zscore_clip,
        feature_cols=feature_cols,
        target_col=target_col,
        forecast_target=forecast_target,
        timestamp_col=timestamp_col,
        sentiment_path=sentiment_path,
        validation_fraction=validation_fraction,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        data_end_date=data_end_date,
    )

    # Resolve the deterministic transformed order once and reuse the training
    # normalizer state for validation/test without refitting.
    feature_cols = list(train_loader.dataset.feature_cols)
    feature_dim = len(feature_cols)
    normalization_stats = train_loader.dataset.normalization_stats
    config["feature_cols"] = feature_cols
    config["feature_names"] = list(feature_cols)
    config["feature_dim"] = feature_dim
    config["normalization_stats"] = normalization_stats
    config["warmup_report"] = train_loader.dataset.warmup_report
    config["market_alignment_report"] = train_loader.dataset.market_alignment_report
    if not 0 <= target_feature_index < feature_dim:
        raise ValueError(
            f"target_feature_index={target_feature_index} is outside resolved "
            f"feature_cols={feature_cols}"
        )

    print("\n=== Financial preprocessing summary ===")
    print("Feature transform:", feature_transform)
    print("Features:")
    for feature_name in feature_cols:
        print(" ", feature_name)
    print("Normalization:", normalization)
    print("Normalization fit split: train only")
    print("Forecast target:", forecast_target)
    print(f"Forecast horizons: 1..{patch_size}")
    print(f"Window length: {context_size * patch_size}")
    print("Patch size:", patch_size)
    print("Market data:", market_data or "disabled")
    print("Warm-up rows:", config["warmup_report"])
    preprocessing_metadata = {
        "feature_transform": feature_transform,
        "use_sentiment": use_sentiment,
        "market_features": list(config["market_features"]),
        "sentiment_features": list(config["sentiment_features"]),
        "feature_names": feature_cols,
        "normalization": normalization,
        "normalization_fit_split": "train",
        "normalization_stats": normalization_stats,
        "robust_zscore_clip": robust_zscore_clip,
        "forecast_target": forecast_target,
        "target_definition": {
            "value": "future target values in the configured normalized feature space",
            "relative_return": "Close[t+h] / Close[t] - 1",
            "cumulative_log_return": "log(Close[t+h] / Close[t])",
            "excess_log_return": (
                "log(Close[t+h] / Close[t]) - "
                "log(Market[t+h] / Market[t])"
            ),
        }[forecast_target],
        "metric_definition": (
            "MSE and MAE over every saved rolling-step/horizon target value"
        ),
        "direction_accuracy_definition": (
            "sign of consecutive forecast-horizon differences equals sign of "
            "consecutive target differences; relative-return paths include the "
            "known zero origin; cumulative/excess log-return targets compare "
            "the binary indicators (forecast > 0) and (target > 0) at each horizon"
        ),
        "forecast_horizons": list(range(1, patch_size + 1)),
        "forecast_horizon": patch_size,
        "market_data": market_data,
        "warmup_report": config["warmup_report"],
        "market_alignment_report": config["market_alignment_report"],
        "window_length": context_size * patch_size,
        "context_size": context_size,
        "patch_size": patch_size,
        "eval_stride": eval_stride,
        "sampling_mode": sampling_mode,
        "train_end": train_end_date,
        "test_start": test_start_date,
        "test_end": data_end_date,
        "validation_fraction": validation_fraction,
        "batch_size": config["batch_size"],
        "downstream_epochs": config["num_epochs"],
        "forecast_head_lr": config["lr"],
        "checkpoint_path": config.get("pretrain_checkpoint_path"),
        "checkpoint_selection": config.get("checkpoint_selection"),
        "checkpoint_epoch": config.get("checkpoint_to_use"),
        "encoder_weight_source": config.get("pretrain_encoder_weights"),
        "fine_tune_encoder": bool(config.get("fine_tune_encoder", False)),
        "encoder_finetune_lr": config.get("encoder_finetune_lr"),
        "trend_weight": config.get("trend_weight"),
        "trend_loss_temperature": config.get("trend_loss_temperature"),
        "trend_loss_threshold": config.get("trend_loss_threshold"),
        "trend_selection_weight": config.get("trend_selection_weight"),
        **runtime_provenance(),
    }
    metadata_path = os.path.join(results_dir, "preprocessing_config.json")
    with open(metadata_path, "w") as metadata_file:
        json.dump(preprocessing_metadata, metadata_file, indent=2, sort_keys=True)
    print("Preprocessing metadata saved to:", metadata_path)

    val_loader = get_evaluation_loaders(
        config["path_data"],
        config["batch_size"],
        split="val",
        patch_size=patch_size,
        context_size=context_size,
        stride=eval_stride,
        sampling_mode=sampling_mode,
        normalization=normalization,
        normalization_stats=normalization_stats,
        feature_transform=feature_transform,
        market_data=market_data,
        robust_zscore_clip=robust_zscore_clip,
        feature_cols=feature_cols,
        target_col=target_col,
        forecast_target=forecast_target,
        timestamp_col=timestamp_col,
        sentiment_path=sentiment_path,
        validation_fraction=validation_fraction,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        data_end_date=data_end_date,
    )

    test_loader = get_evaluation_loaders(
        config["path_data"],
        config["batch_size"],
        split="test",
        patch_size=patch_size,
        context_size=context_size,
        stride=eval_stride,
        sampling_mode=sampling_mode,
        normalization=normalization,
        normalization_stats=normalization_stats,
        feature_transform=feature_transform,
        market_data=market_data,
        robust_zscore_clip=robust_zscore_clip,
        feature_cols=feature_cols,
        target_col=target_col,
        forecast_target=forecast_target,
        timestamp_col=timestamp_col,
        sentiment_path=sentiment_path,
        validation_fraction=validation_fraction,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        data_end_date=data_end_date,
    )

    preprocessing_metadata.update(
        {
            "test_sample_count": len(test_loader.dataset),
            "test_target_start": _maybe_get_dataset_date(
                test_loader.dataset, 0, 0
            ),
            "test_target_end": _maybe_get_dataset_date(
                test_loader.dataset,
                len(test_loader.dataset) - 1,
                patch_size - 1,
            ),
        }
    )
    with open(metadata_path, "w") as metadata_file:
        json.dump(preprocessing_metadata, metadata_file, indent=2, sort_keys=True)

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

    decoder = build_reconstruction_decoder(
        decoder_type=config["decoder_type"],
        embedding_dim=config["pretrain_encoder_embed_dim"],
        output_dim=patch_size,
        hidden_dim=config["decoder_hidden_dim"],
        num_layers=config["decoder_num_layers"],
        dropout=config["decoder_dropout"],
    )

    print("decoder_type =", config.get("decoder_type", "linear"))
    if config.get("decoder_type", "linear") in ("mlp", "residual_mlp"):
        print("decoder_hidden_dim =", config.get("decoder_hidden_dim", 256))
        print("decoder_dropout =", config.get("decoder_dropout", 0.1))
    if config.get("decoder_type", "linear") == "mlp":
        print("decoder_num_layers =", config.get("decoder_num_layers", 2))

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

    if config.get("pretrain_checkpoint_path"):
        checkpoint_path = config["pretrain_checkpoint_path"]
    else:
        checkpoint_path = config["path_save"] + path_name + ".pt"

    print("Load checkpoint:", checkpoint_path)

    checkpoint = load_pretraining_checkpoint(
        checkpoint_path,
        device,
    )
    checkpoint_config = checkpoint.get("config", {})
    checkpoint_features = checkpoint_config.get(
        "feature_names", checkpoint_config.get("feature_cols")
    )
    if checkpoint_features is not None and list(checkpoint_features) != feature_cols:
        raise ValueError(
            "Checkpoint feature order/dimension does not match evaluation data: "
            f"checkpoint={list(checkpoint_features)}, evaluation={feature_cols}. "
            "Use the preprocessing settings stored in the checkpoint."
        )
    checkpoint_transform = checkpoint_config.get("feature_transform")
    if checkpoint_transform is not None and checkpoint_transform != feature_transform:
        raise ValueError(
            "Checkpoint feature_transform does not match evaluation: "
            f"checkpoint={checkpoint_transform!r}, evaluation={feature_transform!r}"
        )

    encoder_key = (
        "encoder_ema"
        if config.get("pretrain_encoder_weights", "ema") == "ema"
        else "encoder"
    )
    if encoder_key not in checkpoint:
        print(
            f"Warning: checkpoint has no {encoder_key!r}; falling back to online encoder"
        )
        encoder_key = "encoder"
    load_pretrained_encoder_state(encoder, checkpoint[encoder_key])

    print(f"Pretrained encoder loaded from {encoder_key}")

    fine_tune_encoder = bool(config.get("fine_tune_encoder", False))
    encoder_finetune_lr = float(config.get("encoder_finetune_lr", 1e-5))

    for p in encoder.parameters():
        p.requires_grad = fine_tune_encoder

    if fine_tune_encoder:
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder.parameters(), "lr": encoder_finetune_lr},
                {"params": decoder.parameters(), "lr": config["lr"]},
            ],
        )
        print(
            "Fine-tuning encoder during downstream training "
            f"(encoder lr={encoder_finetune_lr}, decoder lr={config['lr']})"
        )
    else:
        encoder.eval()
        optimizer = torch.optim.AdamW(
            decoder.parameters(),
            lr=config["lr"],
        )
        print("Encoder frozen during downstream training")

    # =========================
    # Train decoder
    # =========================

    print("Start downstream decoder training")

    loss_history = []
    mse_loss_history = []
    trend_loss_history = []
    val_mse_history = []
    val_mae_history = []
    val_trend_history = []

    best_val_score = float("inf")
    best_encoder_state = None
    best_decoder_state = None
    trend_weight = float(config.get("trend_weight", 0.0))
    trend_loss_temperature = float(config.get("trend_loss_temperature", 0.01))
    trend_loss_threshold = float(config.get("trend_loss_threshold", 0.0))
    trend_selection_weight = float(config.get("trend_selection_weight", 0.0))

    for epoch in range(config["num_epochs"]):
        if fine_tune_encoder:
            encoder.train()
        else:
            encoder.eval()
        decoder.train()

        total_loss = 0.0
        total_mse_loss = 0.0
        total_trend_loss = 0.0

        for context_patches, target_patch in train_loader:
            optimizer.zero_grad()

            context_patches = context_patches.to(device)
            target_patch = target_patch.to(device)
            target_patch = select_target_feature_tensor(
                target_patch,
                patch_size=patch_size,
                feature_dim=feature_dim,
                target_feature_index=target_feature_index,
            )

            if fine_tune_encoder:
                encoded_patches = encoder(context_patches)
            else:
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

            trend_loss = directional_auxiliary_loss(
                predicted_patch=predicted_next_patch,
                target_patch=target_patch,
                temperature=trend_loss_temperature,
                threshold=trend_loss_threshold,
                include_origin=forecast_target == "relative_return",
                direct_return=forecast_target
                in ("cumulative_log_return", "excess_log_return"),
            )

            loss = mse_loss + trend_weight * trend_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_trend_loss += trend_loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_mse_loss = total_mse_loss / len(train_loader)
        avg_train_trend_loss = total_trend_loss / len(train_loader)
        loss_history.append(avg_train_loss)
        mse_loss_history.append(avg_train_mse_loss)
        trend_loss_history.append(avg_train_trend_loss)

        # =========================
        # Validation
        # =========================

        val_mse, val_mae, val_preds, val_targets = evaluate_model(
            encoder=encoder,
            decoder=decoder,
            loader=val_loader,
            device=device,
            eval_type=config["eval_type"],
        )
        val_trend_acc = compute_trend_accuracy(
            all_preds=val_preds,
            all_targets=val_targets,
            include_origin=forecast_target == "relative_return",
            direct_return=forecast_target
            in ("cumulative_log_return", "excess_log_return"),
        )
        val_score = val_mse + trend_selection_weight * (1.0 - val_trend_acc)

        val_mse_history.append(val_mse)
        val_mae_history.append(val_mae)
        val_trend_history.append(val_trend_acc)

        if val_score < best_val_score:
            best_val_score = val_score
            best_encoder_state = {
                k: v.detach().cpu().clone()
                for k, v in encoder.state_dict().items()
            }
            best_decoder_state = {
                k: v.detach().cpu().clone()
                for k, v in decoder.state_dict().items()
            }

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: "
                f"train_loss={avg_train_loss:.6f}, "
                f"mse_loss={avg_train_mse_loss:.6f}, "
                f"trend_loss={avg_train_trend_loss:.6f}, "
                f"val_mse={val_mse:.6f}, "
                f"val_mae={val_mae:.6f}, "
                f"val_trend_acc={val_trend_acc:.4f}"
            )

    # Restore best decoder
    if best_encoder_state is not None:
        encoder.load_state_dict(best_encoder_state)
        encoder.to(device)
    if best_decoder_state is not None:
        decoder.load_state_dict(best_decoder_state)
        decoder.to(device)

    # =========================
    # Save training history
    # =========================

    save_path = os.path.join(results_dir, "loss.txt")

    with open(save_path, "w") as f:
        f.write("epoch,train_loss,mse_loss,trend_loss,val_mse,val_mae,val_trend_acc\n")
        for epoch in range(len(loss_history)):
            f.write(
                f"{epoch},"
                f"{loss_history[epoch]},"
                f"{mse_loss_history[epoch]},"
                f"{trend_loss_history[epoch]},"
                f"{val_mse_history[epoch]},"
                f"{val_mae_history[epoch]},"
                f"{val_trend_history[epoch]}\n"
            )

    print(f"Loss saved to {save_path}")

    print("Start GRU baseline training")

    gru_model = GRUForecastModel(
        input_size=feature_dim,
        hidden_size=config.get("gru_hidden_size", 64),
        num_layers=config.get("gru_num_layers", 2),
        output_size=patch_size,
        dropout=config.get("gru_dropout", 0.1),
    ).to(device)

    gru_optimizer = torch.optim.AdamW(
        gru_model.parameters(),
        lr=config.get("gru_lr", 1e-3),
    )

    best_gru_val_mse = float("inf")
    best_gru_state = None

    for epoch in range(config.get("gru_num_epochs", config["num_epochs"])):
        gru_model.train()

        total_gru_loss = 0.0

        for context_patches, target_patch in train_loader:
            context_patches = context_patches.to(device)
            target_patch = target_patch.to(device)
            target_patch = select_target_feature_tensor(
                target_patch,
                patch_size=patch_size,
                feature_dim=feature_dim,
                target_feature_index=target_feature_index,
            )

            pred = gru_model(context_patches)

            loss = torch.nn.functional.mse_loss(
                pred,
                target_patch,
                reduction="mean",
            )

            gru_optimizer.zero_grad()
            loss.backward()
            gru_optimizer.step()

            total_gru_loss += loss.item()

        avg_gru_loss = total_gru_loss / len(train_loader)

        gru_val_mse, gru_val_mae, _, _ = evaluate_gru_model(
            model=gru_model,
            loader=val_loader,
            device=device,
        )

        if gru_val_mse < best_gru_val_mse:
            best_gru_val_mse = gru_val_mse
            best_gru_state = {
                k: v.detach().cpu().clone()
                for k, v in gru_model.state_dict().items()
            }

        if epoch % 10 == 0:
            print(
                f"GRU Epoch {epoch}: "
                f"train_loss={avg_gru_loss:.6f}, "
                f"val_mse={gru_val_mse:.6f}, "
                f"val_mae={gru_val_mae:.6f}"
            )

    if best_gru_state is not None:
        gru_model.load_state_dict(best_gru_state)
        gru_model.to(device)

    print("GRU baseline training finished")

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
            save_dir=results_dir,
        )
    )

    trend_accuracy = compute_trend_accuracy(
        all_preds=all_preds,
        all_targets=all_targets,
        include_origin=forecast_target == "relative_return",
        direct_return=forecast_target
        in ("cumulative_log_return", "excess_log_return"),
    )

    print(f"========== Final Test ({data_title(config)}) ==========")
    print("TS-JEPA Test MSE is: {:.6f}".format(test_mse))
    print("TS-JEPA Test MAE is: {:.6f}".format(test_mae))
    print("TS-JEPA Trend Accuracy is: {:.4f}".format(trend_accuracy))

    gru_test_mse, gru_test_mae, gru_preds, gru_targets, _, _ = (
        prequential_gru_evaluate(
            model=gru_model,
            dataset=test_loader.dataset,
            device=device,
            config=config,
            save_dir=results_dir,
        )
    )

    gru_trend_accuracy = compute_trend_accuracy(
        all_preds=gru_preds,
        all_targets=gru_targets,
        include_origin=forecast_target == "relative_return",
        direct_return=forecast_target
        in ("cumulative_log_return", "excess_log_return"),
    )

    print(f"========== GRU Final Test ({data_title(config)}) ==========")
    print("GRU Test MSE is: {:.6f}".format(gru_test_mse))
    print("GRU Test MAE is: {:.6f}".format(gru_test_mae))
    print("GRU Trend Accuracy is: {:.4f}".format(gru_trend_accuracy))

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
            save_dir=results_dir,
        )
    )

    model_comparison_rows = [
        {
            "model": "TS-JEPA",
            "forecast_target": forecast_target,
            "mse": test_mse,
            "mae": test_mae,
            "trend_accuracy": trend_accuracy,
        },
        {
            "model": "GRU",
            "forecast_target": forecast_target,
            "mse": gru_test_mse,
            "mae": gru_test_mae,
            "trend_accuracy": gru_trend_accuracy,
        },
    ] + baseline_summary_rows

    comparison_paths = save_model_comparison(
        model_rows=model_comparison_rows,
        config=config,
        save_dir=results_dir,
    )

    visualize_model_comparison(
        model_rows=model_comparison_rows,
        config=config,
        save_dir=results_dir,
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
    plt.title(f"{data_title(config)} - Rolling Forecast Error over Time")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metric_png_path = os.path.join(
        results_dir,
        config["eval_type"] + f"_rolling_metrics_test_{timestamp}.png",
    )

    plt.savefig(metric_png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Rolling metric figure saved to {metric_png_path}")


    # =========================
    # Visualization 2:
    # Last point of each predicted patch
    # TS-JEPA vs GRU
    # =========================

    tsjepa_last = all_preds[:, -1]
    gru_last = gru_preds[:, -1]
    true_last = all_targets[:, -1]

    plt.figure(figsize=(12, 5))

    plt.plot(true_last, label="Ground Truth", linewidth=2)
    plt.plot(tsjepa_last, label="TS-JEPA Prediction", linewidth=2)
    plt.plot(gru_last, label="GRU Prediction", linewidth=2, linestyle="--")

    plt.legend()
    plt.xlabel("Rolling evaluation step")
    plt.ylabel(forecast_axis_label(config))
    plt.title(
        f"{data_title(config)} - Rolling Forecast: "
        "Last Point of Each Predicted Patch"
    )

    last_point_png_path = os.path.join(
        results_dir,
        config["eval_type"]
        + f"_rolling_last_point_tsjepa_vs_gru_test_{timestamp}.png",
    )

    plt.savefig(last_point_png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Rolling last-point TS-JEPA vs GRU figure saved to {last_point_png_path}")


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
        save_dir=os.path.join(results_dir, "rolling_windows_with_baselines"),
        sample_indices=[60],
        make_gif=False,
        baseline_names=[
            "naive_last",
            "previous_patch",
            "mean_context",
            "drift",
        ],
    )
    visualize_all_rolling_predictions_as_series(
        all_preds=all_preds,
        all_targets=all_targets,
        config=config,
        gru_preds=gru_preds,
        save_dir=results_dir,
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
