#!/usr/bin/env python3
"""Rebuild the Chapter 5 figures from immutable thesis-result snapshots."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "thesis_results" / "chapter5_figures"
STOCKS = {"NVDA", "AAPL", "AVGO", "TSLA", "WMT"}
SEEDS = {42, 44, 46}
SHARED = "Shared-target JEPA--MAE"
LOCAL = "Local-MAE/Long-JEPA"

SNAPSHOTS = {
    "stage00": "thesis_results/00_dual_loss_smoke/e56db7d33c56",
    "stage01_zscore": "thesis_results/01_preprocessing_train_zscore/b643221e52a1-c018af4432c2",
    "stage01_window": "thesis_results/01_preprocessing_window_return/568f317f7e2c-1b543b18e1bc",
    "stage02_sentiment": "thesis_results/02_sentiment_included/afc29d4b16b6-655ec105d05b",
    "stage03_shared_6": "thesis_results/03_shared_context_6_patches/49b848da7bb2-2609b9618086",
    "stage03_shared_12": "thesis_results/03_shared_context_12_patches/afc29d4b16b6-8b70a34f4982",
    "stage03_shared_24": "thesis_results/03_shared_context_24_patches/4f778fee2b9b-6803fe2fb974",
    "stage03_local_6": "thesis_results/03_local_long_context_6_patches/643dd2d957d2-00670fa6e96b",
    "stage03_local_12": "thesis_results/03_local_long_context_12_patches/a60306e1c2cc-41dcaf5bc18d",
    "stage03_local_24": "thesis_results/03_local_long_context_24_patches/8ac63bf9b1eb-cfc753ae10fe",
    "stage04_shared_0_2": "thesis_results/04_shared_joint_loss_jepa_0_mae_2/ccf448a9d694-22d950d16002",
    "stage04_shared_05_15": "thesis_results/04_shared_joint_loss_jepa_0_5_mae_1_5/2e7d28d68c3c-d65be5e42b4d",
    "stage04_shared_1_1": "thesis_results/04_shared_joint_loss_jepa_1_mae_1/429e4337b133-4bb8562f87e0",
    "stage04_shared_15_05": "thesis_results/04_shared_joint_loss_jepa_1_5_mae_0_5/bc257b6d2afb-9286499b1a2c",
    "stage04_shared_2_0": "thesis_results/04_shared_joint_loss_jepa_2_mae_0/a67f8a1d8485-f64f1511a92a",
    "stage04_local_0_2": "thesis_results/04_local_long_joint_loss_jepa_0_mae_2/eb1cd970301e-c62410937c40",
    "stage04_local_05_15": "thesis_results/04_local_long_joint_loss_jepa_0_5_mae_1_5/a599bb380215-6a3e8f8b241b",
    "stage04_local_1_1": "thesis_results/04_local_long_joint_loss_jepa_1_mae_1/62ab3dac50e6-d7532ba2336b",
    "stage04_local_15_05": "thesis_results/04_local_long_joint_loss_jepa_1_5_mae_0_5/6ae736eff556-87de0125c7ce",
    "stage04_local_2_0": "thesis_results/04_local_long_joint_loss_jepa_2_mae_0/f68c6f9bff75-bc2d1a5dbd02",
}

EXPECTED = {
    "stage01_window": (0.052634, 0.5478),
    "stage01_zscore": (0.056598, 0.5356),
    "stage02_sentiment": (0.049247, 0.5233),
    "stage03_shared_6": (0.049227, None),
    "stage03_shared_12": (0.047260, None),
    "stage03_shared_24": (0.049475, None),
    "stage03_local_6": (0.049541, None),
    "stage03_local_12": (0.048152, None),
    "stage03_local_24": (0.049353, None),
    "stage04_shared_0_2": (0.047470, 0.5178),
    "stage04_shared_05_15": (0.047751, 0.5222),
    "stage04_shared_1_1": (0.047898, 0.5156),
    "stage04_shared_15_05": (0.047690, 0.5200),
    "stage04_shared_2_0": (0.048501, 0.5211),
    "stage04_local_0_2": (0.048328, 0.5233),
    "stage04_local_05_15": (0.048021, 0.5261),
    "stage04_local_1_1": (0.047529, 0.5378),
    "stage04_local_15_05": (0.047548, 0.5539),
    "stage04_local_2_0": (0.047736, 0.5444),
}

BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#666666"
PDF_METADATA = {
    "Title": "TS-JEPA Chapter 5 figure",
    "Author": "TS-JEPA",
    "Creator": "analysis/plot_chapter5_figures.py",
    "CreationDate": None,
    "ModDate": None,
}


def snapshot(key: str) -> Path:
    path = REPO / SNAPSHOTS[key]
    if not path.is_dir():
        raise FileNotFoundError(f"Required snapshot is missing: {path}")
    return path


def verify_file(path: Path, root: Path) -> None:
    checksums = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        checksums[relative.strip()] = digest
    relative = path.relative_to(root).as_posix()
    expected = checksums.get(relative)
    if expected is None:
        raise ValueError(f"{relative} is not covered by {root / 'SHA256SUMS'}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Checksum mismatch: {path}")


def load_snapshot(
    key: str,
    method: str,
    strategy: str,
    context: int,
    origins: int,
    target_start: str,
    target_end: str,
) -> dict:
    root = snapshot(key)
    paths = {
        name: root / relative
        for name, relative in {
            "config": "provenance/experiment_config.json",
            "overall": "data/overall_summary.csv",
            "stock": "data/stock_summary.csv",
            "predictions": "data/predictions_tidy.csv",
        }.items()
    }
    for path in paths.values():
        verify_file(path, root)

    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    runner = config["runner"]
    if runner["downstream"]["evaluation_split"] != "validation":
        raise ValueError(f"{key} is not validation-only")
    if int(runner["downstream"]["context_size"]) != context:
        raise ValueError(f"{key} has the wrong context size")
    if set(runner["masking"]["strategies"]) != {strategy}:
        raise ValueError(f"{key} has the wrong strategy")
    if set(config["common"]["stocks"]) != STOCKS or set(config["common"]["seeds"]) != SEEDS:
        raise ValueError(f"{key} has the wrong stock or seed set")
    if key.startswith("stage04_shared"):
        provenance = config["provenance"]
        if provenance.get("parent_candidate_id") != "shared_context_12":
            raise ValueError(f"{key} does not inherit corrected Stage 03")

    overall = pd.read_csv(paths["overall"])
    stock = pd.read_csv(paths["stock"])
    predictions = pd.read_csv(paths["predictions"])
    row = overall.loc[overall["method"] == method]
    stock_rows = stock.loc[stock["method"] == method].copy()
    prediction_rows = predictions.loc[predictions["method"] == method].copy()
    if len(row) != 1 or set(stock_rows["stock"]) != STOCKS:
        raise ValueError(f"{key} is missing aggregate rows for {method}")
    if not (stock_rows["n_valid_seeds"] == len(SEEDS)).all() or not (
        stock_rows["n_direction_seeds"] == len(SEEDS)
    ).all():
        raise ValueError(f"{key} is missing seed coverage")
    if set(prediction_rows["strategy"]) != {strategy}:
        raise ValueError(f"{key} prediction strategy mismatch")
    if prediction_rows["rolling_step"].nunique() != origins:
        raise ValueError(f"{key} has the wrong number of validation origins")
    dates = pd.to_datetime(prediction_rows["target_date"])
    if dates.min().isoformat() != target_start or dates.max().isoformat() != target_end:
        raise ValueError(f"{key} has the wrong validation target period")
    horizon = int(runner["downstream"]["forecast_horizon"])
    expected_rows = origins * horizon
    group_sizes = prediction_rows.groupby(["stock", "seed"]).size()
    if len(group_sizes) != len(STOCKS) * len(SEEDS) or not (group_sizes == expected_rows).all():
        raise ValueError(f"{key} has incomplete prediction coverage")

    stored_means = stock_rows.set_index("stock")[["rmse_mean", "direction_accuracy_mean"]].sort_index()
    selected = row.iloc[0]
    if not np.isclose(selected["rmse"], stored_means["rmse_mean"].mean(), rtol=0, atol=1e-12):
        raise ValueError(f"{key} overall RMSE aggregation mismatch")
    if not np.isclose(
        selected["direction_accuracy"], stored_means["direction_accuracy_mean"].mean(), rtol=0, atol=1e-12
    ):
        raise ValueError(f"{key} overall direction aggregation mismatch")
    if not np.isclose(selected["rmse_std_across_stocks"], stored_means["rmse_mean"].std(ddof=1), rtol=0, atol=1e-12):
        raise ValueError(f"{key} RMSE uncertainty aggregation mismatch")
    if not np.isclose(
        selected["direction_accuracy_std_across_stocks"],
        stored_means["direction_accuracy_mean"].std(ddof=1),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(f"{key} direction uncertainty aggregation mismatch")

    return {
        "key": key,
        "root": root,
        "method": method,
        "rmse": float(selected["rmse"]),
        "rmse_sd": float(selected["rmse_std_across_stocks"]),
        "direction": float(selected["direction_accuracy"]),
        "direction_sd": float(selected["direction_accuracy_std_across_stocks"]),
        "stock": stock_rows.set_index("stock").sort_index(),
        "inputs": list(paths.values()),
    }


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", metadata=PDF_METADATA)
    fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.11, 1.04, label, transform=ax.transAxes, fontweight="bold", va="bottom")


def metric_axes(ax: plt.Axes, metric: str, upper: float) -> None:
    ax.set_ylim(0, upper)
    ax.set_ylabel(metric)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def bar_comparison(title: str, labels: list[str], data: list[dict]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), constrained_layout=True)
    colors = [BLUE, ORANGE]
    hatches = ["///", "..."]
    for index, (ax, metric, sd, ylabel, panel) in enumerate(
        zip(
            axes,
            ("rmse", "direction"),
            ("rmse_sd", "direction_sd"),
            ("RMSE (lower is better)", "Direction Accuracy (higher is better)"),
            ("(a)", "(b)"),
        )
    ):
        values = np.array([item[metric] for item in data])
        errors = np.array([item[sd] for item in data])
        bars = ax.bar(
            np.arange(len(labels)), values, yerr=errors, width=0.62, color=colors,
            edgecolor="black", linewidth=0.8, hatch=hatches, capsize=4,
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
        )
        upper = max(0.11 if index == 0 else 0.65, float((values + errors).max() * 1.18))
        metric_axes(ax, ylabel, upper)
        ax.set_xticks(range(len(labels)), labels)
        for bar, value, error in zip(bars, values, errors):
            text = f"{value:.4f}" if index == 0 else f"{100 * value:.1f}%"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + error + upper * 0.012,
                text,
                ha="center",
                va="bottom",
            )
        panel_label(ax, panel)
    fig.suptitle(title, fontweight="bold")
    return fig


def line_panels(
    title: str,
    x: np.ndarray,
    tick_labels: list[str],
    shared: list[dict],
    local: list[dict],
    selected_x: float | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), constrained_layout=True)
    for index, (ax, metric, sd, ylabel, panel) in enumerate(
        zip(
            axes,
            ("rmse", "direction"),
            ("rmse_sd", "direction_sd"),
            ("RMSE (lower is better)", "Direction Accuracy (higher is better)"),
            ("(a)", "(b)"),
        )
    ):
        shared_values = np.array([item[metric] for item in shared])
        local_values = np.array([item[metric] for item in local])
        shared_errors = np.array([item[sd] for item in shared])
        local_errors = np.array([item[sd] for item in local])
        offset = 0.0 if selected_x is not None else 0.045
        ax.errorbar(
            x - offset, shared_values, yerr=shared_errors, color=BLUE, marker="o",
            linestyle="-", linewidth=1.7, markersize=5, capsize=3, label="Shared-Target",
        )
        ax.errorbar(
            x + offset, local_values, yerr=local_errors, color=ORANGE, marker="s",
            linestyle="--", linewidth=1.7, markersize=5, capsize=3, label="Local-MAE/Long-JEPA",
        )
        upper = max(0.09 if index == 0 else 0.65, float(max((shared_values + shared_errors).max(), (local_values + local_errors).max()) * 1.12))
        metric_axes(ax, ylabel, upper)
        ax.set_xticks(x, tick_labels)
        if selected_x is not None:
            ax.axvline(selected_x, color=GRAY, linewidth=1.0, linestyle=":", zorder=0)
            if index == 0:
                ax.text(
                    selected_x + 0.45,
                    upper * 0.025,
                    "selected context",
                    rotation=90,
                    ha="left",
                    va="bottom",
                    color=GRAY,
                    fontsize=8,
                )
        ax.legend(
            frameon=False,
            loc="lower left" if index == 0 else "upper left",
            bbox_to_anchor=(0, 0.04) if index == 0 else None,
        )
        panel_label(ax, panel)
        if index == 0:
            inset = ax.inset_axes([0.39, 0.56, 0.58, 0.34])
            inset.plot(x - offset, shared_values, color=BLUE, marker="o", linewidth=1.2, markersize=3)
            inset.plot(x + offset, local_values, color=ORANGE, marker="s", linestyle="--", linewidth=1.2, markersize=3)
            low = float(min(shared_values.min(), local_values.min()))
            high = float(max(shared_values.max(), local_values.max()))
            padding = max((high - low) * 0.22, 0.00015)
            inset.set_ylim(low - padding, high + padding)
            inset.set_xticks([])
            inset.tick_params(axis="y", labelsize=6.5, length=2)
            inset.grid(axis="y", color="#D0D0D0", linewidth=0.4, alpha=0.6)
            inset.set_title("Means only (expanded scale)", fontsize=7, pad=2)
            inset.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontweight="bold")
    return fig


def source_list(items: list[dict], filenames: list[str]) -> list[str]:
    lines = []
    for item in items:
        root = item["root"]
        lines.append(f"- Snapshot: `{root.relative_to(REPO).as_posix()}`")
        for filename in filenames:
            lines.append(f"  - `{(root / filename).relative_to(REPO).as_posix()}`")
    return lines


def values_table(labels: list[str], items: list[dict]) -> list[str]:
    lines = ["| Configuration | RMSE | Direction Accuracy |", "|---|---:|---:|"]
    lines.extend(
        f"| {label} | {item['rmse']:.6f} | {100 * item['direction']:.2f}% |"
        for label, item in zip(labels, items)
    )
    return lines


def main() -> int:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)

    stage00 = snapshot("stage00")
    stage00_summary = stage00 / "summary.csv"
    stage00_image = stage00 / "smoke_mnist_rows_reconstruction.png"
    verify_file(stage00_summary, stage00)
    verify_file(stage00_image, stage00)
    smoke = pd.read_csv(stage00_summary)
    mnist = smoke.loc[smoke["case"] == "MNIST_ROWS"]
    if len(mnist) != 1 or not np.isclose(mnist.iloc[0]["model_rmse"], 0.111085, atol=1e-12):
        raise ValueError("Stage 00 MNIST diagnostic does not match its validated summary")
    shutil.copyfile(stage00_image, OUT / "stage00_mnist_reconstruction.png")
    image = np.asarray(Image.open(stage00_image))
    fig = plt.figure(figsize=(6.6, 6.6 * image.shape[0] / image.shape[1]))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.axis("off")
    fig.savefig(OUT / "stage00_mnist_reconstruction.pdf", metadata=PDF_METADATA)
    plt.close(fig)

    stage01 = [
        load_snapshot("stage01_window", SHARED, "random", 12, 12, "2024-10-02T00:00:00", "2024-12-26T00:00:00"),
        load_snapshot("stage01_zscore", SHARED, "random", 12, 12, "2024-10-02T00:00:00", "2024-12-26T00:00:00"),
    ]
    save_figure(
        bar_comparison("Stage 01: normalization comparison", ["Window-relative", "Train z-score"], stage01),
        "stage01_normalization_comparison",
    )

    stage02 = [
        stage01[0],
        load_snapshot("stage02_sentiment", SHARED, "random", 12, 12, "2024-10-02T00:00:00", "2024-12-26T00:00:00"),
    ]
    save_figure(
        bar_comparison("Stage 02: sentiment ablation", ["Market only", "Market + sentiment"], stage02),
        "stage02_sentiment_ablation",
    )

    contexts = [6, 12, 24]
    stage03_shared = [
        load_snapshot(f"stage03_shared_{context}", SHARED, "random", context, 24, "2024-07-09T00:00:00", "2024-12-26T00:00:00")
        for context in contexts
    ]
    stage03_local = [
        load_snapshot(f"stage03_local_{context}", LOCAL, "local_long", context, 24, "2024-07-09T00:00:00", "2024-12-26T00:00:00")
        for context in contexts
    ]
    save_figure(
        line_panels(
            "Stage 03: context-length sensitivity",
            np.array(contexts, dtype=float),
            ["6\n(30 obs.)", "12\n(60 obs.)", "24\n(120 obs.)"],
            stage03_shared,
            stage03_local,
            selected_x=12,
        ),
        "stage03_context_sensitivity",
    )

    shared12, local12 = stage03_shared[1], stage03_local[1]
    stock_order = ["AAPL", "AVGO", "NVDA", "TSLA", "WMT"]
    fig, ax = plt.subplots(figsize=(6.6, 4.25), constrained_layout=True)
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#666666"]
    for stock_name, color in zip(stock_order, palette):
        values = [
            shared12["stock"].loc[stock_name, "rmse_mean"],
            local12["stock"].loc[stock_name, "rmse_mean"],
        ]
        ax.plot([0, 1], values, color=color, marker="o", linewidth=1.3, markersize=4.5)
        ax.text(1.035, values[1], stock_name, color=color, va="center", fontsize=8.5)
    means = [shared12["rmse"], local12["rmse"]]
    ax.plot([0, 1], means, color="black", marker="D", linestyle="--", linewidth=2.0, markersize=6, label="Overall mean")
    ax.set_xlim(-0.15, 1.28)
    metric_axes(ax, "RMSE (lower is better)", 0.09)
    ax.set_xticks([0, 1], ["Shared-Target", "Local-MAE/Long-JEPA"])
    ax.set_title("Stage 03: matched-context stock-level comparison", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.5, -0.16,
        f"Direction Accuracy: Shared-Target {100 * shared12['direction']:.2f}%; Local-MAE/Long-JEPA {100 * local12['direction']:.2f}%",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
    )
    save_figure(fig, "stage03_shared_vs_local")

    weight_keys = ["0_2", "05_15", "1_1", "15_05", "2_0"]
    stage04_shared = [
        load_snapshot(f"stage04_shared_{key}", SHARED, "random", 12, 24, "2024-07-09T00:00:00", "2024-12-26T00:00:00")
        for key in weight_keys
    ]
    stage04_local = [
        load_snapshot(f"stage04_local_{key}", LOCAL, "local_long", 12, 24, "2024-07-09T00:00:00", "2024-12-26T00:00:00")
        for key in weight_keys
    ]
    weight_labels = [
        "MAE only\n(0, 2)", "MAE-heavy\n(0.5, 1.5)", "Balanced\n(1, 1)",
        "JEPA-heavy\n(1.5, 0.5)", "JEPA only\n(2, 0)",
    ]
    save_figure(
        line_panels(
            "Stage 04: JEPA-MAE loss-weight sensitivity",
            np.arange(5, dtype=float),
            weight_labels,
            stage04_shared,
            stage04_local,
        ),
        "stage04_loss_weight_sensitivity",
    )

    actual_by_key = {item["key"]: item for item in stage01 + stage02[1:] + stage03_shared + stage03_local + stage04_shared + stage04_local}
    mismatches = []
    for key, (expected_rmse, expected_direction) in EXPECTED.items():
        actual = actual_by_key[key]
        if abs(actual["rmse"] - expected_rmse) > 5e-7:
            mismatches.append(f"{key} RMSE: expected approximately {expected_rmse:.6f}, actual {actual['rmse']:.9f}")
        if expected_direction is not None and abs(actual["direction"] - expected_direction) > 5e-5:
            mismatches.append(f"{key} Direction Accuracy: expected approximately {100 * expected_direction:.2f}%, actual {100 * actual['direction']:.4f}%")

    common_inputs = [
        "provenance/experiment_config.json",
        "data/overall_summary.csv",
        "data/stock_summary.csv",
        "data/predictions_tidy.csv",
    ]
    readme = [
        "# Chapter 5 publication figures",
        "",
        "These figures are generated deterministically by `analysis/plot_chapter5_figures.py` from immutable, checksum-verified thesis-result snapshots. No experiment output is modified and no training is performed.",
        "",
        "The stock set is AAPL, AVGO, NVDA, TSLA, and WMT; the seed set is 42, 44, and 46. Learned-model metrics are averaged over seeds within each stock and then across the five stock means. Error bars show the sample standard deviation across those five stock-level seed means; they are cross-stock dispersion, not confidence intervals.",
        "",
        "Stage 01-02 use 12 validation origins (2024-10-02 to 2024-12-26). Stage 03-04 use 24 validation origins (2024-07-09 to 2024-12-26). They are not plotted as an additive learning curve.",
        "",
        "## Figure 1 - Stage 00 model validation",
        "",
        "Caption: Representative controlled-data diagnostic used to verify the implementation before the financial forecasting experiments.",
        "",
        f"- Snapshot: `{stage00.relative_to(REPO).as_posix()}`",
        f"- Inputs: `{stage00_summary.relative_to(REPO).as_posix()}`, `{stage00_image.relative_to(REPO).as_posix()}`",
        f"- Plotted diagnostic: MNIST-row model RMSE {mnist.iloc[0]['model_rmse']:.6f}; previous-row RMSE {mnist.iloc[0]['naive_last_rmse']:.6f}.",
        "- No stock, seed, or financial validation period applies. No uncertainty bar is shown.",
        "- This is an implementation diagnostic, not financial forecasting evidence.",
        "",
        "## Figure 2 - Stage 01 normalization comparison",
        "",
        *source_list(stage01, common_inputs),
        "",
        *values_table(["Window-relative", "Train z-score"], stage01),
        "",
        "Error bars: sample standard deviation across five stock-level seed means.",
        "",
        "## Figure 3 - Stage 02 sentiment ablation",
        "",
        *source_list(stage02, common_inputs),
        "",
        *values_table(["Market only", "Market + sentiment"], stage02),
        "",
        "The market-only row reuses the validated Stage 01 window-relative snapshot, as Stage 02 changed only sentiment inclusion. Error bars show the sample standard deviation across five stock-level seed means.",
        "",
        "## Figure 4 - Stage 03 context-length sensitivity",
        "",
        *source_list(stage03_shared + stage03_local, common_inputs),
        "",
        *values_table(
            [f"Shared-Target, {context} patches" for context in contexts] + [f"Local-MAE/Long-JEPA, {context} patches" for context in contexts],
            stage03_shared + stage03_local,
        ),
        "",
        "All six snapshots have identical 24-origin validation support. Error bars show the sample standard deviation across five stock-level seed means. Twelve patches correspond to 60 observations and are the selected Stage 03 context.",
        "",
        "## Figure 5 - Stage 03 Shared-Target versus Local-MAE/Long-JEPA",
        "",
        *source_list([shared12, local12], common_inputs),
        "",
        "Each connected line is one stock after averaging its three seeds. The black diamond is the mean across the five stocks. No error bars are shown; the paired stock lines display the cross-stock variation directly.",
        "",
        *values_table(["Shared-Target, 12 patches", "Local-MAE/Long-JEPA, 12 patches"], [shared12, local12]),
        "",
        "## Figure 6 - Stage 04 JEPA-MAE loss-weight sensitivity",
        "",
        *source_list(stage04_shared + stage04_local, common_inputs),
        "",
        *values_table(
            [f"Shared-Target, {label.replace(chr(10), ' ')}" for label in weight_labels]
            + [f"Local-MAE/Long-JEPA, {label.replace(chr(10), ' ')}" for label in weight_labels],
            stage04_shared + stage04_local,
        ),
        "",
        "All ten snapshots use context size 12 and identical 24-origin validation support. Every Shared-Target snapshot records `shared_context_12` as its parent. Error bars show the sample standard deviation across five stock-level seed means.",
        "",
        f"The best Stage 04 RMSE is Shared-Target (0, 2) at {stage04_shared[0]['rmse']:.6f}. The best Local-MAE/Long-JEPA RMSE is (1, 1) at {stage04_local[2]['rmse']:.6f}; the relative difference is {100 * (stage04_local[2]['rmse'] - stage04_shared[0]['rmse']) / stage04_local[2]['rmse']:.3f}%.",
        "",
        "## Expected-value audit",
        "",
        *(mismatches if mismatches else ["No material mismatch: every requested approximate value agrees with the checksum-verified snapshot value at the stated precision."]),
        "",
    ]
    (OUT / "README.md").write_text("\n".join(readme), encoding="utf-8")

    outputs = sorted(path.relative_to(REPO).as_posix() for path in OUT.iterdir() if path.is_file())
    print("Generated files:")
    for output in outputs:
        print(f"- {output}")
    print("Source experiment directories:")
    for key in SNAPSHOTS:
        print(f"- {key}: {SNAPSHOTS[key]}")
    print("Expected-value mismatches:")
    print("- none" if not mismatches else "\n".join(f"- {item}" for item in mismatches))
    print("Figures not generated:")
    print("- none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
