#!/usr/bin/env python3
"""Copy validated figures and plot published Stage 05 aggregates; run from any cwd."""
import json
import shutil
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_chapter5_figures import REPO, STOCKS, SEEDS, SHARED, LOCAL, PDF_METADATA, verify_file

OUT = REPO / "figures/chapter5"
# Branch-specific validation selections, not a common hyperparameter comparison.
PROCEDURES = [
    ("Shared-Target / Market", "05_benchmark_market_only/0c0edc94f105-41b3195f1267", SHARED, 12, (1, 1), False, (0.00541, 0.00173, 0.00937)),
    ("Local-MAE/Long-JEPA / Market", "05_benchmark_local_long_market_only/9db20417d36d-e4e710020059", LOCAL, 24, (0, 2), False, (0.01213, 0.00251, 0.02665)),
    ("Shared-Target / Sentiment", "05_benchmark_sentiment/92f8a6ebc19a-5fc1ec922264", SHARED, 12, (0.5, 1.5), True, (0.01419, 0.00284, 0.03006)),
    ("Local-MAE/Long-JEPA / Sentiment", "05_benchmark_local_long_sentiment/3804d1fb7d09-f802b55ab4ee", LOCAL, 12, (1.5, 0.5), True, (0.00819, 0.00182, 0.01553)),
]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
MARKERS = ["o", "s", "^", "D"]
COPIES = [
    "stage00_mnist_reconstruction.pdf", "stage01_normalization_comparison.pdf",
    "stage02_sentiment_ablation.pdf", "stage03_context_sensitivity.pdf",
    "stage03_shared_vs_local.pdf", "stage04_loss_weight_sensitivity.pdf",
]


def load_data():
    """Runnable snapshot checks precede all output writes; no raw-data inference."""
    paired, horizons, baseline = [], [], None
    for label, relative, method, context, weights, sentiment, expected in PROCEDURES:
        root = REPO / "thesis_results" / relative
        for name in ("provenance/experiment_config.json", "data/paired_vs_naive.csv", "data/horizon_metrics.csv"):
            verify_file(root / name, root)
        config = json.loads((root / "provenance/experiment_config.json").read_text())
        runner, provenance = config["runner"], config["provenance"]
        assert set(config["common"]["stocks"]) == STOCKS
        assert set(config["common"]["seeds"]) == SEEDS
        assert runner["downstream"]["evaluation_split"] == "test"
        assert runner["downstream"]["context_size"] == context
        assert runner["downstream"]["forecast_horizon"] == 5
        assert tuple(runner["objectives"][key]["weight"] for key in ("jepa", "mae")) == weights
        assert runner["preprocessing"]["custom"]["features"]["sentiment"]["enabled"] == sentiment
        assert provenance["selection_source_split"] == "validation"
        assert provenance["selection_weights"] == {"rmse": 0.3, "direction_accuracy": 0.7}
        frame = pd.read_csv(root / "data/paired_vs_naive.csv")
        selected = frame.loc[frame.model == method]
        assert len(selected) == 1
        row = selected.iloc[0]
        assert row.n_stocks == 5 and row.rmse_holm_p_value >= 0.05
        values = row[["mean_delta_rmse", "rmse_ci_low", "rmse_ci_high"]].to_numpy(dtype=float)
        np.testing.assert_allclose(values, expected, rtol=0, atol=5e-6)
        assert values[1] <= values[0] <= values[2]
        paired.append(values)
        frame = pd.read_csv(root / "data/horizon_metrics.csv")
        selected = frame.loc[frame.method == method].sort_values("horizon").reset_index(drop=True)
        naive = frame.loc[frame.method == "Naive-last"].sort_values("horizon").reset_index(drop=True)
        for curve in (selected, naive):
            assert curve.horizon.tolist() == [1, 2, 3, 4, 5]
            assert (curve.n_stocks == 5).all() and (curve.n_runs == 15).all()
            assert np.isfinite(curve[["rmse", "direction_accuracy"]]).all().all()
        if baseline is None:
            baseline = naive
        else:
            pd.testing.assert_frame_equal(baseline, naive, check_exact=True)
        assert (selected.rmse > naive.rmse).all()
        horizons.append(selected)
    np.testing.assert_allclose([horizons[2].direction_accuracy[1], horizons[3].direction_accuracy[1]], [0.592, 0.5693333333333334], rtol=0, atol=1e-14)
    return np.array(paired), horizons, baseline


def save(fig, stem):
    metadata = {**PDF_METADATA, "Creator": "analysis/finalize_chapter5_figures.py"}
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", metadata=metadata)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage05-only", action="store_true", help="Generate only the two Stage 05 figures.")
    args = parser.parse_args()
    paired, horizons, baseline = load_data()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.stage05_only:
        smoke = REPO / "thesis_results/00_dual_loss_smoke/e56db7d33c56"
        verify_file(smoke / "smoke_sin_cos_predictions.png", smoke)
        for filename in COPIES:
            source = REPO / "thesis_results/chapter5_figures" / filename
            shutil.copyfile(source, OUT / filename)
            assert source.read_bytes() == (OUT / filename).read_bytes()
        shutil.copyfile(smoke / "smoke_sin_cos_predictions.png", OUT / "stage00_sin_cos_predictions.png")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "legend.fontsize": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    fig.subplots_adjust(left=0.34, right=0.98, bottom=0.29, top=0.85)
    for i, ((mean, low, high), color, marker) in enumerate(zip(paired, COLORS, MARKERS)):
        ax.errorbar(mean, i, xerr=[[mean-low], [high-mean]], fmt=marker, color=color, capsize=4, linewidth=1.7)
        ax.text(1.02, i, f"{mean:+.5f}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8.5)
    ax.axvline(0, color="#666666", linestyle="--", linewidth=1)
    ax.set_yticks(range(4), [p[0] for p in PROCEDURES])
    ax.set_ylim(3.6, -0.6)
    ax.set_xlim(-0.0015, 0.034)
    ax.set_xlabel(r"$\Delta$ RMSE = RMSE$_{model}$ - RMSE$_{naive}$ (saved target space)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Stage 05: paired RMSE difference to Naive-last", fontweight="bold")
    fig.text(0.02, 0.04, "5 equities; seeds 42, 44, 46 averaged within each equity before primary inference.\n95% equity-bootstrap CIs; positive Δ RMSE favours Naive-last.\nRetrospective held-out-test evaluation of validation-selected complete procedures.", fontsize=8.5)
    save(fig, "stage05_paired_rmse_vs_naive")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.8), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.79, bottom=0.14, hspace=0.22)
    for i, curve in enumerate(horizons + [baseline]):
        label = PROCEDURES[i][0] if i < 4 else "Naive-last"
        color = COLORS[i] if i < 4 else "#333333"
        for ax, metric, scale in zip(axes, ["rmse", "direction_accuracy"], [1, 100]):
            ax.plot(curve.horizon, curve[metric] * scale, label=label, color=color,
                    marker=MARKERS[i] if i < 4 else "x", linestyle="--" if i in (1, 3, 4) else "-", linewidth=1.6, markersize=5)
    axes[0].set_ylabel("RMSE (saved target space)")
    axes[1].set_ylabel("Direction Accuracy (%)")
    axes[1].set_xlabel("Forecast horizon h")
    axes[1].set_xticks(range(1, 6))
    axes[1].set_ylim(-2, 65)
    for index, offset, precision in [(2, (8, 12), 1), (3, (8, -22), 2)]:
        value = horizons[index].direction_accuracy[1] * 100
        axes[1].annotate(f"{value:.{precision}f}%", (2, value), xytext=offset,
                         textcoords="offset points", color=COLORS[index], fontsize=8)
    for ax, title in zip(axes, ["(a) RMSE", "(b) Direction Accuracy"]):
        ax.set_title(title, loc="left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)
    fig.suptitle("Stage 05: forecast-horizon comparison", fontweight="bold", y=0.98)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    fig.text(0.02, 0.025, "Published aggregates over 5 equities and seeds 42, 44, 46; saved direction metric.\nNaive-last is near zero by definition under this metric; its identical trajectory is shown once.\nRetrospective held-out-test evaluation of validation-selected complete procedures.", fontsize=8)
    save(fig, "stage05_horizon_comparison")
    print("Snapshot checks passed; generated 2 vector PDFs with PNG previews.")


if __name__ == "__main__":
    main()
