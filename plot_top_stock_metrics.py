import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_STOCK_ORDER = [
    "NVDA",
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "AVGO",
    "META",
    "TSLA",
    "GOOG",
    "WMT",
]

DEFAULT_MODELS = [
    "TS-JEPA",
    "GRU",
    "naive_last",
    "drift",
    "mean_context",
]

MODEL_COLORS = {
    "TS-JEPA": "tab:blue",
    "GRU": "tab:orange",
    "naive_last": "tab:green",
    "drift": "tab:red",
    "mean_context": "tab:purple",
}


def parse_data_source(txt_path):
    with txt_path.open() as f:
        for line in f:
            if line.startswith("Data source:"):
                return line.split(":", 1)[1].strip()
    return None


def latest_comparison_files(results_dir, seeds=None):
    latest_by_run = {}
    allowed_seed_parts = (
        {f"seed_{int(seed)}" for seed in seeds} if seeds is not None else None
    )
    for txt_path in sorted(results_dir.rglob("last_model_comparison_*.txt")):
        if allowed_seed_parts is not None and not allowed_seed_parts.intersection(
            txt_path.parent.parts
        ):
            continue
        stock = parse_data_source(txt_path)
        if not stock:
            continue

        csv_path = txt_path.with_suffix(".csv")
        if not csv_path.exists():
            continue

        run_key = (stock, txt_path.parent)
        if run_key not in latest_by_run or txt_path.name > latest_by_run[run_key].name:
            latest_by_run[run_key] = txt_path

    latest_by_stock = {}
    for (stock, _), txt_path in latest_by_run.items():
        latest_by_stock.setdefault(stock, []).append(txt_path)
    for stock, paths in latest_by_stock.items():
        nested_paths = [
            path
            for path in paths
            if stock in path.relative_to(results_dir).parts[:-1]
        ]
        if nested_paths:
            paths = nested_paths
        seed_paths = [
            path
            for path in paths
            if any(part.startswith("seed_") for part in path.parent.parts)
        ]
        latest_by_stock[stock] = sorted(seed_paths or paths)
    return latest_by_stock


def load_rows(results_dir, stock_order, models, seeds=None):
    latest_by_stock = latest_comparison_files(results_dir, seeds=seeds)
    rows = []

    for stock in stock_order:
        txt_paths = latest_by_stock.get(stock, [])
        if not txt_paths:
            print(f"Missing comparison result for {stock}; skipping")
            continue

        for txt_path in txt_paths:
            csv_path = txt_path.with_suffix(".csv")
            with csv_path.open(newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["model"] not in models:
                        continue
                    rows.append(
                        {
                            "stock": stock,
                            "model": row["model"],
                            "mse": float(row["mse"]),
                            "mae": float(row["mae"]),
                            "trend_accuracy": float(row["trend_accuracy"]),
                            "source_file": str(csv_path),
                        }
                    )

    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw

    aggregated = (
        raw.groupby(["stock", "model"], as_index=False)
        .agg(
            mse=("mse", "mean"),
            mse_std=("mse", "std"),
            mae=("mae", "mean"),
            mae_std=("mae", "std"),
            trend_accuracy=("trend_accuracy", "mean"),
            trend_accuracy_std=("trend_accuracy", "std"),
            num_runs=("source_file", "count"),
            source_files=("source_file", lambda values: ";".join(sorted(values))),
        )
    )
    std_columns = ["mse_std", "mae_std", "trend_accuracy_std"]
    aggregated[std_columns] = aggregated[std_columns].fillna(0.0)
    return aggregated


def plot_metric(
    ax,
    df,
    stock_order,
    models,
    metric,
    ylabel,
    title,
    ylim=None,
    hide_zero_models=None,
    reference_line=None,
    model_colors=None,
):
    pivot = (
        df.pivot_table(index="stock", columns="model", values=metric, aggfunc="first")
        .reindex(stock_order)
        .dropna(how="all")
    )
    std_column = metric + "_std"
    std_pivot = None
    if std_column in df.columns:
        std_pivot = (
            df.pivot_table(
                index="stock",
                columns="model",
                values=std_column,
                aggfunc="first",
            )
            .reindex(pivot.index)
        )
    x = range(len(pivot.index))
    group_width = 0.82
    width = group_width / max(len(models), 1)

    for i, model in enumerate(models):
        if model not in pivot.columns:
            continue
        values = pivot[model].copy()
        if hide_zero_models and model in hide_zero_models:
            values = values.mask(values == 0.0)
        offsets = [
            pos - group_width / 2 + width / 2 + i * width
            for pos in x
        ]
        color = model_colors.get(model) if model_colors else None
        errors = None
        if std_pivot is not None and model in std_pivot.columns:
            errors = std_pivot[model].fillna(0.0).values
        ax.bar(
            offsets,
            values.values,
            width=width,
            label=model,
            color=color,
            yerr=errors,
            capsize=2 if errors is not None else 0,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot.index, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    if reference_line is not None:
        ax.axhline(
            reference_line,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )
    if ylim is not None:
        ax.set_ylim(*ylim)


def save_combined_plot(df, stock_order, models, output_path, figure_title):
    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=False)
    fig.suptitle(
        figure_title,
        fontsize=16,
        fontweight="bold",
    )

    plot_metric(
        axes[0],
        df,
        stock_order,
        models,
        "mse",
        "MSE",
        "Mean Squared Error (lower is better)",
        model_colors=MODEL_COLORS,
    )
    plot_metric(
        axes[1],
        df,
        stock_order,
        models,
        "mae",
        "MAE",
        "Mean Absolute Error (lower is better)",
        model_colors=MODEL_COLORS,
    )
    trend_models = [
        model
        for model in models
        if model not in {"naive_last", "mean_context"}
    ]
    plot_metric(
        axes[2],
        df,
        stock_order,
        trend_models,
        "trend_accuracy",
        "Trend Accuracy",
        "Trend Accuracy (higher is better)",
        ylim=(0.0, 1.0),
        reference_line=0.5,
        model_colors=MODEL_COLORS,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Combine MSE, MAE, and trend accuracy for stock comparison runs."
    )
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output filename prefix. Defaults to top_<stock-count>_nasdaq100.",
    )
    parser.add_argument(
        "--figure-title",
        default=None,
        help="Figure title. Defaults to the output filename prefix.",
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--stocks", nargs="+", default=DEFAULT_STOCK_ORDER)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Only aggregate result directories named seed_<N> for these seeds.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_rows(results_dir, args.stocks, args.models, seeds=args.seeds)
    if df.empty:
        raise RuntimeError("No matching comparison rows found.")

    output_prefix = args.output_prefix or f"top_{len(args.stocks)}_nasdaq100"
    figure_title = args.figure_title or output_prefix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{output_prefix}_{timestamp}.csv"
    png_path = output_dir / f"{output_prefix}_{timestamp}.png"

    df.to_csv(csv_path, index=False)
    save_combined_plot(
        df,
        args.stocks,
        args.models,
        png_path,
        figure_title=figure_title,
    )

    print(f"Combined metrics CSV saved to: {csv_path}")
    print(f"Combined metrics figure saved to: {png_path}")


if __name__ == "__main__":
    main()
