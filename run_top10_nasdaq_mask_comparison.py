"""Run random and local-long masking on the top 10 NASDAQ-100 stocks.

Each strategy is run once per stock and seed. The default seed list contains
10 seeds, producing 100 experiments per strategy. Results are summarized with
the mean and sample standard deviation of MSE, MAE, and trend accuracy.
"""

import argparse
import csv
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from download_indices_and_news import TOP_NASDAQ100_STOCKS
from plot_top_stock_metrics import DEFAULT_MODELS, latest_comparison_files


MASK_STRATEGIES = ("random", "local_long")
METRICS = ("mse", "mae", "trend_accuracy")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run random and local-long masking on the top 10 NASDAQ-100 "
            "stocks repeatedly, then report metric means and standard deviations."
        )
    )
    parser.add_argument(
        "--stocks",
        nargs="+",
        default=TOP_NASDAQ100_STOCKS,
        help="Stocks to evaluate. Defaults to the repository's top 10 list.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Runs per strategy and stock (default: 10).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
        help="First seed when --seeds is not supplied (default: 42).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit unique seeds. Overrides --runs and --seed-start.",
    )
    parser.add_argument(
        "--results-dir",
        default="./results/top10_nasdaq100_mask_comparison",
        help="Root directory for strategy runs and summary CSV files.",
    )
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Download/refresh stock data before the first strategy. By default, "
            "the existing data/<TICKER>/<TICKER>.csv files are used."
        ),
    )
    parser.add_argument("--skip-news", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--pretrain-num-epochs", type=int, default=2001)
    parser.add_argument("--eval-num-epochs", type=int, default=501)
    parser.add_argument("--checkpoint-to-use", type=int, default=2000)
    parser.add_argument("--use-best-checkpoint", action="store_true")
    parser.add_argument("--pretrain-stride", type=int, default=5)
    parser.add_argument(
        "--sampling-mode",
        choices=("sliding_window", "temporal_segments"),
        default="sliding_window",
    )
    parser.add_argument(
        "--normalization",
        choices=("window_return", "train_zscore", "none"),
        default="window_return",
    )
    parser.add_argument(
        "--encoder-weights",
        choices=("ema", "online"),
        default="ema",
    )
    parser.add_argument(
        "--forecast-target",
        choices=("value", "relative_return"),
        default="value",
        help="Downstream value path or cutoff-relative return path.",
    )
    parser.add_argument("--lambda-jepa", type=float, default=1.0)
    parser.add_argument("--lambda-mae", type=float, default=0.5)
    parser.add_argument(
        "--jepa-loss",
        choices=("mse", "l1", "smooth_l1"),
        default="mse",
    )
    parser.add_argument(
        "--mae-loss",
        choices=("mse", "l1", "smooth_l1"),
        default="mse",
    )
    parser.add_argument("--mae-window-patches", type=int, default=1)
    parser.add_argument("--jepa-gap-patches", type=int, default=4)
    parser.add_argument("--jepa-target-patches", type=int, default=4)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Do not launch experiments; rebuild summaries from existing results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all experiment commands without training or aggregating.",
    )
    return parser.parse_args(argv)


def resolve_seeds(args):
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    seeds = (
        list(args.seeds)
        if args.seeds is not None
        else list(range(args.seed_start, args.seed_start + args.runs))
    )
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique values")
    return seeds


def selected_stocks(args):
    stocks = [stock.upper() for stock in args.stocks]
    if len(stocks) != 10:
        raise ValueError(
            "This comparison requires exactly 10 stocks; pass exactly 10 with --stocks."
        )
    if len(stocks) != len(set(stocks)):
        raise ValueError("--stocks must contain 10 unique tickers")
    return stocks


def build_strategy_command(args, strategy, stocks, seeds):
    strategy_results_dir = Path(args.results_dir) / strategy
    command = [
        sys.executable,
        "-u",
        "run_top_nasdaq100_stocks.py",
        "--stocks",
        *stocks,
        "--max-stocks",
        "0",
        "--mask-strategy",
        strategy,
        "--seeds",
        *[str(seed) for seed in seeds],
        "--results-dir",
        str(strategy_results_dir),
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--pretrain-num-epochs",
        str(args.pretrain_num_epochs),
        "--eval-num-epochs",
        str(args.eval_num_epochs),
        "--checkpoint-to-use",
        str(args.checkpoint_to_use),
        "--pretrain-stride",
        str(args.pretrain_stride),
        "--sampling-mode",
        args.sampling_mode,
        "--normalization",
        args.normalization,
        "--encoder-weights",
        args.encoder_weights,
        "--forecast-target",
        getattr(args, "forecast_target", "value"),
        "--lambda-jepa",
        str(args.lambda_jepa),
        "--lambda-mae",
        str(args.lambda_mae),
        "--jepa-loss",
        args.jepa_loss,
        "--mae-loss",
        args.mae_loss,
    ]

    if strategy == "local_long":
        command.extend(
            [
                "--mae-window-patches",
                str(args.mae_window_patches),
                "--jepa-gap-patches",
                str(args.jepa_gap_patches),
                "--jepa-target-patches",
                str(args.jepa_target_patches),
            ]
        )
    if not args.download or strategy != MASK_STRATEGIES[0]:
        command.append("--skip-download")
    if args.skip_news:
        command.append("--skip-news")
    if args.skip_pretrain:
        command.append("--skip-pretrain")
    if args.use_best_checkpoint:
        command.append("--use-best-checkpoint")
    if args.dry_run:
        command.append("--dry-run")
    return command


def run_command(command):
    print("=" * 80, flush=True)
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _seed_from_path(path):
    for part in path.parent.parts:
        if part.startswith("seed_"):
            try:
                return int(part.removeprefix("seed_"))
            except ValueError:
                continue
    raise ValueError(f"Could not determine seed from result path: {path}")


def collect_raw_results(results_dir, strategy, stocks, seeds):
    latest_by_stock = latest_comparison_files(results_dir, seeds=seeds)
    rows = []

    for stock in stocks:
        for txt_path in latest_by_stock.get(stock, []):
            seed = _seed_from_path(txt_path)
            with txt_path.with_suffix(".csv").open(newline="") as csv_file:
                for row in csv.DictReader(csv_file):
                    rows.append(
                        {
                            "strategy": strategy,
                            "stock": stock,
                            "seed": seed,
                            "model": row["model"],
                            **{metric: float(row[metric]) for metric in METRICS},
                        }
                    )

    return pd.DataFrame(rows)


def aggregate_metrics(raw_results, group_columns):
    aggregations = {}
    for metric in METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    aggregations["num_runs"] = ("seed", "count")

    summary = (
        raw_results.groupby(group_columns, as_index=False)
        .agg(**aggregations)
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    std_columns = [f"{metric}_std" for metric in METRICS]
    summary[std_columns] = summary[std_columns].fillna(0.0)
    return summary


def aggregate_strategy_runs(raw_results):
    """Average stocks within a seed before measuring variation across seeds."""
    per_run_aggregations = {
        metric: (metric, "mean")
        for metric in METRICS
    }
    per_run_aggregations["num_stocks"] = ("stock", "nunique")
    per_run_summary = (
        raw_results.groupby(
            ["strategy", "seed", "model"],
            as_index=False,
        )
        .agg(**per_run_aggregations)
        .sort_values(["strategy", "seed", "model"])
        .reset_index(drop=True)
    )
    overall_summary = aggregate_metrics(
        per_run_summary,
        ["strategy", "model"],
    )
    return per_run_summary, overall_summary


def validate_run_counts(per_stock_summary, stocks, seeds):
    expected_models = set(DEFAULT_MODELS)
    expected_runs = len(seeds)
    problems = []

    for strategy in MASK_STRATEGIES:
        for stock in stocks:
            rows = per_stock_summary[
                (per_stock_summary["strategy"] == strategy)
                & (per_stock_summary["stock"] == stock)
            ]
            found_models = set(rows["model"])
            missing_models = sorted(expected_models - found_models)
            if missing_models:
                problems.append(
                    f"{strategy}/{stock}: missing models {','.join(missing_models)}"
                )
            for row in rows.itertuples():
                if row.num_runs != expected_runs:
                    problems.append(
                        f"{strategy}/{stock}/{row.model}: "
                        f"found {row.num_runs} runs, expected {expected_runs}"
                    )

    if problems:
        raise RuntimeError(
            "Results are incomplete, so summary statistics were not written:\n- "
            + "\n- ".join(problems)
        )


def write_summaries(args, stocks, seeds):
    frames = []
    for strategy in MASK_STRATEGIES:
        strategy_dir = Path(args.results_dir) / strategy
        frames.append(
            collect_raw_results(strategy_dir, strategy, stocks, seeds)
        )

    nonempty_frames = [frame for frame in frames if not frame.empty]
    if not nonempty_frames:
        raise RuntimeError(f"No experiment results found under {args.results_dir}")

    raw_results = pd.concat(nonempty_frames, ignore_index=True)
    per_stock_summary = aggregate_metrics(
        raw_results,
        ["strategy", "stock", "model"],
    )
    validate_run_counts(per_stock_summary, stocks, seeds)
    per_run_summary, overall_summary = aggregate_strategy_runs(raw_results)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / "all_runs.csv"
    per_stock_path = results_dir / "per_stock_mean_std.csv"
    per_run_path = results_dir / "per_run_stock_mean.csv"
    overall_path = results_dir / "overall_mean_std.csv"
    raw_results.sort_values(
        ["strategy", "stock", "seed", "model"]
    ).to_csv(raw_path, index=False)
    per_stock_summary.to_csv(per_stock_path, index=False)
    per_run_summary.to_csv(per_run_path, index=False)
    overall_summary.to_csv(overall_path, index=False)

    print(f"All run metrics saved to: {raw_path}", flush=True)
    print(f"Per-stock mean/std saved to: {per_stock_path}", flush=True)
    print(f"Per-run stock means saved to: {per_run_path}", flush=True)
    print(f"Overall mean/std saved to: {overall_path}", flush=True)
    return raw_path, per_stock_path, per_run_path, overall_path


def main(argv=None):
    args = parse_args(argv)
    stocks = selected_stocks(args)
    seeds = resolve_seeds(args)

    print(f"Stocks ({len(stocks)}): {', '.join(stocks)}", flush=True)
    print(
        f"Seeds ({len(seeds)} runs/strategy): {', '.join(map(str, seeds))}",
        flush=True,
    )

    if not args.aggregate_only:
        for strategy in MASK_STRATEGIES:
            run_command(build_strategy_command(args, strategy, stocks, seeds))

    if args.dry_run:
        print("Dry run complete; summary aggregation was skipped.", flush=True)
        return

    write_summaries(args, stocks, seeds)


if __name__ == "__main__":
    main()
