import argparse
import copy
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

from download_indices_and_news import TOP_NASDAQ100_STOCKS
from config.config_pretrain import config as pretrain_defaults
from config.file_options import parse_args_with_config
from config.preprocessing_presets import PREPROCESSING_PRESETS
from pretrain_dual_loss import (
    parse_args as parse_pretrain_args,
    validate_strategy_config,
)


MASK_STRATEGIES = ("random", "local_long", "future_block", "causal_multiblock")


def run_command(command, dry_run=False):
    print("=" * 80, flush=True)
    print("Running:", " ".join(command), flush=True)
    if dry_run:
        print("Dry run: command not executed", flush=True)
        return
    subprocess.run(command, check=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Download, pretrain, and evaluate TS-JEPA/GRU for the top "
            "NASDAQ-100 stocks."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "JSON or TOML experiment option file. Reads [common] and [runner]; "
            "explicit command-line options take precedence."
        ),
    )
    parser.add_argument(
        "--stocks",
        nargs="+",
        default=TOP_NASDAQ100_STOCKS,
        help="Stock tickers to run. Defaults to the top NASDAQ-100 holdings.",
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=5,
        help="Limit how many selected stocks to run. Use 0 to run all selected stocks.",
    )
    parser.add_argument("--download-start-date", default="2015-01-01")
    parser.add_argument("--download-end-date", default=date.today().isoformat())
    parser.add_argument(
        "--skip-download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--skip-news",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--skip-pretrain",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--mask-strategies",
        choices=MASK_STRATEGIES,
        nargs="+",
        default=["random"],
        help=(
            "Run one or more mask strategies (default: random). Results are "
            "isolated below RESULTS_DIR/STRATEGY for comparison analysis."
        ),
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
    parser.add_argument("--future-target-patches", type=int, default=4)
    parser.add_argument("--causal-num-blocks", type=int, default=2)
    parser.add_argument("--causal-block-patches", type=int, default=2)
    parser.add_argument("--causal-block-gap-patches", type=int, default=1)
    parser.add_argument(
        "--series-split-size",
        type=int,
        default=pretrain_defaults["series_split_size"],
        help="Rows per pretraining window before patching.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=pretrain_defaults["patch_size"],
        help="Rows per patch; must divide --series-split-size.",
    )
    parser.add_argument("--pretrain-stride", type=int, default=5)
    parser.add_argument(
        "--sampling-mode",
        choices=("sliding_window", "temporal_segments"),
        default="sliding_window",
        help="Use overlapping windows or non-overlapping temporal segments.",
    )
    parser.add_argument(
        "--normalization",
        choices=("window_return", "train_zscore", "train_robust_zscore", "none"),
        default="window_return",
    )
    parser.add_argument(
        "--preprocessing-preset",
        choices=tuple(PREPROCESSING_PRESETS),
        default=None,
        help="Apply one of the P0-P3 preprocessing ablations.",
    )
    parser.add_argument(
        "--feature-transform",
        choices=("raw", "return"),
        default="raw",
    )
    parser.add_argument("--market-data", default=None)
    parser.add_argument("--robust-zscore-clip", type=float, default=None)
    parser.add_argument("--feature-cols", nargs="+", default=None)
    parser.add_argument("--market-features", nargs="+", default=None)
    parser.add_argument("--sentiment-features", nargs="+", default=None)
    sentiment_group = parser.add_mutually_exclusive_group()
    sentiment_group.add_argument(
        "--use-sentiment",
        dest="use_sentiment",
        action="store_true",
        help="Include configured sentiment/news features (default).",
    )
    sentiment_group.add_argument(
        "--no-sentiment",
        dest="use_sentiment",
        action="store_false",
        help="Run the identical pipeline with market features only.",
    )
    parser.set_defaults(use_sentiment=pretrain_defaults["use_sentiment"])
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=42)
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run every stock for multiple seeds; overrides --seed.",
    )
    parser.add_argument(
        "--encoder-weights",
        choices=("ema", "online"),
        default="ema",
        help="Checkpoint encoder used in downstream evaluation.",
    )
    parser.add_argument(
        "--forecast-target",
        choices=(
            "value",
            "relative_return",
            "cumulative_log_return",
            "excess_log_return",
        ),
        default="value",
        help="Downstream value path or cutoff-relative return path.",
    )
    parser.add_argument("--eval-num-epochs", type=int, default=501)
    parser.add_argument("--pretrain-num-epochs", type=int, default=2001)
    parser.add_argument("--checkpoint-to-use", type=int, default=2000)
    parser.add_argument(
        "--use-best-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate the deterministic best-validation checkpoint instead of an epoch.",
    )
    parser.add_argument("--max-news-articles", type=int, default=None)
    parser.add_argument("--news-chunk-days", type=int, default=7)
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument("--write-mode", choices=["append", "overwrite"], default="append")
    parser.add_argument(
        "--results-dir",
        default="./results",
        help=(
            "Root output directory. Singular runs use TICKER/seed_N; plural "
            "strategy runs use STRATEGY/TICKER/seed_N."
        ),
    )
    parser.add_argument(
        "--skip-combined-plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do not generate the combined stock metrics CSV and PNG after evaluation.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the commands and write the summary without executing them.",
    )
    return parse_args_with_config(parser, argv, section="runner")


def resolve_mask_strategies(args):
    strategies = list(args.mask_strategies)
    if not strategies:
        raise ValueError("At least one mask strategy must be configured")
    if len(strategies) != len(set(strategies)):
        raise ValueError("Mask strategies must be unique")
    return strategies


def resolve_seeds(args):
    seeds = list(args.seeds or [args.seed])
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique values")
    return seeds


def validate_runner_mask_geometry(args, strategies):
    series_split_size = int(args.series_split_size)
    patch_size = int(args.patch_size)
    if series_split_size <= 0 or patch_size <= 0:
        raise ValueError("--series-split-size and --patch-size must be positive")
    if series_split_size % patch_size != 0:
        raise ValueError(
            "--series-split-size must be divisible by --patch-size: "
            f"series_split_size={series_split_size}, patch_size={patch_size}"
        )

    num_patches = series_split_size // patch_size
    if num_patches < 2:
        raise ValueError(
            "At least two patches are required: "
            f"series_split_size={series_split_size}, patch_size={patch_size}"
        )

    for strategy in strategies:
        if strategy == "random":
            continue
        strategy_config = {
            "mask_strategy": strategy,
            "mae_window_patches": args.mae_window_patches,
            "jepa_gap_patches": args.jepa_gap_patches,
            "jepa_target_patches": args.jepa_target_patches,
            "future_target_patches": args.future_target_patches,
            "causal_num_blocks": args.causal_num_blocks,
            "causal_block_patches": args.causal_block_patches,
            "causal_block_gap_patches": args.causal_block_gap_patches,
            "anchor_strategy": pretrain_defaults["anchor_strategy"],
            "fixed_anchor": pretrain_defaults["fixed_anchor"],
        }
        try:
            validate_strategy_config(strategy_config, num_patches)
        except ValueError as exc:
            advice = ""
            if strategy == "local_long":
                required_patches = (
                    int(args.jepa_gap_patches)
                    + int(args.jepa_target_patches)
                )
                required_rows = required_patches * patch_size
                advice = (
                    f" Increase --series-split-size to at least {required_rows} "
                    "rows or reduce the gap/target geometry."
                )
            raise ValueError(
                f"Invalid {strategy!r} mask geometry for {num_patches} patches "
                f"(--series-split-size={series_split_size}, "
                f"--patch-size={patch_size}): {exc}{advice}"
            ) from exc


def strategy_results_dir(args, strategy):
    return Path(args.results_dir) / strategy


def resolve_preprocessing_settings(args):
    preset_name = getattr(args, "preprocessing_preset", None)
    if preset_name is not None:
        settings = dict(PREPROCESSING_PRESETS[preset_name])
    else:
        settings = {
            "feature_transform": getattr(args, "feature_transform", "raw"),
            "normalization": getattr(args, "normalization", "window_return"),
            "forecast_target": getattr(args, "forecast_target", "value"),
            "market_data": getattr(args, "market_data", None),
        }
    settings["robust_zscore_clip"] = getattr(args, "robust_zscore_clip", None)
    settings["feature_cols"] = getattr(args, "feature_cols", None)
    settings["market_features"] = getattr(args, "market_features", None)
    settings["sentiment_features"] = getattr(args, "sentiment_features", None)
    settings["use_sentiment"] = bool(
        getattr(args, "use_sentiment", pretrain_defaults["use_sentiment"])
    )
    if (
        settings["use_sentiment"]
        and preset_name in ("P1", "P2", "P3")
        and settings["feature_cols"] is None
        and settings["sentiment_features"] is None
    ):
        # Return mode always adds the eight canonical price/volume features;
        # these names opt into the two optional sentiment features.
        settings["sentiment_features"] = ["sentiment_mean", "news_count"]
    return settings


def build_stock_commands(args, stock, seed=None, strategy=None, results_dir=None):
    commands = []
    seed = args.seed if seed is None else seed
    if strategy is None:
        strategies = resolve_mask_strategies(args)
        if len(strategies) != 1:
            raise ValueError(
                "strategy is required when more than one mask strategy is configured"
            )
        strategy = strategies[0]
    sampling_mode = getattr(args, "sampling_mode", "sliding_window")
    preprocessing = resolve_preprocessing_settings(args)
    preprocessing_args = [
        "--feature-transform",
        preprocessing["feature_transform"],
        "--normalization",
        preprocessing["normalization"],
        "--market-data",
        str(preprocessing["market_data"] or "none"),
    ]
    preprocessing_args.append(
        "--use-sentiment"
        if preprocessing["use_sentiment"]
        else "--no-sentiment"
    )
    if preprocessing["robust_zscore_clip"] is not None:
        preprocessing_args.extend(
            ["--robust-zscore-clip", str(preprocessing["robust_zscore_clip"])]
        )
    if preprocessing["feature_cols"] is not None:
        preprocessing_args.extend(
            ["--feature-cols", *map(str, preprocessing["feature_cols"])]
        )
    if preprocessing["market_features"] is not None:
        preprocessing_args.extend(
            ["--market-features", *map(str, preprocessing["market_features"])]
        )
    if preprocessing["sentiment_features"] is not None:
        preprocessing_args.extend(
            [
                "--sentiment-features",
                *map(str, preprocessing["sentiment_features"]),
            ]
        )
    strategy_args = ["--mask-strategy", strategy]
    if strategy == "local_long":
        strategy_args.extend(
            [
                "--mae-window-patches",
                str(args.mae_window_patches),
                "--jepa-gap-patches",
                str(args.jepa_gap_patches),
                "--jepa-target-patches",
                str(args.jepa_target_patches),
            ]
        )
    elif strategy == "future_block":
        strategy_args.extend(
            ["--future-target-patches", str(args.future_target_patches)]
        )
    elif strategy == "causal_multiblock":
        strategy_args.extend(
            [
                "--causal-num-blocks",
                str(args.causal_num_blocks),
                "--causal-block-patches",
                str(args.causal_block_patches),
                "--causal-block-gap-patches",
                str(args.causal_block_gap_patches),
            ]
        )

    pretrain_command = [
        sys.executable,
        "-u",
        "pretrain_dual_loss.py",
        "--no-run-eval",
        "--data",
        stock,
        *strategy_args,
        "--num_epochs",
        str(args.pretrain_num_epochs),
        "--lambda_jepa",
        str(args.lambda_jepa),
        "--lambda_mae",
        str(args.lambda_mae),
        "--jepa-loss",
        args.jepa_loss,
        "--mae-loss",
        args.mae_loss,
        "--series-split-size",
        str(getattr(args, "series_split_size", pretrain_defaults["series_split_size"])),
        "--patch-size",
        str(getattr(args, "patch_size", pretrain_defaults["patch_size"])),
        "--pretrain-stride",
        str(args.pretrain_stride),
        *preprocessing_args,
        "--sampling-mode",
        sampling_mode,
        "--seed",
        str(seed),
    ]
    if preprocessing["use_sentiment"]:
        pretrain_command.extend(
            [
                "--sentiment-path",
                str(Path("data") / stock / f"{stock}_daily_sentiment.csv"),
            ]
        )

    resolved_pretrain_config = parse_pretrain_args(
        copy.deepcopy(pretrain_defaults),
        argv=pretrain_command[3:],
    )
    if getattr(args, "use_best_checkpoint", False):
        checkpoint_path = resolved_pretrain_config["path_save"] + "_best.pt"
    else:
        checkpoint_path = (
            resolved_pretrain_config["path_save"]
            + "_epoch_"
            + str(args.checkpoint_to_use)
            + ".pt"
        )
    checkpoint_args = ["--pretrain-checkpoint-path", checkpoint_path]
    if args.skip_pretrain and not Path(checkpoint_path).exists():
        # Let eval_dual_loss resolve a legacy non-fingerprinted checkpoint.
        checkpoint_args = []

    if not args.skip_pretrain:
        commands.append(pretrain_command)

    stock_results_dir = Path(results_dir or args.results_dir)
    if getattr(args, "preprocessing_preset", None):
        stock_results_dir /= args.preprocessing_preset
    stock_results_dir = stock_results_dir / stock / f"seed_{seed}"

    commands.append(
        [
            sys.executable,
            "-u",
            "eval_dual_loss.py",
            "--data",
            stock,
            *strategy_args,
            "--checkpoint_to_use",
            str(args.checkpoint_to_use),
            *checkpoint_args,
            "--pretrain-encoder-weights",
            args.encoder_weights,
            "--forecast-target",
            preprocessing["forecast_target"],
            *preprocessing_args,
            "--num_epochs",
            str(args.eval_num_epochs),
            "--seed",
            str(seed),
            "--sampling-mode",
            sampling_mode,
            "--results-dir",
            str(stock_results_dir),
            "--lambda_jepa",
            str(args.lambda_jepa),
            "--lambda_mae",
            str(args.lambda_mae),
        ]
    )

    return commands


def build_combined_plot_command(
    args,
    stocks,
    results_dir=None,
    strategy=None,
):
    output_prefix = f"top_{len(stocks)}_nasdaq100"
    if strategy is not None:
        output_prefix += f"_{strategy}"
    seeds = args.seeds or [args.seed]
    results_dir = str(results_dir or args.results_dir)
    return [
        sys.executable,
        "-u",
        "plot_top_stock_metrics.py",
        "--results-dir",
        results_dir,
        "--output-dir",
        results_dir,
        "--output-prefix",
        output_prefix,
        "--figure-title",
        output_prefix,
        "--seeds",
        *[str(seed) for seed in seeds],
        "--stocks",
        *stocks,
    ]


def main(argv=None):
    args = parse_args(argv)
    preprocessing = resolve_preprocessing_settings(args)
    stocks = [stock.upper() for stock in args.stocks]
    seeds = resolve_seeds(args)
    strategies = resolve_mask_strategies(args)
    validate_runner_mask_geometry(args, strategies)

    if args.max_stocks < 0:
        raise ValueError("--max-stocks must be >= 0")
    if args.max_stocks > 0:
        stocks = stocks[:args.max_stocks]

    if not args.skip_download:
        download_cmd = [
            sys.executable,
            "download_indices_and_news.py",
            "--skip-indices",
            "--stocks",
            *stocks,
            "--download-start-date",
            args.download_start_date,
            "--download-end-date",
            args.download_end_date,
            "--news-chunk-days",
            str(args.news_chunk_days),
            "--request-delay",
            str(args.request_delay),
            "--write-mode",
            args.write_mode,
        ]
        if args.skip_news or not preprocessing["use_sentiment"]:
            download_cmd.append("--skip-news")
        if args.max_news_articles is not None:
            download_cmd.extend(["--max-news-articles", str(args.max_news_articles)])
        run_command(download_cmd, dry_run=args.dry_run)

    summary_path = Path(args.results_dir) / "top_nasdaq100_stock_runs.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "runner": "run_top_nasdaq100_stocks.py",
        "arguments": vars(args),
        "stocks": stocks,
        "seeds": seeds,
        "mask_strategies": strategies,
        "results_dir": str(args.results_dir),
        "strategy_results_dirs": {
            strategy: str(strategy_results_dir(args, strategy))
            for strategy in strategies
        },
        "use_sentiment": preprocessing["use_sentiment"],
        "market_features": preprocessing["market_features"],
        "sentiment_features": preprocessing["sentiment_features"],
        "feature_cols": preprocessing["feature_cols"],
        "feature_transform": preprocessing["feature_transform"],
        "normalization": preprocessing["normalization"],
        "forecast_target": preprocessing["forecast_target"],
        "sampling_mode": args.sampling_mode,
        "pretrain_stride": args.pretrain_stride,
        "pretrain_num_epochs": args.pretrain_num_epochs,
        "eval_num_epochs": args.eval_num_epochs,
        "checkpoint_to_use": args.checkpoint_to_use,
        "use_best_checkpoint": args.use_best_checkpoint,
        "lambda_jepa": args.lambda_jepa,
        "lambda_mae": args.lambda_mae,
        "jepa_loss": args.jepa_loss,
        "mae_loss": args.mae_loss,
    }
    manifest_path = Path(args.results_dir) / "experiment_manifest.json"
    with manifest_path.open("w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")

    with summary_path.open("w") as summary:
        summary.write("Top NASDAQ-100 stock workflow\n")
        summary.write(f"stocks={','.join(stocks)}\n")
        summary.write(f"download_start_date={args.download_start_date}\n")
        summary.write(f"download_end_date={args.download_end_date}\n")
        summary.write(f"results_dir={args.results_dir}\n")
        summary.write("pretrain_script=pretrain_dual_loss.py\n")
        summary.write("eval_script=eval_dual_loss.py\n")
        summary.write(f"mask_strategies={','.join(strategies)}\n")
        summary.write(f"seeds={','.join(str(seed) for seed in seeds)}\n")
        summary.write(
            f"preprocessing_preset={getattr(args, 'preprocessing_preset', None)}\n"
        )
        summary.write(f"feature_transform={preprocessing['feature_transform']}\n")
        summary.write(f"normalization={preprocessing['normalization']}\n")
        summary.write(f"market_data={preprocessing['market_data']}\n")
        summary.write(f"use_sentiment={preprocessing['use_sentiment']}\n")
        summary.write(
            "effective_feature_override="
            f"{preprocessing['feature_cols']}\n"
        )
        summary.write(f"market_features={preprocessing['market_features']}\n")
        summary.write(
            f"sentiment_features={preprocessing['sentiment_features']}\n"
        )
        summary.write(f"sampling_mode={args.sampling_mode}\n")
        summary.write(f"pretrain_stride={args.pretrain_stride}\n")
        summary.write(f"encoder_weights={args.encoder_weights}\n")
        summary.write(
            f"forecast_target={preprocessing['forecast_target']}\n"
        )
        summary.write(f"use_best_checkpoint={args.use_best_checkpoint}\n")
        summary.write(f"lambda_jepa={args.lambda_jepa}\n")
        summary.write(f"lambda_mae={args.lambda_mae}\n")
        summary.write(f"jepa_loss={args.jepa_loss}\n")
        summary.write(f"mae_loss={args.mae_loss}\n")
        summary.write(f"eval_num_epochs={args.eval_num_epochs}\n\n")

        for strategy in strategies:
            strategy_dir = strategy_results_dir(args, strategy)
            for stock in stocks:
                for seed in seeds:
                    for cmd in build_stock_commands(
                        args,
                        stock,
                        seed=seed,
                        strategy=strategy,
                        results_dir=strategy_dir,
                    ):
                        summary.write(
                            f"{strategy}/{stock}[seed={seed}]: {' '.join(cmd)}\n"
                        )
                        summary.flush()
                        run_command(cmd, dry_run=args.dry_run)

            if not args.skip_combined_plot:
                plot_strategy = strategy if len(strategies) > 1 else None
                combined_plot_cmd = build_combined_plot_command(
                    args,
                    stocks,
                    results_dir=strategy_dir,
                    strategy=plot_strategy,
                )
                summary.write(
                    f"combined_plot[{strategy}]: {' '.join(combined_plot_cmd)}\n"
                )
                summary.flush()
                run_command(combined_plot_cmd, dry_run=args.dry_run)

    print(f"Run summary saved to {summary_path}", flush=True)
    print(f"Experiment manifest saved to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
