import argparse
import copy
import subprocess
import sys
from datetime import date
from pathlib import Path

from download_indices_and_news import TOP_NASDAQ100_STOCKS
from config.config_pretrain import config as pretrain_defaults
from pretrain_dual_loss import parse_args as parse_pretrain_args


def run_command(command, dry_run=False):
    print("=" * 80, flush=True)
    print("Running:", " ".join(command), flush=True)
    if dry_run:
        print("Dry run: command not executed", flush=True)
        return
    subprocess.run(command, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download, pretrain, and evaluate TS-JEPA/GRU for the top "
            "NASDAQ-100 stocks."
        )
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
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-news", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument(
        "--mask-strategy",
        choices=("random", "local_long", "future_block", "causal_multiblock"),
        default="random",
        help="Unified pretraining mask strategy used for every selected stock.",
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
    parser.add_argument("--pretrain-stride", type=int, default=5)
    parser.add_argument(
        "--sampling-mode",
        choices=("sliding_window", "temporal_segments"),
        default="sliding_window",
        help="Use overlapping windows or non-overlapping temporal segments.",
    )
    parser.add_argument(
        "--normalization",
        choices=("window_return", "train_zscore", "none"),
        default="window_return",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
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
    parser.add_argument("--eval-num-epochs", type=int, default=501)
    parser.add_argument("--pretrain-num-epochs", type=int, default=2001)
    parser.add_argument("--checkpoint-to-use", type=int, default=2000)
    parser.add_argument(
        "--use-best-checkpoint",
        action="store_true",
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
            "Root output directory. Each run is written to "
            "TICKER/seed_N, and combined outputs are written to this root."
        ),
    )
    parser.add_argument(
        "--skip-combined-plot",
        action="store_true",
        help="Do not generate the combined stock metrics CSV and PNG after evaluation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands and write the summary without executing them.",
    )
    return parser.parse_args()


def build_stock_commands(args, stock, seed=None):
    commands = []
    seed = args.seed if seed is None else seed
    sampling_mode = getattr(args, "sampling_mode", "sliding_window")
    strategy_args = ["--mask-strategy", args.mask_strategy]
    if args.mask_strategy == "local_long":
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
    elif args.mask_strategy == "future_block":
        strategy_args.extend(
            ["--future-target-patches", str(args.future_target_patches)]
        )
    elif args.mask_strategy == "causal_multiblock":
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
        "--pretrain-stride",
        str(args.pretrain_stride),
        "--normalization",
        args.normalization,
        "--sampling-mode",
        sampling_mode,
        "--seed",
        str(seed),
    ]

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

    stock_results_dir = Path(args.results_dir) / stock / f"seed_{seed}"

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
            "--num_epochs",
            str(args.eval_num_epochs),
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


def build_combined_plot_command(args, stocks):
    output_prefix = f"top_{len(stocks)}_nasdaq100"
    seeds = args.seeds or [args.seed]
    return [
        sys.executable,
        "-u",
        "plot_top_stock_metrics.py",
        "--results-dir",
        str(args.results_dir),
        "--output-dir",
        str(args.results_dir),
        "--output-prefix",
        output_prefix,
        "--figure-title",
        output_prefix,
        "--seeds",
        *[str(seed) for seed in seeds],
        "--stocks",
        *stocks,
    ]


def main():
    args = parse_args()
    stocks = [stock.upper() for stock in args.stocks]
    seeds = args.seeds or [args.seed]

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
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            "--news-chunk-days",
            str(args.news_chunk_days),
            "--request-delay",
            str(args.request_delay),
            "--write-mode",
            args.write_mode,
        ]
        if args.skip_news:
            download_cmd.append("--skip-news")
        if args.max_news_articles is not None:
            download_cmd.extend(["--max-news-articles", str(args.max_news_articles)])
        run_command(download_cmd, dry_run=args.dry_run)

    summary_path = Path(args.results_dir) / "top_nasdaq100_stock_runs.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w") as summary:
        summary.write("Top NASDAQ-100 stock workflow\n")
        summary.write(f"stocks={','.join(stocks)}\n")
        summary.write(f"start_date={args.start_date}\n")
        summary.write(f"end_date={args.end_date}\n")
        summary.write(f"results_dir={args.results_dir}\n")
        summary.write("pretrain_script=pretrain_dual_loss.py\n")
        summary.write("eval_script=eval_dual_loss.py\n")
        summary.write(f"mask_strategy={args.mask_strategy}\n")
        summary.write(f"seeds={','.join(str(seed) for seed in seeds)}\n")
        summary.write(f"normalization={args.normalization}\n")
        summary.write(f"sampling_mode={args.sampling_mode}\n")
        summary.write(f"pretrain_stride={args.pretrain_stride}\n")
        summary.write(f"encoder_weights={args.encoder_weights}\n")
        summary.write(f"use_best_checkpoint={args.use_best_checkpoint}\n")
        summary.write(f"lambda_jepa={args.lambda_jepa}\n")
        summary.write(f"lambda_mae={args.lambda_mae}\n")
        summary.write(f"jepa_loss={args.jepa_loss}\n")
        summary.write(f"mae_loss={args.mae_loss}\n")
        summary.write(f"eval_num_epochs={args.eval_num_epochs}\n\n")

        for stock in stocks:
            for seed in seeds:
                for cmd in build_stock_commands(
                    args,
                    stock,
                    seed=seed,
                ):
                    summary.write(f"{stock}[seed={seed}]: {' '.join(cmd)}\n")
                    summary.flush()
                    run_command(cmd, dry_run=args.dry_run)

        if not args.skip_combined_plot:
            combined_plot_cmd = build_combined_plot_command(args, stocks)
            summary.write(f"combined_plot: {' '.join(combined_plot_cmd)}\n")
            summary.flush()
            run_command(combined_plot_cmd, dry_run=args.dry_run)

    print(f"Run summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
