import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

from download_indices_and_news import TOP_NASDAQ100_STOCKS


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
        "--pretrain-script",
        choices=[
            "pretrain_wm.py",
            "pretrain_dual_loss.py",
            "pretrain_local_mae_long_jepa.py",
        ],
        default="pretrain_dual_loss.py",
        help="Pretraining entrypoint to run for each stock.",
    )
    parser.add_argument(
        "--eval-script",
        choices=[
            "eval_dual_loss.py",
            "eval_local_mae_long_jepa.py",
        ],
        default=None,
        help="Evaluation wrapper for decoupled pretrain/eval scripts. Defaults from --pretrain-script.",
    )
    parser.add_argument("--lambda-jepa", type=float, default=1.0)
    parser.add_argument("--lambda-mae", type=float, default=0.5)
    parser.add_argument("--eval-num-epochs", type=int, default=501)
    parser.add_argument("--pretrain-num-epochs", type=int, default=2001)
    parser.add_argument("--checkpoint-to-use", type=int, default=2000)
    parser.add_argument("--max-news-articles", type=int, default=None)
    parser.add_argument("--news-chunk-days", type=int, default=7)
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument("--write-mode", choices=["append", "overwrite"], default="append")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands and write the summary without executing them.",
    )
    return parser.parse_args()


def default_eval_script(pretrain_script):
    if pretrain_script == "pretrain_dual_loss.py":
        return "eval_dual_loss.py"
    if pretrain_script == "pretrain_local_mae_long_jepa.py":
        return "eval_local_mae_long_jepa.py"
    return None


def build_stock_commands(args, stock):
    if args.pretrain_script == "pretrain_wm.py":
        cmd = [
            sys.executable,
            "-u",
            "pretrain_wm.py",
            "--data",
            stock,
            "--run-eval",
            "--num_epochs",
            str(args.pretrain_num_epochs),
            "--eval-num-epochs",
            str(args.eval_num_epochs),
            "--eval-checkpoint-to-use",
            str(args.checkpoint_to_use),
        ]
        if args.skip_pretrain:
            cmd.append("--skip-pretrain")
        return [cmd]

    commands = []
    eval_script = args.eval_script or default_eval_script(args.pretrain_script)

    if not args.skip_pretrain:
        commands.append(
            [
                sys.executable,
                "-u",
                args.pretrain_script,
                "--data",
                stock,
                "--num_epochs",
                str(args.pretrain_num_epochs),
                "--lambda_jepa",
                str(args.lambda_jepa),
                "--lambda_mae",
                str(args.lambda_mae),
            ]
        )

    if eval_script is None:
        raise ValueError(
            f"No evaluation script is known for pretrain_script={args.pretrain_script!r}"
        )

    commands.append(
        [
            sys.executable,
            "-u",
            eval_script,
            "--data",
            stock,
            "--checkpoint_to_use",
            str(args.checkpoint_to_use),
            "--num_epochs",
            str(args.eval_num_epochs),
            "--lambda_jepa",
            str(args.lambda_jepa),
            "--lambda_mae",
            str(args.lambda_mae),
        ]
    )

    return commands


def main():
    args = parse_args()
    stocks = [stock.upper() for stock in args.stocks]

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

    summary_path = Path("results") / "top_nasdaq100_stock_runs.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w") as summary:
        summary.write("Top NASDAQ-100 stock workflow\n")
        summary.write(f"stocks={','.join(stocks)}\n")
        summary.write(f"start_date={args.start_date}\n")
        summary.write(f"end_date={args.end_date}\n")
        summary.write(f"pretrain_script={args.pretrain_script}\n")
        summary.write(f"eval_script={args.eval_script or default_eval_script(args.pretrain_script)}\n")
        summary.write(f"lambda_jepa={args.lambda_jepa}\n")
        summary.write(f"lambda_mae={args.lambda_mae}\n")
        summary.write(f"eval_num_epochs={args.eval_num_epochs}\n\n")

        for stock in stocks:
            for cmd in build_stock_commands(args, stock):
                summary.write(f"{stock}: {' '.join(cmd)}\n")
                summary.flush()
                run_command(cmd, dry_run=args.dry_run)

    print(f"Run summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
