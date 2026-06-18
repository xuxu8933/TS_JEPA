import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

from download_indices_and_news import TOP_NASDAQ100_STOCKS


def run_command(command):
    print("=" * 80, flush=True)
    print("Running:", " ".join(command), flush=True)
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
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-news", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--eval-num-epochs", type=int, default=501)
    parser.add_argument("--pretrain-num-epochs", type=int, default=2001)
    parser.add_argument("--checkpoint-to-use", type=int, default=2000)
    parser.add_argument("--max-news-articles", type=int, default=None)
    parser.add_argument("--news-chunk-days", type=int, default=7)
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument("--write-mode", choices=["append", "overwrite"], default="append")
    return parser.parse_args()


def main():
    args = parse_args()
    stocks = [stock.upper() for stock in args.stocks]

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
        run_command(download_cmd)

    summary_path = Path("results") / "top_nasdaq100_stock_runs.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w") as summary:
        summary.write("Top NASDAQ-100 stock workflow\n")
        summary.write(f"stocks={','.join(stocks)}\n")
        summary.write(f"start_date={args.start_date}\n")
        summary.write(f"end_date={args.end_date}\n")
        summary.write(f"eval_num_epochs={args.eval_num_epochs}\n\n")

        for stock in stocks:
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

            summary.write(f"{stock}: {' '.join(cmd)}\n")
            summary.flush()
            run_command(cmd)

    print(f"Run summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
