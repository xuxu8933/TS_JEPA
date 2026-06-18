import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from news_read import (
    aggregate_daily_sentiment,
    append_existing_news,
    collect_alpaca_news,
    collect_gdelt_news,
    get_alpaca_key_id,
    get_alpaca_secret_key,
    merge_sentiment_into_price,
    score_news_sentiment,
)


INDEX_CONFIGS = {
    "NASDAQ100": {
        "price_symbol": "^NDX",
        "news_symbol": "QQQ",
        "gdelt_query": '("Nasdaq 100" OR "NASDAQ-100" OR QQQ) sourcelang:english',
    },
    "SP500": {
        "price_symbol": "^GSPC",
        "news_symbol": "SPY",
        "gdelt_query": '("S&P 500" OR "SP 500" OR SPY) sourcelang:english',
    },
}


def _flatten_yfinance_columns(df):
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    return df


def download_index_prices(index_name, cfg, start_date, end_date):
    save_dir = Path("data") / index_name
    save_dir.mkdir(parents=True, exist_ok=True)
    price_path = save_dir / f"{index_name}.csv"

    # yfinance treats end as exclusive, so include the requested end date.
    yf_end_date = (
        pd.Timestamp(end_date) + pd.Timedelta(days=1)
    ).date().isoformat()

    df = yf.download(
        cfg["price_symbol"],
        start=start_date,
        end=yf_end_date,
        auto_adjust=False,
        progress=True,
    )

    if df.empty:
        raise RuntimeError(
            f"No price data returned for {cfg['price_symbol']} "
            f"from {start_date} to {end_date}."
        )

    df = _flatten_yfinance_columns(df)
    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df = df.dropna()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df.to_csv(price_path, index=False, lineterminator="\n")

    print(
        f"Saved {index_name} prices from {df['Date'].min()} "
        f"to {df['Date'].max()} at {price_path} ({len(df)} rows)",
        flush=True,
    )

    return price_path


def collect_index_news(index_name, cfg, args):
    if args.news_source == "alpaca":
        return collect_alpaca_news(
            api_key_id=args.alpaca_key_id,
            secret_key=args.alpaca_secret_key,
            ticker=cfg["news_symbol"],
            start_date=args.start_date,
            end_date=args.end_date,
            chunk_days=args.news_chunk_days,
            limit=args.alpaca_limit,
            include_content=args.alpaca_include_content,
            timeout=args.request_timeout,
            request_retries=args.request_retries,
            retry_sleep=args.retry_sleep,
            request_delay=args.request_delay,
        )

    if args.news_source == "gdelt":
        return collect_gdelt_news(
            ticker=cfg["news_symbol"],
            start_date=args.start_date,
            end_date=args.end_date,
            chunk_days=args.news_chunk_days,
            query=cfg["gdelt_query"],
            max_records=args.gdelt_max_records,
            timeout=args.request_timeout,
            request_retries=args.request_retries,
            retry_sleep=args.retry_sleep,
            request_delay=args.request_delay,
        )

    raise ValueError(f"Unsupported news source: {args.news_source}")


def save_index_news_and_sentiment(index_name, price_path, news_df, args):
    save_dir = Path("data") / index_name
    news_path = save_dir / f"{index_name}_news_with_sentiment.csv"
    daily_path = save_dir / f"{index_name}_daily_sentiment.csv"

    if args.max_news_articles is not None:
        news_df = news_df.sort_values("datetime").tail(args.max_news_articles)
        print(
            f"Keeping the most recent {len(news_df)} articles for "
            f"{index_name} because --max-news-articles was set.",
            flush=True,
        )

    news_df = score_news_sentiment(
        news_df,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )

    if args.write_mode == "append":
        news_df = append_existing_news(news_path, news_df)

    daily_sentiment = aggregate_daily_sentiment(news_df)

    news_df.to_csv(news_path, index=False, lineterminator="\n")
    daily_sentiment.to_csv(daily_path, index=False, lineterminator="\n")
    nonzero_rows = merge_sentiment_into_price(price_path, daily_sentiment)

    print(f"Saved {index_name} scored news: {news_path}", flush=True)
    print(f"Saved {index_name} daily sentiment: {daily_path}", flush=True)
    print(f"Merged sentiment into {price_path}; non-zero rows={nonzero_rows}", flush=True)

    return news_path, daily_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download Nasdaq-100 and S&P 500 index prices plus related news "
            "sentiment datasets."
        )
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        choices=["NASDAQ100", "SP500"],
        default=["NASDAQ100", "SP500"],
    )
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument(
        "--news-source",
        choices=["alpaca", "gdelt"],
        default="alpaca",
    )
    parser.add_argument(
        "--skip-news",
        action="store_true",
        help="Only download prices; do not collect or score news.",
    )
    parser.add_argument(
        "--skip-prices",
        action="store_true",
        help="Use existing price CSVs; only collect and score news.",
    )
    parser.add_argument(
        "--max-news-articles",
        type=int,
        default=None,
        help="Optional cap for quick tests; keeps the most recent N articles.",
    )
    parser.add_argument("--news-chunk-days", type=int, default=7)
    parser.add_argument("--alpaca-key-id", default=get_alpaca_key_id())
    parser.add_argument("--alpaca-secret-key", default=get_alpaca_secret_key())
    parser.add_argument("--alpaca-limit", type=int, default=50)
    parser.add_argument("--alpaca-include-content", action="store_true", default=False)
    parser.add_argument("--gdelt-max-records", type=int, default=250)
    parser.add_argument("--request-timeout", type=int, default=30)
    parser.add_argument("--request-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument("--model-name", default="ProsusAI/finbert")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--write-mode",
        choices=["append", "overwrite"],
        default="overwrite",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for index_name in args.indices:
        cfg = INDEX_CONFIGS[index_name]
        print("=" * 80, flush=True)
        print(
            f"Preparing {index_name}: prices={cfg['price_symbol']}, "
            f"news proxy={cfg['news_symbol']}, "
            f"date_range={args.start_date}..{args.end_date}",
            flush=True,
        )

        price_path = Path("data") / index_name / f"{index_name}.csv"
        if args.skip_prices:
            if not price_path.exists():
                raise FileNotFoundError(
                    f"Cannot use --skip-prices because {price_path} does not exist."
                )
            print(f"Using existing price file: {price_path}", flush=True)
        else:
            price_path = download_index_prices(
                index_name=index_name,
                cfg=cfg,
                start_date=args.start_date,
                end_date=args.end_date,
            )

        if args.skip_news:
            continue

        news_df = collect_index_news(index_name, cfg, args)
        print(f"Collected {len(news_df)} deduplicated news rows for {index_name}", flush=True)
        save_index_news_and_sentiment(index_name, price_path, news_df, args)


if __name__ == "__main__":
    main()
