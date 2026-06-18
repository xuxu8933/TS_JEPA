import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests


DEFAULT_API_KEY = "d4j0vapr01queual0k7gd4j0vapr01queual0k80"
ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
SENTIMENT_COLS = [
    "sentiment_mean",
    "sentiment_sum",
    "sentiment_max",
    "sentiment_min",
    "sentiment_std",
    "news_count",
]


def get_alpaca_key_id():
    return (
        os.getenv("APCA_API_KEY_ID")
        or os.getenv("APCA_API_KEY")
        or os.getenv("ALPACA_API_KEY_ID")
        or os.getenv("ALPACA_API_KEY")
    )


def get_alpaca_secret_key():
    return (
        os.getenv("APCA_API_SECRET_KEY")
        or os.getenv("APCA_API_SECRET")
        or os.getenv("ALPACA_API_SECRET_KEY")
        or os.getenv("ALPACA_API_SECRET")
    )


def infer_stock_date_window(price_path, lookback_days):
    price_df = pd.read_csv(price_path, parse_dates=["Date"])
    end_date = price_df["Date"].max().normalize()
    start_date = end_date - pd.Timedelta(days=lookback_days)
    return start_date.date().isoformat(), end_date.date().isoformat()


def _iter_date_windows(start_date, end_date, chunk_days, chunk_mode):
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    window_start = start_ts
    while window_start <= end_ts:
        if chunk_mode == "months":
            month_end = window_start + pd.offsets.MonthEnd(0)
            window_end = min(month_end, end_ts)
        else:
            window_end = min(window_start + pd.Timedelta(days=chunk_days - 1), end_ts)

        yield window_start, window_end
        window_start = window_end + pd.Timedelta(days=1)


def collect_finnhub_news(
    api_key,
    ticker,
    start_date,
    end_date,
    chunk_days,
    chunk_mode,
    request_delay,
):
    import finnhub

    client = finnhub.Client(api_key=api_key)
    news = []

    for window_start, window_end in _iter_date_windows(
        start_date,
        end_date,
        chunk_days=chunk_days,
        chunk_mode=chunk_mode,
    ):
        chunk = client.company_news(
            ticker,
            _from=window_start.date().isoformat(),
            to=window_end.date().isoformat(),
        )
        news.extend(chunk)
        print(
            "Fetched "
            f"{len(chunk)} articles for "
            f"{window_start.date().isoformat()}..{window_end.date().isoformat()}"
        )
        if request_delay > 0:
            time.sleep(request_delay)

    news_df = pd.DataFrame(news)

    if news_df.empty:
        raise RuntimeError(
            f"No Finnhub news returned for {ticker} from {start_date} to {end_date}."
        )

    if "id" in news_df.columns:
        news_df = news_df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    else:
        news_df = news_df.drop_duplicates().reset_index(drop=True)

    news_df["date"] = pd.to_datetime(news_df["datetime"], unit="s").dt.date
    news_df["headline"] = news_df["headline"].fillna("")
    news_df["summary"] = news_df["summary"].fillna("")
    news_df["text"] = news_df["headline"] + ". " + news_df["summary"]

    return news_df


def _normalize_alpaca_article(article, ticker):
    created_at = pd.to_datetime(article.get("created_at"), errors="coerce", utc=True)
    if pd.isna(created_at):
        return None

    images = article.get("images") or []
    image_url = ""
    if images and isinstance(images, list):
        image_url = images[0].get("url", "") if isinstance(images[0], dict) else ""

    symbols = article.get("symbols") or []
    headline = article.get("headline") or ""
    summary = article.get("summary") or ""
    content = article.get("content") or ""
    text_body = content or summary

    return {
        "category": "alpaca",
        "datetime": int(created_at.timestamp()),
        "headline": headline,
        "id": article.get("id", ""),
        "image": image_url,
        "related": ",".join(symbols) if symbols else ticker.upper(),
        "source": article.get("source", ""),
        "author": article.get("author", ""),
        "summary": summary,
        "url": article.get("url", ""),
        "date": created_at.date(),
        "text": f"{headline}. {text_body}".strip(),
    }


def collect_alpaca_news(
    api_key_id,
    secret_key,
    ticker,
    start_date,
    end_date,
    chunk_days,
    limit,
    include_content,
    timeout,
    request_retries,
    retry_sleep,
    request_delay,
):
    if not api_key_id or not secret_key:
        raise ValueError(
            "Alpaca news requires credentials. Set APCA_API_KEY_ID/APCA_API_SECRET_KEY "
            "or ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY, or pass "
            "--alpaca-key-id and --alpaca-secret-key."
        )

    session = requests.Session()
    session.headers.update(
        {
            "APCA-API-KEY-ID": api_key_id,
            "APCA-API-SECRET-KEY": secret_key,
            "User-Agent": "TS-JEPA sentiment research",
        }
    )

    rows = []
    for window_start, window_end in _iter_date_windows(
        start_date,
        end_date,
        chunk_days=chunk_days,
        chunk_mode="days",
    ):
        next_page_token = None
        window_count = 0

        while True:
            params = {
                "symbols": ticker.upper(),
                "start": pd.Timestamp(window_start).isoformat() + "Z",
                "end": (
                    pd.Timestamp(window_end)
                    .replace(hour=23, minute=59, second=59)
                    .isoformat()
                    + "Z"
                ),
                "limit": limit,
                "sort": "asc",
                "include_content": str(include_content).lower(),
            }
            if next_page_token:
                params["page_token"] = next_page_token

            response = None
            for attempt in range(request_retries + 1):
                response = session.get(
                    ALPACA_NEWS_URL,
                    params=params,
                    timeout=timeout,
                )
                if response.status_code not in (429, 500, 502, 503, 504):
                    response.raise_for_status()
                    break

                if attempt == request_retries:
                    response.raise_for_status()

                retry_after = response.headers.get("Retry-After")
                sleep_seconds = float(retry_after) if retry_after else retry_sleep * (attempt + 1)
                print(
                    "Alpaca request was rate limited or temporarily unavailable; "
                    f"sleeping {sleep_seconds:.1f}s before retry {attempt + 1}/{request_retries}"
                )
                time.sleep(sleep_seconds)

            payload = response.json()
            articles = payload.get("news", [])
            for article in articles:
                normalized = _normalize_alpaca_article(article, ticker)
                if normalized is not None:
                    rows.append(normalized)
            window_count += len(articles)

            next_page_token = payload.get("next_page_token")
            if not next_page_token:
                break

        print(
            "Fetched "
            f"{window_count} Alpaca articles for "
            f"{window_start.date().isoformat()}..{window_end.date().isoformat()}"
        )
        if request_delay > 0:
            time.sleep(request_delay)

    news_df = pd.DataFrame(rows)
    if news_df.empty:
        raise RuntimeError(
            f"No Alpaca news returned for {ticker} from {start_date} to {end_date}."
        )

    news_df = news_df.drop_duplicates(subset=["id", "url"], keep="last").reset_index(drop=True)
    news_df["headline"] = news_df["headline"].fillna("")
    news_df["summary"] = news_df["summary"].fillna("")
    news_df["text"] = news_df["text"].fillna(news_df["headline"])

    return news_df


def _format_gdelt_datetime(timestamp, end_of_day=False):
    timestamp = pd.Timestamp(timestamp)
    if end_of_day:
        timestamp = timestamp.replace(hour=23, minute=59, second=59)
    return timestamp.strftime("%Y%m%d%H%M%S")


def _parse_gdelt_seen_date(value):
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        parsed = pd.to_datetime(str(value)[:14], format="%Y%m%d%H%M%S", errors="coerce", utc=True)
    return parsed


def _normalize_gdelt_articles(articles, ticker):
    rows = []
    for article in articles:
        seen_at = _parse_gdelt_seen_date(article.get("seendate"))
        if pd.isna(seen_at):
            continue

        headline = article.get("title") or ""
        url = article.get("url") or ""
        domain = article.get("domain") or ""
        source_country = article.get("sourcecountry") or article.get("sourceCountry") or ""
        language = article.get("language") or ""

        rows.append(
            {
                "category": "gdelt",
                "datetime": int(seen_at.timestamp()),
                "headline": headline,
                "id": url,
                "image": article.get("socialimage") or "",
                "related": ticker,
                "source": domain,
                "source_country": source_country,
                "language": language,
                "summary": "",
                "url": url,
                "date": seen_at.date(),
                "text": headline,
            }
        )

    return rows


def collect_gdelt_news(
    ticker,
    start_date,
    end_date,
    chunk_days,
    query,
    max_records,
    timeout,
    request_retries,
    retry_sleep,
    request_delay,
):
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    query = query or f'("{ticker}" OR "NVIDIA") sourcelang:english'
    rows = []
    session = requests.Session()

    window_start = start_ts
    while window_start <= end_ts:
        window_end = min(window_start + pd.Timedelta(days=chunk_days - 1), end_ts)
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "startdatetime": _format_gdelt_datetime(window_start),
            "enddatetime": _format_gdelt_datetime(window_end, end_of_day=True),
            "maxrecords": max_records,
            "sort": "HybridRel",
        }
        response = None
        for attempt in range(request_retries + 1):
            response = session.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params=params,
                timeout=timeout,
                headers={"User-Agent": "TS-JEPA sentiment research"},
            )
            if response.status_code != 429:
                response.raise_for_status()
                break

            if attempt == request_retries:
                response.raise_for_status()

            retry_after = response.headers.get("Retry-After")
            sleep_seconds = float(retry_after) if retry_after else retry_sleep * (attempt + 1)
            print(
                "GDELT rate limited this request; "
                f"sleeping {sleep_seconds:.1f}s before retry {attempt + 1}/{request_retries}"
            )
            time.sleep(sleep_seconds)

        payload = response.json()
        articles = payload.get("articles", [])
        rows.extend(_normalize_gdelt_articles(articles, ticker))
        print(
            "Fetched "
            f"{len(articles)} GDELT articles for "
            f"{window_start.date().isoformat()}..{window_end.date().isoformat()}"
        )
        if request_delay > 0:
            time.sleep(request_delay)
        window_start = window_end + pd.Timedelta(days=1)

    news_df = pd.DataFrame(rows)

    if news_df.empty:
        raise RuntimeError(
            f"No GDELT news returned for query={query!r} from {start_date} to {end_date}."
        )

    news_df = news_df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    news_df["headline"] = news_df["headline"].fillna("")
    news_df["summary"] = news_df["summary"].fillna("")
    news_df["text"] = news_df["text"].fillna(news_df["headline"])

    return news_df


def _first_present(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def collect_fnspid_news(
    ticker,
    start_date,
    end_date,
    paths,
    chunksize,
    text_field,
):
    start_ts = pd.Timestamp(start_date).tz_localize(None)
    end_ts = pd.Timestamp(end_date).tz_localize(None) + pd.Timedelta(days=1)
    matches = []
    total_matches = 0
    usecols = [
        "Date",
        "Article_title",
        "Stock_symbol",
        "Url",
        "Publisher",
        "Author",
        "Article",
        "Lsa_summary",
        "Luhn_summary",
        "Textrank_summary",
        "Lexrank_summary",
    ]

    for raw_path in str(paths).split(","):
        path = Path(raw_path.strip())
        if not path.exists():
            raise FileNotFoundError(f"FNSPID CSV not found: {path}")

        print(f"Scanning FNSPID CSV: {path}")
        for chunk_idx, chunk in enumerate(
            pd.read_csv(
                path,
                chunksize=chunksize,
                low_memory=False,
                usecols=lambda col: col in usecols,
            ),
            start=1,
        ):
            chunk["Stock_symbol"] = chunk["Stock_symbol"].astype(str).str.upper()
            chunk = chunk[chunk["Stock_symbol"] == ticker.upper()].copy()
            if chunk.empty:
                if chunk_idx % 25 == 0:
                    print(
                        f"Scanned {chunk_idx} chunks in {path.name}; "
                        f"matched {total_matches} rows"
                    )
                continue

            chunk["parsed_date"] = pd.to_datetime(
                chunk["Date"],
                errors="coerce",
                utc=True,
            ).dt.tz_localize(None)
            chunk = chunk[
                (chunk["parsed_date"] >= start_ts)
                & (chunk["parsed_date"] < end_ts)
            ].copy()
            if chunk.empty:
                if chunk_idx % 25 == 0:
                    print(
                        f"Scanned {chunk_idx} chunks in {path.name}; "
                        f"matched {total_matches} rows"
                    )
                continue

            total_matches += len(chunk)
            matches.append(chunk)
            print(
                f"Scanned chunk {chunk_idx} in {path.name}; "
                f"matched {total_matches} rows so far"
            )

    if not matches:
        raise RuntimeError(
            f"No FNSPID rows found for {ticker} from {start_date} to {end_date} in {paths}."
        )

    news_df = pd.concat(matches, ignore_index=True, sort=False)
    news_df = news_df.drop_duplicates(subset=["Url"], keep="last").reset_index(drop=True)

    summary_col = _first_present(
        news_df.columns,
        ["Lsa_summary", "Textrank_summary", "Lexrank_summary", "Luhn_summary"],
    )
    text_col = text_field
    if text_col == "auto":
        text_col = _first_present(
            news_df.columns,
            ["Article", "Lsa_summary", "Textrank_summary", "Lexrank_summary", "Article_title"],
        )
    if text_col not in news_df.columns:
        raise ValueError(
            f"FNSPID text field {text_col!r} not found. Available columns: {list(news_df.columns)}"
        )

    headline = news_df["Article_title"].fillna("")
    body = news_df[text_col].fillna("")
    if text_col == "Article_title":
        text = headline
    else:
        text = headline + ". " + body

    unix_seconds = (
        news_df["parsed_date"] - pd.Timestamp("1970-01-01")
    ).dt.total_seconds().astype("int64")

    normalized_df = pd.DataFrame(
        {
            "category": "fnspid",
            "datetime": unix_seconds,
            "headline": headline,
            "id": news_df["Url"].fillna(""),
            "image": "",
            "related": ticker.upper(),
            "source": news_df["Publisher"].fillna(""),
            "author": news_df["Author"].fillna(""),
            "summary": news_df[summary_col].fillna("") if summary_col else "",
            "url": news_df["Url"].fillna(""),
            "date": news_df["parsed_date"].dt.date,
            "text": text.fillna(""),
        }
    )
    normalized_df = normalized_df[
        normalized_df["text"].astype(str).str.strip().astype(bool)
    ].copy()

    if normalized_df.empty:
        raise RuntimeError(
            f"FNSPID rows were found for {ticker}, but none had usable text in {text_col!r}."
        )

    print(f"Loaded {len(normalized_df)} deduplicated FNSPID rows from {paths}")
    return normalized_df


def score_news_sentiment(news_df, model_name, batch_size):
    from transformers import pipeline

    sentiment_pipe = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        top_k=None,
    )

    texts = [
        text[:512] if isinstance(text, str) and text.strip() else ""
        for text in news_df["text"].tolist()
    ]
    outputs = sentiment_pipe(
        texts,
        batch_size=batch_size,
        truncation=True,
        max_length=512,
    )

    scores = []
    for result in outputs:
        probs = {item["label"].lower(): item["score"] for item in result}
        scores.append(probs.get("positive", 0.0) - probs.get("negative", 0.0))

    scored_df = news_df.copy()
    scored_df["sentiment_score"] = scores
    return scored_df


def aggregate_daily_sentiment(news_df):
    news_df = news_df.copy()
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.date
    news_df["sentiment_score"] = pd.to_numeric(
        news_df["sentiment_score"],
        errors="coerce",
    ).fillna(0.0)

    daily_sentiment = news_df.groupby("date").agg(
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_sum=("sentiment_score", "sum"),
        sentiment_max=("sentiment_score", "max"),
        sentiment_min=("sentiment_score", "min"),
        sentiment_std=("sentiment_score", "std"),
        news_count=("sentiment_score", "count"),
    ).reset_index()

    daily_sentiment["sentiment_std"] = daily_sentiment["sentiment_std"].fillna(0.0)
    return daily_sentiment


def append_existing_news(news_path, news_df):
    if not news_path.exists():
        return news_df

    existing_df = pd.read_csv(news_path, low_memory=False)
    combined_df = pd.concat([existing_df, news_df], ignore_index=True, sort=False)

    dedupe_cols = [col for col in ["id", "url"] if col in combined_df.columns]
    if dedupe_cols:
        combined_df = combined_df.drop_duplicates(subset=dedupe_cols, keep="last")
    else:
        combined_df = combined_df.drop_duplicates(keep="last")

    if "date" in combined_df.columns:
        combined_df["date"] = pd.to_datetime(combined_df["date"]).dt.date
        combined_df = combined_df.sort_values(["date"]).reset_index(drop=True)

    return combined_df


def append_existing_daily(daily_path, daily_sentiment):
    if not daily_path.exists():
        return daily_sentiment

    existing_df = pd.read_csv(daily_path, parse_dates=["date"], low_memory=False)
    new_df = daily_sentiment.copy()
    new_df["date"] = pd.to_datetime(new_df["date"])

    combined_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
    combined_df = combined_df.drop_duplicates(subset=["date"], keep="last")
    combined_df = combined_df.sort_values("date").reset_index(drop=True)
    combined_df["date"] = combined_df["date"].dt.date
    return combined_df


def merge_sentiment_into_price(price_path, daily_sentiment):
    price_df = pd.read_csv(price_path, parse_dates=["Date"])
    sentiment_df = daily_sentiment.copy()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])

    price_df = price_df.drop(
        columns=[col for col in SENTIMENT_COLS if col in price_df.columns]
    )
    combined_df = price_df.merge(
        sentiment_df[["date", *SENTIMENT_COLS]],
        how="left",
        left_on=price_df["Date"].dt.normalize(),
        right_on=sentiment_df["date"].dt.normalize(),
    )
    combined_df = combined_df.drop(columns=["key_0", "date"])
    combined_df[SENTIMENT_COLS] = combined_df[SENTIMENT_COLS].fillna(0.0)
    combined_df["Date"] = combined_df["Date"].dt.strftime("%Y-%m-%d")
    combined_df.to_csv(price_path, index=False, lineterminator="\n")

    return int((combined_df[SENTIMENT_COLS].abs().sum(axis=1) != 0).sum())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect news, score it with FinBERT, and merge daily sentiment into the stock CSV."
    )
    parser.add_argument("--source", choices=["alpaca", "gdelt", "finnhub", "fnspid"], default="alpaca")
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--price-path", default="data/NVDA/NVDA.csv")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--chunk-days", type=int, default=1)
    parser.add_argument(
        "--finnhub-chunk",
        choices=["days", "months"],
        default="months",
        help="Finnhub request window size. Use months for long historical backfills.",
    )
    parser.add_argument("--alpaca-key-id", default=get_alpaca_key_id())
    parser.add_argument("--alpaca-secret-key", default=get_alpaca_secret_key())
    parser.add_argument("--alpaca-limit", type=int, default=50)
    parser.add_argument(
        "--alpaca-include-content",
        action="store_true",
        default=False,
        help="Ask Alpaca to include full article content when available.",
    )
    parser.add_argument("--gdelt-query", default=None)
    parser.add_argument("--gdelt-max-records", type=int, default=250)
    parser.add_argument(
        "--fnspid-path",
        default="data/All_external.csv,data/nasdaq_exteral_data.csv",
        help="Comma-separated local FNSPID CSV path(s).",
    )
    parser.add_argument("--fnspid-chunksize", type=int, default=100000)
    parser.add_argument(
        "--fnspid-text-field",
        default="auto",
        help="FNSPID text column to score: auto, Article, Lsa_summary, Textrank_summary, Lexrank_summary, Luhn_summary, or Article_title.",
    )
    parser.add_argument("--request-timeout", type=int, default=30)
    parser.add_argument("--request-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--model-name", default="ProsusAI/finbert")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--api-key", default=os.getenv("FINNHUB_API_KEY", DEFAULT_API_KEY))
    parser.add_argument(
        "--write-mode",
        choices=["append", "overwrite"],
        default="append",
        help="Append to existing sentiment files by default. Use overwrite to replace them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    price_path = Path(args.price_path)

    start_date = args.start_date
    end_date = args.end_date
    if start_date is None or end_date is None:
        inferred_start, inferred_end = infer_stock_date_window(
            price_path,
            lookback_days=args.lookback_days,
        )
        start_date = start_date or inferred_start
        end_date = end_date or inferred_end

    news_path = Path(f"{args.ticker}_news_with_sentiment.csv")
    daily_path = Path(f"{args.ticker}_daily_sentiment.csv")

    print(f"Collecting {args.ticker} news from {start_date} to {end_date} via {args.source}")
    if args.source == "alpaca":
        news_df = collect_alpaca_news(
            api_key_id=args.alpaca_key_id,
            secret_key=args.alpaca_secret_key,
            ticker=args.ticker,
            start_date=start_date,
            end_date=end_date,
            chunk_days=args.chunk_days,
            limit=args.alpaca_limit,
            include_content=args.alpaca_include_content,
            timeout=args.request_timeout,
            request_retries=args.request_retries,
            retry_sleep=args.retry_sleep,
            request_delay=args.request_delay,
        )
    elif args.source == "finnhub":
        news_df = collect_finnhub_news(
            args.api_key,
            args.ticker,
            start_date,
            end_date,
            chunk_days=args.chunk_days,
            chunk_mode=args.finnhub_chunk,
            request_delay=args.request_delay,
        )
    elif args.source == "gdelt":
        news_df = collect_gdelt_news(
            args.ticker,
            start_date,
            end_date,
            chunk_days=args.chunk_days,
            query=args.gdelt_query,
            max_records=args.gdelt_max_records,
            timeout=args.request_timeout,
            request_retries=args.request_retries,
            retry_sleep=args.retry_sleep,
            request_delay=args.request_delay,
        )
    else:
        news_df = collect_fnspid_news(
            args.ticker,
            start_date,
            end_date,
            paths=args.fnspid_path,
            chunksize=args.fnspid_chunksize,
            text_field=args.fnspid_text_field,
        )
    print(f"Deduplicated news count: {len(news_df)}")

    news_df = score_news_sentiment(
        news_df,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
    daily_sentiment = aggregate_daily_sentiment(news_df)

    if args.write_mode == "append":
        news_df = append_existing_news(news_path, news_df)
        daily_sentiment = aggregate_daily_sentiment(news_df)
        daily_sentiment = append_existing_daily(daily_path, daily_sentiment)

    news_df.to_csv(news_path, index=False, lineterminator="\n")
    daily_sentiment.to_csv(daily_path, index=False, lineterminator="\n")
    nonzero_rows = merge_sentiment_into_price(price_path, daily_sentiment)

    print(f"Saved: {news_path}")
    print(f"Saved: {daily_path}")
    print(f"Updated: {price_path}")
    print(
        "Daily sentiment date range: "
        f"{daily_sentiment['date'].min()} to {daily_sentiment['date'].max()}"
    )
    print(f"Stock rows with non-zero sentiment: {nonzero_rows}")


if __name__ == "__main__":
    main()
