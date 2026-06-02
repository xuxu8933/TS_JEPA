# d4j0vapr01queual0k7gd4j0vapr01queual0k80
import finnhub
import pandas as pd
from transformers import pipeline


# =========================
# 1. Config
# =========================

API_KEY = "d4j0vapr01queual0k7gd4j0vapr01queual0k80"   # 建议不要直接写真实 key，最好用环境变量
TICKER = "NVDA"
START_DATE = "2026-01-01"
END_DATE = "2026-06-01"


# =========================
# 2. Read news from Finnhub
# =========================

client = finnhub.Client(api_key=API_KEY)

news = client.company_news(
    TICKER,
    _from=START_DATE,
    to=END_DATE
)

news_df = pd.DataFrame(news)

print("Raw news count:", len(news_df))
print(news_df.head())


# =========================
# 3. Check empty data
# =========================

if news_df.empty:
    print("No news found. Try another ticker or a more recent date range.")
    exit()


# =========================
# 4. Prepare text
# =========================

news_df["date"] = pd.to_datetime(news_df["datetime"], unit="s").dt.date

news_df["headline"] = news_df["headline"].fillna("")
news_df["summary"] = news_df["summary"].fillna("")

news_df["text"] = news_df["headline"] + ". " + news_df["summary"]


# =========================
# 5. Load FinBERT
# =========================

sentiment_pipe = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
    top_k=None
)


def get_sentiment_score(text):
    """
    Return sentiment score:
    positive probability - negative probability

    Range:
    close to 1  -> positive
    close to 0  -> neutral
    close to -1 -> negative
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0

    # BERT maximum input length is limited, so truncate text
    result = sentiment_pipe(text[:512])[0]

    probs = {item["label"].lower(): item["score"] for item in result}

    positive = probs.get("positive", 0.0)
    negative = probs.get("negative", 0.0)
    neutral = probs.get("neutral", 0.0)

    score = positive - negative

    return score


# =========================
# 6. Add sentiment score
# =========================

news_df["sentiment_score"] = news_df["text"].apply(get_sentiment_score)

print(
    news_df[
        [
            "date",
            "headline",
            "sentiment_score"
        ]
    ].head()
)


# =========================
# 7. Daily aggregation
# =========================

daily_sentiment = news_df.groupby("date").agg(
    sentiment_mean=("sentiment_score", "mean"),
    sentiment_sum=("sentiment_score", "sum"),
    sentiment_max=("sentiment_score", "max"),
    sentiment_min=("sentiment_score", "min"),
    sentiment_std=("sentiment_score", "std"),
    news_count=("sentiment_score", "count")
).reset_index()

daily_sentiment["sentiment_std"] = daily_sentiment["sentiment_std"].fillna(0)

print("Daily sentiment:")
print(daily_sentiment.head())


# =========================
# 8. Save files
# =========================

news_df.to_csv(f"{TICKER}_news_with_sentiment.csv", index=False)
daily_sentiment.to_csv(f"{TICKER}_daily_sentiment.csv", index=False)

print(f"Saved: {TICKER}_news_with_sentiment.csv")
print(f"Saved: {TICKER}_daily_sentiment.csv")