# Sentiment Feature Flow

```mermaid
flowchart TD
    A[Alpaca News API<br/>ticker or index proxy] --> B[collect_alpaca_news<br/>news_read.py]
    B --> C[Raw article table<br/>headline, summary, content, date]
    C --> D[score_news_sentiment<br/>FinBERT sentiment pipeline]
    D --> E[News with sentiment<br/>*_news_with_sentiment.csv]
    E --> F[aggregate_daily_sentiment]
    F --> G[Daily sentiment table<br/>date, sentiment_mean, sentiment_sum,<br/>sentiment_max, sentiment_min,<br/>sentiment_std, news_count]
    G --> H[*_daily_sentiment.csv]

    I[Price data CSV<br/>data/TICKER/TICKER.csv] --> J[merge_sentiment_into_price]
    H --> J
    J --> K[Price CSV with sentiment columns<br/>Close, Volume, MA10, MA50, sentiment_mean, ...]

    K --> L[config_pretrain.py / config_downstream.py<br/>feature_cols includes sentiment_mean]
    L --> M[data_loader_roll_volume.py]
    M --> N[load_price_series<br/>data_class_roll_volume.py]
    N --> O[Normalize selected features]
    O --> P[Patch sequence<br/>context shape = context_size x patch_size*feature_dim]

    P --> Q[TS-JEPA Encoder<br/>pretrain_wm.py]
    Q --> R[Pretrained representation]
    R --> S[Downstream decoder / GRU baseline<br/>eval_forecast_prequential_with_baselines_gru_volume.py]
    S --> T[Forecast target<br/>Close price patch]

    subgraph FeatureVector[Model input feature vector per day]
        U[Close]
        V[Volume]
        W[MA10]
        X[MA50]
        Y[sentiment_mean]
    end

    K -. selected by feature_cols .-> FeatureVector
    FeatureVector -. flattened into patches .-> P
```

Sentiment is included as an input feature, not as the target. The downstream target remains `Close`, while `sentiment_mean` is one of the context features used by the encoder and decoder.
