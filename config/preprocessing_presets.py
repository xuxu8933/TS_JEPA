"""Named preprocessing ablations with unchanged model/evaluation structure."""

PREPROCESSING_PRESETS = {
    "P0": {
        "feature_transform": "raw",
        "normalization": "train_zscore",
        "forecast_target": "cumulative_log_return",
        "market_data": None,
    },
    "P1": {
        "feature_transform": "return",
        "normalization": "train_zscore",
        "forecast_target": "cumulative_log_return",
        "market_data": None,
    },
    "P2": {
        "feature_transform": "return",
        "normalization": "train_robust_zscore",
        "forecast_target": "cumulative_log_return",
        "market_data": None,
    },
    "P3": {
        "feature_transform": "return",
        "normalization": "train_robust_zscore",
        "forecast_target": "excess_log_return",
        "market_data": "NASDAQ100",
    },
}

# Recommended financial representation. Defaults in the main configs remain
# backward compatible; runners can select this explicitly with P2.
RECOMMENDED_FINANCIAL_CONFIG = dict(PREPROCESSING_PRESETS["P2"])
