# Sentiment Mechanism Ablation Design

**Date:** 2026-08-25

**Branch:** `single-dim`
**Execution policy:** implementation, tests, configuration validation, and dry-run only. Full training is explicitly excluded.

## Goal

Prepare three controlled experiments that can determine, after the user runs the full experiments, whether daily FinBERT sentiment underperforms because of:

1. a mismatch between sentiment timing and the five-step forecasting objective;
2. conflation of no-news days with observed neutral-news days; or
3. a scale mismatch between window-relative market channels and raw sentiment.

The existing thesis architecture, chronological split, market preprocessing, training settings, published result packages, ten-stock universe, and seeds 42–51 remain unchanged unless a hypothesis explicitly requires a difference.

## Baseline Contract

The controls are:

- `config/experiments/top10_with_sentiment.json`, with published results at `thesis_results/top10_with_sentiment/5b8f3897bf23-02add88f32d5/`;
- `config/experiments/top10_without_sentiment.json`, with published results at `thesis_results/top10_without_sentiment/2fab810c1e1d-d0fb2944255b/`.

Their recorded effective configurations differ only in `use_sentiment` and `sentiment_features`. Missing daily sentiment is currently filled with zero, and configured sentiment columns are passthrough channels under `window_return`. These semantics must remain unchanged for both controls.

The published result directories are immutable inputs. Neither dry-run validation nor later analysis may write into them.

## Experiment Configurations

Create four configurations under `config/experiments/`:

- `top10_h1_without_sentiment.json`
- `top10_h1_with_sentiment.json`
- `top10_sentiment_has_news.json`
- `top10_sentiment_zscore.json`

Every configuration retains the ten baseline stocks, seeds 42–51, patch size 5, both enabled masking strategies (`random` and `local_long`), the existing GRU control, and all unchanged training/checkpoint settings.

### H1: Short-Horizon Objective

Introduce a downstream `forecast_horizon` setting that is independent of temporal `patch_size`:

- `patch_size` continues to control input patch construction and flattened encoder input size;
- `forecast_horizon` controls target length and downstream output width;
- when omitted, `forecast_horizon` resolves to `patch_size`, preserving all existing behavior and fingerprints where possible.

Both H1 configurations set `forecast_horizon` to 1. Their only semantic difference from each other is the existing sentiment condition. Their input dimensions remain 20 and 25 respectively.

### H2: No-News Indicator

Recognize `has_news` as a derived sentiment feature. During the existing same-trading-date merge:

```text
has_news = 1.0 if news_count > 0 else 0.0
```

Missing daily-news rows therefore produce `sentiment_mean = 0.0` and `has_news = 0.0`; observed neutral rows may produce approximately zero sentiment with `has_news = 1.0`. The merge remains an exact normalized-date join, so later news cannot populate an earlier trading date.

The H2 feature order is `Close`, `Volume`, `MA10`, `MA50`, `sentiment_mean`, `has_news`. Both sentiment channels bypass market `window_return`, and raw `sentiment_mean` is not z-scored.

### H3: Train-Only Sentiment Z-Score

Recognize `sentiment_mean_z` as a derived sentiment feature sourced from raw `sentiment_mean`. The source is merged causally before splitting, but transformation occurs only after chronological train/validation/test separation:

```text
train_mean = mean(train.sentiment_mean)
train_std  = population_std(train.sentiment_mean), clamped by epsilon

sentiment_mean_z(split) =
    (split.sentiment_mean - train_mean) / train_std
```

No validation or test observation contributes to `train_mean` or `train_std`. Statistics are fitted per stock and reused across that stock's pretraining and downstream loaders. They are stored in checkpoint configuration and each run's `preprocessing_config.json`.

Market channels retain `window_return`; only `sentiment_mean_z` receives train-only z-scoring and then passes through the window transform. The H3 feature order is `Close`, `Volume`, `MA10`, `MA50`, `sentiment_mean_z`.

## Configuration and CLI Flow

Extend the existing configuration flow rather than introduce a parallel runner:

```text
experiment JSON
  -> config.file_options flattening/validation
  -> run_top_nasdaq100_stocks.py
  -> pretrain_dual_loss.py / eval_dual_loss.py
  -> loader and downstream evaluator
```

New semantic options are:

- `runner.downstream.forecast_horizon` / `--forecast-horizon`;
- `runner.preprocessing.custom.features.sentiment.normalization`, with supported values `none` and `train_zscore`.

Existing configs that omit these options resolve to horizon = patch size and sentiment normalization = none.

## Dry-Run Contract

`run_top_nasdaq100_stocks.py --config CONFIG --dry-run` must validate and print a machine-readable summary containing:

- experiment/config name and current git branch;
- exact stocks and seeds plus counts;
- resolved forecast horizon;
- ordered feature names and count;
- patch size and flattened patch input dimension;
- sentiment handling and selective sentiment normalization;
- market normalization mode;
- result directory;
- `training_disabled: true`.

Dry-run may parse configs, validate paths, build commands, and inspect existing result coverage. It must return before downloads, subprocess calls, result-directory creation, pretraining, downstream optimization, plotting, or optimizer steps.

The expected flattened dimensions are 20, 25, 30, and 25 for H1-without, H1-with, H2, and H3 respectively.

## Configuration Isolation Validation

Provide reusable semantic comparison logic and tests that normalize defaults before diffing. The allowed differences are:

- H1 configs against their corresponding H=5 controls: forecast horizon only;
- H1 with versus H1 without: sentiment enablement/features only;
- H2 versus the raw-sentiment control: addition of `has_news` only;
- H3 versus the raw-sentiment control: replacement by `sentiment_mean_z` and selective sentiment normalization only.

The validation output is JSON so the baseline and intervention comparison is machine-readable. Any model, optimizer, split, masking, checkpoint, date, or coverage difference is an error.

## Post-Experiment Analysis

Add a standalone analysis command that does not launch training. It reads:

- the two immutable published control `all_runs_tidy.csv` files;
- completed raw result directories for the four new configurations;
- per-run preprocessing metadata and experiment manifests.

For each hypothesis, retain these model rows:

- `TS-JEPA` under `random` as `Shared-target JEPA--MAE`;
- `TS-JEPA` under `local_long` as `Local-MAE/Long-JEPA`;
- `GRU` under `random` as the supervised control.

Pair observations strictly on stock, seed, model, and metric. Error deltas use `intervention - control`, where negative is favorable. Direction-accuracy deltas use the same subtraction, where positive is favorable.

Primary inference first averages seeds within stock and then uses the ten paired stock means. It reports a two-sided paired Student t-test, Cohen's dz, mean difference, and 95% confidence interval. Holm adjustment is applied within each hypothesis across primary model/metric comparisons. The 100 stock-seed pairs are descriptive only.

The verdict rule is intentionally conservative:

- `supported`: MSE and MAE have favorable mean directions, a majority of stocks improve, and at least one primary error metric has Holm-adjusted p < 0.05;
- `not supported`: the mean primary error effects are unfavorable;
- `inconclusive`: favorable effects do not meet the consistency and corrected-inference conditions, or primary metrics disagree.

Direction accuracy is secondary and cannot by itself produce a supported verdict.

The analysis creates a new package at `thesis_results/sentiment_mechanism_ablation/<run_id>/` with:

- `data/mechanism_summary.csv`
- `data/per_stock_deltas.csv`
- `data/per_seed_deltas.csv`
- `data/h1_short_horizon_results.csv`
- `provenance/experiment_manifest.json`
- `sentiment_mechanism_report.md`

If any required result set is absent or incomplete, it prints `Experiment results not found; run the corresponding experiment first.` and exits cleanly without fabricating tables, verdicts, or a result package.

## Tests

Use test-driven development for all changes. Tests cover:

1. `has_news` for missing, neutral, positive, and negative news plus future-row isolation;
2. `sentiment_mean_z` fitting on training only, including adversarial validation/test values;
3. reuse and persistence of selective sentiment statistics;
4. unchanged outputs for both existing configs when new options are absent;
5. independent forecast horizon with input patch geometry unchanged;
6. exact semantic config isolation and expected feature dimensions;
7. dry-run returning before every execution/training boundary;
8. deterministic stock/seed pairing, statistics, Holm correction, and output schemas;
9. clean missing-results behavior.

Only unit tests, integration tests, CLI help, config validation, lightweight loader checks, and the four dry-runs are authorized during implementation.

## Research Impact

This work adds three distinct experiments without changing the original controls. H1 changes only the supervised forecast objective width, H2 changes only observability of news presence, and H3 changes only the sentiment channel's scale using train-only statistics. The implementation does not itself establish any scientific result. H1/H2/H3 conclusions remain unavailable until the user executes all full experiments and runs the prepared analysis.
