# TS-JEPA: Joint Embedding Goes Temporal

This repository implements TS-JEPA from ["Joint Embedding Goes Temporal"](https://openreview.net/forum?id=FIdbozebmy), presented at the NeurIPS Workshop on Time Series in the Age of Large Models.

![TS-JEPA architecture](assets/jepa_architecture.png)

The current pretraining entrypoint combines two losses:

- **JEPA loss:** predicts target-patch representations produced by an EMA target encoder.
- **MAE loss:** decodes predicted representations back into the original patch values.

It supports four masking strategies from one implementation:

| Strategy | CLI value | Behavior |
| --- | --- | --- |
| Random dual loss | `random` | JEPA and MAE operate on the same randomly masked patches. |
| Local MAE + long JEPA | `local_long` | MAE reconstructs a short local window while JEPA predicts a farther latent window from causal context. |
| Future block | `future_block` | Uses only patches before a cutoff to predict one contiguous future block. |
| Causal multi-block | `causal_multiblock` | Uses only patches before a cutoff to predict several separated future blocks. |

## Installation

The tested environment for this repository is the `ts-jepa` Conda environment:

```bash
conda run -n ts-jepa python --version
```

To install dependencies in another environment:

```bash
pip install -r requirements.txt
```

Install a PyTorch build appropriate for your CPU or CUDA version separately if necessary.

## Data layout

Time-series data is resolved from:

```text
data/<DATASET>/<DATASET>.csv
```

For example, `--data NVDA` reads:

```text
data/NVDA/NVDA.csv
```

A minimal price CSV is:

```csv
Date,Close,Volume
2020-01-02,5.99775,237536000
2020-01-03,5.90175,205384000
```

The loader:

- sorts observations chronologically;
- computes `MA10` and `MA50` automatically;
- drops rows without complete moving-average values;
- applies `log1p` to `Volume` when it is selected;
- builds sliding windows or non-overlapping temporal segments and divides each sample into patches;
- uses only the training split during pretraining.

Pretraining stride defaults to `patch_size`, rather than the complete window length. Override it with `--pretrain-stride N`. For example, a 60-row window with `patch_size=5` uses `stride=5` unless explicitly changed.

Select the sample construction mode with `--sampling-mode`:

| Mode | Behavior |
| --- | --- |
| `sliding_window` | Builds chronological fixed-length windows using `pretrain_stride`. This remains the default. |
| `temporal_segments` | Splits each chronological data split into contiguous, non-overlapping `series_split_size` segments. An incomplete tail is dropped so model tensor shapes remain fixed. |

For non-overlapping 60-observation segments:

```bash
conda run --no-capture-output -n ts-jepa python pretrain_dual_loss.py \
  --no-run-eval \
  --data NVDA \
  --series-split-size 60 \
  --sampling-mode temporal_segments
```

In temporal-segment mode, the effective pretraining stride is the complete segment length. The selected mode is stored in the checkpoint and reused by unified downstream evaluation.

Normalization is selected explicitly:

| Mode | Behavior |
| --- | --- |
| `window_return` | Divides every window by its first observable row and subtracts one. This is the default and does not fit global statistics. |
| `train_zscore` | Fits mean and standard deviation on the chronological train split only, stores them in the checkpoint, and reuses them for validation/test. |
| `train_robust_zscore` | Fits per-feature median and MAD on the chronological train split only; an optional `--robust-zscore-clip C` clips the transformed values to `[-C, C]`. |
| `none` | Uses the loaded feature values without additional normalization. |

Example:

```bash
--pretrain-stride 5 --normalization train_zscore
```

## Leakage-safe financial preprocessing

`--feature-transform raw` preserves the historical input format. The explicit
`--feature-transform return` mode constructs this deterministic feature order
before any split or sample window is made:

```text
log_return_1       = log(Close[t] / Close[t-1])
log_return_5       = log(Close[t] / Close[t-5])
close_ma10_gap     = Close[t] / MA10[t] - 1
close_ma50_gap     = Close[t] / MA50[t] - 1
ma10_ma50_gap      = MA10[t] / MA50[t] - 1
log_volume_change  = log1p(Volume[t]) - log1p(Volume[t-1])
rolling_vol_10     = trailing 10-observation std(log_return_1)
rolling_vol_20     = trailing 20-observation std(log_return_1)
sentiment_mean     = optional daily sentiment mean
news_count         = optional daily article count
```

The moving averages and volatility windows are trailing, never centered.
Features are computed over the complete chronological series, after which rows
without the required history are removed with an explicit warm-up report.
Sentiment is optional: omit its names from `--feature-cols` for price-only
experiments.

Robust scaling is fitted independently for every feature:

```text
median_j = median(x_train,j)
MAD_j    = median(abs(x_train,j - median_j))
scale_j  = 1.4826 * MAD_j
z_j      = (x_j - median_j) / scale_j
```

A zero or near-zero `scale_j` safely falls back to `1.0`. Validation and test
statistics are never used for normalization. The exact processing order is:

```text
raw chronological rows
-> causal features and optional date-intersection with market data
-> warm-up filtering
-> chronological train/validation/test split
-> fit normalizer on train only
-> transform train/validation/test with the stored train state
-> construct 60-observation windows
-> form 5-observation patches
```

For `--forecast-target cumulative_log_return`, a forecast made at the last
observable close `Close[t]` has horizons `h=1,...,H`:

```text
y[t,h] = log(Close[t+h] / Close[t])
```

The default five-point output therefore represents cumulative 1-, 2-, 3-, 4-,
and 5-day log returns. Direction accuracy at every horizon compares
`prediction > 0` with `target > 0`. `--market-data NASDAQ100` adds aligned
`market_log_return_1` and `market_log_return_5` features using the deterministic
intersection of stock and market trading dates. The explicit
`--forecast-target excess_log_return` subtracts the corresponding cumulative
market log return; it is never enabled implicitly.

The stock runner exposes the four fixed preprocessing ablations while leaving
the JEPA/MAE architecture, masks, baselines, and evaluation protocol unchanged:

```bash
# P0: raw + train Z-score + cumulative stock log return
conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
  --stocks NVDA --max-stocks 1 --skip-download --preprocessing-preset P0

# P1: return features + train Z-score + cumulative stock log return
conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
  --stocks NVDA --max-stocks 1 --skip-download --preprocessing-preset P1

# P2 (recommended financial representation): robust return features
conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
  --stocks NVDA --max-stocks 1 --skip-download --preprocessing-preset P2

# P3: robust return features + NASDAQ100 inputs + excess log-return target
conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
  --stocks NVDA --max-stocks 1 --skip-download --preprocessing-preset P3
```

Each evaluation writes `preprocessing_config.json`, and checkpoints retain the
feature order, transform, normalizer state, target definition, and market
setting. Choose preprocessing with training/validation evidence only; final
test performance must not be used to select P0-P3.

The dataset must contain enough rows after the first 49 moving-average rows and the validation/test split to create at least one `series_split_size` window.

### Optional sentiment features

The complete configuration hierarchy and parameter table are documented in
[doc/configuration.md](doc/configuration.md). Use `--use-sentiment` (the
repository default) or `--no-sentiment` to run the same pipeline with only the
feature selection changed. The market-only mode never reads or requires a
sentiment file.

Sentiment columns can already exist in the price CSV, or they can be loaded from a daily sentiment file such as:

```text
data/NVDA/NVDA_daily_sentiment.csv
NVDA_daily_sentiment.csv
```

The sentiment file must contain `date` and the requested sentiment columns. Supported columns include `sentiment_mean`, `sentiment_sum`, `sentiment_max`, `sentiment_min`, `sentiment_std`, and `news_count`.

For price-only training, explicitly select price columns and disable the sentiment file:

```bash
--feature-cols Close Volume --sentiment-path none
```

## Configuration files

The repository uses Python dictionaries for configuration; there is no
`--config FILE` option. The two configuration files provide defaults:

| File | Used for |
| --- | --- |
| `config/config_pretrain.py` | Complete pretraining plus automatic downstream-evaluation workflow. |
| `config/config_downstream.py` | Forecast decoder, fine-tuning, checkpoint, and result-output defaults. |

Run the configured NVDA causal pretraining followed automatically by
relative-return forecasting evaluation:

```bash
conda run --no-capture-output -n ts-jepa python pretrain_dual_loss.py
```

The checked-in pretraining config uses `mask_strategy="future_block"`, selects
the lowest-validation-loss checkpoint, and evaluates
`eval_forecast_target="relative_return"`. Use `--no-run-eval` to run only
pretraining. The downstream config remains available for standalone
`eval_dual_loss.py` runs.

Command-line arguments override configuration-file values. The effective
precedence is configuration defaults, then command-line overrides. The complete
resolved configuration is stored in every unified checkpoint. During
evaluation, `eval_dual_loss.py` reads the pretraining data protocol, feature
order, normalization statistics, and model architecture from that checkpoint.
Keep `target_feature_index` consistent with the order of `feature_cols`; index
`0` selects the first feature.

### Recommended leakage-safe NVDA baseline

There is not yet enough comparable multi-seed evidence in this repository to
claim one empirically best configuration. In particular, do not choose a
configuration from test-set performance. The following is the recommended
starting point for financial forecasting because it preserves chronology and
uses sentiment without fitting normalization statistics on validation or test
data:

| Setting | Recommended value | Reason |
| --- | --- | --- |
| Masking | `future_block` | Context is strictly earlier than the prediction target. |
| Features | `Close Volume MA10 MA50 sentiment_mean` | Combines price, activity, trend, and daily news sentiment. |
| Normalization | `train_zscore` | Fits statistics on the chronological training split only. |
| Window geometry | 60 rows, 5-row patches, stride 5 | Produces 12 patches per window without discarding most possible windows. |
| Objective | JEPA `1.0`, MAE `0.5` | Keeps representation prediction primary while retaining reconstruction regularization. |
| Checkpoint | Lowest pretraining validation loss | Avoids selecting the final epoch by test performance. |
| Encoder for evaluation | `ema` | Uses the more stable target-encoder representation. |

Train once and automatically evaluate the lowest-pretraining-validation-loss
checkpoint:

```bash
conda run --no-capture-output -n ts-jepa python pretrain_dual_loss.py \
  --data NVDA \
  --mask-strategy future_block \
  --feature-cols Close Volume MA10 MA50 sentiment_mean \
  --sentiment-path ./NVDA_daily_sentiment.csv \
  --train-end-date 2024-12-31 \
  --test-start-date 2025-01-01 \
  --data-end-date 2026-01-01 \
  --validation-fraction 0.05 \
  --series-split-size 60 \
  --patch-size 5 \
  --normalization train_zscore \
  --pretrain-stride 5 \
  --target-feature-index 0 \
  --future-target-patches 4 \
  --lambda-jepa 1.0 \
  --lambda-mae 0.5 \
  --jepa-loss mse \
  --mae-loss mse \
  --decoder-type residual_mlp \
  --batch-size 32 \
  --lr 1e-5 \
  --end-lr 1e-6 \
  --ema-momentum 0.998 \
  --num-epochs 2001 \
  --validation-interval 10 \
  --seed 42 \
  --run-eval \
  --eval-use-best \
  --eval-encoder-weights ema \
  --eval-num-epochs 501 \
  --eval-results-dir results/NVDA/future_block_sentiment/seed_42
```

`--data-end-date` is an inclusive global cutoff applied before the
chronological train/validation/test split. It prevents pretraining and every
downstream evaluator from using observations after that date. The configured
cutoff is stored in the checkpoint and restored automatically by
`eval_dual_loss.py`.

`--test-start-date` is required for time-series runs. Rows on or after that
date form the held-out test set; `--validation-fraction` only divides the
earlier train/validation period. If `--train-end-date` is omitted, every row
before `--test-start-date` is eligible for the train/validation split.

For a defensible comparison, repeat the complete workflow across several seeds
and report mean and standard deviation instead of selecting the best seed:

```bash
conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
  --stocks NVDA \
  --max-stocks 1 \
  --skip-download \
  --mask-strategies future_block \
  --future-target-patches 4 \
  --lambda-jepa 1.0 \
  --lambda-mae 0.5 \
  --pretrain-stride 5 \
  --normalization train_zscore \
  --seeds 7 17 42 73 101 \
  --pretrain-num-epochs 2001 \
  --use-best-checkpoint \
  --eval-num-epochs 501 \
  --results-dir results/NVDA_future_block_sentiment
```

The stock runner obtains feature columns, sentiment path, dates, and model
geometry from `config/config_pretrain.py`. Before using `--skip-download`, make
sure `data/NVDA/NVDA.csv` and the configured sentiment file cover the requested
dates. Select configurations using validation metrics; use the held-out test
split only for the final report, and always compare against `naive_last`,
`drift`, and GRU.

### Multi-strategy stock comparison

Complete reusable option files are provided for the controlled sentiment
comparison:

```bash
# Market features plus sentiment
conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
  --config config/experiments/top10_with_sentiment.json

# Identical stock/seed/masking setup without sentiment
conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
  --config config/experiments/top10_without_sentiment.json
```

The same file is passed to the standalone analyzer after training:

```bash
conda run --no-capture-output -n ts-jepa python analyze_stock_results.py \
  --config config/experiments/top10_with_sentiment.json
```

Each JSON or TOML file may contain `common`, `runner`, and `analysis` objects.
Both scripts read `common`; the stock runner reads `runner`, and the analyzer
reads `analysis`. Option names use Python/JSON underscores, such as
`mask_strategies` and `use_sentiment`. Explicit command-line options override
file values, so a safe command inspection is:

```bash
python run_top_nasdaq100_stocks.py \
  --config config/experiments/top10_with_sentiment.json \
  --dry-run
```

The stock runner uses one `--mask-strategies` option for both single- and
multi-strategy runs. It always stores each method under its own strategy
directory, preventing result files from different masking methods from being
aggregated together.

Run random and local-long masking on the configured top 10 stocks with the same
10 reproducible seeds and sentiment enabled:

```bash
conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
  --stocks NVDA AAPL MSFT AMZN GOOGL AVGO META TSLA GOOG WMT \
  --max-stocks 0 \
  --mask-strategies random local_long \
  --series-split-size 120 \
  --patch-size 5 \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --use-sentiment \
  --skip-download \
  --results-dir results/top10_nasdaq100_mask_comparison
```

After training, the equivalent explicit command for extracting the comparison
without rerunning either model is:

```bash
conda run --no-capture-output -n ts-jepa python analyze_stock_results.py \
  --results-dir results/top10_nasdaq100_mask_comparison
```

The runner writes `experiment_manifest.json`, so the analyzer can recover the
strategies, stocks, and seeds automatically. It creates:

```text
raw_runs.csv
per_stock_summary.csv
per_seed_summary.csv
overall_summary.csv
paired_strategy_differences.csv
missing_or_failed_runs.csv
strategy_comparison.png
```

By default, missing runs or expected models are recorded in
`missing_or_failed_runs.csv` and partial summary statistics are refused. Use
`--allow-incomplete` only for explicitly exploratory analysis.
In `paired_strategy_differences.csv`, each delta is `strategy_b - strategy_a`;
MSE and MAE are lower-is-better, while trend accuracy is higher-is-better.

## Unified pretraining

Use `pretrain_dual_loss.py` for all strategies. The default seed is `42`; pass `--seed` explicitly for published experiments.

### Random JEPA + MAE targets

```bash
conda run --no-capture-output -n ts-jepa python pretrain_dual_loss.py \
  --no-run-eval \
  --data NVDA \
  --mask-strategy random \
  --feature-cols Close Volume \
  --sentiment-path none \
  --train-end-date none \
  --test-start-date 2025-01-01 \
  --series-split-size 120 \
  --patch-size 5 \
  --pretrain-stride 5 \
  --normalization window_return \
  --mask-ratio 0.7 \
  --lambda-jepa 1.0 \
  --lambda-mae 0.5 \
  --decoder-type residual_mlp \
  --batch-size 32 \
  --lr 1e-5 \
  --end-lr 5e-6 \
  --ema-momentum 0.998 \
  --num-epochs 2001 \
  --seed 7
```

In this mode, `mask_ratio` determines the fraction of patches used as targets. A value strictly between `0` and `1` is required.

For JEPA-only training compatible with the former L1 objective, disable the MAE weight:

```bash
conda run --no-capture-output -n ts-jepa python pretrain_dual_loss.py \
  --no-run-eval \
  --data NVDA \
  --mask-strategy random \
  --feature-cols Close Volume \
  --sentiment-path none \
  --lambda-jepa 1 \
  --lambda-mae 0 \
  --jepa-loss l1
```

### Local MAE + long-horizon JEPA

```bash
conda run --no-capture-output -n ts-jepa python pretrain_dual_loss.py \
  --no-run-eval \
  --data NVDA \
  --mask-strategy local_long \
  --feature-cols Close Volume \
  --sentiment-path none \
  --train-end-date none \
  --test-start-date 2025-01-01 \
  --series-split-size 120 \
  --patch-size 5 \
  --mae-window-patches 1 \
  --jepa-gap-patches 4 \
  --jepa-target-patches 4 \
  --anchor-strategy random \
  --lambda-jepa 1.0 \
  --lambda-mae 0.5 \
  --decoder-type residual_mlp \
  --batch-size 32 \
  --lr 1e-5 \
  --end-lr 5e-6 \
  --ema-momentum 0.998 \
  --num-epochs 2001 \
  --seed 7
```

With `series_split_size=120` and `patch_size=5`, each sample contains 24 patches. The local strategy parameters must satisfy:

```text
mae_window_patches < jepa_gap_patches
jepa_gap_patches + jepa_target_patches <= number_of_patches
```

For a fixed experiment location, use:

```bash
--anchor-strategy fixed --fixed-anchor 0
```

The fixed anchor must be between `0` and:

```text
number_of_patches - jepa_gap_patches - jepa_target_patches
```

### Strictly causal future masking

Predict one contiguous future block using only earlier context:

```bash
conda run --no-capture-output -n ts-jepa python pretrain_dual_loss.py \
  --no-run-eval \
  --data NVDA \
  --mask-strategy future_block \
  --future-target-patches 4 \
  --pretrain-stride 5 \
  --normalization train_zscore \
  --seed 42
```

Predict multiple future blocks after one forecast cutoff:

```bash
conda run --no-capture-output -n ts-jepa python pretrain_dual_loss.py \
  --no-run-eval \
  --data NVDA \
  --mask-strategy causal_multiblock \
  --causal-num-blocks 2 \
  --causal-block-patches 2 \
  --causal-block-gap-patches 1 \
  --pretrain-stride 5 \
  --seed 42
```

For both causal modes, every context index is strictly earlier than every target index. `local_long` guarantees this for its long JEPA target, while its local MAE objective may use patches after the local reconstruction region.

## Quick smoke run

Use one epoch and a small number of batches to validate the data and model configuration:

```bash
conda run --no-capture-output -n ts-jepa python pretrain_dual_loss.py \
  --no-run-eval \
  --data NVDA \
  --mask-strategy random \
  --feature-cols Close Volume \
  --sentiment-path none \
  --train-end-date none \
  --test-start-date 2025-01-01 \
  --num-epochs 1 \
  --max-batches-per-epoch 2 \
  --checkpoint-print 1 \
  --seed 7
```

## Checkpoints

Checkpoints are written under:

```text
logs/output_model/<DATASET>/
```

The filename records the readable strategy fields plus a deterministic configuration fingerprint. The fingerprint includes seed, loss types, data protocol, normalization, stride, feature order, model/decoder geometry, and causal-mask settings, preventing incompatible experiments from silently overwriting one another. The final epoch is saved by default.

Each unified checkpoint contains:

- `encoder`: online encoder weights;
- `encoder_ema`: EMA target-encoder weights;
- `predictor`: JEPA predictor weights;
- `decoder`: MAE reconstruction decoder weights;
- `optimizer` and `scheduler`: complete optimizer state for exact continuation;
- `global_step` and Python/NumPy/PyTorch RNG states;
- train-only normalization statistics when `train_zscore` is used;
- validation history and best validation metric;
- `config`: the complete pretraining configuration;
- `strategy` and `epoch` metadata.

Resume without restarting the optimizer, scheduler, EMA schedule, or random streams:

```bash
conda run --no-capture-output -n ts-jepa python pretrain_dual_loss.py \
  --no-run-eval \
  --data NVDA \
  --resume-from logs/output_model/NVDA/CHECKPOINT_FILE.pt \
  --num-epochs 2501
```

The resume command must use the same training configuration fingerprint. `num_epochs` may be extended.

### Pretraining validation

The chronological validation split uses deterministic masks. Every `--validation-interval` epochs, pretraining reports:

```text
validation total / JEPA / MAE loss
EMA embedding standard deviation
effective rank
mean absolute off-diagonal covariance
```

The best validation checkpoint is written as `..._best.pt`. Use `--validation-max-batches N` to limit validation cost. If the validation split is shorter than one complete pretraining window, validation is disabled with a warning.

Useful checkpoint options:

```text
--checkpoint-save N       Save every N epochs, excluding epoch 0.
--no-save-final           Do not save the final epoch automatically.
--resume-from PATH        Restore complete training state.
--path-suffix SUFFIX      Override the generated strategy suffix.
--compatible-save-name    Omit the strategy suffix for legacy tooling.
```

`ratio_patches` is retained in checkpoint names for compatibility with older scripts. It does not control masking; use `mask_ratio` for random masking.

## Forecast evaluation

The most reliable method is to pass the exact checkpoint printed by pretraining. The evaluation wrapper reads architecture and data settings from the checkpoint:

```bash
conda run --no-capture-output -n ts-jepa python eval_dual_loss.py \
  --data NVDA \
  --pretrain-checkpoint-path logs/output_model/NVDA/CHECKPOINT_FILE.pt \
  --pretrain-encoder-weights ema \
  --num-epochs 501
```

EMA weights are the default for downstream representation evaluation. Use `--pretrain-encoder-weights online` for an explicit online-vs-EMA ablation. Legacy checkpoints without `encoder_ema` fall back to the online encoder with a warning.

### Config-driven best and last checkpoint selection

Checkpoint selection is configured in `config/config_downstream.py`. The
checked-in configuration selects the last random-mask, JEPA `1.0`, MAE `1.0`
checkpoint:

```python
"checkpoint_selection": "last",
"pretrain_checkpoint_path": None,
"mask_strategy": "random",
"lambda_jepa": 1.0,
"lambda_mae": 1.0,
```

Run the configured checkpoint without repeating its filename or geometry:

```bash
conda run --no-capture-output -n ts-jepa python eval_dual_loss.py
```

Inspect the resolved checkpoint and delegated evaluation command first:

```bash
conda run -n ts-jepa python eval_dual_loss.py --dry-run
```

Change only `checkpoint_selection` to choose another checkpoint policy:

| Value | Resolution |
| --- | --- |
| `best` | The matching `_best.pt` checkpoint with the lowest pretraining validation loss. |
| `last` | The matching epoch checkpoint with the largest saved epoch number. |
| `epoch` | The matching `checkpoint_to_use` epoch. |
| `path` | The exact `pretrain_checkpoint_path`. |

A temporary command-line override remains available. For example, evaluate the
last checkpoint while keeping the config default unchanged:

```bash
conda run --no-capture-output -n ts-jepa python eval_dual_loss.py \
  --checkpoint-selection last
```

The strategy, loss weights, and mask geometry in `config_downstream.py` must
match the checkpoint family. After training the recommended MAE `0.5`
configuration, change `lambda_mae` to `0.5`. If several fingerprints match the
same family, selection stops with an ambiguity error; set
`checkpoint_selection` to `path` and provide the exact
`pretrain_checkpoint_path` instead of loading one silently.

Feature order, sentiment path, chronological split, inclusive data cutoff,
train-only normalization statistics, model architecture, and the actual
checkpoint epoch are restored from the selected checkpoint.

### Forecast future relative-return paths

The downstream decoder can predict either the target value in the configured
normalization space (the backward-compatible default) or a cumulative simple
return path relative to the last observation at the forecast cutoff:

```text
relative_return[h] = target[t + h] / target[t] - 1
```

Here `target[t]` is the last context value and `h=1..patch_size`, so label
construction never uses a future value as its base. Input features still use
the checkpoint's pretraining normalization; only the downstream label changes.
Run the additional task with:

```bash
conda run --no-capture-output -n ts-jepa python eval_dual_loss.py \
  --data NVDA \
  --pretrain-checkpoint-path logs/output_model/NVDA/CHECKPOINT_FILE.pt \
  --forecast-target relative_return \
  --num-epochs 501
```

In this mode TS-JEPA, GRU, and all simple baselines are trained or evaluated in
the same return space. The zero-return `naive_last` baseline represents an
unchanged future price, and trend accuracy includes the first move from the
zero-return forecast origin. Set `forecast_target="relative_return"` in
`config/config_downstream.py` to make this the configured default.
The stock batch runners accept the same `--forecast-target relative_return`
option; with `pretrain_dual_loss.py --run-eval`, use
`--eval-forecast-target relative_return`.

For a local/long checkpoint, use the unified evaluator:

```bash
conda run --no-capture-output -n ts-jepa python eval_dual_loss.py \
  --data NVDA \
  --mask-strategy local_long \
  --pretrain-checkpoint-path logs/output_model/NVDA/CHECKPOINT_FILE.pt \
  --num-epochs 501
```

To inspect the resolved checkpoint and delegated forecast command without running evaluation:

```bash
conda run -n ts-jepa python eval_dual_loss.py \
  --data NVDA \
  --pretrain-checkpoint-path logs/output_model/NVDA/CHECKPOINT_FILE.pt \
  --dry-run
```

### Train and evaluate in one command

For time-series input, pretraining can start downstream forecast evaluation after saving:

```bash
conda run --no-capture-output -n ts-jepa python pretrain_dual_loss.py \
  --data NVDA \
  --mask-strategy random \
  --feature-cols Close Volume \
  --sentiment-path none \
  --train-end-date none \
  --test-start-date 2025-01-01 \
  --run-eval \
  --eval-use-best \
  --eval-encoder-weights ema \
  --eval-num-epochs 501
```

The command above evaluates `..._best.pt`. Omit `--eval-use-best` to evaluate
the last checkpoint saved in the current run, or use
`--eval-checkpoint-to-use EPOCH` to select a specific saved epoch. Use
`--eval-encoder-weights ema|online` to select the representation source.

## Batch NASDAQ-100 workflow

`run_top_nasdaq100_stocks.py` orchestrates the complete workflow for several stocks:

1. Download or append price and news-sentiment data for all selected tickers.
2. Pretrain each ticker with `pretrain_dual_loss.py`.
3. Evaluate each checkpoint with `eval_dual_loss.py` and the GRU baseline.
4. Save each ticker's metrics, CSV files, and images under `results/TICKER/seed_N/` for a legacy singular run or `results/STRATEGY/TICKER/seed_N/` for a plural run.
5. Generate a combined metrics CSV and image after all evaluations finish; multi-seed plots show mean with standard-deviation error bars.
6. Record every generated command in `results/top_nasdaq100_stock_runs.txt`.
7. Save the resolved experiment scope and comparison-relevant settings in `results/experiment_manifest.json`.

Commands are executed sequentially. The workflow stops immediately if a download, pretraining, or evaluation command fails.

The default output layout is:

```text
results/
├── NVDA/
│   └── seed_42/
│       ├── loss.txt
│       ├── last_model_comparison_*.csv
│       ├── last_model_comparison_*.txt
│       ├── last_model_comparison_*.png
│       └── rolling_windows_with_baselines/
├── MSFT/
│   └── ...
├── top_2_nasdaq100_*.csv
├── top_2_nasdaq100_*.png
└── top_nasdaq100_stock_runs.txt
```

Here `2` is the actual number of selected stocks after applying `--max-stocks`. The combined PNG title and filename use `top_<count>_nasdaq100`, and the image compares MSE, MAE, and trend accuracy. Timestamped files from earlier runs are preserved. Fixed-name files such as `loss.txt` are safe because each ticker has its own directory.

For repeated-seed experiments, the layout becomes `results/NVDA/seed_7/`, `results/NVDA/seed_17/`, and so on. The combined CSV contains mean, standard deviation, and `num_runs` for every stock/model pair.

With `--mask-strategies random local_long`, both strategy trees are created
below the result root. Each strategy gets its own combined plot, while
`analyze_stock_results.py` produces the cross-strategy summaries at the root.
The supplied comparison configs use `series_split_size=120` and `patch_size=5`,
giving 24 patches. This is required because the local-long defaults need at
least `jepa_gap_patches + jepa_target_patches = 8` patches.

### Inspect commands safely

Start with a dry run. This prints commands without executing them and still writes the run summary:

```bash
conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
  --stocks NVDA MSFT \
  --max-stocks 2 \
  --skip-download \
  --mask-strategies random \
  --pretrain-num-epochs 3 \
  --checkpoint-to-use 2 \
  --eval-num-epochs 4 \
  --dry-run
```

Remove `--skip-download` from a dry run if you also want to inspect the generated download command.

### Full random dual-loss run

This downloads/appends data, trains three stocks, and evaluates epoch 2000:

```bash
conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
  --stocks NVDA MSFT AMD \
  --max-stocks 3 \
  --download-start-date 2015-01-01 \
  --download-end-date 2025-12-31 \
  --write-mode append \
  --news-chunk-days 7 \
  --request-delay 0.5 \
  --mask-strategies random \
  --lambda-jepa 1.0 \
  --lambda-mae 0.5 \
  --jepa-loss mse \
  --mae-loss mse \
  --pretrain-stride 5 \
  --normalization train_zscore \
  --seed 42 \
  --pretrain-num-epochs 2001 \
  --checkpoint-to-use 2000 \
  --eval-num-epochs 501
```

`append` is the default write mode and preserves existing downloaded history. Use `overwrite` only when the ticker data should be rebuilt.

### Multi-seed reporting

Run every selected stock with several seeds and aggregate mean/standard deviation automatically:

```bash
conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
  --stocks NVDA MSFT AMD \
  --max-stocks 3 \
  --skip-download \
  --mask-strategies future_block \
  --future-target-patches 4 \
  --seeds 7 17 42 73 101 \
  --pretrain-stride 5 \
  --normalization train_zscore \
  --pretrain-num-epochs 2001 \
  --use-best-checkpoint
```

### Local MAE + long-JEPA run

Use the unified local/long strategy for every selected stock:

```bash
conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
  --stocks NVDA MSFT \
  --max-stocks 2 \
  --skip-download \
  --mask-strategies local_long \
  --series-split-size 120 \
  --patch-size 5 \
  --mae-window-patches 1 \
  --jepa-gap-patches 4 \
  --jepa-target-patches 4 \
  --lambda-jepa 1.0 \
  --lambda-mae 0.5 \
  --pretrain-num-epochs 2001 \
  --checkpoint-to-use 2000 \
  --eval-num-epochs 501
```

When `--skip-download` is used, every `data/TICKER/TICKER.csv` file must already exist and contain the feature columns configured in `config/config_pretrain.py`.

### JEPA-only run

To reproduce the former L1 JEPA-only objective across stocks:

```bash
conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
  --stocks NVDA MSFT \
  --max-stocks 2 \
  --skip-download \
  --mask-strategies random \
  --lambda-jepa 1 \
  --lambda-mae 0 \
  --jepa-loss l1 \
  --pretrain-num-epochs 2001 \
  --checkpoint-to-use 2000
```

### Evaluate existing checkpoints only

Skip both data download and pretraining:

```bash
conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
  --stocks NVDA MSFT \
  --max-stocks 2 \
  --skip-download \
  --skip-pretrain \
  --mask-strategies random \
  --lambda-jepa 1.0 \
  --lambda-mae 0.5 \
  --checkpoint-to-use 2000 \
  --eval-num-epochs 501
```

The strategy, loss weights, local-window settings, and checkpoint epoch must match the checkpoint filename. If custom architecture or path settings were used, evaluate with `eval_dual_loss.py --pretrain-checkpoint-path ...` directly instead.

### Stock, download, and news selection

| Option | Behavior |
| --- | --- |
| `--stocks NVDA MSFT ...` | Explicit ticker list. Without it, the built-in top NASDAQ-100 list is used. |
| `--max-stocks N` | Uses only the first `N` selected tickers. The default is `5`; use `0` for all. |
| `--skip-download` | Skips both price and news ingestion. Existing local CSVs are required. |
| `--skip-news` | Downloads prices but skips news scoring and ensures zero-valued sentiment columns exist. |
| `--download-start-date`, `--download-end-date` | Raw price/news download range in `YYYY-MM-DD` format. |
| `--write-mode append` | Adds to existing history; this is the default. |
| `--max-news-articles N` | Limits the news articles processed per symbol. |
| `--news-chunk-days N` | Size of each news request window. |
| `--request-delay SECONDS` | Delay between news requests. |
| `--seed N` | Reproducible single-seed run; default `42`. |
| `--seeds N ...` | Runs every stock for every listed seed and aggregates mean/std; overrides `--seed`. |
| `--mask-strategies NAME ...` | Runs one or more strategies in isolated strategy subdirectories; default `random`. |
| `--series-split-size N` | Rows per pretraining window; must be divisible by `--patch-size`. |
| `--patch-size N` | Rows per temporal patch; local/long geometry is validated against the resulting patch count. |
| `--pretrain-stride N` | Sliding-window stride used during pretraining; runner default `5`. |
| `--sampling-mode MODE` | Uses overlapping `sliding_window` samples or non-overlapping `temporal_segments`. |
| `--normalization MODE` | `window_return`, `train_zscore`, `train_robust_zscore`, or `none`. |
| `--feature-transform raw|return` | Selects legacy raw inputs or the causal return representation. |
| `--preprocessing-preset P0|P1|P2|P3` | Applies a fixed preprocessing ablation in the stock runners. |
| `--market-data NAME` | Optionally aligns market returns, for example `NASDAQ100`. |
| `--encoder-weights ema|online` | Chooses downstream checkpoint encoder; default `ema`. |
| `--use-best-checkpoint` | Evaluates each run's deterministic `..._best.pt` instead of `--checkpoint-to-use`. |
| `--results-dir PATH` | Changes the output root; per-stock subdirectories and combined files are created below it. |
| `--skip-combined-plot` | Runs evaluations without creating the final combined CSV and PNG. |

To rebuild the combined image manually from existing per-stock results:

```bash
conda run --no-capture-output -n ts-jepa python plot_top_stock_metrics.py \
  --results-dir results \
  --output-dir results \
  --seeds 7 17 42 \
  --stocks NVDA MSFT AMD
```

The final pretraining epoch is `pretrain_num_epochs - 1`. Normally, set `--checkpoint-to-use` to that value. For example:

```text
pretrain-num-epochs=2001 -> checkpoint-to-use=2000
pretrain-num-epochs=3    -> checkpoint-to-use=2
```

Show all batch-runner options with:

```bash
conda run -n ts-jepa python run_top_nasdaq100_stocks.py --help
```

## MNIST row reconstruction

MNIST mode treats each 28-pixel image row as one token. It supports the random masking strategy.

```bash
conda run --no-capture-output -n ts-jepa python pretrain_dual_loss.py \
  --no-run-eval \
  --data MNIST_ROWS \
  --input-mode mnist_rows \
  --mnist-root data/MNIST \
  --download-mnist \
  --mnist-train-samples 4000 \
  --mask-strategy random \
  --mask-ratio 0.4 \
  --encoder-embed-dim 64 \
  --encoder-nhead 4 \
  --predictor-embed 64 \
  --predictor-nhead 4 \
  --decoder-type mlp \
  --lambda-jepa 0.01 \
  --lambda-mae 1.0 \
  --num-epochs 30 \
  --seed 7
```

Evaluate a saved MNIST checkpoint with:

```bash
conda run --no-capture-output -n ts-jepa python eval_dual_loss.py \
  --data MNIST_ROWS \
  --eval-mode mnist_rows \
  --mnist-root data/MNIST \
  --pretrain-checkpoint-path logs/output_model/MNIST_ROWS/CHECKPOINT_FILE.pt \
  --require-better-than-naive
```

Do not use `--run-eval` for MNIST; that option launches the time-series forecast evaluator.

## Important parameter rules

- `0 < mask_ratio < 1`.
- `0 < end_lr <= lr`.
- `0 <= ema_momentum < 1`.
- `lambda_jepa` and `lambda_mae` must be non-negative, and at least one must be positive.
- Encoder and predictor embedding dimensions must be even and divisible by their respective attention-head counts.
- `encoder_kernel_size` cannot exceed the flattened patch dimension.
- `local_long`, `future_block`, and `causal_multiblock` require time-series input.
- `pretrain_stride` must be positive; omitting it uses `patch_size`.
- `temporal_segments` uses `series_split_size` as its effective stride and drops only the incomplete tail.
- `future_block` and `causal_multiblock` require at least one context patch before all targets.
- `max_batches_per_epoch`, when set, must be positive.

Show every available option with:

```bash
conda run -n ts-jepa python pretrain_dual_loss.py --help
conda run -n ts-jepa python eval_dual_loss.py --help
```

## Tests

Run all smoke and regression tests in the project environment:

```bash
conda run --no-capture-output -n ts-jepa python -m unittest discover -s tests -v
```

The suite covers:

- random dual-loss pretraining;
- local-MAE + long-JEPA pretraining;
- future-block and causal multi-block geometry;
- train-only normalization, configurable stride, and non-overlapping temporal segments;
- complete checkpoint save/resume and evaluation resolution;
- iteration-level EMA scheduling and online/EMA selection;
- deterministic validation and collapse metrics;
- multi-seed metric aggregation;
- downstream context-length compatibility;
- synthetic forecasting and MNIST reconstruction.

## Repository structure

```text
config/                         Training and downstream defaults
data/                           Time-series and optional MNIST data
logs/output_model/              Saved pretraining checkpoints
main/                           Shared CLI and training utilities
src/data_loaders/               Time-series and MNIST loaders
src/models/                     Encoder, predictor, and decoder modules
tests/                          Smoke and regression tests
pretrain_dual_loss.py           Unified pretraining entrypoint
eval_dual_loss.py               Unified evaluation wrapper
run_top_nasdaq100_stocks.py     Multi-stock download, pretrain, and evaluation workflow
analyze_stock_results.py        Saved-run validation and strategy comparison analysis
```

## Citation

If this work is useful in your research, please cite the TS-JEPA paper linked above. For questions about the original implementation, contact `ennadir@kth.se`.
