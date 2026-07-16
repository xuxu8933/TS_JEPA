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
- builds sliding windows and divides each window into patches;
- uses only the training split during pretraining.

Pretraining stride defaults to `patch_size`, rather than the complete window length. Override it with `--pretrain-stride N`. For example, a 60-row window with `patch_size=5` uses `stride=5` unless explicitly changed.

Normalization is selected explicitly:

| Mode | Behavior |
| --- | --- |
| `window_return` | Divides every window by its first observable row and subtracts one. This is the default and does not fit global statistics. |
| `train_zscore` | Fits mean and standard deviation on the chronological train split only, stores them in the checkpoint, and reuses them for validation/test. |
| `none` | Uses the loaded feature values without additional normalization. |

Example:

```bash
--pretrain-stride 5 --normalization train_zscore
```

The dataset must contain enough rows after the first 49 moving-average rows and the validation/test split to create at least one `series_split_size` window.

### Optional sentiment features

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

## Unified pretraining

Use `pretrain_dual_loss.py` for all strategies. The default seed is `42`; pass `--seed` explicitly for published experiments.

### Random JEPA + MAE targets

```bash
conda run --no-capture-output -n ts-jepa python pretrain_dual_loss.py \
  --data NVDA \
  --mask-strategy random \
  --feature-cols Close Volume \
  --sentiment-path none \
  --train-end-date none \
  --test-start-date none \
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
  --data NVDA \
  --mask-strategy local_long \
  --feature-cols Close Volume \
  --sentiment-path none \
  --train-end-date none \
  --test-start-date none \
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
  --data NVDA \
  --mask-strategy random \
  --feature-cols Close Volume \
  --sentiment-path none \
  --train-end-date none \
  --test-start-date none \
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
  --test-start-date none \
  --run-eval \
  --eval-num-epochs 501
```

Use `--eval-checkpoint-to-use EPOCH` to evaluate a specific saved epoch. Otherwise, the last checkpoint saved in the current run is used.

Use `--eval-use-best` to launch downstream evaluation from `..._best.pt`; `--eval-encoder-weights ema|online` selects the representation source.

## Batch NASDAQ-100 workflow

`run_top_nasdaq100_stocks.py` orchestrates the complete workflow for several stocks:

1. Download or append price and news-sentiment data for all selected tickers.
2. Pretrain each ticker with `pretrain_dual_loss.py`.
3. Evaluate each checkpoint with `eval_dual_loss.py` and the GRU baseline.
4. Save each ticker's metrics, CSV files, and images under `results/TICKER/seed_N/`.
5. Generate a combined metrics CSV and image after all evaluations finish; multi-seed plots show mean with standard-deviation error bars.
6. Record every generated command in `results/top_nasdaq100_stock_runs.txt`.

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

### Inspect commands safely

Start with a dry run. This prints commands without executing them and still writes the run summary:

```bash
conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
  --stocks NVDA MSFT \
  --max-stocks 2 \
  --skip-download \
  --mask-strategy random \
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
  --start-date 2015-01-01 \
  --end-date 2025-12-31 \
  --write-mode append \
  --news-chunk-days 7 \
  --request-delay 0.5 \
  --mask-strategy random \
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
  --mask-strategy future_block \
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
  --mask-strategy local_long \
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
  --mask-strategy random \
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
  --mask-strategy random \
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
| `--start-date`, `--end-date` | Download range in `YYYY-MM-DD` format. |
| `--write-mode append` | Adds to existing history; this is the default. |
| `--max-news-articles N` | Limits the news articles processed per symbol. |
| `--news-chunk-days N` | Size of each news request window. |
| `--request-delay SECONDS` | Delay between news requests. |
| `--seed N` | Reproducible single-seed run; default `42`. |
| `--seeds N ...` | Runs every stock for every listed seed and aggregates mean/std; overrides `--seed`. |
| `--pretrain-stride N` | Sliding-window stride used during pretraining; runner default `5`. |
| `--normalization MODE` | `window_return`, `train_zscore`, or `none`. |
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
- train-only normalization and configurable stride;
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
```

## Citation

If this work is useful in your research, please cite the TS-JEPA paper linked above. For questions about the original implementation, contact `ennadir@kth.se`.
