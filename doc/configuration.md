# TS-JEPA configuration reference

TS-JEPA keeps its authoritative defaults in
`config/experiment.py`. The defaults are grouped into data, preprocessing,
masking/checkpoint, model, training, evaluation, and runtime sections. The
legacy modules `config/config_pretrain.py` and `config/config_downstream.py`
flatten those sections into dictionaries so existing entrypoints and saved
checkpoints remain compatible.

The resolved flat dictionary—not an unresolved default—is saved in every
pretraining checkpoint. Downstream evaluation also writes
`preprocessing_config.json`. These artifacts include `use_sentiment`, the
configured market and sentiment feature groups, and the effective feature
order.

## Script option files

`run_top_nasdaq100_stocks.py` and `analyze_stock_results.py` accept
`--config FILE`. JSON and TOML are supported. A shared file uses three objects:

```json
{
  "common": {
    "stocks": ["NVDA", "AAPL"],
    "seeds": [42, 43],
    "results_dir": "./results/example"
  },
  "runner": {
    "mask_strategies": ["random", "local_long"],
    "series_split_size": 120,
    "patch_size": 5,
    "use_sentiment": true,
    "skip_download": true
  },
  "analysis": {
    "strategies": ["random", "local_long"],
    "allow_incomplete": false
  }
}
```

The runner merges `common` with `runner`; the analyzer merges `common` with
`analysis`. A file without these section names is treated as a flat option file
for the receiving script. Keys use underscore-form argument destinations, not
CLI hyphens. Unknown sections and options fail immediately to prevent silent
experiment drift.

File values become parser defaults. Explicit CLI options take precedence,
including strategy and seed selectors. For example, this uses the common
data/output settings but replaces the configured strategy list and seed list:

```bash
python run_top_nasdaq100_stocks.py \
  --config config/experiments/top10_with_sentiment.json \
  --mask-strategies future_block \
  --seed 7
```

Boolean runner/analyzer switches also have explicit negative forms. For
example, `--no-skip-download` overrides `"skip_download": true`, and
`--no-skip-plot` overrides `"skip_plot": true`.

`runner.execution.max_parallel_jobs` (or `--max-parallel-jobs`) controls how
many independent stock/seed/strategy task chains the runner may execute at
once. It defaults to `1`, preserving sequential execution. On a single GPU,
`2` can overlap two complete task chains when memory permits. This option is
runtime-only: it is recorded in runner provenance but excluded from experiment
identity and is not forwarded to pretraining or downstream model commands.

Complete configurations are available at:

- `config/experiments/top10_with_sentiment.json`
- `config/experiments/top10_without_sentiment.json`
- `config/experiments/top10_h1_without_sentiment.json`
- `config/experiments/top10_h1_with_sentiment.json`
- `config/experiments/top10_sentiment_has_news.json`
- `config/experiments/top10_sentiment_zscore.json`

Use the selected file for both stages:

```bash
python run_top_nasdaq100_stocks.py --config CONFIG.json
python analyze_stock_results.py --config CONFIG.json
```

The runner records the supplied filename and all resolved arguments in
`experiment_manifest.json`. The analyzer records its selected filename and
resolved analysis scope in `analysis_manifest.json`.

## Sentiment-mechanism options

The downstream forecast width is independent of the encoder patch geometry.
`forecast_horizon` controls the number of future target values and downstream
head outputs; `patch_size` continues to control how historical rows are grouped
for the encoder. When omitted, `forecast_horizon` defaults to `patch_size`, so
existing configs and checkpoints keep their five-step behavior.

For example, H1 keeps five-row input patches and requests a one-step target:

```json
{
  "runner": {
    "downstream": {
      "epochs": 501,
      "forecast_horizon": 1
    }
  }
}
```

`has_news` is derived from the same trading-date merge as `news_count`:
`1.0` means at least one matched article and `0.0` means none. It is not shifted,
forward-filled, or nearest-date matched, so observed neutral sentiment remains
distinguishable from missing news without changing the available information
set.

Selective sentiment scaling is configured inside the sentiment feature group:

```json
{
  "runner": {
    "preprocessing": {
      "preset": null,
      "custom": {
        "features": {
          "sentiment": {
            "enabled": true,
            "columns": ["sentiment_mean_z"],
            "normalization": "train_zscore"
          }
        }
      }
    }
  }
}
```

The raw `sentiment_mean` source is merged before chronological splitting. Mean
and population standard deviation are then fitted on the training split separately for each stock. That state is stored in the checkpoint and reused
unchanged for validation, test, and downstream evaluation. Market features
retain the configured global mode (for these ablations, `window_return`), while
the transformed sentiment channel passes through the window transform.

## Canonical experiment

The repository default remains the prior sentiment-enabled experiment. The
same entrypoint runs the controlled market-only variant:

```bash
# Market features plus sentiment_mean (repository default)
python pretrain_dual_loss.py --use-sentiment

# Identical pipeline and defaults, but market features only
python pretrain_dual_loss.py --no-sentiment --sentiment-path none
```

The stock runners expose the same pair of flags. `--feature-cols` remains a
compatibility override for existing commands. New configurations should prefer
`--market-features`, `--sentiment-features`, and the explicit toggle.

## Feature selection

| Setting | Effective raw features | Sentiment file access |
| --- | --- | --- |
| `use_sentiment=false` | `Close`, `Volume`, `MA10`, `MA50` | Never read or required |
| `use_sentiment=true` | `Close`, `Volume`, `MA10`, `MA50`, `sentiment_mean` | Required if the selected sentiment column is not already in the market CSV |

`market_features` and `sentiment_features` are ordered and must be disjoint.
The derived `feature_cols` list is the only feature list consumed by the data
pipeline. The derived model input width is:

```text
feature_dim    = len(effective feature_cols)
patch_input_dim = patch_size * feature_dim
```

With the defaults, `patch_input_dim` is 25 with sentiment and 20 without it.
The encoder reads the dataset's actual patch width; no model feature count is
hard-coded.

In `feature_transform=return` mode, the eight canonical causal return features
replace the raw market list. Configured sentiment features remain optional and
market-index features are appended only when `market_data` is enabled.

Daily sentiment is merged by normalized calendar date using a left join.
Missing news days and non-numeric values in an existing selected sentiment
column are filled with `0.0`; the market calendar and sample count are not
changed. Enabling sentiment without an available file fails with a
`FileNotFoundError` rather than silently changing the experiment.

## Parameter reference

`Required` means an entrypoint must receive a value. A dash means the setting
does not apply to that stage. Defaults below are the repository values in
`config/experiment.py`.

| Parameter | Section | Type | Pretrain default | Downstream default | Description | Experiment impact |
| --- | --- | --- | --- | --- | --- | --- |
| `data` | Data | str | `NVDA` | `NVDA` | Dataset/ticker name; resolves to `data/<name>/<name>.csv` | Methodological |
| `input_mode` | Data | str | `timeseries` | — | `timeseries` or the separate `mnist_rows` smoke task | Methodological |
| `timestamp_col` | Data | str | `Date` | `Date` | Chronological timestamp column | Methodological |
| `market_features` | Data | list[str] | `Close, Volume, MA10, MA50` | Same | Ordered market input features | Methodological |
| `sentiment_features` | Data | list[str] | `sentiment_mean` | Same | Ordered news-derived features used when enabled | Methodological |
| `use_sentiment` | Data | bool | `true` | `true` | Include configured sentiment features | Methodological |
| `feature_cols` | Derived | list[str] | Derived | Derived | Final input feature order | Derived |
| `feature_dim` | Derived | int | Derived after loading | Derived after loading | Effective number of input features | Derived |
| `sentiment_path` | Data | path/null | `./NVDA_daily_sentiment.csv` | Same | Optional explicit daily sentiment CSV | Methodological |
| `train_end_date` | Data | ISO date/null | `2024-12-31` | Same | Inclusive end of the train/validation source period | Methodological |
| `test_start_date` | Data | ISO date | `2025-01-01` | Same | First held-out test timestamp | Methodological |
| `data_end_date` | Data | ISO date/null | `2026-01-01` | Same | Inclusive maximum timestamp loaded | Methodological |
| `validation_fraction` | Data | float | `0.05` | `0.05` | Trailing fraction of the pre-test period reserved for validation | Methodological |
| `series_split_size` | Data | int | `20` | — | Time steps in each SSL pretraining window | Methodological |
| `patch_size` | Data/Model | int | `5` | `5` | Time steps per historical input patch | Methodological |
| `pretrain_stride` | Data | int | `5` | — | Start-index stride between pretraining windows | Methodological |
| `sampling_mode` | Data | str | `sliding_window` | `sliding_window` | Sliding windows or non-overlapping temporal segments | Methodological |
| `context_size` | Data | int | — | `12` | Historical patches used downstream (`60` time steps) | Methodological |
| `eval_stride` | Data | int | — | `5` | Start-index stride between downstream samples | Methodological |
| `target_col` | Data | str | `Close` | `Close` | Raw column used to construct forecast labels | Methodological |
| `target_feature_index` | Data | int | `0` | `0` | Selected feature index for value-space targets | Methodological |
| `feature_transform` | Preprocessing | str | `raw` | `raw` | `raw` or causal `return` representation | Methodological |
| `normalization` | Preprocessing | str | `train_zscore` | `window_return` | Input normalization mode | Methodological |
| `normalization_stats` | Preprocessing | dict/null | Derived | `null`/checkpoint | Train-only fitted state reused by validation/test | Derived |
| `sentiment_normalization` | Preprocessing | str | `none` | `none` | `none` or selective `train_zscore` for derived sentiment channels | Methodological |
| `sentiment_normalization_stats` | Preprocessing | dict/null | Derived | `null`/checkpoint | Per-stock training-only sentiment state | Derived |
| `robust_zscore_clip` | Preprocessing | float/null | `null` | `null` | Optional symmetric clip after robust scaling | Methodological |
| `market_data` | Preprocessing | str/path/null | `null` | `null` | Optional aligned market series for market/excess features | Methodological |
| `mask_strategy` | Masking | str | `random` | `random` | `random`, `local_long`, `future_block`, or `causal_multiblock` | Methodological |
| `mask_ratio` | Masking | float | `0.7` | `0.7` | Random-strategy target fraction | Methodological |
| `ratio_patches` | Compatibility | int | `10` | `10` | Legacy checkpoint-name field; does not select masks | Runtime identity |
| `mae_window_patches` | Masking | int | `1` | `1` | Local MAE target width for `local_long` | Methodological |
| `jepa_gap_patches` | Masking | int | `4` | `4` | Offset from local MAE anchor to long JEPA target | Methodological |
| `jepa_target_patches` | Masking | int | `4` | `4` | JEPA target width for `local_long` | Methodological |
| `anchor_strategy` | Masking | str | `random` | — | Random or fixed structured-mask anchor | Methodological |
| `fixed_anchor` | Masking | int | `0` | — | Anchor used when `anchor_strategy=fixed` | Methodological |
| `future_target_patches` | Masking | int | `4` | `4` | Target width for `future_block` | Methodological |
| `causal_num_blocks` | Masking | int | `2` | `2` | Number of future target blocks | Methodological |
| `causal_block_patches` | Masking | int | `2` | `2` | Patches per future block | Methodological |
| `causal_block_gap_patches` | Masking | int | `1` | `1` | Gap between future blocks | Methodological |
| `encoder_embed_dim` | Model | int | `256` | — | SSL encoder latent width | Methodological |
| `encoder_nhead` | Model | int | `2` | — | SSL encoder attention heads | Methodological |
| `encoder_num_layers` | Model | int | `1` | — | SSL encoder transformer blocks | Methodological |
| `encoder_kernel_size` | Model | int | `3` | — | SSL tokenizer convolution kernel | Methodological |
| `encoder_embed_bias` | Model | bool | `true` | — | Tokenizer projection bias | Methodological |
| `predictor_embed` | Model | int | `128` | — | JEPA predictor latent width | Methodological |
| `predictor_nhead` | Model | int | `2` | — | Predictor attention heads | Methodological |
| `predictor_num_layers` | Model | int | `1` | — | Predictor transformer blocks | Methodological |
| `decoder_type` | Model | str | `residual_mlp` | `residual_mlp` | MAE/downstream decoder family | Methodological |
| `decoder_hidden_dim` | Model | int | `128` | `128` | Decoder hidden width | Methodological |
| `decoder_num_layers` | Model | int | `2` | `2` | MLP decoder hidden layers | Methodological |
| `decoder_dropout` | Model | float | `0.1` | `0.1` | Decoder dropout | Methodological |
| `pretrain_encoder_*` | Model | mixed | — | `256 / 2 / 1 / 3 / true` | Downstream reconstruction of the pretrained encoder architecture | Methodological |
| `embed_dim`, `nhead`, `num_layers` | Model | int | — | `128 / 2 / 1` | Non-pretrained downstream transformer settings | Methodological |
| `batch_size` | Training | int | `32` | `32` | Optimization batch size | Training |
| `lr` | Training | float | `1e-5` | `1e-3` | Stage optimizer learning rate | Training |
| `end_lr` | Training | float | `1e-6` | — | Final pretraining scheduler learning rate | Training |
| `num_epochs` | Training | int | `2001` | `501` | Maximum optimization epochs | Training |
| `ema_momentum` | Training | float | `0.998` | — | Target-encoder EMA base momentum | Methodological |
| `lambda_jepa` | Training | float | `1.0` | `1.0` identity | JEPA loss weight | Methodological |
| `lambda_mae` | Training | float | `0.5` | `1.0` identity | MAE loss weight | Methodological |
| `jepa_loss` | Training | str | `mse` | — | JEPA distance (`mse`, `l1`, `smooth_l1`) | Methodological |
| `mae_loss` | Training | str | `mse` | — | Reconstruction distance | Methodological |
| `clip_grad` | Training | float | `1` | — | Gradient norm clipping threshold | Training |
| `validation_interval` | Training | int | `10` | — | Epochs between SSL validation passes | Training |
| `validation_max_batches` | Training | int/null | `null` | — | Optional validation-batch cap | Runtime |
| `checkpoint_save` | Training | int | `500` | — | Periodic checkpoint interval | Runtime |
| `checkpoint_print` | Training | int | `30` | — | Training log interval | Runtime |
| `fine_tune_encoder` | Training | bool | — | `true` | Fine-tune rather than freeze the pretrained encoder | Methodological |
| `encoder_finetune_lr` | Training | float | — | `1e-5` | Encoder parameter-group learning rate | Training |
| `trend_weight` | Training | float | — | `0.001` | Directional auxiliary loss weight | Methodological |
| `forecast_target` | Evaluation | str | — | `value` | Downstream label definition | Methodological |
| `forecast_horizon` | Evaluation | int/null | — | `null` → `patch_size` | Independent downstream target/output width | Methodological |
| `evaluation_split` | Evaluation | str | — | `test` | `validation` for candidate selection or `test` for the frozen final run | Experimental protocol |
| `eval_forecast_target` | Evaluation | str | `relative_return` | — | Target used by automatic post-pretrain evaluation | Methodological |
| `eval_type` | Evaluation | str | — | `last` | Context pooling rule | Methodological |
| `run_eval` | Evaluation | bool | `true` | — | Run downstream evaluation after pretraining | Runtime |
| `eval_use_best` | Evaluation | bool | `true` | — | Select best-validation checkpoint for automatic evaluation | Training protocol |
| `eval_num_epochs` | Evaluation | int | `501` | — | Downstream epochs in automatic evaluation | Training |
| `checkpoint_selection` | Runtime | str | — | `last` | Best, last, epoch, or explicit-path checkpoint selection | Runtime |
| `pretrain_encoder_weights` | Runtime | str | — | `ema` | EMA or online encoder loaded downstream | Methodological |
| `seed` | Runtime | int | `42` | `42` | Python, NumPy, Torch, and CUDA seed | Runtime |
| `deterministic` | Runtime | bool | `true` | — | Request deterministic Torch algorithms | Runtime |
| `resume_from` | Runtime | path/null | `null` | — | Full-state checkpoint used to resume | Runtime |
| `max_batches_per_epoch` | Runtime | int/null | `null` | — | Optional smoke/debug batch cap | Runtime |
| `save_final` | Runtime | bool | `true` | — | Save final pretraining checkpoint | Runtime |
| `path_save` | Runtime | path | Derived | `./logs/output_model/` | Checkpoint root/stem | Runtime |
| `results_dir` | Runtime | path | `eval_results_dir` below | `./results` | Metrics and plot output root | Runtime |

## Dependencies and validation

- `series_split_size % patch_size == 0`; therefore pretraining
  `num_patches = series_split_size / patch_size`.
- Downstream historical length is `context_size * patch_size`.
  `forecast_horizon` must be positive and defaults to `patch_size` when omitted.
- `0 < mask_ratio < 1`, all lengths and batch sizes are positive, and attention
  embedding widths must be divisible by their head counts.
- `train_end_date < test_start_date <= data_end_date` when the optional dates
  are present. The validation fraction is in `[0, 1)`.
- `local_long` requires `mae_window_patches < jepa_gap_patches` and enough total
  patches for the JEPA target. Future strategies require at least one context
  patch before all target blocks.
- `feature_transform=return` cannot be combined with `window_return`, because
  the input is already expressed as returns.
- `forecast_target=excess_log_return` requires `market_data`.
- Validation and test reuse normalization statistics fitted only on the
  chronological training split. The final test period is never used for
  fitting, early stopping, or configuration selection.
- Chapter 5 candidate configs use `evaluation_split=validation` and
  `checkpoint.selection.mode=best`. The selector consumes only
  `validation_metrics.json`; the emitted frozen config switches to
  `evaluation_split=test` for the single held-out evaluation.

## Compatibility behavior

`--feature-cols` is retained because existing scripts and checkpoints use it.
If supplied without a sentiment flag, the list is honored exactly and
`use_sentiment` is inferred from the selected sentiment names. An explicit
`--no-sentiment` removes known sentiment columns; an explicit
`--use-sentiment` appends configured sentiment columns.

`ratio_patches` is retained only in historical checkpoint names. Actual patch
count comes from the window and patch sizes, and actual random target count
comes from `mask_ratio`.
