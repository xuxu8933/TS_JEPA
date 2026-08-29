# TS-JEPA thesis result manifest

This directory is generated exclusively from saved experiment artifacts; no model training is performed.

## Analysis scope and coverage

- Config: `config/experiments/normalization_pilot_train_zscore.json`
- Result root: `results/normalization_pilot_train_zscore`
- Expected equities: NVDA, AAPL, AVGO, TSLA, WMT
- Expected seeds: 42, 44, 46, 48, 50
- Strategies: random
- Canonical baseline/GRU strategy: `random`
- Canonical rows: 125
- Audit issues: 0 errors and 0 warnings

| method | available_runs | expected_runs | coverage_pct |
| --- | --- | --- | --- |
| Shared-target JEPA--MAE | 25 | 25 | 100 |
| GRU | 25 | 25 | 100 |
| Naive-last | 25 | 25 | 100 |
| Drift | 25 | 25 | 100 |
| Mean-context | 25 | 25 | 100 |

`missing_runs.csv` is authoritative for missing stocks, seeds, methods, duplicate reruns, conflicting configurations, non-finite values, test-target mismatches, and deterministic-baseline inconsistencies. Incomplete coverage is never silently imputed.

## Source data and exclusions

- `results/normalization_pilot_train_zscore/random/AAPL/seed_42/last_model_comparison_20260828_085617.csv`
- `results/normalization_pilot_train_zscore/random/AAPL/seed_44/last_model_comparison_20260828_085718.csv`
- `results/normalization_pilot_train_zscore/random/AAPL/seed_46/last_model_comparison_20260828_085817.csv`
- `results/normalization_pilot_train_zscore/random/AAPL/seed_48/last_model_comparison_20260828_085918.csv`
- `results/normalization_pilot_train_zscore/random/AAPL/seed_50/last_model_comparison_20260828_090016.csv`
- `results/normalization_pilot_train_zscore/random/AVGO/seed_42/last_model_comparison_20260828_090116.csv`
- `results/normalization_pilot_train_zscore/random/AVGO/seed_44/last_model_comparison_20260828_090215.csv`
- `results/normalization_pilot_train_zscore/random/AVGO/seed_46/last_model_comparison_20260828_090315.csv`
- `results/normalization_pilot_train_zscore/random/AVGO/seed_48/last_model_comparison_20260828_090414.csv`
- `results/normalization_pilot_train_zscore/random/AVGO/seed_50/last_model_comparison_20260828_090513.csv`
- `results/normalization_pilot_train_zscore/random/NVDA/seed_42/last_model_comparison_20260828_085118.csv`
- `results/normalization_pilot_train_zscore/random/NVDA/seed_44/last_model_comparison_20260828_085218.csv`
- `results/normalization_pilot_train_zscore/random/NVDA/seed_46/last_model_comparison_20260828_085317.csv`
- `results/normalization_pilot_train_zscore/random/NVDA/seed_48/last_model_comparison_20260828_085418.csv`
- `results/normalization_pilot_train_zscore/random/NVDA/seed_50/last_model_comparison_20260828_085517.csv`
- `results/normalization_pilot_train_zscore/random/TSLA/seed_42/last_model_comparison_20260828_090611.csv`
- `results/normalization_pilot_train_zscore/random/TSLA/seed_44/last_model_comparison_20260828_090712.csv`
- `results/normalization_pilot_train_zscore/random/TSLA/seed_46/last_model_comparison_20260828_090812.csv`
- `results/normalization_pilot_train_zscore/random/TSLA/seed_48/last_model_comparison_20260828_090912.csv`
- `results/normalization_pilot_train_zscore/random/TSLA/seed_50/last_model_comparison_20260828_091011.csv`
- `results/normalization_pilot_train_zscore/random/WMT/seed_42/last_model_comparison_20260828_091107.csv`
- `results/normalization_pilot_train_zscore/random/WMT/seed_44/last_model_comparison_20260828_091208.csv`
- `results/normalization_pilot_train_zscore/random/WMT/seed_46/last_model_comparison_20260828_091308.csv`
- `results/normalization_pilot_train_zscore/random/WMT/seed_48/last_model_comparison_20260828_091407.csv`
- `results/normalization_pilot_train_zscore/random/WMT/seed_50/last_model_comparison_20260828_091506.csv`

Excluded inventory rows: 0.

The inventory selects the latest timestamp only for duplicate bundles. A duplicate with multiple recoverable configuration signatures is an error. Strategy-specific GRU rows outside the configured reference strategy and duplicate deterministic baselines are retained in `run_inventory.csv` but excluded from the canonical dataset.

## Metrics and direction-accuracy audit

MSE and MAE are recomputed over all saved rolling-step × horizon values whenever score files exist. Stored summary values are checked against those reconstructions.

Direction accuracy uses `project_within_trajectory_v1: sign of consecutive forecast-horizon differences equals sign of consecutive target differences; relative-return paths additionally include the known zero origin; cumulative/excess log-return targets compare the binary indicators (forecast > 0) and (target > 0) at each horizon`. The identical implementation is applied to learned models and baselines. Naive-last and mean-context therefore have valid direction scores when trajectories are saved; a constant predicted value path generally produces zero-valued predicted differences for a value target, which only count as correct when the corresponding true difference is also zero. Unsupported values remain missing.

All reported values remain in the saved target space (`normalization`, `forecast_target`, and `target_definition` are preserved in `all_runs_tidy.csv`). No conversion to absolute prices is inferred.

## Aggregation and statistical procedure

For each method, metrics are first averaged over seeds within an equity. Overall metrics and ranks are then averaged over equities. Variability in the main table is the standard deviation across equity-level seed means.

Paired differences use Δ = method − naive-last, so negative values favour the method. Relative improvement is `100 × (naive − method) / naive`, so positive values favour the method. Pairing requires the same equity, seed, strategy-specific run bundle, target definition, normalization, metric definition, horizon, and saved target signature whenever available.

Primary inference averages seed-level paired differences within each equity and uses equities as the statistical units. The 95% interval is a percentile bootstrap that resamples equity-level means. The Wilcoxon result is an exact two-sided signed-rank sign-permutation test (zero differences removed); rank-biserial correlation is signed in Δ coordinates, so a negative effect favours the model. P-values are Holm-adjusted separately for the three learned-model MSE and MAE comparisons. Seed-level win counts and distribution figures are descriptive only.

The separate Shared-vs-Local comparison first retains compatible seeds matched by equity, seed, target definition, normalization, metric definition, forecast horizon, test period, and saved target signature. Shared and Local MSE/MAE are averaged over the same matched seeds within each equity. A two-sided paired Student t-test and signed Cohen's dz then use the resulting equity-level values only, with Δ = Shared − Local; negative values favour Shared for these error metrics. No Direction Accuracy test or multiple-comparison correction is applied to this separate comparison.

## Paired results snapshot

| model | mean_delta_mse | mse_ci_low | mse_ci_high | mse_holm_p_value | mse_stock_wins | mse_run_wins | mse_run_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Shared-target JEPA--MAE | 0.001319 | 0.0008161 | 0.002059 | 0.125 | 0 | 0 | 25 |
| GRU | 0.0005014 | 9.616e-05 | 0.001166 | 0.125 | 0 | 4 | 25 |

## Shared-target vs Local-MAE/Long-JEPA snapshot

_None._

## Representative trajectory

Not generated: complete aligned raw predictions were unavailable.

## Interpretation limits

The Shared-target and Local-MAE/Long-JEPA rows are complete forecasting procedures, not causal ablations of JEPA, MAE, EMA, or pre-training. Unless separately controlled rows appear in the inventory, random-initialized Transformer, JEPA-only, MAE-only, and online-vs-EMA causal analyses cannot be produced. Decreasing pre-training or downstream loss demonstrates optimization behaviour only.

Forecast-horizon and qualitative figures are omitted when score-level predictions are absent. Pre-training JEPA/MAE/total-loss diagnostics are omitted when checkpoint validation histories or training histories are absent. These omissions are recorded in `artifact_manifest.csv`.

## Generated artifact manifest

| artifact | status | description |
| --- | --- | --- |
| data/all_runs_tidy.csv | generated | Canonical analysis dataset: all_runs_tidy.csv |
| data/run_inventory.csv | generated | Canonical analysis dataset: run_inventory.csv |
| data/missing_runs.csv | generated | Canonical analysis dataset: missing_runs.csv |
| data/coverage_summary.csv | generated | Canonical analysis dataset: coverage_summary.csv |
| data/configuration_inventory.csv | generated | Canonical analysis dataset: configuration_inventory.csv |
| data/predictions_tidy.csv | generated | Canonical analysis dataset: predictions_tidy.csv |
| data/paired_run_differences.csv | generated | Canonical analysis dataset: paired_run_differences.csv |
| data/paired_stock_differences.csv | generated | Canonical analysis dataset: paired_stock_differences.csv |
| data/stock_summary.csv | generated | Canonical analysis dataset: stock_summary.csv |
| data/overall_summary.csv | generated | Canonical analysis dataset: overall_summary.csv |
| data/paired_vs_naive.csv | generated | Canonical analysis dataset: paired_vs_naive.csv |
| data/paired_shared_vs_local.csv | generated | Canonical analysis dataset: paired_shared_vs_local.csv |
| data/relative_performance_by_stock.csv | generated | Canonical analysis dataset: relative_performance_by_stock.csv |
| data/horizon_metrics.csv | generated | Canonical analysis dataset: horizon_metrics.csv |
| tables/table_main_metrics.csv | generated | Thesis or appendix table |
| tables/table_main_metrics.tex | generated | Thesis or appendix table |
| tables/table_paired_vs_naive.csv | generated | Thesis or appendix table |
| tables/table_paired_vs_naive.tex | generated | Thesis or appendix table |
| tables/table_appendix_stock_metrics.csv | generated | Thesis or appendix table |
| tables/table_appendix_stock_metrics.tex | generated | Thesis or appendix table |
| tables/table_reproducibility.tex | generated | Thesis or appendix table |
| figures/fig_paired_mse_forest.pdf | generated | Publication-quality thesis figure |
| figures/fig_paired_mse_forest.png | generated | Publication-quality thesis figure |
| figures/fig_paired_mae_forest.pdf | generated | Publication-quality thesis figure |
| figures/fig_paired_mae_forest.png | generated | Publication-quality thesis figure |
| figures/fig_relative_mse_heatmap.pdf | generated | Publication-quality thesis figure |
| figures/fig_relative_mse_heatmap.png | generated | Publication-quality thesis figure |
| figures/fig_relative_mae_heatmap.pdf | generated | Publication-quality thesis figure |
| figures/fig_relative_mae_heatmap.png | generated | Publication-quality thesis figure |
| figures/fig_direction_accuracy_heatmap.pdf | generated | Publication-quality thesis figure |
| figures/fig_direction_accuracy_heatmap.png | generated | Publication-quality thesis figure |
| figures/fig_mse_by_horizon.pdf | generated | Publication-quality thesis figure |
| figures/fig_mse_by_horizon.png | generated | Publication-quality thesis figure |
| figures/fig_mae_by_horizon.pdf | generated | Publication-quality thesis figure |
| figures/fig_mae_by_horizon.png | generated | Publication-quality thesis figure |
| figures/fig_direction_by_horizon.pdf | generated | Publication-quality thesis figure |
| figures/fig_direction_by_horizon.png | generated | Publication-quality thesis figure |
| figures/fig_seed_level_delta_mse_distribution.pdf | generated | Publication-quality thesis figure |
| figures/fig_seed_level_delta_mse_distribution.png | generated | Publication-quality thesis figure |
| figures/diagnostics/diagnostic_downstream_training_loss.pdf | generated | Publication-quality thesis figure |
| figures/diagnostics/diagnostic_downstream_training_loss.png | generated | Publication-quality thesis figure |
| figures/diagnostics/diagnostic_downstream_validation_mse.pdf | generated | Publication-quality thesis figure |
| figures/diagnostics/diagnostic_downstream_validation_mse.png | generated | Publication-quality thesis figure |
| tables/table_shared_vs_local.tex | omitted | Compatible Shared-vs-Local stock pairs were unavailable. |
| figures/fig_representative_prediction_trajectory.pdf | omitted | Complete aligned raw predictions were unavailable. |
| figures/diagnostics/diagnostic_pretraining_losses.pdf | omitted | Pre-training JEPA/MAE/total histories were unavailable. |
| README.md | generated | Methodology, coverage, exclusions, interpretation limits, and reproduction command |
| artifact_manifest.csv | generated | Mapping from every planned output artifact to its source data and status |
| analysis_metadata.json | generated | Machine-readable analysis provenance |

## Reproduction command

```bash
conda run --no-capture-output -n ts-jepa python analyze_thesis_results.py \
  --config config/experiments/normalization_pilot_train_zscore.json \
  --reference-strategy random \
  --bootstrap-samples 20000 \
  --analysis-seed 20260822
```

The analysis bootstrap seed and sample count are recorded in the command and output metadata. PDF figures are vector outputs; matching PNG files are previews.

## Git publication

- Immutable snapshot: `d4be34e52029-beffffca1ced`
- Full raw experiment outputs are intentionally excluded from Git.
- `SHA256SUMS` verifies every published file.
- Files omitted by publication policy: 0. See `publication_manifest.csv`.
- Large/raw artifacts should be attached to the matching GitHub Release.
