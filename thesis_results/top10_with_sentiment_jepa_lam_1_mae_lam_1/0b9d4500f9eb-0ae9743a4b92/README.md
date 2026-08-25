# TS-JEPA thesis result manifest

This directory is generated exclusively from saved experiment artifacts; no model training is performed.

## Analysis scope and coverage

- Config: `config/experiments/top10_with_sentiment_jepa_lam_1_mae_lam_1.json`
- Result root: `results/top10_with_sentiment_jepa_lam_1_mae_lam_1`
- Expected equities: NVDA, AVGO, COST
- Expected seeds: 42, 46, 50
- Strategies: random, local_long
- Canonical baseline/GRU strategy: `random`
- Canonical rows: 54
- Audit issues: 0 errors and 18 warnings

| method | available_runs | expected_runs | coverage_pct |
| --- | --- | --- | --- |
| Shared-target JEPA--MAE | 9 | 9 | 100 |
| Local-MAE/Long-JEPA | 9 | 9 | 100 |
| GRU | 9 | 9 | 100 |
| Naive-last | 9 | 9 | 100 |
| Drift | 9 | 9 | 100 |
| Mean-context | 9 | 9 | 100 |

`missing_runs.csv` is authoritative for missing stocks, seeds, methods, duplicate reruns, conflicting configurations, non-finite values, test-target mismatches, and deterministic-baseline inconsistencies. Incomplete coverage is never silently imputed.

## Source data and exclusions

- `results/top10_with_sentiment_jepa_lam_1_mae_lam_1/local_long/AVGO/seed_42/last_model_comparison_20260825_093110.csv`
- `results/top10_with_sentiment_jepa_lam_1_mae_lam_1/local_long/AVGO/seed_46/last_model_comparison_20260825_093534.csv`
- `results/top10_with_sentiment_jepa_lam_1_mae_lam_1/local_long/AVGO/seed_50/last_model_comparison_20260825_093953.csv`
- `results/top10_with_sentiment_jepa_lam_1_mae_lam_1/local_long/COST/seed_42/last_model_comparison_20260825_094415.csv`
- `results/top10_with_sentiment_jepa_lam_1_mae_lam_1/local_long/COST/seed_46/last_model_comparison_20260825_094832.csv`
- `results/top10_with_sentiment_jepa_lam_1_mae_lam_1/local_long/COST/seed_50/last_model_comparison_20260825_095252.csv`
- `results/top10_with_sentiment_jepa_lam_1_mae_lam_1/local_long/NVDA/seed_42/last_model_comparison_20260825_091807.csv`
- `results/top10_with_sentiment_jepa_lam_1_mae_lam_1/local_long/NVDA/seed_46/last_model_comparison_20260825_092228.csv`
- `results/top10_with_sentiment_jepa_lam_1_mae_lam_1/local_long/NVDA/seed_50/last_model_comparison_20260825_092650.csv`
- `results/top10_with_sentiment_jepa_lam_1_mae_lam_1/random/AVGO/seed_42/last_model_comparison_20260825_093114.csv`
- `results/top10_with_sentiment_jepa_lam_1_mae_lam_1/random/AVGO/seed_46/last_model_comparison_20260825_093533.csv`
- `results/top10_with_sentiment_jepa_lam_1_mae_lam_1/random/AVGO/seed_50/last_model_comparison_20260825_093956.csv`
- `results/top10_with_sentiment_jepa_lam_1_mae_lam_1/random/COST/seed_42/last_model_comparison_20260825_094416.csv`
- `results/top10_with_sentiment_jepa_lam_1_mae_lam_1/random/COST/seed_46/last_model_comparison_20260825_094836.csv`
- `results/top10_with_sentiment_jepa_lam_1_mae_lam_1/random/COST/seed_50/last_model_comparison_20260825_095253.csv`
- `results/top10_with_sentiment_jepa_lam_1_mae_lam_1/random/NVDA/seed_42/last_model_comparison_20260825_091811.csv`
- `results/top10_with_sentiment_jepa_lam_1_mae_lam_1/random/NVDA/seed_46/last_model_comparison_20260825_092232.csv`
- `results/top10_with_sentiment_jepa_lam_1_mae_lam_1/random/NVDA/seed_50/last_model_comparison_20260825_092652.csv`

Excluded inventory rows: 36.
- duplicate deterministic baseline; reference strategy selected: 27
- alternate strategy-specific GRU; reference strategy selected: 9

The inventory selects the latest timestamp only for duplicate bundles. A duplicate with multiple recoverable configuration signatures is an error. Strategy-specific GRU rows outside the configured reference strategy and duplicate deterministic baselines are retained in `run_inventory.csv` but excluded from the canonical dataset.

## Metrics and direction-accuracy audit

MSE and MAE are recomputed over all saved rolling-step × horizon values whenever score files exist. Stored summary values are checked against those reconstructions.

Direction accuracy uses `project_within_trajectory_v1: sign of consecutive forecast-horizon differences equals sign of consecutive target differences; relative-return paths additionally include the known zero origin; cumulative/excess log-return targets compare the binary indicators (forecast > 0) and (target > 0) at each horizon`. The identical implementation is applied to learned models and baselines. Naive-last and mean-context therefore have valid direction scores when trajectories are saved; a constant predicted value path generally produces zero-valued predicted differences for a value target, which only count as correct when the corresponding true difference is also zero. Unsupported values remain missing.

All reported values remain in the saved target space (`normalization`, `forecast_target`, and `target_definition` are preserved in `all_runs_tidy.csv`). No conversion to absolute prices is inferred.

## Aggregation and statistical procedure

For each method, metrics are first averaged over seeds within an equity. Overall metrics and ranks are then averaged over equities. Variability in the main table is the standard deviation across equity-level seed means.

Paired differences use Δ = method − naive-last, so negative values favour the method. Relative improvement is `100 × (naive − method) / naive`, so positive values favour the method. Pairing requires the same equity, seed, strategy-specific run bundle, target definition, normalization, metric definition, horizon, and saved target signature whenever available.

Primary inference averages seed-level paired differences within each equity and uses equities as the statistical units. The 95% interval is a percentile bootstrap that resamples equity-level means. The Wilcoxon result is an exact two-sided signed-rank sign-permutation test (zero differences removed); rank-biserial correlation is signed in Δ coordinates, so a negative effect favours the model. P-values are Holm-adjusted separately for the three learned-model MSE and MAE comparisons. Seed-level win counts and distribution figures are descriptive only.

## Paired results snapshot

| model | mean_delta_mse | mse_ci_low | mse_ci_high | mse_holm_p_value | mse_stock_wins | mse_run_wins | mse_run_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Shared-target JEPA--MAE | 0.001216 | 5.574e-05 | 0.002831 | 0.75 | 0 | 0 | 9 |
| Local-MAE/Long-JEPA | 0.002272 | 0.0002216 | 0.004846 | 0.75 | 0 | 0 | 9 |
| GRU | 0.0008393 | 0.0001274 | 0.001799 | 0.75 | 0 | 0 | 9 |

## Representative trajectory

- Stock: NVDA
- Seed: 42
- Rolling step: 0
- Target dates: 2025-04-01T00:00:00, 2025-04-02T00:00:00, 2025-04-03T00:00:00, 2025-04-04T00:00:00, 2025-04-07T00:00:00
- Selection rule: Use NVDA when it has complete saved predictions (otherwise the alphabetically first complete stock); select the Shared-target seed whose overall test MSE is closest to that stock's median, breaking ties by smaller seed; plot the first saved rolling step.

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
| figures/fig_representative_prediction_trajectory.pdf | generated | Publication-quality thesis figure |
| figures/fig_representative_prediction_trajectory.png | generated | Publication-quality thesis figure |
| figures/diagnostics/diagnostic_downstream_training_loss.pdf | generated | Publication-quality thesis figure |
| figures/diagnostics/diagnostic_downstream_training_loss.png | generated | Publication-quality thesis figure |
| figures/diagnostics/diagnostic_downstream_validation_mse.pdf | generated | Publication-quality thesis figure |
| figures/diagnostics/diagnostic_downstream_validation_mse.png | generated | Publication-quality thesis figure |
| figures/diagnostics/diagnostic_pretraining_losses.pdf | omitted | Pre-training JEPA/MAE/total histories were unavailable. |
| data/representative_example.json | generated | Deterministic representative-example selection metadata |
| README.md | generated | Methodology, coverage, exclusions, interpretation limits, and reproduction command |
| artifact_manifest.csv | generated | Mapping from every planned output artifact to its source data and status |
| analysis_metadata.json | generated | Machine-readable analysis provenance |

## Reproduction command

```bash
conda run --no-capture-output -n ts-jepa python analyze_thesis_results.py \
  --config config/experiments/top10_with_sentiment_jepa_lam_1_mae_lam_1.json \
  --reference-strategy random \
  --bootstrap-samples 20000 \
  --analysis-seed 20260822
```

The analysis bootstrap seed and sample count are recorded in the command and output metadata. PDF figures are vector outputs; matching PNG files are previews.

## Git publication

- Immutable snapshot: `0b9d4500f9eb-0ae9743a4b92`
- Full raw experiment outputs are intentionally excluded from Git.
- `SHA256SUMS` verifies every published file.
- Files omitted by publication policy: 0. See `publication_manifest.csv`.
- Large/raw artifacts should be attached to the matching GitHub Release.
