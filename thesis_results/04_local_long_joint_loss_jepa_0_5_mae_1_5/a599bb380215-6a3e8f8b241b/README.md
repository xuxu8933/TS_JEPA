# TS-JEPA thesis result manifest

This directory is generated exclusively from saved experiment artifacts; no model training is performed.

## Analysis scope and coverage

- Config: `config/experiments/chapter5_candidates/04_local_long_joint_loss_jepa_0_5_mae_1_5.json`
- Result root: `results/04_local_long_joint_loss_jepa_0_5_mae_1_5`
- Expected equities: NVDA, AAPL, AVGO, TSLA, WMT
- Expected seeds: 42, 44, 46
- Strategies: local_long
- Canonical baseline/GRU strategy: `local_long`
- Canonical rows: 75
- Audit issues: 0 errors and 0 warnings

| method | available_runs | expected_runs | coverage_pct |
| --- | --- | --- | --- |
| Local-MAE/Long-JEPA | 15 | 15 | 100 |
| GRU | 15 | 15 | 100 |
| Naive-last | 15 | 15 | 100 |
| Drift | 15 | 15 | 100 |
| Mean-context | 15 | 15 | 100 |

`missing_runs.csv` is authoritative for missing stocks, seeds, methods, duplicate reruns, conflicting configurations, non-finite values, test-target mismatches, and deterministic-baseline inconsistencies. Incomplete coverage is never silently imputed.

## Source data and exclusions

- `results/04_local_long_joint_loss_jepa_0_5_mae_1_5/local_long/AAPL/seed_42/last_model_comparison_20260903_012042.csv`
- `results/04_local_long_joint_loss_jepa_0_5_mae_1_5/local_long/AAPL/seed_44/last_model_comparison_20260903_012506.csv`
- `results/04_local_long_joint_loss_jepa_0_5_mae_1_5/local_long/AAPL/seed_46/last_model_comparison_20260903_012504.csv`
- `results/04_local_long_joint_loss_jepa_0_5_mae_1_5/local_long/AVGO/seed_42/last_model_comparison_20260903_012925.csv`
- `results/04_local_long_joint_loss_jepa_0_5_mae_1_5/local_long/AVGO/seed_44/last_model_comparison_20260903_012927.csv`
- `results/04_local_long_joint_loss_jepa_0_5_mae_1_5/local_long/AVGO/seed_46/last_model_comparison_20260903_013346.csv`
- `results/04_local_long_joint_loss_jepa_0_5_mae_1_5/local_long/NVDA/seed_42/last_model_comparison_20260903_011621.csv`
- `results/04_local_long_joint_loss_jepa_0_5_mae_1_5/local_long/NVDA/seed_44/last_model_comparison_20260903_011621.csv`
- `results/04_local_long_joint_loss_jepa_0_5_mae_1_5/local_long/NVDA/seed_46/last_model_comparison_20260903_012042.csv`
- `results/04_local_long_joint_loss_jepa_0_5_mae_1_5/local_long/TSLA/seed_42/last_model_comparison_20260903_013348.csv`
- `results/04_local_long_joint_loss_jepa_0_5_mae_1_5/local_long/TSLA/seed_44/last_model_comparison_20260903_013805.csv`
- `results/04_local_long_joint_loss_jepa_0_5_mae_1_5/local_long/TSLA/seed_46/last_model_comparison_20260903_013808.csv`
- `results/04_local_long_joint_loss_jepa_0_5_mae_1_5/local_long/WMT/seed_42/last_model_comparison_20260903_014225.csv`
- `results/04_local_long_joint_loss_jepa_0_5_mae_1_5/local_long/WMT/seed_44/last_model_comparison_20260903_014227.csv`
- `results/04_local_long_joint_loss_jepa_0_5_mae_1_5/local_long/WMT/seed_46/last_model_comparison_20260903_014625.csv`

Excluded inventory rows: 0.

The inventory selects the latest timestamp only for duplicate bundles. A duplicate with multiple recoverable configuration signatures is an error. Strategy-specific GRU rows outside the configured reference strategy and duplicate deterministic baselines are retained in `run_inventory.csv` but excluded from the canonical dataset.

## Metrics and direction-accuracy audit

RMSE is recomputed over all saved rolling-step × horizon values whenever score files exist. Stored summary values are checked against that reconstruction.

Direction accuracy uses `project_within_trajectory_v1: sign of consecutive forecast-horizon differences equals sign of consecutive target differences; relative-return paths additionally include the known zero origin; cumulative/excess log-return targets compare the binary indicators (forecast > 0) and (target > 0) at each horizon`. The identical implementation is applied to learned models and baselines. Naive-last and mean-context therefore have valid direction scores when trajectories are saved; a constant predicted value path generally produces zero-valued predicted differences for a value target, which only count as correct when the corresponding true difference is also zero. Unsupported values remain missing.

All reported values remain in the saved target space (`normalization`, `forecast_target`, and `target_definition` are preserved in `all_runs_tidy.csv`). No conversion to absolute prices is inferred.

## Aggregation and statistical procedure

For each method, metrics are first averaged over seeds within an equity. Overall metrics and ranks are then averaged over equities. Variability in the main table is the standard deviation across equity-level seed means.

Paired differences use Δ = method − naive-last, so negative values favour the method. Relative improvement is `100 × (naive − method) / naive`, so positive values favour the method. Pairing requires the same equity, seed, strategy-specific run bundle, target definition, normalization, metric definition, horizon, and saved target signature whenever available.

Primary inference averages seed-level paired differences within each equity and uses equities as the statistical units. The 95% interval is a percentile bootstrap that resamples equity-level means. The Wilcoxon result is an exact two-sided signed-rank sign-permutation test (zero differences removed); rank-biserial correlation is signed in Δ coordinates, so a negative effect favours the model. RMSE p-values are Holm-adjusted across the three learned-model comparisons. Seed-level win counts and distribution figures are descriptive only.

The separate Shared-vs-Local comparison first retains compatible seeds matched by equity, seed, target definition, normalization, metric definition, forecast horizon, test period, and saved target signature. Shared and Local RMSE are averaged over the same matched seeds within each equity. A two-sided paired Student t-test and signed Cohen's dz then use the resulting equity-level values only, with Δ = Shared − Local; negative values favour Shared. No Direction Accuracy test or multiple-comparison correction is applied to this separate comparison.

## Paired results snapshot

| model | mean_delta_rmse | rmse_ci_low | rmse_ci_high | rmse_holm_p_value | rmse_stock_wins | rmse_run_wins | rmse_run_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Local-MAE/Long-JEPA | -0.003517 | -0.008428 | 0.0003453 | 0.3125 | 4 | 10 | 15 |
| GRU | -0.002358 | -0.003495 | -0.001449 | 0.125 | 5 | 15 | 15 |

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
| figures/fig_paired_rmse_forest.pdf | generated | Publication-quality thesis figure |
| figures/fig_paired_rmse_forest.png | generated | Publication-quality thesis figure |
| figures/fig_relative_rmse_heatmap.pdf | generated | Publication-quality thesis figure |
| figures/fig_relative_rmse_heatmap.png | generated | Publication-quality thesis figure |
| figures/fig_direction_accuracy_heatmap.pdf | generated | Publication-quality thesis figure |
| figures/fig_direction_accuracy_heatmap.png | generated | Publication-quality thesis figure |
| figures/fig_rmse_by_horizon.pdf | generated | Publication-quality thesis figure |
| figures/fig_rmse_by_horizon.png | generated | Publication-quality thesis figure |
| figures/fig_direction_by_horizon.pdf | generated | Publication-quality thesis figure |
| figures/fig_direction_by_horizon.png | generated | Publication-quality thesis figure |
| figures/fig_seed_level_delta_rmse_distribution.pdf | generated | Publication-quality thesis figure |
| figures/fig_seed_level_delta_rmse_distribution.png | generated | Publication-quality thesis figure |
| figures/diagnostics/diagnostic_downstream_training_loss.pdf | generated | Publication-quality thesis figure |
| figures/diagnostics/diagnostic_downstream_training_loss.png | generated | Publication-quality thesis figure |
| figures/diagnostics/diagnostic_downstream_validation_rmse.pdf | generated | Publication-quality thesis figure |
| figures/diagnostics/diagnostic_downstream_validation_rmse.png | generated | Publication-quality thesis figure |
| tables/table_shared_vs_local.tex | omitted | Compatible Shared-vs-Local stock pairs were unavailable. |
| figures/fig_representative_prediction_trajectory.pdf | omitted | Complete aligned raw predictions were unavailable. |
| figures/diagnostics/diagnostic_pretraining_losses.pdf | omitted | Pre-training JEPA/MAE/total histories were unavailable. |
| README.md | generated | Methodology, coverage, exclusions, interpretation limits, and reproduction command |
| artifact_manifest.csv | generated | Mapping from every planned output artifact to its source data and status |
| analysis_metadata.json | generated | Machine-readable analysis provenance |

## Reproduction command

```bash
conda run --no-capture-output -n ts-jepa python analyze_thesis_results.py \
  --config config/experiments/chapter5_candidates/04_local_long_joint_loss_jepa_0_5_mae_1_5.json \
  --reference-strategy local_long \
  --bootstrap-samples 20000 \
  --analysis-seed 20260822
```

The analysis bootstrap seed and sample count are recorded in the command and output metadata. PDF figures are vector outputs; matching PNG files are previews.

## Git publication

- Immutable snapshot: `a599bb380215-6a3e8f8b241b`
- Full raw experiment outputs are intentionally excluded from Git.
- `SHA256SUMS` verifies every published file.
- Files omitted by publication policy: 0. See `publication_manifest.csv`.
- Large/raw artifacts should be attached to the matching GitHub Release.
