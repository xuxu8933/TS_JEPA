# Chapter 5 publication figures

These figures are generated deterministically by `analysis/plot_chapter5_figures.py` from immutable, checksum-verified thesis-result snapshots. No experiment output is modified and no training is performed.

The stock set is AAPL, AVGO, NVDA, TSLA, and WMT; the seed set is 42, 44, and 46. Learned-model metrics are averaged over seeds within each stock and then across the five stock means. Error bars show the sample standard deviation across those five stock-level seed means; they are cross-stock dispersion, not confidence intervals.

Stage 01-02 use 12 validation origins (2024-10-02 to 2024-12-26). Stage 03-04 use 24 validation origins (2024-07-09 to 2024-12-26). They are not plotted as an additive learning curve.

## Figure 1 - Stage 00 model validation

Caption: Representative controlled-data diagnostic used to verify the implementation before the financial forecasting experiments.

- Snapshot: `thesis_results/00_dual_loss_smoke/e56db7d33c56`
- Inputs: `thesis_results/00_dual_loss_smoke/e56db7d33c56/summary.csv`, `thesis_results/00_dual_loss_smoke/e56db7d33c56/smoke_mnist_rows_reconstruction.png`
- Plotted diagnostic: MNIST-row model RMSE 0.111085; previous-row RMSE 0.188641.
- No stock, seed, or financial validation period applies. No uncertainty bar is shown.
- This is an implementation diagnostic, not financial forecasting evidence.

## Figure 2 - Stage 01 normalization comparison

- Snapshot: `thesis_results/01_preprocessing_window_return/568f317f7e2c-1b543b18e1bc`
  - `thesis_results/01_preprocessing_window_return/568f317f7e2c-1b543b18e1bc/provenance/experiment_config.json`
  - `thesis_results/01_preprocessing_window_return/568f317f7e2c-1b543b18e1bc/data/overall_summary.csv`
  - `thesis_results/01_preprocessing_window_return/568f317f7e2c-1b543b18e1bc/data/stock_summary.csv`
  - `thesis_results/01_preprocessing_window_return/568f317f7e2c-1b543b18e1bc/data/predictions_tidy.csv`
- Snapshot: `thesis_results/01_preprocessing_train_zscore/b643221e52a1-c018af4432c2`
  - `thesis_results/01_preprocessing_train_zscore/b643221e52a1-c018af4432c2/provenance/experiment_config.json`
  - `thesis_results/01_preprocessing_train_zscore/b643221e52a1-c018af4432c2/data/overall_summary.csv`
  - `thesis_results/01_preprocessing_train_zscore/b643221e52a1-c018af4432c2/data/stock_summary.csv`
  - `thesis_results/01_preprocessing_train_zscore/b643221e52a1-c018af4432c2/data/predictions_tidy.csv`

| Configuration | RMSE | Direction Accuracy |
|---|---:|---:|
| Window-relative | 0.052634 | 54.78% |
| Train z-score | 0.056598 | 53.56% |

Error bars: sample standard deviation across five stock-level seed means.

## Figure 3 - Stage 02 sentiment ablation

- Snapshot: `thesis_results/01_preprocessing_window_return/568f317f7e2c-1b543b18e1bc`
  - `thesis_results/01_preprocessing_window_return/568f317f7e2c-1b543b18e1bc/provenance/experiment_config.json`
  - `thesis_results/01_preprocessing_window_return/568f317f7e2c-1b543b18e1bc/data/overall_summary.csv`
  - `thesis_results/01_preprocessing_window_return/568f317f7e2c-1b543b18e1bc/data/stock_summary.csv`
  - `thesis_results/01_preprocessing_window_return/568f317f7e2c-1b543b18e1bc/data/predictions_tidy.csv`
- Snapshot: `thesis_results/02_sentiment_included/afc29d4b16b6-655ec105d05b`
  - `thesis_results/02_sentiment_included/afc29d4b16b6-655ec105d05b/provenance/experiment_config.json`
  - `thesis_results/02_sentiment_included/afc29d4b16b6-655ec105d05b/data/overall_summary.csv`
  - `thesis_results/02_sentiment_included/afc29d4b16b6-655ec105d05b/data/stock_summary.csv`
  - `thesis_results/02_sentiment_included/afc29d4b16b6-655ec105d05b/data/predictions_tidy.csv`

| Configuration | RMSE | Direction Accuracy |
|---|---:|---:|
| Market only | 0.052634 | 54.78% |
| Market + sentiment | 0.049247 | 52.33% |

The market-only row reuses the validated Stage 01 window-relative snapshot, as Stage 02 changed only sentiment inclusion. Error bars show the sample standard deviation across five stock-level seed means.

## Figure 4 - Stage 03 context-length sensitivity

- Snapshot: `thesis_results/03_shared_context_6_patches/49b848da7bb2-2609b9618086`
  - `thesis_results/03_shared_context_6_patches/49b848da7bb2-2609b9618086/provenance/experiment_config.json`
  - `thesis_results/03_shared_context_6_patches/49b848da7bb2-2609b9618086/data/overall_summary.csv`
  - `thesis_results/03_shared_context_6_patches/49b848da7bb2-2609b9618086/data/stock_summary.csv`
  - `thesis_results/03_shared_context_6_patches/49b848da7bb2-2609b9618086/data/predictions_tidy.csv`
- Snapshot: `thesis_results/03_shared_context_12_patches/afc29d4b16b6-8b70a34f4982`
  - `thesis_results/03_shared_context_12_patches/afc29d4b16b6-8b70a34f4982/provenance/experiment_config.json`
  - `thesis_results/03_shared_context_12_patches/afc29d4b16b6-8b70a34f4982/data/overall_summary.csv`
  - `thesis_results/03_shared_context_12_patches/afc29d4b16b6-8b70a34f4982/data/stock_summary.csv`
  - `thesis_results/03_shared_context_12_patches/afc29d4b16b6-8b70a34f4982/data/predictions_tidy.csv`
- Snapshot: `thesis_results/03_shared_context_24_patches/4f778fee2b9b-6803fe2fb974`
  - `thesis_results/03_shared_context_24_patches/4f778fee2b9b-6803fe2fb974/provenance/experiment_config.json`
  - `thesis_results/03_shared_context_24_patches/4f778fee2b9b-6803fe2fb974/data/overall_summary.csv`
  - `thesis_results/03_shared_context_24_patches/4f778fee2b9b-6803fe2fb974/data/stock_summary.csv`
  - `thesis_results/03_shared_context_24_patches/4f778fee2b9b-6803fe2fb974/data/predictions_tidy.csv`
- Snapshot: `thesis_results/03_local_long_context_6_patches/643dd2d957d2-00670fa6e96b`
  - `thesis_results/03_local_long_context_6_patches/643dd2d957d2-00670fa6e96b/provenance/experiment_config.json`
  - `thesis_results/03_local_long_context_6_patches/643dd2d957d2-00670fa6e96b/data/overall_summary.csv`
  - `thesis_results/03_local_long_context_6_patches/643dd2d957d2-00670fa6e96b/data/stock_summary.csv`
  - `thesis_results/03_local_long_context_6_patches/643dd2d957d2-00670fa6e96b/data/predictions_tidy.csv`
- Snapshot: `thesis_results/03_local_long_context_12_patches/a60306e1c2cc-41dcaf5bc18d`
  - `thesis_results/03_local_long_context_12_patches/a60306e1c2cc-41dcaf5bc18d/provenance/experiment_config.json`
  - `thesis_results/03_local_long_context_12_patches/a60306e1c2cc-41dcaf5bc18d/data/overall_summary.csv`
  - `thesis_results/03_local_long_context_12_patches/a60306e1c2cc-41dcaf5bc18d/data/stock_summary.csv`
  - `thesis_results/03_local_long_context_12_patches/a60306e1c2cc-41dcaf5bc18d/data/predictions_tidy.csv`
- Snapshot: `thesis_results/03_local_long_context_24_patches/8ac63bf9b1eb-cfc753ae10fe`
  - `thesis_results/03_local_long_context_24_patches/8ac63bf9b1eb-cfc753ae10fe/provenance/experiment_config.json`
  - `thesis_results/03_local_long_context_24_patches/8ac63bf9b1eb-cfc753ae10fe/data/overall_summary.csv`
  - `thesis_results/03_local_long_context_24_patches/8ac63bf9b1eb-cfc753ae10fe/data/stock_summary.csv`
  - `thesis_results/03_local_long_context_24_patches/8ac63bf9b1eb-cfc753ae10fe/data/predictions_tidy.csv`

| Configuration | RMSE | Direction Accuracy |
|---|---:|---:|
| Shared-Target, 6 patches | 0.049227 | 49.72% |
| Shared-Target, 12 patches | 0.047260 | 50.89% |
| Shared-Target, 24 patches | 0.049475 | 52.00% |
| Local-MAE/Long-JEPA, 6 patches | 0.049541 | 50.94% |
| Local-MAE/Long-JEPA, 12 patches | 0.048152 | 52.61% |
| Local-MAE/Long-JEPA, 24 patches | 0.049353 | 51.33% |

All six snapshots have identical 24-origin validation support. Error bars show the sample standard deviation across five stock-level seed means. Twelve patches correspond to 60 observations and are the selected Stage 03 context.

## Figure 5 - Stage 03 Shared-Target versus Local-MAE/Long-JEPA

- Snapshot: `thesis_results/03_shared_context_12_patches/afc29d4b16b6-8b70a34f4982`
  - `thesis_results/03_shared_context_12_patches/afc29d4b16b6-8b70a34f4982/provenance/experiment_config.json`
  - `thesis_results/03_shared_context_12_patches/afc29d4b16b6-8b70a34f4982/data/overall_summary.csv`
  - `thesis_results/03_shared_context_12_patches/afc29d4b16b6-8b70a34f4982/data/stock_summary.csv`
  - `thesis_results/03_shared_context_12_patches/afc29d4b16b6-8b70a34f4982/data/predictions_tidy.csv`
- Snapshot: `thesis_results/03_local_long_context_12_patches/a60306e1c2cc-41dcaf5bc18d`
  - `thesis_results/03_local_long_context_12_patches/a60306e1c2cc-41dcaf5bc18d/provenance/experiment_config.json`
  - `thesis_results/03_local_long_context_12_patches/a60306e1c2cc-41dcaf5bc18d/data/overall_summary.csv`
  - `thesis_results/03_local_long_context_12_patches/a60306e1c2cc-41dcaf5bc18d/data/stock_summary.csv`
  - `thesis_results/03_local_long_context_12_patches/a60306e1c2cc-41dcaf5bc18d/data/predictions_tidy.csv`

Each connected line is one stock after averaging its three seeds. The black diamond is the mean across the five stocks. No error bars are shown; the paired stock lines display the cross-stock variation directly.

| Configuration | RMSE | Direction Accuracy |
|---|---:|---:|
| Shared-Target, 12 patches | 0.047260 | 50.89% |
| Local-MAE/Long-JEPA, 12 patches | 0.048152 | 52.61% |

## Figure 6 - Stage 04 JEPA-MAE loss-weight sensitivity

- Snapshot: `thesis_results/04_shared_joint_loss_jepa_0_mae_2/ccf448a9d694-22d950d16002`
  - `thesis_results/04_shared_joint_loss_jepa_0_mae_2/ccf448a9d694-22d950d16002/provenance/experiment_config.json`
  - `thesis_results/04_shared_joint_loss_jepa_0_mae_2/ccf448a9d694-22d950d16002/data/overall_summary.csv`
  - `thesis_results/04_shared_joint_loss_jepa_0_mae_2/ccf448a9d694-22d950d16002/data/stock_summary.csv`
  - `thesis_results/04_shared_joint_loss_jepa_0_mae_2/ccf448a9d694-22d950d16002/data/predictions_tidy.csv`
- Snapshot: `thesis_results/04_shared_joint_loss_jepa_0_5_mae_1_5/2e7d28d68c3c-d65be5e42b4d`
  - `thesis_results/04_shared_joint_loss_jepa_0_5_mae_1_5/2e7d28d68c3c-d65be5e42b4d/provenance/experiment_config.json`
  - `thesis_results/04_shared_joint_loss_jepa_0_5_mae_1_5/2e7d28d68c3c-d65be5e42b4d/data/overall_summary.csv`
  - `thesis_results/04_shared_joint_loss_jepa_0_5_mae_1_5/2e7d28d68c3c-d65be5e42b4d/data/stock_summary.csv`
  - `thesis_results/04_shared_joint_loss_jepa_0_5_mae_1_5/2e7d28d68c3c-d65be5e42b4d/data/predictions_tidy.csv`
- Snapshot: `thesis_results/04_shared_joint_loss_jepa_1_mae_1/429e4337b133-4bb8562f87e0`
  - `thesis_results/04_shared_joint_loss_jepa_1_mae_1/429e4337b133-4bb8562f87e0/provenance/experiment_config.json`
  - `thesis_results/04_shared_joint_loss_jepa_1_mae_1/429e4337b133-4bb8562f87e0/data/overall_summary.csv`
  - `thesis_results/04_shared_joint_loss_jepa_1_mae_1/429e4337b133-4bb8562f87e0/data/stock_summary.csv`
  - `thesis_results/04_shared_joint_loss_jepa_1_mae_1/429e4337b133-4bb8562f87e0/data/predictions_tidy.csv`
- Snapshot: `thesis_results/04_shared_joint_loss_jepa_1_5_mae_0_5/bc257b6d2afb-9286499b1a2c`
  - `thesis_results/04_shared_joint_loss_jepa_1_5_mae_0_5/bc257b6d2afb-9286499b1a2c/provenance/experiment_config.json`
  - `thesis_results/04_shared_joint_loss_jepa_1_5_mae_0_5/bc257b6d2afb-9286499b1a2c/data/overall_summary.csv`
  - `thesis_results/04_shared_joint_loss_jepa_1_5_mae_0_5/bc257b6d2afb-9286499b1a2c/data/stock_summary.csv`
  - `thesis_results/04_shared_joint_loss_jepa_1_5_mae_0_5/bc257b6d2afb-9286499b1a2c/data/predictions_tidy.csv`
- Snapshot: `thesis_results/04_shared_joint_loss_jepa_2_mae_0/a67f8a1d8485-f64f1511a92a`
  - `thesis_results/04_shared_joint_loss_jepa_2_mae_0/a67f8a1d8485-f64f1511a92a/provenance/experiment_config.json`
  - `thesis_results/04_shared_joint_loss_jepa_2_mae_0/a67f8a1d8485-f64f1511a92a/data/overall_summary.csv`
  - `thesis_results/04_shared_joint_loss_jepa_2_mae_0/a67f8a1d8485-f64f1511a92a/data/stock_summary.csv`
  - `thesis_results/04_shared_joint_loss_jepa_2_mae_0/a67f8a1d8485-f64f1511a92a/data/predictions_tidy.csv`
- Snapshot: `thesis_results/04_local_long_joint_loss_jepa_0_mae_2/eb1cd970301e-c62410937c40`
  - `thesis_results/04_local_long_joint_loss_jepa_0_mae_2/eb1cd970301e-c62410937c40/provenance/experiment_config.json`
  - `thesis_results/04_local_long_joint_loss_jepa_0_mae_2/eb1cd970301e-c62410937c40/data/overall_summary.csv`
  - `thesis_results/04_local_long_joint_loss_jepa_0_mae_2/eb1cd970301e-c62410937c40/data/stock_summary.csv`
  - `thesis_results/04_local_long_joint_loss_jepa_0_mae_2/eb1cd970301e-c62410937c40/data/predictions_tidy.csv`
- Snapshot: `thesis_results/04_local_long_joint_loss_jepa_0_5_mae_1_5/a599bb380215-6a3e8f8b241b`
  - `thesis_results/04_local_long_joint_loss_jepa_0_5_mae_1_5/a599bb380215-6a3e8f8b241b/provenance/experiment_config.json`
  - `thesis_results/04_local_long_joint_loss_jepa_0_5_mae_1_5/a599bb380215-6a3e8f8b241b/data/overall_summary.csv`
  - `thesis_results/04_local_long_joint_loss_jepa_0_5_mae_1_5/a599bb380215-6a3e8f8b241b/data/stock_summary.csv`
  - `thesis_results/04_local_long_joint_loss_jepa_0_5_mae_1_5/a599bb380215-6a3e8f8b241b/data/predictions_tidy.csv`
- Snapshot: `thesis_results/04_local_long_joint_loss_jepa_1_mae_1/62ab3dac50e6-d7532ba2336b`
  - `thesis_results/04_local_long_joint_loss_jepa_1_mae_1/62ab3dac50e6-d7532ba2336b/provenance/experiment_config.json`
  - `thesis_results/04_local_long_joint_loss_jepa_1_mae_1/62ab3dac50e6-d7532ba2336b/data/overall_summary.csv`
  - `thesis_results/04_local_long_joint_loss_jepa_1_mae_1/62ab3dac50e6-d7532ba2336b/data/stock_summary.csv`
  - `thesis_results/04_local_long_joint_loss_jepa_1_mae_1/62ab3dac50e6-d7532ba2336b/data/predictions_tidy.csv`
- Snapshot: `thesis_results/04_local_long_joint_loss_jepa_1_5_mae_0_5/6ae736eff556-87de0125c7ce`
  - `thesis_results/04_local_long_joint_loss_jepa_1_5_mae_0_5/6ae736eff556-87de0125c7ce/provenance/experiment_config.json`
  - `thesis_results/04_local_long_joint_loss_jepa_1_5_mae_0_5/6ae736eff556-87de0125c7ce/data/overall_summary.csv`
  - `thesis_results/04_local_long_joint_loss_jepa_1_5_mae_0_5/6ae736eff556-87de0125c7ce/data/stock_summary.csv`
  - `thesis_results/04_local_long_joint_loss_jepa_1_5_mae_0_5/6ae736eff556-87de0125c7ce/data/predictions_tidy.csv`
- Snapshot: `thesis_results/04_local_long_joint_loss_jepa_2_mae_0/f68c6f9bff75-bc2d1a5dbd02`
  - `thesis_results/04_local_long_joint_loss_jepa_2_mae_0/f68c6f9bff75-bc2d1a5dbd02/provenance/experiment_config.json`
  - `thesis_results/04_local_long_joint_loss_jepa_2_mae_0/f68c6f9bff75-bc2d1a5dbd02/data/overall_summary.csv`
  - `thesis_results/04_local_long_joint_loss_jepa_2_mae_0/f68c6f9bff75-bc2d1a5dbd02/data/stock_summary.csv`
  - `thesis_results/04_local_long_joint_loss_jepa_2_mae_0/f68c6f9bff75-bc2d1a5dbd02/data/predictions_tidy.csv`

| Configuration | RMSE | Direction Accuracy |
|---|---:|---:|
| Shared-Target, MAE only (0, 2) | 0.047470 | 51.78% |
| Shared-Target, MAE-heavy (0.5, 1.5) | 0.047751 | 52.22% |
| Shared-Target, Balanced (1, 1) | 0.047898 | 51.56% |
| Shared-Target, JEPA-heavy (1.5, 0.5) | 0.047690 | 52.00% |
| Shared-Target, JEPA only (2, 0) | 0.048501 | 52.11% |
| Local-MAE/Long-JEPA, MAE only (0, 2) | 0.048328 | 52.33% |
| Local-MAE/Long-JEPA, MAE-heavy (0.5, 1.5) | 0.048021 | 52.61% |
| Local-MAE/Long-JEPA, Balanced (1, 1) | 0.047529 | 53.78% |
| Local-MAE/Long-JEPA, JEPA-heavy (1.5, 0.5) | 0.047548 | 55.39% |
| Local-MAE/Long-JEPA, JEPA only (2, 0) | 0.047736 | 54.44% |

All ten snapshots use context size 12 and identical 24-origin validation support. Every Shared-Target snapshot records `shared_context_12` as its parent. Error bars show the sample standard deviation across five stock-level seed means.

The best Stage 04 RMSE is Shared-Target (0, 2) at 0.047470. The best Local-MAE/Long-JEPA RMSE is (1, 1) at 0.047529; the relative difference is 0.124%.

## Expected-value audit

No material mismatch: every requested approximate value agrees with the checksum-verified snapshot value at the stated precision.
