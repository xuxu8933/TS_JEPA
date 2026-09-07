# Chapter 5 thesis figures

Regenerate with `python analysis/finalize_chapter5_figures.py` (NumPy, pandas,
Matplotlib and Pillow; available in the `ts-jepa` conda environment).
For only the two Stage 05 figures, append `--stage05-only`.
The script checks input SHA256SUMS, selected configurations, published paired
estimates and confidence intervals, five-equity/three-seed coverage, identical
Naive-last horizon rows, and the two sentiment horizon-2 direction values.
It uses published aggregates directly, without recomputing inference.

The six Stage 00-04 PDFs are byte-for-byte copies from
`thesis_results/chapter5_figures/`. The sine/cosine PNG is copied unchanged from
`thesis_results/00_dual_loss_smoke/e56db7d33c56/smoke_sin_cos_predictions.png`.
Stage 01-04 remain validation sensitivity/model-selection evidence.

The two Stage 05 PDFs are vector plots, with matching PNG previews:

- `stage05_paired_rmse_vs_naive`: model minus Naive-last RMSE; positive values
  favour Naive-last. Bars are published 95% equity-bootstrap CIs. Seeds 42, 44,
  46 are averaged within each of five equities before primary inference.
  No significance stars: published Holm-adjusted Wilcoxon p-values exceed 0.05.
- `stage05_horizon_comparison`: published horizon RMSE and Direction Accuracy.
  Identical Naive-last rows are shown once, never averaged across snapshots.
  Sentiment procedures attain 59.2% (Shared) and 56.93% (Local) at horizon 2.
  RMSE stays in the saved target coordinates; no absolute-price conversion.
  Direction Accuracy uses the saved `project_within_trajectory_v1` metric,
  comparing consecutive trajectory differences (including the known zero
  origin for relative-return paths). Constant Naive-last paths therefore
  score near zero; this is not a conventional up/down directional baseline.

Stage 05 is retrospective held-out-test evaluation of validation-selected
complete procedures, not a causal ablation of objectives, EMA, sentiment or
pre-training. Selection uses `0.30 * RMSE rank + 0.70 * Direction Accuracy rank`
on validation. Branch-specific configurations are preserved:

| Procedure | Context (patches) | (lambda_JEPA, lambda_MAE) | Snapshot under thesis_results/ |
|---|---:|---|---|
| Shared / Market | 12 | (1, 1) | 05_benchmark_market_only/0c0edc94f105-41b3195f1267 |
| Local / Market | 24 | (0, 2) | 05_benchmark_local_long_market_only/9db20417d36d-e4e710020059 |
| Shared / Sentiment | 12 | (0.5, 1.5) | 05_benchmark_sentiment/92f8a6ebc19a-5fc1ec922264 |
| Local / Sentiment | 12 | (1.5, 0.5) | 05_benchmark_local_long_sentiment/3804d1fb7d09-f802b55ab4ee |

Each snapshot supplies `data/paired_vs_naive.csv` and `data/horizon_metrics.csv`;
`provenance/experiment_config.json` supplies the selection checks.
No models were retrained and no snapshot values were changed. The optional
aggregate overview was omitted because it duplicates the tables. No synthetic
trajectory or unavailable pre-training-history diagnostic was generated.
The two figure blocks are enabled in the external revised Chapter 5 source:
`/home/xujiang/Downloads/chapter5_results_discussion_revised_v2.tex`.
Figure paths remain relative to the thesis build root (`figures/chapter5/`).
