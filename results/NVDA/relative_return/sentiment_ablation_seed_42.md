# NVDA sentiment ablation (seed 42)

This comparison changes only the input features:

- With sentiment: `Close`, `Volume`, `MA10`, `MA50`, `sentiment_mean`
- Without sentiment: `Close`, `Volume`, `MA10`, `MA50`

Both runs use the random masking strategy, 20-row pretraining windows, 5-row
patches, `train_zscore` normalization, seed 42, a training cutoff of
2024-12-31, a test start of 2025-01-01, and an inclusive data cutoff of
2026-01-01. The downstream target is relative return. The test set contains 38
rolling windows and 190 horizon predictions. The true values and rolling/horizon
indices are identical between conditions.

| Model | Metric | With sentiment | Without sentiment | Effect of sentiment |
| --- | ---: | ---: | ---: | ---: |
| TS-JEPA | MSE | 0.002647 | 0.003104 | 14.71% lower |
| TS-JEPA | MAE | 0.040564 | 0.042652 | 4.89% lower |
| TS-JEPA | Trend accuracy | 49.47% | 44.74% | +4.74 pp |
| GRU | MSE | 0.003326 | 0.002343 | 41.95% higher |
| GRU | MAE | 0.044670 | 0.036448 | 22.56% higher |
| GRU | Trend accuracy | 51.05% | 50.00% | +1.05 pp |

Sentiment improved TS-JEPA on all reported metrics. For the GRU, sentiment
slightly improved direction accuracy but substantially worsened MSE and MAE.
The simple baselines were identical across conditions. Neither learned model
beat `naive_last`, `mean_context`, or `drift` on error in this single-seed test.

This is one seed, so the result is an ablation result rather than evidence of a
stable population-level effect. A multi-seed comparison is needed for a robust
conclusion.
