# Stage 00: controlled comparison and objective ablations

> **DIAGNOSTIC ONLY — one independent seed (7).** Repeated execution verifies determinism; it does not estimate seed uncertainty.

## Main findings

The full JEPA-MAE model and MAE-only are nearly tied on MNIST: adding the JEPA term reduces pooled RMSE by 0.177%. The full model wins on 69 of 128 images against MAE-only, compared with 123 against JEPA-only and 128 against GRU. This run supports a strong role for pixel reconstruction in this transfer protocol; it does not establish a meaningful advantage for the joint objective.

GRU has the lowest sine-cosine RMSE, 66.70% below full JEPA-MAE. It beats the full model on 73 of 76 forecast origins and at all four forecast horizons. MAE-only also has lower pooled RMSE than the full model. Full JEPA-MAE improves pooled RMSE over JEPA-only, but wins on only 38 of 76 origins; aggregate improvement is not uniform across origins.

## Measured results

| Method | MNIST RMSE | Sine-cosine RMSE |
|---|---:|---:|
| JEPA-MAE | 0.108917266 | 0.008631282 |
| JEPA-only | 0.146452531 | 0.009168757 |
| MAE-only | 0.109110765 | 0.008288070 |
| GRU | 0.170200661 | 0.002874152 |
| Naive-last | 0.246229529 | 0.089596644 |
| Drift | 0.278252304 | 0.092663802 |
| Mean-context | 0.278281718 | 0.074163891 |

Each entry is one run. Mean RMSE equals the displayed value; across-seed SD is unavailable. MNIST errors are pixel intensities; sine errors are window-relative Close values. Do not compare magnitudes across datasets.

## Experiment and model definitions

MNIST retains 4,000 train / 128 validation / 128 test images, 28 row tokens, 11 randomly masked targets and 17 visible rows. The synthetic task retains 20 historical observations, Close/Volume inputs, four future Close targets, and 96/10/76 train/validation/test windows. Its raw row split is February 19–June 17 / June 18–30 / July 1–September 17, 2021, after the original 49-row warmup removal. See [protocol.md](protocol.md) for complete data construction, normalization, masking and model definitions.

Full-model SSL weights are JEPA/MAE = 0.01/1 on MNIST and 0.7/0.3 on sine. Ablations change only those weights to 1/0 or 0/1. MNIST uses the existing width-64 encoder/predictor (two blocks, four heads), 30 SSL epochs, then a fresh identical MLP readout trained for 30 epochs with encoder/predictor frozen. Sine uses the width-16 encoder and width-8 predictor (one block, two heads), the original single SSL update, and 120 downstream epochs with encoder fine-tuning. GRU is one layer with hidden size 64 and no dropout, supervised using the same per-dataset downstream budget. Naive-last, Drift and Mean-context are deterministic context-only baselines. Every learned downstream checkpoint is selected by validation RMSE only.

## Interpretation and confounders

- **Observation:** the full MNIST loss is heavily reconstruction-weighted. At the final SSL epoch its JEPA component contributes approximately 3.04% of the weighted validation loss. Similarity to MAE-only is therefore consistent with this particular weighting; it does not test every possible balance.
- **Observation:** JEPA-only MNIST latent validation loss changes from 0.038214 at SSL epoch 0 to 0.185192 at epoch 29; target-embedding standard deviation changes from 0.050965 to 0.444664. A small moving-teacher latent loss alone is not a reliable proxy for reconstruction transfer. These diagnostics alone do not establish representation collapse. All variants retain their final, fixed-budget SSL checkpoint; no post-test checkpoint substitution was made.
- **Observation:** downstream selection differs by method. Full/JEPA-only/MAE-only/GRU select epochs 30/29/24/30 on MNIST and 119/120/118/86 on sine. Sine GRU's final validation RMSE is 0.006662 versus 0.002562 at its selected epoch, illustrating why the validation checkpoint matters.
- **Limit:** sine pretraining is only one optimizer step. Its 13 validation observations cannot form a 20-observation SSL window; SSL validation is unavailable, while downstream validation has 10 windows. This budget supports a pipeline diagnostic, not a strong SSL comparison.
- **Limit:** there is no matching randomly initialized Transformer control. Comparison with GRU cannot isolate the benefit of pretraining from architecture, parameter count, total optimization work or the frozen/readout protocol. Equal downstream epochs do not imply equal total compute.
- **Limit:** only one learned-model seed exists. MNIST images and overlapping sine forecast origins are not additional random seeds; the four forecast steps and 28 row pixels are not independent model runs. No p-values or seed confidence intervals are claimed.
- **Hypothesis for a separately specified future experiment:** additional pretraining and several seeds may change the ablation ranking. Any new settings should be selected on validation data and evaluated on fresh held-out support; the present test outcomes must not become tuning criteria.

## Legacy compatibility and corrections

The old MNIST previous-row baseline accessed masked pixels in 570/1,408 target rows. The new Naive-last uses only visible rows, so its result is not numerically comparable to the old baseline. MNIST remains reconstruction with visible rows on both sides, not causal forecasting. The old sine SSL cutoff overlapped downstream validation/test; the corrected run aligns SSL to the existing July 1 downstream cutoff. The archived legacy full-model numbers were reproduced separately, including sine RMSE 0.0102686705 with 12 CPU threads. This comparison fixes one CPU thread throughout; floating-point reduction order changed the legacy optimization trajectory. Original published diagnostic snapshots are retained and are not replaced by these scores.

## Verification and publication

The analysis recomputed every RMSE in float64 from saved predictions, checked complete 2×7×1 coverage, identical targets/support across methods, exact repeat predictions and recorded metrics, finite checkpoint weights, and restoration of the minimum-validation checkpoint. The full experiment verification also compares repeated pretrained/downstream weights and confirms split separation; its record is included under provenance.

Published data include raw run metrics, mean/SD/count summaries, per-sample errors, paired win/loss counts, per-horizon metrics, training histories and LaTeX tables. Raw prediction/checkpoint files are represented by paths and SHA-256 hashes and remain in the ignored results archive. The runtime manifest, split audits and exact source text snapshot preserve the executed configuration, including code that was uncommitted at run time.

Reproduce this analysis and publish it with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/xujiang/miniconda3/envs/ts-jepa/bin/python results/00_stage00_verification.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/xujiang/miniconda3/envs/ts-jepa/bin/python analysis/analyze_stage00_results.py
python publish_thesis_results.py --analysis-dir analysis_artifacts/00_model_comparison
```

Use a new `--output-dir` to regenerate analysis staging. Publication is local to the repository's immutable `thesis_results/` workflow.

## Git publication

- Immutable snapshot: `no-manifest-122a67f97c70`
- Full raw experiment outputs are intentionally excluded from Git.
- `SHA256SUMS` verifies every published file.
- Files omitted by publication policy: 0. See `publication_manifest.csv`.
- Large/raw artifacts should be attached to the matching GitHub Release.
- User-requested figure presentation revision: simplified title and removed footnote; numerical data are unchanged. See `provenance/figure_revision.json`.
