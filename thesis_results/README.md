# Published thesis results

This directory contains validated, immutable thesis-analysis snapshots that are
intended to be committed to Git and shared between devices.

The [Stage 04 local-long market-only context-12 analysis](../doc/chapter5_stage04_local_long_market_only_context_12_analysis.md)
compares all five loss weights across five stocks and three seeds, links the
published snapshots, and records the exploratory validation limits.

Directories prefixed with `00_` contain diagnostic smoke-test artifacts and
controlled model comparisons. They are not financial forecasting evidence.

The [Stage 00 controlled comparison](00_model_comparison/no-manifest-122a67f97c70/README.md)
publishes all seven methods on MNIST masked-row reconstruction and sine-cosine
forecasting. It includes the objective-ablation analysis, corrected baselines,
CSV/LaTeX tables, a figure, and reproducibility records. This snapshot has one
independent seed; repeated execution is not an estimate of seed uncertainty.

Generate disposable analysis output under `analysis_artifacts/`, then publish a
validated snapshot with:

```bash
python publish_thesis_results.py \
  --analysis-dir analysis_artifacts/<config-name>
```

Snapshots are stored below `<config-name>/<experiment-and-analysis-signature>/`.
Publication is refused when the analysis reports validity errors or has no
canonical result rows. Each snapshot includes `publication_manifest.csv` and
`SHA256SUMS` for auditing.

Raw stock/seed outputs and analysis staging files are intentionally excluded
from Git. Package those separately as an immutable GitHub Release asset when
full reproduction data must be shared.
