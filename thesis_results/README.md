# Published thesis results

This directory contains validated, immutable thesis-analysis snapshots that are
intended to be committed to Git and shared between devices.

Directories prefixed with `00_` contain diagnostic smoke-test artifacts. They
verify that the pipeline runs, but they are not thesis experiment evidence.

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
