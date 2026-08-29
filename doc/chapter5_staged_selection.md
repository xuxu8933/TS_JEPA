# Chapter 5 staged validation selection

This workflow runs three validation-only stages and performs one held-out test evaluation after the final configuration is frozen.

## Fixed pilot settings

- Stocks: NVDA, AAPL, AVGO, TSLA, WMT
- Seeds: 42, 44, 46
- Parallel jobs: 2
- Pretraining/downstream epochs: 2001/501
- Forecast horizon: 5
- Objective weights: JEPA 1.0, MAE 0.5
- Selection metric: downstream validation MSE
- Expected wall time on the detected RTX 3060: 6--10 hours

Sentiment CSVs must already be cached. The timed candidate configs do not download market or news data.

## Automated execution

First validate all ten candidate configurations without training, selecting, writing repository artifacts, or evaluating test data:

```bash
conda run --no-capture-output -n ts-jepa python run_chapter5_staged.py --dry-run
```

Then run the complete workflow:

```bash
conda run --no-capture-output -n ts-jepa python run_chapter5_staged.py
```

The script runs each stage sequentially, generates later candidates from the actual preceding validation winner, writes the three selection manifests automatically, and finally runs `selection_artifacts/chapter5_automated/final/selected_config.json` on test once. Re-running the command resumes compatible completed stock/seed runs through the existing runner checks. It stops immediately on a failed or incompatible run.

Audit outputs are under `selection_artifacts/chapter5_automated/`. The final workflow summary is `workflow_summary.json`; each stage retains its manifest and selection summary. The held-out test results are written below `selection_artifacts/chapter5_automated/final/results/selected_config/`, separate from all validation candidate roots.

The remaining sections describe the equivalent manual procedure.

## Stage 1: preprocessing and normalization

Inspect both generated command sets without running them:

```bash
for cfg in 01_preprocessing_window_return 01_preprocessing_train_zscore; do
  conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
    --config "config/experiments/chapter5_candidates/${cfg}.json" \
    --dry-run --verbose
done
```

Run both candidates:

```bash
for cfg in 01_preprocessing_window_return 01_preprocessing_train_zscore; do
  conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
    --config "config/experiments/chapter5_candidates/${cfg}.json"
done
```

Locate their validation roots:

```bash
find results/01_preprocessing_window_return results/01_preprocessing_train_zscore \
  -name validation_metrics.json -print
```

Copy `config/experiments/chapter5_stage1_selection.template.jsonc` to `config/experiments/chapter5_stage1_selection.jsonc`. Its validation roots already match the runner output directories. Select stage 1:

```bash
conda run --no-capture-output -n ts-jepa python chapter5_selection.py \
  --manifest config/experiments/chapter5_stage1_selection.jsonc \
  --output-dir selection_artifacts/chapter5_stage1
```

Materialize the two stage-2 configs from the actual validation winner:

```bash
STAGE1_PARENT_ID=$(conda run -n ts-jepa python -c \
  "import json; print(json.load(open('selection_artifacts/chapter5_stage1/selection_summary.json'))['selected_candidate_id'])")

conda run --no-capture-output -n ts-jepa python chapter5_prepare_candidates.py \
  --stage sentiment \
  --base-config selection_artifacts/chapter5_stage1/selected_stage_config.json \
  --parent-candidate-id "$STAGE1_PARENT_ID" \
  --output-dir config/experiments/chapter5_candidates
```

## Stage 2: sentiment

Dry-run, then run both candidates:

```bash
for cfg in 02_sentiment_excluded 02_sentiment_included; do
  conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
    --config "config/experiments/chapter5_candidates/${cfg}.json" \
    --dry-run --verbose
done

for cfg in 02_sentiment_excluded 02_sentiment_included; do
  conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
    --config "config/experiments/chapter5_candidates/${cfg}.json"
done
```

Copy `chapter5_stage2_selection.template.jsonc` to `chapter5_stage2_selection.jsonc`. Set both sentiment `parent_candidate_id` values to `STAGE1_PARENT_ID`, then select:

```bash
conda run --no-capture-output -n ts-jepa python chapter5_selection.py \
  --manifest config/experiments/chapter5_stage2_selection.jsonc \
  --output-dir selection_artifacts/chapter5_stage2
```

Materialize the six architecture-context candidates:

```bash
STAGE2_PARENT_ID=$(conda run -n ts-jepa python -c \
  "import json; print(json.load(open('selection_artifacts/chapter5_stage2/selection_summary.json'))['selected_candidate_id'])")

conda run --no-capture-output -n ts-jepa python chapter5_prepare_candidates.py \
  --stage architecture_context \
  --base-config selection_artifacts/chapter5_stage2/selected_stage_config.json \
  --parent-candidate-id "$STAGE2_PARENT_ID" \
  --output-dir config/experiments/chapter5_candidates
```

## Stage 3: architecture and context

Dry-run all six configurations:

```bash
for cfg in \
  03_shared_context_6_patches \
  03_shared_context_12_patches \
  03_shared_context_24_patches \
  03_local_long_context_6_patches \
  03_local_long_context_12_patches \
  03_local_long_context_24_patches; do
  conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
    --config "config/experiments/chapter5_candidates/${cfg}.json" \
    --dry-run --verbose
done
```

Run all six by removing the dry-run flags:

```bash
for cfg in \
  03_shared_context_6_patches \
  03_shared_context_12_patches \
  03_shared_context_24_patches \
  03_local_long_context_6_patches \
  03_local_long_context_12_patches \
  03_local_long_context_24_patches; do
  conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
    --config "config/experiments/chapter5_candidates/${cfg}.json"
done
```

Copy `chapter5_selection.template.jsonc` to `chapter5_selection.jsonc`. Set both sentiment parent IDs to `STAGE1_PARENT_ID`, and set all six final parent IDs to `STAGE2_PARENT_ID`. Perform complete validation selection:

```bash
conda run --no-capture-output -n ts-jepa python chapter5_selection.py \
  --manifest config/experiments/chapter5_selection.jsonc \
  --output-dir selection_artifacts/chapter5_final
```

Archive `selection_summary.json` before proceeding. It contains the complete validation-only decision trail.

## One final held-out test run

Run the frozen configuration once:

```bash
conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
  --config selection_artifacts/chapter5_final/selected_config.json
```

The final run writes `test_metrics.json`. Never point a selection manifest at this result directory.
