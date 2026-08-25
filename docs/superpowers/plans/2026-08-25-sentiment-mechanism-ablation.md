# Sentiment Mechanism Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and dry-run four controlled sentiment-ablation configurations, together with leakage-safe preprocessing, independent forecast horizons, configuration-isolation validation, and post-experiment paired analysis.

**Architecture:** Extend the existing experiment-config-to-runner pipeline with two orthogonal options: downstream forecast horizon and selective sentiment normalization. Derived sentiment features are created in the financial data layer, the runner gains a side-effect-free JSON dry-run report, and a separate analysis module consumes completed raw results plus immutable published controls.

**Tech Stack:** Python 3.11, PyTorch, pandas, NumPy, `argparse`, standard-library `json`, `csv`, `math`, `statistics`, `subprocess`, `unittest`, and existing TS-JEPA configuration/result utilities.

**Spec:** `docs/superpowers/specs/2026-08-25-sentiment-mechanism-ablation-design.md`

## Global Constraints

- Work only on branch `single-dim`.
- Do not run pretraining, downstream optimization, an optimizer step, a one-stock smoke training run, or any 10-stock × 10-seed experiment.
- Authorized execution is limited to unit/integration tests, CLI help, config validation, analysis tests, provenance tests, deterministic-pairing tests, lightweight data loading, and true dry-runs.
- Do not modify either published package under `thesis_results/top10_with_sentiment/5b8f3897bf23-02add88f32d5/` or `thesis_results/top10_without_sentiment/2fab810c1e1d-d0fb2944255b/`.
- Preserve chronological splitting, date alignment, window-return market preprocessing, existing control-config behavior, patch size 5, both primary mask strategies, all ten stocks, and seeds 42–51.
- H1 changes downstream forecast horizon only; H2 adds raw `has_news` only; H3 replaces raw sentiment with train-only `sentiment_mean_z` only.
- Do not generate H1/H2/H3 scientific verdicts without complete user-run results.

## File Responsibility Map

- `config/experiment.py`: shared defaults, feature recognition, forecast-horizon resolution, and cross-stage validation.
- `config/file_options.py`: nested experiment-file schema and flattening for the two new options.
- `src/data_loaders/financial_preprocessing.py`: declarations for raw and derived sentiment features.
- `src/data_loaders/data_class_roll_volume.py`: same-date feature derivation, chronological train-only sentiment fitting, normalization state, target-horizon construction.
- `src/data_loaders/data_loader_roll_volume.py`: loader argument forwarding.
- `pretrain_dual_loss.py`: pretraining CLI/state persistence and optional downstream handoff.
- `eval_dual_loss.py`: checkpoint-aware downstream command construction.
- `main/utils.py`: downstream CLI parsing for direct evaluator execution.
- `eval_forecast_prequential_with_baselines_gru_volume.py`: target width, model output width, loader reuse, and preprocessing provenance.
- `run_top_nasdaq100_stocks.py`: experiment CLI propagation, semantic fingerprints, dry-run validation report, and early return.
- Four `config/experiments/top10_*.json` files: the production-ready intervention configurations.
- `analysis/sentiment_mechanism.py`: config isolation, result loading, pairing, statistics, verdicts, provenance, and reports.
- `analyze_sentiment_mechanisms.py`: thin CLI entry point.
- `tests/test_sentiment_mechanism_ablation.py`: focused tests for every new behavior and safety boundary.
- `README.md` and `doc/configuration.md`: exact CLI/config semantics and manual commands.

---

### Task 1: Separate Forecast Horizon From Input Patch Size

**Files:**
- Create: `tests/test_sentiment_mechanism_ablation.py`
- Modify: `config/experiment.py`
- Modify: `config/file_options.py`
- Modify: `src/data_loaders/data_loader_roll_volume.py`
- Modify: `src/data_loaders/data_class_roll_volume.py`
- Modify: `main/utils.py`
- Modify: `pretrain_dual_loss.py`
- Modify: `eval_dual_loss.py`
- Modify: `eval_forecast_prequential_with_baselines_gru_volume.py`
- Modify: `run_top_nasdaq100_stocks.py`

**Interfaces:**
- Produces: `resolve_forecast_horizon(forecast_horizon: int | None, patch_size: int) -> int` in `config.experiment`.
- Produces: `forecast_horizon: int | None` accepted by runner/downstream CLIs and `EvaluationDataLoader`.
- Preserves: omitted horizon resolves to patch size and produces the current five-target behavior.

- [ ] **Step 1: Write failing configuration and dataset tests**

Add tests that demonstrate default compatibility and H=1 target sizing without changing patch geometry:

```python
class ForecastHorizonTest(unittest.TestCase):
    def test_omitted_horizon_defaults_to_patch_size(self):
        self.assertEqual(resolve_forecast_horizon(None, 5), 5)

    def test_horizon_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "forecast_horizon must be positive"):
            resolve_forecast_horizon(0, 5)

    def test_h1_target_does_not_change_context_patch_dimension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prices.csv"
            write_price_csv(path, rows=180)
            dataset = EvaluationDataLoader(
                path_data=str(path), patch_size=5, forecast_horizon=1,
                context_size=12, stride=5, split="train",
                normalization="window_return",
                feature_cols=("Close", "Volume", "MA10", "MA50"),
                validation_fraction=0.1, test_start_date="2020-05-25",
            )
            context, target = dataset[0]
            self.assertEqual(tuple(context.shape), (12, 20))
            self.assertEqual(tuple(target.shape), (1,))
```

The test file must include a concrete helper that writes monotonically dated `Close` and `Volume` rows.

- [ ] **Step 2: Run the new tests and verify the expected failures**

Run:

```bash
conda run --no-capture-output -n ts-jepa python -m unittest \
  tests.test_sentiment_mechanism_ablation.ForecastHorizonTest -v
```

Expected: errors because `resolve_forecast_horizon` and the loader argument do not exist.

- [ ] **Step 3: Add shared horizon resolution and config-file mapping**

Add this resolver and use it anywhere a target width is needed:

```python
def resolve_forecast_horizon(forecast_horizon: int | None, patch_size: int) -> int:
    resolved = int(patch_size if forecast_horizon is None else forecast_horizon)
    if resolved <= 0:
        raise ValueError(f"forecast_horizon must be positive, got {resolved}")
    return resolved
```

Add `forecast_horizon: None` to downstream defaults and `eval_forecast_horizon: None` to pretraining evaluation defaults. Extend `[runner].downstream` in `flatten_runner_options` to accept and map `forecast_horizon`. Preserve omission rather than materializing a new result-affecting value in legacy config fingerprints.

- [ ] **Step 4: Thread the horizon through downstream command builders**

Add `--forecast-horizon` to runner, `eval_dual_loss.py`, and the direct evaluator. Pass the flag only when the runner value is not `None`:

```python
if args.forecast_horizon is not None:
    eval_command.extend(["--forecast-horizon", str(args.forecast_horizon)])
```

When pretraining launches optional evaluation, pass `eval_forecast_horizon` conditionally. When `eval_dual_loss.py` launches the evaluator, prefer the explicit downstream value and do not derive it from a pretraining checkpoint.

- [ ] **Step 5: Make target construction and model output use the resolved horizon**

In `EvaluationDataLoader`, store `self.forecast_horizon`, compute sample length as `context_size * patch_size + forecast_horizon`, slice exactly that many future rows, and reshape targets to `forecast_horizon`. Keep context reshaping at `patch_size * feature_dim`.

In the direct evaluator:

```python
forecast_horizon = resolve_forecast_horizon(
    config.get("forecast_horizon"), patch_size
)
config["forecast_horizon"] = forecast_horizon
```

Use `forecast_horizon` for the JEPA forecast head, GRU `output_size`, metadata horizons, test target-end lookup, and every former output-width use of patch size. Continue using patch size for tokenizer/encoder layout and context history.

- [ ] **Step 6: Preserve old experiment and checkpoint identities**

In signature canonicalization, omit the new option when it is `None` or equals patch size. Add regression assertions that existing config signatures still match:

```python
self.assertEqual(
    experiment_config_signature(parse_runner(["--config", str(with_config)])),
    "5b8f3897bf231cf2ad5968d6dbfa03ad0143a75455f5596c90faf2059f5f6eaa",
)
self.assertEqual(
    experiment_config_signature(parse_runner(["--config", str(without_config)])),
    "2fab810c1e1d50cee4c3744e33a802c6ba29d507cbf0da431284b21c4833bbd5",
)
```

- [ ] **Step 7: Run focused and existing downstream tests**

```bash
conda run --no-capture-output -n ts-jepa python -m unittest \
  tests.test_sentiment_mechanism_ablation.ForecastHorizonTest \
  tests.test_financial_preprocessing tests.test_unified_dual_loss \
  tests.test_top10_nasdaq_mask_comparison -v
```

Expected: PASS; default targets remain length 5 and H1 targets have length 1.

- [ ] **Step 8: Commit the horizon slice**

```bash
git add config/experiment.py config/file_options.py \
  src/data_loaders/data_loader_roll_volume.py src/data_loaders/data_class_roll_volume.py \
  main/utils.py pretrain_dual_loss.py eval_dual_loss.py \
  eval_forecast_prequential_with_baselines_gru_volume.py \
  run_top_nasdaq100_stocks.py tests/test_sentiment_mechanism_ablation.py
git commit -m "feat: separate forecast horizon from patch size"
```

---

### Task 2: Add Causal `has_news` and Train-Only `sentiment_mean_z`

**Files:**
- Modify: `config/experiment.py`
- Modify: `config/file_options.py`
- Modify: `src/data_loaders/financial_preprocessing.py`
- Modify: `src/data_loaders/data_class_roll_volume.py`
- Modify: `src/data_loaders/data_loader_roll_volume.py`
- Modify: `pretrain_dual_loss.py`
- Modify: `eval_dual_loss.py`
- Modify: `main/utils.py`
- Modify: `eval_forecast_prequential_with_baselines_gru_volume.py`
- Modify: `run_top_nasdaq100_stocks.py`
- Modify: `tests/test_sentiment_mechanism_ablation.py`

**Interfaces:**
- Produces: `DERIVED_SENTIMENT_FEATURE_SOURCES = {"has_news": "news_count", "sentiment_mean_z": "sentiment_mean"}`.
- Produces: `fit_transform_sentiment_features(frame_splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], feature_cols: Sequence[str], mode: str, state: Mapping[str, Any] | None = None, eps: float = 1e-6) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], dict[str, Any] | None]`.
- Produces: loader property and checkpoint key `sentiment_normalization_stats`.
- Consumes: existing chronological frame split before tensor conversion.

- [ ] **Step 1: Write failing `has_news` tests**

Construct price dates plus daily sentiment rows for neutral, positive, negative, and future-only news. Test through `load_price_series`. Assert missing maps to `(0.0, 0.0)`, observed neutral maps to approximately `(0.0, 1.0)`, positive/negative rows both have `has_news == 1.0`, all indicator values are binary, and date `t+1` never alters date `t`.

- [ ] **Step 2: Write failing selective-z-score tests**

```python
train = pd.DataFrame({"sentiment_mean": [-1.0, 0.0, 1.0], "sentiment_mean_z": [-1.0, 0.0, 1.0]})
val = pd.DataFrame({"sentiment_mean": [1000.0], "sentiment_mean_z": [1000.0]})
test = pd.DataFrame({"sentiment_mean": [-1000.0], "sentiment_mean_z": [-1000.0]})
transformed, state = fit_transform_sentiment_features(
    (train, val, test), ["sentiment_mean_z"], "train_zscore"
)
self.assertEqual(state["fit_split"], "train")
self.assertAlmostEqual(state["features"]["sentiment_mean_z"]["mean"], 0.0)
self.assertAlmostEqual(
    state["features"]["sentiment_mean_z"]["std"], math.sqrt(2.0 / 3.0)
)
```

Change validation/test values and assert fitted state is unchanged. Pass stored state into another call and assert identical transformed values.

- [ ] **Step 3: Run sentiment tests and verify expected failures**

```bash
conda run --no-capture-output -n ts-jepa python -m unittest \
  tests.test_sentiment_mechanism_ablation.SentimentFeatureTest -v
```

Expected: failure because derived features and selective normalization are absent.

- [ ] **Step 4: Implement same-date derived sentiment features**

Extend known sentiment sets with `has_news` and `sentiment_mean_z`. Expand requested derived columns to raw dependencies before reading the daily file. After the normalized-date left join and zero fill, materialize:

```python
if "has_news" in requested_features:
    frame["has_news"] = (frame["news_count"] > 0).astype("float64")
if "sentiment_mean_z" in requested_features:
    frame["sentiment_mean_z"] = frame["sentiment_mean"].astype("float64")
```

Do not shift, forward-fill, nearest-match, or reassign weekend news.

- [ ] **Step 5: Implement post-split selective normalization**

Run immediately after `_chronological_split_frames` and before tensor conversion. Use population standard deviation and replace scales below epsilon with 1.0. Store:

```python
{
    "mode": "train_zscore",
    "fit_split": "train",
    "features": {
        "sentiment_mean_z": {
            "source": "sentiment_mean",
            "mean": fitted_mean,
            "std": fitted_std,
            "eps": 1e-6,
        }
    },
}
```

Reject z-scored derived features under mode `none`, reject `train_zscore` without a `_z` feature, and reject mismatched reusable state.

- [ ] **Step 6: Thread and persist selective normalization state**

Add `sentiment_normalization="none"` and `sentiment_normalization_stats=None` through loaders, CLIs, command generation, and checkpoint handoffs. Reuse the training loader state for validation/test and include it in `preprocessing_config.json`. Keep global `normalization_stats` untouched. Omit mode `none` from fingerprints.

- [ ] **Step 7: Add compatibility and provenance tests**

Load one fixture with omitted new arguments and explicit `sentiment_normalization="none"`; assert equal split tensors, names, passthrough indices, and existing normalization state. For H3 assert passthrough index `[4]`, `fit_split == "train"`, and persisted `sentiment_mean_z` statistics. Use the lightweight checkpoint helper to verify round-trip state.

- [ ] **Step 8: Run focused preprocessing and checkpoint tests**

```bash
conda run --no-capture-output -n ts-jepa python -m unittest \
  tests.test_sentiment_mechanism_ablation.SentimentFeatureTest \
  tests.test_sentiment_toggle tests.test_financial_preprocessing \
  tests.test_unified_dual_loss -v
```

Expected: PASS with unchanged control preprocessing.

- [ ] **Step 9: Commit the sentiment slice**

```bash
git add config/experiment.py config/file_options.py \
  src/data_loaders/financial_preprocessing.py src/data_loaders/data_class_roll_volume.py \
  src/data_loaders/data_loader_roll_volume.py pretrain_dual_loss.py eval_dual_loss.py \
  main/utils.py eval_forecast_prequential_with_baselines_gru_volume.py \
  run_top_nasdaq100_stocks.py tests/test_sentiment_mechanism_ablation.py
git commit -m "feat: add controlled sentiment feature transforms"
```

---

### Task 3: Add Production Configs and Machine-Readable Isolation Checks

**Files:**
- Create: `config/experiments/top10_h1_without_sentiment.json`
- Create: `config/experiments/top10_h1_with_sentiment.json`
- Create: `config/experiments/top10_sentiment_has_news.json`
- Create: `config/experiments/top10_sentiment_zscore.json`
- Create: `analysis/sentiment_mechanism.py`
- Modify: `analysis/__init__.py`
- Modify: `tests/test_sentiment_mechanism_ablation.py`

**Interfaces:**
- Produces: `semantic_experiment_config(config_path: Path) -> dict[str, Any]`.
- Produces: `nested_config_diff(control, intervention) -> dict[str, dict[str, Any]]` using dotted paths.
- Produces: `validate_ablation_configs(repo_root: Path) -> dict[str, Any]` with JSON-serializable verification details.

- [ ] **Step 1: Write failing config-isolation tests**

Use this expected matrix:

```python
EXPECTED = {
    "top10_h1_without_sentiment.json": (1, ["Close", "Volume", "MA10", "MA50"], 20),
    "top10_h1_with_sentiment.json": (1, ["Close", "Volume", "MA10", "MA50", "sentiment_mean"], 25),
    "top10_sentiment_has_news.json": (5, ["Close", "Volume", "MA10", "MA50", "sentiment_mean", "has_news"], 30),
    "top10_sentiment_zscore.json": (5, ["Close", "Volume", "MA10", "MA50", "sentiment_mean_z"], 25),
}
```

For every config assert exact stock order, seeds `list(range(42, 52))`, patch size 5, `random` plus `local_long`, and derived result directory. Assert config validation returns `valid: true` and verifies published manifests.

- [ ] **Step 2: Run config tests and verify expected failure**

```bash
conda run --no-capture-output -n ts-jepa python -m unittest tests.test_sentiment_mechanism_ablation.ConfigIsolationTest -v
```

Expected: failure because configs and comparison functions are absent.

- [ ] **Step 3: Create configs by minimal control transformations**

Copy complete controls and apply only:

```text
top10_h1_without_sentiment: downstream.forecast_horizon = 1
top10_h1_with_sentiment:    downstream.forecast_horizon = 1
top10_sentiment_has_news:   sentiment.columns adds has_news; forecast_horizon = 5
top10_sentiment_zscore:     sentiment.columns becomes [sentiment_mean_z];
                            sentiment.normalization = train_zscore;
                            forecast_horizon = 5
```

Keep `execution.dry_run` false. CLI overrides enable validation.

- [ ] **Step 4: Implement semantic snapshots and dotted diffs**

Parse through `run_top_nasdaq100_stocks.parse_args`, resolve horizon/features/default selective normalization, and emit a canonical semantic dictionary. Exclude only existing execution mechanics. A diff entry has this shape:

```python
{
    "runner.preprocessing.sentiment_features": {
        "control": ["sentiment_mean"],
        "intervention": ["sentiment_mean", "has_news"],
    }
}
```

Validate against explicit allowed dotted paths. Compare current control signatures/effective configs with immutable published manifests.

- [ ] **Step 5: Run tests and emit the config report**

```bash
conda run --no-capture-output -n ts-jepa python -m unittest tests.test_sentiment_mechanism_ablation.ConfigIsolationTest -v
conda run --no-capture-output -n ts-jepa python -c "import json; from pathlib import Path; from analysis.sentiment_mechanism import validate_ablation_configs; print(json.dumps(validate_ablation_configs(Path('.')), indent=2, sort_keys=True))"
```

Expected: PASS and JSON `valid: true` with only approved differences.

- [ ] **Step 6: Commit configs and isolation validation**

```bash
git add config/experiments/top10_h1_without_sentiment.json config/experiments/top10_h1_with_sentiment.json config/experiments/top10_sentiment_has_news.json config/experiments/top10_sentiment_zscore.json analysis/sentiment_mechanism.py analysis/__init__.py tests/test_sentiment_mechanism_ablation.py
git commit -m "feat: configure isolated sentiment ablations"
```

---

### Task 4: Make Runner Dry-Run Structured and Side-Effect Free

**Files:**
- Modify: `run_top_nasdaq100_stocks.py`
- Modify: `tests/test_sentiment_mechanism_ablation.py`
- Modify: `tests/test_top10_nasdaq_mask_comparison.py`

**Interfaces:**
- Produces: `build_dry_run_report(args, stocks, seeds, strategies) -> dict[str, Any]`.
- Produces: stdout marker `DRY_RUN_VALIDATION` followed by one JSON object.
- Guarantees: dry-run returns before `run_command`, `execute_tasks`, `_write_json`, summary-file opening, or directory creation.

- [ ] **Step 1: Write failing report and safety tests**

Parse each new config with `--dry-run` and compare every field. For the boundary test:

```python
with patch("run_top_nasdaq100_stocks.run_command", side_effect=AssertionError("execution reached")), patch("run_top_nasdaq100_stocks.execute_tasks", side_effect=AssertionError("training reached")), patch("run_top_nasdaq100_stocks._write_json", side_effect=AssertionError("write reached")), patch("pathlib.Path.open", side_effect=AssertionError("file write reached")):
    output = io.StringIO()
    with redirect_stdout(output):
        run_stock_main(["--config", str(config_path), "--dry-run"])
self.assertIn('"training_disabled": true', output.getvalue())
```

Patch `plan_incremental_execution` with an in-memory plan so patched `Path.open` cannot affect read-only probes.

- [ ] **Step 2: Run safety tests and verify failure**

```bash
conda run --no-capture-output -n ts-jepa python -m unittest tests.test_sentiment_mechanism_ablation.DryRunSafetyTest -v
```

Expected: failure because current dry-run opens a summary.

- [ ] **Step 3: Build the validation report**

Resolve features through shared logic and emit:

```python
{
    "experiment_name": Path(args.config).stem,
    "git_branch": current_git_branch(),
    "stock_count": len(stocks),
    "stocks": stocks,
    "seed_count": len(seeds),
    "seeds": seeds,
    "forecast_horizon": resolved_horizon,
    "feature_names": feature_names,
    "feature_count": len(feature_names),
    "patch_size": args.patch_size,
    "flattened_patch_input_dimension": args.patch_size * len(feature_names),
    "sentiment_handling": sentiment_description,
    "sentiment_normalization": preprocessing["sentiment_normalization"],
    "normalization_mode": preprocessing["normalization"],
    "output_directory": str(Path(args.results_dir)),
    "training_disabled": True,
}
```

Validate every stock CSV and, for sentiment conditions, daily sentiment CSV. Raise one error listing missing paths.

- [ ] **Step 4: Return before write/execution boundaries**

After config, geometry, paths, and command-plan validation:

```python
print("DRY_RUN_VALIDATION")
print(json.dumps(report, indent=2, sort_keys=True))
return
```

Place before downloads, summaries, manifests, tasks, and plots. Adjust old dry-run tests to the stronger contract.

- [ ] **Step 5: Run runner suites**

```bash
conda run --no-capture-output -n ts-jepa python -m unittest tests.test_sentiment_mechanism_ablation.DryRunSafetyTest tests.test_top10_nasdaq_mask_comparison tests.test_runtime_optimizations -v
```

Expected: PASS and no result directory created.

- [ ] **Step 6: Commit dry-run safety**

```bash
git add run_top_nasdaq100_stocks.py tests/test_sentiment_mechanism_ablation.py tests/test_top10_nasdaq_mask_comparison.py
git commit -m "feat: add side-effect-free experiment dry runs"
```

---

### Task 5: Implement Deterministic Paired Analysis and Statistics

**Files:**
- Modify: `analysis/sentiment_mechanism.py`
- Modify: `tests/test_sentiment_mechanism_ablation.py`

**Interfaces:**
- Produces: `load_published_results(path: Path, condition: str) -> pd.DataFrame`.
- Produces: `load_raw_experiment_results(results_dir: Path, condition: str, stocks: Sequence[str], seeds: Sequence[int]) -> pd.DataFrame`.
- Produces: `pair_condition_results(control: pd.DataFrame, intervention: pd.DataFrame, hypothesis: str) -> pd.DataFrame`.
- Produces: `paired_stock_statistics(pairs: pd.DataFrame) -> pd.DataFrame`.
- Produces: `holm_adjust(p_values: Sequence[float]) -> list[float]`.

- [ ] **Step 1: Write failing loader/pairing tests**

Build shuffled fixture CSVs and assert identifiers:

```python
pairs = pair_condition_results(control, intervention, "H2")
self.assertEqual(list(pairs[["stock", "seed"]].itertuples(index=False, name=None)), sorted(expected_stock_seed_pairs))
self.assertEqual(len(pairs), len(expected_stock_seed_pairs) * 3 * 3)
```

Assert duplicate keys, missing pairs, wrong coverage/horizon, and non-finite metrics raise errors.

- [ ] **Step 2: Write failing known-statistics tests**

For deltas `[1, 2, 3, 4, 5]`, assert mean 3, sample standard deviation `sqrt(2.5)`, t `3 / (sqrt(2.5) / sqrt(5))`, dz `3 / sqrt(2.5)`, and the 95% Student-t CI. Test Holm `[0.01, 0.04, 0.03]` gives `[0.03, 0.06, 0.06]` in original order.

- [ ] **Step 3: Run analysis-core tests and verify failure**

```bash
conda run --no-capture-output -n ts-jepa python -m unittest tests.test_sentiment_mechanism_ablation.PairedAnalysisTest -v
```

Expected: failure because analysis functions are absent.

- [ ] **Step 4: Implement strict canonicalization and pairing**

Canonical columns are `condition, stock, seed, model, metric, value, forecast_horizon, source_file`. Map `trend_accuracy` to `direction_accuracy`. Retain `TS-JEPA/random`, `TS-JEPA/local_long`, and `GRU/random`; map report names. Read sibling preprocessing metadata. Production analysis requires ten stocks and seeds 42–51.

Outer-merge on stock, seed, model, metric with `validate="one_to_one"`; reject non-pairs. Store control, intervention, delta, percent delta, improved. Negative errors and positive direction deltas improve. Reject zero percentage denominators.

- [ ] **Step 5: Implement Student-t inference with standard library**

Use `math.lgamma`, continued-fraction regularized incomplete beta, Student-t CDF identity, and bounded bisection for 95% critical values. All-zero variance gives t=0/p=1; nonzero constant gives signed infinity/p=0. Average seeds within stock and require `n == 10`. dz uses sample standard deviation. Holm uses sorted multiplication, cumulative maxima, original order, cap one.

- [ ] **Step 6: Run tests and commit**

```bash
conda run --no-capture-output -n ts-jepa python -m unittest tests.test_sentiment_mechanism_ablation.PairedAnalysisTest -v
git add analysis/sentiment_mechanism.py tests/test_sentiment_mechanism_ablation.py
git commit -m "feat: add paired sentiment mechanism statistics"
```

Expected: PASS and order-independent outputs.

---

### Task 6: Generate the Deferred Analysis Package and Report

**Files:**
- Create: `analyze_sentiment_mechanisms.py`
- Modify: `analysis/sentiment_mechanism.py`
- Modify: `tests/test_sentiment_mechanism_ablation.py`

**Interfaces:**
- Produces: `run_mechanism_analysis(args: argparse.Namespace) -> Path | None`.
- Produces: `main(argv: Sequence[str] | None = None) -> int`.
- Produces: `data/mechanism_summary.csv`, `data/per_stock_deltas.csv`, `data/per_seed_deltas.csv`, `data/h1_short_horizon_results.csv`, `provenance/experiment_manifest.json`, and `sentiment_mechanism_report.md`.
- Guarantees: missing inputs return zero, print the exact message, create no package.

- [ ] **Step 1: Write failing missing-results test**

```python
output = io.StringIO()
with redirect_stdout(output):
    status = analysis_main(["--output-root", str(output_root)])
self.assertEqual(status, 0)
self.assertEqual(output.getvalue().strip(), "Experiment results not found; run the corresponding experiment first.")
self.assertFalse(output_root.exists())
```

- [ ] **Step 2: Write failing complete-package test**

Create complete synthetic inputs for ten stocks, ten seeds, three models, and three metrics, including manifests and preprocessing metadata. Pass `--run-id test_run`; assert exact files/columns, executive header, H1 horizon 1, and provenance branch/commit/config/coverage/changes/paths/H3 stats.

- [ ] **Step 3: Run report tests and verify failure**

```bash
conda run --no-capture-output -n ts-jepa python -m unittest tests.test_sentiment_mechanism_ablation.MechanismReportTest -v
```

Expected: failure because CLI/writers are absent.

- [ ] **Step 4: Implement CLI defaults and clean preflight**

Default to four `results/<config-stem>` roots and two published control CSVs. Support `--output-root thesis_results/sentiment_mechanism_ablation`, `--run-id RUN_ID`, and `--validate-configs`. Config-only mode prints validation JSON. Full analysis preflights every path and coverage before output creation.

- [ ] **Step 5: Implement tables, Holm groups, and verdicts**

Write per-seed, per-stock, summary, and H1 tables. Apply Holm within hypothesis across model × MSE/MAE; direction is secondary.

`mechanism_summary.csv` must contain `hypothesis`, `intervention`, `control`, `model`, `metric`, `control_mean`, `intervention_mean`, `absolute_delta`, `percent_delta`, `stock_win_count`, `stock_total`, `seed_pair_win_rate`, `paired_t`, `paired_p`, `paired_p_holm`, `cohens_dz`, `ci95_low`, `ci95_high`, and `verdict`. `per_stock_deltas.csv` contains hypothesis/stock/model/metric/control/intervention/delta/percent delta/seeds; `per_seed_deltas.csv` contains every paired stock-seed observation without treating those rows as independent inferential units.

A model is supported with favorable MSE/MAE means, at least six stock wins for both, and one adjusted p below 0.05. Both unfavorable means means not supported; otherwise inconclusive. A hypothesis is supported if a primary JEPA model is supported and neither primary is not supported; it is not supported when neither primary is supported and at least one is not supported; otherwise inconclusive. GRU does not decide the hypothesis verdict.

- [ ] **Step 6: Implement provenance and Markdown**

Write atomically. Sections are executive table, Baseline verification, H1, H2, H3, Statistical summary, Overall conclusion, and Thesis-ready interpretation. Separate observations, statistics, and mechanisms.

- [ ] **Step 7: Run report/analysis tests and commit**

```bash
conda run --no-capture-output -n ts-jepa python -m unittest tests.test_sentiment_mechanism_ablation.MechanismReportTest tests.test_thesis_results_analysis tests.test_package_experiment_results -v
git add analyze_sentiment_mechanisms.py analysis/sentiment_mechanism.py tests/test_sentiment_mechanism_ablation.py
git commit -m "feat: prepare sentiment mechanism analysis package"
```

Expected: PASS; missing-result path creates nothing.

---

### Task 7: Document, Verify, and Dry-Run Without Training

**Files:**
- Modify: `README.md`
- Modify: `doc/configuration.md`
- Modify: `tests/test_sentiment_mechanism_ablation.py`

**Interfaces:**
- Documents exact dry-run, user-only full-run, and post-experiment analysis commands.
- Validates the repository and all four structured dry-runs.

- [ ] **Step 1: Add documentation assertions and update docs**

Test stable command/key fragments. Document:

```json
"features": {
  "sentiment": {
    "enabled": true,
    "columns": ["sentiment_mean_z"],
    "normalization": "train_zscore"
  }
},
"downstream": {
  "epochs": 501,
  "forecast_horizon": 1
}
```

Explain horizon fallback, per-stock train-only fitting, and dry-run's no-write/no-subprocess guarantee.

- [ ] **Step 2: Run CLI help and config validation**

```bash
conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py --help
conda run --no-capture-output -n ts-jepa python analyze_sentiment_mechanisms.py --help
conda run --no-capture-output -n ts-jepa python analyze_sentiment_mechanisms.py --validate-configs
```

Expected: zero exits and config JSON `valid: true`.

- [ ] **Step 3: Run the complete test suite**

```bash
conda run --no-capture-output -n ts-jepa python -m unittest discover -s tests -v
```

Expected: all tests pass and none enters optimization.

- [ ] **Step 4: Run all four required dry-runs**

```bash
conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py --config config/experiments/top10_h1_without_sentiment.json --dry-run
conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py --config config/experiments/top10_h1_with_sentiment.json --dry-run
conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py --config config/experiments/top10_sentiment_has_news.json --dry-run
conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py --config config/experiments/top10_sentiment_zscore.json --dry-run
```

Expected: each reports `single-dim`, exact 10/10 coverage, expected horizon/dimension, output directory, and `training_disabled: true`; no result root is created.

- [ ] **Step 5: Verify missing results and published-package integrity**

```bash
conda run --no-capture-output -n ts-jepa python analyze_sentiment_mechanisms.py --run-id top10_seeds42_51
git diff 37afbf9 -- thesis_results/top10_with_sentiment/5b8f3897bf23-02add88f32d5 thesis_results/top10_without_sentiment/2fab810c1e1d-d0fb2944255b
git diff --check
git status --short
git diff --stat 5e28d85..HEAD
```

Expected: exact missing-results message, no package, no published-package diff, no whitespace errors, and only intended files.

- [ ] **Step 6: Commit documentation and final test adjustments**

```bash
git add README.md doc/configuration.md tests/test_sentiment_mechanism_ablation.py
git commit -m "docs: document sentiment mechanism workflow"
```

- [ ] **Step 7: Prepare the final manual handoff**

List four concrete dry-run commands, four concrete full-run commands with `--dry-run` omitted, and:

```bash
conda run --no-capture-output -n ts-jepa python analyze_sentiment_mechanisms.py --run-id top10_seeds42_51
```

Report expected result roots, exact files changed, tests and dry-run evidence, confirm no full experiment ran, and state: `No scientific conclusion has been drawn from dry-run validation.`
