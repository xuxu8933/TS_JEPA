# Leakage-Safe JEPA/MAE Lambda Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible validation-only sweep that selects one shared JEPA/MAE weight pair from five candidates and generates the final ten-stock, ten-seed experiment config without exposing test data during selection.

**Architecture:** A result-affecting validation-only mode flows through the stock runner, dual-loss wrapper, and downstream evaluator, producing one structured summary per stock/seed/strategy run before test-loader construction or baseline evaluation. A separate pure-Python sweep module materializes complete candidate configs, verifies provenance and exact 90-run coverage, ranks candidates deterministically, and generates a full final config; a thin CLI runs each candidate through the existing resumable runner.

**Tech Stack:** Python 3.11, PyTorch, `argparse`, standard-library `json`, `csv`, `hashlib`, `statistics`, `subprocess`, and `unittest`.

**Spec:** `docs/plans/2026-08-25-lambda-sweep-design.md`

## Global Constraints

- Compare exactly `(2.0, 0.0)`, `(1.5, 0.5)`, `(1.0, 1.0)`, `(0.5, 1.5)`, and `(0.0, 2.0)`; every pair has total weight 2.0.
- Screen exactly stocks `NVDA`, `AAPL`, `MSFT`, seeds `42`, `43`, `44`, and strategies `random`, `local_long`, requiring all 90 validation summaries.
- Select one shared pair by lowest mean best-validation MSE, then lowest sample standard deviation, then smallest absolute weight difference, then declaration order.
- Never construct the held-out test loader, train GRU, evaluate deterministic baselines, generate model-comparison outputs, or generate combined plots during validation-only screening.
- Preserve pretraining, downstream training, validation, loss, masking, split, target, normalization, checkpoint-selection, and seed semantics.
- Deep-copy every source configuration field; change only the screening/final fields approved in the design.
- Treat validation-only mode as result-affecting experiment identity; keep `max_parallel_jobs` runtime-only.
- Bound candidate execution at `max_parallel_jobs = 2` and run candidate configs sequentially.
- Never modify or stage `config/experiments/top10_with_sentiment_jepa_lam_1_mae_lam_1.json`; it contains a pre-existing user change.
- Refuse partial, duplicated, malformed, non-finite, or provenance-incompatible selection input.
- Write JSON and CSV completion artifacts atomically.

## File Structure

- Modify `main/utils.py`: parse downstream validation-only and objective-provenance arguments.
- Modify `eval_dual_loss.py`: accept and forward validation-only mode, strategy, weights, and loss names.
- Modify `eval_forecast_prequential_with_baselines_gru_volume.py`: conditionally avoid test data and emit the best-validation summary.
- Modify `config/file_options.py`: map optional nested output setting `validation_only`.
- Modify `config/experiments/template_experiment.jsonc`: document the new runner output setting.
- Modify `run_top_nasdaq100_stocks.py`: forward the mode, distinguish completion artifacts, record manifests, and skip combined plots.
- Create `lambda_sweep.py`: validate definitions, materialize configs, verify coverage/provenance, rank candidates, and generate outputs.
- Create `run_lambda_sweep.py`: orchestrate candidate runner processes and selection.
- Create `config/sweeps/top10_with_sentiment_lambda_screen.json`: committed five-candidate sweep definition.
- Create `tests/test_lambda_sweep.py`: focused downstream, runner, expansion, selection, and CLI tests.
- Modify `doc/configuration.md`: document validation-only and sweep commands.

---

### Task 1: Add Test-Isolated Validation-Only Downstream Evaluation

**Files:**
- Modify: `main/utils.py:46-350`
- Modify: `eval_dual_loss.py:191-700`
- Modify: `eval_forecast_prequential_with_baselines_gru_volume.py:1-120,1810-2440`
- Create: `tests/test_lambda_sweep.py`

**Interfaces:**
- Consumes: `--validation-only`, `--mask-strategy`, `--lambda-jepa`, `--lambda-mae`, `--jepa-loss`, and `--mae-loss` from `eval_dual_loss.py`.
- Produces: `build_validation_summary(config: dict, *, best_epoch: int, best_val_score: float, best_val_mse: float, best_val_mae: float, best_val_trend_accuracy: float) -> dict`.
- Produces: `write_json_atomic(path: Path, value: dict) -> None`.
- Produces: `load_test_loader_unless_validation_only(validation_only: bool, loader_factory: Callable, loader_kwargs: dict)` returning `None` or the test loader.
- Produces: `<run_dir>/validation_summary.json` with schema version 1.

- [ ] **Step 1: Write failing parser, forwarding, loader-isolation, and summary tests**

Create `tests/test_lambda_sweep.py` with these initial tests:

```python
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from eval_dual_loss import build_eval_argv, parse_args as parse_dual_eval_args
from eval_forecast_prequential_with_baselines_gru_volume import (
    build_validation_summary,
    load_test_loader_unless_validation_only,
    write_json_atomic,
)


class ValidationOnlyDownstreamTest(unittest.TestCase):
    def test_dual_eval_forwards_validation_mode_and_objective_provenance(self):
        args, passthrough = parse_dual_eval_args(
            argv=[
                "--data", "NVDA",
                "--mask-strategy", "local_long",
                "--lambda-jepa", "1.5",
                "--lambda-mae", "0.5",
                "--jepa-loss", "mse",
                "--mae-loss", "smooth_l1",
                "--validation-only",
                "--seed", "43",
            ]
        )
        with patch(
            "eval_dual_loss.resolve_dual_checkpoint_path",
            return_value="/missing/checkpoint.pt",
        ):
            argv, _ = build_eval_argv(args, passthrough)
        self.assertIn("--validation-only", argv)
        self.assertEqual(argv[argv.index("--mask-strategy") + 1], "local_long")
        self.assertEqual(argv[argv.index("--lambda-jepa") + 1], "1.5")
        self.assertEqual(argv[argv.index("--lambda-mae") + 1], "0.5")
        self.assertEqual(argv[argv.index("--jepa-loss") + 1], "mse")
        self.assertEqual(argv[argv.index("--mae-loss") + 1], "smooth_l1")
        self.assertEqual(argv[argv.index("--seed") + 1], "43")

    def test_validation_only_does_not_construct_test_loader(self):
        loader_factory = Mock(return_value="test-loader")
        result = load_test_loader_unless_validation_only(
            True,
            loader_factory,
            {"path": "unused"},
        )
        self.assertIsNone(result)
        loader_factory.assert_not_called()

    def test_final_mode_constructs_test_loader(self):
        loader_factory = Mock(return_value="test-loader")
        result = load_test_loader_unless_validation_only(
            False,
            loader_factory,
            {"path": "data.csv"},
        )
        self.assertEqual(result, "test-loader")
        loader_factory.assert_called_once_with(split="test", path="data.csv")

    def test_summary_records_best_epoch_and_metrics_atomically(self):
        config = {
            "data": "NVDA",
            "seed": 43,
            "mask_strategy": "local_long",
            "lambda_jepa": 1.5,
            "lambda_mae": 0.5,
            "jepa_loss": "mse",
            "mae_loss": "smooth_l1",
            "trend_selection_weight": 0.0,
            "pretrain_checkpoint_path": "/checkpoints/model.pt",
            "checkpoint_selection": "fixed_pretraining_epoch",
        }
        summary = build_validation_summary(
            config,
            best_epoch=7,
            best_val_score=0.11,
            best_val_mse=0.11,
            best_val_mae=0.22,
            best_val_trend_accuracy=0.61,
        )
        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(summary["evaluation_mode"], "validation_only")
        self.assertEqual(summary["best_epoch"], 7)
        self.assertEqual(summary["best_val_mse"], 0.11)
        self.assertEqual(summary["stock"], "NVDA")
        self.assertEqual(summary["seed"], 43)
        self.assertEqual(summary["mask_strategy"], "local_long")

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "validation_summary.json"
            write_json_atomic(path, summary)
            self.assertEqual(json.loads(path.read_text()), summary)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())
```

- [ ] **Step 2: Run the new tests and verify they fail for missing interfaces**

Run:

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest \
  tests.test_lambda_sweep.ValidationOnlyDownstreamTest -v
```

Expected: FAIL because the validation-only parser option and evaluator helpers do not exist.

- [ ] **Step 3: Add downstream argument parsing and dual-wrapper forwarding**

In `main/utils.py`, add parser actions and copy their values into `config`:

```python
parser.add_argument(
    "--validation-only",
    action=argparse.BooleanOptionalAction,
    default=config.get("validation_only", False),
)
parser.add_argument(
    "--mask-strategy",
    choices=("random", "local_long", "future_block", "causal_multiblock"),
    default=config.get("mask_strategy", "random"),
)
parser.add_argument("--lambda-jepa", type=float, default=config.get("lambda_jepa", 1.0))
parser.add_argument("--lambda-mae", type=float, default=config.get("lambda_mae", 1.0))
parser.add_argument(
    "--jepa-loss",
    choices=("mse", "l1", "smooth_l1"),
    default=config.get("jepa_loss", "mse"),
)
parser.add_argument(
    "--mae-loss",
    choices=("mse", "l1", "smooth_l1"),
    default=config.get("mae_loss", "mse"),
)
```

Assign all six values after parsing. In `eval_dual_loss.py`, add a boolean optional `--validation-only` argument, then extend `eval_argv` with the resolved strategy, both weights, and both loss names. Append `--validation-only` only when true. Keep the existing passthrough seed unchanged.

- [ ] **Step 4: Implement evaluator helpers and finite-value validation**

Add these interfaces near the existing provenance helpers:

```python
from pathlib import Path


VALIDATION_SUMMARY_FILENAME = "validation_summary.json"


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    temporary_path.replace(path)


def load_test_loader_unless_validation_only(
    validation_only,
    loader_factory,
    loader_kwargs,
):
    if validation_only:
        return None
    return loader_factory(split="test", **loader_kwargs)


def build_validation_summary(
    config,
    *,
    best_epoch,
    best_val_score,
    best_val_mse,
    best_val_mae,
    best_val_trend_accuracy,
):
    metrics = {
        "best_val_score": float(best_val_score),
        "best_val_mse": float(best_val_mse),
        "best_val_mae": float(best_val_mae),
        "best_val_trend_accuracy": float(best_val_trend_accuracy),
    }
    if best_epoch is None or int(best_epoch) < 0:
        raise ValueError("Validation-only evaluation did not select a best epoch")
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Validation summary metrics must be finite")
    return {
        "schema_version": 1,
        "evaluation_mode": "validation_only",
        "stock": str(config["data"]).upper(),
        "seed": int(config["seed"]),
        "mask_strategy": str(config["mask_strategy"]),
        "lambda_jepa": float(config["lambda_jepa"]),
        "lambda_mae": float(config["lambda_mae"]),
        "jepa_loss": str(config["jepa_loss"]),
        "mae_loss": str(config["mae_loss"]),
        "selection_metric": "val_mse_plus_weighted_trend_error",
        "trend_selection_weight": float(config["trend_selection_weight"]),
        "best_epoch": int(best_epoch),
        **metrics,
        "pretrain_checkpoint_path": config.get("pretrain_checkpoint_path"),
        "checkpoint_selection": config.get("checkpoint_selection"),
    }
```

Import `math` and retain runtime provenance in the existing preprocessing metadata rather than duplicating mutable environment fields into the selector's metric identity.

- [ ] **Step 5: Reorder loader creation and stop before test/baseline work**

Build one `loader_kwargs` dictionary from the existing repeated loader arguments. Create the validation loader with:

```python
val_loader = get_evaluation_loaders(split="val", **loader_kwargs)
test_loader = load_test_loader_unless_validation_only(
    bool(config["validation_only"]),
    get_evaluation_loaders,
    loader_kwargs,
)
```

Only add test sample/date metadata when `test_loader is not None`. Track the exact best epoch and associated metrics inside the existing strict-improvement branch:

```python
best_epoch = None
best_val_mse = None
best_val_mae = None
best_val_trend_accuracy = None

if val_score < best_val_score:
    best_epoch = epoch
    best_val_mse = val_mse
    best_val_mae = val_mae
    best_val_trend_accuracy = val_trend_acc
```

Immediately after writing `loss.txt` and restoring the best states, write the summary and exit successfully:

```python
if config["validation_only"]:
    summary = build_validation_summary(
        config,
        best_epoch=best_epoch,
        best_val_score=best_val_score,
        best_val_mse=best_val_mse,
        best_val_mae=best_val_mae,
        best_val_trend_accuracy=best_val_trend_accuracy,
    )
    write_json_atomic(Path(results_dir) / VALIDATION_SUMMARY_FILENAME, summary)
    print("Validation summary saved to:", Path(results_dir) / VALIDATION_SUMMARY_FILENAME)
    raise SystemExit(0)
```

This branch must remain above `print("Start GRU baseline training")` and every reference to `test_loader` after training.

- [ ] **Step 6: Run focused downstream tests**

Run:

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest \
  tests.test_lambda_sweep.ValidationOnlyDownstreamTest -v
```

Expected: all downstream tests PASS.

- [ ] **Step 7: Commit downstream validation-only evaluation**

```bash
git add main/utils.py eval_dual_loss.py \
  eval_forecast_prequential_with_baselines_gru_volume.py \
  tests/test_lambda_sweep.py
git commit -m "feat: add validation-only downstream evaluation"
```

Expected: only the listed files are committed; the source experiment config remains unstaged.

---

### Task 2: Make the Stock Runner Resume Validation-Only Runs Safely

**Files:**
- Modify: `config/file_options.py:115-550`
- Modify: `config/experiments/template_experiment.jsonc`
- Modify: `run_top_nasdaq100_stocks.py:18-520,638-855,975-1170,1171-1320`
- Modify: `tests/test_lambda_sweep.py`
- Modify: `tests/test_top10_nasdaq_mask_comparison.py`

**Interfaces:**
- Consumes: `validation_summary.json` schema version 1 from Task 1.
- Produces: runner namespace field `validation_only: bool` from CLI or `[runner].output.validation_only`.
- Produces: `validation_summary_complete(args, strategy: str, stock: str, seed: int, run_dir: Path) -> bool`.
- Produces: validation-only run manifests with `validation_summary_file` and no comparison files.

- [ ] **Step 1: Add failing runner schema, identity, forwarding, and resume tests**

Append a `ValidationOnlyRunnerTest` class to `tests/test_lambda_sweep.py`. Use a temporary nested config with `runner.output.validation_only: true` and assert:

```python
args = parse_stock_runner_args(["--config", str(config_path)])
self.assertTrue(args.validation_only)
self.assertNotEqual(
    experiment_config_signature(args),
    experiment_config_signature(
        parse_stock_runner_args(["--config", str(config_path), "--no-validation-only"])
    ),
)
commands = build_stock_commands(
    args,
    "NVDA",
    seed=42,
    strategy="random",
    results_dir=Path(args.results_dir) / "random",
)
self.assertNotIn("--validation-only", commands[0])
self.assertIn("--validation-only", commands[-1])
```

Create a compatible run directory containing a schema-valid validation summary but no model-comparison files. Write the matching experiment and run manifests, then assert `plan_incremental_execution(...)["tasks"] == []`. Add the inverse assertion for final mode, which must still report the run missing.

In `tests/test_top10_nasdaq_mask_comparison.py`, extend the template test to assert the new output key is present and parses false by default.

- [ ] **Step 2: Run the runner tests and verify failures**

Run:

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest \
  tests.test_lambda_sweep.ValidationOnlyRunnerTest \
  tests.test_top10_nasdaq_mask_comparison.StockMaskComparisonTest.test_commented_template_documents_every_config_input -v
```

Expected: FAIL because the nested output key, runner option, forwarding, and summary completion logic are missing.

- [ ] **Step 3: Add config mapping and runner option**

Allow the optional key in `flatten_runner_options`:

```python
_validate_keys(
    output,
    {"skip_combined_plot", "validation_only"},
    "[runner].output",
    config_path,
    required=("skip_combined_plot",),
)
flattened["skip_combined_plot"] = output["skip_combined_plot"]
if "validation_only" in output:
    flattened["validation_only"] = output["validation_only"]
```

Add `"validation_only": false` to the output block in the JSONC template. Add this runner argument:

```python
parser.add_argument(
    "--validation-only",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Stop after downstream validation and do not load or evaluate the test split.",
)
```

Do not add `validation_only` to `NON_RESULT_CONFIG_KEYS`.

- [ ] **Step 4: Validate completion summaries against run identity**

Implement `validation_summary_complete` to load exactly
`run_dir / "validation_summary.json"`, return false when absent, and raise a
`RuntimeError` for unreadable or invalid content. Require:

```python
expected = {
    "schema_version": 1,
    "evaluation_mode": "validation_only",
    "stock": stock.upper(),
    "seed": int(seed),
    "mask_strategy": strategy,
    "lambda_jepa": float(args.lambda_jepa),
    "lambda_mae": float(args.lambda_mae),
    "jepa_loss": args.jepa_loss,
    "mae_loss": args.mae_loss,
}
```

Require exact values for those identity fields, a nonnegative integer
`best_epoch`, and finite numeric values for `best_val_score`, `best_val_mse`,
`best_val_mae`, and `best_val_trend_accuracy`. Dispatch output completion in
`downstream_run_status`:

```python
outputs_complete = (
    validation_summary_complete(args, strategy, stock, seed, run_dir)
    if args.validation_only
    else _comparison_outputs_complete(run_dir)
)
```

When scanning a result root without an experiment manifest, consider either a
model-comparison TXT or `validation_summary.json` an existing run output.

- [ ] **Step 5: Forward mode and update manifests/plot behavior**

Append `--validation-only` only to the downstream command when the runner mode
is enabled. Also pass `--jepa-loss` and `--mae-loss` so the summary records the
same objective names as the candidate identity.

Add `evaluation_mode` to experiment and text manifests. In a complete run
manifest, record either:

```python
"comparison_files": comparison_files,
"validation_summary_file": (
    "validation_summary.json" if args.validation_only else None
),
```

Skip combined plotting with:

```python
if not args.validation_only and not args.skip_combined_plot:
```

Keep all current final-mode paths and completion rules unchanged.

- [ ] **Step 6: Run focused runner and existing compatibility tests**

Run:

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest \
  tests.test_lambda_sweep.ValidationOnlyRunnerTest \
  tests.test_top10_nasdaq_mask_comparison -v
```

Expected: all tests PASS.

- [ ] **Step 7: Commit validation-aware runner support**

```bash
git add config/file_options.py config/experiments/template_experiment.jsonc \
  run_top_nasdaq100_stocks.py tests/test_lambda_sweep.py \
  tests/test_top10_nasdaq_mask_comparison.py
git commit -m "feat: resume validation-only stock runs"
```

Expected: a focused runner/config commit without the user's experiment config.

---

### Task 3: Materialize Complete Candidate Configs with Provenance

**Files:**
- Create: `lambda_sweep.py`
- Create: `config/sweeps/top10_with_sentiment_lambda_screen.json`
- Modify: `tests/test_lambda_sweep.py`

**Interfaces:**
- Consumes: a JSON sweep definition and the complete base experiment JSON.
- Produces: `load_sweep_definition(path: Path) -> dict`.
- Produces: `sha256_file(path: Path) -> str`.
- Produces: `materialize_sweep(sweep_path: Path, artifact_root: Path | None = None) -> dict` returning and atomically writing `sweep_manifest.json`.
- Produces: five materialized candidate config paths and explicit expected summary paths.

- [ ] **Step 1: Write failing definition and materialization tests**

Add `SweepMaterializationTest` using a temporary full nested base config. Verify:

```python
source_before = base_path.read_bytes()
manifest = materialize_sweep(sweep_path, artifact_root)
self.assertEqual(len(manifest["candidates"]), 5)
self.assertEqual(manifest["required_summary_count"], 90)
self.assertEqual(base_path.read_bytes(), source_before)
```

For every materialized candidate, load the JSON and compare it recursively with
the source after removing only these approved paths:

```text
common.stocks
common.seeds
runner.execution.max_stocks
runner.execution.max_seeds
runner.execution.max_parallel_jobs
runner.objectives.jepa.weight
runner.objectives.mae.weight
runner.output.skip_combined_plot
runner.output.validation_only
```

Assert stocks/seeds/strategies, candidate order, total weight 2.0, two-job
execution, validation-only mode, skip-combined-plot, config hashes, and all 18
expected summary paths per candidate.

Add table-driven failures for duplicate candidate names, duplicate weights,
negative weights, a pair whose sum is not 2.0, duplicate stocks/seeds/strategies,
missing strategies in the base config, and an unsupported schema version.

- [ ] **Step 2: Run materialization tests and verify import failure**

Run:

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest \
  tests.test_lambda_sweep.SweepMaterializationTest -v
```

Expected: FAIL because `lambda_sweep.py` and the committed definition do not exist.

- [ ] **Step 3: Implement definition validation and hashing**

Create `lambda_sweep.py` with constants:

```python
SWEEP_SCHEMA_VERSION = 1
VALIDATION_SUMMARY_SCHEMA_VERSION = 1
SWEEP_MANIFEST_FILENAME = "sweep_manifest.json"
```

`load_sweep_definition` must resolve `base_config` relative to the sweep file,
validate exact object/list/scalar types, require nonempty unique stocks, seeds,
strategies, and candidates, require nonnegative finite weights with at least one
positive value, require `math.isclose(jepa + mae, 2.0, rel_tol=0.0,
abs_tol=1e-12)`, and require the declared shared selection policy:

```json
{
  "scope": "shared",
  "primary": "mean_best_val_mse",
  "tie_breakers": ["std_best_val_mse", "balanced_weights", "declaration_order"]
}
```

Implement SHA-256 over exact file bytes and atomic JSON/CSV helpers in this
module; do not reuse the evaluator helper across process-boundary concerns.

- [ ] **Step 4: Implement deterministic candidate expansion**

`materialize_sweep` must:

1. deep-copy the parsed source object for every candidate;
2. set the approved screening fields only;
3. retain both declared strategies and verify they are enabled in the source;
4. name configs `<sweep-name>__<candidate-name>.json` under `artifact_root`;
5. derive each result root with `results_dir_from_config(candidate_path)`;
6. list the exact 18 `<result>/<strategy>/<stock>/seed_<seed>/validation_summary.json` paths;
7. atomically write configs and a manifest with source path/hash, sweep path/hash,
   candidate declaration index, weights, config path/hash, result root, and
   expected summary paths;
8. refuse to replace an existing candidate config or manifest whose contents
   differ from deterministic regeneration.

Use repo-relative paths in the manifest when a path is below the repository and
absolute paths otherwise. Resolve paths before comparing them.

- [ ] **Step 5: Add the committed five-candidate sweep definition**

Create `config/sweeps/top10_with_sentiment_lambda_screen.json`:

```json
{
  "schema_version": 1,
  "name": "top10_with_sentiment_lambda_screen",
  "base_config": "../experiments/top10_with_sentiment_jepa_lam_1_mae_lam_1.json",
  "stocks": ["NVDA", "AAPL", "MSFT"],
  "seeds": [42, 43, 44],
  "strategies": ["random", "local_long"],
  "max_parallel_jobs": 2,
  "candidates": [
    {"name": "j2_m0", "lambda_jepa": 2.0, "lambda_mae": 0.0},
    {"name": "j1p5_m0p5", "lambda_jepa": 1.5, "lambda_mae": 0.5},
    {"name": "j1_m1", "lambda_jepa": 1.0, "lambda_mae": 1.0},
    {"name": "j0p5_m1p5", "lambda_jepa": 0.5, "lambda_mae": 1.5},
    {"name": "j0_m2", "lambda_jepa": 0.0, "lambda_mae": 2.0}
  ],
  "selection": {
    "scope": "shared",
    "primary": "mean_best_val_mse",
    "tie_breakers": [
      "std_best_val_mse",
      "balanced_weights",
      "declaration_order"
    ]
  }
}
```

- [ ] **Step 6: Run materialization tests and a real prepare smoke check**

Run:

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest \
  tests.test_lambda_sweep.SweepMaterializationTest -v
temporary_root=$(mktemp -d /tmp/ts-jepa-lambda-sweep-plan.XXXXXX)
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 - <<PY
from pathlib import Path
from lambda_sweep import materialize_sweep

manifest = materialize_sweep(
    Path("config/sweeps/top10_with_sentiment_lambda_screen.json"),
    Path("$temporary_root"),
)
assert manifest["required_summary_count"] == 90
assert len(manifest["candidates"]) == 5
print("materialized", manifest["required_summary_count"], "summaries")
PY
```

Expected: tests PASS and smoke output is `materialized 90 summaries`; no repository result artifacts are created.

- [ ] **Step 7: Commit sweep definition and expansion**

```bash
git add lambda_sweep.py config/sweeps/top10_with_sentiment_lambda_screen.json \
  tests/test_lambda_sweep.py
git commit -m "feat: materialize lambda sweep candidates"
```

Expected: candidate generation is complete and independently testable.

---

### Task 4: Enforce Exact Coverage and Select the Shared Pair

**Files:**
- Modify: `lambda_sweep.py`
- Modify: `tests/test_lambda_sweep.py`

**Interfaces:**
- Consumes: `sweep_manifest.json` and exactly 90 schema-valid `validation_summary.json` files.
- Produces: `collect_validation_rows(manifest: dict) -> list[dict]`.
- Produces: `rank_candidates(manifest: dict, rows: list[dict]) -> list[dict]`.
- Produces: `select_and_write_outputs(manifest_path: Path) -> dict` returning the selection manifest.
- Produces: `validation_rows.csv`, `candidate_ranking.csv`, `selection_manifest.json`, and the full selected final config.

- [ ] **Step 1: Write failing coverage, ranking, and final-config tests**

Add `LambdaSelectionTest`. Build a materialized temporary sweep and generate all
90 expected summaries with candidate means deliberately ordered. Assert the
lowest mean wins even if another candidate has lower variance. Then add separate
datasets proving each tie-breaker in order:

```python
ranking = rank_candidates(manifest, rows)
self.assertEqual(ranking[0]["candidate_name"], expected_winner)
self.assertEqual(ranking[0]["run_count"], 18)
```

Use subtests that delete one expected summary, create one unexpected summary,
duplicate one logical identity, change one stock/seed/strategy/weight/loss field,
write invalid JSON, and write `NaN`/`Infinity`. Each must raise `ValueError` or
`RuntimeError` with the offending candidate/path.

After selection, assert:

```python
self.assertEqual(selection["required_summary_count"], 90)
self.assertEqual(selection["observed_summary_count"], 90)
self.assertEqual(selection["winner"]["candidate_name"], expected_winner)
self.assertTrue(Path(selection["final_config_path"]).is_file())
```

Load the final config and verify all ten original stocks and seeds remain, both
strategies remain enabled, limits equal ten, validation-only is false, the
source `skip_combined_plot` value is restored, concurrency is two, only the
winning weights change, and the source bytes are unchanged.

- [ ] **Step 2: Run selection tests and verify missing interfaces**

Run:

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest \
  tests.test_lambda_sweep.LambdaSelectionTest -v
```

Expected: FAIL because collection, ranking, and final generation are missing.

- [ ] **Step 3: Implement strict summary validation and coverage checking**

For every expected path, require the Task 1 schema and exact equality with the
candidate record's stock, seed, strategy, weights, and source loss names. Require
finite numeric best metrics and a nonnegative integer best epoch. Reject boolean
values where integer/numeric values are expected.

Discover every `validation_summary.json` below each candidate result root and
compare resolved discovered paths with resolved expected paths. Raise on missing
or extra files before ranking. Build row dictionaries that include candidate
name/index, stock, seed, strategy, weights, best epoch, selection score, MSE,
MAE, trend accuracy, and source path.

Before collecting, recompute and compare the sweep definition, source config,
and every candidate config hash stored in `sweep_manifest.json`.

- [ ] **Step 4: Implement deterministic aggregation and ranking**

Group rows by candidate name, require 18 rows per candidate, then calculate:

```python
mean_mse = statistics.fmean(row["best_val_mse"] for row in candidate_rows)
std_mse = statistics.stdev(row["best_val_mse"] for row in candidate_rows)
balance = abs(candidate["lambda_jepa"] - candidate["lambda_mae"])
sort_key = (mean_mse, std_mse, balance, candidate["declaration_index"])
```

Sort ascending without a fuzzy tolerance. Include all four ranking fields and
one-based rank in the candidate CSV/manifest.

- [ ] **Step 5: Generate atomic selection outputs and final config**

Write the row and ranking CSV files through temporary sibling files followed by
`Path.replace`. Build the final config from a fresh deep copy of the source,
not from a screening candidate. Set:

```python
final_config["runner"]["objectives"]["jepa"]["weight"] = winner["lambda_jepa"]
final_config["runner"]["objectives"]["mae"]["weight"] = winner["lambda_mae"]
final_config["runner"]["execution"]["max_stocks"] = len(final_config["common"]["stocks"])
final_config["runner"]["execution"]["max_seeds"] = len(final_config["common"]["seeds"])
final_config["runner"]["execution"]["max_parallel_jobs"] = 2
final_config["runner"]["output"]["validation_only"] = False
```

Restore `skip_combined_plot` from the source. Name the artifact
`<sweep-name>__selected__<candidate-name>.json`. Write the selection manifest
last so it is the completion marker, including all hashes, rules, coverage,
ranking, winner, and output paths.

- [ ] **Step 6: Run selection and all lambda-sweep tests**

Run:

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest \
  tests.test_lambda_sweep -v
```

Expected: all tests PASS.

- [ ] **Step 7: Commit strict selection**

```bash
git add lambda_sweep.py tests/test_lambda_sweep.py
git commit -m "feat: select shared lambda from validation"
```

Expected: selection is independently testable without launching training.

---

### Task 5: Add the Resumable Sweep CLI and User Documentation

**Files:**
- Create: `run_lambda_sweep.py`
- Modify: `tests/test_lambda_sweep.py`
- Modify: `doc/configuration.md`

**Interfaces:**
- Consumes: `--sweep PATH`, optional `--artifact-root PATH`, and mutually exclusive `--prepare-only`, `--select-only`, or default execute-and-select mode.
- Produces: sequential candidate calls to `run_top_nasdaq100_stocks.py --config <materialized-config>`; each call internally uses at most two jobs.
- Produces: a printed selected-config path after complete selection.

- [ ] **Step 1: Write failing CLI orchestration tests**

Patch `run_lambda_sweep.materialize_sweep`, `subprocess.run`, and
`select_and_write_outputs`. Assert default mode invokes all five candidate
commands in declaration order with `check=True`, then selects once. Assert a
failed subprocess prevents later candidates and selection. Assert:

- `--prepare-only` materializes but neither executes nor selects;
- `--select-only` does not rematerialize or execute and selects the existing
  `<artifact-root>/sweep_manifest.json`;
- `--dry-run` materializes and prints runner commands with `--dry-run --verbose`
  but does not select absent summaries;
- invalid combinations such as `--select-only --dry-run` are rejected by the
  argument parser.

- [ ] **Step 2: Run CLI tests and verify import failure**

Run:

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest \
  tests.test_lambda_sweep.LambdaSweepCliTest -v
```

Expected: FAIL because `run_lambda_sweep.py` is missing.

- [ ] **Step 3: Implement the thin orchestration CLI**

Create `run_lambda_sweep.py` with `parse_args(argv=None)` and `main(argv=None)`.
Derive the default artifact root as:

```python
repo_root / "results" / "lambda_sweeps" / Path(args.sweep).stem
```

Default execution calls each candidate sequentially:

```python
command = [
    sys.executable,
    "-u",
    str(repo_root / "run_top_nasdaq100_stocks.py"),
    "--config",
    candidate["config_path"],
]
subprocess.run(command, cwd=repo_root, check=True)
```

Do not launch candidates concurrently; their generated configs already request
two independent task chains on the single GPU. Let the stock runner's existing
manifests resume completed runs. In dry-run mode append `--dry-run --verbose`,
print commands, execute them only as runner dry runs, and return before
selection. Print artifact root, required coverage, winner, and final-config path
in normal/select-only modes.

- [ ] **Step 4: Document commands and leakage guarantees**

Add a lambda-sweep section to `doc/configuration.md` containing:

```bash
# Inspect generated candidate configs and 90-run coverage without training
python run_lambda_sweep.py \
  --sweep config/sweeps/top10_with_sentiment_lambda_screen.json \
  --prepare-only

# Print the underlying runner commands
python run_lambda_sweep.py \
  --sweep config/sweeps/top10_with_sentiment_lambda_screen.json \
  --dry-run

# Run/resume all candidates and select the shared pair
python run_lambda_sweep.py \
  --sweep config/sweeps/top10_with_sentiment_lambda_screen.json

# Re-run selection from complete existing summaries
python run_lambda_sweep.py \
  --sweep config/sweeps/top10_with_sentiment_lambda_screen.json \
  --select-only
```

State explicitly that screening never constructs the test loader, that
selection requires all 90 runs, and that the generated final config is the only
one intended for full test evaluation.

- [ ] **Step 5: Run CLI tests and non-training command smoke checks**

Run:

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest \
  tests.test_lambda_sweep.LambdaSweepCliTest -v
temporary_root=$(mktemp -d /tmp/ts-jepa-lambda-sweep-cli.XXXXXX)
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 run_lambda_sweep.py \
  --sweep config/sweeps/top10_with_sentiment_lambda_screen.json \
  --artifact-root "$temporary_root" \
  --prepare-only
test "$(find "$temporary_root" -maxdepth 1 -name '*.json' | wc -l)" -eq 6
```

Expected: CLI tests PASS; the smoke check creates five candidate configs and one sweep manifest outside the repository without starting training.

- [ ] **Step 6: Commit CLI and documentation**

```bash
git add run_lambda_sweep.py tests/test_lambda_sweep.py doc/configuration.md
git commit -m "feat: orchestrate leakage-safe lambda sweep"
```

Expected: the end-to-end workflow is documented and runnable.

---

### Task 6: Verify the Integrated Workflow and Research Boundaries

**Files:**
- Inspect: all files changed in Tasks 1-5
- Inspect but do not stage: `config/experiments/top10_with_sentiment_jepa_lam_1_mae_lam_1.json`
- Produce outside repository: prepare-only smoke artifacts

**Interfaces:**
- Consumes: all completed task commits.
- Produces: test evidence, clean scoped diff, and a ready-to-run command; it does not launch the 90 long-running training jobs during implementation verification.

- [ ] **Step 1: Run focused lambda and runner suites**

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest \
  tests.test_lambda_sweep \
  tests.test_top10_nasdaq_mask_comparison \
  tests.test_unified_dual_loss -v
```

Expected: all focused tests PASS.

- [ ] **Step 2: Run the complete repository test suite**

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest discover -s tests -v
```

Expected: all tests PASS with no regression in the prior 100-test suite or the new lambda tests.

- [ ] **Step 3: Materialize and validate the real sweep outside the repository**

```bash
verification_root=$(mktemp -d /tmp/ts-jepa-lambda-sweep-verify.XXXXXX)
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 run_lambda_sweep.py \
  --sweep config/sweeps/top10_with_sentiment_lambda_screen.json \
  --artifact-root "$verification_root" \
  --prepare-only
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 - <<PY
import json
from pathlib import Path

root = Path("$verification_root")
manifest = json.loads((root / "sweep_manifest.json").read_text())
assert manifest["required_summary_count"] == 90
assert len(manifest["candidates"]) == 5
assert sum(len(item["expected_summary_paths"]) for item in manifest["candidates"]) == 90
for item in manifest["candidates"]:
    candidate = json.loads(Path(item["config_path"]).read_text())
    assert candidate["runner"]["output"]["validation_only"] is True
    assert candidate["runner"]["execution"]["max_parallel_jobs"] == 2
print("verified leakage-safe 90-run sweep materialization")
PY
```

Expected: verification message prints and no training, checkpoint, or repository result artifact is created.

- [ ] **Step 4: Inspect source preservation, diff, and status**

```bash
git diff --check
git diff -- config/experiments/top10_with_sentiment_jepa_lam_1_mae_lam_1.json
git status --short
git log --oneline --decorate -8
```

Expected: `git diff --check` is clean; the source config diff contains only the pre-existing `max_parallel_jobs: 2` addition; no datasets, checkpoints, plots, temporary configs, caches, or unrelated files are staged or untracked.

- [ ] **Step 5: Run verification-before-completion review**

Use `superpowers:verification-before-completion`. Recheck the approved spec
against the implementation, confirm the exact selection formula and test-loader
guard in source, and report actual test counts and commands without claiming the
90 training runs were executed.

- [ ] **Step 6: Commit any verification-only documentation correction**

If Step 5 finds a documentation mismatch, edit only `doc/configuration.md`, rerun
the relevant tests, and commit:

```bash
git add doc/configuration.md
git commit -m "docs: clarify lambda sweep workflow"
```

If no documentation correction is needed, do not create an empty commit.
