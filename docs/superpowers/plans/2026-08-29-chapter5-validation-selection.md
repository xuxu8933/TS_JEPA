# Chapter 5 Validation Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, validation-only Chapter 5 configuration selector that freezes a provenance-bearing runnable config for one final held-out test evaluation.

**Architecture:** Extend the existing downstream evaluator with an explicit validation/test evaluation split and a strict metrics artifact. Add a standalone staged selector that validates candidate identities, aggregates seeds within stock then stocks, and emits a deterministic summary plus frozen config. Keep selection and final-test roots distinct through config-derived output paths.

**Tech Stack:** Python 3.11, argparse, JSON/JSONC, hashlib, pathlib, unittest, PyTorch evaluation pipeline.

**Spec:** `docs/superpowers/specs/2026-08-29-chapter5-validation-selection-design.md`

## Global Constraints

- Reuse the existing chronological train/validation/test split and all current checkpoint-selection mechanisms.
- Configuration selection may consume validation forecasting metrics only; it must never consume test metrics.
- Aggregate configured seeds within each stock before averaging stocks.
- Rank by MSE, then MAE, then descending direction accuracy, then candidate ID.
- Require `checkpoint.selection.mode = "best"` for every selectable candidate.
- Do not instantiate a test dataset during validation-only execution.
- Keep validation selection artifacts separate from final test artifacts.

---

### Task 1: Config and runner evaluation controls

**Files:**
- Modify: `config/file_options.py`
- Modify: `run_top_nasdaq100_stocks.py`
- Test: `tests/test_chapter5_selection.py`

**Interfaces:**
- Consumes: existing nested `[runner].downstream` parsing and `build_stock_commands`.
- Produces: runner options `evaluation_split: Literal["validation", "test"]` and `context_size: int | None`; accepts ignored top-level `provenance`.

- [ ] **Step 1: Write failing parser and command tests**

```python
def test_nested_downstream_forwards_context_and_validation_split(self):
    args = parse_runner_args(["--config", str(self.config_path)])
    _, command = build_stock_commands(args, "NVDA", seed=42, strategy="random")
    self.assertEqual(command[command.index("--evaluation-split") + 1], "validation")
    self.assertEqual(command[command.index("--context-size") + 1], "8")

def test_frozen_provenance_is_ignored_by_runner(self):
    args = parse_runner_args(["--config", str(self.provenance_config)])
    self.assertEqual(args.evaluation_split, "test")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest -v tests.test_chapter5_selection.RunnerSelectionConfigTest`
Expected: FAIL because the options and provenance section are unsupported.

- [ ] **Step 3: Implement minimal config/runner support**

Add `context_size` and `evaluation_split` to the downstream nested mapping, parser arguments with strict choices/positivity checks, and command forwarding. Permit `provenance` as an ignored top-level section.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m unittest -v tests.test_chapter5_selection.RunnerSelectionConfigTest`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config/file_options.py run_top_nasdaq100_stocks.py tests/test_chapter5_selection.py
git commit -m "feat: configure validation-only downstream runs"
```

### Task 2: Split-safe downstream evaluation artifact

**Files:**
- Modify: `main/utils.py`
- Modify: `eval_dual_loss.py`
- Modify: `eval_forecast_prequential_with_baselines_gru_volume.py`
- Test: `tests/test_chapter5_selection.py`

**Interfaces:**
- Consumes: `evaluation_split`, config signature, mask strategy, stock, and seed from the runner command.
- Produces: `choose_evaluation_loader(evaluation_split, val_loader, test_loader_factory)` and `write_downstream_metrics_artifact(...) -> Path`.

- [ ] **Step 1: Write failing leakage and artifact tests**

```python
def test_validation_loader_never_calls_test_factory(self):
    marker = object()
    chosen = choose_evaluation_loader("validation", marker, lambda: self.fail("test"))
    self.assertIs(chosen, marker)

def test_metrics_artifacts_have_explicit_disjoint_splits(self):
    validation = build_downstream_metrics_artifact(split="validation", **self.identity)
    test = build_downstream_metrics_artifact(split="test", **self.identity)
    self.assertEqual(validation["split"], "validation")
    self.assertEqual(test["split"], "test")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest -v tests.test_chapter5_selection.DownstreamSplitTest`
Expected: FAIL because the helpers and CLI option do not exist.

- [ ] **Step 3: Implement validation-only loader selection and canonical JSON**

Use the existing validation loader for validation runs. Construct the test loader lazily only for test runs. Evaluate restored best TS-JEPA and GRU states on the chosen split, retain current baselines, and write `validation_metrics.json` or `test_metrics.json` atomically.

- [ ] **Step 4: Run tests and existing evaluator tests**

Run: `python -m unittest -v tests.test_chapter5_selection.DownstreamSplitTest tests.test_unified_dual_loss`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main/utils.py eval_dual_loss.py eval_forecast_prequential_with_baselines_gru_volume.py tests/test_chapter5_selection.py
git commit -m "feat: emit split-safe downstream metrics"
```

### Task 3: Validation artifact reader and leakage guard

**Files:**
- Create: `chapter5_selection.py`
- Test: `tests/test_chapter5_selection.py`

**Interfaces:**
- Produces: `load_validation_artifact(path: Path, expected_identity: Mapping[str, Any]) -> dict[str, float]`.

- [ ] **Step 1: Write failing guard tests**

```python
def test_selector_rejects_test_split_even_when_renamed(self):
    self.write_artifact("validation_metrics.json", split="test")
    with self.assertRaisesRegex(ValueError, "validation-only"):
        load_validation_artifact(self.path, self.identity)

def test_selector_rejects_nested_test_metric_keys(self):
    self.write_artifact("validation_metrics.json", extra={"provenance": {"test_mse": 0.0}})
    with self.assertRaisesRegex(ValueError, "test-result"):
        load_validation_artifact(self.path, self.identity)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest -v tests.test_chapter5_selection.ValidationArtifactGuardTest`
Expected: FAIL because the selector module does not exist.

- [ ] **Step 3: Implement fail-closed parsing**

Require the exact filename, artifact type, schema, validation split, finite three-metric object, and exact config-signature/stock/seed/strategy identity. Recursively reject `test` tokens in keys.

- [ ] **Step 4: Run guard tests and verify GREEN**

Run: `python -m unittest -v tests.test_chapter5_selection.ValidationArtifactGuardTest`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add chapter5_selection.py tests/test_chapter5_selection.py
git commit -m "feat: reject test leakage in candidate selection"
```

### Task 4: Hierarchical deterministic stage selection

**Files:**
- Modify: `chapter5_selection.py`
- Test: `tests/test_chapter5_selection.py`

**Interfaces:**
- Produces: `aggregate_candidate(...) -> dict`, `select_stages(manifest_path: Path) -> dict`, and deterministic ranking key `(mse, mae, -direction_accuracy, candidate_id)`.

- [ ] **Step 1: Write failing hierarchy and determinism tests**

```python
def test_aggregation_means_seeds_within_stock_then_stocks(self):
    summary = aggregate_candidate(self.unbalanced_seed_fixture)
    self.assertEqual(summary["overall"]["mse"], 25.25)

def test_selection_is_identical_when_manifest_and_files_are_reordered(self):
    first = select_stages(self.manifest_a)
    second = select_stages(self.manifest_b)
    self.assertEqual(first, second)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest -v tests.test_chapter5_selection.DeterministicSelectionTest`
Expected: FAIL because aggregation and staged selection are absent.

- [ ] **Step 3: Implement exact stage validation and ranking**

Validate the three canonical stage names/order, parent filtering, unique IDs, identical eligible coverage, enabled strategy, validation evaluation split, and best checkpoint mode. Require complete stock/seed coverage and serialize candidates sorted by ID.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m unittest -v tests.test_chapter5_selection.DeterministicSelectionTest`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add chapter5_selection.py tests/test_chapter5_selection.py
git commit -m "feat: select chapter 5 stages deterministically"
```

### Task 5: Frozen config and auditable CLI

**Files:**
- Modify: `chapter5_selection.py`
- Create: `config/experiments/chapter5_selection.template.jsonc`
- Test: `tests/test_chapter5_selection.py`

**Interfaces:**
- Produces CLI `python chapter5_selection.py --manifest PATH --output-dir DIR`; writes `selection_summary.json` and `selected_config.json`.

- [ ] **Step 1: Write failing output tests**

```python
def test_cli_writes_deterministic_summary_and_runnable_frozen_config(self):
    main(["--manifest", str(self.manifest), "--output-dir", str(self.output)])
    frozen = json.loads((self.output / "selected_config.json").read_text())
    self.assertEqual(frozen["runner"]["downstream"]["evaluation_split"], "test")
    self.assertEqual(frozen["runner"]["checkpoint"]["selection"]["mode"], "best")
    parse_runner_args(["--config", str(self.output / "selected_config.json")])
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest -v tests.test_chapter5_selection.FrozenConfigTest`
Expected: FAIL because output writing and provenance are absent.

- [ ] **Step 3: Implement atomic deterministic outputs**

Hash canonical config and summary JSON with SHA-256, record the current Git commit, inject ignored provenance, change only downstream evaluation split to test, and refuse output directories nested in any candidate validation root.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m unittest -v tests.test_chapter5_selection.FrozenConfigTest`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add chapter5_selection.py config/experiments/chapter5_selection.template.jsonc tests/test_chapter5_selection.py
git commit -m "feat: freeze selected chapter 5 experiment"
```

### Task 6: Documentation and complete verification

**Files:**
- Modify: `README.md`
- Modify: `doc/configuration.md`
- Test: `tests/test_chapter5_selection.py`

**Interfaces:**
- Documents validation candidate execution, selection, inspection, and final frozen test commands.

- [ ] **Step 1: Add a CLI workflow smoke test**

Use temporary candidate configs and artifacts to run the real selector CLI twice, compare byte-identical outputs, parse the frozen config with the production parser, and assert no test metric appears in `selection_summary.json`.

- [ ] **Step 2: Run smoke test and verify RED if any integration is missing**

Run: `python -m unittest -v tests.test_chapter5_selection.Chapter5WorkflowIntegrationTest`
Expected: PASS only when the full workflow is connected.

- [ ] **Step 3: Document exact commands and artifact boundaries**

Document `evaluation_split = "validation"`, `checkpoint.selection.mode = "best"`, the selection manifest, selection command, and running the emitted `selected_config.json` once for final test evaluation.

- [ ] **Step 4: Run focused and complete verification**

Run: `python -m unittest -v tests.test_chapter5_selection`

Run: `python -m unittest discover -s tests -v`

Run: `git diff --check && git status --short`

Expected: all tests pass; only intentional source, config, documentation, and compact provenance fixtures are changed.

- [ ] **Step 5: Commit**

```bash
git add README.md doc/configuration.md tests/test_chapter5_selection.py
git commit -m "docs: explain validation-only experiment selection"
```
