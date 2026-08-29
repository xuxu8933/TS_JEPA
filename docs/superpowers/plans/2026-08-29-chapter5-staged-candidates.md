# Chapter 5 Staged Candidates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a validation-only, three-stage Chapter 5 workflow for 5 stocks × 3 seeds that materializes later candidates from actual preceding winners, compares a 2 × 3 architecture-context grid, and freezes exactly one final test configuration.

**Architecture:** Extend the selector to accept canonical stage prefixes and emit validation-only intermediate snapshots. Add a focused candidate materializer that deep-copies the selected predecessor and applies audited stage-specific deltas. Keep the existing stock runner as the only experiment executor.

**Tech Stack:** Python 3, JSON/JSONC, `argparse`, `hashlib`, `unittest`, existing TS-JEPA runner and selector.

**Spec:** `docs/superpowers/specs/2026-08-29-chapter5-staged-candidates-design.md`

## Global Constraints

- Stocks are exactly `NVDA`, `AAPL`, `AVGO`, `TSLA`, `WMT`; seeds are exactly `42`, `44`, `46`.
- Maximum parallel jobs is `2`.
- Stages 1 and 2 use shared-target JEPA--MAE (`random` masking).
- Stage 3 crosses shared-target and Local-MAE/Long-JEPA with context sizes `6`, `12`, `24` patches.
- JEPA weight is `1.0`; MAE weight is `0.5`; patch size is `5`; pretraining series length is `60`; forecast horizon is `5`.
- Pretraining uses `2001` epochs; downstream training uses `501` epochs.
- Every candidate uses `checkpoint.selection.mode=best` and `downstream.evaluation_split=validation`.
- Intermediate selection never creates a test config or reads a test artifact.
- Only complete three-stage selection writes `selected_config.json` with `evaluation_split=test`.
- Sentiment data is cached before timed runs; candidates do not download data.

---

### Task 1: Canonical stage prefixes and intermediate selected configs

**Files:**
- Modify: `chapter5_selection.py`
- Modify: `tests/test_chapter5_selection.py`
- Modify: `docs/superpowers/specs/2026-08-29-chapter5-validation-selection-design.md`
- Modify: `docs/superpowers/plans/2026-08-29-chapter5-validation-selection.md`

**Interfaces:**
- Consumes: `select_stages(manifest_path: Path)`, `_selected_candidate(summary)`, `freeze_selected_config(...)`, `_write_json_atomic(...)`.
- Produces: `snapshot_selected_stage_config(manifest_path: Path, summary: Mapping[str, Any]) -> dict[str, Any]`; prefix-aware selection; split-specific CLI output.

- [ ] **Step 1: Write failing prefix-selection tests**

Set the test fixture stage names to:

```python
STAGE_NAMES = (
    "preprocessing_normalization",
    "sentiment",
    "architecture_context",
)
```

Add a `stage_count` fixture option. Assert that one- and two-stage CLI runs write `selected_stage_config.json` with literal `evaluation_split == "validation"` and no `selected_config.json`. Assert that a three-stage run writes only `selected_config.json` with literal `evaluation_split == "test"`. Keep explicit rejection tests for a fourth `architecture_objective` stage and the removed `historical_context` name.

- [ ] **Step 2: Run tests and verify RED**

```bash
conda run --no-capture-output -n ts-jepa python -m unittest \
  tests.test_chapter5_selection.DeterministicSelectionTest \
  tests.test_chapter5_selection.FrozenConfigTest
```

Expected: failures because selection still requires one exact stage list and always freezes a test config.

- [ ] **Step 3: Implement prefix validation**

Use:

```python
STAGE_NAMES = (
    "preprocessing_normalization",
    "sentiment",
    "architecture_context",
)

if not actual_stage_names or tuple(actual_stage_names) != STAGE_NAMES[:len(stages)]:
    raise ValueError(
        "Stage order must be a non-empty prefix of "
        f"{list(STAGE_NAMES)}, got {actual_stage_names}"
    )
```

Add `complete = len(stages) == len(STAGE_NAMES)` to the summary.

- [ ] **Step 4: Implement validation-only intermediate snapshots**

Add `snapshot_selected_stage_config`. Resolve and copy the selected source config, assert that its downstream split remains `validation`, and replace top-level provenance with:

```python
{
    "artifact_type": "selected_chapter5_stage_config",
    "schema_version": 1,
    "selection_id": summary["selection_id"],
    "selected_candidate_id": selected["id"],
    "selected_stage": summary["stages"][-1]["name"],
    "source_config": selected["config"],
    "source_config_sha256": selected["config_sha256"],
    "source_config_signature": selected["config_signature"],
    "selection_summary_sha256": canonical_sha256(summary),
    "metric_split": "validation",
}
```

Change `main` to write `selected_stage_config.json` for partial prefixes and `selected_config.json` only when `summary["complete"]` is true.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add chapter5_selection.py tests/test_chapter5_selection.py \
  docs/superpowers/specs/2026-08-29-chapter5-validation-selection-design.md \
  docs/superpowers/plans/2026-08-29-chapter5-validation-selection.md
git commit -m "Support staged Chapter 5 validation selection"
```

---

### Task 2: Five-stock, three-seed stage-1 configs

**Files:**
- Create: `config/experiments/chapter5_candidates/01_preprocessing_window_return.json`
- Create: `config/experiments/chapter5_candidates/01_preprocessing_train_zscore.json`
- Modify: `tests/test_chapter5_selection.py`

**Interfaces:**
- Consumes: `parse_runner_args(["--config", PATH])`, `effective_experiment_config(...)`, `resolve_stocks`, `resolve_seeds`.
- Produces: two runnable stage-1 configs identical except for normalization method.

- [ ] **Step 1: Write failing config contract test**

For both paths, assert literal stocks and seeds, `max_parallel_jobs=2`, only `random` masking, weights `1.0/0.5`, epochs `2001/501`, horizon `5`, context `12`, validation split, best checkpoint, and sentiment disabled. Compare effective config dictionaries after removing only `normalization`; assert the rest are equal. Assert normalization methods are exactly `window_return` and `train_zscore`.

- [ ] **Step 2: Run test and verify RED**

```bash
conda run --no-capture-output -n ts-jepa python -m unittest \
  tests.test_chapter5_selection.StageOneCandidateConfigTest
```

Expected: FAIL because the files do not exist.

- [ ] **Step 3: Create the configs**

Derive both from `normalization_pilot_window_return.json`. Set the Global Constraints, `download.skip=true`, sentiment disabled, `forecast.target=relative_return`, only random masking, best checkpoint, and validation evaluation. Change only `normalization.method` between the files.

- [ ] **Step 4: Run test and verify GREEN**

Run Step 2. Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add config/experiments/chapter5_candidates tests/test_chapter5_selection.py
git commit -m "Add Chapter 5 normalization candidates"
```

---

### Task 3: Deterministic later-stage candidate materializer

**Files:**
- Create: `chapter5_prepare_candidates.py`
- Modify: `tests/test_chapter5_selection.py`

**Interfaces:**
- Consumes: `read_config_file`, `parse_runner_args`, `canonical_sha256`.
- Produces: `materialize_candidates(stage: str, base_config_path: Path, parent_candidate_id: str, output_dir: Path) -> list[Path]`; CLI flags `--stage`, `--base-config`, `--parent-candidate-id`, `--output-dir`.

- [ ] **Step 1: Write failing sentiment materializer tests**

Call:

```python
paths = materialize_candidates(
    "sentiment",
    base,
    "preprocessing_window_return",
    output,
)
```

Assert filenames `02_sentiment_excluded.json` and `02_sentiment_included.json`; byte-identical repeated materialization; preserved coverage, validation, and best-checkpoint settings; exact equality after removing provenance and sentiment `enabled`; parent ID and `canonical_sha256(base_object)` in provenance. Add rejection tests for test-split bases, epoch-selected bases, empty parent IDs, and unsupported stages.

- [ ] **Step 2: Run tests and verify RED**

```bash
conda run --no-capture-output -n ts-jepa python -m unittest \
  tests.test_chapter5_selection.CandidateMaterializerTest
```

Expected: import failure because `chapter5_prepare_candidates.py` does not exist.

- [ ] **Step 3: Implement validation and atomic writes**

Implement `load_validated_base(path)` using the production parser. Reject anything other than validation split and best checkpoints. Write sorted, indented JSON through a `.tmp` sibling and `Path.replace`. If a target exists with different bytes, raise `FileExistsError`; identical reruns are no-ops.

- [ ] **Step 4: Implement sentiment candidates**

Deep-copy the base and change only:

```python
candidate["runner"]["preprocessing"]["custom"]["features"]["sentiment"]["enabled"] = enabled
```

Replace provenance with artifact type, schema version, stage, candidate ID, parent ID, parent canonical SHA-256, and exact delta. Leave downloads disabled for both.

- [ ] **Step 5: Run sentiment tests and verify GREEN**

Run Step 2. Expected: sentiment materialization tests PASS.

- [ ] **Step 6: Write failing architecture-context grid test**

Materialize from the sentiment winner and assert the six filenames from the spec. Assert three resolve only `random`, three only `local_long`, and context sizes are exactly `[6, 12, 24]` for each. Assert fixed objectives, series size, patch size, validation split, and best checkpoint. Assert local-long parameters are MAE window `1`, JEPA gap `4`, JEPA target `4`.

- [ ] **Step 7: Run grid test and verify RED**

Run Step 2. Expected: FAIL because `architecture_context` is unsupported.

- [ ] **Step 8: Implement the six-cell grid**

For each architecture-context pair, deep-copy the base, replace masking strategies with exactly one enabled strategy, and set downstream context. Record architecture, context patches, parent ID, parent hash, and exact deltas in provenance.

- [ ] **Step 9: Run materializer tests and verify GREEN**

Run Step 2. Expected: PASS.

- [ ] **Step 10: Commit Task 3**

```bash
git add chapter5_prepare_candidates.py tests/test_chapter5_selection.py
git commit -m "Add deterministic Chapter 5 candidate materializer"
```

---

### Task 4: Stage manifests and executable command guide

**Files:**
- Create: `config/experiments/chapter5_stage1_selection.template.jsonc`
- Create: `config/experiments/chapter5_stage2_selection.template.jsonc`
- Modify: `config/experiments/chapter5_selection.template.jsonc`
- Create: `doc/chapter5_staged_selection.md`
- Modify: `README.md`
- Modify: `tests/test_chapter5_selection.py`

**Interfaces:**
- Consumes: candidate names from Tasks 2 and 3 and prefix selection from Task 1.
- Produces: parseable templates and copy/paste commands for dry runs, execution, partial selection, materialization, final freezing, and the single test run.

- [ ] **Step 1: Write failing template test**

Load all templates with `read_config_file`. Assert stage sequences are exactly one-, two-, and three-stage canonical prefixes. Assert cumulative candidate counts `[2, 4, 10]` and that the final stage contains the exact six strategy-context candidates.

- [ ] **Step 2: Run test and verify RED**

```bash
conda run --no-capture-output -n ts-jepa python -m unittest \
  tests.test_chapter5_selection.SelectionManifestTemplateTest
```

Expected: FAIL because prefix templates and the six-cell final template are absent.

- [ ] **Step 3: Create the manifests**

Use JSONC comments to mark the only user edits: `validation_root` and later-stage `parent_candidate_id`. Point config paths at `chapter5_candidates/`. Use final IDs `shared_context_6`, `shared_context_12`, `shared_context_24`, `local_long_context_6`, `local_long_context_12`, `local_long_context_24`.

- [ ] **Step 4: Write exact commands**

Document stage commands using this form:

```bash
for cfg in 01_preprocessing_window_return 01_preprocessing_train_zscore; do
  conda run --no-capture-output -n ts-jepa python run_top_nasdaq100_stocks.py \
    --config "config/experiments/chapter5_candidates/${cfg}.json" \
    --dry-run --verbose
done

conda run --no-capture-output -n ts-jepa python chapter5_selection.py \
  --manifest config/experiments/chapter5_stage1_selection.jsonc \
  --output-dir selection_artifacts/chapter5_stage1

STAGE1_PARENT_ID=$(conda run -n ts-jepa python -c \
  "import json; print(json.load(open('selection_artifacts/chapter5_stage1/selection_summary.json'))['selected_candidate_id'])")

conda run --no-capture-output -n ts-jepa python chapter5_prepare_candidates.py \
  --stage sentiment \
  --base-config selection_artifacts/chapter5_stage1/selected_stage_config.json \
  --parent-candidate-id "$STAGE1_PARENT_ID" \
  --output-dir config/experiments/chapter5_candidates
```

Repeat explicitly for the two sentiment candidates, stage-2 selection, six final candidates, complete selection, and one `selected_config.json` test run. State 150 runs, 6--10 hours, and cached sentiment prerequisite.

- [ ] **Step 5: Run template and focused suites**

Run Step 2, then:

```bash
conda run --no-capture-output -n ts-jepa python -m unittest tests.test_chapter5_selection
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add README.md doc/chapter5_staged_selection.md \
  config/experiments/chapter5_selection.template.jsonc \
  config/experiments/chapter5_stage1_selection.template.jsonc \
  config/experiments/chapter5_stage2_selection.template.jsonc \
  tests/test_chapter5_selection.py
git commit -m "Document Chapter 5 staged candidate commands"
```

---

### Task 5: End-to-end verification and handoff

**Files:**
- Modify only if verification exposes a defect in Tasks 1--4.

**Interfaces:**
- Consumes: all prior deliverables.
- Produces: fresh evidence that configs parse, dry runs resolve, selection remains leakage-safe, and the repository stays green.

- [ ] **Step 1: Materialize all generated candidates twice in temporary directories**

Compare SHA-256 sums. Expected: byte-identical outputs.

- [ ] **Step 2: Parse and dry-run all ten candidates**

Use checked-in stage-1 configs and temporary generated later-stage configs. Invoke `run_top_nasdaq100_stocks.py --config PATH --dry-run --verbose` for each. Expected: five stocks, three seeds, best checkpoint, validation split, and only the intended strategy.

- [ ] **Step 3: Run static checks**

```bash
conda run --no-capture-output -n ts-jepa python -m py_compile \
  chapter5_selection.py chapter5_prepare_candidates.py tests/test_chapter5_selection.py
git diff --check
```

Expected: exit 0.

- [ ] **Step 4: Run the complete suite**

```bash
conda run --no-capture-output -n ts-jepa python -m unittest discover -s tests
```

Expected: all tests pass; only environment-dependent tests may skip.

- [ ] **Step 5: Inspect repository state**

```bash
git status --short --branch
git diff --stat
git log --oneline --max-count=8
```

Confirm no datasets, checkpoints, validation/test metrics, temporary configs, caches, or generated experiment outputs are staged.

- [ ] **Step 6: Commit verification fixes only if needed**

If verification exposes a defect, commit its regression test and focused fix. Do not create an empty commit.
