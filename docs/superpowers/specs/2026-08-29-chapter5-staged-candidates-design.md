# Chapter 5 Staged Candidate Execution Design

## Purpose

Create and run the seven three-stage Chapter 5 candidates without assuming an unknown earlier-stage winner. The workflow must preserve validation-only selection, deterministic provenance, and the existing chronological train/validation/test protocol.

## Experimental coverage

Every candidate uses the same pilot coverage:

- stocks: `NVDA`, `AAPL`, `AVGO`, `TSLA`, `WMT`;
- seeds: `42`, `44`, `46`;
- maximum parallel jobs: `2`;
- masking/model: shared-target JEPA--MAE (`random` strategy);
- objective weights: JEPA `1.0`, MAE `0.5`;
- patch size: `5`;
- forecast horizon: `5`;
- pretraining epochs: `2001`;
- downstream epochs: `501`;
- checkpoint selection: `best`;
- downstream evaluation split: `validation`.

The fixed forecast target and all remaining training settings come from a single checked-in base candidate. Later candidates may change only their declared experimental factor.

## Candidate sequence

### Stage 1: preprocessing and normalization

Two checked-in runnable configs are created immediately:

- `01_preprocessing_window_return.json`;
- `01_preprocessing_train_zscore.json`.

They differ only in `runner.preprocessing.custom.normalization.method`. Sentiment is disabled and downstream context is fixed at 12 patches for both candidates.

### Stage 2: sentiment

After stage 1 selection, a materialization command reads the winning stage-1 config and creates:

- `02_sentiment_excluded.json`;
- `02_sentiment_included.json`.

The files are deep copies of the validated stage-1 winner. They differ only in sentiment enablement and the download/news setting needed to make the feature available. Both record the parent candidate ID and parent config SHA-256 in the ignored top-level provenance section.

### Stage 3: historical context

After stage 2 selection, the materialization command reads the winning sentiment config and creates:

- `03_context_6_patches.json`;
- `03_context_12_patches.json`;
- `03_context_24_patches.json`.

They differ only in `runner.downstream.context_size`, corresponding to 30, 60, and 120 historical observations at patch size 5. Each records the stage-2 parent identity and hash.

## Partial-stage selection

`chapter5_selection.py` accepts a non-empty prefix of the canonical three stages:

1. `preprocessing_normalization`;
2. `sentiment`;
3. `historical_context`.

For a one- or two-stage manifest, it writes `selection_summary.json` and `selected_stage_config.json`. The selected stage config remains validation-only and is intended solely as the immutable base for materializing the next stage. It must not contain or produce test metrics.

Only a complete three-stage manifest may write `selected_config.json`, and only that final frozen config changes `runner.downstream.evaluation_split` to `test`.

Four-stage manifests remain invalid.

## Candidate materializer

A new `chapter5_prepare_candidates.py` command has two modes:

- `--stage sentiment --base-config PATH --parent-candidate-id ID`;
- `--stage historical_context --base-config PATH --parent-candidate-id ID`.

It validates that the base config is validation-only and uses best-checkpoint selection. It writes candidates atomically, refuses to overwrite non-identical files unless explicitly requested, and records canonical parent/config hashes. The generated files remain directly runnable by `run_top_nasdaq100_stocks.py`.

## Manifests and commands

Checked-in manifest templates cover each execution checkpoint:

- stage 1 only;
- stages 1--2;
- complete stages 1--3.

Documentation provides, for each stage:

1. dry-run commands for every candidate;
2. execution commands;
3. partial selection command;
4. next-stage materialization command;
5. final freeze and held-out test commands.

The user updates only validation result-root paths after runs. Parent IDs in later manifests must match the selected IDs from the prior summary.

## Validation and failure behavior

Tests cover:

- five-stock/three-seed coverage in both stage-1 configs;
- candidate configs using validation and best checkpoints;
- exact one-factor differences within each stage;
- deterministic materialization and parent hashes;
- rejection of invalid, test, or non-best base configs;
- prefix-stage selection;
- absence of `selected_config.json` before stage 3;
- final test freezing only after all three stages;
- continued rejection of the removed architecture/objective stage.

## Runtime estimate

The workflow executes 7 candidates × 5 stocks × 3 seeds = 105 stock-seed runs. With two parallel jobs on the detected RTX 3060, 2,001 pretraining epochs, and 501 downstream epochs, the expected wall time is approximately 4--7 hours. Context size 24 is expected to be the slowest candidate. The estimate excludes downloading news; sentiment data should be prepared or cached before timed execution.
