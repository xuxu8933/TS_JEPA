# Chapter 5 Validation Selection Design

## Purpose

Add experiment-level selection without changing the repository's existing chronological split, downstream validation checkpointing, GRU validation-MSE checkpointing, pre-training validation, or `*_best.pt` resolution.

## Experimental contract

Selection proceeds in this fixed order:

1. `preprocessing_normalization`
2. `sentiment`
3. `architecture_context`

Every candidate is evaluated on the existing chronological validation split. The final winner is frozen into a new runnable `selected_config.json`, whose downstream evaluation split is changed to `test`. Running that frozen config therefore writes to a result root distinct from every validation candidate config.

All candidate configs must use `runner.checkpoint.selection.mode = "best"`. A candidate may not substitute pre-training loss for downstream forecasting performance.

## Validation artifact contract

Each stock/seed/strategy validation run writes exactly one `validation_metrics.json` with:

- `artifact_type = "downstream_forecast_metrics"`;
- `schema_version = 1`;
- `split = "validation"`;
- exact candidate identity: config signature, stock, seed, and strategy;
- TS-JEPA downstream `mse`, `mae`, and `direction_accuracy`.

Test runs write `test_metrics.json` with `split = "test"`. The selector accepts only the validation filename and validation split. It rejects recursively any metric artifact containing a key whose token is `test`, so a renamed test artifact cannot be consumed accidentally.

Validation-only execution must not instantiate a test dataset. It trains downstream heads on train, selects their checkpoints on validation using existing logic, and evaluates the restored checkpoints on validation.

## Candidate manifest

A selection manifest contains the selection ID and three ordered stages. Each candidate declares:

- stable candidate ID;
- runnable experiment config path;
- validation result root;
- one enabled masking strategy;
- `parent_candidate_id` after the first stage.

At stage 1 all candidates are eligible. At every later stage, only candidates whose parent is the previous stage winner are eligible. Eligible candidates must have identical configured stock and seed coverage.

## Aggregation and deterministic ranking

For each candidate and metric:

1. average all configured seeds within each stock;
2. average those stock means across configured stocks.

All configured stock/seed artifacts are required. Ranking is lexicographic by validation MSE ascending, validation MAE ascending, direction accuracy descending, then candidate ID ascending. Candidate IDs make exact ties deterministic. Candidates and stock summaries are serialized in sorted order and no wall-clock timestamp is recorded.

## Frozen configuration and provenance

`selected_config.json` remains directly consumable by the existing runner. It contains the original `common`, `runner`, and `analysis` sections plus a read-only top-level `provenance` section. The config loader accepts but never maps provenance into runtime options.

Provenance records the selection ID, selected candidate, source config path and full canonical SHA-256, frozen experiment-config SHA-256, selection-summary SHA-256, Git commit, ordered stage winners, aggregation hierarchy, and ranking rule.

The selector also writes `selection_summary.json`, containing every eligible candidate's per-stock and overall validation metrics plus every selected stage winner. Selection artifacts contain no final test metrics.

## Separation of artifacts

Validation candidates retain their own config-derived result roots. Selection outputs go to an explicitly supplied selection-artifact directory. The frozen config derives a new `results/selected_config` root when run, so final test outputs cannot overwrite or masquerade as candidate validation outputs.
