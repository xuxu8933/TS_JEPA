# Leakage-Safe JEPA/MAE Lambda Sweep Design

**Date:** 2026-08-25

## Objective

Add a reproducible, validation-only workflow for choosing one shared JEPA/MAE
objective-weight pair for both supported masking strategies. The selected pair
will then be written into a separate final configuration for the full ten-stock,
ten-seed experiment.

This is a new experiment workflow. It does not change JEPA or MAE loss
definitions, masking semantics, forecasting targets, chronological splits, or
final evaluation metrics.

## Screening experiment

The sweep compares five pairs whose total objective weight remains 2.0:

| Candidate | JEPA weight | MAE weight |
| --- | ---: | ---: |
| `j2_m0` | 2.0 | 0.0 |
| `j1p5_m0p5` | 1.5 | 0.5 |
| `j1_m1` | 1.0 | 1.0 |
| `j0p5_m1p5` | 0.5 | 1.5 |
| `j0_m2` | 0.0 | 2.0 |

Each pair is evaluated with:

- stocks `NVDA`, `AAPL`, and `MSFT`;
- seeds `42`, `43`, and `44`;
- masking strategies `random` and `local_long`;
- all other research settings copied unchanged from
  `config/experiments/top10_with_sentiment_jepa_lam_1_mae_lam_1.json`.

This produces 90 required validation summaries:

```text
5 candidates x 3 stocks x 3 seeds x 2 strategies = 90 runs
```

The execution concurrency remains bounded at two jobs. Concurrency is a runtime
setting and does not change the experimental identity or objective values.

## Architecture and data flow

```text
base experiment config
        |
        v
sweep definition
(candidates, stocks, seeds, strategies, selection policy)
        |
        v
materialized full candidate configs
        |
        v
existing stock runner in validation-only mode
        |
        v
90 structured validation summaries
        |
        v
coverage validation and shared-pair ranking
        |
        v
selection report + generated final 10-stock/10-seed config
```

A dedicated sweep command owns expansion, execution, coverage checking,
selection, and final-config generation. Candidate configurations are fully
materialized under a sweep-specific results directory so every effective
setting can be inspected and reproduced. Candidate and final configs are
generated artifacts; the source experiment config is never edited.

The generated candidate result directories are explicitly recorded in the sweep
manifest rather than inferred later from weight formatting. Existing experiment
identity checks continue to protect individual candidate runs against reuse of
incompatible artifacts.

## Validation-only evaluation mode

An explicit `validation_only` setting is forwarded through:

1. `run_top_nasdaq100_stocks.py`;
2. `eval_dual_loss.py`;
3. `eval_forecast_prequential_with_baselines_gru_volume.py`.

It is a result-affecting execution mode and therefore belongs in effective
experiment identity and manifests. A validation-only run cannot be confused
with, or reused as, a final test-evaluation run.

In validation-only mode, downstream training behaves exactly as before through
selection and restoration of the best validation checkpoint. It then:

1. writes the normal loss history;
2. writes an atomic `validation_summary.json`;
3. exits successfully before GRU training, deterministic-baseline evaluation,
   final model comparison, or combined plotting.

The test loader is not constructed in validation-only mode. This makes the test
period unavailable to the lambda-selection workflow rather than merely
computing test results and promising to ignore them.

The summary contains at least:

- schema version and evaluation mode;
- stock/data identifier and seed;
- masking strategy;
- JEPA and MAE weights and loss names;
- validation selection rule and trend-selection weight;
- best epoch and best selection score;
- validation MSE, MAE, and direction accuracy at the best epoch;
- relevant checkpoint and source configuration provenance.

The runner treats a nonempty, schema-valid validation summary plus a compatible
run manifest as completion in validation-only mode. Final-mode completion keeps
its current model-comparison CSV/TXT requirement.

## Selection policy

Selection is refused unless the exact requested 90-run grid is present. Missing,
duplicate, malformed, non-finite, or identity-incompatible summaries cause a
clear error. Partial results are never ranked.

Each candidate is summarized over all 18 runs, giving equal weight to every
stock/seed/strategy combination. One pair is selected jointly across both
masking strategies so the final comparison does not confound masking strategy
with a different objective balance.

Candidates are ordered by:

1. lowest mean best-validation MSE;
2. lowest sample standard deviation of best-validation MSE;
3. smallest absolute difference between JEPA and MAE weights;
4. declaration order in the sweep definition.

The last two rules only make ties deterministic; they do not replace the
validation criterion.

Selection produces:

- a row-level table of all 90 included summaries;
- a candidate-level ranking table;
- a JSON selection manifest with coverage, rule, winner, source-config hash,
  and candidate-config hashes;
- a full final experiment config with the winning weights, all ten configured
  stocks, all ten configured seeds, both strategies, final evaluation enabled,
  and bounded two-job execution.

The final config is written to the sweep artifact directory with a filename that
contains the winning pair. Running it creates a new result namespace and does
not overwrite the original 1.0/1.0 experiment.

## Resume and failure behavior

The sweep is resumable at individual stock/seed/strategy granularity through the
existing runner manifests. Candidate configs are deterministically regenerated
and compared with their recorded hashes. Existing incompatible data is not
overwritten automatically.

The sweep stops before selection when:

- a candidate command fails;
- expanded configuration differs from recorded provenance;
- an expected validation summary is absent or invalid;
- coverage is not exactly the declared Cartesian product;
- a metric used for ranking is non-finite;
- the source config changes between expansion and selection.

Atomic JSON writes prevent interrupted runs from appearing complete.

## Configuration preservation

Candidate expansion deep-copies the complete source JSON. Only the following
screening fields may change:

- JEPA and MAE objective weights;
- screening stocks and seeds;
- `max_stocks` and `max_seeds` to cover the three-by-three screen;
- validation-only mode;
- combined-plot behavior, because no final comparison exists;
- sweep-specific provenance and result namespace.

The generated final config deep-copies the same source and changes only:

- the selected JEPA and MAE objective weights;
- limits to cover all ten stocks and ten seeds;
- final evaluation mode;
- its unique result namespace/provenance.

The source file, including the user's uncommitted `max_parallel_jobs: 2`
setting, is not modified or staged by this work.

## Verification

Focused automated tests cover:

- parsing and forwarding validation-only mode through all three layers;
- not constructing the test loader in validation-only mode;
- not training GRU or producing final comparison artifacts in that mode;
- correct best-epoch summary contents;
- validation-only versus final experiment-identity separation;
- exact 90-run Cartesian coverage validation;
- rejection of missing, duplicate, malformed, incompatible, and non-finite
  summaries;
- deterministic candidate ranking and all tie-breakers;
- complete candidate expansion with only approved overrides;
- final-config generation without changing the source config;
- resume behavior based on validation summary completion.

After focused tests, the complete repository test suite will run. The final diff
and status will be inspected for generated data, checkpoints, unrelated edits,
or accidental staging of the pre-existing config modification.

## Research impact

This workflow reduces screening cost by omitting test evaluation and invariant
baseline work while preserving the actual pretraining and downstream validation
procedure. Lambda selection never observes the held-out test period. The final
ten-stock, ten-seed run remains the only workflow that computes test metrics for
the selected pair.

No claim of runtime improvement or unchanged numerical results will be made
until the implementation is tested. The lambda screen itself is a methodological
comparison and may intentionally select an objective balance whose results
differ from the current 1.0/1.0 configuration.
