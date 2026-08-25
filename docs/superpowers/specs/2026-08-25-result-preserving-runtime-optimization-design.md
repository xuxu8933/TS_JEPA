# Result-Preserving Runtime Optimization Design

## Objective

Reduce wall-clock time for the complete TS-JEPA stock experiment without
changing its scientific configuration, stochastic trajectories, checkpoint
state tensors, or downstream comparison metrics.

This change is limited to two optimizations:

1. run independent stock/seed/strategy tasks concurrently;
2. remove per-batch GPU synchronization caused by training-loss `.item()`
   calls.

The benchmark uses
`config/experiments/top10_with_sentiment_jepa_lam_1_mae_lam_1.json` with its
existing one-stock and one-seed execution limits. Both enabled masking
strategies, all 2,001 pretraining epochs, all 501 downstream epochs, and every
other configured parameter remain unchanged.

## Acceptance Criterion

For each masking strategy, the optimized run must match the sequential
baseline as follows:

- every tensor in the epoch-2000 checkpoint model and optimizer state is
  exactly equal under `torch.equal`;
- non-tensor checkpoint state that defines training progress is equal;
- the model-comparison CSV values are exactly equal;
- the number and ordering of optimizer steps, samples, masks, validation
  passes, and checkpoints are unchanged.

Elapsed times, filesystem timestamps, output paths, process IDs, hardware
utilization metadata, and serialized checkpoint bytes may differ. Serialized
bytes are not an equality target because container metadata can differ even
when every stored value is equal.

## Non-Goals

This work will not introduce mixed precision, TF32, nondeterministic kernels,
`torch.compile`, fused attention, a different batch size, DataLoader worker
changes, cached baselines, deferred checkpoint writes, cached patches, fewer
epochs, wider strides, or reduced validation. It will not modify the target
experiment JSON.

## Current Behavior

`run_top_nasdaq100_stocks.py` executes every planned strategy/stock/seed task
sequentially. A task runs pretraining and then downstream evaluation. Tasks
already use separate checkpoint fingerprints and result directories, making
them independent except for sharing the selected CUDA device.

The pretraining, downstream TS-JEPA, and downstream GRU loops call `.item()`
on one or more CUDA loss scalars for every batch. Each call forces the host to
wait for queued GPU work even though the values are only needed after the
epoch for reporting.

## Parallel Task Execution

### Interface

Add a runtime option named `max_parallel_jobs`:

- CLI: `--max-parallel-jobs N`;
- config location: `runner.execution.max_parallel_jobs`;
- default: `1`;
- validation: `N` must be positive.

The option is operational metadata, not an experimental parameter. It must
not change the experiment compatibility signature, checkpoint fingerprint,
or model/evaluation command arguments.

The benchmark will pass `--max-parallel-jobs 2` on the command line, leaving
the user-owned target config unchanged.

### Execution Model

Extract the existing per-task pretrain-and-evaluate sequence into one narrow
function. With `max_parallel_jobs=1`, call it directly in the existing task
order. With a larger value, submit tasks to a `ThreadPoolExecutor`; each worker
only coordinates blocking child processes, while model computation remains
inside the existing Python subprocesses.

Within a task, ordering remains strict:

1. run or reuse pretraining;
2. verify the requested checkpoint exists;
3. write the task's `running` manifest;
4. run downstream evaluation;
5. verify comparison outputs;
6. write the task's `complete` manifest.

Each worker writes only to its task-specific result directory and
fingerprinted checkpoint family. The shared command summary is written by the
main thread in deterministic plan order before concurrent execution begins.
Console status updates may reflect completion order.

### Failure Handling

On the first task failure:

- do not submit or start additional pending work where cancellation is still
  possible;
- cancel futures that have not started;
- allow already-running subprocess chains to terminate cleanly;
- propagate the original failure after active workers settle;
- never write a `complete` manifest for a failed or incomplete task;
- do not generate combined plots after a task failure.

Incremental reruns continue to rely on the existing compatibility and
completion checks.

## Synchronization-Free Loss Reporting

Introduce a small helper that reproduces the existing ordered Python-float
sum without synchronizing once per scalar:

1. append `loss.detach()` for each batch to an epoch-local list;
2. at epoch end, stack the scalar tensors;
3. transfer the stack to CPU once;
4. convert it to a Python list;
5. call Python `sum` in original batch order and divide by the unchanged batch
   count.

`tensor.item()` and `tensor.cpu().tolist()` both convert the same stored
floating-point scalar to a Python float. Summing the resulting Python floats
in the same order preserves existing reported loss values while reducing many
host synchronizations to one transfer per collected loss series per epoch.

Apply the helper only to training loss aggregation in:

- unified pretraining total, JEPA, and MAE losses;
- downstream TS-JEPA total, MSE, and directional losses;
- downstream GRU loss.

Do not change validation, prediction, gradient, optimizer, EMA, scheduler,
masking, or checkpoint code. Detached reporting tensors must never retain an
autograd graph or influence training.

## Tests

Add focused tests for:

- exact equality between the old ordered `.item()` accumulation and the new
  batched scalar collection for representative float32 values;
- detached aggregation with no gradient connection;
- default single-job execution preserving planned task order;
- `max_parallel_jobs=2` allowing two task chains to overlap while never
  exceeding the configured bound;
- rejection of zero or negative worker counts;
- experiment compatibility signatures remaining equal for worker counts 1
  and 2;
- failure propagation and absence of a false `complete` manifest.

Run the existing stock-runner, dual-loss, preprocessing, checkpoint, and
result-analysis tests in addition to the focused tests.

## Benchmark Procedure

Use isolated benchmark workspaces that share the same immutable market and
sentiment input files but have separate checkpoint and result directories.
This prevents checkpoint reuse and output collisions.

Measure three runs:

1. **Baseline:** current sequential implementation.
2. **Optimized serial:** new implementation with `max_parallel_jobs=1`,
   isolating the synchronization change.
3. **Optimized parallel:** new implementation with `max_parallel_jobs=2`,
   measuring combined throughput on the single GPU.

For every run, record total wall time and per-task completion time. Do not run
the comparison while unrelated GPU workloads are active. Compare baseline and
both optimized artifacts using the acceptance criterion above. If exact state
or metric equality fails, treat the optimization as invalid and diagnose the
first differing field before proceeding.

## Research Impact

This is an execution optimization, not a methodological change or new
experiment. It preserves chronological splits, normalization, masks,
objectives, model selection, baselines, seeds, and all reported scientific
metrics. The runtime worker count should be recorded as provenance but must
not create a new scientific experiment identity.
