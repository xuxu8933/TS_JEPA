# Result-Preserving Runtime Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run independent TS-JEPA experiment tasks concurrently and eliminate per-batch training-loss GPU synchronizations without changing checkpoint state tensors or downstream metrics.

**Architecture:** The runner gains a runtime-only bounded task executor that keeps each task's pretrain-then-evaluate chain intact. A shared ordered-scalar helper batches detached CUDA scalar transfers at epoch boundaries while preserving Python-float summation order. Baseline, optimized-serial, and optimized-parallel runs use isolated worktrees for exact comparison.

**Tech Stack:** Python 3.11, PyTorch 2.11, `concurrent.futures.ThreadPoolExecutor`, `unittest`, Git worktrees, NVIDIA RTX 3060.

**Spec:** `docs/superpowers/specs/2026-08-25-result-preserving-runtime-optimization-design.md`

## Global Constraints

- Preserve both strategies, 2,001 pretraining epochs, 501 downstream epochs, all samples, validation passes, checkpoints, RNG calls, optimizer/EMA/scheduler steps, chronological splits, and normalization.
- Do not modify `config/experiments/top10_with_sentiment_jepa_lam_1_mae_lam_1.json`.
- Require `torch.equal` for corresponding checkpoint tensors and exact model-comparison CSV values.
- Allow only time, timestamps, paths, process IDs, hardware metadata, and serialized container bytes to differ.
- Exclude `max_parallel_jobs` from experiment identity and model commands while recording it as runner provenance.
- Do not add AMP, TF32, nondeterminism, `torch.compile`, fused kernels, DataLoader changes, caching, or deferred checkpoints.

---

### Task 1: Capture the Sequential Baseline

**Files:**
- Read: `config/experiments/top10_with_sentiment_jepa_lam_1_mae_lam_1.json`
- Produce outside repository: baseline checkpoints, results, logs, and timing

**Interfaces:**
- Consumes: design commit `ce99c83` and target config.
- Produces: `/tmp/ts-jepa-runtime-bench-root.txt` and `baseline_seconds.txt` for Task 7.

- [ ] **Step 1: Create an isolated baseline worktree**

Use `superpowers:using-git-worktrees`, then run:

```bash
BENCH_ROOT=$(mktemp -d /tmp/ts-jepa-runtime-bench.XXXXXX)
git worktree add --detach "$BENCH_ROOT/baseline" ce99c83
cp config/experiments/top10_with_sentiment_jepa_lam_1_mae_lam_1.json \
  "$BENCH_ROOT/baseline/config/experiments/"
ln -s /home/xujiang/TS_JEPA/data "$BENCH_ROOT/baseline/data"
printf '%s\n' "$BENCH_ROOT" > /tmp/ts-jepa-runtime-bench-root.txt
```

Expected: baseline code and outputs are isolated; input data are shared read-only.

- [ ] **Step 2: Require an uncontended GPU**

Run `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader`.

Expected: no unrelated CUDA training process. Monitor existing user work until it exits; never terminate it.

- [ ] **Step 3: Run and time the baseline**

```bash
cd "$BENCH_ROOT/baseline"
/usr/bin/time -f '%e' -o "$BENCH_ROOT/baseline_seconds.txt" \
  /home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 \
  run_top_nasdaq100_stocks.py \
  --config config/experiments/top10_with_sentiment_jepa_lam_1_mae_lam_1.json
```

Expected: complete random and local-long NVDA seed-42 runs.

- [ ] **Step 4: Verify both baseline manifests and checkpoints**

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 - <<'PY'
import json
from pathlib import Path

root = Path(open('/tmp/ts-jepa-runtime-bench-root.txt').read().strip()) / 'baseline'
results = root / 'results' / 'top10_with_sentiment_jepa_lam_1_mae_lam_1'
for strategy in ('random', 'local_long'):
    path = results / strategy / 'NVDA' / 'seed_42' / 'run_manifest.json'
    manifest = json.loads(path.read_text())
    assert manifest['status'] == 'complete'
    assert (root / manifest['checkpoint_path']).is_file()
print('baseline artifacts complete')
PY
```

Expected: `baseline artifacts complete`.

---

### Task 2: Add Exact Ordered Scalar Aggregation

**Files:**
- Modify: `main/utils.py:18`
- Create: `tests/test_runtime_optimizations.py`

**Interfaces:**
- Consumes: non-empty `Sequence[torch.Tensor]` of scalar losses in batch order.
- Produces: `ordered_scalar_mean(values: Sequence[torch.Tensor]) -> float`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_runtime_optimizations.py`:

```python
import unittest

import torch

from main.utils import ordered_scalar_mean


class OrderedScalarMeanTest(unittest.TestCase):
    def test_matches_ordered_item_mean_exactly(self):
        values = [
            torch.tensor(value, dtype=torch.float32, requires_grad=True)
            for value in (0.1, 1000.25, -0.3, 1.0 / 7.0)
        ]
        expected = sum(value.item() for value in values) / len(values)
        actual = ordered_scalar_mean(values)
        self.assertEqual(actual, expected)
        self.assertIsInstance(actual, float)
        self.assertTrue(all(value.grad is None for value in values))

    def test_rejects_empty_values(self):
        with self.assertRaisesRegex(ValueError, 'at least one scalar'):
            ordered_scalar_mean([])

    def test_rejects_non_scalar_tensor(self):
        with self.assertRaisesRegex(ValueError, 'scalar tensors'):
            ordered_scalar_mean([torch.tensor([1.0, 2.0])])


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run and confirm import failure**

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest \
  tests.test_runtime_optimizations.OrderedScalarMeanTest -v
```

Expected: FAIL because the helper is missing.

- [ ] **Step 3: Implement the helper**

Add to `main/utils.py`:

```python
from collections.abc import Sequence


def ordered_scalar_mean(values: Sequence[torch.Tensor]) -> float:
    """Average detached scalars with one device-to-host transfer."""
    if not values:
        raise ValueError('ordered_scalar_mean requires at least one scalar')
    if any(value.numel() != 1 for value in values):
        raise ValueError('ordered_scalar_mean accepts only scalar tensors')
    host_values = torch.stack(
        [value.detach().reshape(()) for value in values]
    ).cpu().tolist()
    return sum(host_values) / len(host_values)
```

- [ ] **Step 4: Run the helper tests**

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest \
  tests.test_runtime_optimizations.OrderedScalarMeanTest -v
```

Expected: three tests PASS.

- [ ] **Step 5: Commit the helper**

```bash
git add main/utils.py tests/test_runtime_optimizations.py
git commit -m "perf: batch training loss scalar transfers"
```

Expected: one focused commit.

---

### Task 3: Remove Per-Batch Training-Loss Synchronization

**Files:**
- Modify: `pretrain_dual_loss.py:38,1769-1884`
- Modify: `eval_forecast_prequential_with_baselines_gru_volume.py:18-24,2253-2436`
- Test: `tests/test_runtime_optimizations.py`

**Interfaces:**
- Consumes: `ordered_scalar_mean` from Task 2.
- Produces: unchanged epoch loss floats with one host transfer per series per epoch.

- [ ] **Step 1: Add a failing source regression test**

Append:

```python
import ast
from pathlib import Path


class TrainingLossSynchronizationTest(unittest.TestCase):
    def test_training_aggregation_does_not_call_item(self):
        repo_root = Path(__file__).resolve().parents[1]
        cases = (
            (repo_root / 'pretrain_dual_loss.py', ('loss', 'jepa_loss', 'mae_loss')),
            (
                repo_root / 'eval_forecast_prequential_with_baselines_gru_volume.py',
                ('loss', 'mse_loss', 'trend_loss'),
            ),
        )
        for path, forbidden_names in cases:
            tree = ast.parse(path.read_text())
            forbidden = []
            for node in ast.walk(tree):
                function = node.func if isinstance(node, ast.Call) else None
                if (
                    isinstance(function, ast.Attribute)
                    and function.attr == 'item'
                    and isinstance(function.value, ast.Name)
                    and function.value.id in forbidden_names
                ):
                    forbidden.append((function.value.id, node.lineno))
            self.assertEqual(forbidden, [], f'{path}: {forbidden}')
```

- [ ] **Step 2: Run and confirm current loops fail**

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest \
  tests.test_runtime_optimizations.TrainingLossSynchronizationTest -v
```

Expected: FAIL with current training-loop lines.

- [ ] **Step 3: Convert pretraining aggregation**

Import `ordered_scalar_mean`. Initialize `epoch_losses`, `epoch_jepa_losses`, and `epoch_mae_losses` each epoch. After the optimizer/EMA step append:

```python
epoch_losses = []
epoch_jepa_losses = []
epoch_mae_losses = []
total_anchor = 0.0
num_batches = 0
```

After the optimizer/EMA step append:

```python
epoch_losses.append(loss.detach())
epoch_jepa_losses.append(jepa_loss.detach())
epoch_mae_losses.append(mae_loss.detach())
```

After the zero-batch guard compute:

```python
total_loss = ordered_scalar_mean(epoch_losses)
total_jepa_loss = ordered_scalar_mean(epoch_jepa_losses)
total_mae_loss = ordered_scalar_mean(epoch_mae_losses)
```

Do not alter validation, diagnostics, masks, gradients, optimizer, EMA, scheduler, or checkpoint code.

- [ ] **Step 4: Convert downstream TS-JEPA and GRU aggregation**

For TS-JEPA, append detached total/MSE/trend scalars and compute:

```python
avg_train_loss = ordered_scalar_mean(epoch_losses)
avg_train_mse_loss = ordered_scalar_mean(epoch_mse_losses)
avg_train_trend_loss = ordered_scalar_mean(epoch_trend_losses)
```

For GRU, append `loss.detach()` to `epoch_gru_losses` and compute:

```python
avg_gru_loss = ordered_scalar_mean(epoch_gru_losses)
```

- [ ] **Step 5: Run scoped tests**

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest \
  tests.test_runtime_optimizations tests.test_unified_dual_loss \
  tests.test_dual_loss_smoke -v
```

Expected: tests PASS.

- [ ] **Step 6: Inspect the reporting-only diff**

```bash
git diff -- pretrain_dual_loss.py \
  eval_forecast_prequential_with_baselines_gru_volume.py main/utils.py
```

Expected: only detached reporting aggregation changed.

- [ ] **Step 7: Commit loop integration**

```bash
git add pretrain_dual_loss.py \
  eval_forecast_prequential_with_baselines_gru_volume.py \
  tests/test_runtime_optimizations.py
git commit -m "perf: defer training loss host synchronization"
```

Expected: one focused integration commit.

---

### Task 4: Add the Runtime-Only Parallelism Interface

**Files:**
- Modify: `config/file_options.py:165-176`
- Modify: `run_top_nasdaq100_stocks.py:26-36,546-749`
- Modify: `config/experiments/template_experiment.jsonc`
- Modify: `doc/configuration.md`
- Test: `tests/test_runtime_optimizations.py`

**Interfaces:**
- Consumes: `runner.execution.max_parallel_jobs` or `--max-parallel-jobs N`.
- Produces: positive `args.max_parallel_jobs`, default `1`, excluded from identity.

- [ ] **Step 1: Write failing option tests**

Append:

```python
import tempfile

from config.file_options import flatten_runner_options
from run_top_nasdaq100_stocks import (
    effective_experiment_config,
    experiment_config_signature,
    parse_args as parse_stock_args,
)


class ParallelJobOptionTest(unittest.TestCase):
    def test_nested_execution_config_maps_parallel_jobs(self):
        runner = {'execution': {'max_parallel_jobs': 2}}
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / 'experiment.json'
            flattened = flatten_runner_options(runner, config_path)
        self.assertEqual(flattened['max_parallel_jobs'], 2)

    def test_cli_default_and_override(self):
        self.assertEqual(parse_stock_args([]).max_parallel_jobs, 1)
        self.assertEqual(
            parse_stock_args(['--max-parallel-jobs', '2']).max_parallel_jobs,
            2,
        )

    def test_rejects_non_positive_parallel_jobs(self):
        for value in ('0', '-1'):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parse_stock_args(['--max-parallel-jobs', value])

    def test_worker_count_does_not_change_experiment_identity(self):
        serial = vars(parse_stock_args(['--max-parallel-jobs', '1']))
        parallel = vars(parse_stock_args(['--max-parallel-jobs', '2']))
        self.assertEqual(
            effective_experiment_config(serial),
            effective_experiment_config(parallel),
        )
        self.assertEqual(
            experiment_config_signature(serial),
            experiment_config_signature(parallel),
        )
        self.assertNotIn('max_parallel_jobs', effective_experiment_config(serial))
```

- [ ] **Step 2: Run and confirm unknown-option failures**

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest \
  tests.test_runtime_optimizations.ParallelJobOptionTest -v
```

Expected: FAIL because the config key and CLI option are unknown.

- [ ] **Step 3: Implement and document the option**

Add `'max_parallel_jobs': 'max_parallel_jobs'` to the nested execution mapping. Add:

```python
parser.add_argument(
    '--max-parallel-jobs',
    type=int,
    default=1,
    help='Maximum independent pretrain/evaluation task chains to run concurrently.',
)
```

Validate after configured parsing:

```python
args = parse_args_with_config(parser, argv, section='runner')
if args.max_parallel_jobs <= 0:
    parser.error('--max-parallel-jobs must be positive')
return args
```

Add `max_parallel_jobs` to `NON_RESULT_CONFIG_KEYS`, add `"max_parallel_jobs": 1` to the template, and document its runtime-only single-GPU behavior. Do not edit the target config or model commands.

- [ ] **Step 4: Run option and config tests**

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest \
  tests.test_runtime_optimizations.ParallelJobOptionTest \
  tests.test_top10_nasdaq_mask_comparison -v
```

Expected: tests PASS.

- [ ] **Step 5: Commit the runtime interface**

```bash
git add config/file_options.py run_top_nasdaq100_stocks.py \
  config/experiments/template_experiment.jsonc doc/configuration.md \
  tests/test_runtime_optimizations.py
git commit -m "feat: configure bounded experiment parallelism"
```

Expected: one focused interface commit.

---

### Task 5: Execute Independent Tasks Concurrently

**Files:**
- Modify: `run_top_nasdaq100_stocks.py:1-18,519-543,1161-1254`
- Test: `tests/test_runtime_optimizations.py`

**Interfaces:**
- Consumes: `args.max_parallel_jobs` and existing task dictionaries.
- Produces: `execute_task(args, task) -> None`, `execute_tasks(args, tasks) -> None`, and `write_task_commands(summary, task) -> None`.

- [ ] **Step 1: Write failing executor tests**

Append:

```python
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from run_top_nasdaq100_stocks import execute_task, execute_tasks


class ParallelTaskExecutionTest(unittest.TestCase):
    def _tasks(self, count):
        return [
            {'strategy': 'random', 'stock': 'NVDA', 'seed': seed}
            for seed in range(count)
        ]

    def test_single_worker_preserves_task_order(self):
        observed = []
        args = SimpleNamespace(max_parallel_jobs=1)
        with patch(
            'run_top_nasdaq100_stocks.execute_task',
            side_effect=lambda unused_args, task: observed.append(task['seed']),
        ):
            execute_tasks(args, self._tasks(4))
        self.assertEqual(observed, [0, 1, 2, 3])

    def test_two_workers_overlap(self):
        barrier = threading.Barrier(2, timeout=5)
        lock = threading.Lock()
        active = 0
        peak = 0

        def fake_execute(unused_args, unused_task):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            barrier.wait()
            with lock:
                active -= 1

        with patch(
            'run_top_nasdaq100_stocks.execute_task',
            side_effect=fake_execute,
        ):
            execute_tasks(SimpleNamespace(max_parallel_jobs=2), self._tasks(2))
        self.assertEqual(peak, 2)

    def test_failure_does_not_start_third_task(self):
        barrier = threading.Barrier(2, timeout=5)
        started = []
        lock = threading.Lock()

        def fake_execute(unused_args, task):
            with lock:
                started.append(task['seed'])
            barrier.wait()
            if task['seed'] == 0:
                raise RuntimeError('planned failure')
            time.sleep(0.2)

        with patch(
            'run_top_nasdaq100_stocks.execute_task',
            side_effect=fake_execute,
        ):
            with self.assertRaisesRegex(RuntimeError, 'planned failure'):
                execute_tasks(SimpleNamespace(max_parallel_jobs=2), self._tasks(3))
        self.assertCountEqual(started, [0, 1])

    def test_evaluation_failure_never_writes_complete_manifest(self):
        args = SimpleNamespace(dry_run=False, verbose=False)
        task = {
            'strategy': 'random',
            'stock': 'NVDA',
            'seed': 42,
            'pretrain_command': None,
            'eval_command': ['python', 'eval_dual_loss.py'],
        }
        with (
            patch('run_top_nasdaq100_stocks.report_run_status'),
            patch(
                'run_top_nasdaq100_stocks.run_command',
                side_effect=RuntimeError('evaluation failed'),
            ),
            patch('run_top_nasdaq100_stocks.write_run_manifest') as write_manifest,
        ):
            with self.assertRaisesRegex(RuntimeError, 'evaluation failed'):
                execute_task(args, task)
        statuses = [call.args[2] for call in write_manifest.call_args_list]
        self.assertEqual(statuses, ['running'])
```

- [ ] **Step 2: Run and confirm missing interface failure**

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest \
  tests.test_runtime_optimizations.ParallelTaskExecutionTest -v
```

Expected: FAIL because `execute_tasks` is undefined.

- [ ] **Step 3: Extract the existing task chain**

Move the current per-task operations into this function. Do not write
`complete` on failure:

```python
def execute_task(args, task):
    if args.dry_run and not args.verbose:
        report_run_status(task, 'dry-run')
    try:
        if task['pretrain_command'] is not None:
            if not args.verbose and not args.dry_run:
                report_run_status(task, 'pretraining')
            run_command(
                task['pretrain_command'],
                dry_run=args.dry_run,
                verbose=args.verbose,
            )
            if (
                not args.dry_run
                and task['checkpoint_path'] is not None
                and not task['checkpoint_path'].is_file()
            ):
                raise RuntimeError(
                    'Pretraining completed without creating the requested '
                    f"checkpoint: {task['checkpoint_path']}"
                )

        if not args.dry_run:
            write_run_manifest(args, task, 'running')
        if not args.verbose and not args.dry_run:
            report_run_status(task, 'evaluating')
        run_command(
            task['eval_command'],
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        if not args.dry_run:
            if not _comparison_outputs_complete(task['run_dir']):
                raise RuntimeError(
                    'Downstream evaluation completed without a '
                    'model-comparison CSV/TXT pair in '
                    f"{task['run_dir']}"
                )
            write_run_manifest(args, task, 'complete')
            if not args.verbose:
                report_run_status(task, 'complete')
    except Exception:
        if not args.verbose and not args.dry_run:
            report_run_status(task, 'failed')
        raise
```

- [ ] **Step 4: Implement bounded scheduling**

Import `FIRST_COMPLETED`, `ThreadPoolExecutor`, and `wait`, then add:

```python
def execute_tasks(args, tasks):
    tasks = list(tasks)
    if args.max_parallel_jobs == 1 or len(tasks) <= 1:
        for task in tasks:
            execute_task(args, task)
        return

    task_iterator = iter(tasks)
    with ThreadPoolExecutor(max_workers=args.max_parallel_jobs) as executor:
        running = set()
        for _ in range(min(args.max_parallel_jobs, len(tasks))):
            running.add(executor.submit(execute_task, args, next(task_iterator)))

        while running:
            completed, running = wait(running, return_when=FIRST_COMPLETED)
            first_error = None
            for future in completed:
                try:
                    future.result()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
            if first_error is not None:
                raise first_error
            for _ in completed:
                try:
                    task = next(task_iterator)
                except StopIteration:
                    break
                running.add(executor.submit(execute_task, args, task))
```

At most `max_parallel_jobs` futures exist. On failure, no later task is submitted; the executor waits for the active peer before propagating the original error.

- [ ] **Step 5: Make summary writes main-thread-only**

Add:

```python
def write_task_commands(summary, task):
    label = f"{task['strategy']}/{task['stock']}[seed={task['seed']}]"
    if task['pretrain_command'] is None:
        summary.write(f'{label}/pretrain: reused or explicitly skipped\n')
    else:
        summary.write(
            f"{label}/pretrain: {' '.join(task['pretrain_command'])}\n"
        )
    summary.write(f"{label}/downstream: {' '.join(task['eval_command'])}\n")
```

In `main` use:

```python
for task in execution_plan['tasks']:
    write_task_commands(summary, task)
summary.flush()
execute_tasks(args, execution_plan['tasks'])
```

Keep combined plots after `execute_tasks`.

- [ ] **Step 6: Run executor and runner tests**

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest \
  tests.test_runtime_optimizations.ParallelTaskExecutionTest \
  tests.test_top10_nasdaq_mask_comparison tests.test_sentiment_toggle -v
```

Expected: tests PASS.

- [ ] **Step 7: Commit concurrent execution**

```bash
git add run_top_nasdaq100_stocks.py tests/test_runtime_optimizations.py
git commit -m "perf: run independent experiment tasks concurrently"
```

Expected: one focused executor commit.

---

### Task 6: Verify the Complete Implementation

**Files:**
- Inspect: all modified source, tests, template, and docs
- Preserve: user-owned target config

**Interfaces:**
- Consumes: Tasks 2-5 commits.
- Produces: clean tested `IMPLEMENTATION_COMMIT` for Task 7.

- [ ] **Step 1: Run focused and full tests**

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest \
  tests.test_runtime_optimizations -v
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest discover \
  -s tests -p 'test_*.py' -v
```

Expected: all tests PASS.

- [ ] **Step 2: Inspect scope and cleanliness**

```bash
git diff --check
git status --short
git diff ce99c83 --stat
git diff ce99c83 -- main/utils.py pretrain_dual_loss.py \
  eval_forecast_prequential_with_baselines_gru_volume.py \
  run_top_nasdaq100_stocks.py config/file_options.py \
  config/experiments/template_experiment.jsonc doc/configuration.md \
  tests/test_runtime_optimizations.py
```

Expected: no generated artifacts or target-config modification; only scoped changes.

- [ ] **Step 3: Record implementation revision**

Run `git rev-parse HEAD` and retain it as `IMPLEMENTATION_COMMIT`. No empty commit is required.

---

### Task 7: Benchmark and Compare Optimized Runs

**Files:**
- Read: baseline artifacts from Task 1
- Produce outside repository: optimized serial/parallel artifacts and timings

**Interfaces:**
- Consumes: `BENCH_ROOT` and `IMPLEMENTATION_COMMIT`.
- Produces: three wall times and exact checkpoint/metric comparison results.

- [ ] **Step 1: Create isolated optimized worktrees**

Use `superpowers:using-git-worktrees`:

```bash
BENCH_ROOT=$(cat /tmp/ts-jepa-runtime-bench-root.txt)
IMPLEMENTATION_COMMIT=$(git rev-parse HEAD)
git worktree add --detach "$BENCH_ROOT/optimized_serial" "$IMPLEMENTATION_COMMIT"
git worktree add --detach "$BENCH_ROOT/optimized_parallel" "$IMPLEMENTATION_COMMIT"
for variant in optimized_serial optimized_parallel; do
  cp config/experiments/top10_with_sentiment_jepa_lam_1_mae_lam_1.json \
    "$BENCH_ROOT/$variant/config/experiments/"
  ln -s /home/xujiang/TS_JEPA/data "$BENCH_ROOT/$variant/data"
done
```

Expected: independent logs/results with identical committed code and shared data.

- [ ] **Step 2: Reconfirm idle GPU, then run serial and parallel**

Run the same `nvidia-smi` check from Task 1. Then:

```bash
cd "$BENCH_ROOT/optimized_serial"
/usr/bin/time -f '%e' -o "$BENCH_ROOT/optimized_serial_seconds.txt" \
  /home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 \
  run_top_nasdaq100_stocks.py \
  --config config/experiments/top10_with_sentiment_jepa_lam_1_mae_lam_1.json \
  --max-parallel-jobs 1

cd "$BENCH_ROOT/optimized_parallel"
/usr/bin/time -f '%e' -o "$BENCH_ROOT/optimized_parallel_seconds.txt" \
  /home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 \
  run_top_nasdaq100_stocks.py \
  --config config/experiments/top10_with_sentiment_jepa_lam_1_mae_lam_1.json \
  --max-parallel-jobs 2
```

Expected: two complete tasks per variant with no reuse.

- [ ] **Step 3: Compare checkpoint values and metrics recursively**

Run this from the main repository:

```bash
BENCH_ROOT=$(cat /tmp/ts-jepa-runtime-bench-root.txt)
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 - \
  "$BENCH_ROOT/baseline" "$BENCH_ROOT/optimized_serial" \
  "$BENCH_ROOT/optimized_parallel" <<'PY'
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


def assert_equal(left, right, path):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right), path
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray) and np.array_equal(left, right), path
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), path
        for key in left:
            assert_equal(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, pair in enumerate(zip(left, right)):
            assert_equal(pair[0], pair[1], f'{path}[{index}]')
    else:
        assert left == right, f'{path}: {left!r} != {right!r}'


def artifacts(worktree, strategy):
    result_dir = (
        worktree / 'results' / 'top10_with_sentiment_jepa_lam_1_mae_lam_1'
        / strategy / 'NVDA' / 'seed_42'
    )
    manifest = json.loads((result_dir / 'run_manifest.json').read_text())
    assert manifest['status'] == 'complete'
    checkpoint = torch.load(
        worktree / manifest['checkpoint_path'],
        map_location='cpu',
        weights_only=False,
    )
    name = next(
        value for value in manifest['comparison_files'] if value.endswith('.csv')
    )
    with (result_dir / name).open(newline='') as comparison_file:
        metrics = list(csv.DictReader(comparison_file))
    return checkpoint, metrics


baseline, serial, parallel = map(Path, sys.argv[1:])
for strategy in ('random', 'local_long'):
    expected_checkpoint, expected_metrics = artifacts(baseline, strategy)
    for label, worktree in (('serial', serial), ('parallel', parallel)):
        checkpoint, metrics = artifacts(worktree, strategy)
        assert_equal(expected_checkpoint, checkpoint, f'{label}.{strategy}')
        assert expected_metrics == metrics, f'{label}.{strategy}.metrics'
        print(f'exact match: {label} {strategy}')
PY
```

Expected: four `exact match` lines. Any mismatch blocks completion and is diagnosed from the first recursive path.

- [ ] **Step 4: Calculate timings**

```bash
BENCH_ROOT=$(cat /tmp/ts-jepa-runtime-bench-root.txt)
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 - \
  "$BENCH_ROOT/baseline_seconds.txt" \
  "$BENCH_ROOT/optimized_serial_seconds.txt" \
  "$BENCH_ROOT/optimized_parallel_seconds.txt" <<'PY'
import sys
from pathlib import Path

baseline, serial, parallel = [
    float(Path(path).read_text().strip()) for path in sys.argv[1:]
]
print(f'baseline_seconds={baseline:.2f}')
print(f'optimized_serial_seconds={serial:.2f}')
print(f'optimized_parallel_seconds={parallel:.2f}')
print(f'serial_speedup={baseline / serial:.3f}x')
print(f'parallel_speedup={baseline / parallel:.3f}x')
print(f'serial_reduction_percent={(1 - serial / baseline) * 100:.2f}')
print(f'parallel_reduction_percent={(1 - parallel / baseline) * 100:.2f}')
PY
```

Expected: all times, speedups, and reductions.

---

### Task 8: Final Verification and Handoff

**Files:**
- Inspect: final commits, diff, tests, and benchmark outputs
- Preserve: target config and benchmark artifacts until reporting

**Interfaces:**
- Consumes: passing tests and exact comparisons.
- Produces: evidence-backed completion report.

- [ ] **Step 1: Use verification-before-completion and rerun critical checks**

Use `superpowers:verification-before-completion`, then:

```bash
/home/xujiang/miniconda3/envs/ts-jepa/bin/python3.11 -m unittest \
  tests.test_runtime_optimizations tests.test_top10_nasdaq_mask_comparison \
  tests.test_unified_dual_loss tests.test_dual_loss_smoke \
  tests.test_sentiment_toggle -v
git diff --check
git status --short
```

Expected: tests PASS; diff check clean; no generated repository artifacts or target-config changes.

- [ ] **Step 2: Inspect final commits and diff**

```bash
git log --oneline ce99c83..HEAD
git diff ce99c83..HEAD --stat
git diff ce99c83..HEAD
```

Expected: only the two optimizations, focused tests, template, and docs.

- [ ] **Step 3: Report exact evidence**

Report baseline, optimized-serial, and optimized-parallel times; both speedups; exact checkpoint/metric comparison outcome; test commands; unchanged scientific config; and that the speedup applies to this RTX 3060 workload. Report the benchmark artifact path and ask whether it should be retained.
