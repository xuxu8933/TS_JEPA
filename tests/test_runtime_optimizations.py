import json
import tempfile
import threading
import time
import unittest
from argparse import Namespace
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from config.file_options import flatten_runner_options
from main.utils import ordered_scalar_mean
from run_top_nasdaq100_stocks import (
    effective_experiment_config,
    execute_task,
    execute_tasks,
    experiment_config_signature,
    parse_args as parse_stock_args,
)


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
        with self.assertRaisesRegex(ValueError, "at least one scalar"):
            ordered_scalar_mean([])

    def test_rejects_non_scalar_tensor(self):
        with self.assertRaisesRegex(ValueError, "scalar tensors"):
            ordered_scalar_mean([torch.tensor([1.0, 2.0])])


class ParallelJobOptionTest(unittest.TestCase):
    def test_nested_execution_config_maps_parallel_jobs(self):
        runner = {"execution": {"max_parallel_jobs": 2}}
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "experiment.json"
            flattened = flatten_runner_options(runner, config_path)

        self.assertEqual(flattened["max_parallel_jobs"], 2)

    def test_cli_default_and_override(self):
        self.assertEqual(parse_stock_args([]).max_parallel_jobs, 1)
        self.assertEqual(
            parse_stock_args(["--max-parallel-jobs", "2"]).max_parallel_jobs,
            2,
        )

    def test_rejects_non_positive_parallel_jobs(self):
        for value in ("0", "-1"):
            with self.subTest(value=value), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    parse_stock_args(["--max-parallel-jobs", value])

    def test_worker_count_does_not_change_experiment_identity(self):
        serial = vars(parse_stock_args(["--max-parallel-jobs", "1"]))
        parallel = vars(parse_stock_args(["--max-parallel-jobs", "2"]))

        self.assertEqual(
            effective_experiment_config(serial),
            effective_experiment_config(parallel),
        )
        self.assertEqual(
            experiment_config_signature(serial),
            experiment_config_signature(parallel),
        )
        self.assertNotIn(
            "max_parallel_jobs",
            effective_experiment_config(serial),
        )


class ParallelTaskExecutionTest(unittest.TestCase):
    def _tasks(self, count):
        return [
            {"strategy": "random", "stock": "NVDA", "seed": seed}
            for seed in range(count)
        ]

    def test_single_worker_preserves_task_order(self):
        observed = []
        args = SimpleNamespace(max_parallel_jobs=1)
        with patch(
            "run_top_nasdaq100_stocks.execute_task",
            side_effect=lambda unused_args, task: observed.append(task["seed"]),
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
            "run_top_nasdaq100_stocks.execute_task",
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
                started.append(task["seed"])
            barrier.wait()
            if task["seed"] == 0:
                raise RuntimeError("planned failure")
            time.sleep(0.2)

        with patch(
            "run_top_nasdaq100_stocks.execute_task",
            side_effect=fake_execute,
        ):
            with self.assertRaisesRegex(RuntimeError, "planned failure"):
                execute_tasks(
                    SimpleNamespace(max_parallel_jobs=2),
                    self._tasks(3),
                )

        self.assertCountEqual(started, [0, 1])

    def test_evaluation_failure_leaves_running_manifest(self):
        args = Namespace(dry_run=False, verbose=False)
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            task = {
                "strategy": "random",
                "stock": "NVDA",
                "seed": 42,
                "run_dir": run_dir,
                "checkpoint_path": None,
                "pretrain_command": None,
                "eval_command": ["python", "eval_dual_loss.py"],
            }
            with (
                patch("run_top_nasdaq100_stocks.report_run_status"),
                patch(
                    "run_top_nasdaq100_stocks.run_command",
                    side_effect=RuntimeError("evaluation failed"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                    execute_task(args, task)

            manifest = json.loads(
                (run_dir / "run_manifest.json").read_text()
            )

        self.assertEqual(manifest["status"], "running")


if __name__ == "__main__":
    unittest.main()
