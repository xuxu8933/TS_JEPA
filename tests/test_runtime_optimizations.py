import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import torch

from config.file_options import flatten_runner_options
from main.utils import ordered_scalar_mean
from run_top_nasdaq100_stocks import (
    effective_experiment_config,
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


if __name__ == "__main__":
    unittest.main()
