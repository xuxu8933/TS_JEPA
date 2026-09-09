import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

from run_top_nasdaq100_stocks import (
    build_stock_commands,
    effective_experiment_config,
    parse_args,
    validate_runner_mask_geometry,
)


class Stage04Context12LauncherTest(unittest.TestCase):
    def test_launches_only_local_long_with_isolated_context12_results(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "run_chapter5_stage04_local_long_market_only_context_12.sh"
        # Capture the launch boundary instead of starting 75 training runs.
        with tempfile.TemporaryDirectory() as workdir:
            result = subprocess.run(
                ["bash", str(script), "--dry-run", "--max-parallel-jobs", "1"],
                cwd=workdir,
                env={**os.environ, "PYTHON_BIN": "/bin/echo"},
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [
            shlex.split(line)
            for line in result.stdout.splitlines()
            if line.startswith("-u ")
        ]
        self.assertEqual(len(commands), 5)
        weights = set()
        for command in commands:
            self.assertEqual(command[:2], ["-u", "run_top_nasdaq100_stocks.py"])
            command[3] = str(root / command[3])
            args = parse_args(command[2:])
            original_path = (
                root / "config/experiments/chapter5_candidates/stage04_market_only"
                / Path(args.config).name.replace("_context_12.json", ".json")
            )
            original = parse_args(["--config", str(original_path)])
            expected = effective_experiment_config(original)
            expected["context_size"] = 12
            self.assertEqual(effective_experiment_config(args), expected)
            self.assertNotEqual(args.results_dir, original.results_dir)
            self.assertEqual(Path(args.results_dir).name, Path(args.config).stem)
            self.assertTrue(args.dry_run)
            self.assertEqual(args.max_parallel_jobs, 1)
            self.assertEqual(args.mask_strategies, ["local_long"])
            validate_runner_mask_geometry(args, args.mask_strategies)
            downstream = build_stock_commands(args, "NVDA", seed=42)[-1]
            self.assertEqual(downstream[downstream.index("--context-size") + 1], "12")
            self.assertEqual(
                downstream[downstream.index("--evaluation-split") + 1], "validation"
            )
            weights.add((args.lambda_jepa, args.lambda_mae))
        self.assertEqual(weights, {(0, 2), (0.5, 1.5), (1, 1), (1.5, 0.5), (2, 0)})


if __name__ == "__main__":
    unittest.main()
