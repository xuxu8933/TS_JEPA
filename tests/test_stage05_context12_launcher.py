import json
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

from chapter5_selection import canonical_sha256
from run_top_nasdaq100_stocks import (
    build_stock_commands,
    effective_experiment_config,
    parse_args,
    resolve_seeds,
    resolve_stocks,
    validate_runner_mask_geometry,
)


class Stage05Context12LauncherTest(unittest.TestCase):
    def test_benchmarks_the_context12_validation_winner_in_a_separate_directory(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "run_chapter5_stage05_local_long_market_only_context_12.sh"
        with tempfile.TemporaryDirectory() as workdir:
            result = subprocess.run(
                ["bash", str(script), "--dry-run", "--max-parallel-jobs", "1"],
                cwd=workdir,
                env={**os.environ, "PYTHON_BIN": "/bin/echo"},
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout)
        self.assertEqual(command[:3], ["-u", "run_top_nasdaq100_stocks.py", "--config"])
        command[3] = str(root / command[3])
        args = parse_args(command[2:])
        parent_path = root / (
            "config/experiments/chapter5_candidates/stage04_market_only_context_12/"
            "04_local_long_joint_loss_jepa_1_5_mae_0_5_market_only_context_12.json"
        )
        parent = parse_args(["--config", str(parent_path)])
        expected = effective_experiment_config(parent)
        # The runner omits its default test split from the effective signature.
        expected.pop("evaluation_split")
        self.assertEqual(effective_experiment_config(args), expected)
        self.assertNotEqual(args.results_dir, parent.results_dir)
        self.assertEqual(Path(args.results_dir).name, Path(args.config).stem)
        self.assertEqual(args.evaluation_split, "test")
        self.assertEqual(args.context_size, 12)
        self.assertEqual((args.lambda_jepa, args.lambda_mae), (1.5, 0.5))
        self.assertFalse(args.use_sentiment)
        self.assertTrue(args.dry_run)
        self.assertEqual(args.max_parallel_jobs, 1)
        self.assertEqual(len(resolve_stocks(args)), 5)
        self.assertEqual(resolve_seeds(args), [42, 44, 46])
        self.assertEqual(args.mask_strategies, ["local_long"])
        validate_runner_mask_geometry(args, args.mask_strategies)
        downstream = build_stock_commands(args, "NVDA", seed=42)[-1]
        self.assertEqual(downstream[downstream.index("--context-size") + 1], "12")
        self.assertEqual(downstream[downstream.index("--evaluation-split") + 1], "test")
        provenance = json.loads(Path(args.config).read_text())["provenance"]
        self.assertEqual(
            provenance["parent_config_sha256"],
            canonical_sha256(json.loads(parent_path.read_text())),
        )
        self.assertEqual(provenance["selection_source_split"], "validation")
        self.assertEqual(provenance["evaluation_status"], "retrospective_test_period")


if __name__ == "__main__":
    unittest.main()
