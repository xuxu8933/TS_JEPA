import inspect
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np
import pandas as pd
import config.experiment as experiment_config
from eval_dual_loss import build_eval_argv, parse_args as parse_eval_args
from config.config_pretrain import config as pretrain_config
from pretrain_dual_loss import parse_args as parse_pretrain_args
from src.data_loaders.data_class_roll_volume import EvaluationDataLoader
from run_top_nasdaq100_stocks import (
    build_stock_commands,
    effective_experiment_config,
    parse_args as parse_runner_args,
)


def write_price_csv(path: Path, rows: int = 180) -> None:
    close = np.linspace(100.0, 200.0, rows)
    pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=rows, freq="D"),
            "Close": close,
            "Volume": np.linspace(1_000.0, 2_000.0, rows),
        }
    ).to_csv(path, index=False)


class ForecastHorizonTest(unittest.TestCase):
    def test_omitted_horizon_defaults_to_patch_size(self):
        resolver = getattr(experiment_config, "resolve_forecast_horizon", None)
        self.assertIsNotNone(
            resolver,
            "resolve_forecast_horizon must define the independent horizon contract",
        )
        self.assertEqual(resolver(None, 5), 5)

    def test_horizon_must_be_positive(self):
        with self.assertRaisesRegex(
            ValueError,
            "forecast_horizon must be positive",
        ):
            experiment_config.resolve_forecast_horizon(0, 5)

    def test_evaluation_loader_accepts_independent_horizon(self):
        parameters = inspect.signature(EvaluationDataLoader.__init__).parameters
        self.assertIn("forecast_horizon", parameters)

    def test_h1_target_does_not_change_context_patch_dimension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prices.csv"
            write_price_csv(path)
            dataset = EvaluationDataLoader(
                path_data=str(path),
                patch_size=5,
                forecast_horizon=1,
                context_size=12,
                stride=5,
                split="train",
                normalization="window_return",
                feature_cols=("Close", "Volume", "MA10", "MA50"),
                validation_fraction=0.1,
                test_start_date="2020-05-25",
            )
            context, target = dataset[0]
            self.assertEqual(tuple(context.shape), (12, 20))
            self.assertEqual(tuple(target.shape), (1,))

    def test_stock_runner_accepts_forecast_horizon(self):
        parsed = None
        with redirect_stderr(io.StringIO()):
            try:
                parsed = parse_runner_args(["--forecast-horizon", "1"])
            except SystemExit:
                pass
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.forecast_horizon, 1)

    def test_forecast_horizon_is_forwarded_only_to_downstream(self):
        args = parse_runner_args(
            [
                "--forecast-horizon",
                "1",
                "--stocks",
                "NVDA",
                "--max-stocks",
                "1",
                "--results-dir",
                "/tmp/ts-jepa-horizon-test",
            ]
        )
        pretrain_command, eval_command = build_stock_commands(
            args,
            "NVDA",
            seed=42,
            strategy="random",
        )
        self.assertNotIn("--forecast-horizon", pretrain_command)
        horizon_index = eval_command.index("--forecast-horizon")
        self.assertEqual(eval_command[horizon_index + 1], "1")

    def test_unified_evaluator_forwards_forecast_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "not-created.pt"
            args, passthrough = parse_eval_args(
                argv=[
                    "--data",
                    "NVDA",
                    "--pretrain-checkpoint-path",
                    str(checkpoint),
                    "--forecast-horizon",
                    "1",
                ]
            )
            self.assertEqual(args.forecast_horizon, 1)
            self.assertNotIn("--forecast-horizon", passthrough)
            eval_argv, _ = build_eval_argv(args, passthrough)
            self.assertEqual(eval_argv.count("--forecast-horizon"), 1)
            horizon_index = eval_argv.index("--forecast-horizon")
            self.assertEqual(eval_argv[horizon_index + 1], "1")

    def test_pretraining_accepts_optional_eval_forecast_horizon(self):
        config = parse_pretrain_args(
            dict(pretrain_config),
            argv=["--eval-forecast-horizon", "1"],
        )
        self.assertEqual(config["eval_forecast_horizon"], 1)

    def test_omitted_horizon_does_not_change_legacy_fingerprint(self):
        args = parse_runner_args([])
        self.assertNotIn("forecast_horizon", effective_experiment_config(args))


if __name__ == "__main__":
    unittest.main()
