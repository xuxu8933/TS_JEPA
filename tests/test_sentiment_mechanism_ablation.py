import inspect
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import config.experiment as experiment_config
import src.data_loaders.data_class_roll_volume as roll_volume
import src.data_loaders.financial_preprocessing as financial_preprocessing
import run_top_nasdaq100_stocks as stock_runner
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


def write_sentiment_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


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


class SentimentFeatureTest(unittest.TestCase):
    @staticmethod
    def _write_dense_sentiment(path: Path, rows: int) -> None:
        values = np.linspace(-2.0, 3.0, rows)
        write_sentiment_csv(
            path,
            [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "sentiment_mean": float(value),
                    "news_count": int(index % 3 == 0),
                }
                for index, (date, value) in enumerate(
                    zip(pd.date_range("2020-01-01", periods=rows), values)
                )
            ],
        )

    def test_has_news_distinguishes_missing_neutral_and_signed_news_causally(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            price_path = root / "NVDA.csv"
            sentiment_path = root / "NVDA_daily_sentiment.csv"
            write_price_csv(price_path, rows=90)
            write_sentiment_csv(
                sentiment_path,
                [
                    {"date": "2020-02-20", "sentiment_mean": 0.0, "news_count": 1},
                    {"date": "2020-02-21", "sentiment_mean": 0.4, "news_count": 2},
                    {"date": "2020-02-22", "sentiment_mean": -0.3, "news_count": 1},
                    {"date": "2020-02-24", "sentiment_mean": 0.8, "news_count": 1},
                ],
            )
            prepared = roll_volume.load_price_series(
                path_data=str(price_path),
                feature_cols=("sentiment_mean", "has_news"),
                sentiment_path=str(sentiment_path),
                validation_fraction=0.1,
                test_start_date="2020-03-20",
                return_metadata=True,
            )
            rows = {}
            for dates, values in zip(prepared["dates"], prepared["features"]):
                rows.update(
                    {
                        pd.Timestamp(date).strftime("%Y-%m-%d"): tuple(value.tolist())
                        for date, value in zip(dates, values)
                    }
                )

            self.assertEqual(rows["2020-02-23"], (0.0, 0.0))
            self.assertEqual(rows["2020-02-20"], (0.0, 1.0))
            self.assertEqual(rows["2020-02-21"][1], 1.0)
            self.assertEqual(rows["2020-02-22"][1], 1.0)
            self.assertEqual(
                {row[1] for row in rows.values()},
                {0.0, 1.0},
            )
            self.assertEqual(rows["2020-02-23"], (0.0, 0.0))

    def test_selective_zscore_fits_train_only_and_reuses_state(self):
        transform = getattr(
            financial_preprocessing,
            "fit_transform_sentiment_features",
            None,
        )
        self.assertIsNotNone(transform)
        splits = (
            pd.DataFrame(
                {
                    "sentiment_mean": [-1.0, 0.0, 1.0],
                    "sentiment_mean_z": [-1.0, 0.0, 1.0],
                }
            ),
            pd.DataFrame(
                {"sentiment_mean": [1000.0], "sentiment_mean_z": [1000.0]}
            ),
            pd.DataFrame(
                {"sentiment_mean": [-1000.0], "sentiment_mean_z": [-1000.0]}
            ),
        )
        transformed, state = transform(
            splits,
            ["sentiment_mean_z"],
            "train_zscore",
        )
        feature_state = state["features"]["sentiment_mean_z"]
        self.assertEqual(state["fit_split"], "train")
        self.assertAlmostEqual(feature_state["mean"], 0.0)
        self.assertAlmostEqual(feature_state["std"], math.sqrt(2.0 / 3.0))
        self.assertEqual(transformed[0]["sentiment_mean_z"].mean(), 0.0)

        changed_holdout = (splits[0], splits[1] * 7.0, splits[2] * 11.0)
        _, changed_state = transform(
            changed_holdout,
            ["sentiment_mean_z"],
            "train_zscore",
        )
        self.assertEqual(changed_state, state)

        reused, reused_state = transform(
            splits,
            ["sentiment_mean_z"],
            "train_zscore",
            state=state,
        )
        self.assertEqual(reused_state, state)
        for first, second in zip(transformed, reused):
            self.assertTrue(first.equals(second))

    def test_loader_persists_train_only_selective_zscore_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            price_path = root / "NVDA.csv"
            sentiment_path = root / "NVDA_daily_sentiment.csv"
            write_price_csv(price_path, rows=180)
            self._write_dense_sentiment(sentiment_path, rows=180)
            common = {
                "path_data": str(price_path),
                "feature_cols": (
                    "Close",
                    "Volume",
                    "MA10",
                    "MA50",
                    "sentiment_mean_z",
                ),
                "sentiment_path": str(sentiment_path),
                "sentiment_normalization": "train_zscore",
                "validation_fraction": 0.1,
                "test_start_date": "2020-05-25",
                "series_split_size": 5,
                "patch_size": 5,
            }
            train = roll_volume.CSVDataLoader(split="train", **common)
            validation = roll_volume.CSVDataLoader(
                split="val",
                sentiment_normalization_stats=train.sentiment_normalization_stats,
                **common,
            )
            self.assertEqual(train.passthrough_indices, [4])
            self.assertEqual(
                train.sentiment_normalization_stats["fit_split"],
                "train",
            )
            self.assertEqual(
                validation.sentiment_normalization_stats,
                train.sentiment_normalization_stats,
            )
            self.assertAlmostEqual(float(train.train_df[:, 4].mean()), 0.0, places=6)

    def test_explicit_none_preserves_legacy_loader_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            price_path = root / "NVDA.csv"
            sentiment_path = root / "NVDA_daily_sentiment.csv"
            write_price_csv(price_path, rows=180)
            self._write_dense_sentiment(sentiment_path, rows=180)
            common = {
                "path_data": str(price_path),
                "feature_cols": ("Close", "Volume", "sentiment_mean"),
                "sentiment_path": str(sentiment_path),
                "validation_fraction": 0.1,
                "test_start_date": "2020-05-25",
                "series_split_size": 20,
                "patch_size": 5,
            }
            legacy = roll_volume.CSVDataLoader(split="train", **common)
            explicit = roll_volume.CSVDataLoader(
                split="train",
                sentiment_normalization="none",
                **common,
            )
            self.assertEqual(legacy.feature_cols, explicit.feature_cols)
            self.assertEqual(legacy.passthrough_indices, explicit.passthrough_indices)
            self.assertEqual(legacy.normalization_stats, explicit.normalization_stats)
            self.assertTrue(torch.equal(legacy.train_df, explicit.train_df))

    def test_runner_forwards_selective_sentiment_mode_to_both_stages(self):
        args = parse_runner_args(
            [
                "--sentiment-features",
                "sentiment_mean_z",
                "--sentiment-normalization",
                "train_zscore",
                "--stocks",
                "NVDA",
                "--max-stocks",
                "1",
                "--results-dir",
                "/tmp/ts-jepa-sentiment-test",
            ]
        )
        commands = build_stock_commands(
            args,
            "NVDA",
            seed=42,
            strategy="random",
        )
        for command in commands:
            mode_index = command.index("--sentiment-normalization")
            self.assertEqual(command[mode_index + 1], "train_zscore")

    def test_unified_evaluator_forwards_checkpoint_sentiment_state(self):
        state = {
            "mode": "train_zscore",
            "fit_split": "train",
            "features": {
                "sentiment_mean_z": {
                    "source": "sentiment_mean",
                    "mean": 0.25,
                    "std": 0.5,
                    "eps": 1e-6,
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.pt"
            torch.save(
                {
                    "epoch": 10,
                    "config": {
                        "feature_cols": [
                            "Close",
                            "Volume",
                            "MA10",
                            "MA50",
                            "sentiment_mean_z",
                        ],
                        "sentiment_features": ["sentiment_mean_z"],
                        "use_sentiment": True,
                        "sentiment_normalization": "train_zscore",
                        "sentiment_normalization_stats": state,
                    },
                },
                checkpoint,
            )
            args, passthrough = parse_eval_args(
                argv=[
                    "--data",
                    "NVDA",
                    "--pretrain-checkpoint-path",
                    str(checkpoint),
                    "--sentiment-normalization",
                    "train_zscore",
                ]
            )
            eval_argv, _ = build_eval_argv(args, passthrough)
            state_index = eval_argv.index(
                "--sentiment-normalization-stats-json"
            )
            self.assertEqual(json.loads(eval_argv[state_index + 1]), state)


class ConfigIsolationTest(unittest.TestCase):
    EXPECTED_STOCKS = [
        "NVDA",
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOGL",
        "AVGO",
        "META",
        "TSLA",
        "COST",
        "WMT",
    ]
    EXPECTED = {
        "top10_h1_without_sentiment.json": (
            1,
            ["Close", "Volume", "MA10", "MA50"],
            20,
        ),
        "top10_h1_with_sentiment.json": (
            1,
            ["Close", "Volume", "MA10", "MA50", "sentiment_mean"],
            25,
        ),
        "top10_sentiment_has_news.json": (
            5,
            [
                "Close",
                "Volume",
                "MA10",
                "MA50",
                "sentiment_mean",
                "has_news",
            ],
            30,
        ),
        "top10_sentiment_zscore.json": (
            5,
            ["Close", "Volume", "MA10", "MA50", "sentiment_mean_z"],
            25,
        ),
    }

    def test_new_configs_resolve_exact_semantics(self):
        from analysis.sentiment_mechanism import semantic_experiment_config

        repo_root = Path(__file__).resolve().parents[1]
        for filename, (horizon, features, dimension) in self.EXPECTED.items():
            with self.subTest(config=filename):
                snapshot = semantic_experiment_config(
                    repo_root / "config" / "experiments" / filename
                )
                self.assertEqual(snapshot["stocks"], self.EXPECTED_STOCKS)
                self.assertEqual(snapshot["seeds"], list(range(42, 52)))
                self.assertEqual(snapshot["patch_size"], 5)
                self.assertEqual(snapshot["forecast_horizon"], horizon)
                self.assertEqual(snapshot["feature_cols"], features)
                self.assertEqual(snapshot["input_dimension"], dimension)
                self.assertEqual(snapshot["mask_strategies"], ["random", "local_long"])
                self.assertEqual(
                    snapshot["results_dir"],
                    str(repo_root / "results" / Path(filename).stem),
                )

    def test_only_approved_semantic_differences_are_present(self):
        from analysis.sentiment_mechanism import validate_ablation_configs

        repo_root = Path(__file__).resolve().parents[1]
        report = validate_ablation_configs(repo_root)
        self.assertTrue(report["valid"], report)
        self.assertTrue(report["published_controls_verified"])
        self.assertEqual(set(report["configs"]), set(self.EXPECTED))


class DryRunSafetyTest(unittest.TestCase):
    def test_reports_exact_structured_values_for_all_configs(self):
        repo_root = Path(__file__).resolve().parents[1]
        expected = ConfigIsolationTest.EXPECTED
        for filename, (horizon, features, dimension) in expected.items():
            with self.subTest(config=filename):
                config_path = repo_root / "config" / "experiments" / filename
                args = stock_runner.parse_args(
                    ["--config", str(config_path), "--dry-run"]
                )
                report_builder = getattr(
                    stock_runner,
                    "build_dry_run_report",
                    None,
                )
                self.assertIsNotNone(report_builder)
                report = report_builder(
                    args,
                    stock_runner.resolve_stocks(args),
                    stock_runner.resolve_seeds(args),
                    stock_runner.resolve_mask_strategies(args),
                )
                self.assertEqual(report["experiment_name"], Path(filename).stem)
                self.assertEqual(report["git_branch"], "single-dim")
                self.assertEqual(report["stock_count"], 10)
                self.assertEqual(report["stocks"], ConfigIsolationTest.EXPECTED_STOCKS)
                self.assertEqual(report["seed_count"], 10)
                self.assertEqual(report["seeds"], list(range(42, 52)))
                self.assertEqual(report["forecast_horizon"], horizon)
                self.assertEqual(report["feature_names"], features)
                self.assertEqual(report["feature_count"], len(features))
                self.assertEqual(report["patch_size"], 5)
                self.assertEqual(
                    report["flattened_patch_input_dimension"],
                    dimension,
                )
                self.assertTrue(report["training_disabled"])

    def test_dry_run_returns_before_writes_or_execution(self):
        repo_root = Path(__file__).resolve().parents[1]
        config_path = (
            repo_root
            / "config/experiments/top10_h1_with_sentiment.json"
        )
        fake_plan = {
            "requested_runs": [("NVDA", 42)],
            "completed_runs": set(),
            "missing_runs": [("NVDA", 42)],
            "tasks": [{"would": "train"}],
        }
        real_path_open = Path.open

        def reject_writes(path, mode="r", *args, **kwargs):
            if any(marker in mode for marker in ("w", "a", "+", "x")):
                raise AssertionError("file write reached")
            return real_path_open(path, mode, *args, **kwargs)

        output = io.StringIO()
        with (
            patch.object(
                stock_runner,
                "validate_existing_experiment",
                return_value=False,
            ),
            patch.object(stock_runner, "reject_duplicate_experiment_config"),
            patch.object(
                stock_runner,
                "plan_incremental_execution",
                return_value=fake_plan,
            ),
            patch.object(
                stock_runner,
                "run_command",
                side_effect=AssertionError("execution reached"),
            ),
            patch.object(
                stock_runner,
                "execute_tasks",
                side_effect=AssertionError("training reached"),
            ),
            patch.object(
                stock_runner,
                "_write_json",
                side_effect=AssertionError("write reached"),
            ),
            patch.object(
                Path,
                "mkdir",
                side_effect=AssertionError("directory creation reached"),
            ),
            patch.object(Path, "open", new=reject_writes),
            redirect_stdout(output),
        ):
            stock_runner.main(
                ["--config", str(config_path), "--dry-run"]
            )
        rendered = output.getvalue()
        self.assertIn("DRY_RUN_VALIDATION", rendered)
        self.assertIn('"training_disabled": true', rendered)


if __name__ == "__main__":
    unittest.main()
