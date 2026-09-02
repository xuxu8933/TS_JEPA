import inspect
import io
import json
import math
import subprocess
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
from eval_forecast_prequential_with_baselines_gru_volume import (
    compute_trend_accuracy,
    directional_auxiliary_loss,
    prequential_baseline_evaluate,
)
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

    def test_explicit_patch_sized_horizon_keeps_legacy_fingerprint(self):
        implicit = parse_runner_args([])
        explicit = parse_runner_args(["--forecast-horizon", "5"])
        self.assertEqual(
            effective_experiment_config(explicit),
            effective_experiment_config(implicit),
        )

    def test_h1_value_direction_uses_last_observed_context(self):
        predictions = np.array([[0.60], [0.40]], dtype=np.float32)
        targets = np.array([[0.55], [0.45]], dtype=np.float32)
        origins = np.array([0.50, 0.50], dtype=np.float32)
        accuracy = compute_trend_accuracy(
            predictions,
            targets,
            origins=origins,
        )
        self.assertTrue(math.isfinite(accuracy))
        self.assertEqual(accuracy, 1.0)

    def test_h1_value_direction_requires_an_observed_origin(self):
        with self.assertRaisesRegex(ValueError, "forecast origin"):
            compute_trend_accuracy(
                np.array([[0.60]], dtype=np.float32),
                np.array([[0.55]], dtype=np.float32),
            )

    def test_h1_directional_loss_uses_observed_origin(self):
        target = torch.tensor([[0.55]], dtype=torch.float32)
        origin = torch.tensor([0.50], dtype=torch.float32)
        aligned = directional_auxiliary_loss(
            torch.tensor([[0.60]], dtype=torch.float32),
            target,
            origin=origin,
        )
        opposed = directional_auxiliary_loss(
            torch.tensor([[0.40]], dtype=torch.float32),
            target,
            origin=origin,
        )
        self.assertTrue(torch.isfinite(aligned))
        self.assertLess(aligned.item(), opposed.item())

    def test_h1_multifeature_baseline_scores_finite_direction(self):
        class TinyDataset:
            def __len__(self):
                return 2

            def __getitem__(self, index):
                context = torch.tensor(
                    [
                        [0.1, 1.0, 0.2, 1.0, 0.3, 1.0, 0.4, 1.0, 0.5, 1.0],
                        [0.6, 1.0, 0.7, 1.0, 0.8, 1.0, 0.9, 1.0, 1.0, 1.0],
                    ],
                    dtype=torch.float32,
                )
                target = torch.tensor([1.1 - 0.2 * index], dtype=torch.float32)
                return context, target

        config = {
            "patch_size": 5,
            "forecast_horizon": 1,
            "feature_dim": 2,
            "target_feature_index": 0,
            "forecast_target": "value",
            "normalization": "none",
            "feature_transform": "raw",
            "eval_type": "last",
        }
        with tempfile.TemporaryDirectory() as tmp:
            summary, *_ = prequential_baseline_evaluate(
                TinyDataset(),
                config,
                baseline_names=["naive_last"],
                save_dir=tmp,
            )
        self.assertTrue(math.isfinite(summary[0]["trend_accuracy"]))


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
        expected_branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
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
                self.assertEqual(report["git_branch"], expected_branch)
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


class PairedAnalysisTest(unittest.TestCase):
    @staticmethod
    def _canonical_fixture(condition: str, offset: float = 0.0) -> pd.DataFrame:
        rows = []
        for stock in ("AAPL", "NVDA"):
            for seed in (42, 43):
                for model in (
                    "TS-JEPA/random",
                    "TS-JEPA/local_long",
                    "GRU/random",
                ):
                    for metric in ("rmse", "direction_accuracy"):
                        rows.append(
                            {
                                "condition": condition,
                                "stock": stock,
                                "seed": seed,
                                "model": model,
                                "metric": metric,
                                "value": 1.0 + offset,
                                "forecast_horizon": 5,
                                "source_file": f"{condition}.csv",
                            }
                        )
        return pd.DataFrame(rows).sample(frac=1.0, random_state=7).reset_index(drop=True)

    def test_pairing_is_deterministic_and_strict(self):
        from analysis.sentiment_mechanism import pair_condition_results

        control = self._canonical_fixture("control")
        intervention = self._canonical_fixture("intervention", offset=0.25)
        pairs = pair_condition_results(control, intervention, "H2")
        identifiers = list(
            pairs[["stock", "seed"]].drop_duplicates().itertuples(
                index=False, name=None
            )
        )
        self.assertEqual(identifiers, sorted(identifiers))
        self.assertEqual(len(pairs), 2 * 2 * 3 * 2)
        self.assertTrue(np.allclose(pairs["delta"], 0.25))

        duplicate = pd.concat([control, control.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            pair_condition_results(duplicate, intervention, "H2")
        with self.assertRaisesRegex(ValueError, "missing paired"):
            pair_condition_results(control.iloc[:-1], intervention, "H2")
        wrong_horizon = intervention.copy()
        wrong_horizon["forecast_horizon"] = 1
        with self.assertRaisesRegex(ValueError, "forecast horizon"):
            pair_condition_results(control, wrong_horizon, "H2")
        non_finite = intervention.copy()
        non_finite.loc[0, "value"] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            pair_condition_results(control, non_finite, "H2")

    def test_known_student_t_statistics_and_holm_adjustment(self):
        from analysis.sentiment_mechanism import (
            holm_adjust,
            paired_stock_statistics,
        )

        deltas = [1.0, 2.0, 3.0, 4.0, 5.0]
        pairs = pd.DataFrame(
            {
                "hypothesis": ["H2"] * 5,
                "stock": [f"S{index}" for index in range(5)],
                "seed": [42] * 5,
                "model": ["TS-JEPA/random"] * 5,
                "metric": ["rmse"] * 5,
                "delta": deltas,
            }
        )
        row = paired_stock_statistics(pairs, expected_stock_count=5).iloc[0]
        sample_std = math.sqrt(2.5)
        expected_t = 3.0 / (sample_std / math.sqrt(5.0))
        half_width = 2.7764451051977987 * sample_std / math.sqrt(5.0)
        self.assertAlmostEqual(row["mean_delta"], 3.0)
        self.assertAlmostEqual(row["std_delta"], sample_std)
        self.assertAlmostEqual(row["t_stat"], expected_t)
        self.assertAlmostEqual(row["dz"], 3.0 / sample_std)
        self.assertAlmostEqual(row["ci_low"], 3.0 - half_width, places=6)
        self.assertAlmostEqual(row["ci_high"], 3.0 + half_width, places=6)
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_raw_loader_rejects_run_manifest_identity_mismatch(self):
        from analysis.sentiment_mechanism import (
            load_raw_experiment_results,
            semantic_experiment_config,
        )
        from run_top_nasdaq100_stocks import experiment_config_signature

        repo_root = Path(__file__).resolve().parents[1]
        expected = semantic_experiment_config(
            repo_root / "config/experiments/top10_sentiment_has_news.json"
        )
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            root_manifest = {
                "effective_config": expected["effective_config"],
                "config_signature": experiment_config_signature(
                    expected["effective_config"]
                ),
                "stocks": ["NVDA"],
                "seeds": [42],
                "mask_strategies": ["random", "local_long"],
            }
            (results_dir / "experiment_manifest.json").write_text(
                json.dumps(root_manifest), encoding="utf-8"
            )
            for strategy in ("random", "local_long"):
                run_dir = results_dir / strategy / "NVDA" / "seed_42"
                run_dir.mkdir(parents=True)
                rows = [
                    {
                        "model": "TS-JEPA",
                        "rmse": 0.1,
                        "direction_accuracy": 0.6,
                    }
                ]
                if strategy == "random":
                    rows.append(
                        {
                            "model": "GRU",
                            "rmse": 0.2,
                            "direction_accuracy": 0.5,
                        }
                    )
                comparison = "last_model_comparison_fixture.csv"
                pd.DataFrame(rows).to_csv(run_dir / comparison, index=False)
                metadata = {
                    **expected["data_protocol"],
                    "forecast_horizon": expected["forecast_horizon"],
                    "forecast_target": expected["forecast_target"],
                    "feature_names": expected["feature_cols"],
                    "normalization": expected["normalization"],
                    "sentiment_normalization": expected[
                        "sentiment_normalization"
                    ],
                }
                (run_dir / "preprocessing_config.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                run_manifest = {
                    "status": "complete",
                    "config_signature": root_manifest["config_signature"],
                    "effective_config": expected["effective_config"],
                    "strategy": strategy,
                    "stock": "NVDA",
                    "seed": 42,
                    "comparison_files": [comparison],
                }
                (run_dir / "run_manifest.json").write_text(
                    json.dumps(run_manifest), encoding="utf-8"
                )

            loaded = load_raw_experiment_results(
                results_dir,
                "has_news",
                ["NVDA"],
                [42],
                expected_semantics=expected,
            )
            self.assertEqual(len(loaded), 6)

            bad_manifest_path = (
                results_dir / "local_long/NVDA/seed_42/run_manifest.json"
            )
            bad_manifest = json.loads(bad_manifest_path.read_text())
            bad_manifest["seed"] = 999
            bad_manifest_path.write_text(json.dumps(bad_manifest))
            with self.assertRaisesRegex(ValueError, "manifest identity"):
                load_raw_experiment_results(
                    results_dir,
                    "has_news",
                    ["NVDA"],
                    [42],
                    expected_semantics=expected,
                )

            bad_manifest["seed"] = 42
            bad_manifest_path.write_text(json.dumps(bad_manifest))
            metadata_path = (
                results_dir / "local_long/NVDA/seed_42/preprocessing_config.json"
            )
            bad_metadata = json.loads(metadata_path.read_text())
            bad_metadata["test_start"] = "2099-01-01"
            metadata_path.write_text(json.dumps(bad_metadata))
            with self.assertRaisesRegex(ValueError, "preprocessing metadata"):
                load_raw_experiment_results(
                    results_dir,
                    "has_news",
                    ["NVDA"],
                    [42],
                    expected_semantics=expected,
                )


class MechanismReportTest(unittest.TestCase):
    @staticmethod
    def _complete_results(condition: str, horizon: int, offset: float = 0.0):
        rows = []
        for stock in ConfigIsolationTest.EXPECTED_STOCKS:
            for seed in range(42, 52):
                for model in (
                    "TS-JEPA/random",
                    "TS-JEPA/local_long",
                    "GRU/random",
                ):
                    for metric in ("rmse", "direction_accuracy"):
                        base = 1.0 if metric != "direction_accuracy" else 0.5
                        rows.append(
                            {
                                "condition": condition,
                                "stock": stock,
                                "seed": seed,
                                "model": model,
                                "metric": metric,
                                "value": base + offset,
                                "forecast_horizon": horizon,
                                "source_file": f"{condition}.csv",
                            }
                        )
        return pd.DataFrame(rows)

    def test_missing_results_exit_cleanly_without_package(self):
        from analysis.sentiment_mechanism import main as analysis_main

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "not-created"
            output = io.StringIO()
            with redirect_stdout(output):
                status = analysis_main(["--output-root", str(output_root)])
            self.assertEqual(status, 0)
            self.assertEqual(
                output.getvalue().strip(),
                "Experiment results not found; run the corresponding experiment first.",
            )
            self.assertFalse(output_root.exists())

    def test_integrity_error_exits_nonzero_instead_of_claiming_missing_results(self):
        from analysis.sentiment_mechanism import main as analysis_main

        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch(
                "analysis.sentiment_mechanism.run_mechanism_analysis",
                side_effect=ValueError("corrupt manifest"),
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            status = analysis_main([])
        self.assertEqual(status, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("corrupt manifest", stderr.getvalue())

    def test_complete_inputs_create_exact_deferred_package(self):
        from analysis.sentiment_mechanism import main as analysis_main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_paths = {
                "h1_without": root / "h1_without",
                "h1_with": root / "h1_with",
                "has_news": root / "has_news",
                "zscore": root / "zscore",
                "with_control": root / "with_control.csv",
                "without_control": root / "without_control.csv",
            }
            for name, path in input_paths.items():
                if name.endswith("control"):
                    path.touch()
                else:
                    path.mkdir()
            output_root = root / "packages"
            datasets = {
                "h1_without": self._complete_results("h1_without", 1),
                "h1_with": self._complete_results("h1_with", 1, -0.1),
                "has_news": self._complete_results("has_news", 5, -0.1),
                "zscore": self._complete_results("zscore", 5, -0.05),
                "with_control": self._complete_results("with_control", 5),
                "without_control": self._complete_results("without_control", 5),
            }

            def raw_loader(path, condition, stocks, seeds, expected_semantics=None):
                return datasets[condition].copy()

            def published_loader(path, condition):
                return datasets[condition].copy()

            h3_stats = {
                stock: {
                    "mode": "train_zscore",
                    "fit_split": "train",
                    "features": {
                        "sentiment_mean_z": {
                            "source": "sentiment_mean",
                            "mean": 0.0,
                            "std": 1.0,
                            "eps": 1e-6,
                        }
                    },
                }
                for stock in ConfigIsolationTest.EXPECTED_STOCKS
            }
            argv = [
                "--output-root",
                str(output_root),
                "--run-id",
                "test_run",
                "--h1-without-results",
                str(input_paths["h1_without"]),
                "--h1-with-results",
                str(input_paths["h1_with"]),
                "--has-news-results",
                str(input_paths["has_news"]),
                "--zscore-results",
                str(input_paths["zscore"]),
                "--with-control",
                str(input_paths["with_control"]),
                "--without-control",
                str(input_paths["without_control"]),
            ]
            with (
                patch(
                    "analysis.sentiment_mechanism.load_raw_experiment_results",
                    side_effect=raw_loader,
                ),
                patch(
                    "analysis.sentiment_mechanism.load_published_results",
                    side_effect=published_loader,
                ),
                patch(
                    "analysis.sentiment_mechanism._collect_h3_normalization_stats",
                    return_value=h3_stats,
                ),
            ):
                status = analysis_main(argv)
            self.assertEqual(status, 0)
            package = output_root / "test_run"
            expected_files = {
                "data/mechanism_summary.csv",
                "data/per_stock_deltas.csv",
                "data/per_seed_deltas.csv",
                "data/h1_short_horizon_results.csv",
                "provenance/experiment_manifest.json",
                "sentiment_mechanism_report.md",
            }
            self.assertEqual(
                {
                    str(path.relative_to(package))
                    for path in package.rglob("*")
                    if path.is_file()
                },
                expected_files,
            )
            summary = pd.read_csv(package / "data/mechanism_summary.csv")
            self.assertEqual(
                summary.columns.tolist(),
                [
                    "hypothesis",
                    "intervention",
                    "control",
                    "model",
                    "metric",
                    "control_mean",
                    "intervention_mean",
                    "absolute_delta",
                    "percent_delta",
                    "stock_win_count",
                    "stock_total",
                    "seed_pair_win_rate",
                    "paired_t",
                    "paired_p",
                    "paired_p_holm",
                    "cohens_dz",
                    "ci95_low",
                    "ci95_high",
                    "verdict",
                ],
            )
            h1 = pd.read_csv(package / "data/h1_short_horizon_results.csv")
            self.assertEqual(set(h1["forecast_horizon"]), {1})
            report = (package / "sentiment_mechanism_report.md").read_text()
            self.assertTrue(report.startswith("# Sentiment Mechanism Ablation Report"))
            self.assertIn("## Executive summary", report)
            provenance = json.loads(
                (package / "provenance/experiment_manifest.json").read_text()
            )
            for key in (
                "git_branch",
                "git_commit",
                "configs",
                "coverage",
                "approved_changes",
                "input_paths",
                "h3_sentiment_normalization_stats",
            ):
                self.assertIn(key, provenance)
            self.assertEqual(
                provenance["h3_sentiment_normalization_stats"], h3_stats
            )


class SentimentMechanismDocumentationTest(unittest.TestCase):
    def test_workflow_and_new_config_keys_are_documented(self):
        repo_root = Path(__file__).resolve().parents[1]
        readme = (repo_root / "README.md").read_text(encoding="utf-8")
        configuration = (repo_root / "doc/configuration.md").read_text(
            encoding="utf-8"
        )
        for config_name in (
            "top10_h1_without_sentiment.json",
            "top10_h1_with_sentiment.json",
            "top10_sentiment_has_news.json",
            "top10_sentiment_zscore.json",
        ):
            self.assertIn(config_name, readme)
        self.assertIn("analyze_sentiment_mechanisms.py", readme)
        self.assertIn("no directories, files, or subprocesses", readme)
        self.assertIn('"forecast_horizon": 1', configuration)
        self.assertIn('"normalization": "train_zscore"', configuration)
        self.assertIn("defaults to `patch_size`", configuration)
        self.assertIn("training split separately for each stock", configuration)


if __name__ == "__main__":
    unittest.main()
