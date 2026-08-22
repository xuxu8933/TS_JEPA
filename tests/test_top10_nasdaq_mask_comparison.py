import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import pandas as pd

from analyze_stock_results import (
    aggregate_metrics,
    aggregate_strategy_runs,
    collect_raw_results,
    parse_args as parse_analysis_args,
    resolve_analysis_scope,
    write_summaries,
)
from run_top_nasdaq100_stocks import (
    build_stock_commands,
    parse_args as parse_stock_runner_args,
    resolve_mask_strategies,
    resolve_seeds,
    strategy_results_dir,
    validate_runner_mask_geometry,
)


class StockMaskComparisonTest(unittest.TestCase):
    def test_shared_config_files_drive_runner_and_analyzer(self):
        repo_root = Path(__file__).resolve().parents[1]
        cases = (
            ("top10_with_sentiment.json", True, "top10_with_sentiment"),
            ("top10_without_sentiment.json", False, "top10_without_sentiment"),
        )

        for filename, use_sentiment, result_name in cases:
            config_path = repo_root / "config" / "experiments" / filename
            config_data = json.loads(config_path.read_text())
            runner_args = parse_stock_runner_args(["--config", str(config_path)])
            analysis_args = parse_analysis_args(["--config", str(config_path)])

            runner_config_keys = set(config_data["common"]) | set(
                config_data["runner"]
            )
            analysis_config_keys = set(config_data["common"]) | set(
                config_data["analysis"]
            )
            self.assertEqual(
                set(vars(runner_args)) - {"config"},
                runner_config_keys,
            )
            self.assertEqual(
                set(vars(analysis_args)) - {"config"},
                analysis_config_keys,
            )

            self.assertEqual(len(runner_args.stocks), 10)
            self.assertEqual(resolve_seeds(runner_args), list(range(42, 52)))
            self.assertEqual(
                resolve_mask_strategies(runner_args),
                ["random", "local_long"],
            )
            self.assertEqual(runner_args.use_sentiment, use_sentiment)
            self.assertEqual(runner_args.series_split_size, 120)
            self.assertEqual(runner_args.patch_size, 5)
            self.assertEqual(runner_args.download_start_date, "2015-01-01")
            self.assertEqual(runner_args.download_end_date, "2026-01-01")
            self.assertFalse(hasattr(runner_args, "start_date"))
            self.assertFalse(hasattr(runner_args, "end_date"))
            validate_runner_mask_geometry(
                runner_args,
                resolve_mask_strategies(runner_args),
            )
            self.assertTrue(runner_args.results_dir.endswith(result_name))
            self.assertEqual(analysis_args.stocks, runner_args.stocks)
            self.assertEqual(analysis_args.seeds, runner_args.seeds)
            self.assertEqual(
                analysis_args.strategies,
                runner_args.mask_strategies,
            )

    def test_cli_strategy_and_seed_override_configured_values(self):
        repo_root = Path(__file__).resolve().parents[1]
        config_path = (
            repo_root
            / "config"
            / "experiments"
            / "top10_with_sentiment.json"
        )
        args = parse_stock_runner_args(
            [
                "--config",
                str(config_path),
                "--mask-strategies",
                "future_block",
                "--seed",
                "7",
                "--no-sentiment",
                "--no-skip-download",
            ]
        )

        self.assertEqual(resolve_mask_strategies(args), ["future_block"])
        self.assertEqual(resolve_seeds(args), [7])
        self.assertFalse(args.use_sentiment)
        self.assertFalse(args.skip_download)

    def test_runner_rejects_invalid_structured_mask_geometry_before_training(self):
        args = parse_stock_runner_args(
            [
                "--mask-strategies",
                "local_long",
                "--series-split-size",
                "20",
                "--patch-size",
                "5",
            ]
        )

        with self.assertRaisesRegex(ValueError, "4 patches"):
            validate_runner_mask_geometry(args, ["local_long"])

    def test_toml_config_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "experiment.toml"
            config_path.write_text(
                "[common]\n"
                'stocks = ["NVDA"]\n'
                "seeds = [3, 5]\n"
                f'results_dir = "{tmp}/results"\n'
                "[runner]\n"
                'mask_strategies = ["random", "local_long"]\n'
                "use_sentiment = false\n"
                "[analysis]\n"
                'strategies = ["random", "local_long"]\n'
                'models = ["TS-JEPA"]\n'
            )

            runner_args = parse_stock_runner_args(["--config", str(config_path)])
            analysis_args = parse_analysis_args(["--config", str(config_path)])

            self.assertEqual(resolve_seeds(runner_args), [3, 5])
            self.assertFalse(runner_args.use_sentiment)
            self.assertEqual(analysis_args.models, ["TS-JEPA"])

    def test_config_rejects_unknown_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "invalid.json"
            config_path.write_text('{"runner": {"mask_stratey": "random"}}')

            with self.assertRaisesRegex(ValueError, "mask_stratey"):
                parse_stock_runner_args(["--config", str(config_path)])

    def test_runner_rejects_removed_singular_strategy_option(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_stock_runner_args(["--mask-strategy", "random"])

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "invalid.json"
            config_path.write_text('{"runner": {"mask_strategy": "random"}}')

            with self.assertRaisesRegex(ValueError, "mask_strategy"):
                parse_stock_runner_args(["--config", str(config_path)])

    def test_runner_isolates_all_strategies(self):
        with tempfile.TemporaryDirectory() as tmp:
            multi_args = parse_stock_runner_args(
                [
                    "--stocks",
                    "NVDA",
                    "--max-stocks",
                    "0",
                    "--mask-strategies",
                    "random",
                    "local_long",
                    "--results-dir",
                    tmp,
                ]
            )
            strategies = resolve_mask_strategies(multi_args)
            self.assertEqual(strategies, ["random", "local_long"])

            for strategy in strategies:
                strategy_dir = strategy_results_dir(
                    multi_args,
                    strategy,
                )
                commands = build_stock_commands(
                    multi_args,
                    "NVDA",
                    strategy=strategy,
                    results_dir=strategy_dir,
                )
                eval_command = commands[-1]
                output_dir = eval_command[eval_command.index("--results-dir") + 1]
                self.assertIn(f"/{strategy}/NVDA/seed_42", output_dir)

            single_args = parse_stock_runner_args(
                ["--mask-strategies", "random", "--results-dir", tmp]
            )
            self.assertEqual(
                strategy_results_dir(single_args, "random"),
                Path(tmp) / "random",
            )

    def test_collect_and_aggregate_calculates_sample_standard_deviation(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            for seed, mse in ((1, 0.1), (2, 0.3)):
                run_dir = results_dir / "NVDA" / f"seed_{seed}"
                run_dir.mkdir(parents=True)
                txt_path = run_dir / "last_model_comparison_20260101_000000.txt"
                txt_path.write_text("Data source: NVDA\n")
                txt_path.with_suffix(".csv").write_text(
                    "model,mse,mae,trend_accuracy\n"
                    f"TS-JEPA,{mse},{mse + 0.1},0.6\n"
                )

            raw = collect_raw_results(
                results_dir,
                "random",
                ["NVDA"],
                [1, 2],
            )
            summary = aggregate_metrics(
                raw,
                ["strategy", "stock", "model"],
            )

        self.assertEqual(int(summary.iloc[0]["num_runs"]), 2)
        self.assertAlmostEqual(float(summary.iloc[0]["mse_mean"]), 0.2)
        self.assertAlmostEqual(
            float(summary.iloc[0]["mse_std"]),
            0.1414213562,
        )

    def test_overall_deviation_is_across_seeded_runs_not_across_stocks(self):
        raw = pd.DataFrame(
            [
                {
                    "strategy": "random",
                    "stock": stock,
                    "seed": seed,
                    "model": "TS-JEPA",
                    "mse": mse,
                    "mae": mse,
                    "trend_accuracy": mse,
                }
                for seed, stock_values in (
                    (1, (("NVDA", 0.1), ("AAPL", 0.9))),
                    (2, (("NVDA", 0.3), ("AAPL", 1.1))),
                )
                for stock, mse in stock_values
            ]
        )

        per_run, overall = aggregate_strategy_runs(raw)

        self.assertAlmostEqual(float(per_run.iloc[0]["mse"]), 0.5)
        self.assertAlmostEqual(float(per_run.iloc[1]["mse"]), 0.7)
        self.assertEqual(int(overall.iloc[0]["num_runs"]), 2)
        self.assertAlmostEqual(float(overall.iloc[0]["mse_mean"]), 0.6)
        self.assertAlmostEqual(
            float(overall.iloc[0]["mse_std"]),
            0.1414213562,
        )

    def test_standalone_analyzer_writes_paired_strategy_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            for strategy, mse, trend in (
                ("random", 0.1, 0.55),
                ("local_long", 0.2, 0.65),
            ):
                run_dir = results_dir / strategy / "NVDA" / "seed_1"
                run_dir.mkdir(parents=True)
                txt_path = run_dir / "last_model_comparison_20260101_000000.txt"
                txt_path.write_text("Data source: NVDA\n")
                txt_path.with_suffix(".csv").write_text(
                    "model,mse,mae,trend_accuracy\n"
                    f"TS-JEPA,{mse},{mse},{trend}\n"
                )

            args = parse_analysis_args(
                [
                    "--results-dir",
                    str(results_dir),
                    "--strategies",
                    "random",
                    "local_long",
                    "--stocks",
                    "NVDA",
                    "--seeds",
                    "1",
                    "--models",
                    "TS-JEPA",
                ]
            )
            strategies, stocks, seeds = resolve_analysis_scope(args)
            write_summaries(args, stocks, seeds, strategies)

            paired = pd.read_csv(
                results_dir / "paired_strategy_differences.csv"
            )
            missing = pd.read_csv(results_dir / "missing_or_failed_runs.csv")

            self.assertTrue((results_dir / "raw_runs.csv").exists())
            self.assertTrue((results_dir / "per_stock_summary.csv").exists())
            self.assertTrue((results_dir / "per_seed_summary.csv").exists())
            self.assertTrue((results_dir / "overall_summary.csv").exists())
            self.assertTrue((results_dir / "strategy_comparison.png").exists())
            self.assertTrue(missing.empty)
            mse_row = paired[paired["metric"] == "mse"].iloc[0]
            self.assertAlmostEqual(float(mse_row["mean_delta_b_minus_a"]), 0.1)
            self.assertEqual(mse_row["better_strategy"], "random")

    def test_analyzer_records_missing_runs_before_refusing_partial_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            run_dir = results_dir / "random" / "NVDA" / "seed_1"
            run_dir.mkdir(parents=True)
            txt_path = run_dir / "last_model_comparison_20260101_000000.txt"
            txt_path.write_text("Data source: NVDA\n")
            txt_path.with_suffix(".csv").write_text(
                "model,mse,mae,trend_accuracy\nTS-JEPA,0.1,0.2,0.6\n"
            )
            args = parse_analysis_args(
                [
                    "--results-dir",
                    str(results_dir),
                    "--strategies",
                    "random",
                    "--stocks",
                    "NVDA",
                    "--seeds",
                    "1",
                    "2",
                    "--models",
                    "TS-JEPA",
                    "--skip-plot",
                ]
            )

            with self.assertRaisesRegex(RuntimeError, "incomplete or invalid"):
                write_summaries(args, ["NVDA"], [1, 2], ["random"])

            issues = pd.read_csv(results_dir / "missing_or_failed_runs.csv")
            self.assertEqual(issues.iloc[0]["status"], "missing_result_file")
            self.assertFalse((results_dir / "overall_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
