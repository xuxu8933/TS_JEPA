import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from config.file_options import (
    flatten_runner_options,
    read_config_file,
    results_dir_from_config,
)
from analyze_stock_results import (
    aggregate_metrics,
    aggregate_strategy_runs,
    collect_raw_results,
    parse_args as parse_analysis_args,
    resolve_analysis_scope,
    write_summaries,
)
from run_top_nasdaq100_stocks import (
    build_combined_plot_command,
    build_experiment_manifest,
    build_stock_commands,
    compatible_result_directories,
    effective_experiment_config,
    experiment_config_signature,
    main as run_stock_main,
    parse_args as parse_stock_runner_args,
    plan_incremental_execution,
    requested_stock_seed_runs,
    reject_duplicate_experiment_config,
    run_command,
    resolve_mask_strategies,
    resolve_seeds,
    resolve_stocks,
    stock_result_dir,
    strategy_results_dir,
    validate_existing_experiment,
    validate_runner_mask_geometry,
)


class StockMaskComparisonTest(unittest.TestCase):
    def _incremental_args(self, results_dir, stocks, seeds, *extra):
        return parse_stock_runner_args(
            [
                "--stocks",
                *stocks,
                "--seeds",
                *map(str, seeds),
                "--max-stocks",
                "0",
                "--max-seeds",
                "0",
                "--results-dir",
                str(results_dir),
                "--mask-strategies",
                "random",
                *extra,
            ]
        )

    def _write_experiment_manifest(self, args, stocks, seeds):
        manifest = build_experiment_manifest(args, stocks, seeds, ["random"])
        manifest_path = Path(args.results_dir) / "experiment_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest))

    def _write_complete_run(self, args, stock, seed):
        run_dir = stock_result_dir(args, "random", stock, seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        txt_path = run_dir / "last_model_comparison_20260101_000000.txt"
        txt_path.write_text(f"Data source: {stock}\n")
        txt_path.with_suffix(".csv").write_text(
            "model,mse,mae,trend_accuracy\nTS-JEPA,0.1,0.2,0.6\n"
        )
        return run_dir

    def _fake_stock_commands(self, checkpoint_root):
        def build(args, stock, seed=None, strategy=None, results_dir=None):
            checkpoint_path = checkpoint_root / strategy / stock / f"seed_{seed}.pt"
            run_dir = stock_result_dir(args, strategy, stock, seed)
            return [
                ["python", "pretrain_dual_loss.py", "--data", stock],
                [
                    "python",
                    "eval_dual_loss.py",
                    "--pretrain-checkpoint-path",
                    str(checkpoint_path),
                    "--results-dir",
                    str(run_dir),
                ],
            ]

        return build

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

            flattened_runner = flatten_runner_options(
                config_data["runner"], config_path
            )
            analysis_config_keys = (
                set(config_data["common"])
                | set(config_data["analysis"])
                | {"strategies"}
            )
            for key, value in {**config_data["common"], **flattened_runner}.items():
                self.assertEqual(getattr(runner_args, key), value)
            self.assertEqual(
                set(vars(analysis_args)) - {"config", "results_dir"},
                analysis_config_keys,
            )

            self.assertEqual(len(runner_args.stocks), 10)
            configured_stocks = list(config_data["common"]["stocks"])
            if runner_args.max_stocks > 0:
                configured_stocks = configured_stocks[:runner_args.max_stocks]
            self.assertEqual(resolve_stocks(runner_args), configured_stocks)
            configured_seeds = list(config_data["common"]["seeds"])
            if runner_args.max_seeds > 0:
                configured_seeds = configured_seeds[:runner_args.max_seeds]
            self.assertEqual(resolve_seeds(runner_args), configured_seeds)
            self.assertFalse(runner_args.verbose)
            self.assertEqual(
                resolve_mask_strategies(runner_args),
                ["random", "local_long"],
            )
            self.assertEqual(runner_args.use_sentiment, use_sentiment)
            self.assertFalse(hasattr(runner_args, "feature_cols"))
            self.assertEqual(runner_args.patch_size, 5)
            self.assertTrue(runner_args.skip_download)
            self.assertFalse(hasattr(runner_args, "start_date"))
            self.assertFalse(hasattr(runner_args, "end_date"))
            validate_runner_mask_geometry(
                runner_args,
                resolve_mask_strategies(runner_args),
            )
            self.assertTrue(runner_args.results_dir.endswith(result_name))
            self.assertEqual(
                Path(runner_args.results_dir),
                results_dir_from_config(config_path),
            )
            self.assertEqual(analysis_args.stocks, runner_args.stocks)
            self.assertEqual(analysis_args.seeds, runner_args.seeds)
            self.assertEqual(
                analysis_args.strategies,
                runner_args.mask_strategies,
            )

    def test_all_named_configs_use_matching_result_directories_and_common_coverage(self):
        config_dir = Path(__file__).resolve().parents[1] / "config" / "experiments"
        for config_path in sorted(config_dir.glob("*.json")):
            _, config_data = read_config_file(config_path)
            self.assertIn("stocks", config_data["common"])
            self.assertIn("seeds", config_data["common"])
            self.assertNotIn("stocks", config_data.get("runner", {}))
            self.assertNotIn("seeds", config_data.get("runner", {}))
            self.assertNotIn("stocks", config_data.get("analysis", {}))
            self.assertNotIn("seeds", config_data.get("analysis", {}))
            self.assertNotIn("results_dir", config_data["common"])
            runner_args = parse_stock_runner_args(
                ["--config", str(config_path)]
            )
            self.assertEqual(
                Path(runner_args.results_dir),
                results_dir_from_config(config_path),
            )

    def test_commented_template_documents_every_config_input(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "experiments"
            / "template_experiment.jsonc"
        )
        template_text = config_path.read_text(encoding="utf-8")
        _, config_data = read_config_file(config_path)
        self.assertNotIn('"_comment', template_text)
        self.assertEqual(template_text.count("//"), 2)

        runner_args = parse_stock_runner_args(["--config", str(config_path)])
        analysis_args = parse_analysis_args(["--config", str(config_path)])
        common_keys = set(config_data["common"])
        flattened_runner = flatten_runner_options(
            config_data["runner"], config_path
        )
        analysis_keys = common_keys | set(config_data["analysis"]) | {"strategies"}
        for key, value in {**config_data["common"], **flattened_runner}.items():
            self.assertEqual(getattr(runner_args, key), value)
        self.assertEqual(
            set(vars(analysis_args)) - {"config", "results_dir"},
            analysis_keys,
        )
        validate_runner_mask_geometry(
            runner_args,
            resolve_mask_strategies(runner_args),
        )
        self.assertFalse(hasattr(runner_args, "feature_cols"))
        self.assertEqual(analysis_args.strategies, runner_args.mask_strategies)
        self.assertEqual(
            runner_args.mask_strategies,
            ["random", "local_long"],
        )
        nested_runner = config_data["runner"]
        self.assertEqual(
            set(nested_runner["masking"]["strategies"]),
            {"random", "local_long", "future_block", "causal_multiblock"},
        )
        self.assertIn("start_date", nested_runner["download"])
        self.assertIn("robust_zscore", nested_runner["preprocessing"]["custom"]["normalization"])
        self.assertIn(
            "market_data",
            nested_runner["preprocessing"]["custom"]["forecast"],
        )
        self.assertTrue(runner_args.dry_run)

    def test_json_comments_do_not_modify_double_slashes_inside_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "comments.json"
            config_path.write_text(
                "{\n"
                "  // Coverage comment.\n"
                '  "common": {\n'
                '    "stocks": ["NVDA"], // Inline comment.\n'
                '    "seeds": [42]\n'
                "  },\n"
                '  "runner": {\n'
                '    "market_data": "https://example.com/a//b"\n'
                "  }\n"
                "}\n",
                encoding="utf-8",
            )

            args = parse_stock_runner_args(["--config", str(config_path)])

            self.assertEqual(args.market_data, "https://example.com/a//b")

    def test_cli_strategy_override_preserves_common_seed_coverage(self):
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
                "--no-sentiment",
                "--no-skip-download",
            ]
        )

        self.assertEqual(resolve_mask_strategies(args), ["future_block"])
        self.assertEqual(resolve_seeds(args), list(range(42, 52)))
        self.assertFalse(args.use_sentiment)
        self.assertFalse(args.skip_download)

    def test_configured_seed_coverage_can_be_limited_by_runner_options(self):
        repo_root = Path(__file__).resolve().parents[1]
        config_path = (
            repo_root
            / "config"
            / "experiments"
            / "top10_with_sentiment.json"
        )
        args = parse_stock_runner_args(
            ["--config", str(config_path), "--max-seeds", "1"]
        )
        self.assertEqual(args.seeds, list(range(42, 52)))
        self.assertEqual(resolve_seeds(args), [42])

    def test_configured_stock_and_seed_limits_control_current_run_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "limited.json"
            config_path.write_text(
                json.dumps(
                    {
                        "common": {
                            "stocks": ["NVDA", "AAPL", "MSFT"],
                            "seeds": [42, 43, 44],
                        },
                        "runner": {
                            "max_stocks": 2,
                            "max_seeds": 2,
                            "skip_download": True,
                            "skip_combined_plot": True,
                            "dry_run": True,
                        },
                    }
                )
            )

            output = io.StringIO()
            with patch(
                "run_top_nasdaq100_stocks.build_stock_commands",
                side_effect=self._fake_stock_commands(Path(tmp) / "checkpoints"),
            ):
                with redirect_stdout(output):
                    run_stock_main(["--config", str(config_path)])

            self.assertEqual(
                output.getvalue().splitlines(),
                [
                    "stock=NVDA seed=42 status=random:dry-run",
                    "stock=NVDA seed=43 status=random:dry-run",
                    "stock=AAPL seed=42 status=random:dry-run",
                    "stock=AAPL seed=43 status=random:dry-run",
                ],
            )
            args = parse_stock_runner_args(["--config", str(config_path)])
            manifest = build_experiment_manifest(
                args,
                resolve_stocks(args),
                resolve_seeds(args),
                ["random"],
            )
            self.assertEqual(manifest["stocks"], ["NVDA", "AAPL", "MSFT"])
            self.assertEqual(manifest["seeds"], [42, 43, 44])
            self.assertEqual(manifest["run_stocks"], ["NVDA", "AAPL"])
            self.assertEqual(manifest["run_seeds"], [42, 43])

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

    def test_runner_limits_the_ordered_seed_pool(self):
        args = parse_stock_runner_args(
            ["--seeds", "7", "17", "42", "--max-seeds", "2"]
        )
        self.assertEqual(resolve_seeds(args), [7, 17])

    def test_runner_rejects_negative_seed_limit(self):
        args = parse_stock_runner_args(["--max-seeds", "-1"])
        with self.assertRaisesRegex(ValueError, "--max-seeds must be >= 0"):
            resolve_seeds(args)

    def test_verbose_is_default_off_and_does_not_change_experiment_identity(self):
        quiet_args = parse_stock_runner_args([])
        verbose_args = parse_stock_runner_args(["--verbose"])

        self.assertFalse(quiet_args.verbose)
        self.assertTrue(verbose_args.verbose)
        self.assertEqual(
            experiment_config_signature(quiet_args),
            experiment_config_signature(verbose_args),
        )

    def test_run_command_suppresses_child_output_unless_verbose(self):
        with patch("run_top_nasdaq100_stocks.subprocess.run") as process:
            output = io.StringIO()
            with redirect_stdout(output):
                run_command(["child", "command"])

            self.assertEqual(output.getvalue(), "")
            process.assert_called_once_with(
                ["child", "command"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

        output = io.StringIO()
        with redirect_stdout(output):
            run_command(
                ["child", "command"],
                dry_run=True,
                verbose=True,
            )
        self.assertIn("Running: child command", output.getvalue())
        self.assertIn("Dry run: command not executed", output.getvalue())

    def test_quiet_main_reports_only_active_run_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "quiet.json"
            config_path.write_text(
                json.dumps(
                    {
                        "common": {"stocks": ["QUIET"], "seeds": [7]},
                        "runner": {
                            "max_stocks": 0,
                            "max_seeds": 0,
                            "skip_download": True,
                            "skip_pretrain": True,
                            "skip_combined_plot": True,
                        },
                    }
                )
            )

            def finish_downstream(command, **kwargs):
                self.assertFalse(kwargs["verbose"])
                results_index = command.index("--results-dir") + 1
                run_dir = Path(command[results_index])
                run_dir.mkdir(parents=True, exist_ok=True)
                txt_path = run_dir / "last_model_comparison_20260101_000000.txt"
                txt_path.write_text("Data source: QUIET\n")
                txt_path.with_suffix(".csv").write_text(
                    "model,mse,mae,trend_accuracy\nTS-JEPA,0.1,0.2,0.6\n"
                )

            output = io.StringIO()
            with patch(
                "run_top_nasdaq100_stocks.run_command",
                side_effect=finish_downstream,
            ) as mocked_run:
                with redirect_stdout(output):
                    run_stock_main(["--config", str(config_path)])

            self.assertEqual(mocked_run.call_count, 1)
            self.assertEqual(
                output.getvalue().splitlines(),
                [
                    "stock=QUIET seed=7 status=random:evaluating",
                    "stock=QUIET seed=7 status=random:complete",
                ],
            )

    def test_toml_config_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "experiment.toml"
            config_path.write_text(
                "[common]\n"
                'stocks = ["NVDA"]\n'
                "seeds = [3, 5]\n"
                "[runner]\n"
                'mask_strategies = ["random", "local_long"]\n'
                "use_sentiment = false\n"
                "[analysis]\n"
                'models = ["TS-JEPA"]\n'
            )

            runner_args = parse_stock_runner_args(["--config", str(config_path)])
            analysis_args = parse_analysis_args(["--config", str(config_path)])

            self.assertEqual(resolve_seeds(runner_args), [3, 5])
            self.assertFalse(runner_args.use_sentiment)
            self.assertEqual(analysis_args.models, ["TS-JEPA"])
            self.assertEqual(
                analysis_args.strategies,
                runner_args.mask_strategies,
            )
            self.assertEqual(
                Path(runner_args.results_dir),
                Path(tmp) / "results" / "experiment",
            )

    def test_config_rejects_unknown_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "invalid.json"
            config_path.write_text('{"runner": {"mask_stratey": "random"}}')

            with self.assertRaisesRegex(ValueError, "mask_stratey"):
                parse_stock_runner_args(["--config", str(config_path)])

    def test_removed_feature_cols_runner_option_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "legacy.json"
            config_path.write_text(
                '{"runner": {"feature_cols": ["Close", "Volume"]}}'
            )

            with self.assertRaisesRegex(ValueError, "feature_cols"):
                parse_stock_runner_args(["--config", str(config_path)])

    def test_analysis_strategies_cannot_duplicate_runner_strategies(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "duplicate.json"
            config_path.write_text(
                json.dumps(
                    {
                        "runner": {"mask_strategies": ["random"]},
                        "analysis": {"strategies": ["random"]},
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "derived from"):
                parse_analysis_args(["--config", str(config_path)])

    def test_nested_runner_rejects_conditional_conflicts(self):
        def custom_preprocessing(*, normalization=None, sentiment=None, forecast=None):
            return {
                "preset": None,
                "custom": {
                    "feature_transform": "raw",
                    "normalization": normalization or {"method": "window_return"},
                    "features": {
                        "market": ["Close", "Volume"],
                        "sentiment": sentiment
                        or {"enabled": True, "columns": ["sentiment_mean"]},
                    },
                    "forecast": forecast
                    or {
                        "target": "value",
                        "market_data": {
                            "enabled": False,
                            "name": "NASDAQ100",
                        },
                    },
                },
            }

        cases = (
            (
                {"execution": {}, "verbose": False},
                "cannot be mixed",
            ),
            (
                {"masking": {"strategies": {"future_block": {}}}},
                "Missing required options",
            ),
            (
                {
                    "masking": {
                        "strategies": {
                            "random": {
                                "enabled": False,
                            }
                        }
                    }
                },
                "At least one masking strategy",
            ),
            (
                {"preprocessing": {"preset": "P2", "custom": {}}},
                "mutually exclusive",
            ),
            (
                {
                    "preprocessing": custom_preprocessing(
                        forecast={
                            "target": "excess_log_return",
                            "market_data": {
                                "enabled": False,
                                "name": "NASDAQ100",
                            },
                        }
                    )
                },
                "must be enabled",
            ),
            (
                {
                    "preprocessing": custom_preprocessing(
                        forecast={
                            "target": "value",
                            "market_data": {
                                "enabled": True,
                                "name": "NASDAQ100",
                            },
                        }
                    )
                },
                "can only be enabled",
            ),
            (
                {
                    "checkpoint": {
                        "selection": {"mode": "best", "epoch": 2000},
                        "encoder_weights": "ema",
                    }
                },
                "epoch is not allowed",
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "invalid.json"
            for runner, message in cases:
                with self.subTest(message=message):
                    config_path.write_text(json.dumps({"runner": runner}))
                    with self.assertRaisesRegex(ValueError, message):
                        parse_stock_runner_args(["--config", str(config_path)])

    def test_parent_flags_keep_but_do_not_apply_inactive_children(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "experiments"
            / "template_experiment.jsonc"
        )
        _, config_data = read_config_file(config_path)
        runner = config_data["runner"]
        flattened = flatten_runner_options(runner, config_path)

        self.assertIn("start_date", runner["download"])
        self.assertNotIn("download_start_date", flattened)
        self.assertIn("future_block", runner["masking"]["strategies"])
        self.assertNotIn("future_target_patches", flattened)
        normalization = runner["preprocessing"]["custom"]["normalization"]
        self.assertIn("robust_zscore", normalization)
        self.assertNotIn("robust_zscore_clip", flattened)
        forecast = runner["preprocessing"]["custom"]["forecast"]
        self.assertIn("market_data", forecast)
        self.assertFalse(forecast["market_data"]["enabled"])
        self.assertNotIn("market_data", flattened)

        enabled_runner = json.loads(json.dumps(runner))
        enabled_runner["download"]["skip"] = False
        enabled_runner["masking"]["strategies"]["future_block"]["enabled"] = True
        enabled_custom = enabled_runner["preprocessing"]["custom"]
        enabled_custom["normalization"]["method"] = "train_robust_zscore"
        enabled_custom["forecast"]["target"] = "excess_log_return"
        enabled_custom["forecast"]["market_data"]["enabled"] = True
        enabled = flatten_runner_options(enabled_runner, config_path)

        self.assertEqual(enabled["download_start_date"], "2015-01-01")
        self.assertEqual(enabled["future_target_patches"], 4)
        self.assertEqual(enabled["robust_zscore_clip"], 5.0)
        self.assertEqual(enabled["market_data"], "NASDAQ100")

    def test_nested_and_flat_equivalents_share_experiment_identity(self):
        repo_root = Path(__file__).resolve().parents[1]
        nested_path = (
            repo_root / "config" / "experiments" / "top10_with_sentiment.json"
        )
        _, nested_data = read_config_file(nested_path)
        nested_args = parse_stock_runner_args(["--config", str(nested_path)])

        with tempfile.TemporaryDirectory() as tmp:
            flat_path = Path(tmp) / "legacy.json"
            flat_path.write_text(
                json.dumps(
                    {
                        "common": nested_data["common"],
                        "runner": flatten_runner_options(
                            nested_data["runner"], nested_path
                        ),
                    }
                )
            )
            flat_args = parse_stock_runner_args(["--config", str(flat_path)])

        self.assertEqual(
            experiment_config_signature(nested_args),
            experiment_config_signature(flat_args),
        )

        inactive_changes = vars(nested_args).copy()
        inactive_changes.update(
            {
                "download_start_date": "1900-01-01",
                "download_end_date": "1900-01-02",
                "future_target_patches": 99,
                "causal_num_blocks": 99,
                "robust_zscore_clip": 99.0,
            }
        )
        self.assertEqual(
            experiment_config_signature(nested_args),
            experiment_config_signature(inactive_changes),
        )

        active_change = vars(nested_args).copy()
        active_change["mae_window_patches"] += 1
        self.assertNotEqual(
            experiment_config_signature(nested_args),
            experiment_config_signature(active_change),
        )

    def test_stocks_and_seeds_cannot_be_overridden_outside_common(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "invalid.json"
            config_path.write_text(
                json.dumps(
                    {
                        "common": {"stocks": ["NVDA"], "seeds": [42]},
                        "runner": {"stocks": ["AAPL"]},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "only in \[common\]"):
                parse_stock_runner_args(["--config", str(config_path)])

            config_path.write_text(
                json.dumps(
                    {
                        "common": {"stocks": ["NVDA"], "seeds": [42]},
                        "runner": {},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "only from \[common\]"):
                parse_stock_runner_args(
                    ["--config", str(config_path), "--seeds", "7"]
                )
            with self.assertRaisesRegex(ValueError, "cannot be overridden"):
                parse_stock_runner_args(
                    ["--config", str(config_path), "--results-dir", "elsewhere"]
                )

            config_path.write_text(
                json.dumps(
                    {
                        "common": {
                            "stocks": ["NVDA"],
                            "seeds": [42],
                            "results_dir": "results/elsewhere",
                        },
                        "runner": {},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "must not be configured"):
                parse_stock_runner_args(["--config", str(config_path)])

    def test_complete_ten_by_ten_coverage_plans_no_work(self):
        stocks = [f"S{index}" for index in range(10)]
        seeds = list(range(42, 52))
        with tempfile.TemporaryDirectory() as tmp:
            args = self._incremental_args(Path(tmp) / "results", stocks, seeds)
            self._write_experiment_manifest(args, stocks, seeds)
            for stock, seed in requested_stock_seed_runs(stocks, seeds):
                self._write_complete_run(args, stock, seed)

            compatible = validate_existing_experiment(args)
            plan = plan_incremental_execution(
                args,
                stocks,
                seeds,
                ["random"],
                legacy_manifest_compatible=compatible,
            )

        self.assertTrue(compatible)
        self.assertEqual(len(plan["requested_runs"]), 100)
        self.assertEqual(len(plan["completed_runs"]), 100)
        self.assertEqual(plan["missing_runs"], set())
        self.assertEqual(plan["tasks"], [])

    def test_complete_config_main_runs_no_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "experiment.json"
            results_dir = root / "results" / "experiment"
            config_path.write_text(
                json.dumps(
                    {
                        "common": {
                            "stocks": ["NVDA"],
                            "seeds": [42],
                        },
                        "runner": {
                            "max_stocks": 0,
                            "max_seeds": 0,
                            "skip_download": True,
                            "skip_combined_plot": True,
                            "verbose": True,
                        },
                    }
                )
            )
            args = parse_stock_runner_args(["--config", str(config_path)])
            self._write_experiment_manifest(args, ["NVDA"], [42])
            self._write_complete_run(args, "NVDA", 42)
            output = io.StringIO()
            with patch("run_top_nasdaq100_stocks.run_command") as run_command:
                with redirect_stdout(output):
                    run_stock_main(["--config", str(config_path)])

            run_command.assert_not_called()
            self.assertIn("Configuration unchanged.", output.getvalue())
            self.assertIn("Requested coverage: 1 runs.", output.getvalue())
            self.assertIn("Completed compatible runs: 1.", output.getvalue())
            self.assertIn("Missing runs: 0.", output.getvalue())
            self.assertIn("Nothing to run.", output.getvalue())

    def test_adding_one_common_seed_plans_only_that_seed(self):
        stocks = ["NVDA", "AAPL"]
        old_seeds = [42, 43]
        new_seeds = [42, 43, 44]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_args = self._incremental_args(root / "results", stocks, old_seeds)
            self._write_experiment_manifest(old_args, stocks, old_seeds)
            for stock, seed in requested_stock_seed_runs(stocks, old_seeds):
                self._write_complete_run(old_args, stock, seed)
            current_args = self._incremental_args(root / "results", stocks, new_seeds)

            with patch(
                "run_top_nasdaq100_stocks.build_stock_commands",
                side_effect=self._fake_stock_commands(root / "checkpoints"),
            ):
                plan = plan_incremental_execution(
                    current_args,
                    stocks,
                    new_seeds,
                    ["random"],
                    legacy_manifest_compatible=validate_existing_experiment(
                        current_args
                    ),
                )

        self.assertEqual(plan["missing_runs"], {("NVDA", 44), ("AAPL", 44)})
        self.assertEqual(
            {(task["stock"], task["seed"]) for task in plan["tasks"]},
            plan["missing_runs"],
        )

    def test_adding_one_common_stock_plans_only_that_stock(self):
        old_stocks = ["NVDA", "AAPL"]
        current_stocks = ["NVDA", "AAPL", "MSFT"]
        seeds = [42, 43]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_args = self._incremental_args(root / "results", old_stocks, seeds)
            self._write_experiment_manifest(old_args, old_stocks, seeds)
            for stock, seed in requested_stock_seed_runs(old_stocks, seeds):
                self._write_complete_run(old_args, stock, seed)
            current_args = self._incremental_args(
                root / "results", current_stocks, seeds
            )

            with patch(
                "run_top_nasdaq100_stocks.build_stock_commands",
                side_effect=self._fake_stock_commands(root / "checkpoints"),
            ):
                plan = plan_incremental_execution(
                    current_args,
                    current_stocks,
                    seeds,
                    ["random"],
                    legacy_manifest_compatible=validate_existing_experiment(
                        current_args
                    ),
                )

        self.assertEqual(plan["missing_runs"], {("MSFT", 42), ("MSFT", 43)})

    def test_sparse_missing_detection_is_exact_tuple_level(self):
        stocks = ["NVDA", "AAPL", "MSFT"]
        seeds = [42, 43, 44]
        missing = {("NVDA", 43), ("MSFT", 44)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._incremental_args(root / "results", stocks, seeds)
            self._write_experiment_manifest(args, stocks, seeds)
            for stock, seed in requested_stock_seed_runs(stocks, seeds):
                if (stock, seed) not in missing:
                    self._write_complete_run(args, stock, seed)

            with patch(
                "run_top_nasdaq100_stocks.build_stock_commands",
                side_effect=self._fake_stock_commands(root / "checkpoints"),
            ):
                plan = plan_incremental_execution(
                    args,
                    stocks,
                    seeds,
                    ["random"],
                    legacy_manifest_compatible=validate_existing_experiment(args),
                )

        self.assertEqual(plan["missing_runs"], missing)
        self.assertEqual(
            {(task["stock"], task["seed"]) for task in plan["tasks"]},
            missing,
        )

    def test_removing_seed_keeps_historical_result_outside_current_coverage(self):
        stocks = ["NVDA"]
        old_seeds = [42, 43, 44]
        current_seeds = [42, 43]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_args = self._incremental_args(root / "results", stocks, old_seeds)
            self._write_experiment_manifest(old_args, stocks, old_seeds)
            for stock, seed in requested_stock_seed_runs(stocks, old_seeds):
                self._write_complete_run(old_args, stock, seed)
            historical_dir = stock_result_dir(old_args, "random", "NVDA", 44)
            current_args = self._incremental_args(
                root / "results", stocks, current_seeds
            )
            plan = plan_incremental_execution(
                current_args,
                stocks,
                current_seeds,
                ["random"],
                legacy_manifest_compatible=validate_existing_experiment(current_args),
            )

            self.assertTrue(historical_dir.is_dir())
            self.assertEqual(plan["tasks"], [])
            self.assertEqual(set(plan["requested_runs"]), {("NVDA", 42), ("NVDA", 43)})
            aggregate_command = build_combined_plot_command(
                current_args,
                stocks,
                results_dir=Path(tmp) / "results" / "random",
            )
            seed_index = aggregate_command.index("--seeds")
            stock_index = aggregate_command.index("--stocks")
            self.assertEqual(
                aggregate_command[seed_index + 1 : stock_index],
                ["42", "43"],
            )

    def test_result_affecting_experiment_change_is_incompatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_args = self._incremental_args(root / "results", ["NVDA"], [42])
            self._write_experiment_manifest(old_args, ["NVDA"], [42])
            self._write_complete_run(old_args, "NVDA", 42)
            current_args = self._incremental_args(
                root / "results",
                ["NVDA"],
                [42],
                "--lambda-mae",
                "0.75",
            )

            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                validate_existing_experiment(current_args)

    def test_renamed_config_is_rejected_when_runner_identity_already_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_config = root / "original.json"
            renamed_config = root / "renamed.json"
            changed_config = root / "changed.json"

            def write_config(path, stocks, seeds, lambda_mae=0.5):
                path.write_text(
                    json.dumps(
                        {
                            "common": {"stocks": stocks, "seeds": seeds},
                            "runner": {
                                "max_stocks": 0,
                                "max_seeds": 0,
                                "skip_download": True,
                                "skip_combined_plot": True,
                                "lambda_mae": lambda_mae,
                            },
                        }
                    )
                )

            write_config(old_config, ["NVDA"], [42])
            old_args = parse_stock_runner_args(["--config", str(old_config)])
            self._write_experiment_manifest(old_args, ["NVDA"], [42])
            self._write_complete_run(old_args, "NVDA", 42)

            # A different filename and expanded coverage are still the same
            # experiment because runner identity is unchanged.
            write_config(renamed_config, ["NVDA", "AAPL"], [42, 43])
            renamed_args = parse_stock_runner_args(
                ["--config", str(renamed_config)]
            )
            self.assertEqual(
                compatible_result_directories(renamed_args),
                [Path(old_args.results_dir).resolve()],
            )
            with self.assertRaisesRegex(RuntimeError, "Renaming the config"):
                reject_duplicate_experiment_config(renamed_args)
            with patch("run_top_nasdaq100_stocks.run_command") as run_command:
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    run_stock_main(["--config", str(renamed_config)])
                run_command.assert_not_called()

            # A true runner change has a different identity and may use its own
            # filename-derived result folder.
            write_config(changed_config, ["NVDA"], [42], lambda_mae=0.75)
            changed_args = parse_stock_runner_args(
                ["--config", str(changed_config)]
            )
            self.assertEqual(compatible_result_directories(changed_args), [])
            reject_duplicate_experiment_config(changed_args)

    def test_noncoverage_common_option_changes_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "experiment.json"

            def parse_common(stocks, seeds, normalization):
                config_path.write_text(
                    json.dumps(
                        {
                            "common": {
                                "stocks": stocks,
                                "seeds": seeds,
                                "normalization": normalization,
                            },
                            "runner": {"max_stocks": 0, "max_seeds": 0},
                        }
                    )
                )
                return parse_stock_runner_args(["--config", str(config_path)])

            base = parse_common(["NVDA"], [42], "window_return")
            coverage_only = parse_common(
                ["NVDA", "AAPL"], [42, 43], "window_return"
            )
            changed = parse_common(
                ["NVDA", "AAPL"], [42, 43], "train_zscore"
            )

            self.assertEqual(
                experiment_config_signature(base),
                experiment_config_signature(coverage_only),
            )
            self.assertNotEqual(
                experiment_config_signature(base),
                experiment_config_signature(changed),
            )
            self.assertNotIn("stocks", effective_experiment_config(base))
            self.assertNotIn("seeds", effective_experiment_config(base))
            self.assertIn("normalization", effective_experiment_config(base))

            self._write_experiment_manifest(base, ["NVDA"], [42])
            self._write_complete_run(base, "NVDA", 42)
            self.assertTrue(validate_existing_experiment(coverage_only))
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                validate_existing_experiment(changed)

    def test_legacy_null_feature_cols_manifest_remains_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._incremental_args(
                Path(tmp) / "results" / "experiment",
                ["NVDA"],
                [42],
            )
            manifest = build_experiment_manifest(args, ["NVDA"], [42], ["random"])
            manifest["effective_config"]["feature_cols"] = None
            manifest_path = Path(args.results_dir) / "experiment_manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(json.dumps(manifest))

            self.assertTrue(validate_existing_experiment(args))

    def test_existing_pretraining_is_reused_when_downstream_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._incremental_args(root / "results", ["NVDA"], [42])
            checkpoint_path = root / "checkpoints" / "random" / "NVDA" / "seed_42.pt"
            checkpoint_path.parent.mkdir(parents=True)
            checkpoint_path.write_bytes(b"existing checkpoint")

            with patch(
                "run_top_nasdaq100_stocks.build_stock_commands",
                side_effect=self._fake_stock_commands(root / "checkpoints"),
            ):
                plan = plan_incremental_execution(
                    args,
                    ["NVDA"],
                    [42],
                    ["random"],
                    legacy_manifest_compatible=False,
                )

        self.assertEqual(len(plan["tasks"]), 1)
        self.assertIsNone(plan["tasks"][0]["pretrain_command"])
        self.assertIn("eval_dual_loss.py", plan["tasks"][0]["eval_command"])

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
