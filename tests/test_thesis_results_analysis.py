import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.thesis_results import (
    METHOD_ORDER,
    _student_t_cdf,
    build_shared_vs_local_comparison,
    compute_direction_accuracy,
    exact_wilcoxon,
    load_scope,
    paired_difference_statistics,
    parse_args,
    run_analysis,
    write_shared_vs_local_table,
)
from eval_forecast_prequential_with_baselines_gru_volume import (
    _maybe_get_dataset_date,
    _maybe_get_dataset_index,
)


class ThesisStatisticsTest(unittest.TestCase):
    @staticmethod
    def _strategy_row(
        stock,
        seed,
        method,
        mse,
        mae,
        *,
        test_signature="common-targets",
        **metadata,
    ):
        return {
            "stock": stock,
            "seed": seed,
            "method": method,
            "strategy": "random" if method == METHOD_ORDER[0] else "local_long",
            "split": "test",
            "mse": mse,
            "mae": mae,
            "forecast_horizon": 5,
            "forecast_target": "value",
            "target_definition": "future-close",
            "normalization": "window_return",
            "metric_definition": "rolling-origin-v1",
            "test_start": "2025-01-01",
            "test_end": "2025-12-31",
            "test_signature": test_signature,
            "original_source_file": f"{stock}/{seed}/{method}.csv",
            **metadata,
        }

    def test_thesis_scope_accepts_commented_experiment_template(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "experiments"
            / "template_experiment.jsonc"
        )
        args = parse_args(["--config", str(config_path)])

        scope, results_dir = load_scope(args)

        self.assertEqual(scope["stocks"], ["NVDA", "AAPL"])
        self.assertEqual(scope["seeds"], [42, 43])
        self.assertEqual(results_dir.name, "template_experiment")

    def test_exact_signed_rank_and_effect_direction(self):
        statistic, p_value, effect, count = exact_wilcoxon([-1.0, -2.0, -3.0])
        self.assertEqual(statistic, 0.0)
        self.assertEqual(p_value, 0.25)
        self.assertEqual(effect, -1.0)
        self.assertEqual(count, 3)

    def test_paired_statistics_use_signed_sample_standard_deviation(self):
        statistics = paired_difference_statistics([-1.0, -2.0, -3.0, -4.0, -5.0])

        self.assertEqual(statistics["n_stocks"], 5)
        self.assertAlmostEqual(statistics["mean_delta"], -3.0)
        self.assertAlmostEqual(statistics["median_delta"], -3.0)
        self.assertAlmostEqual(statistics["cohens_dz"], -3.0 / np.sqrt(2.5))
        self.assertAlmostEqual(statistics["t_statistic"], -3.0 / np.sqrt(2.5 / 5.0))
        self.assertAlmostEqual(statistics["p_value"], 0.0132355995636827)
        self.assertEqual(statistics["status"], "ok")

    def test_student_t_probabilities_cover_low_and_high_degrees_of_freedom(self):
        self.assertEqual(_student_t_cdf(0.0, 1), 0.5)
        self.assertAlmostEqual(2.0 * (1.0 - _student_t_cdf(1.0, 1)), 0.5)
        self.assertAlmostEqual(
            2.0 * (1.0 - _student_t_cdf(np.sqrt(3.0), 2)),
            0.2254033307585166,
        )
        self.assertAlmostEqual(
            2.0 * (1.0 - _student_t_cdf(2.0, 30)),
            0.0546250449629831,
        )

    def test_paired_statistics_report_insufficient_and_zero_variance_samples(self):
        insufficient = paired_difference_statistics([-0.25])
        constant = paired_difference_statistics([-0.25, -0.25, -0.25])
        roundoff_only = paired_difference_statistics(
            [-0.00012000000000000010, -0.00012000000000000011]
        )

        self.assertEqual(insufficient["status"], "insufficient_stock_observations")
        self.assertTrue(np.isnan(insufficient["t_statistic"]))
        self.assertTrue(np.isnan(insufficient["p_value"]))
        self.assertTrue(np.isnan(insufficient["cohens_dz"]))
        self.assertEqual(constant["status"], "zero_variance_differences")
        self.assertTrue(np.isnan(constant["t_statistic"]))
        self.assertTrue(np.isnan(constant["p_value"]))
        self.assertTrue(np.isnan(constant["cohens_dz"]))
        self.assertEqual(roundoff_only["status"], "zero_variance_differences")
        self.assertTrue(np.isnan(roundoff_only["t_statistic"]))

    def test_shared_local_latex_marks_unavailable_statistics_explicitly(self):
        summary = pd.DataFrame(
            [
                {
                    "metric": "MSE",
                    "n_stocks": 1,
                    "mean_delta": -0.25,
                    "median_delta": -0.25,
                    "t_statistic": np.nan,
                    "p_value": np.nan,
                    "cohens_dz": np.nan,
                    "status": "insufficient_stock_observations",
                },
                {
                    "metric": "MAE",
                    "n_stocks": 0,
                    "mean_delta": np.nan,
                    "median_delta": np.nan,
                    "t_statistic": np.nan,
                    "p_value": np.nan,
                    "cohens_dz": np.nan,
                    "status": "insufficient_stock_observations",
                },
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_shared_vs_local_table(summary, Path(temporary))
            rendered = paths[1].read_text(encoding="utf-8")

        self.assertNotIn("nan", rendered.lower())
        self.assertIn("--", rendered)
        self.assertIn("insufficient stocks", rendered)

    def test_shared_local_pairing_uses_configured_matched_seeds_and_stock_units(self):
        shared, local = METHOD_ORDER[:2]
        rows = [
            self._strategy_row("AAA", 7, shared, 1.0, 3.0),
            self._strategy_row("AAA", 7, local, 2.0, 4.0),
            self._strategy_row("AAA", 9, shared, 3.0, 6.0),
            self._strategy_row("AAA", 9, local, 5.0, 8.0),
            self._strategy_row("AAA", 11, shared, 100.0, 100.0),
            self._strategy_row("BBB", 7, shared, 4.0, 4.0),
            self._strategy_row("BBB", 7, local, 2.0, 2.0),
            self._strategy_row("BBB", 9, shared, 1.0, 1.0, test_signature="left"),
            self._strategy_row("BBB", 9, local, 9.0, 9.0, test_signature="right"),
            self._strategy_row("CCC", 7, shared, 1.0, 1.0),
            self._strategy_row("OUT", 7, shared, -100.0, -100.0),
            self._strategy_row("OUT", 7, local, 100.0, 100.0),
        ]
        scope = {
            "stocks": ["AAA", "BBB", "CCC"],
            "seeds": [7, 9, 11],
            "strategies": ["random", "local_long"],
        }
        issues = []

        stock_pairs, summary = build_shared_vs_local_comparison(
            pd.DataFrame(rows), scope, issues
        )

        self.assertEqual(stock_pairs["stock"].tolist(), ["AAA", "BBB"])
        aaa = stock_pairs.set_index("stock").loc["AAA"]
        self.assertEqual(aaa["n_matched_seeds"], 2)
        self.assertEqual(aaa["matched_seeds"], "7;9")
        self.assertAlmostEqual(aaa["shared_mse"], 2.0)
        self.assertAlmostEqual(aaa["local_mse"], 3.5)
        self.assertAlmostEqual(aaa["delta_mse"], -1.5)
        self.assertAlmostEqual(aaa["delta_mae"], -1.5)
        bbb = stock_pairs.set_index("stock").loc["BBB"]
        self.assertEqual(bbb["n_matched_seeds"], 1)
        self.assertEqual(bbb["matched_seeds"], "7")
        self.assertAlmostEqual(bbb["delta_mse"], 2.0)
        self.assertEqual(set(summary["metric"]), {"MSE", "MAE"})
        self.assertTrue((summary["n_stocks"] == 2).all())
        self.assertAlmostEqual(
            summary.set_index("metric").loc["MSE", "mean_delta"], 0.25
        )
        self.assertIn("unmatched_shared_local_seed", {row["status"] for row in issues})
        self.assertIn("incompatible_shared_local_pair", {row["status"] for row in issues})
        self.assertIn("shared_local_stock_excluded", {row["status"] for row in issues})

    def test_shared_local_pairing_rejects_different_saved_experiment_identities(self):
        shared, local = METHOD_ORDER[:2]
        rows = [
            self._strategy_row(
                "AAA", 7, shared, 1.0, 1.0,
                experiment_id="experiment-a", config_signature="config-a",
            ),
            self._strategy_row(
                "AAA", 7, local, 2.0, 2.0,
                experiment_id="experiment-a", config_signature="config-a",
            ),
            self._strategy_row(
                "AAA", 9, shared, 3.0, 3.0,
                experiment_id="experiment-a", config_signature="config-a",
            ),
            self._strategy_row(
                "AAA", 9, local, 9.0, 9.0,
                experiment_id="experiment-b", config_signature="config-b",
            ),
        ]
        scope = {"stocks": ["AAA"], "seeds": [7, 9]}
        issues = []

        stock_pairs, summary = build_shared_vs_local_comparison(
            pd.DataFrame(rows), scope, issues
        )

        row = stock_pairs.iloc[0]
        self.assertEqual(row["matched_seeds"], "7")
        self.assertEqual(row["n_matched_seeds"], 1)
        self.assertAlmostEqual(row["delta_mse"], -1.0)
        self.assertTrue((summary["n_stocks"] == 1).all())
        incompatibility = next(
            item for item in issues if item["status"] == "incompatible_shared_local_pair"
        )
        self.assertIn("experiment_id", incompatibility["details"])
        self.assertIn("config_signature", incompatibility["details"])

    def test_published_top10_shared_local_statistics_are_reproduced_dynamically(self):
        repository = Path(__file__).resolve().parents[1]
        published = (
            repository
            / "thesis_results"
            / "top10_with_sentiment"
            / "5b8f3897bf23-02add88f32d5"
            / "data"
            / "all_runs_tidy.csv"
        )
        if not published.is_file():
            self.skipTest("published top10_with_sentiment canonical data are unavailable")
        args = parse_args(
            ["--config", str(repository / "config/experiments/top10_with_sentiment.json")]
        )
        scope, _ = load_scope(args)

        stock_pairs, summary = build_shared_vs_local_comparison(
            pd.read_csv(published), scope, []
        )

        self.assertEqual(len(stock_pairs), len(scope["stocks"]))
        self.assertTrue((stock_pairs["n_matched_seeds"] == len(scope["seeds"])).all())
        indexed = summary.set_index("metric")
        self.assertEqual(indexed.loc["MSE", "n_stocks"], len(scope["stocks"]))
        self.assertAlmostEqual(indexed.loc["MSE", "cohens_dz"], -0.43, delta=0.02)
        self.assertAlmostEqual(indexed.loc["MSE", "p_value"], 0.21, delta=0.02)
        self.assertAlmostEqual(indexed.loc["MAE", "cohens_dz"], -0.61, delta=0.02)
        self.assertAlmostEqual(indexed.loc["MAE", "p_value"], 0.09, delta=0.02)

    def test_direction_accuracy_uses_same_trajectory_rule_for_baseline(self):
        targets = np.array([[0.1, 0.2, 0.15], [-0.1, -0.2, -0.1]])
        matching = targets.copy()
        constant = np.zeros_like(targets)
        self.assertEqual(compute_direction_accuracy(matching, targets, "value"), 1.0)
        self.assertEqual(compute_direction_accuracy(constant, targets, "value"), 0.0)
        self.assertEqual(
            compute_direction_accuracy(matching, targets, "relative_return"),
            1.0,
        )

    def test_saved_target_index_and_date_include_context_offset(self):
        class Dataset:
            context_size = 2
            patch_size = 3
            sample_starts = [4]
            dates = pd.date_range("2025-01-01", periods=20)

        dataset = Dataset()
        self.assertEqual(_maybe_get_dataset_index(dataset, 0, 1, 2), 12)
        self.assertEqual(
            _maybe_get_dataset_date(dataset, 0, 2),
            "2025-01-13T00:00:00",
        )


class ThesisPipelineIntegrationTest(unittest.TestCase):
    @staticmethod
    def _score_frame(model, predictions, targets):
        rows = []
        for rolling_step in range(targets.shape[0]):
            for horizon_index in range(targets.shape[1]):
                predicted = float(predictions[rolling_step, horizon_index])
                true = float(targets[rolling_step, horizon_index])
                error = predicted - true
                rows.append(
                    {
                        "model": model,
                        "forecast_target": "value",
                        "rolling_step": rolling_step,
                        "horizon_step": horizon_index + 1,
                        "target_index": rolling_step + horizon_index,
                        "target_date": f"2025-02-{horizon_index + 3:02d}",
                        "predicted_value": predicted,
                        "true_value": true,
                        "error": error,
                        "absolute_error": abs(error),
                        "squared_error": error**2,
                    }
                )
        return pd.DataFrame(rows)

    def _write_bundle(self, root, strategy, stock, seed, stock_offset):
        run_dir = root / strategy / stock / f"seed_{seed}"
        run_dir.mkdir(parents=True)
        timestamp = "20260101_010101"
        targets = np.array(
            [
                [0.10 + stock_offset, 0.20 + stock_offset, 0.16 + stock_offset],
                [0.05 + stock_offset, 0.12 + stock_offset, 0.18 + stock_offset],
            ]
        )
        naive = np.array(
            [
                [0.08 + stock_offset] * 3,
                [0.04 + stock_offset] * 3,
            ]
        )
        drift = naive + np.array([[0.01, 0.02, 0.03], [0.01, 0.02, 0.03]])
        mean_context = np.array([[stock_offset] * 3, [stock_offset] * 3])
        seed_shift = (seed - 1) * 0.002
        strategy_shift = 0.006 if strategy == "random" else 0.012
        tsjepa = targets + strategy_shift + seed_shift
        gru = targets + 0.018 + seed_shift
        frames = {
            "TS-JEPA": self._score_frame("TS-JEPA", tsjepa, targets),
            "GRU": self._score_frame("GRU", gru, targets),
            "naive_last": self._score_frame("naive_last", naive, targets),
            "drift": self._score_frame("drift", drift, targets),
            "mean_context": self._score_frame("mean_context", mean_context, targets),
        }
        frames["TS-JEPA"].drop(columns="model").to_csv(
            run_dir / f"last_scores_after_observation_{timestamp}.csv", index=False
        )
        frames["GRU"].drop(columns="model").to_csv(
            run_dir / f"last_gru_scores_after_observation_{timestamp}.csv", index=False
        )
        pd.concat(
            [frames[name] for name in ("naive_last", "drift", "mean_context")],
            ignore_index=True,
        ).to_csv(
            run_dir / f"last_baseline_scores_after_observation_{timestamp}.csv",
            index=False,
        )
        comparison_rows = []
        for model, frame in frames.items():
            predictions = frame.pivot(
                index="rolling_step", columns="horizon_step", values="predicted_value"
            ).to_numpy()
            true_values = frame.pivot(
                index="rolling_step", columns="horizon_step", values="true_value"
            ).to_numpy()
            comparison_rows.append(
                {
                    "model": model,
                    "forecast_target": "value",
                    "mse": float(np.mean((predictions - true_values) ** 2)),
                    "mae": float(np.mean(np.abs(predictions - true_values))),
                    "trend_accuracy": compute_direction_accuracy(
                        predictions, true_values, "value"
                    ),
                }
            )
        pd.DataFrame(comparison_rows).to_csv(
            run_dir / f"last_model_comparison_{timestamp}.csv", index=False
        )
        (run_dir / f"last_model_comparison_{timestamp}.txt").write_text(
            f"Data source: {stock}\n"
            "Evaluation type: last\n"
            "Forecast target: value\n"
            "Feature transform: raw\n"
            "Features: Close,Volume,MA10,MA50,sentiment_mean\n"
            "Normalization: window_return\n"
            f"Generated at: {timestamp}\n",
            encoding="utf-8",
        )
        (run_dir / "preprocessing_config.json").write_text(
            json.dumps(
                {
                    "forecast_target": "value",
                    "feature_transform": "raw",
                    "feature_names": [
                        "Close",
                        "Volume",
                        "MA10",
                        "MA50",
                        "sentiment_mean",
                    ],
                    "normalization": "window_return",
                    "forecast_horizons": [1, 2, 3],
                    "patch_size": 3,
                    "window_length": 60,
                    "test_start": "2025-01-01",
                    "test_end": "2025-12-31",
                }
            ),
            encoding="utf-8",
        )
        history = pd.DataFrame(
            {
                "epoch": [0, 1, 2],
                "train_loss": [0.3, 0.2, 0.1],
                "mse_loss": [0.3, 0.2, 0.1],
                "trend_loss": [0.0, 0.0, 0.0],
                "val_mse": [0.4, 0.3, 0.2],
                "val_mae": [0.5, 0.4, 0.3],
                "val_trend_acc": [0.5, 0.6, 0.7],
            }
        )
        history.to_csv(run_dir / "loss.txt", index=False)

    def test_complete_pipeline_generates_traceable_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results_dir = root / "results" / "experiment"
            output_dir = root / "analysis_artifacts" / "experiment"
            stocks = ["AAA", "BBB"]
            seeds = [1, 2]
            for strategy in ("random", "local_long"):
                for stock_index, stock in enumerate(stocks):
                    for seed in seeds:
                        self._write_bundle(
                            results_dir,
                            strategy,
                            stock,
                            seed,
                            stock_offset=stock_index * 0.02,
                        )
            manifest = {
                "arguments": {
                    "patch_size": 3,
                    "context_size": 20,
                    "eval_stride": 1,
                    "checkpoint_to_use": 2000,
                    "encoder_weights": "ema",
                    "fine_tune_encoder": True,
                    "trend_weight": 0.001,
                },
                "stocks": stocks,
                "seeds": seeds,
                "mask_strategies": ["random", "local_long"],
                "normalization": "window_return",
                "forecast_target": "value",
            }
            (results_dir / "experiment_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            config_path = root / "experiment.json"
            config_path.write_text(
                json.dumps(
                    {
                        "common": {
                            "stocks": stocks,
                            "seeds": seeds,
                        },
                        "runner": {
                            "mask_strategies": ["random", "local_long"],
                            "patch_size": 3,
                            "context_size": 20,
                            "normalization": "window_return",
                            "forecast_target": "value",
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--config",
                    str(config_path),
                    "--bootstrap-samples",
                    "200",
                ]
            )
            self.assertEqual(run_analysis(args), 0)
            tidy = pd.read_csv(output_dir / "data" / "all_runs_tidy.csv")
            self.assertEqual(len(tidy), len(stocks) * len(seeds) * len(METHOD_ORDER))
            issues = pd.read_csv(output_dir / "data" / "missing_runs.csv")
            self.assertFalse((issues["severity"] == "error").any())
            paired = pd.read_csv(output_dir / "data" / "paired_vs_naive.csv")
            self.assertEqual(set(paired["model"]), set(METHOD_ORDER[:3]))
            shared_naive = paired.set_index("model").loc[METHOD_ORDER[0]]
            self.assertAlmostEqual(shared_naive["mean_delta_mse"], -0.0078333333333333)
            self.assertAlmostEqual(shared_naive["mean_delta_mae"], -0.068)
            for metric, mean_delta in (
                ("mse", -0.0078333333333333),
                ("mae", -0.068),
            ):
                self.assertAlmostEqual(shared_naive[f"{metric}_ci_low"], mean_delta)
                self.assertAlmostEqual(shared_naive[f"{metric}_ci_high"], mean_delta)
                self.assertEqual(shared_naive[f"{metric}_wilcoxon_statistic"], 0.0)
                self.assertEqual(shared_naive[f"{metric}_p_value"], 0.5)
                self.assertEqual(shared_naive[f"{metric}_holm_p_value"], 1.0)
                self.assertEqual(shared_naive[f"{metric}_rank_biserial"], -1.0)
                self.assertEqual(shared_naive[f"{metric}_wilcoxon_n"], len(stocks))
            shared_local = pd.read_csv(
                output_dir / "data" / "paired_shared_vs_local.csv"
            )
            self.assertEqual(len(shared_local), len(stocks))
            self.assertTrue((shared_local["n_matched_seeds"] == len(seeds)).all())
            shared_local_table = pd.read_csv(
                output_dir / "tables" / "table_shared_vs_local.csv"
            )
            self.assertEqual(set(shared_local_table["metric"]), {"MSE", "MAE"})
            self.assertTrue((shared_local_table["n_stocks"] == len(stocks)).all())
            self.assertTrue((output_dir / "tables" / "table_main_metrics.tex").exists())
            self.assertTrue((output_dir / "tables" / "table_shared_vs_local.tex").exists())
            self.assertTrue((output_dir / "figures" / "fig_paired_mse_forest.pdf").exists())
            self.assertTrue((output_dir / "figures" / "fig_mse_by_horizon.pdf").exists())
            self.assertTrue((output_dir / "figures" / "fig_representative_prediction_trajectory.pdf").exists())
            self.assertIn(
                "equities as the statistical units",
                (output_dir / "README.md").read_text(encoding="utf-8"),
            )
            artifact_manifest = pd.read_csv(output_dir / "artifact_manifest.csv")
            self.assertEqual(
                set(
                    artifact_manifest.loc[
                        artifact_manifest["artifact"].str.contains("shared_vs_local"),
                        "artifact",
                    ]
                ),
                {
                    "data/paired_shared_vs_local.csv",
                    "tables/table_shared_vs_local.csv",
                    "tables/table_shared_vs_local.tex",
                },
            )
            if shutil.which("pdflatex"):
                compile_dir = root / "latex_check"
                compile_dir.mkdir()
                document = compile_dir / "tables.tex"
                document.write_text(
                    "\\documentclass{article}\n"
                    "\\usepackage[margin=1in]{geometry}\n"
                    "\\usepackage{booktabs,longtable}\n"
                    "\\begin{document}\n"
                    f"\\input{{{output_dir / 'tables' / 'table_main_metrics.tex'}}}\n"
                    f"\\input{{{output_dir / 'tables' / 'table_paired_vs_naive.tex'}}}\n"
                    f"\\input{{{output_dir / 'tables' / 'table_shared_vs_local.tex'}}}\n"
                    f"\\input{{{output_dir / 'tables' / 'table_appendix_stock_metrics.tex'}}}\n"
                    f"\\input{{{output_dir / 'tables' / 'table_reproducibility.tex'}}}\n"
                    "\\end{document}\n",
                    encoding="utf-8",
                )
                completed = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", document.name],
                    cwd=compile_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout)
                self.assertNotIn("Overfull \\hbox", completed.stdout)

            missing_config = root / "missing_experiment.json"
            missing_config.write_text(
                json.dumps(
                    {
                        "common": {
                            "stocks": ["AAA"],
                            "seeds": [1],
                        },
                        "runner": {
                            "mask_strategies": ["random", "local_long"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            strict_args = parse_args(
                [
                    "--config",
                    str(missing_config),
                    "--bootstrap-samples",
                    "20",
                    "--skip-figures",
                ]
            )
            with self.assertRaisesRegex(RuntimeError, "validity error"):
                run_analysis(strict_args)
            missing_output_dir = (
                root / "analysis_artifacts" / "missing_experiment"
            )
            self.assertFalse(
                (missing_output_dir / "tables" / "table_main_metrics.tex").exists()
            )
            self.assertTrue(
                (missing_output_dir / "data" / "missing_runs.csv").exists()
            )


if __name__ == "__main__":
    unittest.main()
