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
    compute_direction_accuracy,
    exact_wilcoxon,
    load_scope,
    parse_args,
    run_analysis,
)
from eval_forecast_prequential_with_baselines_gru_volume import (
    _maybe_get_dataset_date,
    _maybe_get_dataset_index,
)


class ThesisStatisticsTest(unittest.TestCase):
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
            output_dir = root / "analysis_artifacts"
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
                    "--output-dir",
                    str(output_dir),
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
            self.assertTrue((output_dir / "tables" / "table_main_metrics.tex").exists())
            self.assertTrue((output_dir / "figures" / "fig_paired_mse_forest.pdf").exists())
            self.assertTrue((output_dir / "figures" / "fig_mse_by_horizon.pdf").exists())
            self.assertTrue((output_dir / "figures" / "fig_representative_prediction_trajectory.pdf").exists())
            self.assertIn(
                "equities as the statistical units",
                (output_dir / "README.md").read_text(encoding="utf-8"),
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
                    "--output-dir",
                    str(output_dir),
                    "--bootstrap-samples",
                    "20",
                    "--skip-figures",
                ]
            )
            with self.assertRaisesRegex(RuntimeError, "validity error"):
                run_analysis(strict_args)
            self.assertFalse(
                (output_dir / "tables" / "table_main_metrics.tex").exists()
            )
            self.assertTrue((output_dir / "data" / "missing_runs.csv").exists())


if __name__ == "__main__":
    unittest.main()
