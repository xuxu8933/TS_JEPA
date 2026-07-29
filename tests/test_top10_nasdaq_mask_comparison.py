import tempfile
import unittest
from pathlib import Path

from run_top10_nasdaq_mask_comparison import (
    aggregate_metrics,
    aggregate_strategy_runs,
    build_strategy_command,
    collect_raw_results,
    parse_args,
    resolve_seeds,
)


class Top10NasdaqMaskComparisonTest(unittest.TestCase):
    def test_defaults_build_ten_reproducible_runs_for_both_strategies(self):
        args = parse_args(["--dry-run"])
        seeds = resolve_seeds(args)
        stocks = list(args.stocks)

        self.assertEqual(seeds, list(range(42, 52)))
        for strategy in ("random", "local_long"):
            command = build_strategy_command(args, strategy, stocks, seeds)
            self.assertEqual(
                command[command.index("--mask-strategy") + 1],
                strategy,
            )
            seed_index = command.index("--seeds")
            results_index = command.index("--results-dir")
            self.assertEqual(
                command[seed_index + 1 : results_index],
                [str(seed) for seed in seeds],
            )
            self.assertIn(f"/{strategy}", command[results_index + 1])

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
        import pandas as pd

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


if __name__ == "__main__":
    unittest.main()
