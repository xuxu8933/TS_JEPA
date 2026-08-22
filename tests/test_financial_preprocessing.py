import copy
import csv
import math
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pretrain_dual_loss import save_checkpoint
from eval_forecast_prequential_with_baselines_gru_volume import (
    compute_trend_accuracy,
    make_baseline_prediction,
)
from src.data_loaders.data_class_roll_volume import (
    CSVDataLoader,
    EvaluationDataLoader,
    normalization_tensors,
    train_robust_zscore_state,
)
from src.data_loaders.financial_preprocessing import (
    RETURN_FEATURE_COLS,
    align_market_data,
    construct_return_features,
)


def _write_price_csv(path, closes, missing_indices=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = set(missing_indices)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Date", "Close", "Volume"])
        writer.writeheader()
        start = date(2020, 1, 1)
        for index, close in enumerate(closes):
            if index in missing:
                continue
            writer.writerow(
                {
                    "Date": (start + timedelta(days=index)).isoformat(),
                    "Close": float(close),
                    "Volume": 1000.0 + index,
                }
            )


class FinancialPreprocessingTest(unittest.TestCase):
    def test_log_return_calculation(self):
        frame = pd.DataFrame(
            {"Close": [100.0, 101.0, 103.0], "Volume": [10.0, 11.0, 12.0]}
        )
        transformed = construct_return_features(frame)
        self.assertAlmostEqual(
            transformed.loc[1, "log_return_1"], math.log(101.0 / 100.0)
        )
        self.assertAlmostEqual(
            transformed.loc[2, "log_return_1"], math.log(103.0 / 101.0)
        )

    def test_future_perturbation_does_not_change_earlier_features(self):
        close = np.exp(np.linspace(math.log(100.0), math.log(150.0), 90))
        original = pd.DataFrame(
            {"Close": close, "Volume": np.arange(1000.0, 1090.0)}
        )
        perturbed = original.copy()
        perturbed.loc[70:, "Close"] *= 20.0
        perturbed.loc[70:, "Volume"] *= 50.0
        before = construct_return_features(original)
        after = construct_return_features(perturbed)
        pd.testing.assert_frame_equal(
            before.loc[:69, list(RETURN_FEATURE_COLS)],
            after.loc[:69, list(RETURN_FEATURE_COLS)],
        )

    def test_robust_normalization_is_train_only_and_handles_zero_mad(self):
        values = torch.tensor(
            [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [1000.0, 5.0]],
            dtype=torch.float32,
        )
        state = train_robust_zscore_state(values, ["normal", "constant"])
        center, scale = normalization_tensors(state, ["normal", "constant"])
        self.assertTrue(torch.isfinite((values - center) / scale).all())
        self.assertEqual(state["scale"][1], 1.0)
        self.assertLess(state["median"][0], 10.0)

        with tempfile.TemporaryDirectory() as tmp:
            first_path = Path(tmp) / "first.csv"
            second_path = Path(tmp) / "second.csv"
            closes = np.linspace(100.0, 220.0, 180)
            changed = closes.copy()
            # After the 49-row warm-up, validation begins at source index 136
            # and test begins at 145. Perturb both while leaving train intact.
            changed[136:] *= 100.0
            _write_price_csv(first_path, closes)
            _write_price_csv(second_path, changed)
            kwargs = dict(
                series_split_size=20,
                patch_size=5,
                normalization="train_robust_zscore",
                feature_transform="return",
                feature_cols=(),
                validation_fraction=0.1,
                test_start_date="2020-05-25",
            )
            first = CSVDataLoader(path_data=str(first_path), **kwargs)
            second = CSVDataLoader(path_data=str(second_path), **kwargs)
            self.assertEqual(first.normalization_stats, second.normalization_stats)

    def test_cumulative_log_target_uses_forecast_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prices.csv"
            closes = 100.0 * np.exp(np.arange(180) * 0.01)
            _write_price_csv(path, closes)
            dataset = EvaluationDataLoader(
                path_data=str(path),
                patch_size=5,
                context_size=12,
                stride=5,
                split="train",
                normalization="none",
                feature_transform="return",
                feature_cols=(),
                target_col="Close",
                forecast_target="cumulative_log_return",
                validation_fraction=0.1,
                test_start_date="2020-05-25",
            )
            _, target = dataset[0]
            start = dataset.sample_starts[0]
            cutoff = start + dataset.context_size * dataset.patch_size - 1
            expected = torch.log(
                dataset.close_series[cutoff + 1 : cutoff + 6]
                / dataset.close_series[cutoff]
            )
            self.assertTrue(torch.allclose(target, expected, atol=1e-7))
            self.assertTrue(
                torch.allclose(target, torch.arange(1, 6) * 0.01, atol=2e-6)
            )

    def test_market_alignment_is_a_deterministic_date_intersection(self):
        with tempfile.TemporaryDirectory() as tmp:
            stock_path = Path(tmp) / "stock.csv"
            market_path = Path(tmp) / "market.csv"
            closes = np.linspace(100.0, 110.0, 12)
            _write_price_csv(stock_path, closes)
            _write_price_csv(market_path, closes * 2.0, missing_indices=(4, 8))
            stock = pd.read_csv(stock_path, parse_dates=["Date"])
            aligned, report, _ = align_market_data(
                stock,
                market_data=str(market_path),
                stock_path=str(stock_path),
            )
            self.assertEqual(len(aligned), 10)
            self.assertEqual(report["stock_rows_dropped"], 2)
            expected_dates = set(stock["Date"]) - {
                pd.Timestamp("2020-01-05"),
                pd.Timestamp("2020-01-09"),
            }
            self.assertEqual(set(aligned["Date"]), expected_dates)

    def test_cumulative_return_baselines_and_direction_use_same_target(self):
        # Two patches, two time steps, two features. Interleaving matches the
        # encoder's [patch, time, feature] flattening convention.
        context = torch.tensor(
            [[0.01, 0.0, -0.02, 0.0], [0.03, 0.0, 0.04, 0.0]]
        )
        config = {
            "feature_transform": "return",
            "feature_cols": ["log_return_1", "log_return_5"],
            "feature_dim": 2,
            "patch_size": 2,
            "normalization": "none",
            "forecast_target": "cumulative_log_return",
            "target_feature_index": 0,
        }
        previous = make_baseline_prediction(
            context, 2, "previous_patch", config=config
        )
        mean = make_baseline_prediction(context, 2, "mean_context", config=config)
        naive = make_baseline_prediction(context, 2, "naive_last", config=config)
        self.assertTrue(np.allclose(previous, [0.03, 0.07]))
        self.assertTrue(np.allclose(mean, [0.015, 0.03]))
        self.assertTrue(np.allclose(naive, [0.0, 0.0]))
        self.assertEqual(
            compute_trend_accuracy(
                np.array([[0.1, -0.2]]),
                np.array([[0.01, -0.01]]),
                direct_return=True,
            ),
            1.0,
        )

    def test_checkpoint_round_trip_preserves_preprocessing_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            features = [*RETURN_FEATURE_COLS, "sentiment_mean", "news_count"]
            stats = train_robust_zscore_state(
                torch.arange(100, dtype=torch.float32).reshape(10, 10),
                features,
                clip=3.0,
            )
            config = {
                "mask_strategy": "random",
                "feature_cols": features,
                "feature_names": features,
                "feature_transform": "return",
                "normalization": "train_robust_zscore",
                "normalization_stats": stats,
                "eval_forecast_target": "cumulative_log_return",
                "target_definition": "log(Close[t+h] / Close[t])",
                "market_data": None,
                "target_feature_index": 0,
            }
            path = save_checkpoint(
                torch.nn.Linear(2, 2),
                torch.nn.Linear(2, 2),
                torch.nn.Linear(2, 2),
                str(Path(tmp) / "financial"),
                epoch=0,
                config=copy.deepcopy(config),
            )
            restored = torch.load(path, map_location="cpu", weights_only=False)[
                "config"
            ]
            for key in (
                "feature_names",
                "feature_transform",
                "normalization",
                "normalization_stats",
                "eval_forecast_target",
                "target_definition",
                "market_data",
                "target_feature_index",
            ):
                self.assertEqual(restored[key], config[key])


if __name__ == "__main__":
    unittest.main()
