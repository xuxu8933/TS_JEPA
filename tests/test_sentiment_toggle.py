import csv
import copy
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import torch

from config.config_pretrain import config as pretrain_defaults
from config.experiment import resolve_feature_selection
from pretrain_dual_loss import parse_args
from eval_dual_loss import build_eval_argv, parse_args as parse_eval_args
from run_top_nasdaq100_stocks import (
    build_stock_commands,
    parse_args as parse_stock_args,
)
from src.data_loaders.data_class_roll_volume import (
    CSVDataLoader,
    EvaluationDataLoader,
)
from src.models.encoder import Encoder


MARKET_FEATURES = ["Close", "Volume", "MA10", "MA50"]
SENTIMENT_FEATURES = ["sentiment_mean"]


def _write_market(path, num_rows=180):
    path.parent.mkdir(parents=True, exist_ok=True)
    start = date(2021, 1, 1)
    with path.open("w", newline="") as market_file:
        writer = csv.DictWriter(
            market_file,
            fieldnames=["Date", "Close", "Volume"],
        )
        writer.writeheader()
        for index in range(num_rows):
            writer.writerow(
                {
                    "Date": (start + timedelta(days=index)).isoformat(),
                    "Close": 100.0 + index * 0.25,
                    "Volume": 1_000.0 + index,
                }
            )


def _write_sparse_sentiment(path, num_rows=180):
    start = date(2021, 1, 1)
    with path.open("w", newline="") as sentiment_file:
        writer = csv.DictWriter(
            sentiment_file,
            fieldnames=["date", "sentiment_mean"],
        )
        writer.writeheader()
        for index in range(0, num_rows, 3):
            writer.writerow(
                {
                    "date": (start + timedelta(days=index)).isoformat(),
                    "sentiment_mean": (index % 9 - 4) / 4.0,
                }
            )


class SentimentToggleTest(unittest.TestCase):
    def _selection(self, use_sentiment):
        return resolve_feature_selection(
            pretrain_defaults,
            market_features=MARKET_FEATURES,
            sentiment_features=SENTIMENT_FEATURES,
            use_sentiment=use_sentiment,
        )

    def _dataset_kwargs(self, market_path, sentiment_path, feature_cols):
        return {
            "path_data": str(market_path),
            "patch_size": 5,
            "normalization": "none",
            "feature_cols": feature_cols,
            "sentiment_path": str(sentiment_path),
            "validation_fraction": 0.1,
            "test_start_date": "2021-06-01",
        }

    def test_default_and_cli_toggle_resolve_one_effective_feature_list(self):
        default_config = parse_args(copy.deepcopy(pretrain_defaults), argv=[])
        market_only = parse_args(
            copy.deepcopy(pretrain_defaults),
            argv=["--no-sentiment"],
        )

        self.assertTrue(default_config["use_sentiment"])
        self.assertEqual(
            default_config["feature_cols"],
            [*MARKET_FEATURES, *SENTIMENT_FEATURES],
        )
        self.assertFalse(market_only["use_sentiment"])
        self.assertEqual(market_only["feature_cols"], MARKET_FEATURES)

    def test_both_modes_derive_dataset_and_encoder_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            market_path = Path(tmp) / "TOGGLE.csv"
            sentiment_path = Path(tmp) / "TOGGLE_daily_sentiment.csv"
            _write_market(market_path)
            _write_sparse_sentiment(sentiment_path)

            datasets = {}
            for use_sentiment, expected_dim in ((False, 4), (True, 5)):
                selection = self._selection(use_sentiment)
                dataset = CSVDataLoader(
                    series_split_size=20,
                    mask_ratio=0.5,
                    **self._dataset_kwargs(
                        market_path,
                        sentiment_path,
                        selection["feature_cols"],
                    ),
                )
                patches, _, _ = dataset[0]
                self.assertEqual(dataset.feature_dim, expected_dim)
                self.assertEqual(patches.shape[1], 5 * expected_dim)

                encoder = Encoder(
                    num_patches=patches.shape[0],
                    dim_in=patches.shape[1],
                    kernel_size=5,
                    embed_dim=8,
                    embed_bias=True,
                    nhead=2,
                    num_layers=1,
                    jepa=True,
                )
                encoded = encoder(patches.unsqueeze(0))
                self.assertEqual(tuple(encoded.shape), (1, 4, 8))
                datasets[use_sentiment] = dataset

            self.assertEqual(
                datasets[False].sample_starts,
                datasets[True].sample_starts,
            )
            self.assertTrue(
                torch.equal(datasets[False].train_df[:, 0], datasets[True].train_df[:, 0])
            )

    def test_disabled_mode_does_not_require_a_sentiment_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            market_path = Path(tmp) / "MARKET_ONLY.csv"
            missing_sentiment_path = Path(tmp) / "does-not-exist.csv"
            _write_market(market_path)

            selection = self._selection(False)
            dataset = CSVDataLoader(
                series_split_size=20,
                mask_ratio=0.5,
                **self._dataset_kwargs(
                    market_path,
                    missing_sentiment_path,
                    selection["feature_cols"],
                ),
            )

            self.assertEqual(dataset.feature_cols, MARKET_FEATURES)
            self.assertGreater(len(dataset), 0)

    def test_enabled_mode_fails_clearly_without_sentiment_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            market_path = Path(tmp) / "MISSING_NEWS.csv"
            missing_sentiment_path = Path(tmp) / "does-not-exist.csv"
            _write_market(market_path)

            selection = self._selection(True)
            with self.assertRaisesRegex(
                FileNotFoundError,
                "Sentiment features were requested",
            ):
                CSVDataLoader(
                    series_split_size=20,
                    mask_ratio=0.5,
                    **self._dataset_kwargs(
                        market_path,
                        missing_sentiment_path,
                        selection["feature_cols"],
                    ),
                )

    def test_toggle_preserves_chronology_samples_and_forecast_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            market_path = Path(tmp) / "CONTROLLED.csv"
            sentiment_path = Path(tmp) / "CONTROLLED_daily_sentiment.csv"
            _write_market(market_path)
            _write_sparse_sentiment(sentiment_path)

            datasets = {}
            for use_sentiment in (False, True):
                selection = self._selection(use_sentiment)
                datasets[use_sentiment] = EvaluationDataLoader(
                    context_size=4,
                    stride=1,
                    split="train",
                    forecast_target="relative_return",
                    **self._dataset_kwargs(
                        market_path,
                        sentiment_path,
                        selection["feature_cols"],
                    ),
                )

            market_only = datasets[False]
            with_sentiment = datasets[True]
            self.assertEqual(market_only.train_dates, with_sentiment.train_dates)
            self.assertEqual(market_only.val_dates, with_sentiment.val_dates)
            self.assertEqual(market_only.test_dates, with_sentiment.test_dates)
            self.assertEqual(market_only.sample_starts, with_sentiment.sample_starts)
            self.assertEqual(len(market_only), len(with_sentiment))
            for index in range(len(market_only)):
                _, market_target = market_only[index]
                _, sentiment_target = with_sentiment[index]
                self.assertTrue(torch.equal(market_target, sentiment_target))

    def test_invalid_patch_relationship_fails_during_config_resolution(self):
        with self.assertRaisesRegex(
            ValueError,
            "series_split_size must be divisible by patch_size",
        ):
            parse_args(
                copy.deepcopy(pretrain_defaults),
                argv=["--series-split-size", "21", "--patch-size", "5"],
            )

    def test_evaluation_wrapper_forwards_the_explicit_toggle(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, passthrough = parse_eval_args(
                argv=[
                    "--data",
                    "NO_SENTIMENT_CHECKPOINT",
                    "--checkpoint-dir",
                    tmp,
                    "--no-sentiment",
                    "--dry-run",
                ],
            )
            eval_argv, _ = build_eval_argv(args, passthrough)

        self.assertIn("--no-sentiment", eval_argv)
        self.assertNotIn("--use-sentiment", eval_argv)

    def test_stock_runner_does_not_pass_a_sentiment_path_when_disabled(self):
        with patch(
            "sys.argv",
            ["run_top_nasdaq100_stocks.py", "--no-sentiment", "--dry-run"],
        ):
            args = parse_stock_args()
        pretrain_command, eval_command = build_stock_commands(args, "NVDA")

        self.assertIn("--no-sentiment", pretrain_command)
        self.assertIn("--no-sentiment", eval_command)
        self.assertNotIn("--sentiment-path", pretrain_command)


if __name__ == "__main__":
    unittest.main()
