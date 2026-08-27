import copy
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, timedelta
from pathlib import Path

from validate_normalization_pilot import (
    compare_target_datasets,
    main,
    validate_config_pair,
    validate_result_metadata_pair,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ZSCORE_CONFIG = (
    REPO_ROOT / "config/experiments/normalization_pilot_train_zscore.json"
)
WINDOW_RETURN_CONFIG = (
    REPO_ROOT / "config/experiments/normalization_pilot_window_return.json"
)


def _write_price_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = date(2020, 1, 1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Date", "Close", "Volume"])
        writer.writeheader()
        for index in range(2200):
            writer.writerow(
                {
                    "Date": (start + timedelta(days=index)).isoformat(),
                    "Close": 100.0 + 0.2 * index + (index % 7) * 0.03,
                    "Volume": 1_000_000.0 + 101.0 * index,
                }
            )


class NormalizationPilotConfigTest(unittest.TestCase):
    def test_configs_isolate_input_normalization_with_relative_return_target(self):
        report = validate_config_pair(TRAIN_ZSCORE_CONFIG, WINDOW_RETURN_CONFIG)

        self.assertTrue(report["valid"], report)
        self.assertEqual(report["forecast_target"], "relative_return")
        self.assertEqual(report["stocks"], ["NVDA"])
        self.assertEqual(report["seeds"], [42, 44, 46])
        self.assertEqual(
            report["semantic_differences"],
            {
                "normalization": {
                    "train_zscore": "train_zscore",
                    "window_return": "window_return",
                }
            },
        )

    def test_value_target_is_rejected_as_normalization_coupled(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            paths = []
            for source in (TRAIN_ZSCORE_CONFIG, WINDOW_RETURN_CONFIG):
                payload = json.loads(source.read_text(encoding="utf-8"))
                payload["runner"]["preprocessing"]["custom"]["forecast"][
                    "target"
                ] = "value"
                destination = temp_root / source.name
                destination.write_text(json.dumps(payload), encoding="utf-8")
                paths.append(destination)

            with self.assertRaisesRegex(
                ValueError,
                "relative_return.*identical target space",
            ):
                validate_config_pair(*paths)


class NormalizationPilotTargetTest(unittest.TestCase):
    def test_targets_are_bitwise_identical_across_all_chronological_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "NVDA.csv"
            _write_price_csv(data_path)

            report = compare_target_datasets(
                TRAIN_ZSCORE_CONFIG,
                WINDOW_RETURN_CONFIG,
                data_paths={"NVDA": data_path},
            )

        self.assertTrue(report["valid"], report)
        for split in ("train", "val", "test"):
            split_report = report["stocks"]["NVDA"][split]
            self.assertTrue(split_report["dates_equal"])
            self.assertTrue(split_report["sample_starts_equal"])
            self.assertTrue(split_report["targets_bitwise_equal"])
            self.assertTrue(split_report["contexts_differ"])
            self.assertEqual(
                split_report["train_zscore_target_sha256"],
                split_report["window_return_target_sha256"],
            )

    def test_cli_emits_one_machine_readable_json_document(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            return_code = main([])

        report = json.loads(stdout.getvalue())
        self.assertEqual(return_code, 0)
        self.assertTrue(report["valid"])
        self.assertTrue(report["targets"]["valid"])


class NormalizationPilotResultMetadataTest(unittest.TestCase):
    BASE_METADATA = {
        "forecast_target": "relative_return",
        "target_definition": "Close[t+h] / Close[t] - 1",
        "metric_definition": (
            "MSE and MAE over every saved rolling-step/horizon target value"
        ),
        "forecast_horizon": 5,
        "test_sample_count": 38,
        "test_target_start": "2025-03-28T00:00:00",
        "test_target_end": "2025-12-31T00:00:00",
    }

    def _write_metadata_pair(self, root: Path):
        roots = {
            "train_zscore": root / "train_zscore",
            "window_return": root / "window_return",
        }
        relative = Path("random/NVDA/seed_42/preprocessing_config.json")
        for normalization, result_root in roots.items():
            path = result_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = copy.deepcopy(self.BASE_METADATA)
            payload["normalization"] = normalization
            path.write_text(json.dumps(payload), encoding="utf-8")
        return roots, relative

    def test_result_metadata_uses_the_same_target_and_metric_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots, _ = self._write_metadata_pair(Path(tmp))
            report = validate_result_metadata_pair(
                roots["train_zscore"], roots["window_return"]
            )

        self.assertTrue(report["valid"], report)
        self.assertEqual(report["matched_runs"], 1)
        self.assertEqual(report["forecast_target"], "relative_return")

    def test_result_metadata_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots, relative = self._write_metadata_pair(Path(tmp))
            path = roots["window_return"] / relative
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["target_definition"] = "different target"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "target_definition"):
                validate_result_metadata_pair(
                    roots["train_zscore"], roots["window_return"]
                )


if __name__ == "__main__":
    unittest.main()
