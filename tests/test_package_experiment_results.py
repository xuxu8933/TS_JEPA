import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from config.file_options import results_dir_from_config
from package_experiment_results import package_experiment_results


class PackageExperimentResultsTest(unittest.TestCase):
    def _write_complete_experiment(self, root):
        config_path = root / "experiment.json"
        config_path.write_text(
            json.dumps(
                {
                    "common": {"stocks": ["NVDA"], "seeds": [42]},
                    "runner": {"mask_strategies": ["random"]},
                }
            ),
            encoding="utf-8",
        )
        results_dir = results_dir_from_config(config_path)
        run_dir = results_dir / "random" / "NVDA" / "seed_42"
        run_dir.mkdir(parents=True)
        comparison_name = "last_model_comparison_20260101_000000.csv"
        (run_dir / comparison_name).write_text(
            "model,mse\nTS-JEPA,0.1\n",
            encoding="utf-8",
        )
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "strategy": "random",
                    "stock": "NVDA",
                    "seed": 42,
                    "comparison_files": [comparison_name],
                }
            ),
            encoding="utf-8",
        )
        (results_dir / "experiment_manifest.json").write_text(
            json.dumps(
                {
                    "config_signature": "b" * 64,
                    "stocks": ["NVDA"],
                    "seeds": [42],
                    "run_stocks": ["NVDA"],
                    "run_seeds": [42],
                    "mask_strategies": ["random"],
                }
            ),
            encoding="utf-8",
        )
        return config_path, results_dir

    def test_packages_complete_experiment_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, _ = self._write_complete_experiment(root)
            archive_root = root / "release_assets"

            archive_path = package_experiment_results(config_path, archive_root)

            self.assertEqual(archive_path.name, f"experiment-{'b' * 12}.zip")
            self.assertTrue(archive_path.with_suffix(".zip.sha256").is_file())
            with zipfile.ZipFile(archive_path) as archive:
                names = set(archive.namelist())
                self.assertIn("experiment/ARCHIVE_MANIFEST.csv", names)
                self.assertIn("experiment/ARCHIVE_README.txt", names)
                self.assertIn(
                    "experiment/random/NVDA/seed_42/run_manifest.json",
                    names,
                )
                rows = list(
                    csv.DictReader(
                        io.StringIO(
                            archive.read(
                                "experiment/ARCHIVE_MANIFEST.csv"
                            ).decode("utf-8")
                        )
                    )
                )
                self.assertTrue(rows)

            first_bytes = archive_path.read_bytes()
            self.assertEqual(
                package_experiment_results(config_path, archive_root),
                archive_path,
            )
            self.assertEqual(archive_path.read_bytes(), first_bytes)

    def test_refuses_incomplete_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, results_dir = self._write_complete_experiment(root)
            manifest_path = results_dir / "experiment_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["run_seeds"] = [42, 43]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "incomplete experiment"):
                package_experiment_results(config_path, root / "release_assets")


if __name__ == "__main__":
    unittest.main()
