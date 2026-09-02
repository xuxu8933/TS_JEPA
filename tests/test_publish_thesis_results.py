import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from publish_thesis_results import publish_thesis_results


class PublishThesisResultsTest(unittest.TestCase):
    def _write_valid_analysis(self, root, *, error_issues=0):
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
        results_dir = root / "results" / "experiment"
        results_dir.mkdir(parents=True)
        (results_dir / "experiment_manifest.json").write_text(
            json.dumps({"config_signature": "a" * 64}),
            encoding="utf-8",
        )

        analysis_dir = root / "analysis_artifacts"
        (analysis_dir / "data").mkdir(parents=True)
        (analysis_dir / "tables").mkdir()
        (analysis_dir / "figures").mkdir()
        (analysis_dir / "README.md").write_text(
            "# Analysis\n",
            encoding="utf-8",
        )
        (analysis_dir / "artifact_manifest.csv").write_text(
            "path,status\ndata/overall_summary.csv,generated\n",
            encoding="utf-8",
        )
        (analysis_dir / "analysis_metadata.json").write_text(
            json.dumps(
                {
                    "error_issues": error_issues,
                    "canonical_rows": 1,
                    "results_dir": str(results_dir),
                    "scope": {"config_path": str(config_path)},
                }
            ),
            encoding="utf-8",
        )
        (analysis_dir / "data" / "overall_summary.csv").write_text(
            "model,rmse\nTS-JEPA,0.1\n",
            encoding="utf-8",
        )
        (analysis_dir / "tables" / "table_main_metrics.tex").write_text(
            "\\begin{tabular}{lr}model & rmse\\\\\\end{tabular}\n",
            encoding="utf-8",
        )
        (analysis_dir / "figures" / "main.png").write_bytes(b"png")
        return analysis_dir

    def test_publishes_valid_snapshot_with_provenance_and_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis_dir = self._write_valid_analysis(root)
            large_path = analysis_dir / "data" / "predictions_tidy.csv"
            large_path.write_bytes(b"x" * 1025)

            destination = publish_thesis_results(
                analysis_dir,
                root / "thesis_results",
                max_file_bytes=1024,
            )

            self.assertEqual(destination.parent.name, "experiment")
            self.assertTrue(destination.name.startswith("aaaaaaaaaaaa-"))
            self.assertTrue((destination / "data" / "overall_summary.csv").is_file())
            self.assertFalse(
                (destination / "data" / "predictions_tidy.csv").exists()
            )
            self.assertTrue(
                (destination / "provenance" / "experiment_config.json").is_file()
            )
            self.assertTrue(
                (destination / "provenance" / "experiment_manifest.json").is_file()
            )

            with (destination / "publication_manifest.csv").open(
                encoding="utf-8",
                newline="",
            ) as handle:
                records = list(csv.DictReader(handle))
            prediction_record = next(
                record
                for record in records
                if record["source"] == "data/predictions_tidy.csv"
            )
            self.assertEqual(prediction_record["status"], "omitted")
            self.assertIn("larger-than-1024-bytes", prediction_record["reason"])

            checksum_lines = (destination / "SHA256SUMS").read_text(
                encoding="utf-8"
            )
            summary = destination / "data" / "overall_summary.csv"
            expected_hash = hashlib.sha256(summary.read_bytes()).hexdigest()
            self.assertIn(
                f"{expected_hash}  data/overall_summary.csv",
                checksum_lines,
            )
            self.assertEqual(
                publish_thesis_results(
                    analysis_dir,
                    root / "thesis_results",
                    max_file_bytes=1024,
                ),
                destination,
            )

    def test_refuses_analysis_with_validity_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis_dir = self._write_valid_analysis(root, error_issues=2)

            with self.assertRaisesRegex(RuntimeError, "2 validity error"):
                publish_thesis_results(
                    analysis_dir,
                    root / "thesis_results",
                )

    def test_allows_clearly_marked_incomplete_test_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis_dir = self._write_valid_analysis(root, error_issues=2)

            destination = publish_thesis_results(
                analysis_dir,
                root / "test_publications",
                allow_incomplete=True,
            )

            self.assertTrue(destination.name.startswith("incomplete-aaaaaaaaaaaa-"))
            readme = (destination / "README.md").read_text(encoding="utf-8")
            self.assertIn("INCOMPLETE TEST SNAPSHOT", readme)
            self.assertIn("2 validity error(s)", readme)
            self.assertIn("not a validated thesis result", readme)


if __name__ == "__main__":
    unittest.main()
