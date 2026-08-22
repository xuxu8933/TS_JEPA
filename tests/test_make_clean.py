import subprocess
import tempfile
import unittest
from pathlib import Path


class MakeCleanTest(unittest.TestCase):
    def test_clean_removes_all_analysis_artifacts(self):
        project_root = Path(__file__).resolve().parents[1]
        roots = {
            name: project_root / name
            for name in ("results", "logs", "analysis_artifacts")
        }
        thesis_root = project_root / "thesis_results"
        for root in roots.values():
            root.mkdir(exist_ok=True)
        thesis_root.mkdir(exist_ok=True)

        with (
            tempfile.TemporaryDirectory(dir=roots["results"]) as results_dir,
            tempfile.TemporaryDirectory(dir=roots["logs"]) as logs_dir,
            tempfile.TemporaryDirectory(dir=roots["analysis_artifacts"])
            as analysis_dir,
            tempfile.TemporaryDirectory(dir=thesis_root) as thesis_dir,
        ):
            Path(results_dir, "result.csv").write_text("metric,value\n", encoding="utf-8")
            Path(logs_dir, "checkpoint.pt").write_bytes(b"checkpoint")
            nested = Path(analysis_dir, "tables")
            nested.mkdir()
            Path(nested, "table.tex").write_text("generated", encoding="utf-8")
            Path(analysis_dir, "extensionless").write_text("generated", encoding="utf-8")
            published = Path(thesis_dir, "overall_summary.csv")
            published.write_text("model,mse\nTS-JEPA,0.1\n", encoding="utf-8")

            completed = subprocess.run(
                [
                    "make",
                    "clean",
                    f"RESULTS_DIR={results_dir}",
                    f"LOGS_DIR={logs_dir}",
                    f"ANALYSIS_DIR={analysis_dir}",
                ],
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertEqual(list(Path(results_dir).iterdir()), [])
            self.assertEqual(list(Path(logs_dir).iterdir()), [])
            self.assertEqual(list(Path(analysis_dir).iterdir()), [])
            self.assertTrue(published.is_file())


if __name__ == "__main__":
    unittest.main()
