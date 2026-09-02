import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from config.file_options import read_config_file
from eval_forecast_prequential_with_baselines_gru_volume import (
    build_downstream_metrics_artifact,
    write_downstream_metrics_artifact,
)
from run_top_nasdaq100_stocks import (
    experiment_config_signature,
    parse_args as parse_runner_args,
    resolve_mask_strategies,
    resolve_seeds,
    resolve_stocks,
    strategy_results_dir,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = REPO_ROOT / "config" / "experiments" / "chapter5_candidates"


class Chapter5AutomationDryRunTest(unittest.TestCase):
    def test_dry_run_validates_all_ten_candidates_without_repository_artifacts(self):
        before = sorted(path.name for path in CANDIDATE_DIR.glob("*.json"))
        self.assertTrue(
            {
                "01_preprocessing_train_zscore.json",
                "01_preprocessing_window_return.json",
            }.issubset(before)
        )

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp) / "selection_artifacts"
            completed = subprocess.run(
                [
                    sys.executable,
                    "run_chapter5_staged.py",
                    "--dry-run",
                    "--artifacts-dir",
                    str(artifacts),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.count("DRY_RUN_VALIDATION"), 10)
            marker = "CHAPTER5_DRY_RUN_SUMMARY\n"
            self.assertIn(marker, completed.stdout)
            report = json.loads(completed.stdout.split(marker, 1)[1])
            self.assertEqual(report["validated_candidate_count"], 10)
            self.assertEqual(report["stage_candidate_counts"], [2, 2, 6])
            self.assertFalse(report["selection_performed"])
            self.assertFalse(report["test_evaluation_performed"])
            self.assertFalse(artifacts.exists())

        after = sorted(path.name for path in CANDIDATE_DIR.glob("*.json"))
        self.assertEqual(after, before)

    def test_checked_in_manifests_use_the_runner_result_roots_directly(self):
        templates = [
            REPO_ROOT
            / "config"
            / "experiments"
            / "chapter5_stage1_selection.template.jsonc",
            REPO_ROOT
            / "config"
            / "experiments"
            / "chapter5_stage2_selection.template.jsonc",
            REPO_ROOT
            / "config"
            / "experiments"
            / "chapter5_selection.template.jsonc",
        ]
        for template in templates:
            _, manifest = read_config_file(template)
            for stage in manifest["stages"]:
                for candidate in stage["candidates"]:
                    root = candidate["validation_root"]
                    self.assertNotIn("REPLACE_CONFIG_ID", root)
                    self.assertTrue(
                        root.endswith(
                            f"/{Path(candidate['config']).stem}/"
                            f"{candidate['strategy']}"
                        )
                    )


class Chapter5CompleteAutomationTest(unittest.TestCase):
    def test_complete_process_selects_each_stage_before_one_final_test(self):
        from run_chapter5_staged import run_complete_process

        scores = {
            "01_preprocessing_window_return": 0.10,
            "01_preprocessing_train_zscore": 0.20,
            "02_sentiment_excluded": 0.09,
            "02_sentiment_included": 0.05,
            "03_shared_context_6_patches": 0.08,
            "03_shared_context_12_patches": 0.07,
            "03_shared_context_24_patches": 0.09,
            "03_local_long_context_6_patches": 0.06,
            "03_local_long_context_12_patches": 0.01,
            "03_local_long_context_24_patches": 0.04,
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage_one_dir = root / "stage_one_configs"
            stage_one_dir.mkdir()
            stage_one = []
            for candidate_id, filename in (
                (
                    "preprocessing_window_return",
                    "01_preprocessing_window_return.json",
                ),
                (
                    "preprocessing_train_zscore",
                    "01_preprocessing_train_zscore.json",
                ),
            ):
                source = CANDIDATE_DIR / filename
                destination = stage_one_dir / filename
                destination.write_bytes(source.read_bytes())
                stage_one.append((candidate_id, destination))

            artifacts_dir = root / "selection_artifacts"
            calls = []

            def complete_without_training(config_path, *, dry_run):
                self.assertFalse(dry_run)
                config_path = Path(config_path)
                args = parse_runner_args(["--config", str(config_path)])
                split = args.evaluation_split
                calls.append((config_path.stem, split))
                if split == "test":
                    self.assertTrue(
                        (artifacts_dir / "final" / "selection_summary.json").is_file()
                    )
                    return

                rmse = scores[config_path.stem]
                signature = experiment_config_signature(args)
                for strategy in resolve_mask_strategies(args):
                    validation_root = strategy_results_dir(args, strategy)
                    if args.preprocessing_preset:
                        validation_root /= args.preprocessing_preset
                    for stock in resolve_stocks(args):
                        for seed in resolve_seeds(args):
                            artifact = build_downstream_metrics_artifact(
                                split="validation",
                                config_signature=signature,
                                stock=stock,
                                seed=seed,
                                strategy=strategy,
                                metrics={
                                    "rmse": rmse,
                                    "direction_accuracy": 1.0 - rmse,
                                },
                            )
                            write_downstream_metrics_artifact(
                                validation_root / stock / f"seed_{seed}",
                                artifact,
                            )

            report = run_complete_process(
                artifacts_dir=artifacts_dir,
                candidate_dir=root / "generated_candidates",
                stage_one=tuple(stage_one),
                candidate_executor=complete_without_training,
            )

            self.assertEqual(
                [name for name, split in calls if split == "validation"],
                [
                    "01_preprocessing_window_return",
                    "01_preprocessing_train_zscore",
                    "02_sentiment_included",
                    "03_shared_context_6_patches",
                    "03_shared_context_24_patches",
                    "03_local_long_context_6_patches",
                    "03_local_long_context_12_patches",
                    "03_local_long_context_24_patches",
                ],
            )
            stage_two_manifest = json.loads(
                (artifacts_dir / "stage2" / "selection_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                stage_two_manifest["stages"][1]["candidates"][0][
                    "validation_root"
                ],
                stage_two_manifest["stages"][0]["candidates"][0][
                    "validation_root"
                ],
            )
            self.assertEqual(calls[-1], ("selected_config", "test"))
            self.assertEqual(
                report["selected_candidate_id"],
                "local_long_context_12",
            )
            self.assertEqual(report["validation_candidate_count"], 10)
            self.assertTrue(report["test_evaluation_performed"])

            final_config_path = Path(report["selected_config"])
            _, final_config = read_config_file(final_config_path)
            self.assertEqual(
                final_config["runner"]["downstream"]["evaluation_split"],
                "test",
            )
            stage_one_manifest = json.loads(
                (artifacts_dir / "stage1" / "selection_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                stage_one_manifest["stages"][0]["candidates"][0][
                    "validation_root"
                ],
                str(
                    root
                    / "stage_one_configs"
                    / "results"
                    / "01_preprocessing_window_return"
                    / "random"
                ),
            )


if __name__ == "__main__":
    unittest.main()
