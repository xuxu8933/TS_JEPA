import json
import tempfile
import unittest
from pathlib import Path

from eval_dual_loss import build_eval_argv, parse_args as parse_eval_args
from eval_forecast_prequential_with_baselines_gru_volume import (
    build_downstream_metrics_artifact,
    choose_evaluation_loader,
    write_downstream_metrics_artifact,
)
from run_top_nasdaq100_stocks import (
    build_stock_commands,
    experiment_config_signature,
    parse_args as parse_runner_args,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    REPO_ROOT / "config" / "experiments" / "normalization_pilot_window_return.json"
)


class RunnerSelectionConfigTest(unittest.TestCase):
    def _write_config(self, root: Path, *, context_size=8):
        payload = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
        payload["runner"]["downstream"].update(
            {
                "context_size": context_size,
                "evaluation_split": "validation",
            }
        )
        payload["runner"]["checkpoint"] = {
            "selection": {"mode": "best"},
            "encoder_weights": "ema",
        }
        payload["provenance"] = {
            "artifact_type": "frozen_chapter5_experiment_config",
            "experiment_config_sha256": "abc123",
        }
        path = root / "candidate.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_nested_downstream_forwards_context_and_validation_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_config(Path(tmp))
            args = parse_runner_args(["--config", str(config_path)])
            _, command = build_stock_commands(
                args,
                "NVDA",
                seed=42,
                strategy="random",
            )

        self.assertEqual(args.evaluation_split, "validation")
        self.assertEqual(args.context_size, 8)
        self.assertEqual(
            command[command.index("--evaluation-split") + 1],
            "validation",
        )
        self.assertEqual(command[command.index("--context-size") + 1], "8")
        self.assertEqual(
            command[command.index("--experiment-config-signature") + 1],
            experiment_config_signature(args),
        )

    def test_frozen_provenance_is_ignored_by_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_config(Path(tmp))
            args = parse_runner_args(["--config", str(config_path)])

        self.assertEqual(args.evaluation_split, "validation")
        self.assertFalse(hasattr(args, "provenance"))

    def test_context_size_must_be_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_config(Path(tmp), context_size=0)
            with self.assertRaisesRegex(ValueError, "context_size must be positive"):
                parse_runner_args(["--config", str(config_path)])


class DownstreamSplitTest(unittest.TestCase):
    IDENTITY = {
        "config_signature": "a" * 64,
        "stock": "NVDA",
        "seed": 42,
        "strategy": "random",
        "metrics": {
            "mse": 0.125,
            "mae": 0.25,
            "direction_accuracy": 0.75,
        },
    }

    def test_validation_loader_never_calls_test_factory(self):
        validation_loader = object()

        def reject_test_loader():
            self.fail("validation-only execution constructed the test loader")

        chosen = choose_evaluation_loader(
            "validation",
            validation_loader,
            reject_test_loader,
        )
        self.assertIs(chosen, validation_loader)

    def test_test_loader_is_constructed_lazily(self):
        test_loader = object()
        calls = []

        chosen = choose_evaluation_loader(
            "test",
            object(),
            lambda: calls.append("test") or test_loader,
        )

        self.assertIs(chosen, test_loader)
        self.assertEqual(calls, ["test"])

    def test_metrics_artifacts_have_explicit_disjoint_splits(self):
        validation = build_downstream_metrics_artifact(
            split="validation",
            **self.IDENTITY,
        )
        test = build_downstream_metrics_artifact(split="test", **self.IDENTITY)

        self.assertEqual(validation["split"], "validation")
        self.assertEqual(test["split"], "test")
        self.assertNotEqual(
            validation["artifact_filename"],
            test["artifact_filename"],
        )

    def test_metrics_writer_uses_canonical_split_filename(self):
        artifact = build_downstream_metrics_artifact(
            split="validation",
            **self.IDENTITY,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_downstream_metrics_artifact(Path(tmp), artifact)
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(path.name, "validation_metrics.json")
        self.assertEqual(saved, artifact)

    def test_unified_wrapper_forwards_split_identity_and_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            explicit_checkpoint = Path(tmp) / "fixture_checkpoint.pt"
            args, passthrough = parse_eval_args(
                argv=[
                    "--data",
                    "NVDA",
                    "--mask-strategy",
                    "local_long",
                    "--pretrain-checkpoint-path",
                    str(explicit_checkpoint),
                    "--evaluation-split",
                    "validation",
                    "--experiment-config-signature",
                    "a" * 64,
                ]
            )
            argv, _ = build_eval_argv(args, passthrough)

        self.assertEqual(argv[argv.index("--mask-strategy") + 1], "local_long")
        self.assertEqual(
            argv[argv.index("--evaluation-split") + 1],
            "validation",
        )
        self.assertEqual(
            argv[argv.index("--experiment-config-signature") + 1],
            "a" * 64,
        )


class ValidationArtifactGuardTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / "validation_metrics.json"
        self.identity = {
            "config_signature": "b" * 64,
            "stock": "NVDA",
            "seed": 42,
            "strategy": "random",
        }

    def tearDown(self):
        self.temporary.cleanup()

    def _write_artifact(self, *, split="validation", extra=None, path=None):
        artifact = build_downstream_metrics_artifact(
            split=split,
            metrics={
                "mse": 0.1,
                "mae": 0.2,
                "direction_accuracy": 0.6,
            },
            **self.identity,
        )
        if extra:
            artifact.update(extra)
        destination = path or self.path
        destination.write_text(json.dumps(artifact), encoding="utf-8")
        return destination

    def test_selector_accepts_exact_validation_identity(self):
        from chapter5_selection import load_validation_artifact

        self._write_artifact()
        metrics = load_validation_artifact(self.path, self.identity)
        self.assertEqual(
            metrics,
            {"mse": 0.1, "mae": 0.2, "direction_accuracy": 0.6},
        )

    def test_selector_rejects_test_split_even_when_renamed(self):
        from chapter5_selection import load_validation_artifact

        self._write_artifact(split="test")
        with self.assertRaisesRegex(ValueError, "validation-only"):
            load_validation_artifact(self.path, self.identity)

    def test_selector_rejects_nested_test_metric_keys(self):
        from chapter5_selection import load_validation_artifact

        self._write_artifact(extra={"provenance": {"test_mse": 0.0}})
        with self.assertRaisesRegex(ValueError, "test-result"):
            load_validation_artifact(self.path, self.identity)

    def test_selector_rejects_test_filename_and_identity_mismatch(self):
        from chapter5_selection import load_validation_artifact

        wrong_path = self.root / "test_metrics.json"
        self._write_artifact(path=wrong_path)
        with self.assertRaisesRegex(ValueError, "validation_metrics.json"):
            load_validation_artifact(wrong_path, self.identity)

        self._write_artifact()
        wrong_identity = {**self.identity, "seed": 99}
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            load_validation_artifact(self.path, wrong_identity)

    def test_selector_rejects_non_finite_metrics(self):
        from chapter5_selection import load_validation_artifact

        artifact = build_downstream_metrics_artifact(
            split="validation",
            metrics={
                "mse": float("nan"),
                "mae": 0.2,
                "direction_accuracy": 0.6,
            },
            **self.identity,
        )
        self.path.write_text(json.dumps(artifact), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "finite"):
            load_validation_artifact(self.path, self.identity)

    def test_selector_rejects_extra_metrics(self):
        from chapter5_selection import load_validation_artifact

        self._write_artifact()
        artifact = json.loads(self.path.read_text(encoding="utf-8"))
        artifact["metrics"]["loss"] = 1.0
        self.path.write_text(json.dumps(artifact), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly"):
            load_validation_artifact(self.path, self.identity)


class DeterministicSelectionTest(unittest.TestCase):
    STAGE_NAMES = (
        "preprocessing_normalization",
        "sentiment",
        "historical_context",
        "architecture_objective",
    )

    def test_aggregation_means_seeds_within_stock_then_stocks(self):
        from chapter5_selection import aggregate_candidate

        summary = aggregate_candidate(
            {
                "AAPL": {
                    42: {"mse": 0.0, "mae": 1.0, "direction_accuracy": 0.4},
                    43: {"mse": 2.0, "mae": 3.0, "direction_accuracy": 0.6},
                },
                "NVDA": {
                    42: {"mse": 100.0, "mae": 5.0, "direction_accuracy": 0.8},
                    43: {"mse": 100.0, "mae": 7.0, "direction_accuracy": 1.0},
                },
            }
        )

        self.assertEqual(summary["per_stock"]["AAPL"]["mse"], 1.0)
        self.assertEqual(summary["per_stock"]["NVDA"]["mse"], 100.0)
        self.assertEqual(summary["overall"]["mse"], 50.5)
        self.assertEqual(summary["overall"]["mae"], 4.0)
        self.assertEqual(summary["overall"]["direction_accuracy"], 0.7)

    def _write_candidate(
        self,
        root,
        candidate_id,
        mse,
        *,
        parent=None,
        checkpoint_mode="best",
    ):
        from run_top_nasdaq100_stocks import experiment_config_signature

        payload = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
        payload["common"] = {"stocks": ["NVDA"], "seeds": [42]}
        payload["runner"]["execution"]["max_stocks"] = 1
        payload["runner"]["execution"]["max_seeds"] = 1
        payload["runner"]["downstream"]["evaluation_split"] = "validation"
        selection = {"mode": checkpoint_mode}
        if checkpoint_mode == "epoch":
            selection["epoch"] = 500
        payload["runner"]["checkpoint"]["selection"] = selection
        config_path = root / f"{candidate_id}.json"
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        args = parse_runner_args(["--config", str(config_path)])
        signature = experiment_config_signature(args)
        validation_root = root / "validation" / candidate_id / "random"
        artifact_path = validation_root / "NVDA" / "seed_42"
        artifact_path.mkdir(parents=True, exist_ok=True)
        artifact = build_downstream_metrics_artifact(
            split="validation",
            config_signature=signature,
            stock="NVDA",
            seed=42,
            strategy="random",
            metrics={
                "mse": mse,
                "mae": mse + 0.1,
                "direction_accuracy": 1.0 - mse / 10.0,
            },
        )
        write_downstream_metrics_artifact(artifact_path, artifact)
        candidate = {
            "id": candidate_id,
            "config": config_path.name,
            "validation_root": validation_root.relative_to(root).as_posix(),
            "strategy": "random",
        }
        if parent is not None:
            candidate["parent_candidate_id"] = parent
        return candidate

    def _write_workflow(self, root, *, reverse=False, invalid_best=False):
        stages = []
        previous_winner = None
        for index, stage_name in enumerate(self.STAGE_NAMES):
            winner_id = f"stage{index + 1}_winner"
            loser_id = f"stage{index + 1}_loser"
            checkpoint_mode = "epoch" if invalid_best and index == 0 else "best"
            candidates = [
                self._write_candidate(
                    root,
                    winner_id,
                    0.1 + index,
                    parent=previous_winner,
                    checkpoint_mode=checkpoint_mode,
                ),
                self._write_candidate(
                    root,
                    loser_id,
                    0.2 + index,
                    parent=previous_winner,
                ),
            ]
            if index > 0:
                candidates.append(
                    self._write_candidate(
                        root,
                        f"stage{index + 1}_ineligible",
                        0.0,
                        parent=f"stage{index}_loser",
                    )
                )
            if reverse:
                candidates.reverse()
            stages.append({"name": stage_name, "candidates": candidates})
            previous_winner = winner_id
        manifest = {
            "schema_version": 1,
            "selection_id": "chapter5_fixture",
            "stages": stages,
        }
        path = root / ("selection_reversed.json" if reverse else "selection.json")
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_selection_is_identical_when_candidates_are_reordered(self):
        from chapter5_selection import select_stages

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_manifest = self._write_workflow(root)
            first = select_stages(first_manifest)
            second_manifest = self._write_workflow(root, reverse=True)
            second = select_stages(second_manifest)

        first["manifest"] = "normalized"
        second["manifest"] = "normalized"
        self.assertEqual(first, second)
        self.assertEqual(
            [stage["selected_candidate_id"] for stage in first["stages"]],
            [f"stage{index}_winner" for index in range(1, 5)],
        )
        self.assertTrue(
            any(
                candidate["status"] == "ineligible_parent"
                for candidate in first["stages"][1]["candidates"]
            )
        )

    def test_selection_requires_best_checkpoint_mode(self):
        from chapter5_selection import select_stages

        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._write_workflow(Path(tmp), invalid_best=True)
            with self.assertRaisesRegex(ValueError, "checkpoint.selection.mode.*best"):
                select_stages(manifest)


class FrozenConfigTest(unittest.TestCase):
    def _workflow(self, root):
        builder = DeterministicSelectionTest()
        return builder._write_workflow(root)

    def test_cli_writes_deterministic_summary_and_runnable_frozen_config(self):
        from chapter5_selection import canonical_sha256, main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._workflow(root)
            first_output = root / "selection_artifacts_a"
            second_output = root / "selection_artifacts_b"

            self.assertEqual(
                main(["--manifest", str(manifest), "--output-dir", str(first_output)]),
                0,
            )
            self.assertEqual(
                main(["--manifest", str(manifest), "--output-dir", str(second_output)]),
                0,
            )

            first_summary_bytes = (first_output / "selection_summary.json").read_bytes()
            second_summary_bytes = (second_output / "selection_summary.json").read_bytes()
            first_config_bytes = (first_output / "selected_config.json").read_bytes()
            second_config_bytes = (second_output / "selected_config.json").read_bytes()
            frozen = json.loads(first_config_bytes)
            summary = json.loads(first_summary_bytes)
            args = parse_runner_args(
                ["--config", str(first_output / "selected_config.json")]
            )

        self.assertEqual(first_summary_bytes, second_summary_bytes)
        self.assertEqual(first_config_bytes, second_config_bytes)
        self.assertEqual(args.evaluation_split, "test")
        self.assertTrue(args.use_best_checkpoint)
        self.assertEqual(
            frozen["runner"]["downstream"]["evaluation_split"],
            "test",
        )
        self.assertEqual(
            frozen["runner"]["checkpoint"]["selection"]["mode"],
            "best",
        )
        experiment_sections = {
            key: frozen[key]
            for key in ("common", "runner", "analysis")
            if key in frozen
        }
        self.assertEqual(
            frozen["provenance"]["experiment_config_sha256"],
            canonical_sha256(experiment_sections),
        )
        self.assertEqual(summary["metric_split"], "validation")
        self.assertNotIn("test_mse", first_summary_bytes.decode("utf-8").lower())

    def test_cli_rejects_selection_output_inside_validation_results(self):
        from chapter5_selection import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._workflow(root)
            nested_output = (
                root / "validation" / "stage1_winner" / "random" / "selection"
            )
            with self.assertRaisesRegex(ValueError, "separate from validation"):
                main(
                    [
                        "--manifest",
                        str(manifest),
                        "--output-dir",
                        str(nested_output),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
