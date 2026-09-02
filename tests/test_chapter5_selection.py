import json
import copy
import tempfile
import unittest
from pathlib import Path

from config.file_options import read_config_file
from eval_dual_loss import build_eval_argv, parse_args as parse_eval_args
from eval_forecast_prequential_with_baselines_gru_volume import (
    build_downstream_metrics_artifact,
    choose_evaluation_loader,
    write_downstream_metrics_artifact,
)
from run_top_nasdaq100_stocks import (
    build_stock_commands,
    effective_experiment_config,
    experiment_config_signature,
    parse_args as parse_runner_args,
    resolve_seeds,
    resolve_stocks,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    REPO_ROOT / "config" / "experiments" / "normalization_pilot_window_return.json"
)
CANDIDATE_CONFIG_DIR = REPO_ROOT / "config" / "experiments" / "chapter5_candidates"


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


class StageOneCandidateConfigTest(unittest.TestCase):
    def test_normalization_candidates_change_only_normalization_method(self):
        paths = {
            "window_return": CANDIDATE_CONFIG_DIR
            / "01_preprocessing_window_return.json",
            "train_zscore": CANDIDATE_CONFIG_DIR
            / "01_preprocessing_train_zscore.json",
        }
        parsed = {
            method: parse_runner_args(["--config", str(path)])
            for method, path in paths.items()
        }

        for method, args in parsed.items():
            with self.subTest(method=method):
                self.assertEqual(
                    resolve_stocks(args),
                    ["NVDA", "AAPL", "AVGO", "TSLA", "WMT"],
                )
                self.assertEqual(resolve_seeds(args), [42, 44, 46])
                self.assertEqual(args.max_parallel_jobs, 2)
                self.assertEqual(args.mask_strategies, ["random"])
                self.assertEqual(args.lambda_jepa, 1.0)
                self.assertEqual(args.lambda_mae, 0.5)
                self.assertEqual(args.pretrain_num_epochs, 2001)
                self.assertEqual(args.eval_num_epochs, 501)
                self.assertEqual(args.forecast_horizon, 5)
                self.assertEqual(args.context_size, 12)
                self.assertEqual(args.evaluation_split, "validation")
                self.assertTrue(args.use_best_checkpoint)
                self.assertFalse(args.use_sentiment)
                self.assertEqual(args.normalization, method)
                self.assertEqual(
                    Path(args.results_dir),
                    REPO_ROOT / "results" / paths[method].stem,
                )

        effective = {
            method: effective_experiment_config(args)
            for method, args in parsed.items()
        }
        for options in effective.values():
            options.pop("normalization")
        self.assertEqual(effective["window_return"], effective["train_zscore"])


class CandidateMaterializerTest(unittest.TestCase):
    BASE = CANDIDATE_CONFIG_DIR / "01_preprocessing_window_return.json"

    def test_sentiment_candidates_are_deterministic_one_factor_derivations(self):
        from chapter5_prepare_candidates import materialize_candidates
        from chapter5_selection import canonical_sha256

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            paths = materialize_candidates(
                "sentiment",
                self.BASE,
                "preprocessing_window_return",
                output,
            )
            first_bytes = [path.read_bytes() for path in paths]
            repeated = materialize_candidates(
                "sentiment",
                self.BASE,
                "preprocessing_window_return",
                output,
            )
            repeated_bytes = [path.read_bytes() for path in repeated]
            payloads = [json.loads(content) for content in first_bytes]
            parsed = [parse_runner_args(["--config", str(path)]) for path in paths]

        self.assertEqual(
            [path.name for path in paths],
            ["02_sentiment_excluded.json", "02_sentiment_included.json"],
        )
        self.assertEqual(first_bytes, repeated_bytes)
        self.assertEqual(
            [
                payload["runner"]["preprocessing"]["custom"]["features"]
                ["sentiment"]["enabled"]
                for payload in payloads
            ],
            [False, True],
        )
        normalized = []
        for payload in payloads:
            candidate = copy.deepcopy(payload)
            candidate.pop("provenance")
            candidate["runner"]["preprocessing"]["custom"]["features"][
                "sentiment"
            ]["enabled"] = False
            normalized.append(candidate)
        self.assertEqual(normalized[0], normalized[1])

        _, base = read_config_file(self.BASE)
        for payload, args in zip(payloads, parsed):
            self.assertEqual(
                payload["provenance"]["parent_candidate_id"],
                "preprocessing_window_return",
            )
            self.assertEqual(
                payload["provenance"]["parent_config_sha256"],
                canonical_sha256(base),
            )
            self.assertEqual(
                resolve_stocks(args),
                ["NVDA", "AAPL", "AVGO", "TSLA", "WMT"],
            )
            self.assertEqual(resolve_seeds(args), [42, 44, 46])
            self.assertEqual(args.evaluation_split, "validation")
            self.assertTrue(args.use_best_checkpoint)

    def test_materializer_uses_selected_snapshot_source_hash_for_lineage(self):
        from chapter5_prepare_candidates import materialize_candidates
        from chapter5_selection import canonical_sha256

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, source = read_config_file(self.BASE)
            source_hash = canonical_sha256(source)
            snapshot = copy.deepcopy(source)
            snapshot["provenance"] = {
                "artifact_type": "selected_chapter5_stage_config",
                "schema_version": 1,
                "selected_candidate_id": "preprocessing_window_return",
                "source_config_sha256": source_hash,
                "metric_split": "validation",
            }
            snapshot_path = root / "selected_stage_config.json"
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

            paths = materialize_candidates(
                "sentiment",
                snapshot_path,
                "preprocessing_window_return",
                root / "candidates",
            )
            payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
            with self.assertRaisesRegex(ValueError, "must match"):
                materialize_candidates(
                    "sentiment",
                    snapshot_path,
                    "wrong_parent",
                    root / "wrong_parent",
                )

        self.assertNotEqual(canonical_sha256(snapshot), source_hash)
        self.assertEqual(
            {payload["provenance"]["parent_config_sha256"] for payload in payloads},
            {source_hash},
        )

    def test_materializer_rejects_unsafe_bases_and_overwrites(self):
        from chapter5_prepare_candidates import materialize_candidates

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = json.loads(self.BASE.read_text(encoding="utf-8"))

            test_base = root / "test_base.json"
            test_payload = copy.deepcopy(base)
            test_payload["runner"]["downstream"]["evaluation_split"] = "test"
            test_base.write_text(json.dumps(test_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation-only"):
                materialize_candidates("sentiment", test_base, "parent", root / "a")

            epoch_base = root / "epoch_base.json"
            epoch_payload = copy.deepcopy(base)
            epoch_payload["runner"]["checkpoint"]["selection"] = {
                "mode": "epoch",
                "epoch": 2000,
            }
            epoch_base.write_text(json.dumps(epoch_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mode=best"):
                materialize_candidates("sentiment", epoch_base, "parent", root / "b")

            with self.assertRaisesRegex(ValueError, "parent_candidate_id"):
                materialize_candidates("sentiment", self.BASE, "", root / "c")
            with self.assertRaisesRegex(ValueError, "stage"):
                materialize_candidates("unknown", self.BASE, "parent", root / "d")

            output = root / "overwrite"
            paths = materialize_candidates("sentiment", self.BASE, "parent", output)
            paths[0].write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                materialize_candidates("sentiment", self.BASE, "parent", output)

    def test_architecture_context_materializes_complete_fixed_objective_grid(self):
        from chapter5_prepare_candidates import materialize_candidates

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sentiment_paths = materialize_candidates(
                "sentiment",
                self.BASE,
                "preprocessing_window_return",
                root / "sentiment",
            )
            paths = materialize_candidates(
                "architecture_context",
                sentiment_paths[1],
                "sentiment_included",
                root / "grid",
            )
            parsed = [parse_runner_args(["--config", str(path)]) for path in paths]

        self.assertEqual(
            [path.name for path in paths],
            [
                "03_shared_context_6_patches.json",
                "03_shared_context_12_patches.json",
                "03_shared_context_24_patches.json",
                "03_local_long_context_6_patches.json",
                "03_local_long_context_12_patches.json",
                "03_local_long_context_24_patches.json",
            ],
        )
        self.assertEqual(
            [(args.mask_strategies, args.context_size) for args in parsed],
            [
                (["random"], 6),
                (["random"], 12),
                (["random"], 24),
                (["local_long"], 6),
                (["local_long"], 12),
                (["local_long"], 24),
            ],
        )
        for args in parsed:
            self.assertEqual(args.lambda_jepa, 1.0)
            self.assertEqual(args.lambda_mae, 0.5)
            self.assertEqual(args.series_split_size, 60)
            self.assertEqual(args.patch_size, 5)
            self.assertEqual(args.evaluation_split, "validation")
            self.assertTrue(args.use_best_checkpoint)
        for args in parsed[3:]:
            self.assertEqual(args.mae_window_patches, 1)
            self.assertEqual(args.jepa_gap_patches, 4)
            self.assertEqual(args.jepa_target_patches, 4)


class SelectionManifestTemplateTest(unittest.TestCase):
    def test_templates_are_canonical_stage_prefixes_with_complete_grid(self):
        paths = [
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
        manifests = [read_config_file(path)[1] for path in paths]

        self.assertEqual(
            [[stage["name"] for stage in manifest["stages"]] for manifest in manifests],
            [
                ["preprocessing_normalization"],
                ["preprocessing_normalization", "sentiment"],
                [
                    "preprocessing_normalization",
                    "sentiment",
                    "architecture_context",
                ],
            ],
        )
        self.assertEqual(
            [
                sum(len(stage["candidates"]) for stage in manifest["stages"])
                for manifest in manifests
            ],
            [2, 4, 10],
        )
        final_candidates = manifests[-1]["stages"][-1]["candidates"]
        self.assertEqual(
            [(item["id"], item["strategy"], Path(item["config"]).name) for item in final_candidates],
            [
                ("shared_context_6", "random", "03_shared_context_6_patches.json"),
                ("shared_context_12", "random", "03_shared_context_12_patches.json"),
                ("shared_context_24", "random", "03_shared_context_24_patches.json"),
                (
                    "local_long_context_6",
                    "local_long",
                    "03_local_long_context_6_patches.json",
                ),
                (
                    "local_long_context_12",
                    "local_long",
                    "03_local_long_context_12_patches.json",
                ),
                (
                    "local_long_context_24",
                    "local_long",
                    "03_local_long_context_24_patches.json",
                ),
            ],
        )


class DownstreamSplitTest(unittest.TestCase):
    IDENTITY = {
        "config_signature": "a" * 64,
        "stock": "NVDA",
        "seed": 42,
        "strategy": "random",
        "metrics": {
            "rmse": 0.125,
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
                "rmse": 0.1,
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
            {"rmse": 0.1, "direction_accuracy": 0.6},
        )

    def test_selector_rejects_test_split_even_when_renamed(self):
        from chapter5_selection import load_validation_artifact

        self._write_artifact(split="test")
        with self.assertRaisesRegex(ValueError, "validation-only"):
            load_validation_artifact(self.path, self.identity)

    def test_selector_rejects_nested_test_metric_keys(self):
        from chapter5_selection import load_validation_artifact

        self._write_artifact(extra={"provenance": {"test_rmse": 0.0}})
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
                "rmse": float("nan"),
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
        "architecture_context",
    )

    def test_aggregation_means_seeds_within_stock_then_stocks(self):
        from chapter5_selection import aggregate_candidate

        summary = aggregate_candidate(
            {
                "AAPL": {
                    42: {"rmse": 0.0, "direction_accuracy": 0.4},
                    43: {"rmse": 2.0, "direction_accuracy": 0.6},
                },
                "NVDA": {
                    42: {"rmse": 100.0, "direction_accuracy": 0.8},
                    43: {"rmse": 100.0, "direction_accuracy": 1.0},
                },
            }
        )

        self.assertEqual(summary["per_stock"]["AAPL"]["rmse"], 1.0)
        self.assertEqual(summary["per_stock"]["NVDA"]["rmse"], 100.0)
        self.assertEqual(summary["overall"]["rmse"], 50.5)
        self.assertEqual(summary["overall"]["direction_accuracy"], 0.7)

    def _write_candidate(
        self,
        root,
        candidate_id,
        rmse,
        *,
        parent=None,
        checkpoint_mode="best",
    ):
        from chapter5_selection import canonical_sha256
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
        if parent is not None:
            parent_payload = json.loads(
                (root / f"{parent}.json").read_text(encoding="utf-8")
            )
            payload["provenance"] = {
                "artifact_type": "chapter5_candidate_config",
                "schema_version": 1,
                "stage": "fixture",
                "candidate_id": candidate_id,
                "candidate_filename": f"{candidate_id}.json",
                "parent_candidate_id": parent,
                "parent_config_sha256": canonical_sha256(parent_payload),
                "delta": {},
            }
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
                "rmse": rmse,
                "direction_accuracy": 1.0 - rmse / 10.0,
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

    def _write_workflow(
        self,
        root,
        *,
        reverse=False,
        invalid_best=False,
        stage_count=None,
    ):
        stages = []
        previous_winner = None
        selected_stage_names = self.STAGE_NAMES[:stage_count]
        for index, stage_name in enumerate(selected_stage_names):
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
            [
                f"stage{index}_winner"
                for index in range(1, len(self.STAGE_NAMES) + 1)
            ],
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

    def test_selection_rejects_removed_architecture_objective_stage(self):
        from chapter5_selection import select_stages

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_workflow(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["stages"].append(
                {
                    "name": "architecture_objective",
                    "candidates": [
                        self._write_candidate(
                            root,
                            "removed_architecture_candidate",
                            0.01,
                            parent="stage3_winner",
                        )
                    ],
                }
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Stage order"):
                select_stages(manifest_path)

    def test_selection_rejects_removed_historical_context_stage_name(self):
        from chapter5_selection import select_stages

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_workflow(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["stages"][-1]["name"] = "historical_context"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Stage order"):
                select_stages(manifest_path)

    def test_selection_rejects_candidate_with_wrong_parent_config_hash(self):
        from chapter5_selection import select_stages

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_workflow(root)
            candidate_path = root / "stage2_winner.json"
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            candidate["provenance"]["parent_config_sha256"] = "0" * 64
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "parent config hash mismatch"):
                select_stages(manifest_path)


class FrozenConfigTest(unittest.TestCase):
    def _workflow(self, root, *, stage_count=None):
        builder = DeterministicSelectionTest()
        return builder._write_workflow(root, stage_count=stage_count)

    def test_partial_cli_writes_validation_snapshot_without_test_config(self):
        from chapter5_selection import main

        for stage_count in (1, 2):
            with self.subTest(stage_count=stage_count), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest = self._workflow(root, stage_count=stage_count)
                output = root / "selection_artifacts"

                self.assertEqual(
                    main(["--manifest", str(manifest), "--output-dir", str(output)]),
                    0,
                )
                snapshot = json.loads(
                    (output / "selected_stage_config.json").read_text(encoding="utf-8")
                )
                summary = json.loads(
                    (output / "selection_summary.json").read_text(encoding="utf-8")
                )

                self.assertEqual(
                    snapshot["runner"]["downstream"]["evaluation_split"],
                    "validation",
                )
                self.assertEqual(
                    snapshot["provenance"]["selected_stage"],
                    DeterministicSelectionTest.STAGE_NAMES[stage_count - 1],
                )
                self.assertFalse(summary["complete"])
                self.assertFalse((output / "selected_config.json").exists())

    def test_freeze_rejects_partial_selection_summary(self):
        from chapter5_selection import freeze_selected_config, select_stages

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._workflow(root, stage_count=1)
            summary = select_stages(manifest)

            with self.assertRaisesRegex(ValueError, "complete three-stage"):
                freeze_selected_config(manifest, summary)

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
        self.assertTrue(summary["complete"])
        self.assertFalse((first_output / "selected_stage_config.json").exists())
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
