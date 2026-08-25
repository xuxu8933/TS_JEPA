import copy
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from config.config_pretrain import config as base_config
from eval_dual_loss import (
    build_eval_argv,
    dual_checkpoint_stem,
    parse_args as parse_eval_args,
    resolve_dual_checkpoint_path,
)
from eval_forecast_prequential_with_baselines_gru_volume import (
    compute_trend_accuracy,
    load_pretraining_checkpoint,
    load_pretrained_encoder_state,
    make_baseline_prediction,
)
from pretrain_dual_loss import (
    ema_momentum_at_step,
    initialize_models,
    make_strategy_masks,
    parse_args,
    restore_training_state,
    save_checkpoint,
)
from plot_top_stock_metrics import latest_comparison_files, load_rows
from run_top_nasdaq100_stocks import (
    build_combined_plot_command,
    build_stock_commands,
)
from src.models.decoder import ResidualMLPDecoder
from src.models.encoder import Encoder
from src.models.tokenizer import TS_Tokenizer
from src.data_loaders.data_class_roll_volume import CSVDataLoader, EvaluationDataLoader
from tests.test_dual_loss_smoke import REPO_ROOT, _run_command, _sin_cos_rows, _write_rows


class UnifiedDualLossTest(unittest.TestCase):
    def _checkpoint_selector_args(self, checkpoint_dir, selection):
        args, passthrough = parse_eval_args(
            argv=[
                "--data",
                "NVDA",
                "--mask-strategy",
                "future_block",
                "--lambda-jepa",
                "1.0",
                "--lambda-mae",
                "0.5",
                "--future-target-patches",
                "4",
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--checkpoint-selection",
                selection,
            ]
        )
        self.assertEqual(passthrough, [])
        return args

    def test_checkpoint_selector_resolves_best_and_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            data_dir = checkpoint_dir / "NVDA"
            data_dir.mkdir(parents=True)

            best_args = self._checkpoint_selector_args(checkpoint_dir, "best")
            stem = dual_checkpoint_stem(best_args)
            best_path = data_dir / f"{stem}_cfg_abc123_best.pt"
            epoch_500 = data_dir / f"{stem}_cfg_abc123_epoch_500.pt"
            epoch_2000 = data_dir / f"{stem}_cfg_abc123_epoch_2000.pt"
            for path in (best_path, epoch_500, epoch_2000):
                path.touch()

            self.assertEqual(resolve_dual_checkpoint_path(best_args), str(best_path))

            last_args = self._checkpoint_selector_args(checkpoint_dir, "last")
            self.assertEqual(resolve_dual_checkpoint_path(last_args), str(epoch_2000))

    def test_checkpoint_selector_rejects_ambiguous_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            data_dir = checkpoint_dir / "NVDA"
            data_dir.mkdir(parents=True)
            args = self._checkpoint_selector_args(checkpoint_dir, "best")
            stem = dual_checkpoint_stem(args)
            (data_dir / f"{stem}_cfg_first_best.pt").touch()
            (data_dir / f"{stem}_cfg_second_best.pt").touch()

            with self.assertRaisesRegex(RuntimeError, "Multiple checkpoints match"):
                resolve_dual_checkpoint_path(args)

    def test_checkpoint_selector_path_requires_explicit_path(self):
        args = self._checkpoint_selector_args("./logs/output_model", "path")
        with self.assertRaisesRegex(ValueError, "requires pretrain_checkpoint_path"):
            resolve_dual_checkpoint_path(args)

    def test_stock_runner_uses_unified_entrypoints_for_both_strategies(self):
        args = SimpleNamespace(
            mask_strategies=["local_long"],
            mae_window_patches=1,
            jepa_gap_patches=4,
            jepa_target_patches=4,
            future_target_patches=4,
            causal_num_blocks=2,
            causal_block_patches=2,
            causal_block_gap_patches=1,
            pretrain_stride=5,
            sampling_mode="temporal_segments",
            normalization="window_return",
            seeds=[42],
            max_seeds=0,
            encoder_weights="ema",
            skip_pretrain=False,
            pretrain_num_epochs=3,
            lambda_jepa=1.0,
            lambda_mae=0.5,
            jepa_loss="mse",
            mae_loss="mse",
            checkpoint_to_use=2,
            use_best_checkpoint=False,
            eval_num_epochs=4,
            results_dir="./custom-results",
        )

        pretrain_command, eval_command = build_stock_commands(args, "NVDA")

        self.assertIn("pretrain_dual_loss.py", pretrain_command)
        self.assertIn("eval_dual_loss.py", eval_command)
        self.assertIn("local_long", pretrain_command)
        self.assertIn("local_long", eval_command)
        self.assertIn("--results-dir", eval_command)
        self.assertIn("custom-results/NVDA/seed_42", eval_command)
        self.assertIn("--pretrain-checkpoint-path", eval_command)
        self.assertIn("--seed", pretrain_command)
        self.assertIn("--seed", eval_command)
        self.assertEqual(
            eval_command[eval_command.index("--seed") + 1],
            "42",
        )
        self.assertIn("--sampling-mode", pretrain_command)
        self.assertIn("temporal_segments", pretrain_command)
        self.assertIn("--sampling-mode", eval_command)

        combined_command = build_combined_plot_command(args, ["NVDA", "MSFT"])
        self.assertIn("plot_top_stock_metrics.py", combined_command)
        self.assertEqual(combined_command[-2:], ["NVDA", "MSFT"])
        self.assertEqual(
            combined_command[combined_command.index("--output-prefix") + 1],
            "top_2_nasdaq100",
        )
        self.assertEqual(
            combined_command[combined_command.index("--figure-title") + 1],
            "top_2_nasdaq100",
        )
        self.assertEqual(
            combined_command[combined_command.index("--results-dir") + 1],
            "./custom-results",
        )

    def test_combined_plot_discovers_nested_stock_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            stock_dir = results_dir / "NVDA"
            stock_dir.mkdir(parents=True)
            txt_path = stock_dir / "last_model_comparison_20260101_000000.txt"
            txt_path.write_text("Data source: NVDA\n")
            txt_path.with_suffix(".csv").write_text(
                "model,mse,mae,trend_accuracy\nTS-JEPA,0.1,0.2,0.6\n"
            )

            latest = latest_comparison_files(results_dir)

        self.assertEqual(latest["NVDA"], [txt_path])

    def test_combined_plot_aggregates_multiple_seed_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            for seed, mse in ((7, 0.1), (17, 0.3)):
                stock_dir = results_dir / "NVDA" / f"seed_{seed}"
                stock_dir.mkdir(parents=True)
                txt_path = stock_dir / "last_model_comparison_20260101_000000.txt"
                txt_path.write_text("Data source: NVDA\n")
                txt_path.with_suffix(".csv").write_text(
                    "model,mse,mae,trend_accuracy\n"
                    f"TS-JEPA,{mse},0.2,0.6\n"
                )

            rows = load_rows(results_dir, ["NVDA"], ["TS-JEPA"])
            seed_7_rows = load_rows(
                results_dir,
                ["NVDA"],
                ["TS-JEPA"],
                seeds=[7],
            )

        self.assertEqual(int(rows.iloc[0]["num_runs"]), 2)
        self.assertAlmostEqual(float(rows.iloc[0]["mse"]), 0.2)
        self.assertGreater(float(rows.iloc[0]["mse_std"]), 0.0)
        self.assertEqual(int(seed_7_rows.iloc[0]["num_runs"]), 1)
        self.assertAlmostEqual(float(seed_7_rows.iloc[0]["mse"]), 0.1)

    def test_unified_parser_builds_strategy_specific_paths(self):
        default_config = parse_args(copy.deepcopy(base_config), argv=[])
        random_config = parse_args(
            copy.deepcopy(base_config),
            argv=["--mask-strategy", "random", "--seed", "7"],
        )
        local_config = parse_args(
            copy.deepcopy(base_config),
            argv=["--mask-strategy", "local_long", "--seed", "7"],
        )

        self.assertEqual(default_config["seed"], 42)
        self.assertEqual(default_config["mask_strategy"], "random")
        self.assertTrue(default_config["run_eval"])
        self.assertTrue(default_config["eval_use_best"])
        self.assertEqual(default_config["eval_forecast_target"], "relative_return")
        self.assertEqual(default_config["eval_num_epochs"], 501)
        self.assertEqual(default_config["data_end_date"], "2026-01-01")
        self.assertEqual(default_config["test_start_date"], "2025-01-01")
        self.assertNotIn("test_fraction", default_config)
        self.assertEqual(random_config["mask_strategy"], "random")
        self.assertIn("_dual_jepa_mae_", random_config["path_save"])
        self.assertEqual(local_config["mask_strategy"], "local_long")
        self.assertIn("_local_mae_long_jepa_", local_config["path_save"])
        self.assertEqual(random_config["seed"], 7)
        self.assertEqual(random_config["pretrain_stride"], random_config["patch_size"])
        configured_future = copy.deepcopy(base_config)
        configured_future["mask_strategy"] = "future_block"
        self.assertEqual(
            parse_args(configured_future, argv=[])["mask_strategy"],
            "future_block",
        )
        segmented_config = parse_args(
            copy.deepcopy(base_config),
            argv=[
                "--sampling-mode",
                "temporal_segments",
                "--pretrain-stride",
                "1",
            ],
        )
        self.assertEqual(segmented_config["sampling_mode"], "temporal_segments")
        self.assertEqual(
            segmented_config["pretrain_stride"],
            segmented_config["series_split_size"],
        )
        eval_args, passthrough = parse_eval_args(
            argv=["--sampling-mode", "temporal_segments"]
        )
        self.assertEqual(passthrough, [])
        self.assertEqual(eval_args.sampling_mode, "temporal_segments")
        other_seed_config = parse_args(
            copy.deepcopy(base_config),
            argv=["--mask-strategy", "random", "--seed", "17"],
        )
        self.assertNotEqual(
            random_config["config_fingerprint"],
            other_seed_config["config_fingerprint"],
        )
        no_eval_config = parse_args(
            copy.deepcopy(base_config),
            argv=["--no-run-eval", "--no-eval-use-best"],
        )
        self.assertFalse(no_eval_config["run_eval"])
        self.assertFalse(no_eval_config["eval_use_best"])

    def test_local_long_masks_are_causal_disjoint_and_ordered(self):
        config = {
            "mae_window_patches": 2,
            "jepa_gap_patches": 5,
            "jepa_target_patches": 3,
            "anchor_strategy": "fixed",
            "fixed_anchor": 2,
        }
        masks = make_strategy_masks(
            config=config,
            batch_size=3,
            num_patches=12,
            device=torch.device("cpu"),
        )

        self.assertEqual(masks["anchor"], 2)
        self.assertEqual(masks["mae"][0].tolist(), [2, 3])
        self.assertEqual(masks["jepa"][0].tolist(), [7, 8, 9])
        self.assertEqual(masks["predict"][0].tolist(), [2, 3, 7, 8, 9])
        self.assertEqual(masks["context"][0].tolist(), [0, 1, 4, 5, 6])
        self.assertTrue(torch.equal(masks["context"][0], masks["context"][2]))

    def test_future_and_multiblock_masks_use_only_past_context(self):
        future_config = {
            "mask_strategy": "future_block",
            "future_target_patches": 3,
            "anchor_strategy": "fixed",
            "fixed_anchor": 4,
        }
        future = make_strategy_masks(
            future_config,
            batch_size=2,
            num_patches=12,
            device=torch.device("cpu"),
        )
        self.assertEqual(future["context"][0].tolist(), [0, 1, 2, 3])
        self.assertEqual(future["jepa"][0].tolist(), [4, 5, 6])
        self.assertLess(future["context"].max(), future["jepa"].min())

        multiblock_config = {
            "mask_strategy": "causal_multiblock",
            "causal_num_blocks": 2,
            "causal_block_patches": 2,
            "causal_block_gap_patches": 1,
            "anchor_strategy": "fixed",
            "fixed_anchor": 3,
        }
        multiblock = make_strategy_masks(
            multiblock_config,
            batch_size=2,
            num_patches=12,
            device=torch.device("cpu"),
        )
        self.assertEqual(multiblock["context"][0].tolist(), [0, 1, 2])
        self.assertEqual(multiblock["jepa"][0].tolist(), [3, 4, 6, 7])
        self.assertLess(multiblock["context"].max(), multiblock["jepa"].min())

    def test_train_zscore_and_pretrain_stride_use_train_split_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "data" / "NORMALIZE" / "NORMALIZE.csv"
            _write_rows(data_path, _sin_cos_rows(180))
            dataset = CSVDataLoader(
                path_data=str(data_path),
                series_split_size=20,
                patch_size=5,
                stride=5,
                normalization="train_zscore",
                feature_cols=("Close", "Volume"),
                sentiment_path=None,
                validation_fraction=0.25,
                test_start_date="2021-06-01",
            )
            val_dataset = CSVDataLoader(
                path_data=str(data_path),
                series_split_size=20,
                patch_size=5,
                stride=5,
                normalization="train_zscore",
                normalization_stats=dataset.normalization_stats,
                split="val",
                mask_seed=10_000,
                feature_cols=("Close", "Volume"),
                sentiment_path=None,
                validation_fraction=0.25,
                test_start_date="2021-06-01",
            )

            expected_windows = (len(dataset.train_df) - 20) // 5 + 1
            self.assertEqual(dataset.stride, 5)
            self.assertEqual(len(dataset), expected_windows)
            self.assertTrue(
                torch.allclose(
                    torch.tensor(dataset.normalization_stats["mean"]),
                    dataset.train_df.mean(dim=0),
                )
            )
            self.assertEqual(
                val_dataset.normalization_stats,
                dataset.normalization_stats,
            )

    def test_time_series_split_requires_test_start_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "data" / "REQUIRED_DATE" / "REQUIRED_DATE.csv"
            _write_rows(data_path, _sin_cos_rows(180))

            with self.assertRaisesRegex(ValueError, "test_start_date must be defined"):
                CSVDataLoader(
                    path_data=str(data_path),
                    series_split_size=20,
                    patch_size=5,
                    normalization="none",
                    feature_cols=("Close", "Volume"),
                    sentiment_path=None,
                )

    def test_data_end_date_caps_pretraining_and_evaluation_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "data" / "CUTOFF" / "CUTOFF.csv"
            _write_rows(data_path, _sin_cos_rows(220))
            loader_kwargs = {
                "path_data": str(data_path),
                "feature_cols": ("Close", "Volume"),
                "sentiment_path": None,
                "train_end_date": "2021-04-30",
                "test_start_date": "2021-05-01",
                "data_end_date": "2021-06-15",
                "validation_fraction": 0.1,
            }

            pretrain_dataset = CSVDataLoader(
                series_split_size=20,
                patch_size=5,
                stride=5,
                normalization="none",
                **loader_kwargs,
            )
            evaluation_dataset = EvaluationDataLoader(
                patch_size=5,
                context_size=2,
                stride=5,
                split="test",
                normalization="none",
                **loader_kwargs,
            )

            # May 1 through June 15 is 46 calendar-daily observations,
            # including the configured end date and excluding all later rows.
            self.assertEqual(len(pretrain_dataset.test_df), 46)
            self.assertEqual(len(evaluation_dataset.test_df), 46)
            self.assertEqual(len(evaluation_dataset.series), 46)

    def test_temporal_segments_are_non_overlapping_and_keep_tensor_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "data" / "SEGMENTS" / "SEGMENTS.csv"
            _write_rows(data_path, _sin_cos_rows(210))

            pretrain_dataset = CSVDataLoader(
                path_data=str(data_path),
                series_split_size=20,
                patch_size=5,
                stride=1,
                sampling_mode="temporal_segments",
                normalization="none",
                feature_cols=("Close", "Volume"),
                sentiment_path=None,
                validation_fraction=0.1,
                test_start_date="2021-06-01",
            )

            expected_starts = list(range(0, len(pretrain_dataset.train_df) - 19, 20))
            self.assertEqual(pretrain_dataset.sample_starts, expected_starts)
            self.assertEqual(pretrain_dataset.stride, 20)
            self.assertEqual(len(pretrain_dataset), len(expected_starts))
            self.assertTrue(
                torch.equal(
                    pretrain_dataset.split_series[1],
                    pretrain_dataset.time_series[20:40],
                )
            )
            patches, _, _ = pretrain_dataset[0]
            self.assertEqual(tuple(patches.shape), (4, 10))

            evaluation_dataset = EvaluationDataLoader(
                path_data=str(data_path),
                patch_size=5,
                context_size=2,
                stride=1,
                sampling_mode="temporal_segments",
                split="train",
                normalization="none",
                feature_cols=("Close", "Volume"),
                sentiment_path=None,
                validation_fraction=0.1,
                test_start_date="2021-06-01",
            )

            expected_eval_starts = list(
                range(0, len(evaluation_dataset.train_df) - 14, 15)
            )
            self.assertEqual(evaluation_dataset.sample_starts, expected_eval_starts)
            self.assertEqual(evaluation_dataset.stride, 15)
            self.assertEqual(
                evaluation_dataset.indices,
                [start + 10 for start in expected_eval_starts],
            )
            context, target = evaluation_dataset[0]
            self.assertEqual(tuple(context.shape), (2, 10))
            self.assertEqual(tuple(target.shape), (5,))

    def test_relative_return_target_is_anchored_at_forecast_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "data" / "RETURNS" / "RETURNS.csv"
            _write_rows(data_path, _sin_cos_rows(180))
            dataset = EvaluationDataLoader(
                path_data=str(data_path),
                patch_size=4,
                context_size=2,
                stride=4,
                split="train",
                normalization="train_zscore",
                forecast_target="relative_return",
                feature_cols=("Close", "Volume"),
                sentiment_path=None,
                validation_fraction=0.1,
                test_start_date="2021-06-01",
            )

            raw_context, raw_target = dataset.samples[0]
            _, target = dataset[0]
            expected = raw_target[:, 0] / raw_context[-1, 0] - 1.0

            self.assertTrue(torch.allclose(target, expected, atol=1e-7))
            self.assertEqual(dataset.forecast_target, "relative_return")

    def test_relative_return_baselines_and_first_move_trend(self):
        context = torch.tensor([[100.0, 102.0], [101.0, 103.0]])
        baseline_config = {
            "feature_dim": 1,
            "patch_size": 2,
            "target_feature_index": 0,
            "normalization": "none",
            "forecast_target": "relative_return",
        }

        naive = make_baseline_prediction(
            context, horizon=2, baseline_name="naive_last", config=baseline_config
        )
        drift = make_baseline_prediction(
            context, horizon=2, baseline_name="drift", config=baseline_config
        )
        self.assertTrue(np.allclose(naive, [0.0, 0.0]))
        self.assertTrue(
            np.allclose(drift, np.array([104.0, 105.0]) / 103.0 - 1.0)
        )

        preds = np.array([[0.1, 0.2]], dtype=np.float32)
        targets = np.array([[-0.1, 0.2]], dtype=np.float32)
        self.assertEqual(compute_trend_accuracy(preds, targets), 1.0)
        self.assertEqual(
            compute_trend_accuracy(preds, targets, include_origin=True),
            0.5,
        )

    def test_unified_evaluator_forwards_relative_return_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            explicit_checkpoint = Path(tmp) / "not-created.pt"
            args, passthrough = parse_eval_args(
                argv=[
                    "--data",
                    "NVDA",
                    "--pretrain-checkpoint-path",
                    str(explicit_checkpoint),
                    "--forecast-target",
                    "relative_return",
                    "--data-end-date",
                    "2026-01-01",
                ]
            )
            eval_argv, _ = build_eval_argv(args, passthrough)

            target_arg = eval_argv.index("--forecast-target")
            self.assertEqual(eval_argv[target_arg + 1], "relative_return")
            cutoff_arg = eval_argv.index("--data_end_date")
            self.assertEqual(eval_argv[cutoff_arg + 1], "2026-01-01")

    def test_ema_schedule_advances_per_optimizer_step(self):
        values = [ema_momentum_at_step(0.9, step, 10) for step in range(11)]
        self.assertEqual(values[0], 0.9)
        self.assertAlmostEqual(values[-1], 1.0)
        self.assertTrue(all(left <= right for left, right in zip(values, values[1:])))

    def test_complete_checkpoint_restores_training_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            encoder = torch.nn.Linear(3, 3)
            encoder_ema = copy.deepcopy(encoder)
            predictor = torch.nn.Linear(3, 3)
            decoder = torch.nn.Linear(3, 3)
            optimizer = torch.optim.AdamW(
                list(encoder.parameters())
                + list(predictor.parameters())
                + list(decoder.parameters()),
                lr=0.01,
            )
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.5,
                total_iters=4,
            )
            checkpoint_path = save_checkpoint(
                encoder,
                predictor,
                decoder,
                str(Path(tmp) / "resume"),
                epoch=2,
                config={"mask_strategy": "random", "config_fingerprint": "abc"},
                encoder_ema=encoder_ema,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=7,
                best_validation_loss=0.25,
            )
            original_weight = encoder.weight.detach().clone()
            with torch.no_grad():
                encoder.weight.zero_()

            restored = restore_training_state(
                checkpoint_path,
                encoder,
                predictor,
                decoder,
                encoder_ema,
                optimizer,
                scheduler,
                device=torch.device("cpu"),
                expected_fingerprint="abc",
            )

            self.assertTrue(torch.equal(encoder.weight, original_weight))
            self.assertEqual(restored["start_epoch"], 3)
            self.assertEqual(restored["global_step"], 7)
            self.assertEqual(restored["best_validation_loss"], 0.25)

    def test_residual_decoder_keeps_zero_initialized_residual_branch(self):
        encoder = torch.nn.Sequential(torch.nn.Linear(4, 8))
        predictor = torch.nn.Sequential(torch.nn.Linear(8, 8))
        decoder = ResidualMLPDecoder(emb_dim=8, patch_size=4, hidden_dim=8)

        initialize_models(encoder, predictor, decoder)

        self.assertEqual(torch.count_nonzero(decoder.residual_head[-1].weight), 0)
        self.assertEqual(torch.count_nonzero(decoder.residual_head[-1].bias), 0)

    def test_tokenizer_honors_disabled_embedding_bias(self):
        tokenizer = TS_Tokenizer(
            dim_in=8,
            kernel_size=2,
            embed_dim=8,
            embed_bias=False,
        )
        self.assertIsNone(tokenizer.proj.bias)
        self.assertIsNone(tokenizer.fc.bias)

    def test_downstream_encoder_can_use_a_different_context_length(self):
        kwargs = {
            "dim_in": 8,
            "kernel_size": 2,
            "embed_dim": 8,
            "embed_bias": True,
            "nhead": 2,
            "num_layers": 1,
            "jepa": True,
        }
        pretrained = Encoder(num_patches=8, **kwargs)
        downstream = Encoder(num_patches=5, **kwargs)
        pretrained_state = pretrained.state_dict()

        load_pretrained_encoder_state(downstream, pretrained_state)

        self.assertEqual(tuple(downstream.pos_embed.shape), (1, 5, 8))
        self.assertTrue(
            torch.equal(
                downstream.tokenizer.fc.weight,
                pretrained.tokenizer.fc.weight,
            )
        )

    def test_downstream_loads_full_checkpoint_with_numpy_rng_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "full_state.pt"
            payload = {
                "encoder": {"weight": torch.ones(1)},
                "rng_state": {"numpy": np.random.get_state()},
            }
            torch.save(payload, checkpoint_path)

            loaded = load_pretraining_checkpoint(checkpoint_path, "cpu")

            self.assertIn("rng_state", loaded)
            self.assertEqual(loaded["rng_state"]["numpy"][0], "MT19937")

    def test_local_strategy_trains_and_saves_unified_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            data_name = "SMOKE_LOCAL_LONG"
            data_path = workdir / "data" / data_name / f"{data_name}.csv"
            _write_rows(data_path, _sin_cos_rows())

            output = _run_command(
                [
                    sys.executable,
                    str(REPO_ROOT / "pretrain_dual_loss.py"),
                    "--no-run-eval",
                    "--data",
                    data_name,
                    "--mask-strategy",
                    "local_long",
                    "--feature-cols",
                    "Close",
                    "Volume",
                    "--sentiment-path",
                    "none",
                    "--train-end-date",
                    "none",
                    "--test-start-date",
                    "2021-09-07",
                    "--validation-fraction",
                    "0.25",
                    "--series-split-size",
                    "40",
                    "--patch-size",
                    "4",
                    "--batch-size",
                    "2",
                    "--num-epochs",
                    "1",
                    "--max-batches-per-epoch",
                    "1",
                    "--checkpoint-save",
                    "99",
                    "--checkpoint-print",
                    "1",
                    "--lr",
                    "0.001",
                    "--end-lr",
                    "0.001",
                    "--ema-momentum",
                    "0.9",
                    "--mask-ratio",
                    "0.4",
                    "--encoder-embed-dim",
                    "16",
                    "--encoder-nhead",
                    "2",
                    "--encoder-num-layers",
                    "1",
                    "--encoder-kernel-size",
                    "3",
                    "--predictor-embed",
                    "8",
                    "--predictor-nhead",
                    "2",
                    "--predictor-num-layers",
                    "1",
                    "--mae-window-patches",
                    "1",
                    "--jepa-gap-patches",
                    "4",
                    "--jepa-target-patches",
                    "3",
                    "--anchor-strategy",
                    "fixed",
                    "--fixed-anchor",
                    "1",
                    "--seed",
                    "7",
                ],
                cwd=workdir,
            )

            match = re.search(r"Saved checkpoint:\s*(.+\.pt)", output)
            self.assertIsNotNone(match, output)
            checkpoint_path = workdir / match.group(1)
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(checkpoint["strategy"], "local_mae_long_jepa")
            self.assertEqual(checkpoint["config"]["mask_strategy"], "local_long")
            self.assertIn("encoder_ema", checkpoint)
            self.assertIn("optimizer", checkpoint)
            self.assertIn("scheduler", checkpoint)
            self.assertIn("global_step", checkpoint)
            self.assertIn("ema_schedule_steps", checkpoint)
            self.assertIn("rng_state", checkpoint)
            self.assertIn("avg_anchor: 1.00", output)
            self.assertIn("Validation epoch 0", output)
            self.assertIn("validation_history", checkpoint["config"])

            eval_output = _run_command(
                [
                    sys.executable,
                    str(REPO_ROOT / "eval_dual_loss.py"),
                    "--data",
                    data_name,
                    "--mask-strategy",
                    "local_long",
                    "--pretrain-checkpoint-path",
                    str(checkpoint_path),
                    "--dry-run",
                ],
                cwd=workdir,
            )
            self.assertIn("Local-MAE + long-JEPA checkpoint:", eval_output)
            self.assertIn("--patch_size 4", eval_output)
            self.assertIn("--feature_cols Close Volume", eval_output)


if __name__ == "__main__":
    unittest.main()
