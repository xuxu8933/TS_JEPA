import sys
import tempfile
import unittest
from pathlib import Path

# Re-export the historical helpers for existing test/plot callers.
from src.stage00_legacy import (
    REPO_ROOT,
    PRETRAIN_SCRIPT,
    EVAL_SCRIPT,
    LR,
    EMA,
    MASK_RATIO,
    RATIO_PATCHES,
    ENCODER_EMBED,
    ENCODER_NHEAD,
    ENCODER_LAYERS,
    PREDICTOR_EMBED,
    PREDICTOR_NHEAD,
    PREDICTOR_LAYERS,
    LAMBDA_JEPA,
    LAMBDA_MAE,
    _write_rows,
    _dated_rows,
    _sin_cos_rows,
    _load_forecast_loaders,
    _target_context_values,
    _naive_last_rmse,
    _naive_last_predictions,
    _fit_downstream_and_predict,
    _model_rmse_after_downstream_fit,
    _run_command,
    _pretrain_smoke_case,
    _pretrain_mnist_row_case,
    _eval_mnist_row_case,
)


class DualLossSmokeTest(unittest.TestCase):
    def _run(self, cmd, cwd):
        try:
            return _run_command(cmd, cwd)
        except Exception as exc:
            self.fail(str(exc))

    def _pretrain(self, workdir, data_name, rows):
        return _pretrain_smoke_case(workdir, data_name, rows)

    def _assert_checkpoint_payload(self, checkpoint_path, data_name, epoch=0):
        import torch

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(checkpoint["strategy"], "dual_jepa_mae")
        self.assertEqual(checkpoint["epoch"], epoch)
        self.assertEqual(checkpoint["config"]["data"], data_name)
        self.assertIn("encoder", checkpoint)
        self.assertIn("predictor", checkpoint)
        self.assertIn("decoder", checkpoint)

    def _assert_eval_dry_run(self, workdir, data_name, checkpoint_path):
        output = self._run(
            [
                sys.executable,
                str(EVAL_SCRIPT),
                "--data",
                data_name,
                "--checkpoint-dir",
                str(workdir / "logs" / "output_model"),
                "--checkpoint-to-use",
                "0",
                "--lr-pretrain",
                LR,
                "--ema-pretrain",
                EMA,
                "--mask-ratio",
                MASK_RATIO,
                "--ratio-patches",
                RATIO_PATCHES,
                "--pretrain-encoder-embed-dim",
                ENCODER_EMBED,
                "--pretrain-encoder-nhead",
                ENCODER_NHEAD,
                "--pretrain-encoder-num-layers",
                ENCODER_LAYERS,
                "--pretrain-decoder-embed-dim",
                PREDICTOR_EMBED,
                "--pretrain-decoder-nhead",
                PREDICTOR_NHEAD,
                "--pretrain-decoder-num-layers",
                PREDICTOR_LAYERS,
                "--lambda-jepa",
                LAMBDA_JEPA,
                "--lambda-mae",
                LAMBDA_MAE,
                "--dry-run",
            ],
            cwd=workdir,
        )
        self.assertIn(str(checkpoint_path), output)
        self.assertIn("Delegated argv:", output)
        self.assertIn("--pretrain_checkpoint_path", output)

    def _assert_better_than_naive_last(self, checkpoint_path, data_path):
        model_rmse, naive_rmse = _model_rmse_after_downstream_fit(
            checkpoint_path=checkpoint_path,
            data_path=data_path,
        )
        print(
            f"{checkpoint_path.parent.name}: "
            f"model_rmse={model_rmse:.6f}, naive_last_rmse={naive_rmse:.6f}"
        )
        self.assertLess(
            model_rmse,
            naive_rmse,
            f"model_rmse={model_rmse:.6f}, naive_last_rmse={naive_rmse:.6f}",
        )

    def _run_case(self, data_name, rows):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            data_path = workdir / "data" / data_name / f"{data_name}.csv"
            checkpoint_path, pretrain_output = self._pretrain(workdir, data_name, rows)
            self.assertIn("JEPA:", pretrain_output)
            self.assertIn("MAE:", pretrain_output)
            self._assert_checkpoint_payload(checkpoint_path, data_name)
            self._assert_eval_dry_run(workdir, data_name, checkpoint_path)
            self._assert_better_than_naive_last(checkpoint_path, data_path)

    def test_sin_cos_artifact_dual_loss_pretrain_and_eval_path(self):
        self._run_case("SMOKE_SIN_COS", _sin_cos_rows())

    def test_real_mnist_row_wise_dual_loss_pretrain_and_eval(self):
        mnist_root = REPO_ROOT / "data" / "MNIST"
        if not (mnist_root / "MNIST" / "raw").exists():
            self.skipTest(
                "MNIST is not cached; run pretrain_dual_loss.py with --download-mnist once"
            )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            checkpoint_path, pretrain_output = _pretrain_mnist_row_case(
                workdir=workdir,
                mnist_root=mnist_root,
            )
            self.assertIn("input_mode = mnist_rows", pretrain_output)
            self.assertIn("num_patches = 28", pretrain_output)
            self.assertIn("patch_dim = 28", pretrain_output)
            self._assert_checkpoint_payload(
                checkpoint_path,
                "SMOKE_MNIST_ROWS",
                epoch=29,
            )

            result = _eval_mnist_row_case(
                workdir=workdir,
                checkpoint_path=checkpoint_path,
                mnist_root=mnist_root,
            )
            print(
                "SMOKE_MNIST_ROWS: "
                f"model_rmse={result['model_rmse']:.6f}, "
                f"naive_previous_row_rmse={result['naive_rmse']:.6f}"
            )
            self.assertLess(result["model_rmse"], result["naive_rmse"])


if __name__ == "__main__":
    unittest.main()
