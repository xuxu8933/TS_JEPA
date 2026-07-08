import csv
import math
import re
import subprocess
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_SCRIPT = REPO_ROOT / "pretrain_dual_loss.py"
EVAL_SCRIPT = REPO_ROOT / "eval_dual_loss.py"

LR = "0.001"
EMA = "0.9"
MASK_RATIO = "0.4"
RATIO_PATCHES = "5"
ENCODER_EMBED = "16"
ENCODER_NHEAD = "2"
ENCODER_LAYERS = "1"
PREDICTOR_EMBED = "8"
PREDICTOR_NHEAD = "2"
PREDICTOR_LAYERS = "1"
LAMBDA_JEPA = "0.7"
LAMBDA_MAE = "0.3"


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "Close", "Volume"])
        writer.writeheader()
        writer.writerows(rows)


def _dated_rows(close_values, volume_values):
    start = date(2021, 1, 1)
    rows = []
    for i, (close, volume) in enumerate(zip(close_values, volume_values)):
        rows.append(
            {
                "Date": (start + timedelta(days=i)).isoformat(),
                "Close": f"{close:.6f}",
                "Volume": f"{volume:.6f}",
            }
        )
    return rows


def _sin_cos_rows(n=260):
    closes = []
    volumes = []
    for i in range(n):
        close = (
            20.0
            + 1.8 * math.sin(2.0 * math.pi * i / 16.0)
            + 1.1 * math.cos(2.0 * math.pi * i / 7.0)
            + 0.002 * i
        )
        volume = (
            1000.0
            + 85.0 * math.cos(2.0 * math.pi * i / 11.0)
            + 25.0 * math.sin(2.0 * math.pi * i / 5.0)
        )
        closes.append(close)
        volumes.append(volume)
    return _dated_rows(closes, volumes)


def _seven_segment_digit(digit):
    segments_by_digit = {
        0: "abcfed",
        1: "bc",
        2: "abged",
        3: "abgcd",
        4: "fgbc",
        5: "afgcd",
        6: "afgecd",
        7: "abc",
        8: "abcdefg",
        9: "abfgcd",
    }
    segments = set(segments_by_digit[digit])
    grid = [[0.0 for _ in range(28)] for _ in range(28)]

    def h(row, c0=6, c1=22):
        for rr in range(row - 1, row + 2):
            for cc in range(c0, c1):
                grid[rr][cc] = 1.0

    def v(col, r0, r1):
        for rr in range(r0, r1):
            for cc in range(col - 1, col + 2):
                grid[rr][cc] = 1.0

    if "a" in segments:
        h(5)
    if "b" in segments:
        v(22, 6, 14)
    if "c" in segments:
        v(22, 15, 24)
    if "d" in segments:
        h(24)
    if "e" in segments:
        v(6, 15, 24)
    if "f" in segments:
        v(6, 6, 14)
    if "g" in segments:
        h(14)

    values = []
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            halo = 0.20 if value == 0.0 and (r + c + digit) % 29 == 0 else 0.0
            values.append(max(value, halo))
    return values


def _mnist_artifact_rows(n=260):
    digit_signatures = []
    for digit in range(10):
        pixels = _seven_segment_digit(digit)
        weighted_sum = sum((idx % 31 + 1) * value for idx, value in enumerate(pixels))
        digit_signatures.append(weighted_sum / 31.0)

    max_signature = max(digit_signatures)
    digit_signatures = [value / max_signature for value in digit_signatures]

    closes = []
    volumes = []
    for i in range(n):
        digit = (i // 4) % 10
        next_digit = ((i // 4) + 1) % 10
        phase = i % 4
        pixel = digit_signatures[digit]
        neighbor = digit_signatures[next_digit]
        closes.append(12.0 + 2.6 * pixel + 0.7 * neighbor + 0.12 * phase)
        volumes.append(800.0 + 150.0 * neighbor + 12.0 * phase)
    return _dated_rows(closes, volumes)


def _load_forecast_loaders(data_path):
    from src.data_loaders.data_loader_roll_volume import get_evaluation_loaders

    loader_kwargs = {
        "path_data": str(data_path),
        "batch_size": 16,
        "ratio_patches": int(RATIO_PATCHES),
        "mask_ratio": float(MASK_RATIO),
        "patch_size": 4,
        "context_size": 5,
        "stride": 1,
        "normalize": True,
        "feature_cols": ("Close", "Volume"),
        "sentiment_path": None,
        "validation_fraction": 0.1,
        "test_fraction": 0.3,
        "train_end_date": None,
        "test_start_date": None,
    }

    train_loader = get_evaluation_loaders(split="train", **loader_kwargs)
    test_loader = get_evaluation_loaders(split="test", **loader_kwargs)
    return train_loader, test_loader


def _target_context_values(context_patches):
    return context_patches.reshape(context_patches.shape[0], 5, 4, 2)[:, :, :, 0].reshape(
        context_patches.shape[0],
        -1,
    )


def _naive_last_mse(loader):
    preds, targets = _naive_last_predictions(loader)
    return ((preds - targets) ** 2).mean().item()


def _naive_last_predictions(loader):
    import torch

    preds = []
    targets = []
    for context_patches, target_patch in loader:
        context_target = _target_context_values(context_patches)
        last_values = context_target[:, -1].unsqueeze(-1)
        pred = last_values.repeat(1, target_patch.shape[-1])
        preds.append(pred)
        targets.append(target_patch)
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)


def _fit_downstream_and_predict(checkpoint_path, data_path):
    import torch

    from main.utils import init_weights
    from src.models.decoder import MLPDecoder
    from src.models.encoder import Encoder

    torch.manual_seed(123)
    train_loader, test_loader = _load_forecast_loaders(data_path)

    sample_context, _ = train_loader.dataset[0]
    encoder = Encoder(
        num_patches=sample_context.shape[0],
        dim_in=sample_context.shape[1],
        kernel_size=3,
        embed_dim=int(ENCODER_EMBED),
        embed_bias=True,
        nhead=int(ENCODER_NHEAD),
        num_layers=int(ENCODER_LAYERS),
        jepa=True,
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    encoder.load_state_dict(checkpoint["encoder"])

    decoder = MLPDecoder(
        emb_dim=int(ENCODER_EMBED),
        patch_size=4,
        hidden_dim=64,
        num_layers=2,
        dropout=0.0,
    )
    for module in decoder.modules():
        init_weights(module)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=3e-3,
    )

    for _ in range(120):
        encoder.train()
        decoder.train()
        for context_patches, target_patch in train_loader:
            optimizer.zero_grad()
            encoded_patches = encoder(context_patches)
            predicted_patch = decoder(encoded_patches.mean(dim=1))
            loss = torch.nn.functional.mse_loss(predicted_patch, target_patch)
            loss.backward()
            optimizer.step()

    model_preds = []
    model_targets = []
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for context_patches, target_patch in test_loader:
            encoded_patches = encoder(context_patches)
            predicted_patch = decoder(encoded_patches.mean(dim=1))
            model_preds.append(predicted_patch)
            model_targets.append(target_patch)

    model_preds = torch.cat(model_preds, dim=0)
    model_targets = torch.cat(model_targets, dim=0)
    naive_preds, naive_targets = _naive_last_predictions(test_loader)

    model_mse = ((model_preds - model_targets) ** 2).mean().item()
    naive_mse = ((naive_preds - naive_targets) ** 2).mean().item()

    return {
        "model_mse": model_mse,
        "naive_mse": naive_mse,
        "model_preds": model_preds.detach().cpu().numpy(),
        "targets": model_targets.detach().cpu().numpy(),
        "naive_preds": naive_preds.detach().cpu().numpy(),
    }


def _model_mse_after_downstream_fit(checkpoint_path, data_path):
    result = _fit_downstream_and_predict(checkpoint_path, data_path)
    return result["model_mse"], result["naive_mse"]


def _run_command(cmd, cwd, timeout=120):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code "
            f"{result.returncode}:\n{' '.join(map(str, cmd))}\n\n{result.stdout}"
        )
    return result.stdout


def _pretrain_smoke_case(workdir, data_name, rows, python_executable=None):
    python_executable = python_executable or sys.executable
    data_path = workdir / "data" / data_name / f"{data_name}.csv"
    _write_rows(data_path, rows)

    output = _run_command(
        [
            python_executable,
            str(PRETRAIN_SCRIPT),
            "--data",
            data_name,
            "--feature-cols",
            "Close",
            "Volume",
            "--sentiment-path",
            "none",
            "--train-end-date",
            "none",
            "--test-start-date",
            "none",
            "--validation-fraction",
            "0.1",
            "--test-fraction",
            "0.1",
            "--series-split-size",
            "20",
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
            LR,
            "--end-lr",
            LR,
            "--ema-momentum",
            EMA,
            "--mask-ratio",
            MASK_RATIO,
            "--ratio-patches",
            RATIO_PATCHES,
            "--encoder-embed-dim",
            ENCODER_EMBED,
            "--encoder-nhead",
            ENCODER_NHEAD,
            "--encoder-num-layers",
            ENCODER_LAYERS,
            "--encoder-kernel-size",
            "3",
            "--predictor-embed",
            PREDICTOR_EMBED,
            "--predictor-nhead",
            PREDICTOR_NHEAD,
            "--predictor-num-layers",
            PREDICTOR_LAYERS,
            "--lambda-jepa",
            LAMBDA_JEPA,
            "--lambda-mae",
            LAMBDA_MAE,
            "--decoder-type",
            "linear",
            "--seed",
            "7",
        ],
        cwd=workdir,
    )

    match = re.search(r"Saved checkpoint:\s*(.+\.pt)", output)
    if match is None:
        raise RuntimeError(f"Could not find checkpoint path in output:\n{output}")
    checkpoint_path = Path(match.group(1))
    if not checkpoint_path.is_absolute():
        checkpoint_path = workdir / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    return checkpoint_path, output


class DualLossSmokeTest(unittest.TestCase):
    def _run(self, cmd, cwd):
        try:
            return _run_command(cmd, cwd)
        except Exception as exc:
            self.fail(str(exc))

    def _pretrain(self, workdir, data_name, rows):
        return _pretrain_smoke_case(workdir, data_name, rows)

    def _assert_checkpoint_payload(self, checkpoint_path, data_name):
        import torch

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(checkpoint["strategy"], "dual_jepa_mae")
        self.assertEqual(checkpoint["epoch"], 0)
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
        model_mse, naive_mse = _model_mse_after_downstream_fit(
            checkpoint_path=checkpoint_path,
            data_path=data_path,
        )
        print(
            f"{checkpoint_path.parent.name}: "
            f"model_mse={model_mse:.6f}, naive_last_mse={naive_mse:.6f}"
        )
        self.assertLess(
            model_mse,
            naive_mse,
            f"model_mse={model_mse:.6f}, naive_last_mse={naive_mse:.6f}",
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

    def test_mnist_artifact_dual_loss_pretrain_and_eval_path(self):
        self._run_case("SMOKE_MNIST_ARTIFACT", _mnist_artifact_rows())


if __name__ == "__main__":
    unittest.main()
