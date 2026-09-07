"""Run the controlled Stage 00 comparison; see doc/stage00_comparison.md."""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from main.utils import set_seed
from src.data_loaders.data_loader_roll_volume import get_jepa_loaders
from src.stage00_baselines import METHODS as BASELINES, forecast, reconstruct_rows
from src.stage00_comparison import (
    DATASETS, METHODS, GRUModel, PretrainedModel, evaluate, fit,
    load_mnist_splits, load_sine_splits, summarize,
)
from src.stage00_legacy import (
    REPO_ROOT, _pretrain_mnist_row_case, _pretrain_smoke_case, _sin_cos_rows, _write_rows,
)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summaries(output, rows):
    summaries = summarize(rows)
    for dataset in DATASETS:
        selected = [row for row in summaries if row["dataset"] == dataset]
        if not selected:
            continue
        write_csv(output / f"summary_{dataset}.csv", selected)
        table = [r"\begin{tabular}{lrrr}", r"\hline",
                 r"Method & Mean RMSE & SD & Runs \\", r"\hline"]
        for row in selected:
            sd = "--" if row["std_rmse"] is None else f"{row['std_rmse']:.6f}"
            table.append(f"{row['model']} & {row['mean_rmse']:.6f} & {sd} & {row['num_runs']} " + r"\\")
        table.extend([r"\hline", r"\end{tabular}"])
        (output / f"table_{dataset}.tex").write_text("\n".join(table) + "\n")
    (output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        with (output / "raw_results.csv").open(newline="") as stream:
            write_summaries(output, list(csv.DictReader(stream)))
        return
    if (output / "manifest.json").exists() or (output / "raw_results.csv").exists():
        raise FileExistsError(f"Use a new output directory to preserve existing runs: {output}")
    seeds = args.seeds if args.seeds is not None else [args.seed if args.seed is not None else 7]
    if len(seeds) != len(set(seeds)) or any(not 0 <= seed < 2**32 - 10000 for seed in seeds):
        raise ValueError("Seeds must be distinct and in [0, 2**32 - 10000)")
    if len(set(args.models)) != len(args.models) or len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Models and datasets must not contain duplicates")
    manifest = {
        "argv": sys.argv, "cwd": str(Path.cwd()), "python_executable": sys.executable,
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "seeds": seeds, "downstream_seeds": [seed + 116 for seed in seeds],
        "datasets": args.datasets, "models": args.models,
        "environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")},
        "protocol": "Stage 00 controlled comparison v1; legacy reconstruction preserved; see doc/stage00_comparison.md",
        "source_sha256": {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [Path(__file__).resolve(), *sorted((REPO_ROOT / "src").glob("stage00_*.py")),
                                       REPO_ROOT / "pretrain_dual_loss.py", REPO_ROOT / "main/utils.py",
                                       REPO_ROOT / "eval_forecast_prequential_with_baselines_gru_volume.py",
                                       *sorted((REPO_ROOT / "config").glob("*.py")),
                                       *sorted((REPO_ROOT / "src/data_loaders").glob("*.py")),
                                       *sorted((REPO_ROOT / "src/models").rglob("*.py"))]},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    for dataset in args.datasets:
        for seed in seeds:
            group = output / dataset / f"seed_{seed}"
            group.mkdir(parents=True, exist_ok=True)
            set_seed(seed, deterministic=True)
            if dataset == "MNIST_ROWS":
                train, val, test, audit = load_mnist_splits(args.mnist_root, seed)
                epochs, lr = 30, 0.002
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                data_path = group / "data/SMOKE_SIN_COS/SMOKE_SIN_COS.csv"
                _write_rows(data_path, _sin_cos_rows())
                train, val, test, audit = load_sine_splits(data_path)
                # Check the actual SSL input split equals the supervised train
                # split. A July cutoff is necessary: legacy used September.
                ssl = get_jepa_loaders(path=str(data_path), batch_size=2, mask_ratio=0.4,
                                       series_split_size=20, patch_size=4, feature_cols=("Close", "Volume"),
                                       sentiment_path=None, validation_fraction=0.1, test_start_date="2021-07-01")
                assert torch.equal(ssl.dataset.time_series, train.dataset.train_df)
                audit["pretraining_equals_supervised_training_split"] = True
                audit["pretraining_windows"] = len(ssl.dataset)
                audit["pretraining_validation"] = "unavailable: 13 validation rows < 20-row SSL window; fixed one-step checkpoint"
                epochs, lr, device = 120, 0.003, "cpu"
            (group / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
            evaluation_identity = None
            for method in args.models:
                workdir = group / method
                workdir.mkdir()
                print(f"{dataset} {method} seed={seed}", flush=True)
                metadata = {"best_validation_loss": None, "final_validation_loss": None,
                            "selected_epoch": None, "selected_checkpoint": None,
                            "pretrain_checkpoint": None, "pretrain_best_validation_loss": None,
                            "pretrain_final_validation_loss": None, "lambda_jepa": None, "lambda_mae": None}
                if method in BASELINES:
                    if dataset == "MNIST_ROWS":
                        model = lambda x, m, v: reconstruct_rows(x, m, v, method)
                    else:
                        model = lambda x: forecast(x.reshape(-1, 20, 2)[:, :, :1], 4, method).squeeze(-1)
                    eval_device = "cpu"
                else:
                    if method != "GRU":
                        overrides = ["--seed", str(seed)]
                        if method != "JEPA-MAE":
                            weights = (1, 0) if method == "JEPA-only" else (0, 1)
                            overrides += ["--lambda-jepa", str(weights[0]), "--lambda-mae", str(weights[1])]
                        if dataset == "MNIST_ROWS":
                            checkpoint_path, _ = _pretrain_mnist_row_case(
                                workdir, args.mnist_root.resolve(), extra_args=overrides,
                                log_path=workdir / "pretrain.log")
                        else:
                            overrides += ["--test-start-date", "2021-07-01"]
                            checkpoint_path, _ = _pretrain_smoke_case(
                                workdir, "SMOKE_SIN_COS", _sin_cos_rows(), extra_args=overrides,
                                log_path=workdir / "pretrain.log")
                        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                        for state_key in ("encoder", "encoder_ema", "predictor", "decoder"):
                            assert all(torch.isfinite(t).all() for t in checkpoint[state_key].values())
                        config = checkpoint["config"]
                        metadata.update(pretrain_checkpoint=str(checkpoint_path),
                                        pretrain_best_validation_loss=checkpoint["best_validation_loss"],
                                        pretrain_final_validation_loss=(config["validation_history"][-1]["total_loss"]
                                                                        if config.get("validation_history") else None),
                                        lambda_jepa=config["lambda_jepa"], lambda_mae=config["lambda_mae"])
                    set_seed(seed + 116, deterministic=True)
                    model = GRUModel(dataset) if method == "GRU" else PretrainedModel(checkpoint, dataset)
                    metadata.update(fit(model, train, val, epochs=epochs, lr=lr,
                                        checkpoint=workdir / "downstream_best.pt", device=device))
                    eval_device = device
                # Final evaluation only, after checkpoint selection is complete.
                result = evaluate(model, test, eval_device)
                if evaluation_identity is None:
                    evaluation_identity = result["evaluation_sha256"]
                assert result["evaluation_sha256"] == evaluation_identity, "Methods received different test samples"
                prediction_path = workdir / "test_predictions.npz"
                np.savez(prediction_path, predictions=result["predictions"], targets=result["targets"])
                row = {"dataset": dataset, "model": method, "seed": seed, "downstream_seed": seed + 116,
                       "rmse": result["rmse"], **metadata, "num_test_values": result["num_values"],
                       "evaluation_sha256": evaluation_identity, "predictions_path": str(prediction_path),
                       "training_epochs": epochs if method not in BASELINES else 0,
                       "training_lr": lr if method not in BASELINES else None,
                       "training_batch_size": train.batch_size, "evaluation_device": eval_device}
                rows.append(row)
                write_csv(output / "raw_results.csv", rows)
                (output / "raw_results.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
                print(f"  TEST RMSE={result['rmse']:.9f}", flush=True)
                del model
            write_summaries(output, rows)
    print(f"Results: {output / 'raw_results.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/00_model_comparison")
    parser.add_argument("--mnist-root", type=Path, default=REPO_ROOT / "data/MNIST")
    parser.add_argument("--aggregate-only", action="store_true", help="Rebuild summaries and LaTeX from raw_results.csv")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
