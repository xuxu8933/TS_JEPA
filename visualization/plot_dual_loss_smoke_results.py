import csv
import importlib.util
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_HELPERS_PATH = REPO_ROOT / "tests" / "test_dual_loss_smoke.py"
OUTPUT_DIR = REPO_ROOT / "results" / "dual_loss_smoke"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_smoke_helpers():
    spec = importlib.util.spec_from_file_location(
        "dual_loss_smoke_helpers",
        TEST_HELPERS_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _flatten(x):
    return np.asarray(x).reshape(-1)


def _plot_case(case_name, result, output_path):
    target = _flatten(result["targets"])
    model = _flatten(result["model_preds"])
    naive = _flatten(result["naive_preds"])

    plt.figure(figsize=(13, 5))
    plt.plot(target, label="Target", linewidth=2)
    plt.plot(model, label="Dual-loss model", linewidth=2)
    plt.plot(naive, label="Naive last", linewidth=2, linestyle="--")
    plt.title(
        f"{case_name}: model MSE={result['model_mse']:.6f}, "
        f"naive-last MSE={result['naive_mse']:.6f}"
    )
    plt.xlabel("Flattened test forecast step")
    plt.ylabel("Normalized target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_summary(rows, output_path):
    names = [row["case"] for row in rows]
    model_mse = [float(row["model_mse"]) for row in rows]
    naive_mse = [float(row["naive_last_mse"]) for row in rows]

    x = np.arange(len(names))
    width = 0.36

    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width / 2, model_mse, width=width, label="Dual-loss model")
    plt.bar(x + width / 2, naive_mse, width=width, label="Naive last")
    plt.xticks(x, names)
    plt.ylabel("Test MSE")
    plt.title("Dual-loss Smoke Benchmark")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    helpers = _load_smoke_helpers()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = [
        ("SMOKE_MNIST_ARTIFACT", helpers._mnist_artifact_rows()),
        ("SMOKE_SIN_COS", helpers._sin_cos_rows()),
    ]

    summary_rows = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)

        for case_name, rows in cases:
            checkpoint_path, _ = helpers._pretrain_smoke_case(
                workdir=workdir,
                data_name=case_name,
                rows=rows,
                python_executable=sys.executable,
            )
            data_path = workdir / "data" / case_name / f"{case_name}.csv"
            result = helpers._fit_downstream_and_predict(
                checkpoint_path=checkpoint_path,
                data_path=data_path,
            )

            plot_path = OUTPUT_DIR / f"{case_name.lower()}_predictions.png"
            _plot_case(case_name, result, plot_path)

            summary_rows.append(
                {
                    "case": case_name,
                    "model_mse": f"{result['model_mse']:.10f}",
                    "naive_last_mse": f"{result['naive_mse']:.10f}",
                    "plot_path": str(plot_path),
                }
            )

            print(
                f"{case_name}: model_mse={result['model_mse']:.6f}, "
                f"naive_last_mse={result['naive_mse']:.6f}, plot={plot_path}"
            )

    summary_csv = OUTPUT_DIR / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "model_mse", "naive_last_mse", "plot_path"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_plot = OUTPUT_DIR / "mse_comparison.png"
    _plot_summary(summary_rows, summary_plot)

    print(f"Summary CSV: {summary_csv}")
    print(f"Summary plot: {summary_plot}")


if __name__ == "__main__":
    main()
