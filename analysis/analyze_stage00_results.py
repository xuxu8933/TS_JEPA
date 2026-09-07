"""Validate and analyze Stage 00 outputs for the existing thesis publisher."""

import argparse
import hashlib
import itertools
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from run_stage00_comparison import write_csv, write_summaries
from src.stage00_comparison import DATASETS, METHODS, summarize


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_predictions(row):
    with np.load(row["predictions_path"]) as data:
        pred, target = data["predictions"], data["targets"]
    if pred.shape != target.shape or target.size != row["num_test_values"]:
        raise ValueError("Prediction/target shape or count mismatch")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite prediction or target")
    errors = (pred.astype(np.float64) - target.astype(np.float64)) ** 2
    # Independent float64 recomputation of the experiment's pooled float32 RMSE.
    if not math.isclose(float(np.sqrt(errors.mean())), row["rmse"], rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("Recorded RMSE does not match saved predictions")
    return errors, target


def analyze(results, repeat, output):
    rows, repeated = read_json(results / "raw_results.json"), read_json(repeat / "raw_results.json")
    manifest = read_json(results / "manifest.json")
    seeds = manifest["seeds"]
    if seeds != [7]:
        raise ValueError("This analysis describes the completed seed-7 experiment; update the interpretation for a new seed scope")
    expected = set(itertools.product(DATASETS, METHODS, seeds))
    key = lambda row: (row["dataset"], row["model"], row["seed"])
    if len(rows) != len(expected) or {key(r) for r in rows} != expected:
        raise ValueError("Incomplete or duplicated primary model/dataset/seed coverage")
    if len(repeated) != len(expected) or {key(r) for r in repeated} != expected:
        raise ValueError("Incomplete or duplicated repeat coverage")
    repeated = {key(row): row for row in repeated}
    sources = {}
    for name, digest in manifest["source_sha256"].items():
        path = REPO / name
        if sha256(path) != digest:
            raise ValueError(f"Source changed since the recorded run: {name}")
        sources[name] = {"sha256": digest, "text": path.read_text()}
    for path in (Path(__file__).resolve(), REPO / "publish_thesis_results.py"):
        sources[str(path.relative_to(REPO))] = {"sha256": sha256(path), "text": path.read_text()}

    errors, targets, identities = {}, {}, {}
    per_sample, horizons, paired, downstream, pretraining, artifacts = [], [], [], [], [], []
    for row in rows:
        identity = key(row)
        again = repeated[identity]
        error, target = load_predictions(row)
        _, repeated_target = load_predictions(again)
        for field in ("rmse", "best_validation_loss", "final_validation_loss", "selected_epoch", "evaluation_sha256"):
            if row[field] != again[field]:
                raise ValueError(f"Repeat mismatch: {identity}, {field}")
        with np.load(row["predictions_path"]) as first, np.load(again["predictions_path"]) as second:
            if not np.array_equal(first["predictions"], second["predictions"]):
                raise ValueError(f"Repeat predictions differ: {identity}")
        group = row["dataset"], row["seed"]
        targets.setdefault(group, target)
        identities.setdefault(group, row["evaluation_sha256"])
        if (not np.array_equal(target, targets[group]) or not np.array_equal(target, repeated_target)
                or identities[group] != row["evaluation_sha256"]):
            raise ValueError(f"Evaluation support differs: {identity}")
        errors[identity] = error.reshape(len(error), -1).mean(1)
        base = dict(zip(("dataset", "model", "seed"), identity))
        for idx, mse in enumerate(errors[identity]):
            per_sample.append({**base, "sample_index": idx, "mse": float(mse), "rmse": math.sqrt(mse)})
        if row["dataset"] == "SIN_COS":
            for horizon, score in enumerate(np.sqrt(error.mean(0)), 1):
                horizons.append({**base, "horizon": horizon, "rmse": float(score), "num_origins": len(error)})
        for field in ("predictions_path", "selected_checkpoint", "pretrain_checkpoint"):
            if row[field]:
                path = Path(row[field])
                artifacts.append({**base, "kind": field, "source": str(path.relative_to(REPO)),
                                  "bytes": path.stat().st_size, "sha256": sha256(path)})
        if row["selected_checkpoint"]:
            path = Path(row["selected_checkpoint"])
            history = read_json(path.with_suffix(".history.json"))
            if not all(math.isfinite(entry["validation_rmse"]) for entry in history):
                raise ValueError(f"Nonfinite validation history: {identity}")
            best = min(history, key=lambda item: item["validation_rmse"])
            state = torch.load(path, map_location="cpu", weights_only=False)
            if (best["epoch"] != row["selected_epoch"] or state["epoch"] != best["epoch"]
                    or best["validation_rmse"] != row["best_validation_loss"]
                    or state["validation_rmse"] != best["validation_rmse"]):
                raise ValueError(f"Validation selection mismatch: {identity}")
            if not all(torch.isfinite(value).all() for value in state["model"].values()):
                raise ValueError(f"Nonfinite checkpoint: {identity}")
            downstream.extend({**base, **entry} for entry in history)
        if row["pretrain_checkpoint"]:
            state = torch.load(row["pretrain_checkpoint"], map_location="cpu", weights_only=False)
            if row["dataset"] == "SIN_COS" and state["config"]["test_start_date"] != "2021-07-01":
                raise ValueError("Pretraining cutoff differs from downstream cutoff")
            history = state["config"].get("validation_history", [])
            if not all(math.isfinite(value) for entry in history for value in entry.values()):
                raise ValueError(f"Nonfinite pretraining history: {identity}")
            pretraining.extend({**base, **entry} for entry in history)
    for dataset in DATASETS:
        full = errors[dataset, "JEPA-MAE", 7]
        for method in METHODS[1:]:
            other = errors[dataset, method, 7]
            paired.append({"dataset": dataset, "method": "JEPA-MAE", "reference": method, "seed": 7,
                           "wins": int((full < other).sum()), "ties": int((full == other).sum()),
                           "losses": int((full > other).sum()), "num_samples": len(full),
                           "mean_mse_difference": float((full - other).mean())})
    summary = summarize(rows)
    scores = {(row["dataset"], row["model"]): row["rmse"] for row in rows}
    for row in summary:
        naive = scores[row["dataset"], "Naive-last"]
        row["rmse_difference_vs_naive"] = row["mean_rmse"] - naive
        row["relative_improvement_vs_naive_pct"] = 100 * (1 - row["mean_rmse"] / naive)

    if output.exists():
        raise FileExistsError(f"Use a fresh analysis directory: {output}")
    output.mkdir(parents=True)
    for directory in ("data", "tables", "figures", "provenance"):
        (output / directory).mkdir()
    write_csv(output / "data/raw_results.csv", rows)
    save_json(output / "data/raw_results.json", rows)
    write_csv(output / "data/overall_summary.csv", summary)
    write_csv(output / "data/per_sample_errors.csv", per_sample)
    write_csv(output / "data/paired_comparisons.csv", paired)
    write_csv(output / "data/sine_horizon_metrics.csv", horizons)
    write_csv(output / "data/downstream_validation_history.csv", downstream)
    write_csv(output / "data/mnist_pretraining_history.csv", pretraining)
    write_csv(output / "provenance/raw_artifact_hashes.csv", artifacts)
    write_summaries(output / "tables", rows)
    save_json(output / "provenance/runtime_manifest.json", manifest)
    save_json(output / "provenance/source_snapshot.json", sources)
    for dataset in DATASETS:
        shutil.copyfile(results / dataset / "seed_7/split_audit.json", output / "provenance" / f"split_audit_{dataset}.json")
    for filename in ("00_stage00_verification.json", "00_stage00_verification.py"):
        path = REPO / "results" / filename
        if path.suffix == ".json":
            shutil.copyfile(path, output / "provenance/experiment_verification.json")
        else:
            save_json(output / "provenance/verification_source.json", {"source": path.read_text(), "sha256": sha256(path)})
    shutil.copyfile(REPO / "doc/stage00_comparison.md", output / "protocol.md")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#246394", "#759ab6", "#abc4d6", "#b76332", "#777777", "#999999", "#bbbbbb"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, dataset, title, unit in zip(axes, DATASETS, ("MNIST masked-row reconstruction", "Sine-cosine forecasting"),
                                       ("Pixel-intensity RMSE", "Window-relative Close RMSE")):
        values = [scores[dataset, method] for method in METHODS]
        ax.barh(METHODS, values, color=colors)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.32)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(unit)
        ax.spines[["top", "right"]].set_visible(False)
        for index, value in enumerate(values):
            ax.text(value + max(values) * .02, index, f"{value:.6f}", va="center", fontsize=9)
    fig.suptitle("Stage 00: seed 7; lower RMSE is better", fontsize=13)
    fig.text(.5, .015, "One independent seed; no uncertainty bars. These are diagnostic tasks, not financial evidence.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .94))
    fig.savefig(output / "figures/rmse_comparison.png", dpi=180)
    plt.close(fig)

    mnist_gain = 100 * (1 - scores["MNIST_ROWS", "JEPA-MAE"] / scores["MNIST_ROWS", "MAE-only"])
    sine_gru_gain = 100 * (1 - scores["SIN_COS", "GRU"] / scores["SIN_COS", "JEPA-MAE"])
    pair = {(r["dataset"], r["reference"]): r for r in paired}
    runs = {(r["dataset"], r["model"]): r for r in rows}
    joint_last = [r for r in pretraining if r["model"] == "JEPA-MAE"][-1]
    jepa_history = [r for r in pretraining if r["model"] == "JEPA-only"]
    jepa_first, jepa_last = jepa_history[0], jepa_history[-1]
    jepa_share = 100 * runs["MNIST_ROWS", "JEPA-MAE"]["lambda_jepa"] * joint_last["jepa_loss"] / joint_last["total_loss"]
    mnist_epochs = "/".join(str(runs["MNIST_ROWS", method]["selected_epoch"]) for method in METHODS[:4])
    sine_epochs = "/".join(str(runs["SIN_COS", method]["selected_epoch"]) for method in METHODS[:4])
    legacy_mnist_path = REPO / "results/00_stage00_legacy_reproduction/mnist_baseline_audit.json"
    legacy_sine_path = REPO / "results/00_stage00_legacy_threads/results.json"
    legacy_mnist, legacy_sine = read_json(legacy_mnist_path), read_json(legacy_sine_path)
    shutil.copyfile(legacy_mnist_path, output / "provenance/legacy_mnist_audit.json")
    shutil.copyfile(legacy_sine_path, output / "provenance/legacy_sine_reproduction.json")
    lines = ["# Stage 00: controlled comparison and objective ablations", "",
             "> **DIAGNOSTIC ONLY — one independent seed (7).** Repeated execution verifies determinism; it does not estimate seed uncertainty.", "",
             "## Main findings", "",
             f"The full JEPA-MAE model and MAE-only are nearly tied on MNIST: adding the JEPA term reduces pooled RMSE by {mnist_gain:.3f}%. "
             f"The full model wins on {pair['MNIST_ROWS', 'MAE-only']['wins']} of {pair['MNIST_ROWS', 'MAE-only']['num_samples']} images against MAE-only, "
             f"compared with {pair['MNIST_ROWS', 'JEPA-only']['wins']} against JEPA-only and {pair['MNIST_ROWS', 'GRU']['wins']} against GRU. "
             "This run supports a strong role for pixel reconstruction in this transfer protocol; it does not establish a meaningful advantage for the joint objective.", "",
             f"GRU has the lowest sine-cosine RMSE, {sine_gru_gain:.2f}% below full JEPA-MAE. It beats the full model on "
             f"{pair['SIN_COS', 'GRU']['losses']} of {pair['SIN_COS', 'GRU']['num_samples']} forecast origins "
             "and at all four forecast horizons. MAE-only also has lower pooled RMSE than the full model. "
             f"Full JEPA-MAE improves pooled RMSE over JEPA-only, but wins on only {pair['SIN_COS', 'JEPA-only']['wins']} of "
             f"{pair['SIN_COS', 'JEPA-only']['num_samples']} origins; aggregate improvement is not uniform across origins.", "",
             "## Measured results", "", "| Method | MNIST RMSE | Sine-cosine RMSE |", "|---|---:|---:|"]
    lines.extend(f"| {method} | {scores['MNIST_ROWS', method]:.9f} | {scores['SIN_COS', method]:.9f} |" for method in METHODS)
    lines += ["", "Each entry is one run. Mean RMSE equals the displayed value; across-seed SD is unavailable. "
              "MNIST errors are pixel intensities; sine errors are window-relative Close values. Do not compare magnitudes across datasets.", "",
              "## Experiment and model definitions", "",
              "MNIST retains 4,000 train / 128 validation / 128 test images, 28 row tokens, 11 randomly masked targets and 17 visible rows. "
              "The synthetic task retains 20 historical observations, Close/Volume inputs, four future Close targets, and 96/10/76 train/validation/test windows. "
              "Its raw row split is February 19–June 17 / June 18–30 / July 1–September 17, 2021, after the original 49-row warmup removal. "
              "See [protocol.md](protocol.md) for complete data construction, normalization, masking and model definitions.", "",
              "Full-model SSL weights are JEPA/MAE = 0.01/1 on MNIST and 0.7/0.3 on sine. Ablations change only those weights to 1/0 or 0/1. "
              "MNIST uses the existing width-64 encoder/predictor (two blocks, four heads), 30 SSL epochs, then a fresh identical MLP readout trained for 30 epochs with encoder/predictor frozen. "
              "Sine uses the width-16 encoder and width-8 predictor (one block, two heads), the original single SSL update, and 120 downstream epochs with encoder fine-tuning. "
              "GRU is one layer with hidden size 64 and no dropout, supervised using the same per-dataset downstream budget. "
              "Naive-last, Drift and Mean-context are deterministic context-only baselines. Every learned downstream checkpoint is selected by validation RMSE only.", "",
              "## Interpretation and confounders", "",
              f"- **Observation:** the full MNIST loss is heavily reconstruction-weighted. At the final SSL epoch its JEPA component contributes approximately {jepa_share:.2f}% of the weighted validation loss. "
              "Similarity to MAE-only is therefore consistent with this particular weighting; it does not test every possible balance.",
              f"- **Observation:** JEPA-only MNIST latent validation loss changes from {jepa_first['jepa_loss']:.6f} at SSL epoch {jepa_first['epoch']} "
              f"to {jepa_last['jepa_loss']:.6f} at epoch {jepa_last['epoch']}; target-embedding standard deviation changes from "
              f"{jepa_first['embedding_std']:.6f} to {jepa_last['embedding_std']:.6f}. A small moving-teacher latent loss alone is not a reliable proxy for reconstruction transfer. "
              "These diagnostics alone do not establish representation collapse. All variants retain their final, fixed-budget SSL checkpoint; no post-test checkpoint substitution was made.",
              f"- **Observation:** downstream selection differs by method. Full/JEPA-only/MAE-only/GRU select epochs {mnist_epochs} on MNIST and {sine_epochs} on sine. "
              f"Sine GRU's final validation RMSE is {runs['SIN_COS', 'GRU']['final_validation_loss']:.6f} versus "
              f"{runs['SIN_COS', 'GRU']['best_validation_loss']:.6f} at its selected epoch, illustrating why the validation checkpoint matters.",
              "- **Limit:** sine pretraining is only one optimizer step. Its 13 validation observations cannot form a 20-observation SSL window; SSL validation is unavailable, "
              "while downstream validation has 10 windows. This budget supports a pipeline diagnostic, not a strong SSL comparison.",
              "- **Limit:** there is no matching randomly initialized Transformer control. Comparison with GRU cannot isolate the benefit of pretraining from architecture, "
              "parameter count, total optimization work or the frozen/readout protocol. Equal downstream epochs do not imply equal total compute.",
              "- **Limit:** only one learned-model seed exists. MNIST images and overlapping sine forecast origins are not additional random seeds; "
              "the four forecast steps and 28 row pixels are not independent model runs. No p-values or seed confidence intervals are claimed.",
              "- **Hypothesis for a separately specified future experiment:** additional pretraining and several seeds may change the ablation ranking. "
              "Any new settings should be selected on validation data and evaluated on fresh held-out support; the present test outcomes must not become tuning criteria.", "",
              "## Legacy compatibility and corrections", "",
              f"The old MNIST previous-row baseline accessed masked pixels in {legacy_mnist['masked_rows_read_by_legacy_baseline']:,}/"
              f"{legacy_mnist['num_target_rows']:,} target rows. The new Naive-last uses only visible rows, "
              "so its result is not numerically comparable to the old baseline. MNIST remains reconstruction with visible rows on both sides, not causal forecasting. "
              "The old sine SSL cutoff overlapped downstream validation/test; the corrected run aligns SSL to the existing July 1 downstream cutoff. "
              f"The archived legacy full-model numbers were reproduced separately, including sine RMSE {legacy_sine['model_rmse']:.10f} with "
              f"{legacy_sine['torch_num_threads']} CPU threads. "
              "This comparison fixes one CPU thread throughout; floating-point reduction order changed the legacy optimization trajectory. "
              "Original published diagnostic snapshots are retained and are not replaced by these scores.", "",
              "## Verification and publication", "",
              "The analysis recomputed every RMSE in float64 from saved predictions, checked complete 2×7×1 coverage, identical targets/support across methods, "
              "exact repeat predictions and recorded metrics, finite checkpoint weights, and restoration of the minimum-validation checkpoint. "
              "The full experiment verification also compares repeated pretrained/downstream weights and confirms split separation; its record is included under provenance.", "",
              "Published data include raw run metrics, mean/SD/count summaries, per-sample errors, paired win/loss counts, per-horizon metrics, training histories and LaTeX tables. "
              "Raw prediction/checkpoint files are represented by paths and SHA-256 hashes and remain in the ignored results archive. "
              "The runtime manifest, split audits and exact source text snapshot preserve the executed configuration, including code that was uncommitted at run time.", "",
              "Reproduce this analysis and publish it with:", "", "```bash",
              "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/xujiang/miniconda3/envs/ts-jepa/bin/python results/00_stage00_verification.py",
              "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/xujiang/miniconda3/envs/ts-jepa/bin/python analysis/analyze_stage00_results.py",
              "python publish_thesis_results.py --analysis-dir analysis_artifacts/00_model_comparison", "```", "",
              "Use a new `--output-dir` to regenerate analysis staging. Publication is local to the repository's immutable `thesis_results/` workflow."]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    scope_path = output / "00_model_comparison.json"
    save_json(scope_path, {"purpose": "post-run analysis scope; not a new training configuration", "datasets": list(DATASETS),
                          "models": list(METHODS), "seeds": seeds, "results_dir": str(results), "repeat_dir": str(repeat),
                          "protocol": "protocol.md", "runtime_manifest": "provenance/runtime_manifest.json"})
    save_json(output / "analysis_metadata.json", {
        "error_issues": 0, "canonical_rows": len(rows), "results_dir": str(results),
        "scope": {"config_path": str(scope_path)}, "evidence_class": "single-seed controlled diagnostic",
        "warnings": ["One independent seed", "Sine SSL budget is one update", "No matching scratch Transformer control",
                     "Overlapping sine test windows", "MNIST is reconstruction, not causal forecasting"],
        "repeat_predictions_exact": True, "pooled_rmse_recomputed": True,
        "analysis_source_sha256": sha256(Path(__file__).resolve()),
    })
    write_csv(output / "artifact_manifest.csv", [
        {"path": str(path.relative_to(output)), "status": "generated", "sha256": sha256(path)}
        for path in sorted(output.rglob("*")) if path.is_file()
    ])
    print(f"Validated {len(rows)} results; analysis: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO / "results/00_model_comparison_repeat")
    parser.add_argument("--repeat-dir", type=Path, default=REPO / "results/00_model_comparison")
    parser.add_argument("--output-dir", type=Path, default=REPO / "analysis_artifacts/00_model_comparison")
    args = parser.parse_args()
    analyze(args.results_dir.resolve(), args.repeat_dir.resolve(), args.output_dir.resolve())
