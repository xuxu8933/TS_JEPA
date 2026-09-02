"""Build reproducible cross-stock and cross-strategy comparison summaries."""

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path

import pandas as pd

from config.file_options import parse_args_with_config
from plot_top_stock_metrics import (
    DEFAULT_MODELS,
    DEFAULT_STOCK_ORDER,
    latest_comparison_files,
)


DEFAULT_MASK_STRATEGIES = ("random", "local_long")
METRICS = ("rmse", "trend_accuracy")
RAW_COLUMNS = ("strategy", "stock", "seed", "model", *METRICS, "source_file")
ISSUE_COLUMNS = ("strategy", "stock", "seed", "model", "status", "details")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Extract stock-run metrics and compare masking strategies without "
            "launching training."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "JSON, JSONC, or TOML experiment option file. Reads [common] and "
            "[analysis], with strategies inherited from "
            "[runner].masking.strategies; "
            "explicit command-line options except stocks/seeds take precedence."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default="./results/top10_nasdaq100_mask_comparison",
        help=(
            "Standalone analysis root. Configured analyses derive "
            "results/<config filename>; multi-strategy runs are expected below "
            "RESULTS_DIR/STRATEGY/STOCK/seed_N."
        ),
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Strategies to compare. Defaults to the experiment manifest.",
    )
    parser.add_argument(
        "--stocks",
        nargs="+",
        default=None,
        help="Expected stocks. Defaults to the experiment manifest or top-10 list.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Expected seeds. Defaults to the experiment manifest or discovered runs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models required in every completed stock/seed result.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Write summaries from available runs even when expected results are "
            "missing. The default writes diagnostics and refuses partial summaries."
        ),
    )
    parser.add_argument(
        "--skip-plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write CSV summaries without the overall strategy comparison plot.",
    )
    return parse_args_with_config(parser, argv, section="analysis")


def _seed_from_path(path):
    for part in path.parent.parts:
        if part.startswith("seed_"):
            try:
                return int(part.removeprefix("seed_"))
            except ValueError:
                continue
    raise ValueError(f"Could not determine seed from result path: {path}")


def _load_experiment_manifest(results_dir):
    manifest_path = Path(results_dir) / "experiment_manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open() as manifest_file:
        return json.load(manifest_file)


def _strategy_results_dir(results_dir, strategy, strategy_count):
    nested_dir = Path(results_dir) / strategy
    if nested_dir.is_dir() or strategy_count > 1:
        return nested_dir
    # Backward compatibility for a legacy single-strategy flat result root.
    return Path(results_dir)


def _discover_seeds(results_dir, strategies):
    seeds = set()
    for strategy in strategies:
        strategy_dir = _strategy_results_dir(
            results_dir,
            strategy,
            len(strategies),
        )
        for txt_path in strategy_dir.rglob("last_model_comparison_*.txt"):
            try:
                seeds.add(_seed_from_path(txt_path))
            except ValueError:
                continue
    return sorted(seeds)


def resolve_analysis_scope(args):
    manifest = _load_experiment_manifest(args.results_dir)
    strategies = list(
        args.strategies
        or manifest.get("mask_strategies")
        or DEFAULT_MASK_STRATEGIES
    )
    stocks = [
        stock.upper()
        for stock in (args.stocks or manifest.get("stocks") or DEFAULT_STOCK_ORDER)
    ]
    seeds = list(args.seeds or manifest.get("seeds") or [])
    if not seeds:
        seeds = _discover_seeds(args.results_dir, strategies)

    if not strategies:
        raise ValueError("At least one strategy is required")
    if len(strategies) != len(set(strategies)):
        raise ValueError("--strategies must contain unique values")
    if not stocks:
        raise ValueError("At least one stock is required")
    if len(stocks) != len(set(stocks)):
        raise ValueError("--stocks must contain unique tickers")
    if not seeds:
        raise RuntimeError(
            "No seeds were supplied, recorded in experiment_manifest.json, or "
            f"discovered under {args.results_dir}"
        )
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique values")
    return strategies, stocks, seeds


def collect_results_and_issues(results_dir, strategy, stocks, seeds, models):
    latest_by_stock = latest_comparison_files(results_dir, seeds=seeds)
    expected_models = set(models)
    rows = []
    issues = []

    for stock in stocks:
        paths_by_seed = {}
        ambiguous_seeds = set()
        for txt_path in latest_by_stock.get(stock, []):
            try:
                seed = _seed_from_path(txt_path)
            except ValueError as exc:
                issues.append(
                    {
                        "strategy": strategy,
                        "stock": stock,
                        "seed": "",
                        "model": "",
                        "status": "invalid_result_path",
                        "details": str(exc),
                    }
                )
                continue
            if seed in paths_by_seed:
                ambiguous_seeds.add(seed)
                issues.append(
                    {
                        "strategy": strategy,
                        "stock": stock,
                        "seed": seed,
                        "model": "",
                        "status": "ambiguous_result_files",
                        "details": (
                            f"Both {paths_by_seed[seed]} and {txt_path} match this "
                            "stock/seed. Use a dedicated experiment result root."
                        ),
                    }
                )
            else:
                paths_by_seed[seed] = txt_path

        for seed in seeds:
            if seed in ambiguous_seeds:
                continue
            txt_path = paths_by_seed.get(seed)
            if txt_path is None:
                issues.append(
                    {
                        "strategy": strategy,
                        "stock": stock,
                        "seed": seed,
                        "model": "",
                        "status": "missing_result_file",
                        "details": "No model-comparison CSV was found for this run.",
                    }
                )
                continue

            csv_path = txt_path.with_suffix(".csv")
            found_models = set()
            file_rows = []
            try:
                with csv_path.open(newline="") as csv_file:
                    for row in csv.DictReader(csv_file):
                        model = row["model"]
                        if model in found_models:
                            raise ValueError(
                                f"duplicate model row for {model!r}"
                            )
                        found_models.add(model)
                        metric_values = {
                            metric: float(row[metric]) for metric in METRICS
                        }
                        if not all(
                            math.isfinite(value)
                            for value in metric_values.values()
                        ):
                            raise ValueError(
                                f"non-finite metric value for model {model!r}"
                            )
                        file_rows.append(
                            {
                                "strategy": strategy,
                                "stock": stock,
                                "seed": seed,
                                "model": model,
                                **metric_values,
                                "source_file": str(csv_path),
                            }
                        )
            except (KeyError, TypeError, ValueError, OSError) as exc:
                issues.append(
                    {
                        "strategy": strategy,
                        "stock": stock,
                        "seed": seed,
                        "model": "",
                        "status": "invalid_result_file",
                        "details": f"{csv_path}: {exc}",
                    }
                )
                continue

            rows.extend(file_rows)
            for model in sorted(expected_models - found_models):
                issues.append(
                    {
                        "strategy": strategy,
                        "stock": stock,
                        "seed": seed,
                        "model": model,
                        "status": "missing_model",
                        "details": f"Model is absent from {csv_path}.",
                    }
                )

    return (
        pd.DataFrame(rows, columns=RAW_COLUMNS),
        pd.DataFrame(issues, columns=ISSUE_COLUMNS),
    )


def collect_raw_results(results_dir, strategy, stocks, seeds):
    """Compatibility helper that extracts rows without enforcing model coverage."""
    raw_results, _ = collect_results_and_issues(
        results_dir,
        strategy,
        stocks,
        seeds,
        models=(),
    )
    return raw_results.drop(columns=["source_file"])


def aggregate_metrics(raw_results, group_columns):
    aggregations = {f"{metric}_mean": (metric, "mean") for metric in METRICS}
    aggregations.update(
        {f"{metric}_std": (metric, "std") for metric in METRICS}
    )
    aggregations["num_runs"] = ("seed", "count")

    summary = (
        raw_results.groupby(group_columns, as_index=False)
        .agg(**aggregations)
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    std_columns = [f"{metric}_std" for metric in METRICS]
    summary[std_columns] = summary[std_columns].fillna(0.0)
    return summary


def aggregate_strategy_runs(raw_results):
    """Average stocks within a seed before measuring variation across seeds."""
    per_seed_aggregations = {metric: (metric, "mean") for metric in METRICS}
    per_seed_aggregations["num_stocks"] = ("stock", "nunique")
    per_seed_summary = (
        raw_results.groupby(["strategy", "seed", "model"], as_index=False)
        .agg(**per_seed_aggregations)
        .sort_values(["strategy", "seed", "model"])
        .reset_index(drop=True)
    )
    overall_summary = aggregate_metrics(
        per_seed_summary,
        ["strategy", "model"],
    )
    return per_seed_summary, overall_summary


def validate_run_counts(
    per_stock_summary,
    stocks,
    seeds,
    strategies=DEFAULT_MASK_STRATEGIES,
    models=DEFAULT_MODELS,
):
    expected_models = set(models)
    expected_runs = len(seeds)
    problems = []

    for strategy in strategies:
        for stock in stocks:
            rows = per_stock_summary[
                (per_stock_summary["strategy"] == strategy)
                & (per_stock_summary["stock"] == stock)
            ]
            missing_models = sorted(expected_models - set(rows["model"]))
            if missing_models:
                problems.append(
                    f"{strategy}/{stock}: missing models {','.join(missing_models)}"
                )
            for row in rows.itertuples():
                if row.num_runs != expected_runs:
                    problems.append(
                        f"{strategy}/{stock}/{row.model}: found {row.num_runs} "
                        f"runs, expected {expected_runs}"
                    )

    if problems:
        raise RuntimeError(
            "Results are incomplete, so summary statistics were not written:\n- "
            + "\n- ".join(problems)
        )


def paired_strategy_differences(per_seed_summary, strategies):
    columns = (
        "strategy_a",
        "strategy_b",
        "model",
        "metric",
        "mean_delta_b_minus_a",
        "std_delta",
        "num_paired_seeds",
        "better_strategy",
    )
    rows = []
    for strategy_a, strategy_b in combinations(strategies, 2):
        left = per_seed_summary[per_seed_summary["strategy"] == strategy_a]
        right = per_seed_summary[per_seed_summary["strategy"] == strategy_b]
        paired = left.merge(
            right,
            on=["seed", "model"],
            suffixes=("_a", "_b"),
        )
        for model, model_rows in paired.groupby("model"):
            for metric in METRICS:
                deltas = model_rows[f"{metric}_b"] - model_rows[f"{metric}_a"]
                mean_delta = float(deltas.mean())
                if mean_delta == 0.0:
                    better_strategy = "tie"
                elif metric == "rmse":
                    better_strategy = strategy_a if mean_delta > 0 else strategy_b
                else:
                    better_strategy = strategy_b if mean_delta > 0 else strategy_a
                rows.append(
                    {
                        "strategy_a": strategy_a,
                        "strategy_b": strategy_b,
                        "model": model,
                        "metric": metric,
                        "mean_delta_b_minus_a": mean_delta,
                        "std_delta": float(deltas.std()) if len(deltas) > 1 else 0.0,
                        "num_paired_seeds": len(deltas),
                        "better_strategy": better_strategy,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def save_strategy_comparison_plot(overall_summary, strategies, output_path):
    import matplotlib.pyplot as plt
    import numpy as np

    models = list(dict.fromkeys(overall_summary["model"]))
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(13, 9))
    x = np.arange(len(models))
    width = 0.8 / max(len(strategies), 1)

    for axis, metric in zip(axes, METRICS):
        for strategy_index, strategy in enumerate(strategies):
            rows = (
                overall_summary[overall_summary["strategy"] == strategy]
                .set_index("model")
                .reindex(models)
            )
            offset = (strategy_index - (len(strategies) - 1) / 2) * width
            axis.bar(
                x + offset,
                rows[f"{metric}_mean"],
                width,
                yerr=rows[f"{metric}_std"].fillna(0.0),
                capsize=3,
                label=strategy,
            )
        axis.set_ylabel(metric.replace("_", " ").upper())
        axis.set_title(
            f"{metric.replace('_', ' ').title()} across seeded stock means"
        )
        axis.set_xticks(x)
        axis.set_xticklabels(models, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)

    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summaries(args, stocks, seeds, strategies=None):
    strategies = list(
        strategies
        or getattr(args, "strategies", None)
        or DEFAULT_MASK_STRATEGIES
    )
    models = list(getattr(args, "models", DEFAULT_MODELS))
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_frames = []
    issue_frames = []
    for strategy in strategies:
        strategy_dir = _strategy_results_dir(
            results_dir,
            strategy,
            len(strategies),
        )
        raw, issues = collect_results_and_issues(
            strategy_dir,
            strategy,
            stocks,
            seeds,
            models,
        )
        raw_frames.append(raw)
        issue_frames.append(issues)

    raw_results = pd.concat(raw_frames, ignore_index=True)
    issues = pd.concat(issue_frames, ignore_index=True)
    raw_path = results_dir / "raw_runs.csv"
    issues_path = results_dir / "missing_or_failed_runs.csv"
    raw_results.sort_values(["strategy", "stock", "seed", "model"]).to_csv(
        raw_path,
        index=False,
    )
    issues.to_csv(issues_path, index=False)

    if raw_results.empty:
        raise RuntimeError(f"No experiment results found under {results_dir}")
    if not issues.empty and not getattr(args, "allow_incomplete", False):
        raise RuntimeError(
            f"Found {len(issues)} incomplete or invalid run entries. Diagnostics "
            f"were written to {issues_path}; pass --allow-incomplete to summarize "
            "only the available results."
        )

    per_stock_summary = aggregate_metrics(
        raw_results,
        ["strategy", "stock", "model"],
    )
    per_seed_summary, overall_summary = aggregate_strategy_runs(raw_results)
    paired_summary = paired_strategy_differences(per_seed_summary, strategies)

    per_stock_path = results_dir / "per_stock_summary.csv"
    per_seed_path = results_dir / "per_seed_summary.csv"
    overall_path = results_dir / "overall_summary.csv"
    paired_path = results_dir / "paired_strategy_differences.csv"
    per_stock_summary.to_csv(per_stock_path, index=False)
    per_seed_summary.to_csv(per_seed_path, index=False)
    overall_summary.to_csv(overall_path, index=False)
    paired_summary.to_csv(paired_path, index=False)

    analysis_manifest = {
        "config": getattr(args, "config", None),
        "results_dir": str(results_dir),
        "strategies": strategies,
        "stocks": stocks,
        "seeds": seeds,
        "models": models,
        "complete": issues.empty,
        "available_metric_rows": len(raw_results),
        "issue_rows": len(issues),
    }
    with (results_dir / "analysis_manifest.json").open("w") as manifest_file:
        json.dump(analysis_manifest, manifest_file, indent=2)
        manifest_file.write("\n")

    if not getattr(args, "skip_plot", False):
        plot_path = results_dir / "strategy_comparison.png"
        save_strategy_comparison_plot(overall_summary, strategies, plot_path)
        print(f"Strategy comparison plot saved to: {plot_path}", flush=True)

    for label, path in (
        ("Raw run metrics", raw_path),
        ("Missing/failed run diagnostics", issues_path),
        ("Per-stock summary", per_stock_path),
        ("Per-seed summary", per_seed_path),
        ("Overall summary", overall_path),
        ("Paired strategy differences", paired_path),
    ):
        print(f"{label} saved to: {path}", flush=True)
    return raw_path, per_stock_path, per_seed_path, overall_path


def main(argv=None):
    args = parse_args(argv)
    strategies, stocks, seeds = resolve_analysis_scope(args)
    print(f"Strategies: {', '.join(strategies)}", flush=True)
    print(f"Stocks ({len(stocks)}): {', '.join(stocks)}", flush=True)
    print(f"Seeds ({len(seeds)}): {', '.join(map(str, seeds))}", flush=True)
    write_summaries(args, stocks, seeds, strategies=strategies)


if __name__ == "__main__":
    main()
