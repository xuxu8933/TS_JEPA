import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date
from pathlib import Path

from download_indices_and_news import TOP_NASDAQ100_STOCKS
from config.config_pretrain import config as pretrain_defaults
from config.experiment import effective_feature_columns, resolve_forecast_horizon
from config.file_options import parse_args_with_config
from config.preprocessing_presets import PREPROCESSING_PRESETS
from pretrain_dual_loss import (
    parse_args as parse_pretrain_args,
    validate_strategy_config,
)


MASK_STRATEGIES = ("random", "local_long", "future_block", "causal_multiblock")
RUN_MANIFEST_FILENAME = "run_manifest.json"
COVERAGE_CONFIG_KEYS = frozenset(
    ("stocks", "seeds", "seed", "max_stocks", "max_seeds")
)
NON_RESULT_CONFIG_KEYS = frozenset(
    (
        "config",
        "results_dir",
        "skip_pretrain",
        "skip_combined_plot",
        "dry_run",
        "max_parallel_jobs",
        "request_delay",
        "verbose",
    )
)


def _jsonable_config_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable_config_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_config_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _jsonable_config_value(item)
            for key, item in value.items()
        }
    return value


def effective_experiment_config(args_or_options):
    """Return result-affecting runner options, excluding execution coverage."""
    options = (
        vars(args_or_options)
        if isinstance(args_or_options, argparse.Namespace)
        else dict(args_or_options)
    )
    excluded = COVERAGE_CONFIG_KEYS | NON_RESULT_CONFIG_KEYS
    effective = {
        key: _jsonable_config_value(options[key])
        for key in sorted(options)
        if key not in excluded
    }
    # ``feature_cols`` was a legacy runner override. Its historical null value
    # did not affect results, so omit it when comparing old and new manifests.
    if effective.get("feature_cols") is None:
        effective.pop("feature_cols", None)
    if effective.get("forecast_horizon") is None:
        effective.pop("forecast_horizon", None)
    if effective.get("sentiment_normalization") in (None, "none"):
        effective.pop("sentiment_normalization", None)

    if effective.get("skip_download") is True:
        for key in (
            "download_start_date",
            "download_end_date",
            "skip_news",
            "max_news_articles",
            "news_chunk_days",
            "request_delay",
            "write_mode",
        ):
            effective.pop(key, None)
    elif effective.get("skip_news") is True or effective.get("use_sentiment") is False:
        for key in ("max_news_articles", "news_chunk_days", "request_delay"):
            effective.pop(key, None)

    strategies = set(effective.get("mask_strategies") or ())
    conditional_strategy_options = {
        "local_long": (
            "mae_window_patches",
            "jepa_gap_patches",
            "jepa_target_patches",
        ),
        "future_block": ("future_target_patches",),
        "causal_multiblock": (
            "causal_num_blocks",
            "causal_block_patches",
            "causal_block_gap_patches",
        ),
    }
    for strategy, keys in conditional_strategy_options.items():
        if strategy not in strategies:
            for key in keys:
                effective.pop(key, None)

    if effective.get("normalization") != "train_robust_zscore":
        effective.pop("robust_zscore_clip", None)
    if effective.get("preprocessing_preset") is not None:
        for key in (
            "feature_transform",
            "normalization",
            "forecast_target",
            "market_data",
        ):
            effective.pop(key, None)
    if effective.get("use_sentiment") is False:
        effective.pop("sentiment_features", None)
    if effective.get("use_best_checkpoint") is True:
        effective.pop("checkpoint_to_use", None)
    return effective


def experiment_config_signature(args_or_options):
    encoded = json.dumps(
        effective_experiment_config(args_or_options),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def requested_stock_seed_runs(stocks, seeds):
    normalized_stocks = [str(stock).upper() for stock in stocks]
    normalized_seeds = [int(seed) for seed in seeds]
    if not normalized_stocks:
        raise ValueError("At least one stock must be configured in [common]")
    if not normalized_seeds:
        raise ValueError("At least one seed must be configured in [common]")
    if len(normalized_stocks) != len(set(normalized_stocks)):
        raise ValueError("[common].stocks must contain unique tickers")
    if len(normalized_seeds) != len(set(normalized_seeds)):
        raise ValueError("[common].seeds must contain unique values")
    return [
        (stock, seed)
        for stock in normalized_stocks
        for seed in normalized_seeds
    ]


def stock_result_dir(args, strategy, stock, seed):
    result_dir = strategy_results_dir(args, strategy)
    if getattr(args, "preprocessing_preset", None):
        result_dir /= args.preprocessing_preset
    return result_dir / stock / f"seed_{seed}"


def _load_json(path):
    try:
        with path.open() as input_file:
            value = json.load(input_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read compatibility metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Compatibility metadata must be an object: {path}")
    return value


def _manifest_effective_config(manifest):
    recorded = manifest.get("effective_config")
    if isinstance(recorded, dict):
        return effective_experiment_config(recorded)
    arguments = manifest.get("arguments")
    if isinstance(arguments, dict):
        return effective_experiment_config(arguments)
    return None


def _config_difference(previous, current):
    changed = []
    for key in sorted(set(previous) | set(current)):
        if previous.get(key) != current.get(key):
            changed.append(
                f"{key}: previous={previous.get(key)!r}, current={current.get(key)!r}"
            )
    return changed


def validate_existing_experiment(args):
    """Validate the result root and return whether legacy runs are compatible."""
    results_dir = Path(args.results_dir)
    manifest_path = results_dir / "experiment_manifest.json"
    existing_run_outputs = any(
        results_dir.rglob("last_model_comparison_*.txt")
    ) if results_dir.exists() else False

    if not manifest_path.exists():
        if existing_run_outputs:
            raise RuntimeError(
                f"Existing results under {results_dir} have no experiment manifest, "
                "so their configuration cannot be verified. They will not be reused "
                "or overwritten."
            )
        return False

    manifest = _load_json(manifest_path)
    previous = _manifest_effective_config(manifest)
    if previous is None:
        raise RuntimeError(
            f"Existing manifest {manifest_path} does not contain enough resolved "
            "configuration metadata. Existing data will not be reused or overwritten."
        )

    current = effective_experiment_config(args)
    if previous != current:
        differences = _config_difference(previous, current)
        details = "\n  ".join(differences[:12])
        if len(differences) > 12:
            details += f"\n  ... and {len(differences) - 12} more"
        raise RuntimeError(
            "The current result-affecting configuration is incompatible with "
            f"{manifest_path}. Existing results will not be reused or overwritten. "
            "Use a config file/result directory with a different matching name."
            + (f"\n  {details}" if details else "")
        )
    return True


def compatible_result_directories(args):
    """Find sibling result roots with the same result-affecting configuration."""
    current_results_dir = Path(args.results_dir).resolve()
    current_config = effective_experiment_config(args)
    compatible = []
    for manifest_path in sorted(
        current_results_dir.parent.glob("*/experiment_manifest.json")
    ):
        candidate_results_dir = manifest_path.parent.resolve()
        if candidate_results_dir == current_results_dir:
            continue
        try:
            manifest = _load_json(manifest_path)
        except RuntimeError:
            continue
        candidate_config = _manifest_effective_config(manifest)
        if candidate_config == current_config:
            compatible.append(candidate_results_dir)
    return compatible


def reject_duplicate_experiment_config(args):
    """Reject a renamed config that would duplicate an existing experiment."""
    compatible = compatible_result_directories(args)
    if not compatible:
        return
    locations = "\n  ".join(str(path) for path in compatible)
    raise RuntimeError(
        "An experiment with the same result-affecting runner configuration "
        "already exists. Renaming the config file does not create a new "
        "experiment. Reuse the existing config/result folder and change only "
        "[common].stocks or [common].seeds to extend coverage:\n  "
        + locations
    )


def validate_config_result_mapping(args):
    if not args.config:
        return
    config_name = Path(args.config).stem
    result_name = Path(args.results_dir).resolve().name
    if result_name != config_name:
        raise ValueError(
            "Experiment config and result directory names must match: "
            f"config={config_name!r}, result_directory={result_name!r}"
        )


def _comparison_outputs_complete(run_dir):
    for txt_path in sorted(run_dir.glob("last_model_comparison_*.txt")):
        csv_path = txt_path.with_suffix(".csv")
        if txt_path.is_file() and txt_path.stat().st_size > 0:
            if csv_path.is_file() and csv_path.stat().st_size > 0:
                return True
    return False


def downstream_run_status(
    args,
    strategy,
    stock,
    seed,
    *,
    legacy_manifest_compatible,
):
    """Return complete, missing, or incompatible for one downstream run."""
    run_dir = stock_result_dir(args, strategy, stock, seed)
    run_manifest_path = run_dir / RUN_MANIFEST_FILENAME
    outputs_complete = _comparison_outputs_complete(run_dir)

    if run_manifest_path.exists():
        run_manifest = _load_json(run_manifest_path)
        expected_signature = experiment_config_signature(args)
        try:
            recorded_seed = int(run_manifest.get("seed"))
        except (TypeError, ValueError):
            recorded_seed = None
        recorded_run_config = run_manifest.get("effective_config")
        config_matches = run_manifest.get("config_signature") == expected_signature
        if isinstance(recorded_run_config, dict):
            config_matches = (
                effective_experiment_config(recorded_run_config)
                == effective_experiment_config(args)
            )
        elif legacy_manifest_compatible:
            # The compatible experiment manifest validates legacy run manifests
            # whose signature predates conditional-option canonicalization.
            config_matches = True
        identity_matches = (
            config_matches
            and run_manifest.get("strategy") == strategy
            and str(run_manifest.get("stock", "")).upper() == stock
            and recorded_seed == int(seed)
        )
        if not identity_matches:
            return "incompatible"
        if run_manifest.get("status") == "complete" and outputs_complete:
            return "complete"
        return "missing"

    if outputs_complete:
        return "complete" if legacy_manifest_compatible else "incompatible"

    if run_dir.exists() and any(run_dir.iterdir()):
        return "incompatible"
    return "missing"


def _command_option(command, option):
    try:
        index = command.index(option)
    except ValueError:
        return None
    if index + 1 >= len(command):
        raise ValueError(f"Command option {option} has no value: {command}")
    return command[index + 1]


def _related_checkpoint_paths(checkpoint_path):
    match = re.match(r"^(.*)_(?:epoch_\d+|best)\.pt$", str(checkpoint_path))
    if not match:
        return []
    prefix = Path(match.group(1))
    return sorted(
        set(prefix.parent.glob(prefix.name + "_epoch_*.pt"))
        | set(prefix.parent.glob(prefix.name + "_best.pt"))
    )


def plan_incremental_execution(
    args,
    stocks,
    seeds,
    strategies,
    *,
    legacy_manifest_compatible,
):
    requested_runs = requested_stock_seed_runs(stocks, seeds)
    completed_runs = set()
    tasks = []

    for stock, seed in requested_runs:
        tuple_complete = True
        for strategy in strategies:
            status = downstream_run_status(
                args,
                strategy,
                stock,
                seed,
                legacy_manifest_compatible=legacy_manifest_compatible,
            )
            if status == "incompatible":
                run_dir = stock_result_dir(args, strategy, stock, seed)
                raise RuntimeError(
                    f"Existing data for {strategy}/{stock}/seed_{seed} at "
                    f"{run_dir} is incompatible or cannot be verified. It will "
                    "not be reused or overwritten."
                )
            if status == "complete":
                continue

            tuple_complete = False
            commands = build_stock_commands(
                args,
                stock,
                seed=seed,
                strategy=strategy,
                results_dir=strategy_results_dir(args, strategy),
            )
            eval_command = commands[-1]
            pretrain_command = (
                commands[0]
                if len(commands) > 1 and "pretrain_dual_loss.py" in commands[0]
                else None
            )
            checkpoint_value = _command_option(
                eval_command,
                "--pretrain-checkpoint-path",
            )
            checkpoint_path = Path(checkpoint_value) if checkpoint_value else None

            if checkpoint_path is not None and checkpoint_path.is_file():
                pretrain_command = None
            elif pretrain_command is not None and checkpoint_path is not None:
                related = _related_checkpoint_paths(checkpoint_path)
                if related:
                    raise RuntimeError(
                        f"Compatible pretraining for {strategy}/{stock}/seed_{seed} "
                        f"is incomplete: expected {checkpoint_path}, but related "
                        "checkpoints already exist. Automatic retraining would "
                        "overwrite them, so execution was stopped."
                    )

            tasks.append(
                {
                    "strategy": strategy,
                    "stock": stock,
                    "seed": seed,
                    "run_dir": stock_result_dir(args, strategy, stock, seed),
                    "checkpoint_path": checkpoint_path,
                    "pretrain_command": pretrain_command,
                    "eval_command": eval_command,
                }
            )

        if tuple_complete:
            completed_runs.add((stock, seed))

    missing_runs = set(requested_runs) - completed_runs
    return {
        "requested_runs": requested_runs,
        "completed_runs": completed_runs,
        "missing_runs": missing_runs,
        "tasks": tasks,
    }


def build_experiment_manifest(args, stocks, seeds, strategies):
    preprocessing = resolve_preprocessing_settings(args)
    return {
        "runner": "run_top_nasdaq100_stocks.py",
        "config_name": Path(args.config).stem if args.config else None,
        "arguments": _jsonable_config_value(vars(args)),
        "effective_config": effective_experiment_config(args),
        "config_signature": experiment_config_signature(args),
        "stocks": [str(stock).upper() for stock in args.stocks],
        "seeds": [int(seed) for seed in args.seeds],
        "run_stocks": list(stocks),
        "run_seeds": list(seeds),
        "mask_strategies": list(strategies),
        "results_dir": str(args.results_dir),
        "strategy_results_dirs": {
            strategy: str(strategy_results_dir(args, strategy))
            for strategy in strategies
        },
        "use_sentiment": preprocessing["use_sentiment"],
        "market_features": preprocessing["market_features"],
        "sentiment_features": preprocessing["sentiment_features"],
        "feature_transform": preprocessing["feature_transform"],
        "normalization": preprocessing["normalization"],
        "forecast_target": preprocessing["forecast_target"],
        "sampling_mode": args.sampling_mode,
        "pretrain_stride": args.pretrain_stride,
        "pretrain_num_epochs": args.pretrain_num_epochs,
        "eval_num_epochs": args.eval_num_epochs,
        "checkpoint_to_use": args.checkpoint_to_use,
        "use_best_checkpoint": args.use_best_checkpoint,
        "lambda_jepa": args.lambda_jepa,
        "lambda_mae": args.lambda_mae,
        "jepa_loss": args.jepa_loss,
        "mae_loss": args.mae_loss,
    }


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    temporary_path.replace(path)


def write_run_manifest(args, task, status):
    run_dir = task["run_dir"]
    comparison_files = []
    if status == "complete":
        comparison_files = [
            path.name
            for path in sorted(run_dir.glob("last_model_comparison_*.*"))
            if path.suffix in (".csv", ".txt")
        ]
    _write_json(
        run_dir / RUN_MANIFEST_FILENAME,
        {
            "status": status,
            "config_signature": experiment_config_signature(args),
            "effective_config": effective_experiment_config(args),
            "strategy": task["strategy"],
            "stock": task["stock"],
            "seed": task["seed"],
            "checkpoint_path": (
                str(task["checkpoint_path"])
                if task["checkpoint_path"] is not None
                else None
            ),
            "comparison_files": comparison_files,
        },
    )


def run_command(command, dry_run=False, verbose=False):
    if verbose:
        print("=" * 80, flush=True)
        print("Running:", " ".join(command), flush=True)
    if dry_run:
        if verbose:
            print("Dry run: command not executed", flush=True)
        return
    if verbose:
        subprocess.run(command, check=True)
    else:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )


def report_run_status(task, status):
    print(
        f"stock={task['stock']} seed={task['seed']} "
        f"status={task['strategy']}:{status}",
        flush=True,
    )


def execute_task(args, task):
    if args.dry_run and not args.verbose:
        report_run_status(task, "dry-run")
    try:
        if task["pretrain_command"] is not None:
            if not args.verbose and not args.dry_run:
                report_run_status(task, "pretraining")
            run_command(
                task["pretrain_command"],
                dry_run=args.dry_run,
                verbose=args.verbose,
            )
            if (
                not args.dry_run
                and task["checkpoint_path"] is not None
                and not task["checkpoint_path"].is_file()
            ):
                raise RuntimeError(
                    "Pretraining completed without creating the requested "
                    f"checkpoint: {task['checkpoint_path']}"
                )

        if not args.dry_run:
            write_run_manifest(args, task, "running")
        if not args.verbose and not args.dry_run:
            report_run_status(task, "evaluating")
        run_command(
            task["eval_command"],
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        if not args.dry_run:
            if not _comparison_outputs_complete(task["run_dir"]):
                raise RuntimeError(
                    "Downstream evaluation completed without a "
                    "model-comparison CSV/TXT pair in "
                    f"{task['run_dir']}"
                )
            write_run_manifest(args, task, "complete")
            if not args.verbose:
                report_run_status(task, "complete")
    except Exception:
        if not args.verbose and not args.dry_run:
            report_run_status(task, "failed")
        raise


def execute_tasks(args, tasks):
    tasks = list(tasks)
    if args.max_parallel_jobs == 1 or len(tasks) <= 1:
        for task in tasks:
            execute_task(args, task)
        return

    task_iterator = iter(tasks)
    with ThreadPoolExecutor(max_workers=args.max_parallel_jobs) as executor:
        running = set()
        for _ in range(min(args.max_parallel_jobs, len(tasks))):
            running.add(executor.submit(execute_task, args, next(task_iterator)))

        while running:
            completed, running = wait(running, return_when=FIRST_COMPLETED)
            first_error = None
            for future in completed:
                try:
                    future.result()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
            if first_error is not None:
                raise first_error
            for _ in completed:
                try:
                    task = next(task_iterator)
                except StopIteration:
                    break
                running.add(executor.submit(execute_task, args, task))


def write_task_commands(summary, task):
    label = f"{task['strategy']}/{task['stock']}[seed={task['seed']}]"
    if task["pretrain_command"] is None:
        summary.write(f"{label}/pretrain: reused or explicitly skipped\n")
    else:
        summary.write(
            f"{label}/pretrain: {' '.join(task['pretrain_command'])}\n"
        )
    summary.write(f"{label}/downstream: {' '.join(task['eval_command'])}\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Download, pretrain, and evaluate TS-JEPA/GRU for the top "
            "NASDAQ-100 stocks."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "JSON, JSONC, or TOML experiment option file. Reads [common] and "
            "[runner]; "
            "explicit command-line options except stocks/seeds take precedence."
        ),
    )
    parser.add_argument(
        "--stocks",
        nargs="+",
        default=TOP_NASDAQ100_STOCKS,
        help="Stock tickers to run. Defaults to the top NASDAQ-100 holdings.",
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=5,
        help="Limit how many selected stocks to run. Use 0 to run all selected stocks.",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=0,
        help="Limit how many configured seeds to run. Use 0 to run all seeds.",
    )
    parser.add_argument(
        "--max-parallel-jobs",
        type=int,
        default=1,
        help=(
            "Maximum independent pretrain/evaluation task chains to run "
            "concurrently."
        ),
    )
    parser.add_argument("--download-start-date", default="2015-01-01")
    parser.add_argument("--download-end-date", default=date.today().isoformat())
    parser.add_argument(
        "--skip-download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--skip-news",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--skip-pretrain",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--mask-strategies",
        choices=MASK_STRATEGIES,
        nargs="+",
        default=["random"],
        help=(
            "Run one or more mask strategies (default: random). Results are "
            "isolated below RESULTS_DIR/STRATEGY for comparison analysis."
        ),
    )
    parser.add_argument("--lambda-jepa", type=float, default=1.0)
    parser.add_argument("--lambda-mae", type=float, default=0.5)
    parser.add_argument(
        "--jepa-loss",
        choices=("mse", "l1", "smooth_l1"),
        default="mse",
    )
    parser.add_argument(
        "--mae-loss",
        choices=("mse", "l1", "smooth_l1"),
        default="mse",
    )
    parser.add_argument("--mae-window-patches", type=int, default=1)
    parser.add_argument("--jepa-gap-patches", type=int, default=4)
    parser.add_argument("--jepa-target-patches", type=int, default=4)
    parser.add_argument("--future-target-patches", type=int, default=4)
    parser.add_argument("--causal-num-blocks", type=int, default=2)
    parser.add_argument("--causal-block-patches", type=int, default=2)
    parser.add_argument("--causal-block-gap-patches", type=int, default=1)
    parser.add_argument(
        "--series-split-size",
        type=int,
        default=pretrain_defaults["series_split_size"],
        help="Rows per pretraining window before patching.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=pretrain_defaults["patch_size"],
        help="Rows per patch; must divide --series-split-size.",
    )
    parser.add_argument("--pretrain-stride", type=int, default=5)
    parser.add_argument(
        "--sampling-mode",
        choices=("sliding_window", "temporal_segments"),
        default="sliding_window",
        help="Use overlapping windows or non-overlapping temporal segments.",
    )
    parser.add_argument(
        "--normalization",
        choices=("window_return", "train_zscore", "train_robust_zscore", "none"),
        default="window_return",
    )
    parser.add_argument(
        "--preprocessing-preset",
        choices=tuple(PREPROCESSING_PRESETS),
        default=None,
        help="Apply one of the P0-P3 preprocessing ablations.",
    )
    parser.add_argument(
        "--feature-transform",
        choices=("raw", "return"),
        default="raw",
    )
    parser.add_argument("--market-data", default=None)
    parser.add_argument("--robust-zscore-clip", type=float, default=None)
    parser.add_argument("--market-features", nargs="+", default=None)
    parser.add_argument("--sentiment-features", nargs="+", default=None)
    parser.add_argument(
        "--sentiment-normalization",
        choices=("none", "train_zscore"),
        default="none",
        help="Optional train-only transform applied only to derived sentiment channels.",
    )
    sentiment_group = parser.add_mutually_exclusive_group()
    sentiment_group.add_argument(
        "--use-sentiment",
        dest="use_sentiment",
        action="store_true",
        help="Include configured sentiment/news features (default).",
    )
    sentiment_group.add_argument(
        "--no-sentiment",
        dest="use_sentiment",
        action="store_false",
        help="Run the identical pipeline with market features only.",
    )
    parser.set_defaults(use_sentiment=pretrain_defaults["use_sentiment"])
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Ordered pool of reproducible seeds to run.",
    )
    parser.add_argument(
        "--encoder-weights",
        choices=("ema", "online"),
        default="ema",
        help="Checkpoint encoder used in downstream evaluation.",
    )
    parser.add_argument(
        "--forecast-target",
        choices=(
            "value",
            "relative_return",
            "cumulative_log_return",
            "excess_log_return",
        ),
        default="value",
        help="Downstream value path or cutoff-relative return path.",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=None,
        help=(
            "Number of downstream forecast steps. Defaults to --patch-size "
            "for backward compatibility."
        ),
    )
    parser.add_argument("--eval-num-epochs", type=int, default=501)
    parser.add_argument("--pretrain-num-epochs", type=int, default=2001)
    parser.add_argument("--checkpoint-to-use", type=int, default=2000)
    parser.add_argument(
        "--use-best-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate the deterministic best-validation checkpoint instead of an epoch.",
    )
    parser.add_argument("--max-news-articles", type=int, default=None)
    parser.add_argument("--news-chunk-days", type=int, default=7)
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument("--write-mode", choices=["append", "overwrite"], default="append")
    parser.add_argument(
        "--results-dir",
        default="./results",
        help=(
            "Standalone-run output root. Configured runs derive "
            "results/<config filename>; strategy runs use "
            "STRATEGY/TICKER/seed_N below that root."
        ),
    )
    parser.add_argument(
        "--skip-combined-plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do not generate the combined stock metrics CSV and PNG after evaluation.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Do not execute commands; combine with --verbose to print the "
            "generated commands."
        ),
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Show configuration summaries, commands, and child-process output. "
            "By default, show only per-run stock/seed status."
        ),
    )
    args = parse_args_with_config(parser, argv, section="runner")
    if args.max_parallel_jobs <= 0:
        parser.error("--max-parallel-jobs must be positive")
    return args


def resolve_mask_strategies(args):
    strategies = list(args.mask_strategies)
    if not strategies:
        raise ValueError("At least one mask strategy must be configured")
    if len(strategies) != len(set(strategies)):
        raise ValueError("Mask strategies must be unique")
    return strategies


def resolve_stocks(args):
    stocks = [str(stock).upper() for stock in args.stocks]
    max_stocks = int(args.max_stocks)
    if max_stocks < 0:
        raise ValueError("--max-stocks must be >= 0")
    if len(stocks) != len(set(stocks)):
        raise ValueError("--stocks must contain unique tickers")
    if max_stocks > 0:
        stocks = stocks[:max_stocks]
    return stocks


def resolve_seeds(args):
    seeds = list(args.seeds)
    max_seeds = int(args.max_seeds)
    if max_seeds < 0:
        raise ValueError("--max-seeds must be >= 0")
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique values")
    if max_seeds > 0:
        seeds = seeds[:max_seeds]
    return seeds


def validate_runner_mask_geometry(args, strategies):
    series_split_size = int(args.series_split_size)
    patch_size = int(args.patch_size)
    if series_split_size <= 0 or patch_size <= 0:
        raise ValueError("--series-split-size and --patch-size must be positive")
    if series_split_size % patch_size != 0:
        raise ValueError(
            "--series-split-size must be divisible by --patch-size: "
            f"series_split_size={series_split_size}, patch_size={patch_size}"
        )

    num_patches = series_split_size // patch_size
    if num_patches < 2:
        raise ValueError(
            "At least two patches are required: "
            f"series_split_size={series_split_size}, patch_size={patch_size}"
        )

    for strategy in strategies:
        if strategy == "random":
            continue
        strategy_config = {
            "mask_strategy": strategy,
            "mae_window_patches": args.mae_window_patches,
            "jepa_gap_patches": args.jepa_gap_patches,
            "jepa_target_patches": args.jepa_target_patches,
            "future_target_patches": args.future_target_patches,
            "causal_num_blocks": args.causal_num_blocks,
            "causal_block_patches": args.causal_block_patches,
            "causal_block_gap_patches": args.causal_block_gap_patches,
            "anchor_strategy": pretrain_defaults["anchor_strategy"],
            "fixed_anchor": pretrain_defaults["fixed_anchor"],
        }
        try:
            validate_strategy_config(strategy_config, num_patches)
        except ValueError as exc:
            advice = ""
            if strategy == "local_long":
                required_patches = (
                    int(args.jepa_gap_patches)
                    + int(args.jepa_target_patches)
                )
                required_rows = required_patches * patch_size
                advice = (
                    f" Increase --series-split-size to at least {required_rows} "
                    "rows or reduce the gap/target geometry."
                )
            raise ValueError(
                f"Invalid {strategy!r} mask geometry for {num_patches} patches "
                f"(--series-split-size={series_split_size}, "
                f"--patch-size={patch_size}): {exc}{advice}"
            ) from exc


def strategy_results_dir(args, strategy):
    return Path(args.results_dir) / strategy


def resolve_preprocessing_settings(args):
    preset_name = getattr(args, "preprocessing_preset", None)
    if preset_name is not None:
        settings = dict(PREPROCESSING_PRESETS[preset_name])
    else:
        settings = {
            "feature_transform": getattr(args, "feature_transform", "raw"),
            "normalization": getattr(args, "normalization", "window_return"),
            "forecast_target": getattr(args, "forecast_target", "value"),
            "market_data": getattr(args, "market_data", None),
        }
    settings["robust_zscore_clip"] = getattr(args, "robust_zscore_clip", None)
    settings["market_features"] = getattr(args, "market_features", None)
    settings["sentiment_features"] = getattr(args, "sentiment_features", None)
    settings["sentiment_normalization"] = getattr(
        args,
        "sentiment_normalization",
        "none",
    )
    settings["use_sentiment"] = bool(
        getattr(args, "use_sentiment", pretrain_defaults["use_sentiment"])
    )
    if (
        settings["use_sentiment"]
        and preset_name in ("P1", "P2", "P3")
        and settings["sentiment_features"] is None
    ):
        # Return mode always adds the eight canonical price/volume features;
        # these names opt into the two optional sentiment features.
        settings["sentiment_features"] = ["sentiment_mean", "news_count"]
    return settings


def build_stock_commands(args, stock, seed=None, strategy=None, results_dir=None):
    commands = []
    seed = resolve_seeds(args)[0] if seed is None else seed
    if strategy is None:
        strategies = resolve_mask_strategies(args)
        if len(strategies) != 1:
            raise ValueError(
                "strategy is required when more than one mask strategy is configured"
            )
        strategy = strategies[0]
    sampling_mode = getattr(args, "sampling_mode", "sliding_window")
    preprocessing = resolve_preprocessing_settings(args)
    preprocessing_args = [
        "--feature-transform",
        preprocessing["feature_transform"],
        "--normalization",
        preprocessing["normalization"],
        "--market-data",
        str(preprocessing["market_data"] or "none"),
        "--sentiment-normalization",
        preprocessing["sentiment_normalization"],
    ]
    preprocessing_args.append(
        "--use-sentiment"
        if preprocessing["use_sentiment"]
        else "--no-sentiment"
    )
    if preprocessing["robust_zscore_clip"] is not None:
        preprocessing_args.extend(
            ["--robust-zscore-clip", str(preprocessing["robust_zscore_clip"])]
        )
    if preprocessing["market_features"] is not None:
        preprocessing_args.extend(
            ["--market-features", *map(str, preprocessing["market_features"])]
        )
    if preprocessing["sentiment_features"] is not None:
        preprocessing_args.extend(
            [
                "--sentiment-features",
                *map(str, preprocessing["sentiment_features"]),
            ]
        )
    strategy_args = ["--mask-strategy", strategy]
    if strategy == "local_long":
        strategy_args.extend(
            [
                "--mae-window-patches",
                str(args.mae_window_patches),
                "--jepa-gap-patches",
                str(args.jepa_gap_patches),
                "--jepa-target-patches",
                str(args.jepa_target_patches),
            ]
        )
    elif strategy == "future_block":
        strategy_args.extend(
            ["--future-target-patches", str(args.future_target_patches)]
        )
    elif strategy == "causal_multiblock":
        strategy_args.extend(
            [
                "--causal-num-blocks",
                str(args.causal_num_blocks),
                "--causal-block-patches",
                str(args.causal_block_patches),
                "--causal-block-gap-patches",
                str(args.causal_block_gap_patches),
            ]
        )

    pretrain_command = [
        sys.executable,
        "-u",
        "pretrain_dual_loss.py",
        "--no-run-eval",
        "--data",
        stock,
        *strategy_args,
        "--num_epochs",
        str(args.pretrain_num_epochs),
        "--lambda_jepa",
        str(args.lambda_jepa),
        "--lambda_mae",
        str(args.lambda_mae),
        "--jepa-loss",
        args.jepa_loss,
        "--mae-loss",
        args.mae_loss,
        "--series-split-size",
        str(getattr(args, "series_split_size", pretrain_defaults["series_split_size"])),
        "--patch-size",
        str(getattr(args, "patch_size", pretrain_defaults["patch_size"])),
        "--pretrain-stride",
        str(args.pretrain_stride),
        *preprocessing_args,
        "--sampling-mode",
        sampling_mode,
        "--seed",
        str(seed),
    ]
    if preprocessing["use_sentiment"]:
        pretrain_command.extend(
            [
                "--sentiment-path",
                str(Path("data") / stock / f"{stock}_daily_sentiment.csv"),
            ]
        )

    resolved_pretrain_config = parse_pretrain_args(
        copy.deepcopy(pretrain_defaults),
        argv=pretrain_command[3:],
    )
    if getattr(args, "use_best_checkpoint", False):
        checkpoint_path = resolved_pretrain_config["path_save"] + "_best.pt"
    else:
        checkpoint_path = (
            resolved_pretrain_config["path_save"]
            + "_epoch_"
            + str(args.checkpoint_to_use)
            + ".pt"
        )
    checkpoint_args = ["--pretrain-checkpoint-path", checkpoint_path]
    if args.skip_pretrain and not Path(checkpoint_path).exists():
        # Let eval_dual_loss resolve a legacy non-fingerprinted checkpoint.
        checkpoint_args = []

    if not args.skip_pretrain:
        commands.append(pretrain_command)

    stock_results_dir = Path(results_dir or args.results_dir)
    if getattr(args, "preprocessing_preset", None):
        stock_results_dir /= args.preprocessing_preset
    stock_results_dir = stock_results_dir / stock / f"seed_{seed}"

    commands.append(
        [
            sys.executable,
            "-u",
            "eval_dual_loss.py",
            "--data",
            stock,
            *strategy_args,
            "--checkpoint_to_use",
            str(args.checkpoint_to_use),
            *checkpoint_args,
            "--pretrain-encoder-weights",
            args.encoder_weights,
            "--forecast-target",
            preprocessing["forecast_target"],
            *(
                ["--forecast-horizon", str(args.forecast_horizon)]
                if getattr(args, "forecast_horizon", None) is not None
                else []
            ),
            *preprocessing_args,
            "--num_epochs",
            str(args.eval_num_epochs),
            "--seed",
            str(seed),
            "--sampling-mode",
            sampling_mode,
            "--results-dir",
            str(stock_results_dir),
            "--lambda_jepa",
            str(args.lambda_jepa),
            "--lambda_mae",
            str(args.lambda_mae),
        ]
    )

    return commands


def build_combined_plot_command(
    args,
    stocks,
    results_dir=None,
    strategy=None,
):
    output_prefix = f"top_{len(stocks)}_nasdaq100"
    if strategy is not None:
        output_prefix += f"_{strategy}"
    seeds = resolve_seeds(args)
    results_dir = str(results_dir or args.results_dir)
    return [
        sys.executable,
        "-u",
        "plot_top_stock_metrics.py",
        "--results-dir",
        results_dir,
        "--output-dir",
        results_dir,
        "--output-prefix",
        output_prefix,
        "--figure-title",
        output_prefix,
        "--seeds",
        *[str(seed) for seed in seeds],
        "--stocks",
        *stocks,
    ]


def current_git_branch(repo_root=None):
    """Read the current branch without launching a subprocess."""
    repo_root = Path(repo_root or Path(__file__).resolve().parent)
    git_path = repo_root / ".git"
    if git_path.is_file():
        pointer = git_path.read_text(encoding="utf-8").strip()
        if not pointer.startswith("gitdir:"):
            return "unknown"
        git_path = (git_path.parent / pointer.split(":", 1)[1].strip()).resolve()
    head = (git_path / "HEAD").read_text(encoding="utf-8").strip()
    prefix = "ref: refs/heads/"
    return head[len(prefix):] if head.startswith(prefix) else head[:12]


def build_dry_run_report(args, stocks, seeds, strategies):
    """Validate inputs and describe a run without creating or executing anything."""
    preprocessing = resolve_preprocessing_settings(args)
    market_features = list(
        preprocessing.get("market_features")
        or ["Close", "Volume", "MA10", "MA50"]
    )
    sentiment_features = list(preprocessing.get("sentiment_features") or [])
    feature_names = effective_feature_columns(
        market_features,
        sentiment_features or ["sentiment_mean"],
        preprocessing["use_sentiment"],
    )
    forecast_horizon = resolve_forecast_horizon(
        getattr(args, "forecast_horizon", None),
        args.patch_size,
    )

    repo_root = Path(__file__).resolve().parent
    missing_paths = []
    for stock in stocks:
        price_path = repo_root / "data" / stock / f"{stock}.csv"
        if not price_path.is_file():
            missing_paths.append(price_path)
        if preprocessing["use_sentiment"]:
            sentiment_path = (
                repo_root / "data" / stock / f"{stock}_daily_sentiment.csv"
            )
            if not sentiment_path.is_file():
                missing_paths.append(sentiment_path)
    if missing_paths:
        rendered = "\n  ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Dry-run validation missing required input files:\n  " + rendered
        )

    if not preprocessing["use_sentiment"]:
        sentiment_handling = "disabled"
    elif preprocessing["sentiment_normalization"] == "train_zscore":
        sentiment_handling = "same-date merge with per-stock train-only z-score"
    else:
        sentiment_handling = "same-date raw sentiment/news features"

    return {
        "experiment_name": (
            Path(args.config).stem if args.config else "standalone"
        ),
        "git_branch": current_git_branch(repo_root),
        "stock_count": len(stocks),
        "stocks": list(stocks),
        "seed_count": len(seeds),
        "seeds": list(seeds),
        "mask_strategies": list(strategies),
        "forecast_horizon": forecast_horizon,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "patch_size": int(args.patch_size),
        "flattened_patch_input_dimension": int(args.patch_size)
        * len(feature_names),
        "sentiment_handling": sentiment_handling,
        "sentiment_normalization": preprocessing["sentiment_normalization"],
        "normalization_mode": preprocessing["normalization"],
        "output_directory": str(Path(args.results_dir)),
        "training_disabled": True,
    }


def main(argv=None):
    args = parse_args(argv)
    preprocessing = resolve_preprocessing_settings(args)
    stocks = resolve_stocks(args)
    seeds = resolve_seeds(args)
    strategies = resolve_mask_strategies(args)
    validate_runner_mask_geometry(args, strategies)

    validate_config_result_mapping(args)
    requested_stock_seed_runs(stocks, seeds)
    legacy_manifest_compatible = validate_existing_experiment(args)
    if not legacy_manifest_compatible:
        reject_duplicate_experiment_config(args)
    execution_plan = plan_incremental_execution(
        args,
        stocks,
        seeds,
        strategies,
        legacy_manifest_compatible=legacy_manifest_compatible,
    )

    requested_count = len(execution_plan["requested_runs"])
    completed_count = len(execution_plan["completed_runs"])
    missing_count = len(execution_plan["missing_runs"])
    if args.verbose:
        if legacy_manifest_compatible:
            print("Configuration unchanged.", flush=True)
        else:
            print("No existing compatible experiment configuration.", flush=True)
        print(f"Requested coverage: {requested_count} runs.", flush=True)
        print(f"Completed compatible runs: {completed_count}.", flush=True)
        print(f"Missing runs: {missing_count}.", flush=True)

    if args.dry_run:
        report = build_dry_run_report(args, stocks, seeds, strategies)
        print("DRY_RUN_VALIDATION")
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if not execution_plan["tasks"]:
        if not args.dry_run:
            current_manifest = build_experiment_manifest(
                args,
                stocks,
                seeds,
                strategies,
            )
            manifest_path = Path(args.results_dir) / "experiment_manifest.json"
            previous_manifest = (
                _load_json(manifest_path) if manifest_path.exists() else None
            )
            if previous_manifest != current_manifest:
                _write_json(manifest_path, current_manifest)
        if args.verbose:
            print("Nothing to run.", flush=True)
        return

    missing_stocks = list(
        dict.fromkeys(
            stock
            for stock, seed in execution_plan["requested_runs"]
            if (stock, seed) in execution_plan["missing_runs"]
        )
    )
    if not args.skip_download:
        download_cmd = [
            sys.executable,
            "download_indices_and_news.py",
            "--skip-indices",
            "--stocks",
            *missing_stocks,
            "--download-start-date",
            args.download_start_date,
            "--download-end-date",
            args.download_end_date,
            "--news-chunk-days",
            str(args.news_chunk_days),
            "--request-delay",
            str(args.request_delay),
            "--write-mode",
            args.write_mode,
        ]
        if args.skip_news or not preprocessing["use_sentiment"]:
            download_cmd.append("--skip-news")
        if args.max_news_articles is not None:
            download_cmd.extend(["--max-news-articles", str(args.max_news_articles)])
        run_command(
            download_cmd,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

    summary_path = Path(args.results_dir) / "top_nasdaq100_stock_runs.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_experiment_manifest(args, stocks, seeds, strategies)
    manifest_path = Path(args.results_dir) / "experiment_manifest.json"
    if not args.dry_run:
        _write_json(manifest_path, manifest)

    with summary_path.open("w") as summary:
        summary.write("Top NASDAQ-100 stock workflow\n")
        summary.write(f"stocks={','.join(stocks)}\n")
        summary.write(f"download_start_date={args.download_start_date}\n")
        summary.write(f"download_end_date={args.download_end_date}\n")
        summary.write(f"results_dir={args.results_dir}\n")
        summary.write("pretrain_script=pretrain_dual_loss.py\n")
        summary.write("eval_script=eval_dual_loss.py\n")
        summary.write(f"mask_strategies={','.join(strategies)}\n")
        summary.write(f"seeds={','.join(str(seed) for seed in seeds)}\n")
        summary.write(
            f"preprocessing_preset={getattr(args, 'preprocessing_preset', None)}\n"
        )
        summary.write(f"feature_transform={preprocessing['feature_transform']}\n")
        summary.write(f"normalization={preprocessing['normalization']}\n")
        summary.write(f"market_data={preprocessing['market_data']}\n")
        summary.write(f"use_sentiment={preprocessing['use_sentiment']}\n")
        summary.write(f"market_features={preprocessing['market_features']}\n")
        summary.write(
            f"sentiment_features={preprocessing['sentiment_features']}\n"
        )
        summary.write(f"sampling_mode={args.sampling_mode}\n")
        summary.write(f"pretrain_stride={args.pretrain_stride}\n")
        summary.write(f"encoder_weights={args.encoder_weights}\n")
        summary.write(
            f"forecast_target={preprocessing['forecast_target']}\n"
        )
        summary.write(f"use_best_checkpoint={args.use_best_checkpoint}\n")
        summary.write(f"lambda_jepa={args.lambda_jepa}\n")
        summary.write(f"lambda_mae={args.lambda_mae}\n")
        summary.write(f"jepa_loss={args.jepa_loss}\n")
        summary.write(f"mae_loss={args.mae_loss}\n")
        summary.write(f"eval_num_epochs={args.eval_num_epochs}\n\n")

        for task in execution_plan["tasks"]:
            write_task_commands(summary, task)
        summary.flush()
        execute_tasks(args, execution_plan["tasks"])

        if not args.skip_combined_plot:
            for strategy in strategies:
                strategy_dir = strategy_results_dir(args, strategy)
                plot_strategy = strategy if len(strategies) > 1 else None
                combined_plot_cmd = build_combined_plot_command(
                    args,
                    stocks,
                    results_dir=strategy_dir,
                    strategy=plot_strategy,
                )
                summary.write(
                    f"combined_plot[{strategy}]: {' '.join(combined_plot_cmd)}\n"
                )
                summary.flush()
                run_command(
                    combined_plot_cmd,
                    dry_run=args.dry_run,
                    verbose=args.verbose,
                )

    if args.verbose:
        print(f"Run summary saved to {summary_path}", flush=True)
        if args.dry_run:
            print("Dry run: experiment/run manifests were not written", flush=True)
        else:
            print(f"Experiment manifest saved to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
