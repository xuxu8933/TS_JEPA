"""Run the complete Chapter 5 validation-selection workflow."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

from chapter5_prepare_candidates import materialize_candidates
from chapter5_selection import load_validation_coverage, main as run_selection
from run_top_nasdaq100_stocks import (
    build_experiment_manifest,
    execute_tasks,
    experiment_config_signature,
    parse_args as parse_runner_args,
    plan_incremental_execution,
    resolve_mask_strategies,
    resolve_seeds,
    resolve_stocks,
    strategy_results_dir,
    validate_existing_experiment,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_DIR = "selection_artifacts/chapter5_automated"
REPAIR_ARTIFACTS_DIR = "selection_artifacts/chapter5_stage3_validation_repair"
CHECKED_IN_CANDIDATE_DIR = (
    REPO_ROOT / "config" / "experiments" / "chapter5_candidates"
)
STAGE_ONE = (
    (
        "preprocessing_window_return",
        CHECKED_IN_CANDIDATE_DIR / "01_preprocessing_window_return.json",
    ),
    (
        "preprocessing_train_zscore",
        CHECKED_IN_CANDIDATE_DIR / "01_preprocessing_train_zscore.json",
    ),
)
SENTIMENT_IDS = ("sentiment_excluded", "sentiment_included")
ARCHITECTURE_CONTEXT_IDS = (
    "shared_context_6",
    "shared_context_12",
    "shared_context_24",
    "local_long_context_6",
    "local_long_context_12",
    "local_long_context_24",
)
ARCHITECTURE_CONTEXT_STRATEGIES = (
    "random",
    "random",
    "random",
    "local_long",
    "local_long",
    "local_long",
)
CandidateExecutor = Callable[..., None]


def _run_candidate(config_path: Path, *, dry_run: bool) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "run_top_nasdaq100_stocks.py"),
        "--config",
        str(Path(config_path).resolve()),
    ]
    if dry_run:
        command.extend(["--dry-run", "--verbose"])
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validation_root(config_path: Path, strategy: str) -> Path:
    args = parse_runner_args(["--config", str(Path(config_path).resolve())])
    if strategy not in resolve_mask_strategies(args):
        raise ValueError(f"Strategy {strategy!r} is not enabled by {config_path}")
    root = strategy_results_dir(args, strategy)
    if args.preprocessing_preset:
        root /= args.preprocessing_preset
    return root.resolve()


def _candidate_entry(
    candidate_id: str,
    config_path: Path,
    strategy: str,
    *,
    parent_candidate_id: str | None = None,
) -> dict[str, Any]:
    entry = {
        "id": candidate_id,
        "config": str(Path(config_path).resolve()),
        "validation_root": str(_validation_root(config_path, strategy)),
        "strategy": strategy,
    }
    if parent_candidate_id is not None:
        entry["parent_candidate_id"] = parent_candidate_id
    return entry


def _selection_manifest(
    selection_id: str,
    stages: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "selection_id": selection_id,
        "stages": list(stages),
    }


def _select_stage(
    artifacts_dir: Path,
    stage_directory: str,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    output_dir = artifacts_dir / stage_directory
    manifest_path = output_dir / "selection_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    run_selection(
        [
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    summary_path = output_dir / "selection_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary, output_dir


def _resolved_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _candidate_coverage(candidate: dict[str, Any], manifest_root: Path) -> tuple:
    args = parse_runner_args(
        ["--config", str(_resolved_path(manifest_root, candidate["config"]))]
    )
    validation_root = _resolved_path(manifest_root, candidate["validation_root"])
    return tuple(
        (
            stock,
            seed,
            *load_validation_coverage(
                validation_root
                / stock
                / f"seed_{seed}"
                / "validation_metrics.json"
            ).values(),
        )
        for stock in resolve_stocks(args)
        for seed in resolve_seeds(args)
    )


def repair_stage3_validation(
    *,
    manifest_path: Path,
    artifacts_dir: Path,
    repair_results_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Rerun only Stage 3 candidates whose validation coverage is inconsistent."""
    resolved_manifest = Path(manifest_path).resolve()
    manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    stage = next(
        item
        for item in manifest["stages"]
        if item["name"] == "architecture_context"
    )
    candidates = stage["candidates"]
    observed = {}
    for candidate in candidates:
        try:
            observed[candidate["id"]] = _candidate_coverage(
                candidate, resolved_manifest.parent
            )
        except (OSError, ValueError):
            pass
    if not observed:
        raise RuntimeError("No complete Stage 3 validation coverage was found")
    ranked = Counter(observed.values()).most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        raise RuntimeError("Stage 3 validation coverage has no unique consensus")
    expected_coverage, agreement = ranked[0]
    if agreement <= len(candidates) // 2:
        raise RuntimeError("Stage 3 validation coverage has no majority consensus")

    repaired_manifest = copy.deepcopy(manifest)
    repaired_stage = next(
        item
        for item in repaired_manifest["stages"]
        if item["name"] == "architecture_context"
    )
    repaired_by_id = {
        candidate["id"]: candidate for candidate in repaired_stage["candidates"]
    }
    affected = [
        candidate
        for candidate in candidates
        if observed.get(candidate["id"]) != expected_coverage
    ]
    reused = [
        candidate["id"] for candidate in candidates if candidate not in affected
    ]

    if not dry_run:
        for candidate in affected:
            config_path = _resolved_path(resolved_manifest.parent, candidate["config"])
            args = parse_runner_args(["--config", str(config_path)])
            candidate_root = Path(repair_results_dir).resolve() / config_path.stem
            args.results_dir = str(candidate_root)
            args.skip_pretrain = True
            args.skip_combined_plot = True
            stocks = resolve_stocks(args)
            seeds = resolve_seeds(args)
            strategies = resolve_mask_strategies(args)
            legacy_compatible = validate_existing_experiment(args)
            plan = plan_incremental_execution(
                args,
                stocks,
                seeds,
                strategies,
                legacy_manifest_compatible=legacy_compatible,
            )
            _write_json_atomic(
                candidate_root / "experiment_manifest.json",
                build_experiment_manifest(args, stocks, seeds, strategies),
            )
            execute_tasks(args, plan["tasks"])
            validation_root = strategy_results_dir(args, candidate["strategy"])
            repaired_entry = repaired_by_id[candidate["id"]]
            repaired_entry["validation_root"] = str(validation_root.resolve())
            if (
                _candidate_coverage(repaired_entry, resolved_manifest.parent)
                != expected_coverage
            ):
                raise RuntimeError(
                    f"Repaired coverage is still invalid for {candidate['id']}"
                )

        resolved_artifacts = Path(artifacts_dir).resolve()
        repaired_manifest_path = resolved_artifacts / "selection_manifest.json"
        _write_json_atomic(repaired_manifest_path, repaired_manifest)
        run_selection(
            [
                "--manifest",
                str(repaired_manifest_path),
                "--output-dir",
                str(resolved_artifacts),
            ]
        )
        selected = json.loads(
            (resolved_artifacts / "selection_summary.json").read_text(
                encoding="utf-8"
            )
        )["selected_candidate_id"]
    else:
        selected = None

    return {
        "artifact_type": "chapter5_stage3_validation_repair",
        "schema_version": 1,
        "affected_candidates": [candidate["id"] for candidate in affected],
        "reused_candidates": reused,
        "pretraining_reused": True,
        "test_evaluation_performed": False,
        "selected_candidate_id": selected,
    }


def _run_candidates(
    paths: Sequence[Path],
    candidate_executor: CandidateExecutor,
    parent_entry: dict[str, Any] | None = None,
) -> dict[Path, str]:
    reused = {}
    parent_signature = None
    if parent_entry is not None:
        parent_args = parse_runner_args(["--config", parent_entry["config"]])
        parent_signature = experiment_config_signature(parent_args)
    for path in paths:
        args = parse_runner_args(["--config", str(Path(path).resolve())])
        if experiment_config_signature(args) == parent_signature:
            reused[Path(path).resolve()] = parent_entry["validation_root"]
            continue
        candidate_executor(path, dry_run=False)
    return reused


def _apply_reused_roots(
    entries: Sequence[dict[str, Any]], reused: dict[Path, str]
) -> None:
    for entry in entries:
        root = reused.get(Path(entry["config"]).resolve())
        if root is not None:
            entry["validation_root"] = root


def run_complete_process(
    *,
    artifacts_dir: Path,
    candidate_dir: Path = CHECKED_IN_CANDIDATE_DIR,
    stage_one: Sequence[tuple[str, Path]] = STAGE_ONE,
    candidate_executor: CandidateExecutor = _run_candidate,
) -> dict[str, Any]:
    """Run three validation stages, then evaluate the frozen test config once."""
    resolved_artifacts = Path(artifacts_dir).resolve()
    resolved_candidate_dir = Path(candidate_dir).resolve()
    stage_one = tuple(
        (candidate_id, Path(path).resolve())
        for candidate_id, path in stage_one
    )
    if len(stage_one) != 2:
        raise ValueError("Stage 1 requires exactly two normalization candidates")

    stage_one_paths = [path for _, path in stage_one]
    _run_candidates(stage_one_paths, candidate_executor)
    stage_one_entries = [
        _candidate_entry(candidate_id, path, "random")
        for candidate_id, path in stage_one
    ]
    stage_one_stage = {
        "name": "preprocessing_normalization",
        "candidates": stage_one_entries,
    }
    stage_one_summary, stage_one_output = _select_stage(
        resolved_artifacts,
        "stage1",
        _selection_manifest("chapter5_stage1", [stage_one_stage]),
    )
    stage_one_parent = stage_one_summary["selected_candidate_id"]

    sentiment_paths = materialize_candidates(
        "sentiment",
        stage_one_output / "selected_stage_config.json",
        stage_one_parent,
        resolved_candidate_dir,
    )
    stage_one_selected = next(
        entry
        for entry in stage_one_entries
        if entry["id"] == stage_one_parent
    )
    reused = _run_candidates(
        sentiment_paths, candidate_executor, stage_one_selected
    )
    sentiment_entries = [
        _candidate_entry(
            candidate_id,
            path,
            "random",
            parent_candidate_id=stage_one_parent,
        )
        for candidate_id, path in zip(SENTIMENT_IDS, sentiment_paths)
    ]
    _apply_reused_roots(sentiment_entries, reused)
    sentiment_stage = {"name": "sentiment", "candidates": sentiment_entries}
    stage_two_summary, stage_two_output = _select_stage(
        resolved_artifacts,
        "stage2",
        _selection_manifest(
            "chapter5_stage2",
            [stage_one_stage, sentiment_stage],
        ),
    )
    stage_two_parent = stage_two_summary["selected_candidate_id"]

    architecture_paths = materialize_candidates(
        "architecture_context",
        stage_two_output / "selected_stage_config.json",
        stage_two_parent,
        resolved_candidate_dir,
    )
    stage_two_selected = next(
        entry
        for entry in sentiment_entries
        if entry["id"] == stage_two_parent
    )
    reused = _run_candidates(
        architecture_paths, candidate_executor, stage_two_selected
    )
    architecture_entries = [
        _candidate_entry(
            candidate_id,
            path,
            strategy,
            parent_candidate_id=stage_two_parent,
        )
        for candidate_id, strategy, path in zip(
            ARCHITECTURE_CONTEXT_IDS,
            ARCHITECTURE_CONTEXT_STRATEGIES,
            architecture_paths,
        )
    ]
    _apply_reused_roots(architecture_entries, reused)
    architecture_stage = {
        "name": "architecture_context",
        "candidates": architecture_entries,
    }
    final_summary, final_output = _select_stage(
        resolved_artifacts,
        "final",
        _selection_manifest(
            "chapter5_final",
            [stage_one_stage, sentiment_stage, architecture_stage],
        ),
    )
    selected_config = final_output / "selected_config.json"
    selected_args = parse_runner_args(["--config", str(selected_config)])
    if selected_args.evaluation_split != "test":
        raise RuntimeError("Frozen configuration is not test-only")
    candidate_executor(selected_config, dry_run=False)

    report = {
        "artifact_type": "chapter5_automated_workflow",
        "schema_version": 1,
        "stage_winners": [
            stage_one_parent,
            stage_two_parent,
            final_summary["selected_candidate_id"],
        ],
        "selected_candidate_id": final_summary["selected_candidate_id"],
        "selected_config": str(selected_config.resolve()),
        "selection_summary": str(
            (final_output / "selection_summary.json").resolve()
        ),
        "validation_candidate_count": 10,
        "test_evaluation_performed": True,
    }
    _write_json_atomic(resolved_artifacts / "workflow_summary.json", report)
    return report


def run_dry_run() -> dict[str, object]:
    """Validate all candidate runner commands without selecting or testing."""
    with tempfile.TemporaryDirectory(prefix="chapter5_dry_run_") as temporary:
        generated_dir = Path(temporary) / "chapter5_candidates"
        stage_one_paths = [path for _, path in STAGE_ONE]
        for path in stage_one_paths:
            _run_candidate(path, dry_run=True)

        sentiment_paths = materialize_candidates(
            "sentiment",
            stage_one_paths[0],
            STAGE_ONE[0][0],
            generated_dir,
        )
        for path in sentiment_paths:
            _run_candidate(path, dry_run=True)

        architecture_paths = materialize_candidates(
            "architecture_context",
            sentiment_paths[1],
            "sentiment_included",
            generated_dir,
        )
        for path in architecture_paths:
            _run_candidate(path, dry_run=True)

        validated_candidate_count = (
            len(stage_one_paths) + len(sentiment_paths) + len(architecture_paths)
        )

    return {
        "artifact_type": "chapter5_automation_dry_run",
        "schema_version": 1,
        "validated_candidate_count": validated_candidate_count,
        "stage_candidate_counts": [
            len(stage_one_paths),
            len(sentiment_paths),
            len(architecture_paths),
        ],
        "selection_performed": False,
        "test_evaluation_performed": False,
        "temporary_config_lineage": {
            "stage_1_parent": STAGE_ONE[0][0],
            "stage_2_parent": "sentiment_included",
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Automate the three-stage Chapter 5 experiment workflow."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repair-stage3", action="store_true")
    parser.add_argument(
        "--artifacts-dir",
        default=None,
    )
    parser.add_argument(
        "--manifest",
        default="selection_artifacts/chapter5_automated/final/selection_manifest.json",
    )
    parser.add_argument(
        "--repair-results-dir",
        default="results/chapter5_stage3_validation_repair",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.repair_stage3:
        artifacts_dir = args.artifacts_dir or REPAIR_ARTIFACTS_DIR
        report = repair_stage3_validation(
            manifest_path=REPO_ROOT / args.manifest,
            artifacts_dir=REPO_ROOT / artifacts_dir,
            repair_results_dir=REPO_ROOT / args.repair_results_dir,
            dry_run=args.dry_run,
        )
        marker = "CHAPTER5_STAGE3_REPAIR_SUMMARY"
    elif args.dry_run:
        report = run_dry_run()
        marker = "CHAPTER5_DRY_RUN_SUMMARY"
    else:
        artifacts_dir = Path(args.artifacts_dir or DEFAULT_ARTIFACTS_DIR)
        if not artifacts_dir.is_absolute():
            artifacts_dir = REPO_ROOT / artifacts_dir
        report = run_complete_process(artifacts_dir=artifacts_dir)
        marker = "CHAPTER5_WORKFLOW_SUMMARY"
    print(marker)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
