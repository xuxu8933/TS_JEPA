"""Run the complete Chapter 5 validation-selection workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

from chapter5_prepare_candidates import materialize_candidates
from chapter5_selection import main as run_selection
from run_top_nasdaq100_stocks import (
    parse_args as parse_runner_args,
    resolve_mask_strategies,
    strategy_results_dir,
)


REPO_ROOT = Path(__file__).resolve().parent
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


def _run_candidates(
    paths: Sequence[Path],
    candidate_executor: CandidateExecutor,
) -> None:
    for path in paths:
        candidate_executor(path, dry_run=False)


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
    _run_candidates(sentiment_paths, candidate_executor)
    sentiment_entries = [
        _candidate_entry(
            candidate_id,
            path,
            "random",
            parent_candidate_id=stage_one_parent,
        )
        for candidate_id, path in zip(SENTIMENT_IDS, sentiment_paths)
    ]
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
    _run_candidates(architecture_paths, candidate_executor)
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
    parser.add_argument(
        "--artifacts-dir",
        default="selection_artifacts/chapter5_automated",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        report = run_dry_run()
        marker = "CHAPTER5_DRY_RUN_SUMMARY"
    else:
        artifacts_dir = Path(args.artifacts_dir)
        if not artifacts_dir.is_absolute():
            artifacts_dir = REPO_ROOT / artifacts_dir
        report = run_complete_process(artifacts_dir=artifacts_dir)
        marker = "CHAPTER5_WORKFLOW_SUMMARY"
    print(marker)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
