"""Deterministic, validation-only experiment selection for Chapter 5."""

from __future__ import annotations

import argparse
import copy
import json
import hashlib
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from config.file_options import read_config_file
from run_top_nasdaq100_stocks import (
    experiment_config_signature,
    parse_args as parse_runner_args,
    resolve_mask_strategies,
    resolve_seeds,
    resolve_stocks,
)


VALIDATION_ARTIFACT_FILENAME = "validation_metrics.json"
PREPROCESSING_ARTIFACT_FILENAME = "preprocessing_config.json"
METRIC_NAMES = ("rmse", "direction_accuracy")
STAGE_NAMES = (
    "preprocessing_normalization",
    "sentiment",
    "architecture_context",
)


def _contains_test_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            tokens = re.split(r"[^a-z0-9]+", str(key).lower().replace("_", "-"))
            if "test" in tokens or _contains_test_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_test_key(item) for item in value)
    return False


def load_validation_artifact(
    path: Path,
    expected_identity: Mapping[str, Any],
) -> dict[str, float]:
    """Load one exact validation artifact and reject any test-result content."""
    artifact_path = Path(path)
    if artifact_path.name != VALIDATION_ARTIFACT_FILENAME:
        raise ValueError(
            "Selection accepts only files named validation_metrics.json; "
            f"got {artifact_path.name!r}"
        )
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read validation artifact {artifact_path}: {exc}") from exc
    if not isinstance(artifact, dict):
        raise ValueError(f"Validation artifact must be an object: {artifact_path}")
    if _contains_test_key(artifact):
        raise ValueError(
            f"Validation selection rejected test-result key in {artifact_path}"
        )
    expected_fields = {
        "artifact_type",
        "schema_version",
        "artifact_filename",
        "split",
        "model",
        "config_signature",
        "stock",
        "seed",
        "strategy",
        "metrics",
    }
    if set(artifact) != expected_fields:
        raise ValueError(
            f"Unexpected validation artifact fields in {artifact_path}: "
            f"{sorted(set(artifact) - expected_fields)}"
        )
    if artifact.get("split") != "validation":
        raise ValueError(
            "Selection is validation-only; artifact split must be 'validation', "
            f"got {artifact.get('split')!r} in {artifact_path}"
        )
    if artifact.get("artifact_type") != "downstream_forecast_metrics":
        raise ValueError(f"Unexpected artifact_type in {artifact_path}")
    if artifact.get("schema_version") != 2:
        raise ValueError(f"Unsupported validation artifact schema in {artifact_path}")
    if artifact.get("artifact_filename") != VALIDATION_ARTIFACT_FILENAME:
        raise ValueError(f"Artifact filename metadata mismatch in {artifact_path}")
    if artifact.get("model") != "TS-JEPA":
        raise ValueError(f"Selection metric model must be TS-JEPA in {artifact_path}")

    normalized_expected = {
        "config_signature": str(expected_identity["config_signature"]),
        "stock": str(expected_identity["stock"]).upper(),
        "seed": int(expected_identity["seed"]),
        "strategy": str(expected_identity["strategy"]),
    }
    actual_identity = {
        "config_signature": str(artifact.get("config_signature", "")),
        "stock": str(artifact.get("stock", "")).upper(),
        "seed": artifact.get("seed"),
        "strategy": str(artifact.get("strategy", "")),
    }
    try:
        actual_identity["seed"] = int(actual_identity["seed"])
    except (TypeError, ValueError):
        pass
    if actual_identity != normalized_expected:
        raise ValueError(
            f"Validation artifact identity mismatch in {artifact_path}: "
            f"expected={normalized_expected}, actual={actual_identity}"
        )

    metrics = artifact.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(METRIC_NAMES):
        raise ValueError(
            f"Validation artifact metrics must be exactly {list(METRIC_NAMES)} "
            f"in {artifact_path}"
        )
    normalized_metrics = {}
    for name in METRIC_NAMES:
        try:
            value = float(metrics[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Validation metric {name!r} is not numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"Validation metric {name!r} must be finite")
        normalized_metrics[name] = value
    if not 0.0 <= normalized_metrics["direction_accuracy"] <= 1.0:
        raise ValueError("direction_accuracy must be between 0 and 1")
    return normalized_metrics


def load_validation_coverage(path: Path) -> dict[str, Any]:
    """Load the target coverage associated with one validation metric artifact."""
    metadata_path = Path(path).with_name(PREPROCESSING_ARTIFACT_FILENAME)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read validation coverage metadata {metadata_path}: {exc}"
        ) from exc
    if metadata.get("evaluation_split") != "validation":
        raise ValueError(
            f"Validation coverage split must be 'validation' in {metadata_path}"
        )
    try:
        sample_count = int(metadata["evaluation_sample_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid evaluation_sample_count in {metadata_path}"
        ) from exc
    target_start = metadata.get("evaluation_target_start")
    target_end = metadata.get("evaluation_target_end")
    if sample_count <= 0 or not isinstance(target_start, str) or not isinstance(
        target_end, str
    ):
        raise ValueError(f"Invalid validation target coverage in {metadata_path}")
    return {
        "sample_count": sample_count,
        "target_start": target_start,
        "target_end": target_end,
    }


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def aggregate_candidate(
    metrics_by_stock: Mapping[str, Mapping[int, Mapping[str, float]]],
) -> dict[str, Any]:
    """Average seeds within each stock, then average the stock means."""
    if not metrics_by_stock:
        raise ValueError("Candidate metrics contain no stocks")
    per_stock = {}
    for stock in sorted(metrics_by_stock):
        seed_metrics = metrics_by_stock[stock]
        if not seed_metrics:
            raise ValueError(f"Candidate metrics contain no seeds for {stock}")
        ordered_seeds = sorted(seed_metrics, key=int)
        per_stock[stock] = {
            name: math.fsum(float(seed_metrics[seed][name]) for seed in ordered_seeds)
            / len(ordered_seeds)
            for name in METRIC_NAMES
        }
    ordered_stocks = sorted(per_stock)
    overall = {
        name: math.fsum(per_stock[stock][name] for stock in ordered_stocks)
        / len(ordered_stocks)
        for name in METRIC_NAMES
    }
    return {
        "aggregation_hierarchy": "seeds_within_stock_then_stocks",
        "per_stock": per_stock,
        "overall": overall,
    }


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        _, value = read_config_file(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _resolve_manifest_path(manifest_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest_root / path).resolve()


def _candidate_config(
    candidate: Mapping[str, Any],
    manifest_root: Path,
) -> tuple[Path, dict[str, Any], Any, str]:
    config_path = _resolve_manifest_path(
        manifest_root,
        candidate.get("config"),
        "candidate.config",
    )
    _, raw_config = read_config_file(config_path)
    args = parse_runner_args(["--config", str(config_path)])
    if not args.use_best_checkpoint:
        raise ValueError(
            f"{candidate.get('id')}: checkpoint.selection.mode must be best"
        )
    if args.evaluation_split != "validation":
        raise ValueError(
            f"{candidate.get('id')}: downstream evaluation_split must be validation"
        )
    signature = experiment_config_signature(args)
    return config_path, raw_config, args, signature


def _candidate_summary(
    candidate: Mapping[str, Any],
    manifest_root: Path,
    *,
    eligible: bool,
) -> dict[str, Any]:
    candidate_id = candidate.get("id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("Every candidate requires a non-empty string id")
    allowed = {
        "id",
        "config",
        "validation_root",
        "strategy",
        "parent_candidate_id",
    }
    unknown = sorted(set(candidate) - allowed)
    if unknown:
        raise ValueError(f"Unknown fields for candidate {candidate_id}: {unknown}")
    required = {"id", "config", "validation_root", "strategy"}
    missing = sorted(required - set(candidate))
    if missing:
        raise ValueError(f"Missing fields for candidate {candidate_id}: {missing}")
    config_path, raw_config, args, signature = _candidate_config(
        candidate,
        manifest_root,
    )
    parent_candidate_id = candidate.get("parent_candidate_id")
    parent_config_sha256 = None
    if parent_candidate_id is not None:
        provenance = raw_config.get("provenance")
        if not isinstance(provenance, dict) or provenance.get(
            "artifact_type"
        ) != "chapter5_candidate_config":
            raise ValueError(
                f"{candidate_id}: later-stage config requires candidate provenance"
            )
        if provenance.get("parent_candidate_id") != parent_candidate_id:
            raise ValueError(
                f"{candidate_id}: manifest parent does not match config provenance"
            )
        parent_config_sha256 = provenance.get("parent_config_sha256")
        if not isinstance(parent_config_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", parent_config_sha256
        ) is None:
            raise ValueError(
                f"{candidate_id}: invalid parent_config_sha256 provenance"
            )
    strategy = candidate.get("strategy")
    if strategy not in resolve_mask_strategies(args):
        raise ValueError(
            f"{candidate_id}: strategy {strategy!r} is not enabled by the config"
        )
    stocks = resolve_stocks(args)
    seeds = resolve_seeds(args)
    summary = {
        "id": candidate_id,
        "status": "eligible" if eligible else "ineligible_parent",
        "parent_candidate_id": parent_candidate_id,
        "parent_config_sha256": parent_config_sha256,
        "config": str(candidate["config"]),
        "config_sha256": canonical_sha256(raw_config),
        "config_signature": signature,
        "validation_root": str(candidate["validation_root"]),
        "strategy": strategy,
        "stocks": stocks,
        "seeds": seeds,
        "evaluation_coverage": None,
        "per_stock": None,
        "overall": None,
    }
    if not eligible:
        return summary

    validation_root = _resolve_manifest_path(
        manifest_root,
        candidate.get("validation_root"),
        f"{candidate_id}.validation_root",
    )
    metrics_by_stock = {}
    coverage_by_stock = {}
    for stock in stocks:
        metrics_by_stock[stock] = {}
        stock_coverage = None
        for seed in seeds:
            artifact_path = (
                validation_root
                / stock
                / f"seed_{seed}"
                / VALIDATION_ARTIFACT_FILENAME
            )
            metrics_by_stock[stock][seed] = load_validation_artifact(
                artifact_path,
                {
                    "config_signature": signature,
                    "stock": stock,
                    "seed": seed,
                    "strategy": strategy,
                },
            )
            coverage = load_validation_coverage(artifact_path)
            if stock_coverage is None:
                stock_coverage = coverage
            elif coverage != stock_coverage:
                raise ValueError(
                    f"{candidate_id}: validation target coverage differs across "
                    f"seeds for {stock}"
                )
        coverage_by_stock[stock] = stock_coverage
    aggregation = aggregate_candidate(metrics_by_stock)
    summary["evaluation_coverage"] = coverage_by_stock
    summary["per_stock"] = aggregation["per_stock"]
    summary["overall"] = aggregation["overall"]
    return summary


def _ranking_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    overall = candidate["overall"]
    return (
        overall["rmse"],
        -overall["direction_accuracy"],
        candidate["id"],
    )


def select_stages(manifest_path: Path) -> dict[str, Any]:
    """Select the three Chapter 5 stages using validation artifacts only."""
    resolved_manifest = Path(manifest_path).resolve()
    manifest = _read_object(resolved_manifest, "selection manifest")
    if set(manifest) != {"schema_version", "selection_id", "stages"}:
        raise ValueError(
            "Selection manifest fields must be schema_version, selection_id, stages"
        )
    if manifest.get("schema_version") != 1:
        raise ValueError("Selection manifest schema_version must be 1")
    selection_id = manifest.get("selection_id")
    if not isinstance(selection_id, str) or not selection_id:
        raise ValueError("selection_id must be a non-empty string")
    stages = manifest.get("stages")
    if not isinstance(stages, list):
        raise ValueError("stages must be a list")
    actual_stage_names = [stage.get("name") for stage in stages if isinstance(stage, dict)]
    if not stages or tuple(actual_stage_names) != STAGE_NAMES[: len(stages)]:
        raise ValueError(
            "Stage order must be a non-empty prefix of "
            f"{list(STAGE_NAMES)}, got {actual_stage_names}"
        )

    manifest_root = resolved_manifest.parent
    seen_ids = set()
    previous_winner = None
    previous_stage_ids = set()
    previous_candidates_by_id = {}
    stage_summaries = []
    for stage_index, (expected_name, stage) in enumerate(zip(STAGE_NAMES, stages)):
        if set(stage) != {"name", "candidates"}:
            raise ValueError(f"Stage {expected_name} fields must be name and candidates")
        candidates = stage.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Stage {expected_name} requires candidates")
        candidate_summaries = []
        for candidate in sorted(candidates, key=lambda item: str(item.get("id", ""))):
            candidate_id = candidate.get("id")
            if candidate_id in seen_ids:
                raise ValueError(f"Duplicate candidate id: {candidate_id}")
            seen_ids.add(candidate_id)
            parent = candidate.get("parent_candidate_id")
            if stage_index == 0 and parent is not None:
                raise ValueError("First-stage candidates cannot declare a parent")
            if stage_index > 0 and not isinstance(parent, str):
                raise ValueError(
                    f"{candidate_id}: parent_candidate_id is required after stage 1"
                )
            if stage_index > 0 and parent not in previous_stage_ids:
                raise ValueError(
                    f"{candidate_id}: parent_candidate_id {parent!r} is not a "
                    "candidate in the previous stage"
                )
            eligible = stage_index == 0 or parent == previous_winner
            candidate_summary = _candidate_summary(
                candidate,
                manifest_root,
                eligible=eligible,
            )
            if stage_index > 0:
                expected_parent_hash = previous_candidates_by_id[parent][
                    "config_sha256"
                ]
                if candidate_summary["parent_config_sha256"] != expected_parent_hash:
                    raise ValueError(
                        f"{candidate_id}: parent config hash mismatch for {parent!r}"
                    )
            candidate_summaries.append(candidate_summary)
        eligible_candidates = [
            candidate
            for candidate in candidate_summaries
            if candidate["status"] == "eligible"
        ]
        if not eligible_candidates:
            raise ValueError(
                f"Stage {expected_name} has no candidate inheriting {previous_winner!r}"
            )
        expected_coverage = (
            eligible_candidates[0]["stocks"],
            eligible_candidates[0]["seeds"],
        )
        for candidate in eligible_candidates[1:]:
            coverage = (candidate["stocks"], candidate["seeds"])
            if coverage != expected_coverage:
                raise ValueError(
                    f"Stage {expected_name} candidates use different stock/seed coverage"
                )
            if (
                candidate["evaluation_coverage"]
                != eligible_candidates[0]["evaluation_coverage"]
            ):
                raise ValueError(
                    f"Stage {expected_name} candidates use different validation "
                    "target coverage"
                )
        selected = min(eligible_candidates, key=_ranking_key)
        previous_winner = selected["id"]
        stage_summaries.append(
            {
                "name": expected_name,
                "selected_candidate_id": previous_winner,
                "candidates": candidate_summaries,
            }
        )
        previous_stage_ids = {candidate["id"] for candidate in candidate_summaries}
        previous_candidates_by_id = {
            candidate["id"]: candidate for candidate in candidate_summaries
        }

    return {
        "artifact_type": "chapter5_validation_selection",
        "schema_version": 1,
        "selection_id": selection_id,
        "manifest": resolved_manifest.name,
        "metric_split": "validation",
        "primary_metric": "rmse",
        "secondary_metrics": ["direction_accuracy"],
        "aggregation_hierarchy": "seeds_within_stock_then_stocks",
        "ranking_rule": [
            "rmse_ascending",
            "direction_accuracy_descending",
            "candidate_id_ascending",
        ],
        "stages": stage_summaries,
        "selected_candidate_id": previous_winner,
        "complete": len(stages) == len(STAGE_NAMES),
    }


def _selected_candidate(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    final_stage = summary["stages"][-1]
    selected_id = final_stage["selected_candidate_id"]
    for candidate in final_stage["candidates"]:
        if candidate["id"] == selected_id:
            return candidate
    raise ValueError(f"Selected candidate {selected_id!r} is missing from summary")


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def freeze_selected_config(
    manifest_path: Path,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy the final validation winner and switch only its holdout to test."""
    if not summary.get("complete"):
        raise ValueError("Final freezing requires a complete three-stage selection")
    resolved_manifest = Path(manifest_path).resolve()
    selected = _selected_candidate(summary)
    source_path = _resolve_manifest_path(
        resolved_manifest.parent,
        selected["config"],
        "selected candidate config",
    )
    _, source = read_config_file(source_path)
    frozen = copy.deepcopy(source)
    downstream = frozen.get("runner", {}).get("downstream")
    if not isinstance(downstream, dict):
        raise ValueError("Selected config requires nested runner.downstream settings")
    downstream["evaluation_split"] = "test"
    checkpoint_selection = (
        frozen.get("runner", {}).get("checkpoint", {}).get("selection", {})
    )
    if checkpoint_selection.get("mode") != "best":
        raise ValueError("Frozen config requires checkpoint.selection.mode = best")
    frozen.pop("provenance", None)
    experiment_sections = {
        key: frozen[key]
        for key in ("common", "runner", "analysis")
        if key in frozen
    }
    frozen["provenance"] = {
        "artifact_type": "frozen_chapter5_experiment_config",
        "schema_version": 1,
        "selection_id": summary["selection_id"],
        "selected_candidate_id": selected["id"],
        "source_config": selected["config"],
        "source_config_sha256": selected["config_sha256"],
        "source_config_signature": selected["config_signature"],
        "experiment_config_sha256": canonical_sha256(experiment_sections),
        "selection_summary_sha256": canonical_sha256(summary),
        "git_commit": _git_commit(Path(__file__).resolve().parent),
        "metric_split": "validation",
        "aggregation_hierarchy": summary["aggregation_hierarchy"],
        "ranking_rule": summary["ranking_rule"],
        "stage_winners": [
            {
                "stage": stage["name"],
                "candidate_id": stage["selected_candidate_id"],
            }
            for stage in summary["stages"]
        ],
        "final_evaluation_split": "test",
    }
    return frozen


def snapshot_selected_stage_config(
    manifest_path: Path,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy an intermediate validation winner without enabling test evaluation."""
    resolved_manifest = Path(manifest_path).resolve()
    selected = _selected_candidate(summary)
    source_path = _resolve_manifest_path(
        resolved_manifest.parent,
        selected["config"],
        "selected candidate config",
    )
    _, source = read_config_file(source_path)
    snapshot = copy.deepcopy(source)
    downstream = snapshot.get("runner", {}).get("downstream")
    if not isinstance(downstream, dict) or downstream.get("evaluation_split") != "validation":
        raise ValueError("Intermediate selected config must remain validation-only")
    snapshot["provenance"] = {
        "artifact_type": "selected_chapter5_stage_config",
        "schema_version": 1,
        "selection_id": summary["selection_id"],
        "selected_candidate_id": selected["id"],
        "selected_stage": summary["stages"][-1]["name"],
        "source_config": selected["config"],
        "source_config_sha256": selected["config_sha256"],
        "source_config_signature": selected["config_signature"],
        "selection_summary_sha256": canonical_sha256(summary),
        "metric_split": "validation",
    }
    return snapshot


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _validate_output_separation(manifest_path: Path, output_dir: Path) -> None:
    resolved_manifest = Path(manifest_path).resolve()
    manifest = _read_object(resolved_manifest, "selection manifest")
    resolved_output = Path(output_dir).resolve()
    for stage in manifest.get("stages", []):
        for candidate in stage.get("candidates", []):
            validation_root = _resolve_manifest_path(
                resolved_manifest.parent,
                candidate.get("validation_root"),
                "candidate.validation_root",
            )
            if resolved_output == validation_root or resolved_output.is_relative_to(
                validation_root
            ):
                raise ValueError(
                    "Selection output must remain separate from validation results: "
                    f"output={resolved_output}, validation_root={validation_root}"
                )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Select and freeze Chapter 5 experiments from validation only."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    _validate_output_separation(manifest_path, output_dir)
    summary = select_stages(manifest_path)
    _write_json_atomic(output_dir / "selection_summary.json", summary)
    print(f"Validation selection summary: {output_dir / 'selection_summary.json'}")
    if summary["complete"]:
        frozen = freeze_selected_config(manifest_path, summary)
        _write_json_atomic(output_dir / "selected_config.json", frozen)
        print(f"Frozen final test config: {output_dir / 'selected_config.json'}")
    else:
        snapshot = snapshot_selected_stage_config(manifest_path, summary)
        _write_json_atomic(output_dir / "selected_stage_config.json", snapshot)
        print(
            "Selected validation-stage config: "
            f"{output_dir / 'selected_stage_config.json'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
