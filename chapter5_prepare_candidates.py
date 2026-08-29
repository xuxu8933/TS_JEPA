"""Materialize later Chapter 5 candidates from a validation-selected base."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from chapter5_selection import canonical_sha256
from config.file_options import read_config_file
from run_top_nasdaq100_stocks import (
    parse_args as parse_runner_args,
    resolve_seeds,
    resolve_stocks,
)


EXPECTED_STOCKS = ["NVDA", "AAPL", "AVGO", "TSLA", "WMT"]
EXPECTED_SEEDS = [42, 44, 46]
SUPPORTED_STAGES = ("sentiment", "architecture_context")


def _parent_config_hash(
    base: dict[str, Any],
    parent_candidate_id: str,
) -> str:
    provenance = base.get("provenance")
    if not isinstance(provenance, dict) or provenance.get(
        "artifact_type"
    ) != "selected_chapter5_stage_config":
        return canonical_sha256(base)

    if provenance.get("metric_split") != "validation":
        raise ValueError("Selected stage snapshot must be validation-only")
    if provenance.get("selected_candidate_id") != parent_candidate_id:
        raise ValueError(
            "parent_candidate_id must match the selected stage snapshot"
        )
    source_hash = provenance.get("source_config_sha256")
    if not isinstance(source_hash, str) or re.fullmatch(
        r"[0-9a-f]{64}", source_hash
    ) is None:
        raise ValueError("Selected stage snapshot has an invalid source config hash")
    return source_hash


def _load_validated_base(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    _, raw = read_config_file(resolved)
    if not isinstance(raw, dict):
        raise ValueError(f"Candidate base must be a JSON object: {resolved}")
    args = parse_runner_args(["--config", str(resolved)])
    if args.evaluation_split != "validation":
        raise ValueError("Candidate base must be validation-only")
    if not args.use_best_checkpoint:
        raise ValueError("Candidate base must use checkpoint.selection.mode=best")
    if resolve_stocks(args) != EXPECTED_STOCKS:
        raise ValueError(f"Candidate base stocks must be exactly {EXPECTED_STOCKS}")
    if resolve_seeds(args) != EXPECTED_SEEDS:
        raise ValueError(f"Candidate base seeds must be exactly {EXPECTED_SEEDS}")
    expected_settings = {
        "max_parallel_jobs": 2,
        "lambda_jepa": 1.0,
        "lambda_mae": 0.5,
        "series_split_size": 60,
        "patch_size": 5,
        "forecast_horizon": 5,
        "pretrain_num_epochs": 2001,
        "eval_num_epochs": 501,
    }
    for name, expected in expected_settings.items():
        actual = getattr(args, name)
        if actual != expected:
            raise ValueError(
                f"Candidate base {name} must be {expected!r}, got {actual!r}"
            )
    return raw


def _provenance(
    *,
    stage: str,
    candidate_id: str,
    filename: str,
    parent_candidate_id: str,
    parent_hash: str,
    delta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_type": "chapter5_candidate_config",
        "schema_version": 1,
        "stage": stage,
        "candidate_id": candidate_id,
        "candidate_filename": filename,
        "parent_candidate_id": parent_candidate_id,
        "parent_config_sha256": parent_hash,
        "delta": delta,
    }


def _sentiment_candidates(
    base: dict[str, Any],
    parent_candidate_id: str,
    parent_hash: str,
) -> list[tuple[str, dict[str, Any]]]:
    candidates = []
    for label, enabled in (("excluded", False), ("included", True)):
        candidate = copy.deepcopy(base)
        candidate["runner"]["preprocessing"]["custom"]["features"][
            "sentiment"
        ]["enabled"] = enabled
        filename = f"02_sentiment_{label}.json"
        candidate["provenance"] = _provenance(
            stage="sentiment",
            candidate_id=f"sentiment_{label}",
            filename=filename,
            parent_candidate_id=parent_candidate_id,
            parent_hash=parent_hash,
            delta={"sentiment_enabled": enabled},
        )
        candidates.append((filename, candidate))
    return candidates


def _architecture_context_candidates(
    base: dict[str, Any],
    parent_candidate_id: str,
    parent_hash: str,
) -> list[tuple[str, dict[str, Any]]]:
    candidates = []
    strategies = (
        ("shared", "random", {"random": {"enabled": True}}),
        (
            "local_long",
            "local_long",
            {
                "local_long": {
                    "enabled": True,
                    "mae_window_patches": 1,
                    "jepa_gap_patches": 4,
                    "jepa_target_patches": 4,
                }
            },
        ),
    )
    for architecture, strategy, strategy_config in strategies:
        for context_size in (6, 12, 24):
            candidate = copy.deepcopy(base)
            candidate["runner"]["masking"]["strategies"] = copy.deepcopy(
                strategy_config
            )
            candidate["runner"]["downstream"]["context_size"] = context_size
            filename = f"03_{architecture}_context_{context_size}_patches.json"
            candidate_id = f"{architecture}_context_{context_size}"
            candidate["provenance"] = _provenance(
                stage="architecture_context",
                candidate_id=candidate_id,
                filename=filename,
                parent_candidate_id=parent_candidate_id,
                parent_hash=parent_hash,
                delta={
                    "strategy": strategy,
                    "context_size_patches": context_size,
                },
            )
            candidates.append((filename, candidate))
    return candidates


def _encoded_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def materialize_candidates(
    stage: str,
    base_config_path: Path,
    parent_candidate_id: str,
    output_dir: Path,
) -> list[Path]:
    """Create one later-stage candidate set without assuming earlier winners."""
    if stage not in SUPPORTED_STAGES:
        raise ValueError(f"Unsupported candidate stage {stage!r}")
    if not isinstance(parent_candidate_id, str) or not parent_candidate_id.strip():
        raise ValueError("parent_candidate_id must be a non-empty string")

    base = _load_validated_base(Path(base_config_path))
    parent_hash = _parent_config_hash(base, parent_candidate_id)
    if stage == "sentiment":
        candidates = _sentiment_candidates(base, parent_candidate_id, parent_hash)
    else:
        candidates = _architecture_context_candidates(
            base,
            parent_candidate_id,
            parent_hash,
        )

    resolved_output = Path(output_dir)
    encoded = [
        (resolved_output / filename, _encoded_json(candidate))
        for filename, candidate in candidates
    ]
    for path, content in encoded:
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(
                f"Refusing to overwrite different candidate config: {path}"
            )

    resolved_output.mkdir(parents=True, exist_ok=True)
    for path, content in encoded:
        if path.exists():
            continue
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
    return [path for path, _ in encoded]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Materialize Chapter 5 candidates from a selected stage config."
    )
    parser.add_argument("--stage", required=True, choices=SUPPORTED_STAGES)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--parent-candidate-id", required=True)
    parser.add_argument(
        "--output-dir",
        default="config/experiments/chapter5_candidates",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = materialize_candidates(
        args.stage,
        Path(args.base_config),
        args.parent_candidate_id,
        Path(args.output_dir),
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
