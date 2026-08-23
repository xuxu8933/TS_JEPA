"""Publish a validated thesis-analysis snapshot into a Git-tracked directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parent
PUBLISHABLE_SUFFIXES = frozenset(
    (".csv", ".json", ".md", ".pdf", ".png", ".tex")
)
PORTABLE_TEXT_SUFFIXES = frozenset((".csv", ".json", ".md", ".tex"))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() not in PORTABLE_TEXT_SUFFIXES:
        return data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    project_root = str(PROJECT_ROOT)
    text = text.replace(project_root + "/", "")
    text = text.replace(project_root + "\\", "")
    return text.encode("utf-8")


def _portable_sha256(path: Path) -> str:
    return hashlib.sha256(_portable_bytes(path)).hexdigest()


def _safe_name(value: str, *, label: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not name:
        raise ValueError(f"Cannot derive a safe {label} from {value!r}")
    return name


def _source_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            continue
        if path.is_file():
            yield path


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in _source_files(root)
    }


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_portable_bytes(source))


def _analysis_identity(
    analysis_dir: Path,
    metadata: dict[str, Any],
) -> tuple[str, str, Path | None, Path | None]:
    scope = metadata.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("analysis_metadata.json has no valid scope object")

    config_path_value = scope.get("config_path")
    if not config_path_value:
        raise ValueError("analysis metadata does not record scope.config_path")
    config_path = Path(str(config_path_value))
    config_name = _safe_name(config_path.stem, label="config name")

    results_manifest = None
    experiment_signature = "no-manifest"
    results_dir_value = metadata.get("results_dir")
    if results_dir_value:
        candidate = Path(str(results_dir_value)) / "experiment_manifest.json"
        if candidate.is_file():
            results_manifest = candidate
            manifest = _load_json(candidate)
            recorded_signature = manifest.get("config_signature")
            if recorded_signature:
                experiment_signature = _safe_name(
                    str(recorded_signature),
                    label="experiment signature",
                )

    digest = hashlib.sha256()
    digest.update(experiment_signature.encode("utf-8"))
    for path in _source_files(analysis_dir):
        relative = path.relative_to(analysis_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(_portable_sha256(path).encode("ascii"))
    if config_path.is_file():
        digest.update(_portable_sha256(config_path).encode("ascii"))
    if results_manifest is not None:
        digest.update(_portable_sha256(results_manifest).encode("ascii"))

    snapshot_name = (
        f"{experiment_signature[:12]}-{digest.hexdigest()[:12]}"
    )
    return config_name, snapshot_name, (
        config_path if config_path.is_file() else None
    ), results_manifest


def _append_publication_note(
    readme_path: Path,
    *,
    snapshot_name: str,
    omitted_count: int,
    validity_error_count: int,
) -> None:
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    validity_note = ""
    if validity_error_count:
        validity_note = (
            "\n> **INCOMPLETE TEST SNAPSHOT:** This analysis contains "
            f"{validity_error_count} validity error(s) and is not a validated "
            "thesis result.\n"
        )
    note = (
        "\n## Git publication\n\n"
        f"{validity_note}"
        f"- Immutable snapshot: `{snapshot_name}`\n"
        "- Full raw experiment outputs are intentionally excluded from Git.\n"
        "- `SHA256SUMS` verifies every published file.\n"
        f"- Files omitted by publication policy: {omitted_count}. See "
        "`publication_manifest.csv`.\n"
        "- Large/raw artifacts should be attached to the matching GitHub Release.\n"
    )
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(existing.rstrip() + "\n" + note, encoding="utf-8")


def publish_thesis_results(
    analysis_dir: Path | str = "analysis_artifacts",
    publish_root: Path | str = "thesis_results",
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    allow_incomplete: bool = False,
) -> Path:
    analysis_dir = Path(analysis_dir).resolve()
    publish_root = Path(publish_root).resolve()
    if max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be positive")
    if not analysis_dir.is_dir():
        raise FileNotFoundError(f"Analysis directory does not exist: {analysis_dir}")
    if (
        publish_root == analysis_dir
        or publish_root.is_relative_to(analysis_dir)
        or analysis_dir.is_relative_to(publish_root)
    ):
        raise ValueError("Analysis and publication directories must not overlap")

    metadata_path = analysis_dir / "analysis_metadata.json"
    artifact_manifest_path = analysis_dir / "artifact_manifest.csv"
    if not metadata_path.is_file() or not artifact_manifest_path.is_file():
        raise ValueError(
            "Analysis publication requires analysis_metadata.json and "
            "artifact_manifest.csv"
        )
    metadata = _load_json(metadata_path)
    try:
        error_issues = int(metadata["error_issues"])
        canonical_rows = int(metadata["canonical_rows"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Analysis metadata must record integer error_issues and canonical_rows"
        ) from exc
    if error_issues and not allow_incomplete:
        raise RuntimeError(
            f"Refusing to publish analysis with {error_issues} validity error(s)"
        )
    if canonical_rows <= 0:
        raise RuntimeError("Refusing to publish analysis with no canonical rows")

    config_name, snapshot_name, config_path, results_manifest = _analysis_identity(
        analysis_dir,
        metadata,
    )
    if error_issues:
        snapshot_name = f"incomplete-{snapshot_name}"
    destination = publish_root / config_name / snapshot_name
    destination.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix=f".{snapshot_name}.",
        dir=destination.parent,
    ) as temporary_name:
        temporary = Path(temporary_name)
        for source in _source_files(analysis_dir):
            relative = source.relative_to(analysis_dir)
            size = source.stat().st_size
            if source.suffix.lower() not in PUBLISHABLE_SUFFIXES:
                status = "omitted"
                reason = "unsupported-file-type"
            elif size > max_file_bytes:
                status = "omitted"
                reason = f"larger-than-{max_file_bytes}-bytes"
            else:
                status = "published"
                reason = ""
                _copy_file(source, temporary / relative)
            recorded_bytes = (
                len(_portable_bytes(source)) if status == "published" else size
            )
            recorded_sha256 = (
                _portable_sha256(source)
                if status == "published"
                else _sha256(source)
            )
            records.append(
                {
                    "source": relative.as_posix(),
                    "published_path": (
                        relative.as_posix() if status == "published" else ""
                    ),
                    "status": status,
                    "reason": reason,
                    "bytes": recorded_bytes,
                    "sha256": recorded_sha256,
                }
            )

        provenance_sources = (
            (
                config_path,
                Path("provenance")
                / f"experiment_config{config_path.suffix if config_path else '.json'}",
                "configured experiment file",
            ),
            (
                results_manifest,
                Path("provenance") / "experiment_manifest.json",
                "runtime experiment manifest",
            ),
        )
        for source, relative, source_label in provenance_sources:
            if source is None:
                continue
            _copy_file(source, temporary / relative)
            records.append(
                {
                    "source": source_label,
                    "published_path": relative.as_posix(),
                    "status": "published",
                    "reason": "provenance",
                    "bytes": len(_portable_bytes(source)),
                    "sha256": _portable_sha256(source),
                }
            )

        omitted_count = sum(record["status"] == "omitted" for record in records)
        _append_publication_note(
            temporary / "README.md",
            snapshot_name=snapshot_name,
            omitted_count=omitted_count,
            validity_error_count=error_issues,
        )

        publication_manifest = temporary / "publication_manifest.csv"
        with publication_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "source",
                    "published_path",
                    "status",
                    "reason",
                    "bytes",
                    "sha256",
                ),
            )
            writer.writeheader()
            writer.writerows(records)

        checksum_paths = [
            path
            for path in _source_files(temporary)
            if path.name != "SHA256SUMS"
        ]
        checksum_text = "".join(
            f"{_sha256(path)}  {path.relative_to(temporary).as_posix()}\n"
            for path in checksum_paths
        )
        (temporary / "SHA256SUMS").write_text(checksum_text, encoding="utf-8")

        if destination.exists():
            if _tree_hashes(destination) != _tree_hashes(temporary):
                raise FileExistsError(
                    "A different published snapshot already exists at "
                    f"{destination}; published thesis results are append-only"
                )
            return destination
        temporary.rename(destination)

    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a complete, validity-clean thesis analysis into an immutable "
            "Git-tracked snapshot."
        )
    )
    parser.add_argument("--analysis-dir", default="analysis_artifacts")
    parser.add_argument("--publish-root", default="thesis_results")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Publish despite recorded validity errors as a clearly marked "
            "incomplete test snapshot."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = publish_thesis_results(
        args.analysis_dir,
        args.publish_root,
        allow_incomplete=args.allow_incomplete,
    )
    if args.allow_incomplete:
        print(
            "WARNING: --allow-incomplete is enabled; this snapshot is for "
            "testing and is not a validated thesis result."
        )
    print(f"Published thesis snapshot: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
