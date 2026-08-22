"""Create an immutable GitHub Release ZIP for one complete experiment scope."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from config.file_options import results_dir_from_config


ARCHIVE_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ALREADY_COMPRESSED_SUFFIXES = frozenset((".gif", ".gz", ".jpg", ".jpeg", ".png", ".zip"))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _safe_name(value: str, *, label: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not name:
        raise ValueError(f"Cannot derive a safe {label} from {value!r}")
    return name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _complete_run_keys(results_dir: Path) -> dict[tuple[str, str, int], Path]:
    completed: dict[tuple[str, str, int], Path] = {}
    for path in sorted(results_dir.rglob("run_manifest.json")):
        manifest = _load_json(path)
        if manifest.get("status") != "complete":
            raise RuntimeError(
                f"Refusing to package non-complete run manifest: {path} "
                f"(status={manifest.get('status')!r})"
            )
        try:
            key = (
                str(manifest["strategy"]),
                str(manifest["stock"]).upper(),
                int(manifest["seed"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid complete run manifest: {path}") from exc
        if key in completed:
            raise RuntimeError(
                f"Duplicate complete run manifests for {key}: "
                f"{completed[key]} and {path}"
            )
        comparison_files = manifest.get("comparison_files")
        if not isinstance(comparison_files, list) or not comparison_files:
            raise RuntimeError(
                f"Complete run manifest has no comparison files: {path}"
            )
        missing_files = [
            name
            for name in comparison_files
            if not (path.parent / str(name)).is_file()
        ]
        if missing_files:
            raise RuntimeError(
                f"Complete run manifest references missing files at {path.parent}: "
                + ", ".join(map(str, missing_files))
            )
        completed[key] = path
    return completed


def validate_archive_coverage(
    results_dir: Path,
    experiment_manifest: dict[str, Any],
) -> None:
    stocks = experiment_manifest.get("run_stocks") or experiment_manifest.get("stocks")
    seeds = experiment_manifest.get("run_seeds") or experiment_manifest.get("seeds")
    strategies = experiment_manifest.get("mask_strategies")
    if not isinstance(stocks, list) or not stocks:
        raise ValueError("Experiment manifest has no run stock coverage")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("Experiment manifest has no run seed coverage")
    if not isinstance(strategies, list) or not strategies:
        raise ValueError("Experiment manifest has no mask strategies")

    expected = {
        (str(strategy), str(stock).upper(), int(seed))
        for strategy in strategies
        for stock in stocks
        for seed in seeds
    }
    completed = _complete_run_keys(results_dir)
    missing = sorted(expected - set(completed))
    if missing:
        preview = ", ".join(
            f"{strategy}/{stock}/seed_{seed}"
            for strategy, stock, seed in missing[:10]
        )
        if len(missing) > 10:
            preview += f", ... and {len(missing) - 10} more"
        raise RuntimeError(
            f"Refusing to package incomplete experiment coverage: {preview}"
        )


def _zip_info(name: str, *, compressed: bool) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ARCHIVE_TIMESTAMP)
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = (
        zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    )
    return info


def _write_bytes(
    archive: zipfile.ZipFile,
    name: str,
    value: bytes,
    *,
    compressed: bool = True,
) -> None:
    archive.writestr(_zip_info(name, compressed=compressed), value)


def package_experiment_results(
    config_path: Path | str,
    archive_root: Path | str = "release_assets",
) -> Path:
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config does not exist: {config_path}")
    results_dir = results_dir_from_config(config_path).resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Experiment results do not exist: {results_dir}")
    manifest_path = results_dir / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Experiment manifest does not exist: {manifest_path}")
    manifest = _load_json(manifest_path)
    signature = manifest.get("config_signature")
    if not signature:
        raise ValueError("Experiment manifest has no config_signature")
    validate_archive_coverage(results_dir, manifest)

    config_name = _safe_name(config_path.stem, label="config name")
    signature_name = _safe_name(str(signature), label="config signature")[:12]
    archive_root = Path(archive_root).resolve()
    if archive_root == results_dir or archive_root.is_relative_to(results_dir):
        raise ValueError("Archive output must be outside the experiment result tree")
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_path = archive_root / f"{config_name}-{signature_name}.zip"

    source_files = [
        path
        for path in sorted(results_dir.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]
    if not source_files:
        raise RuntimeError(f"No result files found below {results_dir}")

    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{archive_path.name}.",
        suffix=".tmp",
        dir=archive_root,
    )
    os.close(temporary_fd)
    temporary_path = Path(temporary_name)
    try:
        records: list[dict[str, Any]] = []
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for source in source_files:
                relative = source.relative_to(results_dir).as_posix()
                archive_name = f"{config_name}/{relative}"
                before = source.stat()
                digest = hashlib.sha256()
                info = _zip_info(
                    archive_name,
                    compressed=(
                        source.suffix.lower() not in ALREADY_COMPRESSED_SUFFIXES
                    ),
                )
                with source.open("rb") as input_file, archive.open(info, "w") as output:
                    for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                        digest.update(chunk)
                        output.write(chunk)
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    raise RuntimeError(
                        f"Result changed while it was being packaged: {source}"
                    )
                records.append(
                    {
                        "path": relative,
                        "bytes": before.st_size,
                        "sha256": digest.hexdigest(),
                    }
                )

            manifest_buffer = io.StringIO(newline="")
            writer = csv.DictWriter(
                manifest_buffer,
                fieldnames=("path", "bytes", "sha256"),
            )
            writer.writeheader()
            writer.writerows(records)
            _write_bytes(
                archive,
                f"{config_name}/ARCHIVE_MANIFEST.csv",
                manifest_buffer.getvalue().encode("utf-8"),
            )
            archive_readme = (
                f"TS-JEPA complete result archive\n"
                f"config={config_path.name}\n"
                f"config_signature={signature}\n"
                f"files={len(records)}\n"
                "Verify extracted files against ARCHIVE_MANIFEST.csv.\n"
            )
            _write_bytes(
                archive,
                f"{config_name}/ARCHIVE_README.txt",
                archive_readme.encode("utf-8"),
            )

        if archive_path.exists():
            if _sha256(archive_path) != _sha256(temporary_path):
                raise FileExistsError(
                    f"A different release archive already exists: {archive_path}"
                )
            temporary_path.unlink()
        else:
            temporary_path.replace(archive_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    checksum_path.write_text(
        f"{_sha256(archive_path)}  {archive_path.name}\n",
        encoding="utf-8",
    )
    return archive_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package one complete filename-derived experiment result tree as "
            "a deterministic GitHub Release ZIP."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--archive-root", default="release_assets")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive_path = package_experiment_results(args.config, args.archive_root)
    print(f"Release archive: {archive_path}")
    print(f"SHA-256: {archive_path}.sha256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
