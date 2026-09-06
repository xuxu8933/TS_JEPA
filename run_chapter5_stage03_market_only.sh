#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
python_bin="${PYTHON_BIN:-python}"
legacy_results=results/01_preprocessing_window_return
legacy_archive=archive/legacy_results/01_preprocessing_window_return_12_origins
stage03_shared12_manifest=results/03_shared_context_12_patches_market_only/experiment_manifest.json
configs=(
  config/experiments/chapter5_candidates/03_shared_context_6_patches_market_only.json
  config/experiments/chapter5_candidates/03_shared_context_12_patches_market_only.json
  config/experiments/chapter5_candidates/03_shared_context_24_patches_market_only.json
  config/experiments/chapter5_candidates/03_local_long_context_6_patches_market_only.json
  config/experiments/chapter5_candidates/03_local_long_context_12_patches_market_only.json
  config/experiments/chapter5_candidates/03_local_long_context_24_patches_market_only.json
)

[[ ${#configs[@]} -eq 6 ]] || exit 1
for config in "${configs[@]}"; do
  [[ -f $config ]] || {
    printf 'Missing config: %s\n' "$config" >&2
    exit 1
  }
done

if [[ ! -f $stage03_shared12_manifest && -d $legacy_results ]]; then
  [[ ! -e $legacy_archive ]] || {
    printf 'Refusing to overwrite existing archive: %s\n' "$legacy_archive" >&2
    exit 1
  }
  "$python_bin" - "$legacy_results" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
metadata_paths = sorted(
    root.glob("random/*/seed_*/preprocessing_config.json")
)
if len(metadata_paths) != 15:
    raise SystemExit(
        f"Refusing to archive {root}: expected 15 metadata files, "
        f"found {len(metadata_paths)}"
    )
for path in metadata_paths:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if (
        metadata.get("evaluation_split") != "validation"
        or metadata.get("evaluation_sample_count") != 12
    ):
        raise SystemExit(
            f"Refusing to archive {root}: unexpected coverage in {path}"
        )
PY
  mkdir -p "$(dirname "$legacy_archive")"
  mv -- "$legacy_results" "$legacy_archive"
  printf 'Archived legacy 12-origin results to %s\n' "$legacy_archive"
fi

for config in "${configs[@]}"; do
  printf '\nRunning %s\n' "$config"
  "$python_bin" -u run_top_nasdaq100_stocks.py --config "$config" "$@"
done
