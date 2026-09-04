#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
python_bin="${PYTHON_BIN:-python}"
analysis_dirs=(analysis_artifacts/0[123]_*)

[[ ${#analysis_dirs[@]} -eq 8 ]] || {
  printf 'Expected 8 stage 1-3 analysis directories, found %d\n' "${#analysis_dirs[@]}" >&2
  exit 1
}

for analysis_dir in "${analysis_dirs[@]}"; do
  printf '\nPublishing %s\n' "$analysis_dir"
  "$python_bin" -u publish_thesis_results.py \
    --analysis-dir "$analysis_dir" \
    "$@"
done
