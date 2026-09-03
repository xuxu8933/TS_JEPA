#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
python_bin="${PYTHON_BIN:-python}"
analysis_dirs=(
  analysis_artifacts/04_shared_joint_loss_*
  analysis_artifacts/04_local_long_joint_loss_*
)

[[ ${#analysis_dirs[@]} -eq 10 ]] || {
  printf 'Expected 10 joint-loss analysis directories, found %d\n' "${#analysis_dirs[@]}" >&2
  exit 1
}

for analysis_dir in "${analysis_dirs[@]}"; do
  printf '\nPublishing %s\n' "$analysis_dir"
  "$python_bin" -u publish_thesis_results.py \
    --analysis-dir "$analysis_dir" \
    "$@"
done
