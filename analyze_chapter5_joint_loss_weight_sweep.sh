#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
python_bin="${PYTHON_BIN:-python}"
configs=(
  config/experiments/chapter5_candidates/04_shared_joint_loss_*.json
  config/experiments/chapter5_candidates/04_local_long_joint_loss_*.json
)

[[ ${#configs[@]} -eq 10 ]] || {
  printf 'Expected 10 joint-loss configs, found %d\n' "${#configs[@]}" >&2
  exit 1
}

for config in "${configs[@]}"; do
  if [[ $config == *04_shared_joint_loss_* ]]; then
    reference_strategy=random
  else
    reference_strategy=local_long
  fi
  printf '\nAnalyzing %s\n' "$config"
  "$python_bin" -u analyze_thesis_results.py \
    --config "$config" \
    --reference-strategy "$reference_strategy" \
    "$@"
done
