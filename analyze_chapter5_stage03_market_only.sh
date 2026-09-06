#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
python_bin="${PYTHON_BIN:-python}"
configs=(
  config/experiments/chapter5_candidates/03_shared_context_*_patches_market_only.json
  config/experiments/chapter5_candidates/03_local_long_context_*_patches_market_only.json
)

[[ ${#configs[@]} -eq 6 ]] || {
  printf 'Expected 6 market-only Stage 03 configs, found %d\n' "${#configs[@]}" >&2
  exit 1
}

for config in "${configs[@]}"; do
  if [[ $config == *03_shared_context_* ]]; then
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
