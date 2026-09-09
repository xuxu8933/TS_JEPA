#!/usr/bin/env bash
set -euo pipefail

# Context is 12 downstream patches (60 observations); pretraining stays at 60.
# Pass --dry-run to validate all five candidates without training.
cd "$(dirname "${BASH_SOURCE[0]}")"
python_bin="${PYTHON_BIN:-python}"
shopt -s nullglob
configs=(
  config/experiments/chapter5_candidates/stage04_market_only_context_12/04_local_long_joint_loss_*.json
)

[[ ${#configs[@]} -eq 5 ]] || {
  printf 'Expected 5 market-only Stage 04 local/long context-12 configs, found %d\n' "${#configs[@]}" >&2
  exit 1
}

for config in "${configs[@]}"; do
  printf '\nRunning %s\n' "$config"
  "$python_bin" -u run_top_nasdaq100_stocks.py --config "$config" "$@"
done
