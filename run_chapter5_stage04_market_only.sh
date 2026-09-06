#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
python_bin="${PYTHON_BIN:-python}"
configs=(
  config/experiments/chapter5_candidates/stage04_market_only/04_shared_joint_loss_*.json
  config/experiments/chapter5_candidates/stage04_market_only/04_local_long_joint_loss_*.json
)

[[ ${#configs[@]} -eq 10 ]] || {
  printf 'Expected 10 market-only Stage 04 configs, found %d\n' "${#configs[@]}" >&2
  exit 1
}

for config in "${configs[@]}"; do
  printf '\nRunning %s\n' "$config"
  "$python_bin" -u run_top_nasdaq100_stocks.py --config "$config" "$@"
done
