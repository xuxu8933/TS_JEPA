#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
python_bin="${PYTHON_BIN:-python}"
configs=(
  config/experiments/chapter5_benchmark/05_benchmark_sentiment.json
  config/experiments/chapter5_benchmark/05_benchmark_market_only.json
  config/experiments/chapter5_benchmark/05_benchmark_local_long_sentiment.json
  config/experiments/chapter5_benchmark/05_benchmark_local_long_market_only.json
)

for config in "${configs[@]}"; do
  [[ -f $config ]] || {
    printf 'Missing config: %s\n' "$config" >&2
    exit 1
  }
  printf '\nRunning %s\n' "$config"
  "$python_bin" -u run_top_nasdaq100_stocks.py --config "$config" "$@"
done
