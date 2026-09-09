#!/usr/bin/env bash
set -euo pipefail

# 12 patches = 60 observations; validation-ranked JEPA:MAE weights 1.5:0.5.
# Stage 05 evaluates the previously observed test period. Use --dry-run to inspect.
cd "$(dirname "${BASH_SOURCE[0]}")"
python_bin="${PYTHON_BIN:-python}"
exec "$python_bin" -u run_top_nasdaq100_stocks.py \
  --config config/experiments/chapter5_benchmark/05_benchmark_local_long_market_only_context_12.json \
  "$@"
