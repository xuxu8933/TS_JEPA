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
  if [[ $config == *local_long* ]]; then
    reference_strategy=local_long
  else
    reference_strategy=random
  fi
  analysis_dir="analysis_artifacts/$(basename "$config" .json)"

  printf '\nAnalyzing %s\n' "$config"
  "$python_bin" -u analyze_thesis_results.py \
    --config "$config" \
    --reference-strategy "$reference_strategy"

  printf '\nPublishing %s\n' "$analysis_dir"
  "$python_bin" -u publish_thesis_results.py \
    --analysis-dir "$analysis_dir"
done
