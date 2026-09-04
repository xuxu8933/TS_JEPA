#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
python_bin="${PYTHON_BIN:-python}"

for result_dir in results/0[123]_*; do
  [[ -d $result_dir ]] || continue
  config="config/experiments/chapter5_candidates/$(basename "$result_dir").json"
  [[ -f $config ]] || {
    printf 'Missing config for %s: %s\n' "$result_dir" "$config" >&2
    exit 1
  }

  if [[ -d $result_dir/random ]]; then
    reference_strategy=random
  elif [[ -d $result_dir/local_long ]]; then
    reference_strategy=local_long
  else
    printf 'No supported strategy directory under %s\n' "$result_dir" >&2
    exit 1
  fi

  printf '\nAnalyzing %s\n' "$result_dir"
  "$python_bin" -u analyze_thesis_results.py \
    --config "$config" \
    --reference-strategy "$reference_strategy" \
    "$@"
done
