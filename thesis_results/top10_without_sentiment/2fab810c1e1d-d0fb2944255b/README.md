# TS-JEPA thesis result manifest

This directory is generated exclusively from saved experiment artifacts; no model training is performed.

## Analysis scope and coverage

- Config: `config/experiments/top10_without_sentiment.json`
- Result root: `results/top10_without_sentiment`
- Expected equities: NVDA, AAPL, MSFT, AMZN, GOOGL, AVGO, META, TSLA, COST, WMT
- Expected seeds: 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
- Strategies: random, local_long
- Canonical baseline/GRU strategy: `random`
- Canonical rows: 600
- Audit issues: 0 errors and 200 warnings

| method | available_runs | expected_runs | coverage_pct |
| --- | --- | --- | --- |
| Shared-target JEPA--MAE | 100 | 100 | 100 |
| Local-MAE/Long-JEPA | 100 | 100 | 100 |
| GRU | 100 | 100 | 100 |
| Naive-last | 100 | 100 | 100 |
| Drift | 100 | 100 | 100 |
| Mean-context | 100 | 100 | 100 |

`missing_runs.csv` is authoritative for missing stocks, seeds, methods, duplicate reruns, conflicting configurations, non-finite values, test-target mismatches, and deterministic-baseline inconsistencies. Incomplete coverage is never silently imputed.

## Source data and exclusions

- `results/top10_without_sentiment/local_long/AAPL/seed_42/last_model_comparison_20260823_132825.csv`
- `results/top10_without_sentiment/local_long/AAPL/seed_43/last_model_comparison_20260823_133538.csv`
- `results/top10_without_sentiment/local_long/AAPL/seed_44/last_model_comparison_20260823_134247.csv`
- `results/top10_without_sentiment/local_long/AAPL/seed_45/last_model_comparison_20260823_134958.csv`
- `results/top10_without_sentiment/local_long/AAPL/seed_46/last_model_comparison_20260823_135710.csv`
- `results/top10_without_sentiment/local_long/AAPL/seed_47/last_model_comparison_20260823_140420.csv`
- `results/top10_without_sentiment/local_long/AAPL/seed_48/last_model_comparison_20260823_141132.csv`
- `results/top10_without_sentiment/local_long/AAPL/seed_49/last_model_comparison_20260823_141849.csv`
- `results/top10_without_sentiment/local_long/AAPL/seed_50/last_model_comparison_20260823_142600.csv`
- `results/top10_without_sentiment/local_long/AAPL/seed_51/last_model_comparison_20260823_143306.csv`
- `results/top10_without_sentiment/local_long/AMZN/seed_42/last_model_comparison_20260823_155212.csv`
- `results/top10_without_sentiment/local_long/AMZN/seed_43/last_model_comparison_20260823_155926.csv`
- `results/top10_without_sentiment/local_long/AMZN/seed_44/last_model_comparison_20260823_160640.csv`
- `results/top10_without_sentiment/local_long/AMZN/seed_45/last_model_comparison_20260823_161351.csv`
- `results/top10_without_sentiment/local_long/AMZN/seed_46/last_model_comparison_20260823_162112.csv`
- `results/top10_without_sentiment/local_long/AMZN/seed_47/last_model_comparison_20260823_162832.csv`
- `results/top10_without_sentiment/local_long/AMZN/seed_48/last_model_comparison_20260823_163553.csv`
- `results/top10_without_sentiment/local_long/AMZN/seed_49/last_model_comparison_20260823_164312.csv`
- `results/top10_without_sentiment/local_long/AMZN/seed_50/last_model_comparison_20260823_165028.csv`
- `results/top10_without_sentiment/local_long/AMZN/seed_51/last_model_comparison_20260823_165742.csv`
- `results/top10_without_sentiment/local_long/AVGO/seed_42/last_model_comparison_20260823_181755.csv`
- `results/top10_without_sentiment/local_long/AVGO/seed_43/last_model_comparison_20260823_182514.csv`
- `results/top10_without_sentiment/local_long/AVGO/seed_44/last_model_comparison_20260823_183231.csv`
- `results/top10_without_sentiment/local_long/AVGO/seed_45/last_model_comparison_20260823_183949.csv`
- `results/top10_without_sentiment/local_long/AVGO/seed_46/last_model_comparison_20260823_184709.csv`
- `results/top10_without_sentiment/local_long/AVGO/seed_47/last_model_comparison_20260823_185429.csv`
- `results/top10_without_sentiment/local_long/AVGO/seed_48/last_model_comparison_20260823_190150.csv`
- `results/top10_without_sentiment/local_long/AVGO/seed_49/last_model_comparison_20260823_190910.csv`
- `results/top10_without_sentiment/local_long/AVGO/seed_50/last_model_comparison_20260823_191627.csv`
- `results/top10_without_sentiment/local_long/AVGO/seed_51/last_model_comparison_20260823_192339.csv`
- `results/top10_without_sentiment/local_long/COST/seed_42/last_model_comparison_20260823_215551.csv`
- `results/top10_without_sentiment/local_long/COST/seed_43/last_model_comparison_20260823_220311.csv`
- `results/top10_without_sentiment/local_long/COST/seed_44/last_model_comparison_20260823_221024.csv`
- `results/top10_without_sentiment/local_long/COST/seed_45/last_model_comparison_20260823_221739.csv`
- `results/top10_without_sentiment/local_long/COST/seed_46/last_model_comparison_20260823_222451.csv`
- `results/top10_without_sentiment/local_long/COST/seed_47/last_model_comparison_20260823_223204.csv`
- `results/top10_without_sentiment/local_long/COST/seed_48/last_model_comparison_20260823_223919.csv`
- `results/top10_without_sentiment/local_long/COST/seed_49/last_model_comparison_20260823_224630.csv`
- `results/top10_without_sentiment/local_long/COST/seed_50/last_model_comparison_20260823_225342.csv`
- `results/top10_without_sentiment/local_long/COST/seed_51/last_model_comparison_20260823_230057.csv`
- `results/top10_without_sentiment/local_long/GOOGL/seed_42/last_model_comparison_20260823_170502.csv`
- `results/top10_without_sentiment/local_long/GOOGL/seed_43/last_model_comparison_20260823_171218.csv`
- `results/top10_without_sentiment/local_long/GOOGL/seed_44/last_model_comparison_20260823_171937.csv`
- `results/top10_without_sentiment/local_long/GOOGL/seed_45/last_model_comparison_20260823_172654.csv`
- `results/top10_without_sentiment/local_long/GOOGL/seed_46/last_model_comparison_20260823_173409.csv`
- `results/top10_without_sentiment/local_long/GOOGL/seed_47/last_model_comparison_20260823_174129.csv`
- `results/top10_without_sentiment/local_long/GOOGL/seed_48/last_model_comparison_20260823_174844.csv`
- `results/top10_without_sentiment/local_long/GOOGL/seed_49/last_model_comparison_20260823_175558.csv`
- `results/top10_without_sentiment/local_long/GOOGL/seed_50/last_model_comparison_20260823_180319.csv`
- `results/top10_without_sentiment/local_long/GOOGL/seed_51/last_model_comparison_20260823_181035.csv`
- `results/top10_without_sentiment/local_long/META/seed_42/last_model_comparison_20260823_193056.csv`
- `results/top10_without_sentiment/local_long/META/seed_43/last_model_comparison_20260823_193814.csv`
- `results/top10_without_sentiment/local_long/META/seed_44/last_model_comparison_20260823_194533.csv`
- `results/top10_without_sentiment/local_long/META/seed_45/last_model_comparison_20260823_195257.csv`
- `results/top10_without_sentiment/local_long/META/seed_46/last_model_comparison_20260823_200014.csv`
- `results/top10_without_sentiment/local_long/META/seed_47/last_model_comparison_20260823_200724.csv`
- `results/top10_without_sentiment/local_long/META/seed_48/last_model_comparison_20260823_201438.csv`
- `results/top10_without_sentiment/local_long/META/seed_49/last_model_comparison_20260823_202148.csv`
- `results/top10_without_sentiment/local_long/META/seed_50/last_model_comparison_20260823_202901.csv`
- `results/top10_without_sentiment/local_long/META/seed_51/last_model_comparison_20260823_203620.csv`
- `results/top10_without_sentiment/local_long/MSFT/seed_42/last_model_comparison_20260823_144017.csv`
- `results/top10_without_sentiment/local_long/MSFT/seed_43/last_model_comparison_20260823_144724.csv`
- `results/top10_without_sentiment/local_long/MSFT/seed_44/last_model_comparison_20260823_145435.csv`
- `results/top10_without_sentiment/local_long/MSFT/seed_45/last_model_comparison_20260823_150149.csv`
- `results/top10_without_sentiment/local_long/MSFT/seed_46/last_model_comparison_20260823_150856.csv`
- `results/top10_without_sentiment/local_long/MSFT/seed_47/last_model_comparison_20260823_151610.csv`
- `results/top10_without_sentiment/local_long/MSFT/seed_48/last_model_comparison_20260823_152325.csv`
- `results/top10_without_sentiment/local_long/MSFT/seed_49/last_model_comparison_20260823_153034.csv`
- `results/top10_without_sentiment/local_long/MSFT/seed_50/last_model_comparison_20260823_153746.csv`
- `results/top10_without_sentiment/local_long/MSFT/seed_51/last_model_comparison_20260823_154456.csv`
- `results/top10_without_sentiment/local_long/NVDA/seed_42/last_model_comparison_20260823_121706.csv`
- `results/top10_without_sentiment/local_long/NVDA/seed_43/last_model_comparison_20260823_122416.csv`
- `results/top10_without_sentiment/local_long/NVDA/seed_44/last_model_comparison_20260823_123128.csv`
- `results/top10_without_sentiment/local_long/NVDA/seed_45/last_model_comparison_20260823_123837.csv`
- `results/top10_without_sentiment/local_long/NVDA/seed_46/last_model_comparison_20260823_124541.csv`
- `results/top10_without_sentiment/local_long/NVDA/seed_47/last_model_comparison_20260823_125251.csv`
- `results/top10_without_sentiment/local_long/NVDA/seed_48/last_model_comparison_20260823_130000.csv`
- `results/top10_without_sentiment/local_long/NVDA/seed_49/last_model_comparison_20260823_130704.csv`
- `results/top10_without_sentiment/local_long/NVDA/seed_50/last_model_comparison_20260823_131411.csv`
- `results/top10_without_sentiment/local_long/NVDA/seed_51/last_model_comparison_20260823_132115.csv`
- `results/top10_without_sentiment/local_long/TSLA/seed_42/last_model_comparison_20260823_204329.csv`
- `results/top10_without_sentiment/local_long/TSLA/seed_43/last_model_comparison_20260823_205040.csv`
- `results/top10_without_sentiment/local_long/TSLA/seed_44/last_model_comparison_20260823_205756.csv`
- `results/top10_without_sentiment/local_long/TSLA/seed_45/last_model_comparison_20260823_210510.csv`
- `results/top10_without_sentiment/local_long/TSLA/seed_46/last_model_comparison_20260823_211222.csv`
- `results/top10_without_sentiment/local_long/TSLA/seed_47/last_model_comparison_20260823_211932.csv`
- `results/top10_without_sentiment/local_long/TSLA/seed_48/last_model_comparison_20260823_212647.csv`
- `results/top10_without_sentiment/local_long/TSLA/seed_49/last_model_comparison_20260823_213405.csv`
- `results/top10_without_sentiment/local_long/TSLA/seed_50/last_model_comparison_20260823_214120.csv`
- `results/top10_without_sentiment/local_long/TSLA/seed_51/last_model_comparison_20260823_214833.csv`
- `results/top10_without_sentiment/local_long/WMT/seed_42/last_model_comparison_20260823_230813.csv`
- `results/top10_without_sentiment/local_long/WMT/seed_43/last_model_comparison_20260823_231530.csv`
- `results/top10_without_sentiment/local_long/WMT/seed_44/last_model_comparison_20260823_232245.csv`
- `results/top10_without_sentiment/local_long/WMT/seed_45/last_model_comparison_20260823_232959.csv`
- `results/top10_without_sentiment/local_long/WMT/seed_46/last_model_comparison_20260823_233707.csv`
- `results/top10_without_sentiment/local_long/WMT/seed_47/last_model_comparison_20260823_234421.csv`
- `results/top10_without_sentiment/local_long/WMT/seed_48/last_model_comparison_20260823_235136.csv`
- `results/top10_without_sentiment/local_long/WMT/seed_49/last_model_comparison_20260823_235848.csv`
- `results/top10_without_sentiment/local_long/WMT/seed_50/last_model_comparison_20260824_000601.csv`
- `results/top10_without_sentiment/local_long/WMT/seed_51/last_model_comparison_20260824_001313.csv`
- `results/top10_without_sentiment/random/AAPL/seed_42/last_model_comparison_20260823_132445.csv`
- `results/top10_without_sentiment/random/AAPL/seed_43/last_model_comparison_20260823_133157.csv`
- `results/top10_without_sentiment/random/AAPL/seed_44/last_model_comparison_20260823_133909.csv`
- `results/top10_without_sentiment/random/AAPL/seed_45/last_model_comparison_20260823_134621.csv`
- `results/top10_without_sentiment/random/AAPL/seed_46/last_model_comparison_20260823_135331.csv`
- `results/top10_without_sentiment/random/AAPL/seed_47/last_model_comparison_20260823_140043.csv`
- `results/top10_without_sentiment/random/AAPL/seed_48/last_model_comparison_20260823_140754.csv`
- `results/top10_without_sentiment/random/AAPL/seed_49/last_model_comparison_20260823_141509.csv`
- `results/top10_without_sentiment/random/AAPL/seed_50/last_model_comparison_20260823_142222.csv`
- `results/top10_without_sentiment/random/AAPL/seed_51/last_model_comparison_20260823_142932.csv`
- `results/top10_without_sentiment/random/AMZN/seed_42/last_model_comparison_20260823_154836.csv`
- `results/top10_without_sentiment/random/AMZN/seed_43/last_model_comparison_20260823_155548.csv`
- `results/top10_without_sentiment/random/AMZN/seed_44/last_model_comparison_20260823_160300.csv`
- `results/top10_without_sentiment/random/AMZN/seed_45/last_model_comparison_20260823_161011.csv`
- `results/top10_without_sentiment/random/AMZN/seed_46/last_model_comparison_20260823_161730.csv`
- `results/top10_without_sentiment/random/AMZN/seed_47/last_model_comparison_20260823_162448.csv`
- `results/top10_without_sentiment/random/AMZN/seed_48/last_model_comparison_20260823_163212.csv`
- `results/top10_without_sentiment/random/AMZN/seed_49/last_model_comparison_20260823_163932.csv`
- `results/top10_without_sentiment/random/AMZN/seed_50/last_model_comparison_20260823_164648.csv`
- `results/top10_without_sentiment/random/AMZN/seed_51/last_model_comparison_20260823_165404.csv`
- `results/top10_without_sentiment/random/AVGO/seed_42/last_model_comparison_20260823_181414.csv`
- `results/top10_without_sentiment/random/AVGO/seed_43/last_model_comparison_20260823_182133.csv`
- `results/top10_without_sentiment/random/AVGO/seed_44/last_model_comparison_20260823_182849.csv`
- `results/top10_without_sentiment/random/AVGO/seed_45/last_model_comparison_20260823_183608.csv`
- `results/top10_without_sentiment/random/AVGO/seed_46/last_model_comparison_20260823_184325.csv`
- `results/top10_without_sentiment/random/AVGO/seed_47/last_model_comparison_20260823_185045.csv`
- `results/top10_without_sentiment/random/AVGO/seed_48/last_model_comparison_20260823_185808.csv`
- `results/top10_without_sentiment/random/AVGO/seed_49/last_model_comparison_20260823_190527.csv`
- `results/top10_without_sentiment/random/AVGO/seed_50/last_model_comparison_20260823_191247.csv`
- `results/top10_without_sentiment/random/AVGO/seed_51/last_model_comparison_20260823_192001.csv`
- `results/top10_without_sentiment/random/COST/seed_42/last_model_comparison_20260823_215211.csv`
- `results/top10_without_sentiment/random/COST/seed_43/last_model_comparison_20260823_215929.csv`
- `results/top10_without_sentiment/random/COST/seed_44/last_model_comparison_20260823_220644.csv`
- `results/top10_without_sentiment/random/COST/seed_45/last_model_comparison_20260823_221358.csv`
- `results/top10_without_sentiment/random/COST/seed_46/last_model_comparison_20260823_222114.csv`
- `results/top10_without_sentiment/random/COST/seed_47/last_model_comparison_20260823_222827.csv`
- `results/top10_without_sentiment/random/COST/seed_48/last_model_comparison_20260823_223539.csv`
- `results/top10_without_sentiment/random/COST/seed_49/last_model_comparison_20260823_224252.csv`
- `results/top10_without_sentiment/random/COST/seed_50/last_model_comparison_20260823_225003.csv`
- `results/top10_without_sentiment/random/COST/seed_51/last_model_comparison_20260823_225716.csv`
- `results/top10_without_sentiment/random/GOOGL/seed_42/last_model_comparison_20260823_170119.csv`
- `results/top10_without_sentiment/random/GOOGL/seed_43/last_model_comparison_20260823_170839.csv`
- `results/top10_without_sentiment/random/GOOGL/seed_44/last_model_comparison_20260823_171554.csv`
- `results/top10_without_sentiment/random/GOOGL/seed_45/last_model_comparison_20260823_172315.csv`
- `results/top10_without_sentiment/random/GOOGL/seed_46/last_model_comparison_20260823_173030.csv`
- `results/top10_without_sentiment/random/GOOGL/seed_47/last_model_comparison_20260823_173746.csv`
- `results/top10_without_sentiment/random/GOOGL/seed_48/last_model_comparison_20260823_174506.csv`
- `results/top10_without_sentiment/random/GOOGL/seed_49/last_model_comparison_20260823_175216.csv`
- `results/top10_without_sentiment/random/GOOGL/seed_50/last_model_comparison_20260823_175936.csv`
- `results/top10_without_sentiment/random/GOOGL/seed_51/last_model_comparison_20260823_180656.csv`
- `results/top10_without_sentiment/random/META/seed_42/last_model_comparison_20260823_192716.csv`
- `results/top10_without_sentiment/random/META/seed_43/last_model_comparison_20260823_193433.csv`
- `results/top10_without_sentiment/random/META/seed_44/last_model_comparison_20260823_194151.csv`
- `results/top10_without_sentiment/random/META/seed_45/last_model_comparison_20260823_194915.csv`
- `results/top10_without_sentiment/random/META/seed_46/last_model_comparison_20260823_195635.csv`
- `results/top10_without_sentiment/random/META/seed_47/last_model_comparison_20260823_200348.csv`
- `results/top10_without_sentiment/random/META/seed_48/last_model_comparison_20260823_201058.csv`
- `results/top10_without_sentiment/random/META/seed_49/last_model_comparison_20260823_201810.csv`
- `results/top10_without_sentiment/random/META/seed_50/last_model_comparison_20260823_202525.csv`
- `results/top10_without_sentiment/random/META/seed_51/last_model_comparison_20260823_203238.csv`
- `results/top10_without_sentiment/random/MSFT/seed_42/last_model_comparison_20260823_143637.csv`
- `results/top10_without_sentiment/random/MSFT/seed_43/last_model_comparison_20260823_144347.csv`
- `results/top10_without_sentiment/random/MSFT/seed_44/last_model_comparison_20260823_145056.csv`
- `results/top10_without_sentiment/random/MSFT/seed_45/last_model_comparison_20260823_145808.csv`
- `results/top10_without_sentiment/random/MSFT/seed_46/last_model_comparison_20260823_150520.csv`
- `results/top10_without_sentiment/random/MSFT/seed_47/last_model_comparison_20260823_151229.csv`
- `results/top10_without_sentiment/random/MSFT/seed_48/last_model_comparison_20260823_151945.csv`
- `results/top10_without_sentiment/random/MSFT/seed_49/last_model_comparison_20260823_152659.csv`
- `results/top10_without_sentiment/random/MSFT/seed_50/last_model_comparison_20260823_153409.csv`
- `results/top10_without_sentiment/random/MSFT/seed_51/last_model_comparison_20260823_154120.csv`
- `results/top10_without_sentiment/random/NVDA/seed_42/last_model_comparison_20260823_121329.csv`
- `results/top10_without_sentiment/random/NVDA/seed_43/last_model_comparison_20260823_122037.csv`
- `results/top10_without_sentiment/random/NVDA/seed_44/last_model_comparison_20260823_122748.csv`
- `results/top10_without_sentiment/random/NVDA/seed_45/last_model_comparison_20260823_123503.csv`
- `results/top10_without_sentiment/random/NVDA/seed_46/last_model_comparison_20260823_124206.csv`
- `results/top10_without_sentiment/random/NVDA/seed_47/last_model_comparison_20260823_124916.csv`
- `results/top10_without_sentiment/random/NVDA/seed_48/last_model_comparison_20260823_125625.csv`
- `results/top10_without_sentiment/random/NVDA/seed_49/last_model_comparison_20260823_130330.csv`
- `results/top10_without_sentiment/random/NVDA/seed_50/last_model_comparison_20260823_131037.csv`
- `results/top10_without_sentiment/random/NVDA/seed_51/last_model_comparison_20260823_131743.csv`
- `results/top10_without_sentiment/random/TSLA/seed_42/last_model_comparison_20260823_203954.csv`
- `results/top10_without_sentiment/random/TSLA/seed_43/last_model_comparison_20260823_204701.csv`
- `results/top10_without_sentiment/random/TSLA/seed_44/last_model_comparison_20260823_205414.csv`
- `results/top10_without_sentiment/random/TSLA/seed_45/last_model_comparison_20260823_210130.csv`
- `results/top10_without_sentiment/random/TSLA/seed_46/last_model_comparison_20260823_210844.csv`
- `results/top10_without_sentiment/random/TSLA/seed_47/last_model_comparison_20260823_211556.csv`
- `results/top10_without_sentiment/random/TSLA/seed_48/last_model_comparison_20260823_212308.csv`
- `results/top10_without_sentiment/random/TSLA/seed_49/last_model_comparison_20260823_213024.csv`
- `results/top10_without_sentiment/random/TSLA/seed_50/last_model_comparison_20260823_213738.csv`
- `results/top10_without_sentiment/random/TSLA/seed_51/last_model_comparison_20260823_214457.csv`
- `results/top10_without_sentiment/random/WMT/seed_42/last_model_comparison_20260823_230435.csv`
- `results/top10_without_sentiment/random/WMT/seed_43/last_model_comparison_20260823_231152.csv`
- `results/top10_without_sentiment/random/WMT/seed_44/last_model_comparison_20260823_231905.csv`
- `results/top10_without_sentiment/random/WMT/seed_45/last_model_comparison_20260823_232622.csv`
- `results/top10_without_sentiment/random/WMT/seed_46/last_model_comparison_20260823_233331.csv`
- `results/top10_without_sentiment/random/WMT/seed_47/last_model_comparison_20260823_234042.csv`
- `results/top10_without_sentiment/random/WMT/seed_48/last_model_comparison_20260823_234757.csv`
- `results/top10_without_sentiment/random/WMT/seed_49/last_model_comparison_20260823_235511.csv`
- `results/top10_without_sentiment/random/WMT/seed_50/last_model_comparison_20260824_000226.csv`
- `results/top10_without_sentiment/random/WMT/seed_51/last_model_comparison_20260824_000932.csv`

Excluded inventory rows: 400.
- duplicate deterministic baseline; reference strategy selected: 300
- alternate strategy-specific GRU; reference strategy selected: 100

The inventory selects the latest timestamp only for duplicate bundles. A duplicate with multiple recoverable configuration signatures is an error. Strategy-specific GRU rows outside the configured reference strategy and duplicate deterministic baselines are retained in `run_inventory.csv` but excluded from the canonical dataset.

## Metrics and direction-accuracy audit

MSE and MAE are recomputed over all saved rolling-step × horizon values whenever score files exist. Stored summary values are checked against those reconstructions.

Direction accuracy uses `project_within_trajectory_v1: sign of consecutive forecast-horizon differences equals sign of consecutive target differences; relative-return paths additionally include the known zero origin; cumulative/excess log-return targets compare the binary indicators (forecast > 0) and (target > 0) at each horizon`. The identical implementation is applied to learned models and baselines. Naive-last and mean-context therefore have valid direction scores when trajectories are saved; a constant predicted value path generally produces zero-valued predicted differences for a value target, which only count as correct when the corresponding true difference is also zero. Unsupported values remain missing.

All reported values remain in the saved target space (`normalization`, `forecast_target`, and `target_definition` are preserved in `all_runs_tidy.csv`). No conversion to absolute prices is inferred.

## Aggregation and statistical procedure

For each method, metrics are first averaged over seeds within an equity. Overall metrics and ranks are then averaged over equities. Variability in the main table is the standard deviation across equity-level seed means.

Paired differences use Δ = method − naive-last, so negative values favour the method. Relative improvement is `100 × (naive − method) / naive`, so positive values favour the method. Pairing requires the same equity, seed, strategy-specific run bundle, target definition, normalization, metric definition, horizon, and saved target signature whenever available.

Primary inference averages seed-level paired differences within each equity and uses equities as the statistical units. The 95% interval is a percentile bootstrap that resamples equity-level means. The Wilcoxon result is an exact two-sided signed-rank sign-permutation test (zero differences removed); rank-biserial correlation is signed in Δ coordinates, so a negative effect favours the model. P-values are Holm-adjusted separately for the three learned-model MSE and MAE comparisons. Seed-level win counts and distribution figures are descriptive only.

## Paired results snapshot

| model | mean_delta_mse | mse_ci_low | mse_ci_high | mse_holm_p_value | mse_stock_wins | mse_run_wins | mse_run_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Shared-target JEPA--MAE | 0.0006662 | 0.000285 | 0.001176 | 0.005859 | 0 | 0 | 100 |
| Local-MAE/Long-JEPA | 0.0008396 | 0.0003078 | 0.001564 | 0.005859 | 0 | 2 | 100 |
| GRU | 0.0002975 | 0.0001014 | 0.0005421 | 0.005859 | 1 | 12 | 100 |

## Representative trajectory

- Stock: NVDA
- Seed: 46
- Rolling step: 0
- Target dates: 2025-04-01T00:00:00, 2025-04-02T00:00:00, 2025-04-03T00:00:00, 2025-04-04T00:00:00, 2025-04-07T00:00:00
- Selection rule: Use NVDA when it has complete saved predictions (otherwise the alphabetically first complete stock); select the Shared-target seed whose overall test MSE is closest to that stock's median, breaking ties by smaller seed; plot the first saved rolling step.

## Interpretation limits

The Shared-target and Local-MAE/Long-JEPA rows are complete forecasting procedures, not causal ablations of JEPA, MAE, EMA, or pre-training. Unless separately controlled rows appear in the inventory, random-initialized Transformer, JEPA-only, MAE-only, and online-vs-EMA causal analyses cannot be produced. Decreasing pre-training or downstream loss demonstrates optimization behaviour only.

Forecast-horizon and qualitative figures are omitted when score-level predictions are absent. Pre-training JEPA/MAE/total-loss diagnostics are omitted when checkpoint validation histories or training histories are absent. These omissions are recorded in `artifact_manifest.csv`.

## Generated artifact manifest

| artifact | status | description |
| --- | --- | --- |
| data/all_runs_tidy.csv | generated | Canonical analysis dataset: all_runs_tidy.csv |
| data/run_inventory.csv | generated | Canonical analysis dataset: run_inventory.csv |
| data/missing_runs.csv | generated | Canonical analysis dataset: missing_runs.csv |
| data/coverage_summary.csv | generated | Canonical analysis dataset: coverage_summary.csv |
| data/configuration_inventory.csv | generated | Canonical analysis dataset: configuration_inventory.csv |
| data/predictions_tidy.csv | generated | Canonical analysis dataset: predictions_tidy.csv |
| data/paired_run_differences.csv | generated | Canonical analysis dataset: paired_run_differences.csv |
| data/paired_stock_differences.csv | generated | Canonical analysis dataset: paired_stock_differences.csv |
| data/stock_summary.csv | generated | Canonical analysis dataset: stock_summary.csv |
| data/overall_summary.csv | generated | Canonical analysis dataset: overall_summary.csv |
| data/paired_vs_naive.csv | generated | Canonical analysis dataset: paired_vs_naive.csv |
| data/relative_performance_by_stock.csv | generated | Canonical analysis dataset: relative_performance_by_stock.csv |
| data/horizon_metrics.csv | generated | Canonical analysis dataset: horizon_metrics.csv |
| tables/table_main_metrics.csv | generated | Thesis or appendix table |
| tables/table_main_metrics.tex | generated | Thesis or appendix table |
| tables/table_paired_vs_naive.csv | generated | Thesis or appendix table |
| tables/table_paired_vs_naive.tex | generated | Thesis or appendix table |
| tables/table_appendix_stock_metrics.csv | generated | Thesis or appendix table |
| tables/table_appendix_stock_metrics.tex | generated | Thesis or appendix table |
| tables/table_reproducibility.tex | generated | Thesis or appendix table |
| figures/fig_paired_mse_forest.pdf | generated | Publication-quality thesis figure |
| figures/fig_paired_mse_forest.png | generated | Publication-quality thesis figure |
| figures/fig_paired_mae_forest.pdf | generated | Publication-quality thesis figure |
| figures/fig_paired_mae_forest.png | generated | Publication-quality thesis figure |
| figures/fig_relative_mse_heatmap.pdf | generated | Publication-quality thesis figure |
| figures/fig_relative_mse_heatmap.png | generated | Publication-quality thesis figure |
| figures/fig_relative_mae_heatmap.pdf | generated | Publication-quality thesis figure |
| figures/fig_relative_mae_heatmap.png | generated | Publication-quality thesis figure |
| figures/fig_direction_accuracy_heatmap.pdf | generated | Publication-quality thesis figure |
| figures/fig_direction_accuracy_heatmap.png | generated | Publication-quality thesis figure |
| figures/fig_mse_by_horizon.pdf | generated | Publication-quality thesis figure |
| figures/fig_mse_by_horizon.png | generated | Publication-quality thesis figure |
| figures/fig_mae_by_horizon.pdf | generated | Publication-quality thesis figure |
| figures/fig_mae_by_horizon.png | generated | Publication-quality thesis figure |
| figures/fig_direction_by_horizon.pdf | generated | Publication-quality thesis figure |
| figures/fig_direction_by_horizon.png | generated | Publication-quality thesis figure |
| figures/fig_seed_level_delta_mse_distribution.pdf | generated | Publication-quality thesis figure |
| figures/fig_seed_level_delta_mse_distribution.png | generated | Publication-quality thesis figure |
| figures/fig_representative_prediction_trajectory.pdf | generated | Publication-quality thesis figure |
| figures/fig_representative_prediction_trajectory.png | generated | Publication-quality thesis figure |
| figures/diagnostics/diagnostic_downstream_training_loss.pdf | generated | Publication-quality thesis figure |
| figures/diagnostics/diagnostic_downstream_training_loss.png | generated | Publication-quality thesis figure |
| figures/diagnostics/diagnostic_downstream_validation_mse.pdf | generated | Publication-quality thesis figure |
| figures/diagnostics/diagnostic_downstream_validation_mse.png | generated | Publication-quality thesis figure |
| figures/diagnostics/diagnostic_pretraining_losses.pdf | omitted | Pre-training JEPA/MAE/total histories were unavailable. |
| data/representative_example.json | generated | Deterministic representative-example selection metadata |
| README.md | generated | Methodology, coverage, exclusions, interpretation limits, and reproduction command |
| artifact_manifest.csv | generated | Mapping from every planned output artifact to its source data and status |
| analysis_metadata.json | generated | Machine-readable analysis provenance |

## Reproduction command

```bash
conda run --no-capture-output -n ts-jepa python analyze_thesis_results.py \
  --config config/experiments/top10_without_sentiment.json \
  --reference-strategy random \
  --bootstrap-samples 20000 \
  --analysis-seed 20260822
```

The analysis bootstrap seed and sample count are recorded in the command and output metadata. PDF figures are vector outputs; matching PNG files are previews.

## Git publication

- Immutable snapshot: `2fab810c1e1d-d0fb2944255b`
- Full raw experiment outputs are intentionally excluded from Git.
- `SHA256SUMS` verifies every published file.
- Files omitted by publication policy: 1. See `publication_manifest.csv`.
- Large/raw artifacts should be attached to the matching GitHub Release.
