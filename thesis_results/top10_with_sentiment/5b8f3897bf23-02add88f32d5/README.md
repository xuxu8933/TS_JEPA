# TS-JEPA thesis result manifest

This directory is generated exclusively from saved experiment artifacts; no model training is performed.

## Analysis scope and coverage

- Config: `config/experiments/top10_with_sentiment.json`
- Result root: `results/top10_with_sentiment`
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

- `results/top10_with_sentiment/local_long/AAPL/seed_42/last_model_comparison_20260824_081227.csv`
- `results/top10_with_sentiment/local_long/AAPL/seed_43/last_model_comparison_20260824_082032.csv`
- `results/top10_with_sentiment/local_long/AAPL/seed_44/last_model_comparison_20260824_082834.csv`
- `results/top10_with_sentiment/local_long/AAPL/seed_45/last_model_comparison_20260824_083640.csv`
- `results/top10_with_sentiment/local_long/AAPL/seed_46/last_model_comparison_20260824_084444.csv`
- `results/top10_with_sentiment/local_long/AAPL/seed_47/last_model_comparison_20260824_085238.csv`
- `results/top10_with_sentiment/local_long/AAPL/seed_48/last_model_comparison_20260824_090041.csv`
- `results/top10_with_sentiment/local_long/AAPL/seed_49/last_model_comparison_20260824_090836.csv`
- `results/top10_with_sentiment/local_long/AAPL/seed_50/last_model_comparison_20260824_091638.csv`
- `results/top10_with_sentiment/local_long/AAPL/seed_51/last_model_comparison_20260824_092435.csv`
- `results/top10_with_sentiment/local_long/AMZN/seed_42/last_model_comparison_20260824_105202.csv`
- `results/top10_with_sentiment/local_long/AMZN/seed_43/last_model_comparison_20260824_105959.csv`
- `results/top10_with_sentiment/local_long/AMZN/seed_44/last_model_comparison_20260824_110800.csv`
- `results/top10_with_sentiment/local_long/AMZN/seed_45/last_model_comparison_20260824_111555.csv`
- `results/top10_with_sentiment/local_long/AMZN/seed_46/last_model_comparison_20260824_112349.csv`
- `results/top10_with_sentiment/local_long/AMZN/seed_47/last_model_comparison_20260824_113150.csv`
- `results/top10_with_sentiment/local_long/AMZN/seed_48/last_model_comparison_20260824_113948.csv`
- `results/top10_with_sentiment/local_long/AMZN/seed_49/last_model_comparison_20260824_114746.csv`
- `results/top10_with_sentiment/local_long/AMZN/seed_50/last_model_comparison_20260824_115543.csv`
- `results/top10_with_sentiment/local_long/AMZN/seed_51/last_model_comparison_20260824_120346.csv`
- `results/top10_with_sentiment/local_long/AVGO/seed_42/last_model_comparison_20260824_133123.csv`
- `results/top10_with_sentiment/local_long/AVGO/seed_43/last_model_comparison_20260824_133922.csv`
- `results/top10_with_sentiment/local_long/AVGO/seed_44/last_model_comparison_20260824_134722.csv`
- `results/top10_with_sentiment/local_long/AVGO/seed_45/last_model_comparison_20260824_135521.csv`
- `results/top10_with_sentiment/local_long/AVGO/seed_46/last_model_comparison_20260824_140317.csv`
- `results/top10_with_sentiment/local_long/AVGO/seed_47/last_model_comparison_20260824_141119.csv`
- `results/top10_with_sentiment/local_long/AVGO/seed_48/last_model_comparison_20260824_141917.csv`
- `results/top10_with_sentiment/local_long/AVGO/seed_49/last_model_comparison_20260824_142719.csv`
- `results/top10_with_sentiment/local_long/AVGO/seed_50/last_model_comparison_20260824_143521.csv`
- `results/top10_with_sentiment/local_long/AVGO/seed_51/last_model_comparison_20260824_144323.csv`
- `results/top10_with_sentiment/local_long/COST/seed_42/last_model_comparison_20260824_173026.csv`
- `results/top10_with_sentiment/local_long/COST/seed_43/last_model_comparison_20260824_173825.csv`
- `results/top10_with_sentiment/local_long/COST/seed_44/last_model_comparison_20260824_174623.csv`
- `results/top10_with_sentiment/local_long/COST/seed_45/last_model_comparison_20260824_175424.csv`
- `results/top10_with_sentiment/local_long/COST/seed_46/last_model_comparison_20260824_180226.csv`
- `results/top10_with_sentiment/local_long/COST/seed_47/last_model_comparison_20260824_181024.csv`
- `results/top10_with_sentiment/local_long/COST/seed_48/last_model_comparison_20260824_181827.csv`
- `results/top10_with_sentiment/local_long/COST/seed_49/last_model_comparison_20260824_182634.csv`
- `results/top10_with_sentiment/local_long/COST/seed_50/last_model_comparison_20260824_183439.csv`
- `results/top10_with_sentiment/local_long/COST/seed_51/last_model_comparison_20260824_184241.csv`
- `results/top10_with_sentiment/local_long/GOOGL/seed_42/last_model_comparison_20260824_121142.csv`
- `results/top10_with_sentiment/local_long/GOOGL/seed_43/last_model_comparison_20260824_121948.csv`
- `results/top10_with_sentiment/local_long/GOOGL/seed_44/last_model_comparison_20260824_122745.csv`
- `results/top10_with_sentiment/local_long/GOOGL/seed_45/last_model_comparison_20260824_123542.csv`
- `results/top10_with_sentiment/local_long/GOOGL/seed_46/last_model_comparison_20260824_124339.csv`
- `results/top10_with_sentiment/local_long/GOOGL/seed_47/last_model_comparison_20260824_125139.csv`
- `results/top10_with_sentiment/local_long/GOOGL/seed_48/last_model_comparison_20260824_125933.csv`
- `results/top10_with_sentiment/local_long/GOOGL/seed_49/last_model_comparison_20260824_130729.csv`
- `results/top10_with_sentiment/local_long/GOOGL/seed_50/last_model_comparison_20260824_131531.csv`
- `results/top10_with_sentiment/local_long/GOOGL/seed_51/last_model_comparison_20260824_132326.csv`
- `results/top10_with_sentiment/local_long/META/seed_42/last_model_comparison_20260824_145118.csv`
- `results/top10_with_sentiment/local_long/META/seed_43/last_model_comparison_20260824_145912.csv`
- `results/top10_with_sentiment/local_long/META/seed_44/last_model_comparison_20260824_150708.csv`
- `results/top10_with_sentiment/local_long/META/seed_45/last_model_comparison_20260824_151503.csv`
- `results/top10_with_sentiment/local_long/META/seed_46/last_model_comparison_20260824_152301.csv`
- `results/top10_with_sentiment/local_long/META/seed_47/last_model_comparison_20260824_153052.csv`
- `results/top10_with_sentiment/local_long/META/seed_48/last_model_comparison_20260824_153847.csv`
- `results/top10_with_sentiment/local_long/META/seed_49/last_model_comparison_20260824_154645.csv`
- `results/top10_with_sentiment/local_long/META/seed_50/last_model_comparison_20260824_155444.csv`
- `results/top10_with_sentiment/local_long/META/seed_51/last_model_comparison_20260824_160240.csv`
- `results/top10_with_sentiment/local_long/MSFT/seed_42/last_model_comparison_20260824_093232.csv`
- `results/top10_with_sentiment/local_long/MSFT/seed_43/last_model_comparison_20260824_094030.csv`
- `results/top10_with_sentiment/local_long/MSFT/seed_44/last_model_comparison_20260824_094822.csv`
- `results/top10_with_sentiment/local_long/MSFT/seed_45/last_model_comparison_20260824_095621.csv`
- `results/top10_with_sentiment/local_long/MSFT/seed_46/last_model_comparison_20260824_100418.csv`
- `results/top10_with_sentiment/local_long/MSFT/seed_47/last_model_comparison_20260824_101212.csv`
- `results/top10_with_sentiment/local_long/MSFT/seed_48/last_model_comparison_20260824_102007.csv`
- `results/top10_with_sentiment/local_long/MSFT/seed_49/last_model_comparison_20260824_102808.csv`
- `results/top10_with_sentiment/local_long/MSFT/seed_50/last_model_comparison_20260824_103606.csv`
- `results/top10_with_sentiment/local_long/MSFT/seed_51/last_model_comparison_20260824_104404.csv`
- `results/top10_with_sentiment/local_long/NVDA/seed_42/last_model_comparison_20260824_065234.csv`
- `results/top10_with_sentiment/local_long/NVDA/seed_43/last_model_comparison_20260824_070036.csv`
- `results/top10_with_sentiment/local_long/NVDA/seed_44/last_model_comparison_20260824_070838.csv`
- `results/top10_with_sentiment/local_long/NVDA/seed_45/last_model_comparison_20260824_071635.csv`
- `results/top10_with_sentiment/local_long/NVDA/seed_46/last_model_comparison_20260824_072436.csv`
- `results/top10_with_sentiment/local_long/NVDA/seed_47/last_model_comparison_20260824_073241.csv`
- `results/top10_with_sentiment/local_long/NVDA/seed_48/last_model_comparison_20260824_074036.csv`
- `results/top10_with_sentiment/local_long/NVDA/seed_49/last_model_comparison_20260824_074833.csv`
- `results/top10_with_sentiment/local_long/NVDA/seed_50/last_model_comparison_20260824_075630.csv`
- `results/top10_with_sentiment/local_long/NVDA/seed_51/last_model_comparison_20260824_080428.csv`
- `results/top10_with_sentiment/local_long/TSLA/seed_42/last_model_comparison_20260824_161046.csv`
- `results/top10_with_sentiment/local_long/TSLA/seed_43/last_model_comparison_20260824_161847.csv`
- `results/top10_with_sentiment/local_long/TSLA/seed_44/last_model_comparison_20260824_162649.csv`
- `results/top10_with_sentiment/local_long/TSLA/seed_45/last_model_comparison_20260824_163449.csv`
- `results/top10_with_sentiment/local_long/TSLA/seed_46/last_model_comparison_20260824_164247.csv`
- `results/top10_with_sentiment/local_long/TSLA/seed_47/last_model_comparison_20260824_165047.csv`
- `results/top10_with_sentiment/local_long/TSLA/seed_48/last_model_comparison_20260824_165841.csv`
- `results/top10_with_sentiment/local_long/TSLA/seed_49/last_model_comparison_20260824_170635.csv`
- `results/top10_with_sentiment/local_long/TSLA/seed_50/last_model_comparison_20260824_171433.csv`
- `results/top10_with_sentiment/local_long/TSLA/seed_51/last_model_comparison_20260824_172226.csv`
- `results/top10_with_sentiment/local_long/WMT/seed_42/last_model_comparison_20260824_185045.csv`
- `results/top10_with_sentiment/local_long/WMT/seed_43/last_model_comparison_20260824_185850.csv`
- `results/top10_with_sentiment/local_long/WMT/seed_44/last_model_comparison_20260824_190654.csv`
- `results/top10_with_sentiment/local_long/WMT/seed_45/last_model_comparison_20260824_191455.csv`
- `results/top10_with_sentiment/local_long/WMT/seed_46/last_model_comparison_20260824_192259.csv`
- `results/top10_with_sentiment/local_long/WMT/seed_47/last_model_comparison_20260824_193107.csv`
- `results/top10_with_sentiment/local_long/WMT/seed_48/last_model_comparison_20260824_193912.csv`
- `results/top10_with_sentiment/local_long/WMT/seed_49/last_model_comparison_20260824_194716.csv`
- `results/top10_with_sentiment/local_long/WMT/seed_50/last_model_comparison_20260824_195522.csv`
- `results/top10_with_sentiment/local_long/WMT/seed_51/last_model_comparison_20260824_200324.csv`
- `results/top10_with_sentiment/random/AAPL/seed_42/last_model_comparison_20260824_080829.csv`
- `results/top10_with_sentiment/random/AAPL/seed_43/last_model_comparison_20260824_081629.csv`
- `results/top10_with_sentiment/random/AAPL/seed_44/last_model_comparison_20260824_082432.csv`
- `results/top10_with_sentiment/random/AAPL/seed_45/last_model_comparison_20260824_083233.csv`
- `results/top10_with_sentiment/random/AAPL/seed_46/last_model_comparison_20260824_084043.csv`
- `results/top10_with_sentiment/random/AAPL/seed_47/last_model_comparison_20260824_084838.csv`
- `results/top10_with_sentiment/random/AAPL/seed_48/last_model_comparison_20260824_085639.csv`
- `results/top10_with_sentiment/random/AAPL/seed_49/last_model_comparison_20260824_090436.csv`
- `results/top10_with_sentiment/random/AAPL/seed_50/last_model_comparison_20260824_091233.csv`
- `results/top10_with_sentiment/random/AAPL/seed_51/last_model_comparison_20260824_092036.csv`
- `results/top10_with_sentiment/random/AMZN/seed_42/last_model_comparison_20260824_104802.csv`
- `results/top10_with_sentiment/random/AMZN/seed_43/last_model_comparison_20260824_105558.csv`
- `results/top10_with_sentiment/random/AMZN/seed_44/last_model_comparison_20260824_110357.csv`
- `results/top10_with_sentiment/random/AMZN/seed_45/last_model_comparison_20260824_111157.csv`
- `results/top10_with_sentiment/random/AMZN/seed_46/last_model_comparison_20260824_111951.csv`
- `results/top10_with_sentiment/random/AMZN/seed_47/last_model_comparison_20260824_112748.csv`
- `results/top10_with_sentiment/random/AMZN/seed_48/last_model_comparison_20260824_113546.csv`
- `results/top10_with_sentiment/random/AMZN/seed_49/last_model_comparison_20260824_114344.csv`
- `results/top10_with_sentiment/random/AMZN/seed_50/last_model_comparison_20260824_115143.csv`
- `results/top10_with_sentiment/random/AMZN/seed_51/last_model_comparison_20260824_115943.csv`
- `results/top10_with_sentiment/random/AVGO/seed_42/last_model_comparison_20260824_132721.csv`
- `results/top10_with_sentiment/random/AVGO/seed_43/last_model_comparison_20260824_133520.csv`
- `results/top10_with_sentiment/random/AVGO/seed_44/last_model_comparison_20260824_134321.csv`
- `results/top10_with_sentiment/random/AVGO/seed_45/last_model_comparison_20260824_135122.csv`
- `results/top10_with_sentiment/random/AVGO/seed_46/last_model_comparison_20260824_135914.csv`
- `results/top10_with_sentiment/random/AVGO/seed_47/last_model_comparison_20260824_140716.csv`
- `results/top10_with_sentiment/random/AVGO/seed_48/last_model_comparison_20260824_141518.csv`
- `results/top10_with_sentiment/random/AVGO/seed_49/last_model_comparison_20260824_142315.csv`
- `results/top10_with_sentiment/random/AVGO/seed_50/last_model_comparison_20260824_143118.csv`
- `results/top10_with_sentiment/random/AVGO/seed_51/last_model_comparison_20260824_143922.csv`
- `results/top10_with_sentiment/random/COST/seed_42/last_model_comparison_20260824_172624.csv`
- `results/top10_with_sentiment/random/COST/seed_43/last_model_comparison_20260824_173424.csv`
- `results/top10_with_sentiment/random/COST/seed_44/last_model_comparison_20260824_174222.csv`
- `results/top10_with_sentiment/random/COST/seed_45/last_model_comparison_20260824_175021.csv`
- `results/top10_with_sentiment/random/COST/seed_46/last_model_comparison_20260824_175823.csv`
- `results/top10_with_sentiment/random/COST/seed_47/last_model_comparison_20260824_180623.csv`
- `results/top10_with_sentiment/random/COST/seed_48/last_model_comparison_20260824_181423.csv`
- `results/top10_with_sentiment/random/COST/seed_49/last_model_comparison_20260824_182229.csv`
- `results/top10_with_sentiment/random/COST/seed_50/last_model_comparison_20260824_183033.csv`
- `results/top10_with_sentiment/random/COST/seed_51/last_model_comparison_20260824_183838.csv`
- `results/top10_with_sentiment/random/GOOGL/seed_42/last_model_comparison_20260824_120743.csv`
- `results/top10_with_sentiment/random/GOOGL/seed_43/last_model_comparison_20260824_121545.csv`
- `results/top10_with_sentiment/random/GOOGL/seed_44/last_model_comparison_20260824_122342.csv`
- `results/top10_with_sentiment/random/GOOGL/seed_45/last_model_comparison_20260824_123141.csv`
- `results/top10_with_sentiment/random/GOOGL/seed_46/last_model_comparison_20260824_123939.csv`
- `results/top10_with_sentiment/random/GOOGL/seed_47/last_model_comparison_20260824_124737.csv`
- `results/top10_with_sentiment/random/GOOGL/seed_48/last_model_comparison_20260824_125531.csv`
- `results/top10_with_sentiment/random/GOOGL/seed_49/last_model_comparison_20260824_130331.csv`
- `results/top10_with_sentiment/random/GOOGL/seed_50/last_model_comparison_20260824_131129.csv`
- `results/top10_with_sentiment/random/GOOGL/seed_51/last_model_comparison_20260824_131925.csv`
- `results/top10_with_sentiment/random/META/seed_42/last_model_comparison_20260824_144719.csv`
- `results/top10_with_sentiment/random/META/seed_43/last_model_comparison_20260824_145512.csv`
- `results/top10_with_sentiment/random/META/seed_44/last_model_comparison_20260824_150307.csv`
- `results/top10_with_sentiment/random/META/seed_45/last_model_comparison_20260824_151105.csv`
- `results/top10_with_sentiment/random/META/seed_46/last_model_comparison_20260824_151900.csv`
- `results/top10_with_sentiment/random/META/seed_47/last_model_comparison_20260824_152656.csv`
- `results/top10_with_sentiment/random/META/seed_48/last_model_comparison_20260824_153445.csv`
- `results/top10_with_sentiment/random/META/seed_49/last_model_comparison_20260824_154244.csv`
- `results/top10_with_sentiment/random/META/seed_50/last_model_comparison_20260824_155043.csv`
- `results/top10_with_sentiment/random/META/seed_51/last_model_comparison_20260824_155839.csv`
- `results/top10_with_sentiment/random/MSFT/seed_42/last_model_comparison_20260824_092830.csv`
- `results/top10_with_sentiment/random/MSFT/seed_43/last_model_comparison_20260824_093630.csv`
- `results/top10_with_sentiment/random/MSFT/seed_44/last_model_comparison_20260824_094426.csv`
- `results/top10_with_sentiment/random/MSFT/seed_45/last_model_comparison_20260824_095218.csv`
- `results/top10_with_sentiment/random/MSFT/seed_46/last_model_comparison_20260824_100017.csv`
- `results/top10_with_sentiment/random/MSFT/seed_47/last_model_comparison_20260824_100813.csv`
- `results/top10_with_sentiment/random/MSFT/seed_48/last_model_comparison_20260824_101607.csv`
- `results/top10_with_sentiment/random/MSFT/seed_49/last_model_comparison_20260824_102408.csv`
- `results/top10_with_sentiment/random/MSFT/seed_50/last_model_comparison_20260824_103207.csv`
- `results/top10_with_sentiment/random/MSFT/seed_51/last_model_comparison_20260824_104004.csv`
- `results/top10_with_sentiment/random/NVDA/seed_42/last_model_comparison_20260824_064831.csv`
- `results/top10_with_sentiment/random/NVDA/seed_43/last_model_comparison_20260824_065634.csv`
- `results/top10_with_sentiment/random/NVDA/seed_44/last_model_comparison_20260824_070438.csv`
- `results/top10_with_sentiment/random/NVDA/seed_45/last_model_comparison_20260824_071235.csv`
- `results/top10_with_sentiment/random/NVDA/seed_46/last_model_comparison_20260824_072030.csv`
- `results/top10_with_sentiment/random/NVDA/seed_47/last_model_comparison_20260824_072835.csv`
- `results/top10_with_sentiment/random/NVDA/seed_48/last_model_comparison_20260824_073638.csv`
- `results/top10_with_sentiment/random/NVDA/seed_49/last_model_comparison_20260824_074433.csv`
- `results/top10_with_sentiment/random/NVDA/seed_50/last_model_comparison_20260824_075229.csv`
- `results/top10_with_sentiment/random/NVDA/seed_51/last_model_comparison_20260824_080028.csv`
- `results/top10_with_sentiment/random/TSLA/seed_42/last_model_comparison_20260824_160643.csv`
- `results/top10_with_sentiment/random/TSLA/seed_43/last_model_comparison_20260824_161444.csv`
- `results/top10_with_sentiment/random/TSLA/seed_44/last_model_comparison_20260824_162245.csv`
- `results/top10_with_sentiment/random/TSLA/seed_45/last_model_comparison_20260824_163048.csv`
- `results/top10_with_sentiment/random/TSLA/seed_46/last_model_comparison_20260824_163845.csv`
- `results/top10_with_sentiment/random/TSLA/seed_47/last_model_comparison_20260824_164642.csv`
- `results/top10_with_sentiment/random/TSLA/seed_48/last_model_comparison_20260824_165441.csv`
- `results/top10_with_sentiment/random/TSLA/seed_49/last_model_comparison_20260824_170236.csv`
- `results/top10_with_sentiment/random/TSLA/seed_50/last_model_comparison_20260824_171036.csv`
- `results/top10_with_sentiment/random/TSLA/seed_51/last_model_comparison_20260824_171826.csv`
- `results/top10_with_sentiment/random/WMT/seed_42/last_model_comparison_20260824_184642.csv`
- `results/top10_with_sentiment/random/WMT/seed_43/last_model_comparison_20260824_185446.csv`
- `results/top10_with_sentiment/random/WMT/seed_44/last_model_comparison_20260824_190252.csv`
- `results/top10_with_sentiment/random/WMT/seed_45/last_model_comparison_20260824_191053.csv`
- `results/top10_with_sentiment/random/WMT/seed_46/last_model_comparison_20260824_191854.csv`
- `results/top10_with_sentiment/random/WMT/seed_47/last_model_comparison_20260824_192702.csv`
- `results/top10_with_sentiment/random/WMT/seed_48/last_model_comparison_20260824_193507.csv`
- `results/top10_with_sentiment/random/WMT/seed_49/last_model_comparison_20260824_194312.csv`
- `results/top10_with_sentiment/random/WMT/seed_50/last_model_comparison_20260824_195116.csv`
- `results/top10_with_sentiment/random/WMT/seed_51/last_model_comparison_20260824_195922.csv`

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
| Shared-target JEPA--MAE | 0.0008896 | 0.0004758 | 0.001386 | 0.005859 | 0 | 0 | 100 |
| Local-MAE/Long-JEPA | 0.00115 | 0.0006352 | 0.001733 | 0.005859 | 0 | 0 | 100 |
| GRU | 0.0003069 | 0.000147 | 0.0004859 | 0.005859 | 1 | 16 | 100 |

## Representative trajectory

- Stock: NVDA
- Seed: 45
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
  --config config/experiments/top10_with_sentiment.json \
  --reference-strategy random \
  --bootstrap-samples 20000 \
  --analysis-seed 20260822
```

The analysis bootstrap seed and sample count are recorded in the command and output metadata. PDF figures are vector outputs; matching PNG files are previews.

## Git publication

- Immutable snapshot: `5b8f3897bf23-02add88f32d5`
- Full raw experiment outputs are intentionally excluded from Git.
- `SHA256SUMS` verifies every published file.
- Files omitted by publication policy: 1. See `publication_manifest.csv`.
- Large/raw artifacts should be attached to the matching GitHub Release.
