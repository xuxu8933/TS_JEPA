# TS-JEPA thesis result manifest

This directory is generated exclusively from saved experiment artifacts; no model training is performed.

## Analysis scope and coverage

- Config: `config/experiments/top10_market_windows_return.json`
- Result root: `results/top10_market_windows_return`
- Expected equities: NVDA, AAPL, MSFT, AMZN, GOOGL, AVGO, META, TSLA, COST, WMT
- Expected seeds: 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
- Strategies: random, local_long
- Canonical baseline/GRU strategy: `random`
- Canonical rows: 600
- Audit issues: 0 errors and 0 warnings

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

- `results/top10_market_windows_return/local_long/AAPL/seed_42/last_model_comparison_20260827_105901.csv`
- `results/top10_market_windows_return/local_long/AAPL/seed_43/last_model_comparison_20260827_105953.csv`
- `results/top10_market_windows_return/local_long/AAPL/seed_44/last_model_comparison_20260827_110045.csv`
- `results/top10_market_windows_return/local_long/AAPL/seed_45/last_model_comparison_20260827_110138.csv`
- `results/top10_market_windows_return/local_long/AAPL/seed_46/last_model_comparison_20260827_110230.csv`
- `results/top10_market_windows_return/local_long/AAPL/seed_47/last_model_comparison_20260827_191636.csv`
- `results/top10_market_windows_return/local_long/AAPL/seed_48/last_model_comparison_20260827_191730.csv`
- `results/top10_market_windows_return/local_long/AAPL/seed_49/last_model_comparison_20260827_211456.csv`
- `results/top10_market_windows_return/local_long/AAPL/seed_50/last_model_comparison_20260828_092520.csv`
- `results/top10_market_windows_return/local_long/AAPL/seed_51/last_model_comparison_20260828_092611.csv`
- `results/top10_market_windows_return/local_long/AMZN/seed_42/last_model_comparison_20260827_110744.csv`
- `results/top10_market_windows_return/local_long/AMZN/seed_43/last_model_comparison_20260827_110836.csv`
- `results/top10_market_windows_return/local_long/AMZN/seed_44/last_model_comparison_20260827_110928.csv`
- `results/top10_market_windows_return/local_long/AMZN/seed_45/last_model_comparison_20260827_111021.csv`
- `results/top10_market_windows_return/local_long/AMZN/seed_46/last_model_comparison_20260827_111112.csv`
- `results/top10_market_windows_return/local_long/AMZN/seed_47/last_model_comparison_20260827_192013.csv`
- `results/top10_market_windows_return/local_long/AMZN/seed_48/last_model_comparison_20260827_192108.csv`
- `results/top10_market_windows_return/local_long/AMZN/seed_49/last_model_comparison_20260827_211642.csv`
- `results/top10_market_windows_return/local_long/AMZN/seed_50/last_model_comparison_20260828_092845.csv`
- `results/top10_market_windows_return/local_long/AMZN/seed_51/last_model_comparison_20260828_092936.csv`
- `results/top10_market_windows_return/local_long/AVGO/seed_42/last_model_comparison_20260827_192352.csv`
- `results/top10_market_windows_return/local_long/AVGO/seed_43/last_model_comparison_20260827_192446.csv`
- `results/top10_market_windows_return/local_long/AVGO/seed_44/last_model_comparison_20260827_192541.csv`
- `results/top10_market_windows_return/local_long/AVGO/seed_45/last_model_comparison_20260827_192635.csv`
- `results/top10_market_windows_return/local_long/AVGO/seed_46/last_model_comparison_20260827_192729.csv`
- `results/top10_market_windows_return/local_long/AVGO/seed_47/last_model_comparison_20260827_192822.csv`
- `results/top10_market_windows_return/local_long/AVGO/seed_48/last_model_comparison_20260827_192916.csv`
- `results/top10_market_windows_return/local_long/AVGO/seed_49/last_model_comparison_20260827_211827.csv`
- `results/top10_market_windows_return/local_long/AVGO/seed_50/last_model_comparison_20260828_093209.csv`
- `results/top10_market_windows_return/local_long/AVGO/seed_51/last_model_comparison_20260828_093300.csv`
- `results/top10_market_windows_return/local_long/COST/seed_42/last_model_comparison_20260828_093716.csv`
- `results/top10_market_windows_return/local_long/COST/seed_43/last_model_comparison_20260828_093807.csv`
- `results/top10_market_windows_return/local_long/COST/seed_44/last_model_comparison_20260828_093858.csv`
- `results/top10_market_windows_return/local_long/COST/seed_45/last_model_comparison_20260828_093949.csv`
- `results/top10_market_windows_return/local_long/COST/seed_46/last_model_comparison_20260828_094040.csv`
- `results/top10_market_windows_return/local_long/COST/seed_47/last_model_comparison_20260828_094132.csv`
- `results/top10_market_windows_return/local_long/COST/seed_48/last_model_comparison_20260828_094223.csv`
- `results/top10_market_windows_return/local_long/COST/seed_49/last_model_comparison_20260828_094315.csv`
- `results/top10_market_windows_return/local_long/COST/seed_50/last_model_comparison_20260828_094406.csv`
- `results/top10_market_windows_return/local_long/COST/seed_51/last_model_comparison_20260828_094457.csv`
- `results/top10_market_windows_return/local_long/GOOGL/seed_42/last_model_comparison_20260827_111204.csv`
- `results/top10_market_windows_return/local_long/GOOGL/seed_43/last_model_comparison_20260827_111256.csv`
- `results/top10_market_windows_return/local_long/GOOGL/seed_44/last_model_comparison_20260827_111348.csv`
- `results/top10_market_windows_return/local_long/GOOGL/seed_45/last_model_comparison_20260827_111441.csv`
- `results/top10_market_windows_return/local_long/GOOGL/seed_46/last_model_comparison_20260827_111534.csv`
- `results/top10_market_windows_return/local_long/GOOGL/seed_47/last_model_comparison_20260827_192202.csv`
- `results/top10_market_windows_return/local_long/GOOGL/seed_48/last_model_comparison_20260827_192257.csv`
- `results/top10_market_windows_return/local_long/GOOGL/seed_49/last_model_comparison_20260827_211734.csv`
- `results/top10_market_windows_return/local_long/GOOGL/seed_50/last_model_comparison_20260828_093026.csv`
- `results/top10_market_windows_return/local_long/GOOGL/seed_51/last_model_comparison_20260828_093117.csv`
- `results/top10_market_windows_return/local_long/META/seed_42/last_model_comparison_20260827_193010.csv`
- `results/top10_market_windows_return/local_long/META/seed_43/last_model_comparison_20260827_193103.csv`
- `results/top10_market_windows_return/local_long/META/seed_44/last_model_comparison_20260827_193156.csv`
- `results/top10_market_windows_return/local_long/META/seed_45/last_model_comparison_20260827_193248.csv`
- `results/top10_market_windows_return/local_long/META/seed_46/last_model_comparison_20260827_193339.csv`
- `results/top10_market_windows_return/local_long/META/seed_47/last_model_comparison_20260827_193432.csv`
- `results/top10_market_windows_return/local_long/META/seed_48/last_model_comparison_20260827_193524.csv`
- `results/top10_market_windows_return/local_long/META/seed_49/last_model_comparison_20260827_211919.csv`
- `results/top10_market_windows_return/local_long/META/seed_50/last_model_comparison_20260828_093351.csv`
- `results/top10_market_windows_return/local_long/META/seed_51/last_model_comparison_20260828_093442.csv`
- `results/top10_market_windows_return/local_long/MSFT/seed_42/last_model_comparison_20260827_110322.csv`
- `results/top10_market_windows_return/local_long/MSFT/seed_43/last_model_comparison_20260827_110414.csv`
- `results/top10_market_windows_return/local_long/MSFT/seed_44/last_model_comparison_20260827_110507.csv`
- `results/top10_market_windows_return/local_long/MSFT/seed_45/last_model_comparison_20260827_110559.csv`
- `results/top10_market_windows_return/local_long/MSFT/seed_46/last_model_comparison_20260827_110651.csv`
- `results/top10_market_windows_return/local_long/MSFT/seed_47/last_model_comparison_20260827_191825.csv`
- `results/top10_market_windows_return/local_long/MSFT/seed_48/last_model_comparison_20260827_191918.csv`
- `results/top10_market_windows_return/local_long/MSFT/seed_49/last_model_comparison_20260827_211550.csv`
- `results/top10_market_windows_return/local_long/MSFT/seed_50/last_model_comparison_20260828_092702.csv`
- `results/top10_market_windows_return/local_long/MSFT/seed_51/last_model_comparison_20260828_092753.csv`
- `results/top10_market_windows_return/local_long/NVDA/seed_42/last_model_comparison_20260827_105441.csv`
- `results/top10_market_windows_return/local_long/NVDA/seed_43/last_model_comparison_20260827_105532.csv`
- `results/top10_market_windows_return/local_long/NVDA/seed_44/last_model_comparison_20260827_105624.csv`
- `results/top10_market_windows_return/local_long/NVDA/seed_45/last_model_comparison_20260827_105717.csv`
- `results/top10_market_windows_return/local_long/NVDA/seed_46/last_model_comparison_20260827_105809.csv`
- `results/top10_market_windows_return/local_long/NVDA/seed_47/last_model_comparison_20260827_191450.csv`
- `results/top10_market_windows_return/local_long/NVDA/seed_48/last_model_comparison_20260827_191543.csv`
- `results/top10_market_windows_return/local_long/NVDA/seed_49/last_model_comparison_20260827_211403.csv`
- `results/top10_market_windows_return/local_long/NVDA/seed_50/last_model_comparison_20260828_092337.csv`
- `results/top10_market_windows_return/local_long/NVDA/seed_51/last_model_comparison_20260828_092429.csv`
- `results/top10_market_windows_return/local_long/TSLA/seed_42/last_model_comparison_20260827_212012.csv`
- `results/top10_market_windows_return/local_long/TSLA/seed_43/last_model_comparison_20260827_212107.csv`
- `results/top10_market_windows_return/local_long/TSLA/seed_44/last_model_comparison_20260827_212204.csv`
- `results/top10_market_windows_return/local_long/TSLA/seed_45/last_model_comparison_20260827_212300.csv`
- `results/top10_market_windows_return/local_long/TSLA/seed_46/last_model_comparison_20260827_212357.csv`
- `results/top10_market_windows_return/local_long/TSLA/seed_47/last_model_comparison_20260827_212450.csv`
- `results/top10_market_windows_return/local_long/TSLA/seed_48/last_model_comparison_20260827_212542.csv`
- `results/top10_market_windows_return/local_long/TSLA/seed_49/last_model_comparison_20260827_212634.csv`
- `results/top10_market_windows_return/local_long/TSLA/seed_50/last_model_comparison_20260828_093534.csv`
- `results/top10_market_windows_return/local_long/TSLA/seed_51/last_model_comparison_20260828_093625.csv`
- `results/top10_market_windows_return/local_long/WMT/seed_42/last_model_comparison_20260828_094549.csv`
- `results/top10_market_windows_return/local_long/WMT/seed_43/last_model_comparison_20260828_094640.csv`
- `results/top10_market_windows_return/local_long/WMT/seed_44/last_model_comparison_20260828_094730.csv`
- `results/top10_market_windows_return/local_long/WMT/seed_45/last_model_comparison_20260828_094822.csv`
- `results/top10_market_windows_return/local_long/WMT/seed_46/last_model_comparison_20260828_094913.csv`
- `results/top10_market_windows_return/local_long/WMT/seed_47/last_model_comparison_20260828_095004.csv`
- `results/top10_market_windows_return/local_long/WMT/seed_48/last_model_comparison_20260828_095054.csv`
- `results/top10_market_windows_return/local_long/WMT/seed_49/last_model_comparison_20260828_095145.csv`
- `results/top10_market_windows_return/local_long/WMT/seed_50/last_model_comparison_20260828_095236.csv`
- `results/top10_market_windows_return/local_long/WMT/seed_51/last_model_comparison_20260828_095327.csv`
- `results/top10_market_windows_return/random/AAPL/seed_42/last_model_comparison_20260827_105901.csv`
- `results/top10_market_windows_return/random/AAPL/seed_43/last_model_comparison_20260827_105953.csv`
- `results/top10_market_windows_return/random/AAPL/seed_44/last_model_comparison_20260827_110046.csv`
- `results/top10_market_windows_return/random/AAPL/seed_45/last_model_comparison_20260827_110138.csv`
- `results/top10_market_windows_return/random/AAPL/seed_46/last_model_comparison_20260827_110230.csv`
- `results/top10_market_windows_return/random/AAPL/seed_47/last_model_comparison_20260827_191634.csv`
- `results/top10_market_windows_return/random/AAPL/seed_48/last_model_comparison_20260827_191729.csv`
- `results/top10_market_windows_return/random/AAPL/seed_49/last_model_comparison_20260827_211456.csv`
- `results/top10_market_windows_return/random/AAPL/seed_50/last_model_comparison_20260828_092520.csv`
- `results/top10_market_windows_return/random/AAPL/seed_51/last_model_comparison_20260828_092611.csv`
- `results/top10_market_windows_return/random/AMZN/seed_42/last_model_comparison_20260827_110743.csv`
- `results/top10_market_windows_return/random/AMZN/seed_43/last_model_comparison_20260827_110835.csv`
- `results/top10_market_windows_return/random/AMZN/seed_44/last_model_comparison_20260827_110927.csv`
- `results/top10_market_windows_return/random/AMZN/seed_45/last_model_comparison_20260827_111019.csv`
- `results/top10_market_windows_return/random/AMZN/seed_46/last_model_comparison_20260827_111111.csv`
- `results/top10_market_windows_return/random/AMZN/seed_47/last_model_comparison_20260827_192013.csv`
- `results/top10_market_windows_return/random/AMZN/seed_48/last_model_comparison_20260827_192107.csv`
- `results/top10_market_windows_return/random/AMZN/seed_49/last_model_comparison_20260827_211642.csv`
- `results/top10_market_windows_return/random/AMZN/seed_50/last_model_comparison_20260828_092844.csv`
- `results/top10_market_windows_return/random/AMZN/seed_51/last_model_comparison_20260828_092935.csv`
- `results/top10_market_windows_return/random/AVGO/seed_42/last_model_comparison_20260827_192351.csv`
- `results/top10_market_windows_return/random/AVGO/seed_43/last_model_comparison_20260827_192446.csv`
- `results/top10_market_windows_return/random/AVGO/seed_44/last_model_comparison_20260827_192541.csv`
- `results/top10_market_windows_return/random/AVGO/seed_45/last_model_comparison_20260827_192635.csv`
- `results/top10_market_windows_return/random/AVGO/seed_46/last_model_comparison_20260827_192729.csv`
- `results/top10_market_windows_return/random/AVGO/seed_47/last_model_comparison_20260827_192822.csv`
- `results/top10_market_windows_return/random/AVGO/seed_48/last_model_comparison_20260827_192915.csv`
- `results/top10_market_windows_return/random/AVGO/seed_49/last_model_comparison_20260827_211826.csv`
- `results/top10_market_windows_return/random/AVGO/seed_50/last_model_comparison_20260828_093208.csv`
- `results/top10_market_windows_return/random/AVGO/seed_51/last_model_comparison_20260828_093259.csv`
- `results/top10_market_windows_return/random/COST/seed_42/last_model_comparison_20260828_093716.csv`
- `results/top10_market_windows_return/random/COST/seed_43/last_model_comparison_20260828_093807.csv`
- `results/top10_market_windows_return/random/COST/seed_44/last_model_comparison_20260828_093858.csv`
- `results/top10_market_windows_return/random/COST/seed_45/last_model_comparison_20260828_093949.csv`
- `results/top10_market_windows_return/random/COST/seed_46/last_model_comparison_20260828_094041.csv`
- `results/top10_market_windows_return/random/COST/seed_47/last_model_comparison_20260828_094132.csv`
- `results/top10_market_windows_return/random/COST/seed_48/last_model_comparison_20260828_094223.csv`
- `results/top10_market_windows_return/random/COST/seed_49/last_model_comparison_20260828_094314.csv`
- `results/top10_market_windows_return/random/COST/seed_50/last_model_comparison_20260828_094405.csv`
- `results/top10_market_windows_return/random/COST/seed_51/last_model_comparison_20260828_094457.csv`
- `results/top10_market_windows_return/random/GOOGL/seed_42/last_model_comparison_20260827_111203.csv`
- `results/top10_market_windows_return/random/GOOGL/seed_43/last_model_comparison_20260827_111256.csv`
- `results/top10_market_windows_return/random/GOOGL/seed_44/last_model_comparison_20260827_111348.csv`
- `results/top10_market_windows_return/random/GOOGL/seed_45/last_model_comparison_20260827_111441.csv`
- `results/top10_market_windows_return/random/GOOGL/seed_46/last_model_comparison_20260827_111533.csv`
- `results/top10_market_windows_return/random/GOOGL/seed_47/last_model_comparison_20260827_192202.csv`
- `results/top10_market_windows_return/random/GOOGL/seed_48/last_model_comparison_20260827_192256.csv`
- `results/top10_market_windows_return/random/GOOGL/seed_49/last_model_comparison_20260827_211734.csv`
- `results/top10_market_windows_return/random/GOOGL/seed_50/last_model_comparison_20260828_093026.csv`
- `results/top10_market_windows_return/random/GOOGL/seed_51/last_model_comparison_20260828_093117.csv`
- `results/top10_market_windows_return/random/META/seed_42/last_model_comparison_20260827_193009.csv`
- `results/top10_market_windows_return/random/META/seed_43/last_model_comparison_20260827_193103.csv`
- `results/top10_market_windows_return/random/META/seed_44/last_model_comparison_20260827_193154.csv`
- `results/top10_market_windows_return/random/META/seed_45/last_model_comparison_20260827_193246.csv`
- `results/top10_market_windows_return/random/META/seed_46/last_model_comparison_20260827_193338.csv`
- `results/top10_market_windows_return/random/META/seed_47/last_model_comparison_20260827_193430.csv`
- `results/top10_market_windows_return/random/META/seed_48/last_model_comparison_20260827_193523.csv`
- `results/top10_market_windows_return/random/META/seed_49/last_model_comparison_20260827_211917.csv`
- `results/top10_market_windows_return/random/META/seed_50/last_model_comparison_20260828_093351.csv`
- `results/top10_market_windows_return/random/META/seed_51/last_model_comparison_20260828_093442.csv`
- `results/top10_market_windows_return/random/MSFT/seed_42/last_model_comparison_20260827_110322.csv`
- `results/top10_market_windows_return/random/MSFT/seed_43/last_model_comparison_20260827_110415.csv`
- `results/top10_market_windows_return/random/MSFT/seed_44/last_model_comparison_20260827_110506.csv`
- `results/top10_market_windows_return/random/MSFT/seed_45/last_model_comparison_20260827_110559.csv`
- `results/top10_market_windows_return/random/MSFT/seed_46/last_model_comparison_20260827_110651.csv`
- `results/top10_market_windows_return/random/MSFT/seed_47/last_model_comparison_20260827_191823.csv`
- `results/top10_market_windows_return/random/MSFT/seed_48/last_model_comparison_20260827_191917.csv`
- `results/top10_market_windows_return/random/MSFT/seed_49/last_model_comparison_20260827_211550.csv`
- `results/top10_market_windows_return/random/MSFT/seed_50/last_model_comparison_20260828_092702.csv`
- `results/top10_market_windows_return/random/MSFT/seed_51/last_model_comparison_20260828_092753.csv`
- `results/top10_market_windows_return/random/NVDA/seed_42/last_model_comparison_20260827_105440.csv`
- `results/top10_market_windows_return/random/NVDA/seed_43/last_model_comparison_20260827_105532.csv`
- `results/top10_market_windows_return/random/NVDA/seed_44/last_model_comparison_20260827_105624.csv`
- `results/top10_market_windows_return/random/NVDA/seed_45/last_model_comparison_20260827_105716.csv`
- `results/top10_market_windows_return/random/NVDA/seed_46/last_model_comparison_20260827_105809.csv`
- `results/top10_market_windows_return/random/NVDA/seed_47/last_model_comparison_20260827_191451.csv`
- `results/top10_market_windows_return/random/NVDA/seed_48/last_model_comparison_20260827_191541.csv`
- `results/top10_market_windows_return/random/NVDA/seed_49/last_model_comparison_20260827_211403.csv`
- `results/top10_market_windows_return/random/NVDA/seed_50/last_model_comparison_20260828_092337.csv`
- `results/top10_market_windows_return/random/NVDA/seed_51/last_model_comparison_20260828_092429.csv`
- `results/top10_market_windows_return/random/TSLA/seed_42/last_model_comparison_20260827_212010.csv`
- `results/top10_market_windows_return/random/TSLA/seed_43/last_model_comparison_20260827_212107.csv`
- `results/top10_market_windows_return/random/TSLA/seed_44/last_model_comparison_20260827_212203.csv`
- `results/top10_market_windows_return/random/TSLA/seed_45/last_model_comparison_20260827_212259.csv`
- `results/top10_market_windows_return/random/TSLA/seed_46/last_model_comparison_20260827_212356.csv`
- `results/top10_market_windows_return/random/TSLA/seed_47/last_model_comparison_20260827_212449.csv`
- `results/top10_market_windows_return/random/TSLA/seed_48/last_model_comparison_20260827_212542.csv`
- `results/top10_market_windows_return/random/TSLA/seed_49/last_model_comparison_20260827_212634.csv`
- `results/top10_market_windows_return/random/TSLA/seed_50/last_model_comparison_20260828_093533.csv`
- `results/top10_market_windows_return/random/TSLA/seed_51/last_model_comparison_20260828_093624.csv`
- `results/top10_market_windows_return/random/WMT/seed_42/last_model_comparison_20260828_094548.csv`
- `results/top10_market_windows_return/random/WMT/seed_43/last_model_comparison_20260828_094639.csv`
- `results/top10_market_windows_return/random/WMT/seed_44/last_model_comparison_20260828_094730.csv`
- `results/top10_market_windows_return/random/WMT/seed_45/last_model_comparison_20260828_094821.csv`
- `results/top10_market_windows_return/random/WMT/seed_46/last_model_comparison_20260828_094912.csv`
- `results/top10_market_windows_return/random/WMT/seed_47/last_model_comparison_20260828_095003.csv`
- `results/top10_market_windows_return/random/WMT/seed_48/last_model_comparison_20260828_095054.csv`
- `results/top10_market_windows_return/random/WMT/seed_49/last_model_comparison_20260828_095145.csv`
- `results/top10_market_windows_return/random/WMT/seed_50/last_model_comparison_20260828_095236.csv`
- `results/top10_market_windows_return/random/WMT/seed_51/last_model_comparison_20260828_095327.csv`

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

The separate Shared-vs-Local comparison first retains compatible seeds matched by equity, seed, target definition, normalization, metric definition, forecast horizon, test period, and saved target signature. Shared and Local MSE/MAE are averaged over the same matched seeds within each equity. A two-sided paired Student t-test and signed Cohen's dz then use the resulting equity-level values only, with Δ = Shared − Local; negative values favour Shared for these error metrics. No Direction Accuracy test or multiple-comparison correction is applied to this separate comparison.

## Paired results snapshot

| model | mean_delta_mse | mse_ci_low | mse_ci_high | mse_holm_p_value | mse_stock_wins | mse_run_wins | mse_run_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Shared-target JEPA--MAE | 0.0004695 | 0.000228 | 0.0007483 | 0.005859 | 0 | 3 | 100 |
| Local-MAE/Long-JEPA | 0.0004772 | 0.0002449 | 0.0007724 | 0.005859 | 0 | 4 | 100 |
| GRU | 0.0004012 | 9.956e-05 | 0.0008053 | 0.01367 | 1 | 16 | 100 |

## Shared-target vs Local-MAE/Long-JEPA snapshot

| metric | n_stocks | mean_delta | median_delta | t_statistic | p_value | cohens_dz | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MSE | 10 | -7.676e-06 | -2.092e-05 | -0.14 | 0.8917 | -0.04428 | ok |
| MAE | 10 | -0.0002382 | -0.0001723 | -0.6233 | 0.5485 | -0.1971 | ok |

## Representative trajectory

- Stock: NVDA
- Seed: 48
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
| data/paired_shared_vs_local.csv | generated | Canonical analysis dataset: paired_shared_vs_local.csv |
| data/relative_performance_by_stock.csv | generated | Canonical analysis dataset: relative_performance_by_stock.csv |
| data/horizon_metrics.csv | generated | Canonical analysis dataset: horizon_metrics.csv |
| tables/table_main_metrics.csv | generated | Thesis or appendix table |
| tables/table_main_metrics.tex | generated | Thesis or appendix table |
| tables/table_paired_vs_naive.csv | generated | Thesis or appendix table |
| tables/table_paired_vs_naive.tex | generated | Thesis or appendix table |
| tables/table_shared_vs_local.csv | generated | Thesis or appendix table |
| tables/table_shared_vs_local.tex | generated | Thesis or appendix table |
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
  --config config/experiments/top10_market_windows_return.json \
  --reference-strategy random \
  --bootstrap-samples 20000 \
  --analysis-seed 20260822
```

The analysis bootstrap seed and sample count are recorded in the command and output metadata. PDF figures are vector outputs; matching PNG files are previews.

## Git publication

- Immutable snapshot: `f2efe84ebe60-55937e54f545`
- Full raw experiment outputs are intentionally excluded from Git.
- `SHA256SUMS` verifies every published file.
- Files omitted by publication policy: 1. See `publication_manifest.csv`.
- Large/raw artifacts should be attached to the matching GitHub Release.
