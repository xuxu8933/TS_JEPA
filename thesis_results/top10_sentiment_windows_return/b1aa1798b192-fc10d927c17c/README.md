# TS-JEPA thesis result manifest

This directory is generated exclusively from saved experiment artifacts; no model training is performed.

## Analysis scope and coverage

- Config: `config/experiments/top10_sentiment_windows_return.json`
- Result root: `results/top10_sentiment_windows_return`
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

- `results/top10_sentiment_windows_return/local_long/AAPL/seed_42/last_model_comparison_20260827_092731.csv`
- `results/top10_sentiment_windows_return/local_long/AAPL/seed_43/last_model_comparison_20260827_093201.csv`
- `results/top10_sentiment_windows_return/local_long/AAPL/seed_44/last_model_comparison_20260827_093633.csv`
- `results/top10_sentiment_windows_return/local_long/AAPL/seed_45/last_model_comparison_20260827_094105.csv`
- `results/top10_sentiment_windows_return/local_long/AAPL/seed_46/last_model_comparison_20260827_094537.csv`
- `results/top10_sentiment_windows_return/local_long/AAPL/seed_47/last_model_comparison_20260827_173739.csv`
- `results/top10_sentiment_windows_return/local_long/AAPL/seed_48/last_model_comparison_20260827_174220.csv`
- `results/top10_sentiment_windows_return/local_long/AAPL/seed_49/last_model_comparison_20260827_201256.csv`
- `results/top10_sentiment_windows_return/local_long/AAPL/seed_50/last_model_comparison_20260828_075709.csv`
- `results/top10_sentiment_windows_return/local_long/AAPL/seed_51/last_model_comparison_20260828_100230.csv`
- `results/top10_sentiment_windows_return/local_long/AMZN/seed_42/last_model_comparison_20260827_101253.csv`
- `results/top10_sentiment_windows_return/local_long/AMZN/seed_43/last_model_comparison_20260827_101726.csv`
- `results/top10_sentiment_windows_return/local_long/AMZN/seed_44/last_model_comparison_20260827_102157.csv`
- `results/top10_sentiment_windows_return/local_long/AMZN/seed_45/last_model_comparison_20260827_102630.csv`
- `results/top10_sentiment_windows_return/local_long/AMZN/seed_46/last_model_comparison_20260827_103100.csv`
- `results/top10_sentiment_windows_return/local_long/AMZN/seed_47/last_model_comparison_20260827_175603.csv`
- `results/top10_sentiment_windows_return/local_long/AMZN/seed_48/last_model_comparison_20260827_180037.csv`
- `results/top10_sentiment_windows_return/local_long/AMZN/seed_49/last_model_comparison_20260827_202207.csv`
- `results/top10_sentiment_windows_return/local_long/AMZN/seed_50/last_model_comparison_20260828_080629.csv`
- `results/top10_sentiment_windows_return/local_long/AMZN/seed_51/last_model_comparison_20260828_101129.csv`
- `results/top10_sentiment_windows_return/local_long/AVGO/seed_42/last_model_comparison_20260827_181416.csv`
- `results/top10_sentiment_windows_return/local_long/AVGO/seed_43/last_model_comparison_20260827_181848.csv`
- `results/top10_sentiment_windows_return/local_long/AVGO/seed_44/last_model_comparison_20260827_182320.csv`
- `results/top10_sentiment_windows_return/local_long/AVGO/seed_45/last_model_comparison_20260827_182752.csv`
- `results/top10_sentiment_windows_return/local_long/AVGO/seed_46/last_model_comparison_20260827_183223.csv`
- `results/top10_sentiment_windows_return/local_long/AVGO/seed_47/last_model_comparison_20260827_183654.csv`
- `results/top10_sentiment_windows_return/local_long/AVGO/seed_48/last_model_comparison_20260827_184128.csv`
- `results/top10_sentiment_windows_return/local_long/AVGO/seed_49/last_model_comparison_20260827_203127.csv`
- `results/top10_sentiment_windows_return/local_long/AVGO/seed_50/last_model_comparison_20260828_081556.csv`
- `results/top10_sentiment_windows_return/local_long/AVGO/seed_51/last_model_comparison_20260828_102030.csv`
- `results/top10_sentiment_windows_return/local_long/COST/seed_42/last_model_comparison_20260828_103359.csv`
- `results/top10_sentiment_windows_return/local_long/COST/seed_43/last_model_comparison_20260828_103829.csv`
- `results/top10_sentiment_windows_return/local_long/COST/seed_44/last_model_comparison_20260828_104300.csv`
- `results/top10_sentiment_windows_return/local_long/COST/seed_45/last_model_comparison_20260828_104730.csv`
- `results/top10_sentiment_windows_return/local_long/COST/seed_46/last_model_comparison_20260828_105159.csv`
- `results/top10_sentiment_windows_return/local_long/COST/seed_47/last_model_comparison_20260828_105630.csv`
- `results/top10_sentiment_windows_return/local_long/COST/seed_48/last_model_comparison_20260828_110100.csv`
- `results/top10_sentiment_windows_return/local_long/COST/seed_49/last_model_comparison_20260828_110530.csv`
- `results/top10_sentiment_windows_return/local_long/COST/seed_50/last_model_comparison_20260828_110959.csv`
- `results/top10_sentiment_windows_return/local_long/COST/seed_51/last_model_comparison_20260828_111428.csv`
- `results/top10_sentiment_windows_return/local_long/GOOGL/seed_42/last_model_comparison_20260827_103533.csv`
- `results/top10_sentiment_windows_return/local_long/GOOGL/seed_43/last_model_comparison_20260827_104004.csv`
- `results/top10_sentiment_windows_return/local_long/GOOGL/seed_44/last_model_comparison_20260827_104438.csv`
- `results/top10_sentiment_windows_return/local_long/GOOGL/seed_45/last_model_comparison_20260827_104909.csv`
- `results/top10_sentiment_windows_return/local_long/GOOGL/seed_46/last_model_comparison_20260827_105342.csv`
- `results/top10_sentiment_windows_return/local_long/GOOGL/seed_47/last_model_comparison_20260827_180509.csv`
- `results/top10_sentiment_windows_return/local_long/GOOGL/seed_48/last_model_comparison_20260827_180943.csv`
- `results/top10_sentiment_windows_return/local_long/GOOGL/seed_49/last_model_comparison_20260827_202646.csv`
- `results/top10_sentiment_windows_return/local_long/GOOGL/seed_50/last_model_comparison_20260828_081120.csv`
- `results/top10_sentiment_windows_return/local_long/GOOGL/seed_51/last_model_comparison_20260828_101558.csv`
- `results/top10_sentiment_windows_return/local_long/META/seed_42/last_model_comparison_20260827_184601.csv`
- `results/top10_sentiment_windows_return/local_long/META/seed_43/last_model_comparison_20260827_185033.csv`
- `results/top10_sentiment_windows_return/local_long/META/seed_44/last_model_comparison_20260827_185512.csv`
- `results/top10_sentiment_windows_return/local_long/META/seed_45/last_model_comparison_20260827_185945.csv`
- `results/top10_sentiment_windows_return/local_long/META/seed_46/last_model_comparison_20260827_190430.csv`
- `results/top10_sentiment_windows_return/local_long/META/seed_47/last_model_comparison_20260827_190908.csv`
- `results/top10_sentiment_windows_return/local_long/META/seed_48/last_model_comparison_20260827_191348.csv`
- `results/top10_sentiment_windows_return/local_long/META/seed_49/last_model_comparison_20260827_203559.csv`
- `results/top10_sentiment_windows_return/local_long/META/seed_50/last_model_comparison_20260828_082025.csv`
- `results/top10_sentiment_windows_return/local_long/META/seed_51/last_model_comparison_20260828_102458.csv`
- `results/top10_sentiment_windows_return/local_long/MSFT/seed_42/last_model_comparison_20260827_095009.csv`
- `results/top10_sentiment_windows_return/local_long/MSFT/seed_43/last_model_comparison_20260827_095444.csv`
- `results/top10_sentiment_windows_return/local_long/MSFT/seed_44/last_model_comparison_20260827_095915.csv`
- `results/top10_sentiment_windows_return/local_long/MSFT/seed_45/last_model_comparison_20260827_100349.csv`
- `results/top10_sentiment_windows_return/local_long/MSFT/seed_46/last_model_comparison_20260827_100820.csv`
- `results/top10_sentiment_windows_return/local_long/MSFT/seed_47/last_model_comparison_20260827_174656.csv`
- `results/top10_sentiment_windows_return/local_long/MSFT/seed_48/last_model_comparison_20260827_175127.csv`
- `results/top10_sentiment_windows_return/local_long/MSFT/seed_49/last_model_comparison_20260827_201729.csv`
- `results/top10_sentiment_windows_return/local_long/MSFT/seed_50/last_model_comparison_20260828_080146.csv`
- `results/top10_sentiment_windows_return/local_long/MSFT/seed_51/last_model_comparison_20260828_100659.csv`
- `results/top10_sentiment_windows_return/local_long/NVDA/seed_42/last_model_comparison_20260827_090449.csv`
- `results/top10_sentiment_windows_return/local_long/NVDA/seed_43/last_model_comparison_20260827_090922.csv`
- `results/top10_sentiment_windows_return/local_long/NVDA/seed_44/last_model_comparison_20260827_091354.csv`
- `results/top10_sentiment_windows_return/local_long/NVDA/seed_45/last_model_comparison_20260827_091826.csv`
- `results/top10_sentiment_windows_return/local_long/NVDA/seed_46/last_model_comparison_20260827_092259.csv`
- `results/top10_sentiment_windows_return/local_long/NVDA/seed_47/last_model_comparison_20260827_172832.csv`
- `results/top10_sentiment_windows_return/local_long/NVDA/seed_48/last_model_comparison_20260827_173307.csv`
- `results/top10_sentiment_windows_return/local_long/NVDA/seed_49/last_model_comparison_20260827_200821.csv`
- `results/top10_sentiment_windows_return/local_long/NVDA/seed_50/last_model_comparison_20260828_075234.csv`
- `results/top10_sentiment_windows_return/local_long/NVDA/seed_51/last_model_comparison_20260828_095759.csv`
- `results/top10_sentiment_windows_return/local_long/TSLA/seed_42/last_model_comparison_20260827_204037.csv`
- `results/top10_sentiment_windows_return/local_long/TSLA/seed_43/last_model_comparison_20260827_204516.csv`
- `results/top10_sentiment_windows_return/local_long/TSLA/seed_44/last_model_comparison_20260827_204949.csv`
- `results/top10_sentiment_windows_return/local_long/TSLA/seed_45/last_model_comparison_20260827_205421.csv`
- `results/top10_sentiment_windows_return/local_long/TSLA/seed_46/last_model_comparison_20260827_205857.csv`
- `results/top10_sentiment_windows_return/local_long/TSLA/seed_47/last_model_comparison_20260827_210340.csv`
- `results/top10_sentiment_windows_return/local_long/TSLA/seed_48/last_model_comparison_20260827_210821.csv`
- `results/top10_sentiment_windows_return/local_long/TSLA/seed_49/last_model_comparison_20260827_211259.csv`
- `results/top10_sentiment_windows_return/local_long/TSLA/seed_50/last_model_comparison_20260828_082459.csv`
- `results/top10_sentiment_windows_return/local_long/TSLA/seed_51/last_model_comparison_20260828_102928.csv`
- `results/top10_sentiment_windows_return/local_long/WMT/seed_42/last_model_comparison_20260828_111858.csv`
- `results/top10_sentiment_windows_return/local_long/WMT/seed_43/last_model_comparison_20260828_112326.csv`
- `results/top10_sentiment_windows_return/local_long/WMT/seed_44/last_model_comparison_20260828_112757.csv`
- `results/top10_sentiment_windows_return/local_long/WMT/seed_45/last_model_comparison_20260828_113228.csv`
- `results/top10_sentiment_windows_return/local_long/WMT/seed_46/last_model_comparison_20260828_113657.csv`
- `results/top10_sentiment_windows_return/local_long/WMT/seed_47/last_model_comparison_20260828_114128.csv`
- `results/top10_sentiment_windows_return/local_long/WMT/seed_48/last_model_comparison_20260828_114556.csv`
- `results/top10_sentiment_windows_return/local_long/WMT/seed_49/last_model_comparison_20260828_115027.csv`
- `results/top10_sentiment_windows_return/local_long/WMT/seed_50/last_model_comparison_20260828_115457.csv`
- `results/top10_sentiment_windows_return/local_long/WMT/seed_51/last_model_comparison_20260828_115928.csv`
- `results/top10_sentiment_windows_return/random/AAPL/seed_42/last_model_comparison_20260827_092733.csv`
- `results/top10_sentiment_windows_return/random/AAPL/seed_43/last_model_comparison_20260827_093205.csv`
- `results/top10_sentiment_windows_return/random/AAPL/seed_44/last_model_comparison_20260827_093635.csv`
- `results/top10_sentiment_windows_return/random/AAPL/seed_45/last_model_comparison_20260827_094109.csv`
- `results/top10_sentiment_windows_return/random/AAPL/seed_46/last_model_comparison_20260827_094540.csv`
- `results/top10_sentiment_windows_return/random/AAPL/seed_47/last_model_comparison_20260827_173744.csv`
- `results/top10_sentiment_windows_return/random/AAPL/seed_48/last_model_comparison_20260827_174225.csv`
- `results/top10_sentiment_windows_return/random/AAPL/seed_49/last_model_comparison_20260827_201258.csv`
- `results/top10_sentiment_windows_return/random/AAPL/seed_50/last_model_comparison_20260828_075714.csv`
- `results/top10_sentiment_windows_return/random/AAPL/seed_51/last_model_comparison_20260828_100232.csv`
- `results/top10_sentiment_windows_return/random/AMZN/seed_42/last_model_comparison_20260827_101256.csv`
- `results/top10_sentiment_windows_return/random/AMZN/seed_43/last_model_comparison_20260827_101728.csv`
- `results/top10_sentiment_windows_return/random/AMZN/seed_44/last_model_comparison_20260827_102201.csv`
- `results/top10_sentiment_windows_return/random/AMZN/seed_45/last_model_comparison_20260827_102632.csv`
- `results/top10_sentiment_windows_return/random/AMZN/seed_46/last_model_comparison_20260827_103104.csv`
- `results/top10_sentiment_windows_return/random/AMZN/seed_47/last_model_comparison_20260827_175607.csv`
- `results/top10_sentiment_windows_return/random/AMZN/seed_48/last_model_comparison_20260827_180039.csv`
- `results/top10_sentiment_windows_return/random/AMZN/seed_49/last_model_comparison_20260827_202205.csv`
- `results/top10_sentiment_windows_return/random/AMZN/seed_50/last_model_comparison_20260828_080642.csv`
- `results/top10_sentiment_windows_return/random/AMZN/seed_51/last_model_comparison_20260828_101131.csv`
- `results/top10_sentiment_windows_return/random/AVGO/seed_42/last_model_comparison_20260827_181420.csv`
- `results/top10_sentiment_windows_return/random/AVGO/seed_43/last_model_comparison_20260827_181851.csv`
- `results/top10_sentiment_windows_return/random/AVGO/seed_44/last_model_comparison_20260827_182325.csv`
- `results/top10_sentiment_windows_return/random/AVGO/seed_45/last_model_comparison_20260827_182754.csv`
- `results/top10_sentiment_windows_return/random/AVGO/seed_46/last_model_comparison_20260827_183226.csv`
- `results/top10_sentiment_windows_return/random/AVGO/seed_47/last_model_comparison_20260827_183659.csv`
- `results/top10_sentiment_windows_return/random/AVGO/seed_48/last_model_comparison_20260827_184131.csv`
- `results/top10_sentiment_windows_return/random/AVGO/seed_49/last_model_comparison_20260827_203127.csv`
- `results/top10_sentiment_windows_return/random/AVGO/seed_50/last_model_comparison_20260828_081555.csv`
- `results/top10_sentiment_windows_return/random/AVGO/seed_51/last_model_comparison_20260828_102030.csv`
- `results/top10_sentiment_windows_return/random/COST/seed_42/last_model_comparison_20260828_103402.csv`
- `results/top10_sentiment_windows_return/random/COST/seed_43/last_model_comparison_20260828_103833.csv`
- `results/top10_sentiment_windows_return/random/COST/seed_44/last_model_comparison_20260828_104302.csv`
- `results/top10_sentiment_windows_return/random/COST/seed_45/last_model_comparison_20260828_104733.csv`
- `results/top10_sentiment_windows_return/random/COST/seed_46/last_model_comparison_20260828_105203.csv`
- `results/top10_sentiment_windows_return/random/COST/seed_47/last_model_comparison_20260828_105632.csv`
- `results/top10_sentiment_windows_return/random/COST/seed_48/last_model_comparison_20260828_110103.csv`
- `results/top10_sentiment_windows_return/random/COST/seed_49/last_model_comparison_20260828_110533.csv`
- `results/top10_sentiment_windows_return/random/COST/seed_50/last_model_comparison_20260828_111002.csv`
- `results/top10_sentiment_windows_return/random/COST/seed_51/last_model_comparison_20260828_111432.csv`
- `results/top10_sentiment_windows_return/random/GOOGL/seed_42/last_model_comparison_20260827_103535.csv`
- `results/top10_sentiment_windows_return/random/GOOGL/seed_43/last_model_comparison_20260827_104006.csv`
- `results/top10_sentiment_windows_return/random/GOOGL/seed_44/last_model_comparison_20260827_104439.csv`
- `results/top10_sentiment_windows_return/random/GOOGL/seed_45/last_model_comparison_20260827_104913.csv`
- `results/top10_sentiment_windows_return/random/GOOGL/seed_46/last_model_comparison_20260827_105344.csv`
- `results/top10_sentiment_windows_return/random/GOOGL/seed_47/last_model_comparison_20260827_180514.csv`
- `results/top10_sentiment_windows_return/random/GOOGL/seed_48/last_model_comparison_20260827_180945.csv`
- `results/top10_sentiment_windows_return/random/GOOGL/seed_49/last_model_comparison_20260827_202653.csv`
- `results/top10_sentiment_windows_return/random/GOOGL/seed_50/last_model_comparison_20260828_081127.csv`
- `results/top10_sentiment_windows_return/random/GOOGL/seed_51/last_model_comparison_20260828_101602.csv`
- `results/top10_sentiment_windows_return/random/META/seed_42/last_model_comparison_20260827_184603.csv`
- `results/top10_sentiment_windows_return/random/META/seed_43/last_model_comparison_20260827_185037.csv`
- `results/top10_sentiment_windows_return/random/META/seed_44/last_model_comparison_20260827_185517.csv`
- `results/top10_sentiment_windows_return/random/META/seed_45/last_model_comparison_20260827_185947.csv`
- `results/top10_sentiment_windows_return/random/META/seed_46/last_model_comparison_20260827_190437.csv`
- `results/top10_sentiment_windows_return/random/META/seed_47/last_model_comparison_20260827_190910.csv`
- `results/top10_sentiment_windows_return/random/META/seed_48/last_model_comparison_20260827_191354.csv`
- `results/top10_sentiment_windows_return/random/META/seed_49/last_model_comparison_20260827_203606.csv`
- `results/top10_sentiment_windows_return/random/META/seed_50/last_model_comparison_20260828_082030.csv`
- `results/top10_sentiment_windows_return/random/META/seed_51/last_model_comparison_20260828_102501.csv`
- `results/top10_sentiment_windows_return/random/MSFT/seed_42/last_model_comparison_20260827_095014.csv`
- `results/top10_sentiment_windows_return/random/MSFT/seed_43/last_model_comparison_20260827_095445.csv`
- `results/top10_sentiment_windows_return/random/MSFT/seed_44/last_model_comparison_20260827_095920.csv`
- `results/top10_sentiment_windows_return/random/MSFT/seed_45/last_model_comparison_20260827_100349.csv`
- `results/top10_sentiment_windows_return/random/MSFT/seed_46/last_model_comparison_20260827_100824.csv`
- `results/top10_sentiment_windows_return/random/MSFT/seed_47/last_model_comparison_20260827_174659.csv`
- `results/top10_sentiment_windows_return/random/MSFT/seed_48/last_model_comparison_20260827_175130.csv`
- `results/top10_sentiment_windows_return/random/MSFT/seed_49/last_model_comparison_20260827_201738.csv`
- `results/top10_sentiment_windows_return/random/MSFT/seed_50/last_model_comparison_20260828_080150.csv`
- `results/top10_sentiment_windows_return/random/MSFT/seed_51/last_model_comparison_20260828_100702.csv`
- `results/top10_sentiment_windows_return/random/NVDA/seed_42/last_model_comparison_20260827_090454.csv`
- `results/top10_sentiment_windows_return/random/NVDA/seed_43/last_model_comparison_20260827_090924.csv`
- `results/top10_sentiment_windows_return/random/NVDA/seed_44/last_model_comparison_20260827_091357.csv`
- `results/top10_sentiment_windows_return/random/NVDA/seed_45/last_model_comparison_20260827_091828.csv`
- `results/top10_sentiment_windows_return/random/NVDA/seed_46/last_model_comparison_20260827_092301.csv`
- `results/top10_sentiment_windows_return/random/NVDA/seed_47/last_model_comparison_20260827_172837.csv`
- `results/top10_sentiment_windows_return/random/NVDA/seed_48/last_model_comparison_20260827_173309.csv`
- `results/top10_sentiment_windows_return/random/NVDA/seed_49/last_model_comparison_20260827_200826.csv`
- `results/top10_sentiment_windows_return/random/NVDA/seed_50/last_model_comparison_20260828_075238.csv`
- `results/top10_sentiment_windows_return/random/NVDA/seed_51/last_model_comparison_20260828_095804.csv`
- `results/top10_sentiment_windows_return/random/TSLA/seed_42/last_model_comparison_20260827_204041.csv`
- `results/top10_sentiment_windows_return/random/TSLA/seed_43/last_model_comparison_20260827_204520.csv`
- `results/top10_sentiment_windows_return/random/TSLA/seed_44/last_model_comparison_20260827_204952.csv`
- `results/top10_sentiment_windows_return/random/TSLA/seed_45/last_model_comparison_20260827_205425.csv`
- `results/top10_sentiment_windows_return/random/TSLA/seed_46/last_model_comparison_20260827_205900.csv`
- `results/top10_sentiment_windows_return/random/TSLA/seed_47/last_model_comparison_20260827_210348.csv`
- `results/top10_sentiment_windows_return/random/TSLA/seed_48/last_model_comparison_20260827_210823.csv`
- `results/top10_sentiment_windows_return/random/TSLA/seed_49/last_model_comparison_20260827_211307.csv`
- `results/top10_sentiment_windows_return/random/TSLA/seed_50/last_model_comparison_20260828_082500.csv`
- `results/top10_sentiment_windows_return/random/TSLA/seed_51/last_model_comparison_20260828_102931.csv`
- `results/top10_sentiment_windows_return/random/WMT/seed_42/last_model_comparison_20260828_111900.csv`
- `results/top10_sentiment_windows_return/random/WMT/seed_43/last_model_comparison_20260828_112330.csv`
- `results/top10_sentiment_windows_return/random/WMT/seed_44/last_model_comparison_20260828_112801.csv`
- `results/top10_sentiment_windows_return/random/WMT/seed_45/last_model_comparison_20260828_113230.csv`
- `results/top10_sentiment_windows_return/random/WMT/seed_46/last_model_comparison_20260828_113700.csv`
- `results/top10_sentiment_windows_return/random/WMT/seed_47/last_model_comparison_20260828_114129.csv`
- `results/top10_sentiment_windows_return/random/WMT/seed_48/last_model_comparison_20260828_114601.csv`
- `results/top10_sentiment_windows_return/random/WMT/seed_49/last_model_comparison_20260828_115031.csv`
- `results/top10_sentiment_windows_return/random/WMT/seed_50/last_model_comparison_20260828_115501.csv`
- `results/top10_sentiment_windows_return/random/WMT/seed_51/last_model_comparison_20260828_115930.csv`

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
| Shared-target JEPA--MAE | 0.00117 | 0.000628 | 0.001813 | 0.005859 | 0 | 0 | 100 |
| Local-MAE/Long-JEPA | 0.0009428 | 0.0005713 | 0.001369 | 0.005859 | 0 | 0 | 100 |
| GRU | 0.0001137 | 4.108e-05 | 0.0001999 | 0.01953 | 2 | 27 | 100 |

## Shared-target vs Local-MAE/Long-JEPA snapshot

| metric | n_stocks | mean_delta | median_delta | t_statistic | p_value | cohens_dz | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MSE | 10 | 0.0002268 | 3.246e-05 | 1.204 | 0.2594 | 0.3806 | ok |
| MAE | 10 | 0.00146 | 0.0005463 | 1.426 | 0.1875 | 0.4511 | ok |

## Representative trajectory

- Stock: NVDA
- Seed: 43
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
  --config config/experiments/top10_sentiment_windows_return.json \
  --reference-strategy random \
  --bootstrap-samples 20000 \
  --analysis-seed 20260822
```

The analysis bootstrap seed and sample count are recorded in the command and output metadata. PDF figures are vector outputs; matching PNG files are previews.

## Git publication

- Immutable snapshot: `b1aa1798b192-fc10d927c17c`
- Full raw experiment outputs are intentionally excluded from Git.
- `SHA256SUMS` verifies every published file.
- Files omitted by publication policy: 1. See `publication_manifest.csv`.
- Large/raw artifacts should be attached to the matching GitHub Release.
