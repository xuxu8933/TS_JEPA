# Dual-loss smoke-test results

> **DIAGNOSTIC ONLY:** These small MNIST and synthetic sin/cos runs verify that
> JEPA + MAE pretraining, checkpoint loading, downstream fitting, RMSE
> evaluation, and plotting execute successfully. They are not Chapter 5 model
> selection or held-out financial test evidence.

| Case | Model RMSE | Naive baseline RMSE |
|---|---:|---:|
| MNIST rows | 0.111085 | 0.188641 |
| Sin/cos | 0.0102686705 | 0.0895966440 |

The MNIST baseline copies the previous image row. The sin/cos baseline repeats
the final observed value. Lower RMSE is better.

Generated with:

```bash
python visualization/plot_dual_loss_smoke_results.py
```

Snapshot `e56db7d33c56` is the first 12 characters of the SHA-256 digest over
the four source artifacts listed in `publication_manifest.csv`.
