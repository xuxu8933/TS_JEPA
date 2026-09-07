# Stage 00 controlled comparison

Implemented and measured on 2026-09-07. This is a diagnostic experiment on
MNIST reconstruction and synthetic forecasting, not evidence of financial
forecasting performance. Stage 01–05 code and published snapshots are unchanged.

## What changed and why

- Added `run_stage00_comparison.py`, shared context-only baselines in
  `src/stage00_baselines.py`, and shared models/evaluation/aggregation in
  `src/stage00_comparison.py`.
- Moved the original Stage 00 data generators, constants, command builders and
  diagnostic helpers from `tests/test_dual_loss_smoke.py` into
  `src/stage00_legacy.py`. The test module re-exports their existing names;
  legacy callers and command defaults remain compatible. Optional CLI overrides
  reuse exactly those pretraining commands for the ablations.
- Exposed MNIST sample indices in `data_loader_mnist_rows.py` for split auditing;
  sampling, images, preprocessing, masks and splits are unchanged.
- Added focused checks in `tests/test_stage00_comparison.py`.

The full pretrained model keeps its existing architecture and hyperparameters.
The necessary comparison-protocol changes below are explicit, separate from
legacy smoke results, and applied equally to the three pretrained variants.

## Methodological issues found

1. **MNIST is masked-row reconstruction, not next-row forecasting.** Its 11
   random targets are interspersed among 17 visible rows. The user approved
   preserving this task and adding the same supervised readout protocol to all
   pretrained variants. Right-hand visible context is legitimate for this
   reconstruction task; it must not be described as causal forecasting.
2. **The old MNIST previous-row baseline reads hidden targets.** For seed 7,
   570 of 1,408 copied rows (40.48%) are themselves masked. Row zero can even
   copy its own target. Its reproduced RMSE, 0.1886414221, therefore cannot be
   renamed and presented as a valid context-only baseline. The comparison has
   one baseline named Naive-last, corrected as defined below. Its score is
   0.2462295294. The old score is retained only in the legacy diagnostic audit.
3. **Legacy sine-cosine SSL and downstream splits disagree.** SSL uses a
   September 7 cutoff, whereas downstream uses July 1. This permits SSL to
   train on downstream validation/test observations and invalidates treating
   the old diagnostic score as out-of-sample evidence. The comparison overrides
   only the SSL cutoff to July 1; downstream data construction and splits stay
   unchanged. The runner checks the actual SSL training tensor equals the
   supervised training split.
4. **Legacy MNIST has no downstream readout fitting.** It directly evaluates
   the pretrained reconstruction decoder; that decoder is untrained under
   JEPA-only. Comparing it directly would not measure representation quality.
   All three variants now receive a newly initialized identical pixel readout,
   trained on training images only, with encoder and predictor frozen.
5. **Legacy sine-cosine evaluates the final downstream epoch.** The new
   comparison restores the best validation RMSE checkpoint for every learned
   method. Its test loader is evaluated only after fitting and selection.
6. The preserved synthetic preprocessing removes 49 MA50 warmup rows even
   though the selected features are Close and Volume. This pre-existing
   behavior is retained. The resulting 13 validation observations cannot form
   a 20-observation SSL window. SSL therefore retains its fixed one-step final
   checkpoint; its validation loss is null, not invented. Downstream validation
   still has 10 valid four-step forecast windows, using preceding training
   observations as historical context.

## Exact tasks and splits

**MNIST:** official training pool supplies 4,000 training and 128 disjoint
validation images from one seeded permutation (offsets 0 and 4000). Official
test pool supplies 128 images. Pixels remain divided by 255. Each image is
28 tokens of 28 pixels, with `int(28 * 0.4) = 11` masked rows. Training masks
remain stochastic. Validation and test masks are deterministic. Every method
uses the same test images, masks, visible rows and targets within a seed.

**Sine-cosine:** unchanged `_sin_cos_rows(260)` generates six-decimal Close and
Volume observations dated January 1–September 17, 2021. Existing feature
preprocessing (including `log1p(Volume)`) and window-return normalization are
reused. After warmup:

| Split | Observation dates | Rows | Forecast windows |
|---|---|---:|---:|
| Train | 2021-02-19–2021-06-17 | 119 | 96 |
| Validation | 2021-06-18–2021-06-30 | 13 | 10 |
| Test | 2021-07-01–2021-09-17 | 79 | 76 |

Input is 20 observations of Close and Volume, patched to `[B,5,8]` using
patch length 4. Output is the next four Close values, `[B,4]`. Close context
and target both use `Close / first_context_Close - 1`. Validation/test windows
may use earlier observations as history; every target stays inside its own
split. Training and selection never update on test windows. Overlapping test
forecast windows remain the original evaluation samples, not independent draws.

## Exact model definitions

All SSL variants use the existing `Encoder`, `Predictor`, MAE decoder, random
masks, initialization, AdamW, linear learning-rate schedule, EMA update and
RMSE objectives from `pretrain_dual_loss.py`. The teacher receives no gradients.
Architecture is kept identical across objective ablations, including the unused
decoder/teacher. Objective weights are the only ablation changes:

| Dataset | Method | JEPA weight | MAE weight |
|---|---|---:|---:|
| MNIST | JEPA-MAE | 0.01 | 1.0 |
| Sine-cosine | JEPA-MAE | 0.7 | 0.3 |
| Both | JEPA-only | 1.0 | 0.0 |
| Both | MAE-only | 0.0 | 1.0 |

- **MNIST SSL:** encoder and predictor width 64, four attention heads, two
  Transformer blocks each; tokenizer convolution kernel/stride 3 within each
  row. Existing reconstruction MLP has two hidden blocks of width 128,
  LayerNorm/GELU, no dropout, output 28. Batch 64; 30 epochs; learning rate
  0.002 linearly to 0.001; EMA starts at 0.9; gradient clipping 1.0;
  EMA schedule scale 1.25. The final epoch-29 checkpoint is used for all three
  variants, matching the legacy fixed-budget selection.
- **MNIST readout:** freeze the online encoder and predictor; discard the
  pretrained pixel decoder's weights and initialize a fresh decoder of the
  same MLP architecture for every variant. Train only this decoder for 30
  epochs, batch 64, AdamW at constant 0.002, minimizing masked-pixel RMSE.
  Select the epoch with the lowest pooled validation RMSE.
- **Sine-cosine SSL:** encoder width 16, two heads, one block; predictor width
  8, two heads, one block; tokenizer kernel/stride 3; linear reconstruction
  decoder to eight patch values. Batch 2; one epoch capped at one batch;
  learning rate 0.001 to 0.001; EMA starts at 0.9; clipping 1.0; EMA schedule
  scale 1.25. Use the fixed final epoch-0 checkpoint.
- **Sine-cosine downstream:** load the online encoder, mean-pool its five
  embeddings, and fine-tune it with the existing fresh `MLPDecoder`: two
  hidden blocks of width 64 with LayerNorm/GELU, no dropout, four outputs.
  AdamW at 0.003, 120 epochs, batch 16, training RMSE; select best validation
  RMSE. This is the original training budget and architecture.
- **GRU:** reuse `GRUForecastModel` from the existing financial evaluator,
  with one recurrent layer, hidden size 64, dropout zero, and a linear output
  head. For sine-cosine its two input features cover exactly the same 20
  observations and its head predicts four values. For MNIST its input is the
  original 28-row grid with masked pixels replaced by zero, plus one visibility
  indicator per row (29 features). The final hidden state predicts 784 pixels,
  reshaped to 28×28; loss/evaluation gather only the same 11 masked rows.
  Row positions and the visibility indicator supply mask geometry, not extra
  observations. GRU uses the same downstream budget, batch size, constant
  learning rate, training loss and validation selection as its dataset's
  pretrained models. All GRU parameters are supervised; there is no SSL.

All downstream optimizers retain AdamW's existing defaults (`weight_decay=0.01`,
betas 0.9/0.999, epsilon 1e-8). No architecture or learning-rate search is run.

**Deterministic baselines:** sine-cosine Naive-last repeats the last context
Close; Mean-context repeats the arithmetic mean of its 20 context Closes;
Drift predicts `last + h * (last-first)/19`, for h=1,…,4, in the same normalized
coordinates. A single-observation context has zero drift.

For MNIST, Naive-last copies the last *visible* row preceding each target row.
Drift uses the first and last visible preceding rows, dividing by their actual
row-index distance and extrapolating to the target index, element-wise. With
only one preceding visible row its slope is zero. With no preceding visible
row both use the earliest visible row, a documented reconstruction boundary
fallback. Mean-context averages all 17 visible rows. No baseline reads a
masked pixel, clips predictions or fits any parameters. These are explicitly
defined reconstruction adaptations, not a claimed causal forecast horizon.

## Seeds, metrics and artifacts

Default experiment seed remains 7. The runner accepts `--seed 7` or
`--seeds 7 8 ...`. MNIST test selection/masks use `seed+10000`; downstream
training uses `seed+116`, preserving the original sine-cosine seed 123 for
seed 7. Python, NumPy, torch and CUDA seeds are set. The existing deterministic
kernel request is retained. Validation iterator creation is isolated from the
training torch RNG. The measured comparison explicitly uses one CPU thread;
MNIST training uses CUDA when available, sine downstream uses CPU as before.

Evaluation concatenates all test predictions/targets and computes
`sqrt(mean((prediction-target)**2))`, exactly the sine-cosine smoke expression
and mathematically the same pooled SSE/count definition as MNIST. It is not
the mean of batch RMSEs. Training reuses `main.utils.rmse_loss`. MNIST RMSE is
in pixel-intensity units; sine RMSE is in window-relative Close units.

`raw_results.csv` and `.json` contain each dataset/model/seed, RMSE, best/final
validation loss, selected epoch/checkpoint, pretraining checkpoint/loss weights,
seeds, test-value count, test-input digest and prediction path. Pretraining
validation losses are weighted SSL objectives and should not be compared
numerically across different weight settings. The unqualified validation-loss
columns are the common downstream RMSE.

`summary_MNIST_ROWS.csv`, `summary_SIN_COS.csv` and `summary.json` report mean,
sample standard deviation (`ddof=1`) and number of runs. Single-seed standard
deviations are null/blank, not zero. Each dataset also has a `.tex` table.
Repeated executions of the same seed are verification runs and are not counted
as additional independent seeds. Per-run folders preserve all selected weights,
training histories, predictions, exact pretraining argv/cwd/logs and split audits.
The runner refuses to overwrite an existing experiment manifest. Generated
datasets/checkpoints remain inside git-ignored `results/`.

## Executed commands

From `/home/xujiang/TS_JEPA`, using Python 3.11 and PyTorch 2.11.0+cu130 on an
RTX 3060. Full pretraining subprocess commands are saved as `pretrain.json`
beside each pretraining log.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/xujiang/miniconda3/envs/ts-jepa/bin/python visualization/plot_dual_loss_smoke_results.py > results/00_stage00_legacy_reproduction/command.log 2>&1

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/xujiang/miniconda3/envs/ts-jepa/bin/python run_stage00_comparison.py --seeds 7 --output-dir results/00_model_comparison > results/00_stage00_comparison.log 2>&1

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/xujiang/miniconda3/envs/ts-jepa/bin/python run_stage00_comparison.py --seeds 7 --output-dir results/00_model_comparison_repeat > results/00_stage00_comparison_repeat.log 2>&1

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/xujiang/miniconda3/envs/ts-jepa/bin/python -m unittest discover -s tests -p 'test_*dual_loss*.py' -v > results/00_stage00_regression_tests.log 2>&1

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/xujiang/miniconda3/envs/ts-jepa/bin/python -m unittest discover -s tests -p test_stage00_comparison.py -v > results/00_stage00_unit_tests.log 2>&1

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/xujiang/miniconda3/envs/ts-jepa/bin/python run_stage00_comparison.py --seeds 7 8 --datasets SIN_COS --models Naive-last Drift Mean-context --output-dir results/00_stage00_multiseed_check > results/00_stage00_multiseed_check.log 2>&1

/home/xujiang/miniconda3/envs/ts-jepa/bin/python run_stage00_comparison.py --aggregate-only --output-dir results/00_model_comparison
/home/xujiang/miniconda3/envs/ts-jepa/bin/python run_stage00_comparison.py --aggregate-only --output-dir results/00_stage00_multiseed_check

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/xujiang/miniconda3/envs/ts-jepa/bin/python results/00_stage00_verification.py
```

The six-run baseline command verifies list-of-seeds execution and aggregation; it does not
constitute a multi-seed learned-model comparison. Each deterministic sine
baseline has two runs and SD=0, as expected for this fixed dataset.

## Legacy reproduction

| Legacy diagnostic | Archived RMSE | Measured RMSE |
|---|---:|---:|
| MNIST JEPA-MAE, original reconstruction decoder | 0.111085 | 0.111085 |
| MNIST previous-row copying (invalid context-only baseline) | 0.188641 | 0.188641 |
| Sine-cosine JEPA-MAE, 12 CPU threads | 0.0102686705 | 0.0102686705 |
| Sine-cosine Naive-last | 0.0895966440 | 0.0895966440 |

The initial one-thread legacy sine run produced RMSE 0.012157 (full precision
in `results/00_dual_loss_smoke/summary.csv`). A targeted rerun using
`torch.set_num_threads(12)` in the original helper reproduced
0.01026867050677538 exactly. Thread-dependent floating-point reduction order
can change the 120-epoch optimization trajectory despite the same seed.
No data, model, loss, seed, or hyperparameter was changed in this reproduction.
Its checkpoint and record are in `results/00_stage00_legacy_threads/`.

The targeted reproduction command was:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/xujiang/miniconda3/envs/ts-jepa/bin/python -u - <<'PY' > results/00_stage00_legacy_threads/command.log 2>&1
import json
from pathlib import Path
import torch
from src.stage00_legacy import _pretrain_smoke_case, _sin_cos_rows, _fit_downstream_and_predict
workdir = Path('results/00_stage00_legacy_threads').resolve()
checkpoint, _ = _pretrain_smoke_case(workdir, 'SMOKE_SIN_COS', _sin_cos_rows(), log_path=workdir / 'pretrain.log')
torch.set_num_threads(12)
result = _fit_downstream_and_predict(checkpoint, workdir / 'data/SMOKE_SIN_COS/SMOKE_SIN_COS.csv')
record = {'torch_num_threads': torch.get_num_threads(), 'model_rmse': result['model_rmse'], 'naive_rmse': result['naive_rmse'], 'pretrain_checkpoint': str(checkpoint)}
(workdir / 'results.json').write_text(json.dumps(record, indent=2) + '\n')
print(record)
PY
```

## Measured primary results

Seed 7, one independent run per method/dataset, one CPU thread. Mean RMSE
therefore equals the raw RMSE and across-seed SD is unavailable. Epochs below
are one-based downstream epochs; SSL checkpoint epochs retain zero-based names.

| Method | MNIST test RMSE | MNIST best val / epoch | Sine test RMSE | Sine best val / epoch |
|---|---:|---:|---:|---:|
| JEPA-MAE | 0.1089172661 | 0.1129009575 / 30 | 0.0086312825 | 0.0105269477 / 119 |
| JEPA-only | 0.1464525312 | 0.1506574154 / 29 | 0.0091687571 | 0.0082756914 / 120 |
| MAE-only | 0.1091107652 | 0.1127941161 / 24 | 0.0082880696 | 0.0098208720 / 118 |
| GRU | 0.1702006608 | 0.1707423478 / 30 | 0.0028741523 | 0.0025618963 / 86 |
| Naive-last | 0.2462295294 | — | 0.0895966440 | — |
| Drift | 0.2782523036 | — | 0.0926638022 | — |
| Mean-context | 0.2782817185 | — | 0.0741638914 | — |

The full model and MAE-only are close on MNIST; the single run does not establish
a meaningful advantage for the full objective. GRU has the lowest measured sine
RMSE. Sine SSL remains a one-optimizer-step smoke budget and cannot support a
strong conclusion about representation learning. The small validation/test
sets, different frozen/fine-tuned transfer protocols across datasets, and
single learned-model seed limit interpretation. No model was retuned after
test results were observed.

Verification passed: 26 existing dual-loss tests plus seven focused comparison
tests. All 14 primary runs were repeated: every saved test prediction,
pretrained weight, selected downstream weight, validation minimum and selected
epoch matched exactly. All saved metrics/predictions were finite, and each
dataset's test tensors matched across all seven methods. The machine-readable
verification is `results/00_stage00_verification.json`.

Main results: `results/00_model_comparison/`. Complete repeat verification:
`results/00_model_comparison_repeat/`. Separate multi-seed runner check:
`results/00_stage00_multiseed_check/`. Legacy output:
`results/00_dual_loss_smoke/`, `results/00_stage00_legacy_reproduction/` and
`results/00_stage00_legacy_threads/`. Machine-readable artifact paths are also
listed in `results/00_stage00_artifact_paths.txt`.
