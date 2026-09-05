# Experimental Results and Analysis: Stages 00–04

## Executive summary

This chapter evaluates the staged development of TS-JEPA, beginning with diagnostic pipeline checks and progressing through preprocessing, sentiment, architecture/context, and loss-weight experiments. The central empirical result is that window-relative normalization, inclusion of historical sentiment, and a shared-target JEPA–MAE encoder with a 12-patch downstream context produced the lowest validation RMSE among the confirmatory Stage 01–03 candidates. After correction of the Stage 03 validation-coverage mismatch, the selected configuration achieved an average RMSE of **0.047260** and direction accuracy of **50.89%**.

The conclusions must nevertheless be bounded carefully. Stages 01 and 02 used 12 validation origins, whereas corrected Stage 03 and Stage 04 used 24. Metrics should therefore be compared directly only within a stage unless the target sets are identical. Stage 04 is exploratory: its shared-target branch inherited the formerly selected 6-patch context, while corrected Stage 03 selects 12 patches, and the Stage 04 runs were performed after the original test evaluation. The previously generated 2025 test result consequently describes the superseded shared-context-6 model and is not a valid final evaluation of the corrected selected model.

## 1. Experimental protocol

### 1.1 Forecasting task

The downstream task uses multivariate historical information to forecast the future five-step trajectory of `Close`. The principal experiments use the following inputs:

- `Close`;
- `Volume`;
- 10-day moving average (`MA10`);
- 50-day moving average (`MA50`);
- daily mean sentiment when enabled.

The forecasting target is the cutoff-relative return

\[
y_{t,h}=\frac{P_{t+h}}{P_t}-1, \qquad h=1,\ldots,5,
\]

and RMSE is computed over all saved rolling-origin and forecast-horizon target values. Direction accuracy compares the signs of consecutive movements within the forecast path. For relative-return forecasts, the first movement is measured from the known zero-return origin at the forecast cutoff.

### 1.2 Data coverage and aggregation

Stages 01–04 use five equities (`NVDA`, `AAPL`, `AVGO`, `TSLA`, and `WMT`) and three seeds (`42`, `44`, and `46`), giving 15 stock–seed runs per candidate. Candidate-level metrics are calculated by averaging seeds within each stock and then averaging the five stock means. RMSE is the primary selection metric; direction accuracy is secondary.

All splits are chronological. Historical observations from the preceding split may supply context, but forecast targets remain entirely inside the requested validation or test period. No held-out test metric is used in the corrected Stage 01–03 selection.

| Stage | Role | Evaluation targets per run | Target period | Evidence status |
|---|---|---:|---|---|
| 00 | Pipeline smoke test | Not comparable to financial runs | Synthetic/diagnostic | Diagnostic only |
| 01 | Normalization selection | 12 origins | 2024-10-02 to 2024-12-26 | Valid within-stage comparison |
| 02 | Sentiment ablation | 12 origins | 2024-10-02 to 2024-12-26 | Valid within-stage comparison |
| 03 | Architecture/context selection | 24 origins | 2024-07-09 to 2024-12-26 | Corrected valid selection |
| 04 | Loss-weight sweep | 24 origins | 2024-07-09 to 2024-12-26 | Exploratory |

Because Stage 01–02 and Stage 03–04 use different validation supports, reductions between their aggregate RMSE values should not be interpreted as pure stage-to-stage improvements. The controlled comparisons inside each stage remain meaningful.

## 2. Stage 00: diagnostic verification

Stage 00 tested whether the dual-loss pipeline could execute end to end. It covered JEPA–MAE pretraining, checkpoint loading, downstream fitting, prediction, RMSE evaluation, and plotting on two small diagnostic cases.

| Diagnostic case | Model RMSE | Naive RMSE | Relative reduction |
|---|---:|---:|---:|
| MNIST image rows | 0.111085 | 0.188641 | 41.1% |
| Synthetic sine/cosine | 0.010269 | 0.089597 | 88.5% |

The model improved substantially over the simple diagnostic baselines, indicating that the implementation could reconstruct or predict structured sequences and that the training/evaluation pipeline was operational. These results do not constitute evidence about financial forecasting, representation transfer, or generalization. Their role is strictly diagnostic, and they should not be combined numerically with Stages 01–04.

## 3. Stage 01: normalization selection

Stage 01 compared training-set z-score normalization with window-relative return normalization. All other experimental components were held fixed.

| Normalization | RMSE | Direction accuracy | Selection |
|---|---:|---:|---|
| Window-relative return | **0.052634** | **54.78%** | Selected |
| Training-set z-score | 0.056598 | 53.56% | Not selected |

Window-relative normalization reduced aggregate RMSE by approximately **7.0%** and increased direction accuracy by **1.22 percentage points**. It achieved lower RMSE for all five stocks, which makes the result more robust than an improvement driven by only one high-volatility equity.

The result is consistent with the hypothesis that a forecast-origin-relative coordinate system is better suited to multi-equity return forecasting than a single training-distribution scale. Relative normalization reduces dependence on absolute price level and ensures that the historical close path and future target share the same local reference. This interpretation is plausible but not causal: the experiment establishes better predictive performance under the tested protocol, not the precise mechanism producing it.

**Stage 01 decision:** carry window-relative return normalization into Stage 02.

## 4. Stage 02: sentiment ablation

Stage 02 compared the selected market-feature pipeline with and without historical daily mean sentiment.

| Sentiment | RMSE | Direction accuracy | Selection |
|---|---:|---:|---|
| Included | **0.049247** | 52.33% | Selected |
| Excluded | 0.052634 | **54.78%** | Not selected |

Adding sentiment reduced aggregate RMSE by approximately **6.44%**, but direction accuracy declined by **2.44 percentage points**. Sentiment improved RMSE for AVGO, NVDA, TSLA, and WMT, but not AAPL. The RMSE benefit is therefore broad but not universal.

This result indicates a metric trade-off. Sentiment appears to help estimate the magnitude of the future return trajectory while not improving—and possibly weakening—the signs of consecutive movements. Because the selection hierarchy defines RMSE as primary, the sentiment-enabled configuration is the correct Stage 02 choice. The result should not be described as evidence that sentiment improves every aspect of forecasting.

Interpretation also depends on timestamp integrity. The sentiment result is valid only under the assumption, enforced by preprocessing, that each news-derived value was available at or before the associated forecasting origin. No future news may be aggregated into a historical input.

**Stage 02 decision:** carry the sentiment-enabled feature set into Stage 03.

## 5. Stage 03: corrected architecture and context comparison

### 5.1 Validity correction

The initial Stage 03 ranking compared candidates with unequal validation support: shared context 6 had 18 origins, shared context 12 had 12, and the remaining candidates had 24. That comparison was invalid because differences in RMSE could reflect different target periods rather than model quality.

The evaluation loader was corrected so that earlier chronological observations provide context while all prediction targets remain inside validation. Only the two affected candidates—shared context 6 and shared context 12—were regenerated. Pretrained checkpoints and the four already valid candidates were reused. The corrected comparison gives all six candidates identical coverage: 24 origins from 2024-07-09 through 2024-12-26 for every stock and seed.

### 5.2 Aggregate ranking

Stage 03 compares shared-target JEPA–MAE and Local-MAE/Long-JEPA with downstream contexts of 6, 12, and 24 patches. At five observations per patch, these correspond to 30, 60, and 120 historical trading days.

| Rank | Configuration | Context | RMSE | Direction accuracy |
|---:|---|---:|---:|---:|
| 1 | Shared-target JEPA–MAE | 12 patches | **0.047260** | 50.89% |
| 2 | Local-MAE/Long-JEPA | 12 patches | 0.048152 | **52.61%** |
| 3 | Shared-target JEPA–MAE | 6 patches | 0.049227 | 49.72% |
| 4 | Local-MAE/Long-JEPA | 24 patches | 0.049353 | 51.33% |
| 5 | Shared-target JEPA–MAE | 24 patches | 0.049475 | 52.00% |
| 6 | Local-MAE/Long-JEPA | 6 patches | 0.049541 | 50.94% |

The shared-target 12-patch configuration achieved an RMSE approximately **1.85%** lower than the second-ranked local-long 12-patch configuration. It also improved RMSE by approximately 4.0% relative to shared context 6 and 4.5% relative to shared context 24. The validity correction therefore changes the Stage 03 selection from shared context 6 to **shared context 12**.

### 5.3 Context-length effect

Both masking formulations achieved their lowest RMSE at 12 patches. The relationship between context length and performance is therefore non-monotonic:

- shared-target RMSE changed from 0.049227 at 6 patches to 0.047260 at 12 and 0.049475 at 24;
- local-long RMSE changed from 0.049541 at 6 patches to 0.048152 at 12 and 0.049353 at 24.

The 60-day history appears to balance recent and medium-range information. Thirty days may omit useful temporal structure, while 120 days may incorporate older regimes or increase the difficulty of downstream optimization. These mechanisms are hypotheses; the evidence directly supports only the observed U-shaped performance pattern.

### 5.4 Shared-target versus local-long masking

Shared-target masking had lower RMSE at 6 and 12 patches, by approximately 0.64% and 1.85%, respectively. At 24 patches, local-long masking was better by only 0.25%, an effect too small to support a general architectural claim.

Direction accuracy gives a different ordering. Local-long context 12 produced the highest direction accuracy at 52.61%, 1.72 percentage points above shared context 12. Thus, the corrected winner is specifically the best RMSE configuration; it is not the best directional classifier.

### 5.5 Stock-level robustness

| Stock | Lowest-RMSE configuration | RMSE |
|---|---|---:|
| AAPL | Shared context 24 | 0.022097 |
| AVGO | Local-long context 6 | 0.063077 |
| NVDA | Shared context 24 | 0.054915 |
| TSLA | Shared context 12 | 0.076272 |
| WMT | Shared context 12 | 0.016379 |

No configuration won every stock. Nevertheless, shared context 12 had lower RMSE than the second-ranked local-long context 12 for all five equities. This makes its advantage over the nearest competitor consistent across stocks, although its larger aggregate advantage over some other candidates was influenced by strong results on TSLA and WMT.

For the selected candidate, seed-level RMSE standard deviation ranged from approximately 0.00044 for WMT to 0.00418 for TSLA. Most equities were relatively stable across the three seeds, while TSLA showed greater sensitivity.

### 5.6 Selected candidate versus baselines

| Model | RMSE | Direction accuracy |
|---|---:|---:|
| Shared-target JEPA–MAE, context 12 | **0.047260** | 50.89% |
| GRU | 0.049179 | 50.00% |
| Naive-last | 0.051538 | 0.33% |
| Drift | 0.052534 | 54.17% |
| Mean-context | 0.052817 | **54.67%** |

Shared context 12 achieved approximately 3.9% lower aggregate RMSE than the supervised GRU. Against naive-last, the stock-aware paired analysis estimated a mean RMSE difference of −0.004277 and a relative improvement of 6.71%. TS-JEPA beat naive-last for four of five stocks and 12 of 15 stock–seed runs.

The bootstrap confidence interval for the mean TS-JEPA-minus-naive RMSE difference was approximately [−0.00835, −0.00059]. The exact Wilcoxon test, however, produced a Holm-adjusted value of \(p=0.125\). With only five equities, the evidence is promising but does not reach conventional significance. The deterministic drift and mean-context baselines also achieved higher direction accuracy, reinforcing the distinction between magnitude accuracy and directional accuracy.

The near-zero naive-last direction score follows from the metric definition: a constant zero-return path produces zero movements, whereas the realized path usually has non-zero movements. It should not be interpreted as ordinary binary direction classification accuracy.

### 5.7 Horizon behaviour

| Horizon | RMSE | Direction accuracy |
|---:|---:|---:|
| 1 | 0.024871 | 52.78% |
| 2 | 0.042169 | 48.89% |
| 3 | 0.051587 | 52.22% |
| 4 | 0.053510 | 49.72% |
| 5 | 0.055961 | 50.83% |

RMSE increased with horizon, as expected when uncertainty accumulates. Shared context 12 had lower RMSE than shared context 6 at every horizon, showing that its aggregate advantage was not isolated to one prediction step. Direction accuracy fluctuated around 50% and did not exhibit a stable horizon trend.

**Stage 03 decision:** select shared-target JEPA–MAE with a 12-patch downstream context for RMSE-oriented model selection.

## 6. Stage 04: exploratory JEPA–MAE loss-weight sweep

Stage 04 varied the JEPA and MAE loss weights while keeping their sum equal to 2. The candidate set included pure MAE, MAE-dominant, equal-weight, JEPA-dominant, and pure JEPA objectives.

### 6.1 Shared-target branch

| JEPA weight | MAE weight | Context | RMSE | Direction accuracy |
|---:|---:|---:|---:|---:|
| 0.0 | 2.0 | 6 | 0.049057 | 51.28% |
| 0.5 | 1.5 | 6 | **0.048799** | 51.11% |
| 1.0 | 1.0 | 6 | 0.049246 | 51.06% |
| 1.5 | 0.5 | 6 | 0.049800 | **53.33%** |
| 2.0 | 0.0 | 6 | 0.049594 | 52.83% |

The lowest shared-target RMSE came from the MAE-dominant 0.5/1.5 objective. It improved RMSE by approximately 0.87% relative to the Stage 03 shared-context-6 default weighting of 1.0/0.5. In contrast, the highest direction accuracy came from the JEPA-dominant 1.5/0.5 objective. This again demonstrates that RMSE and directional performance respond differently to the training objective.

The 0.87% cross-stage difference is descriptive rather than a controlled estimate of the loss-ratio effect because the Stage 03 default weights sum to 1.5, whereas every Stage 04 pair sums to 2. Controlled loss-weight conclusions should therefore be drawn primarily from comparisons among the Stage 04 candidates.

### 6.2 Local-long branch

| JEPA weight | MAE weight | Context | RMSE | Direction accuracy |
|---:|---:|---:|---:|---:|
| 0.0 | 2.0 | 12 | 0.048328 | 52.33% |
| 0.5 | 1.5 | 12 | 0.048021 | 52.61% |
| 1.0 | 1.0 | 12 | **0.047529** | 53.78% |
| 1.5 | 0.5 | 12 | 0.047548 | **55.39%** |
| 2.0 | 0.0 | 12 | 0.047736 | 54.44% |

The equal-weight local-long objective achieved the lowest RMSE, but the JEPA-dominant 1.5/0.5 result was nearly identical: the absolute RMSE difference was only 0.000019. The JEPA-dominant configuration also achieved the best direction accuracy. Pure MAE was the weakest local-long RMSE configuration, while pure JEPA remained competitive. This pattern is consistent with the hypothesis that latent prediction contributes useful longer-range information and that a reconstruction term provides a small complementary benefit.

### 6.3 Stage 04 validity boundary

Stage 04 must be treated as exploratory for two reasons.

First, its branches do not use the same context. The shared-target branch uses 6 patches because it inherited the original, invalid Stage 03 winner; the local-long branch uses 12. Corrected Stage 03 selects shared context 12. Consequently, the lower local-long Stage 04 RMSE cannot be attributed solely to the masking strategy or loss weights because context length is confounded with architecture.

Second, the Stage 04 runs were performed after the original held-out test run. They may be reported as post-hoc sensitivity analysis, but they must not be used to claim that a final model was selected without observing test performance.

Within each branch, the loss-weight comparison remains informative because the candidates share target coverage, context, stocks, seeds, and evaluation protocol. A confirmatory Stage 04 would require rerunning the shared-target weight grid with the corrected 12-patch context under a newly frozen protocol and evaluating the selected configuration only on a new untouched holdout.

## 7. Cross-stage synthesis

| Stage | Controlled question | Selected result | Main implication |
|---|---|---|---|
| 00 | Does the pipeline execute end to end? | Both diagnostics beat naive baselines | Implementation passed a functional smoke test |
| 01 | Which normalization is preferable? | Window-relative return | Local relative coordinates improve RMSE consistently |
| 02 | Does historical sentiment help? | Sentiment included | Better RMSE but weaker direction accuracy |
| 03 | Which masking/context combination is best? | Shared target, 12 patches | Medium context minimizes validation RMSE |
| 04 | How do JEPA/MAE weights affect performance? | Branch-dependent | Loss balance matters, but findings are exploratory |

Three recurring patterns emerge.

1. **Metric trade-offs are systematic.** The configurations with the lowest RMSE are often not those with the highest direction accuracy. Conclusions must identify the metric being optimized.
2. **More history is not automatically better.** Both Stage 03 architectures performed best at 12 patches, with degradation at 24 patches.
3. **Combined representation objectives appear useful but are not uniformly optimal.** Stage 04 favoured an MAE-dominant shared objective and an equal or JEPA-dominant local-long objective. The best weighting depends on the masking structure and potentially on context.

The observed RMSE changes across the selected winners—0.052634 in Stage 01, 0.049247 in Stage 02, and 0.047260 in corrected Stage 03—should not be interpreted as a single additive improvement curve because Stage 03 uses twice as many validation origins. Only the controlled contrasts within each stage support causal comparisons of design choices.

## 8. Status of the held-out test

The original automated workflow evaluated the formerly selected shared-context-6 configuration on the 2025 test period before the Stage 03 correction.

| Model | Historical 2025 test RMSE | Direction accuracy |
|---|---:|---:|
| Naive-last | **0.046404** | 0.16% |
| GRU | 0.048031 | 49.95% |
| Mean-context | 0.050105 | **50.56%** |
| Drift | 0.050282 | 50.40% |
| TS-JEPA, superseded context 6 | 0.054855 | 48.37% |

The superseded TS-JEPA configuration was approximately 18.2% worse than naive-last in RMSE. This is important negative evidence about that specific model, but it is not the final test result for the corrected shared-context-12 selection.

No new test run was performed during the validity repair. Because the 2025 test outcomes have already been observed and Stage 04 was subsequently explored, using that same period repeatedly would weaken the claim of a single untouched final evaluation. The rigorous next step is to freeze the corrected pipeline and use a later, previously unexamined holdout period. Until then, the thesis should describe Stages 01–03 as validation-based model selection and Stage 04 as exploratory analysis, without claiming confirmed out-of-sample superiority.

## 9. Threats to validity

### 9.1 Small cross-sectional sample

Only five equities are used. Three seeds help quantify optimization variability, but seeds do not replace independent assets. Statistical inference should therefore be performed at the stock level, and non-significant exact tests should not be overridden by visually favourable means.

### 9.2 Overlapping forecast windows

Rolling origins and multi-step horizons may share target observations. Individual forecast rows are not independent replicates. Stock-level aggregation and paired stock-level comparisons are more defensible than treating every row as independent.

### 9.3 Asset heterogeneity

RMSE varies substantially across equities, with TSLA producing much larger errors than WMT or AAPL. Equal stock weighting prevents high-volatility assets from completely determining the aggregate, but results may not transfer to a broader universe.

### 9.4 Sequential model selection

Each stage inherits earlier decisions. This reduces the size of the search but means interactions between discarded and later-stage choices are not fully explored. For example, sentiment might interact with normalization, and objective weights might interact with context length.

### 9.5 Direction metric interpretation

Direction accuracy evaluates movements within a predicted trajectory and includes the known zero origin for relative returns. It is not equivalent to conventional binary next-day direction accuracy. Constant forecasts can therefore score near zero rather than near 50%.

### 9.6 Stage 04 procedural status

Stage 04 was run after test observation and used the superseded shared context. Its results are suitable for hypothesis generation but not for retroactive final-model selection.

## 10. Defensible thesis claims

The following claims are supported by the saved evidence:

- Window-relative normalization outperformed training-set z-score normalization across all five equities in the Stage 01 validation comparison.
- Adding historical sentiment reduced aggregate validation RMSE in Stage 02, although direction accuracy declined.
- After aligning validation targets, both shared-target and local-long models achieved their lowest Stage 03 RMSE with a 12-patch context.
- Shared-target JEPA–MAE with 12 patches was the corrected RMSE-selected Stage 03 configuration.
- The selected Stage 03 representation outperformed GRU and deterministic baselines in aggregate validation RMSE, but the five-stock exact test did not establish conventional statistical significance.
- Stage 04 suggests that the preferred JEPA–MAE loss balance depends on masking strategy, but this is exploratory evidence.

The following claims are not supported:

- that Stage 00 demonstrates financial forecasting ability;
- that sentiment improves directional forecasting;
- that shared-target masking is universally superior to local-long masking;
- that a 24-patch context is better because it contains more history;
- that Stage 04 identifies a confirmatory final model;
- that the corrected selected model has demonstrated superiority on an untouched test set.

## 11. Overall conclusion

The corrected staged experiments provide evidence that representation-transfer performance depends materially on preprocessing, auxiliary information, temporal context, and the balance between JEPA and MAE objectives. The most defensible selected configuration is window-relative normalization with historical sentiment, shared-target JEPA–MAE pretraining, and a 12-patch downstream context. It achieves the lowest corrected Stage 03 validation RMSE while showing reasonably stable performance across seeds and a consistent RMSE advantage over the nearest local-long competitor.

The result is encouraging but not conclusive. Direction accuracy remains close to chance-like movement discrimination, deterministic baselines retain directional advantages, asset-level heterogeneity is substantial, and the exact stock-level inference is underpowered. Most importantly, the corrected selected model has not yet been evaluated on a new untouched holdout. The thesis should therefore frame the work as evidence that JEPA–MAE representations can improve validation RMSE under a rigorous chronological protocol, followed by an explicit requirement for future confirmatory out-of-sample evaluation.

## 12. Artifact trail

- Stage 00 diagnostic snapshot: [`thesis_results/00_dual_loss_smoke/e56db7d33c56`](../thesis_results/00_dual_loss_smoke/e56db7d33c56/)
- Corrected Stage 01–03 selection: [`selection_artifacts/chapter5_stage3_validation_repair/selection_summary.json`](../selection_artifacts/chapter5_stage3_validation_repair/selection_summary.json)
- Corrected shared-context-6 publication: [`thesis_results/03_shared_context_6_patches/49b848da7bb2-2609b9618086`](../thesis_results/03_shared_context_6_patches/49b848da7bb2-2609b9618086/)
- Corrected shared-context-12 publication: [`thesis_results/03_shared_context_12_patches/afc29d4b16b6-8b70a34f4982`](../thesis_results/03_shared_context_12_patches/afc29d4b16b6-8b70a34f4982/)
- Candidate configurations: [`config/experiments/chapter5_candidates`](../config/experiments/chapter5_candidates/)

This analysis uses the runtime `evaluation_split`, `evaluation_sample_count`, `evaluation_target_start`, and `evaluation_target_end` metadata. Legacy analysis snapshots that label validation runs as test runs should not be used for split interpretation.
