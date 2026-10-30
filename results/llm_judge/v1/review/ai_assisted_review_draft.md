# Draft claim reviews for Shuharto to check

Prepared by the Codex assistant: 30 October 2026 (Australia/Sydney).

**Status: AI-proposed assessments awaiting your evidence review. These are not recorded as your judgments or as outputs from the two planned judge sessions.**

Read each claim, explanation and cited evidence. You can accept the draft, change it, or mark it uncertain. Your decisions will be recorded separately after you provide them. The procedure change is documented in `notes/llm_judge_protocol_amendment_01.md`.

The sources are the frozen packet at `results/llm_judge/v1/inputs/evidence_packet.md`, based on experiment commit `ed494a3b3c78ecd17f5436f6c9bcfae0050603fe`. Search the packet for the evidence IDs below. Numbers in this reading guide are rounded where stated; the packet retains the saved values.

## What the four categories mean

| Category | Meaning |
|---|---|
| `SUPPORTED` | The evidence backs all material parts of the claim at its stated scope. |
| `PARTIALLY_SUPPORTED` | The main finding is backed, but an important qualification or secondary part is missing. |
| `CONTRADICTED` | A material part of the claim conflicts with the evidence. |
| `INSUFFICIENT_EVIDENCE` | The evidence does not establish the main assertion. This does not automatically mean it is false. |

## C01 — Overall reproduced ROC AUC

**Exact claim:** Across the 30 benchmark datasets, the released-loader reproduction's mean COPOD ROC AUC was lower than the published mean recorded in the project's comparison table.

**AI-proposed category: `SUPPORTED`.**

The recorded paper average is **0.8247** and the reproduced average is **0.813238**. The reproduced average is lower. This statement concerns the overall average; some individual datasets scored higher in the reproduction.

**Evidence to check:** E01.01, row `Metric=Mean ROC-AUC`, columns `Paper` and `Reproduced`; E01.02 lists all 30 datasets.

**Short draft reason:** The reproduced mean ROC AUC, 0.813238, is below the recorded published mean of 0.8247 across the 30-dataset comparison.

## C02 — Correcting feature selection

**Exact claim:** With the correct labels used in both conditions, correcting feature selection did not increase mean AP on every one of the four affected ARFF datasets.

**AI-proposed category: `SUPPORTED`.**

This says that **not all four improved**. It does not say that none improved. When comparing the two conditions that both use the correct labels, the mean AP changes were:

| Dataset | Corrected minus legacy-column mean AP, rounded | Direction |
|---|---:|---|
| HeartDisease | +0.047704 | Increased |
| KDDCup99 | -0.105481 | Decreased |
| Lymphography | 0 | Unchanged |
| Pima | -0.003888 | Decreased |

**Evidence to check:** E02.03, `comparison=column_correction_with_true_labels`, column `average_precision_difference_mean`. E02.02 identifies the separate conditions. Do not mix this comparison with changing the labels themselves.

**Short draft reason:** Mean AP increased for HeartDisease, decreased for KDDCup99 and Pima, and stayed unchanged for Lymphography, so the correction did not improve every dataset.

## C03 — Recovering earlier results versus explaining the paper gap

**Exact claim:** The Wine, WBC and Satimage-2 diagnostics recovered the saved project averages, but did not establish the cause of the remaining differences from the recorded paper results.

**AI-proposed category: `SUPPORTED`.**

The diagnostic runs recovered the earlier project ROC AUC and AP averages for all three datasets within the recorded numerical tolerance. That check used the original rule: round each trial to four decimal places, then average.

The differences from the recorded paper results remained. For example, the recovered Wine AP was **0.43240**, while the recorded paper AP was **0.6082**. Finding variation between test splits does not, by itself, explain why the publication's average differs.

**Evidence to check:** E03.01, E03.05 and E03.09, both metric rows, columns `historical_comparison_status` and `diagnostic_rounded_mean_minus_paper`; E03.04, E03.08 and E03.12, `/interpretation_limits`.

**Short draft reason:** The reruns matched the saved project means, but the paper gaps remained and the diagnostics did not identify their cause.

## C04 — Isolation Forest on the primary APS test set

**Exact claim:** On the full APS primary test set, Isolation Forest's mean AP across its five planned seeds exceeded the AP of both COPOD scoring procedures.

**AI-proposed category: `SUPPORTED`.**

| Method | Primary AP, rounded |
|---|---:|
| Standard batch COPOD | 0.531741 |
| Fixed training-reference COPOD | 0.534231 |
| Isolation Forest, five-seed mean | 0.585954 |

Isolation Forest's mean is higher than both COPOD values. This is a comparison of observed AP on this supplied test set. It does not establish that Isolation Forest is always better. Batch COPOD also has different information access because its scoring reference includes test measurements.

**Evidence to check:** E04.02, rows `condition=primary`, columns `method`, `n_runs`, `average_precision_mean` and `score_reference`.

**Short draft reason:** Isolation Forest's five-seed mean AP was approximately 0.585954, exceeding batch COPOD's 0.531741 and fixed-reference COPOD's 0.534231 on the primary APS test set.

## C05 — Removing heavily missing APS features

**Exact claim:** Removing the eight additional high-missingness APS features reduced the observed mean AP for each of the three evaluated methods relative to the primary condition.

**AI-proposed category: `SUPPORTED`.**

The primary condition retained **169 features**; the secondary condition retained **161**, a reduction of eight. The AP results were:

| Method | Primary | After removing eight features |
|---|---:|---:|
| Standard batch COPOD | 0.531741 | 0.529411 |
| Fixed training-reference COPOD | 0.534231 | 0.532017 |
| Isolation Forest, five-seed mean | 0.585954 | 0.569867 |

Every method's listed value decreased. This statement concerns the method means in this experiment, not every individual Isolation Forest seed or every possible missing-data problem.

**Evidence to check:** E04.03, column `n_features`; E04.02, column `average_precision_mean`; E04.04, column `average_precision_difference_mean`, defined as secondary minus primary.

**Short draft reason:** Removing eight additional features reduced the retained count from 169 to 161, and all three methods had lower mean AP in the secondary condition.

## C06 — Increasing the size of the artificial shift

**Exact claim:** In the synthetic extension, both COPOD procedures' ten-seed mean AP increased across the tested single-feature shift levels 0, 2, 4 and 6.

**AI-proposed category: `SUPPORTED`.**

| Shift size | Batch COPOD mean AP | Fixed-reference COPOD mean AP |
|---:|---:|---:|
| 0 | 0.051916 | 0.051187 |
| 2 | 0.065453 | 0.064402 |
| 4 | 0.095410 | 0.094587 |
| 6 | 0.100823 | 0.100430 |

Both columns increase at each tested level. These are averages across ten data seeds. The claim does not say every seed increased at every step or that performance would keep increasing beyond the tested settings.

**Evidence to check:** E05.01, conditions `marginal_delta0`, `marginal_delta2`, `marginal_delta4` and `marginal_delta6`; methods `pyod_batch` and `fixed_training_reference`; columns `n_seeds` and `average_precision_mean`.

**Short draft reason:** Both COPOD procedures' ten-seed mean AP increased successively across shift sizes 0, 2, 4 and 6 in the saved summaries.

## C07 — Detecting broken relationships in the strong-dependence setting

**Exact claim:** In the synthetic extension's correlation-0.8 dependence setting, Ledoit-Wolf exceeded fixed-reference COPOD on both ROC AUC and AP in every one of the ten planned data seeds.

**AI-proposed category: `SUPPORTED`.**

The paired results compare both methods on the same test rows for each data seed. For all ten seeds, Ledoit-Wolf minus fixed-reference COPOD is positive for both metrics. Even the smallest recorded differences are approximately **+0.422397 ROC AUC** and **+0.918035 AP**.

This is strong evidence for the stated comparison within this particular artificial setting. It does not establish superiority on all datasets or kinds of anomalies.

**Evidence to check:** E05.02, all ten rows with `condition=dependence_rho0p8`, seeds 100-109, columns `roc_auc_difference` and `average_precision_difference`; E05.03 confirms ten positive differences for each metric in this comparison.

**Short draft reason:** Both paired metric differences were positive for all ten planned seeds in the correlation-0.8 dependence setting.

## C08 — What the two null controls mean

**Exact claim:** In each of the synthetic extension's two null controls, the label-1 and label-0 groups follow the same population distribution by construction.

**AI-proposed category: `SUPPORTED`.**

In `marginal_delta0`, both groups use the same correlated Gaussian generation rule and the applied shift is zero. In `dependence_rho0p0`, both groups use independent standard-normal features.

Within each control, the two labelled groups therefore come from the same population distribution. Their realised rows, sample averages and correlations can still differ because they are random draws. The two controls themselves are different: one is correlated and one is independent.

**Evidence to check:** E06.01, original plan lines 55-71, especially line 69; E06.02, generator lines 73-76, 88-91 and 135-147; E06.04, `/conditions` and `/factor_rule`.

**Short draft reason:** Each null setting generates both labelled groups from the same distribution, although separate random draws do not produce identical observed samples.

## C09 — A misplaced decimal

**Exact claim:** In the synthetic extension's correlation-0.8 dependence setting, fixed-reference COPOD achieved a ten-seed mean AP of approximately 0.5000.

**AI-proposed category: `CONTRADICTED`.**

The saved mean AP is **0.05038160953953915**, approximately **0.0504**. The claim instead says **0.5000**. That is a substantial decimal-scale error, not a normal rounding difference.

**Evidence to check:** E05.01, row `condition=dependence_rho0p8`, `method=fixed_training_reference`, column `average_precision_mean`.

**Short draft reason:** The recorded mean AP is approximately 0.0504, which contradicts the stated value of approximately 0.5000.

**Corrected wording, if this were a report claim:** In the synthetic extension's correlation-0.8 dependence setting, fixed-reference COPOD achieved a ten-seed mean AP of approximately 0.0504.

## C10 — Applying synthetic success to future truck data

**Exact claim:** The synthetic extension establishes Ledoit-Wolf as the best detector for future Scania APS observations.

**AI-proposed category: `INSUFFICIENT_EVIDENCE`.**

Ledoit-Wolf performed well in the specified artificial settings. Those features were not designed from Scania measurements, and success there does not establish which detector will perform best on future truck observations. The recorded APS experiment compared COPOD with Isolation Forest; it contains no Ledoit-Wolf APS result.

This is missing evidence for the claimed transfer. It does not prove that Ledoit-Wolf would perform poorly on APS. An appropriate APS comparison and evidence about future or representative unseen observations would be needed; even then, any claim of being "best" would need a defined set of methods and evaluation scope.

**Evidence to check:** E06.01, original plan lines 13, 113 and 170-174; E04.01/E04.02, the methods included in the APS results.

**Short draft reason:** The synthetic study does not establish Ledoit-Wolf's performance on future APS observations or its superiority over the relevant alternatives there.

**Narrower wording supported by the recorded comparison:** In the tested correlation-0.8 synthetic dependence setting, Ledoit-Wolf exceeded fixed-reference COPOD on ROC AUC and AP in all ten data seeds.

## Your review record

Your review fields in `author_review_ai_assisted.csv` are intentionally blank. After checking each claim and its evidence, record whether you accept the draft, revise it or remain uncertain, along with your final category and a short reason. Record your actual name and review time only when that review is completed.

C09 and C10 are the protocol's deliberate diagnostic statements. Their errors must remain in the frozen judge prompts so the later judges can assess them. Suggested corrected wording above is explanatory only; it must not replace the frozen claims.

Do not send this draft or your review to either planned judge. The report should describe your eventual checks as AI-assisted author review. No human-reference agreement, planned judge result or completed author judgment is claimed by this draft.
