# ARFF loader corrections: findings

Written: 26 October 2026.

Evidence: [project commit a22d4fe](https://github.com/Shuharto-Jannat/copod-reproducibility-project/commit/a22d4feac6dd253a5e86c726a30369cccd7415a3).

## Purpose

I investigated whether the benchmark loader selected the intended measurements and anomaly labels from the ARFF files. This follows the Assessment 2 feedback requesting further investigation of differences in the reproduced results.

I kept the historical results and original source unchanged. The new experiment distinguishes reproducing the released loader's behaviour from evaluating COPOD with correctly selected columns and labels.

## Confirmed loading problems

The mapping audit examined all 16 ARFF datasets. The authors' released loader and my project loader selected the same features and labels on these files. Four datasets had an ID column included as a predictor while a measurement was omitted:

| Dataset | Rows | Predictor count in both versions | Incorrectly included | Omitted measurement | Label problem |
|---|---:|---:|---|---|---|
| HeartDisease | 270 | 13 | `id` | `Thal` | None found |
| KDDCup99 | 60,839 | 41 | `id` | `dst_host_srv_rerror_rate` | This measurement was also used as the label instead of `outlier` |
| Lymphography | 148 | 18 | `id` | `att19` | None found |
| Pima | 768 | 8 | `id` | `att8` | None found |

The feature counts stayed unchanged because the ID replaced a measurement. Matching dimensions therefore did not establish that the correct columns were loaded. The other 12 files passed this column-selection audit; that is not a complete assessment of their data quality.

For KDDCup99, the legacy conversion produced 95 positive labels, whereas the named `outlier` column contains 246 positives. There are 339 row-level disagreements. Fractional values in the incorrectly selected measurement were truncated during conversion to integers. These are disagreements between label definitions, not COPOD prediction errors.

## Controlled comparison

I used the authors' unchanged `models/cod.py`, pinned at commit `3ca17400016386546d2c199886e81bdcc859b36b`.

| Condition | Measurements used to score rows | Labels used to evaluate scores |
|---|---|---|
| A: `legacy_columns_legacy_labels` | Original loader's columns | Original loader's labels |
| B: `legacy_columns_true_labels` | Exactly the same scores as A | Named `outlier` labels |
| C: `corrected_columns_true_labels` | Measurements selected by name, excluding `id` and `outlier` | Named `outlier` labels |

I ran ten split seeds (0–9) for each affected dataset: 40 dataset/split combinations and 120 metric rows. Each split used 60% of rows for training and 40% for testing, without stratification or subsampling, following the original ARFF benchmark. A, B and C used identical ordered training/test rows within each trial.

Scaling was fitted on training measurements separately for each feature set. COPOD's `fit` received features without labels. Both scoring runs used the same legacy positive fraction as the contamination parameter; this affects the classification threshold in the inspected source, while the evaluated raw scores do not depend on that threshold.

The authors' scoring function combined training features with the entire test batch to calculate empirical tails and skewness. Consequently, this remains batch/transductive scoring, not evaluation against a fixed training-only reference.

The recorded environment was Python 3.12.1, NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.0, scikit-learn 1.9.0, PyOD 3.6.4, statsmodels 0.14.6 and liac-arff 2.5.0. PyOD supplied preprocessing utilities; the detector itself was the authors' implementation.

## Reproduction check

I first checked condition A against the previously saved project results. The original benchmark rounded each trial's metric to four decimal places before averaging. Applying that same rule, all eight averages (ROC AUC and average precision for four datasets) matched the historical values within `1e-12` numerical tolerance.

This establishes agreement with the saved project averages under the legacy loader. It does not establish that the legacy columns were correct, prove every detail of the historical execution, or verify the published table.

## Effect of correcting the columns

The following comparison uses the correct `outlier` labels in both conditions, B and C. Values are means across ten splits, rounded here to four decimal places. AP means average precision.

| Dataset | ROC AUC: B | ROC AUC: C | AP: B | AP: C |
|---|---:|---:|---:|---:|
| HeartDisease | 0.6505 | 0.6979 | 0.5967 | 0.6444 |
| KDDCup99 | 0.9932 | 0.9924 | 0.4493 | 0.3439 |
| Lymphography | 0.9990 | 0.9990 | 0.9867 | 0.9867 |
| Pima | 0.6439 | 0.6492 | 0.5261 | 0.5223 |

The paired changes below are C minus B, calculated within each split before summarising. SD is the sample standard deviation of those ten changes.

| Dataset | Mean ROC AUC change ± SD | Mean AP change ± SD |
|---|---:|---:|
| HeartDisease | +0.047430 ± 0.011598 | +0.047704 ± 0.013975 |
| KDDCup99 | −0.000842 ± 0.000096 | −0.105481 ± 0.013851 |
| Lymphography | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| Pima | +0.005274 ± 0.012635 | −0.003888 ± 0.015291 |

- **HeartDisease:** both measures increased in all ten splits after correcting the columns.
- **KDDCup99:** both measures decreased in all ten splits when the columns were corrected while retaining the true labels. The AP decrease was much larger than the ROC AUC decrease.
- **Lymphography:** both measures were unchanged in every split. This does not imply that every individual anomaly score was unchanged. Test sets contained only two to four positives.
- **Pima:** effects were mixed. ROC AUC increased in seven splits and decreased in three; AP increased in four and decreased in six.

Correct column selection is required even when it lowers performance. I did not choose the loader according to which version gave the highest score.

## KDDCup99: correcting labels is a separate effect

| Condition | Mean ROC AUC | Mean AP |
|---|---:|---:|
| A: legacy columns and legacy labels | 0.997589 | 0.260476 |
| B: identical legacy scores, true labels | 0.993228 | 0.449339 |
| C: corrected columns, true labels | 0.992386 | 0.343857 |

The increase in AP from A to B does not represent a better detector: the scores are identical. The answer key changed. This changes both which observations count as positives and their prevalence. Across the test splits, mean positive prevalence changed from approximately 0.1574% under the legacy labels to 0.4019% under the true labels.

It would therefore be misleading to describe the combined AP change from A to C as a straightforward model improvement. The comparison B to C isolates the joint column correction on the same labelled task.

## Recorded numerical behaviour

Each KDDCup99 scoring run recorded 121,678 non-finite intermediate feature contributions. The columns `land` and `num_outbounds_cmds` are identically zero throughout the dataset. Their skewness is undefined, producing NaN contributions in the original calculation: two columns × 60,839 concatenated rows = 121,678 entries.

The authors' pandas row summation skips these NaNs, and all final test scores were finite. These are not 121,678 invalid predictions. I retained and documented the original behaviour rather than modifying the detector during the column-selection experiment. Some training splits also flag `urgent` as constant, but it is not constant across the combined training/test batch.

There were 20 scoring runs with numerical flags: two feature versions across ten KDD splits. No non-finite intermediate contributions or recorded warnings were reported for the other three datasets. The metadata and numerical diagnostics retain these checks.

## Interpretation and limits

The experiment confirms that loading choices affected the reported COPOD results on these files, with different effects across datasets. It provides a controlled follow-up to the mapping audit and a reproducible check against the historical project averages.

The ten splits overlap. Their SD describes variation across those splits, not a confidence interval or ten independent datasets. Removing ID and restoring the omitted measurement were performed together, so this experiment does not identify their separate effects. Conclusions apply to the pinned files, source and recorded environment.

Only COPOD was rerun. I cannot use these results to replace the other nine detectors' results or claim a corrected benchmark ranking. Nor do these findings establish the cause of all differences from the publication. The separate hypothesis about AP-table row alignment remains exploratory; it is not a confirmed correction to the published table.

## Evidence and commands

The experiment and summary are reproducible with:

```bash
python scripts/evaluate_arff_loader_corrections.py --datasets HeartDisease KDDCup99 Lymphography Pima --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/summarise_arff_loader_corrections.py
```

These commands reuse verified trial files and check existing summary files rather than replacing differing evidence. The summary also recalculates the metrics from saved scores.

- `results/audits/arff_dataset_mapping/`: feature/label mapping audit.
- `results/loader_corrections/v1/<dataset>/seedXX/`: metrics, test scores, row splits and metadata for each trial.
- `results/loader_corrections/v1/summary/performance_summary.csv`: means, sample SD, positive counts, prevalence and valid-trial counts.
- `results/loader_corrections/v1/summary/paired_summary.csv`: mean paired changes, their sample SD and direction counts.
- `results/loader_corrections/v1/summary/historical_comparison.csv`: comparisons with the historical legacy averages.
- `results/loader_corrections/v1/summary/numerical_diagnostics.csv`: constant-feature, warning and intermediate-value records.
- `results/loader_corrections/v1/summary/summary_metadata.json`: input/output hashes, source and environment information, and interpretation limits.

The historical results in `results/reproduction_arff_16_10_trials/` remain unchanged.
