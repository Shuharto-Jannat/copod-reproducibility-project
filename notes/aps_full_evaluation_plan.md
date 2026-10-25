# APS full-data evaluation plan

Written: 26 October 2026. Plan version: v1.

Status: specified before running the new full-data model comparisons. I will commit this plan before those runs and record any later deviations separately.

## 1. Purpose and previous work

I will extend the preliminary APS experiment to the complete supplied training and test files. I want to establish how well COPOD ranks APS-related failures, whether its scoring reference matters, and whether heavy missingness affects the findings.

The earlier pilot used 20,000 training rows and 8,000 test rows, with ROC AUC 0.9780 and average precision (AP) 0.5016. These were standard PyOD batch-scoring results. I will retain them as preliminary evidence, not replace them with the full-data results.

The pilot already used a subset of the same official test file. This extension is therefore not an evaluation on a completely untouched test population, nor a retrospective claim of preregistration. I will not tune the new settings to improve their full-test scores.

The completed raw-data audit is recorded in [commit 5888583](https://github.com/Shuharto-Jannat/copod-reproducibility-project/commit/5888583) and `results/audits/aps_full_data/v1/`.

## 2. Questions and comparisons

| Question | Planned comparison |
|---|---|
| Does the COPOD reference distribution affect ranking? | Fixed training-reference COPOD versus standard PyOD batch COPOD, using identical preprocessing and test rows. |
| Is COPOD competitive with another unsupervised detector here? | Fixed training-reference COPOD versus Isolation Forest; both use training data only to establish their scoring rules. |
| Do heavily missing features affect the results? | Repeat all methods after excluding features with more than 50% missing training values. |

Standard batch COPOD will also appear alongside the baseline, but it has access to the test-feature distribution. I will identify this difference rather than treat that comparison as identical information access. No direction of improvement is assumed.

## 3. Data and labels

Source: [APS Failure at Scania Trucks, UCI](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks), DOI [10.24432/C51S51](https://doi.org/10.24432/C51S51).

| Supplied partition | Rows | Negative labels | Positive labels | Raw predictors |
|---|---:|---:|---:|---:|
| Training | 60,000 | 59,000 | 1,000 | 170 |
| Test | 16,000 | 15,625 | 375 | 170 |

I will preserve the supplied split, original row order and all rows. There will be no resplitting, class balancing or removal of positive training examples. Labels map `neg` to 0 and `pos` to 1. Positive means APS-related failure; negative means failure unrelated to APS, not necessarily a healthy truck.

Labels will be kept separate from predictors and used for evaluation and descriptive counts. Detector fitting, feature selection, imputation and scaling will not use them. Test prevalence is 375/16,000 = 0.0234375, or 2.34375%.

Before evaluation, source-file hashes must match the committed audit. The audit found no identical complete raw measurement profiles within or across files. This does not establish independence between trucks or rule out matches created by preprocessing.

## 4. Two fixed preprocessing conditions

### Primary: all usable predictors

1. Remove features with zero or one distinct observed training value. The audit identifies `cd_000`; 169 features are expected to remain.
2. Use `SimpleImputer(strategy="median", add_indicator=False)`, fitted on training predictors only. Apply those medians to both partitions.
3. Use `StandardScaler`, fitted on the imputed training predictors only. Apply the same transformation to the test predictors.
4. Record retained columns, exclusion reasons, missing-value counts, medians, and scaling parameters. Require finite prepared values and nonconstant training columns.

Scaling retains the general preprocessing approach of the pilot. It is not a claim that standardisation is intrinsically required for COPOD. There will be no additional log transform, clipping, missingness indicators or feature engineering in this experiment.

### Secondary: exclude heavily missing predictors

Apply the same rules, additionally removing features whose missing fraction in the original training file is strictly greater than 0.50. Exactly 50% missing is retained if otherwise usable. Refit imputation and scaling on the retained training columns.

The threshold means that a feature has more missing than observed training values. It is a fixed sensitivity choice, not an optimised threshold. Test missingness does not determine column selection. Both conditions will be reported, even if the secondary condition performs worse.

All methods within a condition receive the same prepared arrays. Repeat the exact-profile audit after column selection and after imputation/scaling. Record any new within-file duplicates, cross-file matches and label conflicts. Retain the official split and all rows in the primary analysis; any additional overlap-exclusion experiment would require a separately documented analysis.

## 5. Detectors and fixed settings

### A. Standard PyOD batch COPOD

Use the installed PyOD implementation, expected version 3.6.4, with `contamination=0.1` and `n_jobs=1`. Fit on training predictors, then call `decision_function` once on the entire 16,000-row test batch.

The inspected implementation combines stored training features with the supplied test features when computing empirical tails and skewness. I will describe this as batch/transductive scoring. Test labels do not enter the calculation. The contamination setting defines an unused classification threshold; the reported ranking metrics use raw scores.

### B. Modified fixed training-reference COPOD

Reuse `TrainingReferenceCOPOD` from `scripts/evaluate_synthetic_copod.py`. Only training values determine the empirical tails and skewness signs. For each feature, use inclusive counts of training values less than or equal to, and greater than or equal to, the queried value. Divide by 60,000 and clip probabilities to the interval `[1/60000, 1]` before applying the existing negative-log and feature-summing formula.

This is the project's modified scorer, not an exact reproduction of the authors' implementation. It receives the same training and test measurements as A; its scoring reference remains fixed when test batches change.

### C. Isolation Forest baseline

Use `sklearn.ensemble.IsolationForest` with these fixed parameters:

```python
n_estimators=200
max_samples=256
max_features=1.0
bootstrap=False
contamination="auto"
n_jobs=1
warm_start=False
```

Run random seeds 42, 43, 44, 45 and 46 in each preprocessing condition. Fit on all prepared training rows without labels. Use `-model.score_samples(X_test)` so larger values consistently indicate greater abnormality.

This is a bounded baseline comparison, not hyperparameter optimisation. Report every seed and the mean and sample SD of the five separate runs' metrics. Do not select the best seed or average the scores into an unplanned ensemble. Pair the same seeds when comparing preprocessing conditions. These are five random fits on the same data, not five independent datasets.

COPOD has no random fitting step in this setup: run A and B once per condition. Together, the two conditions contain four COPOD configurations and ten Isolation Forest fits.

## 6. Implementation and numerical checks

Before accepting results:

- Verify row identifiers, label alignment, feature order and source hashes.
- Record installed versions and inspected source hashes; do not silently upgrade packages between conditions.
- Compare the modified scorer's training scores against PyOD's stored `decision_scores_` immediately after fitting, before batch scoring overwrites diagnostic arrays. Use `rtol=1e-10`, `atol=1e-10`; record the maximum difference.
- Check that the fixed-reference test scores agree when computed as one batch, four consecutive chunks, and the first ten rows individually, using `rtol=0`, `atol=1e-12`.
- Validate inclusive-tail handling on a small artificial example containing ties. Imputation creates tied values, so this check matters for APS.
- Record warnings, non-finite contributions/scores, changes in skewness signs between training and combined data, and test values beyond each training feature's observed range. Count zero empirical tail probabilities actually replaced by the floor; probabilities already equal to `1/60000` are not replacements.
- Stop and investigate failed checks. Do not hide failures through broad warning suppression or replace invalid scores with `nan_to_num`. The known floor for zero empirical tail probabilities is part of the defined scorer.

These checks validate implementation behaviour; they do not require either method to achieve a particular performance score.

## 7. Performance measures and timing

Evaluate only the supplied test rows, using full-precision scores:

- **AP:** `sklearn.metrics.average_precision_score`; use the non-interpolated AP definition, not trapezoidal PR-curve area. Emphasise AP because positives are rare.
- **ROC AUC:** `sklearn.metrics.roc_auc_score`; include alongside AP to describe overall ranking discrimination.
- Show test prevalence 0.0234375 as an AP reference and 0.5 as the random-ranking ROC AUC reference. Neither reference is a classification threshold.
- Save ROC and precision-recall curves for primary COPOD results. Keep all Isolation Forest seed metrics available; do not choose a favourable seed for a headline comparison.

As a supplementary primary-condition diagnostic, report **Precision@375**, using the known number of positive test labels only to define this retrospective ranking measure. Where a score tie crosses the cutoff, give fractional credit equivalent to random selection within that tie: add `remaining_slots * positive_fraction_in_tie` to positives ranked strictly above it, then divide by 375. Save cutoff tie size and remaining slots. This explicit convention may differ from the original benchmark helper's tie handling.

Precision@375 is not an operational alert budget or a deployable threshold. I will not claim cost savings, diagnostic safety or deployment readiness from these metrics.

Record preprocessing, fitting and scoring times separately; exclude data loading, checks, plotting, file writing and bootstrap analysis. Note possible first-call compilation effects and hardware. Timings are descriptive implementation measurements, not a controlled replication of the paper's scalability study.

## 8. Conditional resampling analysis

For the two COPOD procedures under primary preprocessing, use 2,000 paired, class-stratified bootstrap resamples with seed 20261026. In each replicate, sample 375 positive row indices with replacement and 15,625 negative row indices with replacement. Use the same sampled indices for both procedures.

Keep fitted preprocessing and all original test scores fixed. Recalculate ROC AUC, AP and the **fixed-reference minus standard-batch** difference inside each replicate. Save the 2.5th and 97.5th percentiles, using NumPy's linear quantile convention, as **95% percentile resampling intervals for the observed score-label pairs**. Paired-difference intervals come from the replicate differences, not from subtracting separate interval endpoints.

This describes sensitivity to which observed cases receive more or less weight, at fixed class prevalence. It does not refit models or recompute batch COPOD's reference after resampling. It therefore omits training uncertainty, changed-batch scoring effects, population shifts and uncertainty about prevalence. Treating rows as independent is an assumption; the released anonymous measurements do not establish truck-level independence.

These are exploratory, conditional intervals, not guaranteed coverage for future full-batch performance. I will not turn them into confirmatory significance or equivalence claims. Isolation Forest's seed SD describes fitting randomness on the same test set and is not a confidence interval. The missingness sensitivity will report point-metric changes separately; it does not need another bootstrap analysis.

## 9. Evidence, interpretation and stopping point

Save full-data outputs separately under `results/aps_full_evaluation/v1/` and figures under `figures/aps_full_evaluation/`. Preserve the pilot, raw data and historical benchmark outputs.

Retain configuration and plan hashes, source-file and implementation hashes, environment versions, row references, preprocessing parameters, feature decisions, exact-profile checks, per-row test scores, per-run metrics, warnings, numerical diagnostics and bootstrap summaries. Keep raw APS files under the existing `data_new/` exclusion rule. Reuse existing outputs only after verifying their inputs and configuration; never overwrite differing evidence silently.

The report will present the primary comparison, the Isolation Forest results and the missingness sensitivity regardless of direction. A smaller observed difference does not establish that scoring procedures are equivalent. Any AP change must be interpreted with the same test prevalence and its limited operational meaning in mind.

This extension is complete when both preprocessing conditions, all five baseline seeds, implementation checks and the defined primary resampling analysis are saved and reviewed. I will not add parameter searches merely to obtain stronger scores. Other existing datasets retain their documented preliminary status unless separately extended; this APS experiment does not turn those pilots into full evaluations.

## 10. Reference documentation

- [UCI APS dataset and provenance](https://doi.org/10.24432/C51S51).
- [PyOD COPOD source documentation](https://pyod.readthedocs.io/en/latest/_modules/pyod/models/copod.html). Installed source/version recorded with the runs takes precedence over changing online documentation.
- [scikit-learn IsolationForest API and score direction](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).
- [scikit-learn average precision definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
- [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html), for paired resampling and percentile intervals. The explicit class-stratified index sampling above will be implemented separately.
