# APS full-data evaluation: findings

Written: 29 October 2026.

Evidence snapshot: [project commit 1c8638b](https://github.com/Shuharto-Jannat/copod-reproducibility-project/commit/1c8638bb26daff0de3d3f1b3e0b3f74e866d1ce1). This snapshot includes the full evaluation, summaries, paired resampling and figures.

The protocol was recorded in `notes/aps_full_evaluation_plan.md` and committed as [f005e70](https://github.com/Shuharto-Jannat/copod-reproducibility-project/commit/f005e7041ab47449e2f7fe24a60c70452d6ed247) before the new full-data model runs. This note interprets the saved results; it does not change the protocol or rerun models.

## 1. Purpose and scope

I extended the preliminary APS Failure experiment to the complete supplied training and test files. I investigated three questions: whether COPOD's scoring reference affects its rankings, how it compares with Isolation Forest, and what happens when heavily missing predictors are removed.

APS is the detailed real-world case study following my preliminary assessment of five datasets. Bank Account Fraud, PhiUSIIL, RT-IoT2022 and NATICUSdroid retain their preliminary status. Their sampling and preprocessing differ, so their scores should not be treated as a direct competition between datasets.

The APS pilot already used a subset of the official test file. Although I fixed the full-data protocol before these new runs, this extension is not confirmation on a completely untouched test population. The earlier pilot results and historical benchmark files remain unchanged.

## 2. Data and preprocessing

Source: [APS Failure at Scania Trucks, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks), DOI [10.24432/C51S51](https://doi.org/10.24432/C51S51). The archived dataset description defines positive cases as failures associated with the air-pressure system (APS). Negative cases are failures unrelated to APS; they are not necessarily healthy trucks.

| Supplied partition | Rows | Negative cases | Positive cases | Raw predictors | Missing measurement cells |
|---|---:|---:|---:|---:|---:|
| Training | 60,000 | 59,000 | 1,000 | 170 | 850,015 |
| Test | 16,000 | 15,625 | 375 | 170 | 228,680 |

I retained the supplied split, original row order and all rows, including positive training cases. Labels were kept separate from predictors and used for descriptive checks and evaluation, not detector fitting, feature selection, imputation or scaling. The test positive proportion was 375/16,000 = 0.0234375, or 2.34375%.

The two preprocessing conditions were fixed in advance:

- **Primary:** exclude `cd_000`, which has only one distinct observed training value, leaving 169 predictors.
- **Drop high missingness:** additionally exclude eight predictors with strictly more than 50% missing training values, leaving 161 predictors. These were `ab_000`, `bm_000`, `bn_000`, `bo_000`, `bp_000`, `bq_000`, `br_000` and `cr_000`.

For each condition, I fitted median imputation and standard scaling on training measurements and applied those same settings to test measurements. There were no missingness indicators or additional transformations. All methods within a condition received the same prepared arrays. The missingness threshold was not selected by optimising test performance.

## 3. Methods and evaluation

| Method | Measurements used to establish the scoring reference | Runs per condition |
|---|---|---:|
| Standard PyOD batch COPOD | Training measurements plus the entire supplied test batch | 1 |
| Modified fixed training-reference COPOD | Training measurements only | 1 |
| Isolation Forest | Training measurements only | 5 |

The COPOD implementation was PyOD 3.6.4. Standard batch scoring used `contamination=0.1`, `n_jobs=1` and one call on the entire test batch. Test measurements influence its empirical tails and skewness; test labels do not. The contamination setting determines an unused classification threshold. Evaluation used raw scores.

The modified scorer reused `TrainingReferenceCOPOD` from the synthetic experiment. It used inclusive training-only tail counts and training skewness, with probabilities bounded below by `1/60000`, before applying the COPOD score formula. It is a project modification, not an exact reproduction of the authors' implementation.

Isolation Forest used 200 trees, `max_samples=256`, `max_features=1.0`, `bootstrap=False`, `contamination="auto"`, `n_jobs=1` and `warm_start=False`. Seeds 42-46 were run separately; larger anomaly scores were obtained using `-score_samples`. I did not select a best seed or combine the runs into an ensemble.

Both COPOD procedures are deterministic in this setup. Altogether, I evaluated four COPOD configurations and ten Isolation Forest fits. All performance measures used the supplied test rows only. ROC AUC describes ranking discrimination; average precision (AP) summarises precision across increases in recall and is particularly relevant with rare positives. AP was calculated using scikit-learn's non-interpolated definition, not trapezoidal precision-recall area. Neither measure is classification accuracy.

## 4. Observed performance

Values are rounded to six decimal places. Isolation Forest entries show mean +/- sample standard deviation (SD) across five model seeds. COPOD has one deterministic result per condition, so a between-run SD is not applicable.

| Condition | Method | ROC AUC | Average precision |
|---|---|---:|---:|
| Primary: 169 predictors | Standard batch COPOD | 0.977690 | 0.531741 |
| Primary: 169 predictors | Fixed training-reference COPOD | 0.977821 | 0.534231 |
| Primary: 169 predictors | Isolation Forest | 0.985060 +/- 0.000313 | 0.585954 +/- 0.011787 |
| Drop high missingness: 161 predictors | Standard batch COPOD | 0.976845 | 0.529411 |
| Drop high missingness: 161 predictors | Fixed training-reference COPOD | 0.976990 | 0.532017 |
| Drop high missingness: 161 predictors | Isolation Forest | 0.984477 +/- 0.000418 | 0.569867 +/- 0.008575 |

Both COPOD procedures ranked APS-related failures well above the random-ranking ROC AUC reference of 0.5, and their AP values were well above the test-prevalence reference of 0.0234375. However, ROC AUC near 0.978 does not mean 97.8% classification accuracy, and AP near 0.53 does not mean that 53% of all failures were detected.

Every Isolation Forest seed achieved higher ROC AUC and AP than both COPOD procedures within each condition. The comparison with fixed-reference COPOD is especially useful because both methods establish their scoring rules using training data only. Standard batch COPOD also uses test features, so its information access differs.

These results favour the specified Isolation Forest configuration on this supplied split. They do not establish general superiority across other datasets or tuned configurations. The five-seed SD measures fitting randomness on the same data; it is not a confidence interval or variation across five independent test sets.

The earlier sampled pilot reported ROC AUC 0.9780 and AP 0.5016. The full-data results use different sets of training and test observations and a different test positive proportion, so changes from the pilot cannot be attributed solely to increased training size.

## 5. Effect of COPOD's scoring reference

Under primary preprocessing, using the fixed training reference increased ROC AUC by 0.000130901 and AP by 0.002490037 relative to batch scoring. The secondary condition also showed small positive differences: ROC AUC +0.000145237 and AP +0.002606587.

I completed the planned 2,000 paired, class-stratified resamples of the primary COPOD score-label pairs, using seed 20261026. Each replicate sampled 375 positives and 15,625 negatives with replacement, using identical row indices for both methods. Paired-difference intervals were calculated from the replicate differences, with linear percentile interpolation.

| Primary comparison: fixed reference minus batch | Observed difference | Lower 2.5th percentile | Upper 97.5th percentile |
|---|---:|---:|---:|
| ROC AUC | 0.000130901 | 0.000101193 | 0.000165043 |
| Average precision | 0.002490037 | 0.001395187 | 0.003892862 |

These are **95% percentile resampling intervals for the observed score-label pairs**. Both difference intervals are above zero, indicating that the small observed advantage persists under this conditional reweighting of the observed cases. The magnitude remains small; I do not interpret this as proof of practical superiority or equivalence.

The analysis kept fitted preprocessing, fitted models, original test scores, the original batch reference and class prevalence fixed. It did not refit models or recompute batch scores after resampling. It therefore omits uncertainty from retraining, different test-batch composition affecting scores, changing prevalence and population shift. Rows were treated as independent resampling units, but truck-level independence is not established by the anonymous data. These are exploratory conditional intervals, not confirmatory significance claims or guaranteed coverage for future performance.

No bootstrap comparison with Isolation Forest, the secondary condition or Precision@375 was performed. Individual method intervals are also saved in `percentile_intervals.csv`; overlapping individual intervals do not replace the paired-difference analysis.

## 6. Missingness sensitivity and a supplementary ranking check

The following changes are secondary minus primary, using the same test rows. Isolation Forest differences pair identical model seeds before summarising.

| Method | Change in ROC AUC | Change in AP |
|---|---:|---:|
| Standard batch COPOD | -0.000845 | -0.002330 |
| Fixed training-reference COPOD | -0.000831 | -0.002213 |
| Isolation Forest: mean paired change | -0.000584 | -0.016087 |

Both metrics decreased for each COPOD procedure and for every paired Isolation Forest seed. Removing these eight heavily missing predictors therefore offered no performance benefit in this experiment. This does not establish the usefulness of each removed feature individually or isolate missingness as the cause of the changes. The two conditions were reported as planned, without choosing a new threshold from their results.

Under primary preprocessing, both COPOD procedures had Precision@375 = 0.549333: each placed 206 positive cases among its 375 highest-scoring test rows. Equal counts do not establish identical selected rows or rankings. Isolation Forest's mean Precision@375 was 0.583467, with sample SD 0.003477 across seeds.

Precision@375 used the known number of test positives to set the cutoff retrospectively. The implementation allows fractional credit for ties crossing the cutoff; all seven primary runs had cutoff tie size one. This measure is not a deployed alert threshold or an independently chosen operational budget.

## 7. Integrity, profile overlap and numerical behaviour

The raw audit found no identical complete measurement profiles within or across the supplied files. Column selection changed that result:

- Primary preprocessing produced one matching pair of negative training rows, but no shared training/test profile.
- Dropping heavily missing predictors also produced one profile shared by a negative training row and a negative test row.
- These matches already existed after column selection. Imputation and scaling did not introduce additional matches, and no matching profile had conflicting labels.

I retained all supplied rows and documented the secondary overlap. Matching retained measurements do not establish that two rows describe the same physical truck. Conversely, absence of exact matches does not establish independence. An analysis excluding the shared profile was not performed, so its effect on metrics is not quantified.

All defined implementation checks passed. Training-score agreement between PyOD and the modified scorer had maximum absolute difference approximately 1.14e-13. The modified scorer's recorded chunk and singleton checks had zero maximum difference. No warnings or non-finite final scores were recorded, and no skewness-sign changes occurred between training-only and combined-reference calculations.

Each condition recorded 87 zero-tail probability replacement events and 80 tail events already equal to the floor. These count feature-level tail calculations, not distinct rows. The replacements followed the specified probability floor; invalid final scores were not repaired with `nan_to_num`.

## 8. Figures and evidence for the report

Three figures are saved in PNG and PDF under `figures/aps_full_evaluation/v1/`:

1. `aps_performance_comparison`: both preprocessing conditions and all methods; Isolation Forest bars represent +/-1 sample SD across five seeds. The metric axes are zoomed, not full-range axes.
2. `aps_primary_roc_pr`: primary COPOD ROC and precision-recall curves. Their near overlap is consistent with the small metric differences. Reference lines represent random ranking and test prevalence.
3. `aps_copod_paired_differences`: fixed-reference minus batch differences and their conditional resampling intervals. The dashed zero line means no difference between the two metric values.

The plotted curves were checked against saved ROC AUC and non-interpolated AP values. Plotting produced no recorded warnings and did not rerun models or resampling. Figure metadata includes captions and provenance. For the final report, longer explanatory text can move into captions while preserving readable axes, legends and the definitions of uncertainty bars. The current validated figures should remain available as standalone evidence.

## 9. Interpretation and limitations

The central finding is that COPOD achieved strong ranking discrimination on this APS split, while the tested Isolation Forest configuration performed better. Keeping COPOD's comparison distribution fixed to training data produced a small positive metric difference and scores invariant to the tested changes in query batching. Excluding heavily missing features reduced performance in the specified sensitivity experiment.

The strength of this evidence is its documented comparison on the same supplied test rows, with training-fitted preprocessing, explicit scoring protocols, saved row scores and implementation checks. Its scope is one supplied split, fixed model settings and an APS test population partly examined during the earlier pilot. The results do not establish future operational performance, calibrated failure probabilities, cost savings or deployment readiness. Descriptive execution timings were recorded, but they do not reproduce the paper's controlled scalability experiments.

The planned full APS extension is complete. Its results should be presented alongside the benchmark investigations, four other preliminary dataset assessments and controlled synthetic experiments, with their different purposes and limitations kept explicit.

## 10. Reproducibility

The committed evaluation environment records Python 3.12.1, NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.0, scikit-learn 1.9.0 and PyOD 3.6.4. Detailed versions, input hashes and implementation hashes are retained in the run metadata. Separate summary, resampling and figure metadata record their own provenance.

The following commands document the completed workflow; they are not instructions to rerun it merely to save this note:

```bash
python scripts/audit_aps_full_data.py
python scripts/evaluate_aps_full_data.py --conditions primary
python scripts/evaluate_aps_full_data.py --conditions drop_high_missingness
python scripts/summarise_aps_full_results.py
python scripts/bootstrap_aps_full_results.py
python scripts/plot_aps_full_results.py
```

Raw data remain under the Git-ignored `data_new/aps_failure/` directory. Audit evidence is under `results/audits/aps_full_data/v1/`. The evaluation directory `results/aps_full_evaluation/v1/` contains condition-specific preprocessing records, model scores, metrics and metadata, plus:

- `summary/`: verified performance, paired missingness differences, profile overlap and numerical summaries.
- `paired_bootstrap_primary/`: 2,000 saved replicate metrics, six percentile-interval rows and resampling metadata.
- `curves/`: primary ROC/PR coordinates and curve validation.

The scripts check existing outputs and their provenance before reuse and refuse to silently overwrite differing evidence. Source data, the supplied split, earlier pilot results and historical benchmark results were preserved.
