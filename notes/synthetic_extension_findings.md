# Synthetic extension: findings and limitations

Written: 30 October 2026. Experiment version: v1.

Evidence snapshot: [project commit 0efbded](https://github.com/Shuharto-Jannat/copod-reproducibility-project/commit/0efbdede3c62269bbb7797f131c6057f88aeb391).

Design: `notes/synthetic_extension_plan.md`, committed as [39a2cb6](https://github.com/Shuharto-Jannat/copod-reproducibility-project/commit/39a2cb60318ecd73f9e017f03a94adae25a2678b) before the extension experiments. The original five-seed pilot and historical benchmark results are preserved.

## 1. Purpose and completed work

I extended the synthetic pilot to investigate how anomaly strength affects COPOD, whether a method that accounts for relationships between features can identify the constructed signals, and whether COPOD's scoring reference materially changes the observed results.

I completed all seven planned settings for data seeds 100–109: 70 dataset realizations and 210 method-condition-seed results. The summary includes every planned result. No seeds, settings or methods were selected or removed because of their performance. I reviewed construction plots for the preselected seed 100 and retained numerical construction checks for all ten seeds.

This extension was informed by the earlier pilot. Its design was fixed before these new runs; the whole project was not preregistered before any results were seen. The extension is now complete under its stated scope.

## 2. What I constructed

Every realization contains 10,000 rows and 20 numerical features, arranged into four groups of five. There are 9,500 normal-origin rows and 500 designated positive-origin rows. The features are artificial measurements with no assigned physical meaning, and their parameters were not estimated from APS or another real dataset.

I used NumPy random streams to construct ordinary feature values through a shared group factor:

```text
feature = sqrt(rho) * group_factor + sqrt(1 - rho) * individual_noise
```

The factors and noise are independent standard-normal draws before sharing the group factor. Each ordinary feature has population mean 0 and variance 1. Features in the same group have correlation `rho`; different groups have zero population correlation. Normal and positive-origin base rows use separate random streams.

| Family | Settings | Construction of the positive group | What changes |
|---|---|---|---|
| Single-feature shifts | delta = 0, 2, 4, 6 | Start with the same correlated Gaussian process as the normal group, then shift one randomly selected feature in each positive row by +delta or -delta | Magnitude of the one-feature intervention; normal correlation stays 0.8 |
| Dependence violations | rho = 0, 0.4, 0.8 | Generate independent standard-normal features for positive rows | Strength of the normal group's relationships; positive features remain independent |

Within each family and seed, settings share the planned base random draws, row identities, ordering and split. The marginal settings also share the chosen feature and shift sign. This pairing allows comparisons of the intervention with less variation from unrelated changes in the rows. The two families use separate streams and separate null controls. The 70 realizations are therefore not 70 mutually independent datasets.

The shift intervention changes both individual distributions and relationships. Each positive feature has population variance `1 + delta^2/20`, and within-group positive correlation becomes `0.8 / (1 + delta^2/20)`. It is not a change to individual distributions that preserves every dependence property.

In the dependence family, each individual feature has the same standard-normal population distribution in both groups. The groups differ in their joint relationships when rho is positive. Sample means and other summaries need not match exactly, and matching individual distributions does not force an aggregated detector score to have identical distributions in both groups.

At delta 0 and rho 0, the two labelled groups have the same generating distribution within their respective family. Their 500 positive labels identify a designated comparison group, not deliberately introduced anomalies. In every setting, labels come from the construction process rather than from inspecting detector scores; a positive label does not guarantee that an individual row is easy to separate.

All 70 construction records passed the integrity and pairing checks. They recorded no missing or non-finite feature values, no exact duplicate feature profiles within a realization, and no shared training/test feature profiles within a realization. Checks also verified annotations, allowed coordinate changes, class counts and saved splits. Empirical distribution summaries were compared descriptively with their population targets; samples were not redrawn to force those summaries or detector performance to match an expectation.

## 3. Evaluation and implementation checks

I used the saved stratified 60/40 split with split seed 42: 6,000 training rows, including 300 positives, and 4,000 test rows, including 200 positives. All methods used the same rows within a condition. Labels supported construction checks, stratification and evaluation; they did not enter model fitting or parameter selection. No scaling, imputation, clipping or feature selection was applied.

| Method | Scoring reference and fitted information |
|---|---|
| Batch COPOD | PyOD 3.6.4, with contamination 0.05 and one job. The entire test batch is scored together, using training and test features to calculate tails and skewness. |
| Fixed-reference COPOD | The project's modified scorer, using training-only inclusive tail counts and skewness, with probabilities bounded below by 1/6,000. |
| Ledoit-Wolf distance | Training-only estimated location and shrunk covariance; squared Mahalanobis distance is the anomaly score. The true generating covariance is not supplied. |

Both training-only methods were fitted using all 6,000 training rows, including the 300 positives. Ledoit-Wolf was not fitted on a label-selected normal-only subset. Its parameters were fixed in the plan, with no tuning against test labels. Batch COPOD has access to test measurements; the principal detector comparison is therefore Ledoit-Wolf versus fixed-reference COPOD, which share training-only reference information.

I calculated ROC AUC and non-interpolated average precision (AP) from raw scores and known test labels. ROC AUC measures how well scores rank positives above negatives; AP evaluates how effectively the ranking concentrates positives near the top. These are ranking measures, not the percentage correctly classified at a chosen threshold. The reference values are ROC AUC 0.5 and AP prevalence 0.05.

All 70 saved numerical checks passed. There were no recorded model warnings or non-finite contributions or scores. The largest training-score difference between modified and standard COPOD was about 2.84e-14, within the declared tolerance. Fixed COPOD and Ledoit-Wolf scores were unchanged in the tested full-batch, four-chunk and ten-singleton checks. The covariance-distance formula also passed its separate checks.

The fixed scorer's predeclared probability floor was used when a test value lay beyond the training reference. Across realizations, 15–58 feature-tail cells required zero-tail replacement, affecting 14–47 test rows per realization. These cell counts are distinct from row counts and from probabilities already equal to the floor. Recorded skewness-sign differences between training-only and combined references affected 0–9 features per realization. These diagnostics describe differences between the scoring procedures; they do not establish which component caused a particular metric difference.

## 4. Performance across ten seeds

The following values are means and sample standard deviations across ten data seeds, rounded to four decimals. Calculations and saved comparisons retain full precision. SD describes variation under this simulator and split procedure; it is not a confidence interval or an estimate of deployment uncertainty.

### Single-feature shifts

| Setting | Method | ROC AUC: mean ± SD | AP: mean ± SD |
|---|---|---:|---:|
| Shift 0 (null) | Batch COPOD | 0.5025 ± 0.0196 | 0.0519 ± 0.0044 |
| Shift 0 (null) | Fixed-reference COPOD | 0.5015 ± 0.0180 | 0.0512 ± 0.0038 |
| Shift 0 (null) | Ledoit-Wolf distance | 0.5023 ± 0.0186 | 0.0533 ± 0.0049 |
| Shift 2 | Batch COPOD | 0.5911 ± 0.0178 | 0.0655 ± 0.0060 |
| Shift 2 | Fixed-reference COPOD | 0.5879 ± 0.0148 | 0.0644 ± 0.0050 |
| Shift 2 | Ledoit-Wolf distance | 0.9191 ± 0.0103 | 0.5834 ± 0.0344 |
| Shift 4 | Batch COPOD | 0.7087 ± 0.0207 | 0.0954 ± 0.0123 |
| Shift 4 | Fixed-reference COPOD | 0.7062 ± 0.0171 | 0.0946 ± 0.0108 |
| Shift 4 | Ledoit-Wolf distance | 1.0000 ± 0.0000 | 0.9997 ± 0.0003 |
| Shift 6 | Batch COPOD | 0.7206 ± 0.0184 | 0.1008 ± 0.0116 |
| Shift 6 | Fixed-reference COPOD | 0.7182 ± 0.0171 | 0.1004 ± 0.0103 |
| Shift 6 | Ledoit-Wolf distance | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

COPOD's mean performance improved as shifts increased, but improvement from delta 4 to delta 6 was modest. For example, batch COPOD's mean ROC AUC rose from 0.7087 to 0.7206, while AP rose from 0.0954 to 0.1008. This describes the tested levels, not a demonstrated hard performance ceiling or a guarantee of improvement in every seed.

Ledoit-Wolf identified these signals much more effectively. At delta 4, its unrounded mean ROC AUC was approximately 0.999984, so the table's rounded 1.0000 does not mean every trial was perfect. At delta 6, both ranking metrics were exactly 1 in all ten test sets. Those are perfect rankings for this constructed setting, not evidence of perfect classification or real-world performance.

A single displaced feature can also break the expected relationship with its group. The covariance-based comparator is suitable for this aspect of the construction. I did not separately isolate the contribution of each mechanism to its advantage.

### Dependence violations

| Setting | Method | ROC AUC: mean ± SD | AP: mean ± SD |
|---|---|---:|---:|
| Correlation 0 (null) | Batch COPOD | 0.5091 ± 0.0278 | 0.0536 ± 0.0062 |
| Correlation 0 (null) | Fixed-reference COPOD | 0.5077 ± 0.0242 | 0.0527 ± 0.0055 |
| Correlation 0 (null) | Ledoit-Wolf distance | 0.5072 ± 0.0228 | 0.0521 ± 0.0043 |
| Correlation 0.4 | Batch COPOD | 0.5222 ± 0.0192 | 0.0501 ± 0.0028 |
| Correlation 0.4 | Fixed-reference COPOD | 0.5232 ± 0.0230 | 0.0500 ± 0.0029 |
| Correlation 0.4 | Ledoit-Wolf distance | 0.7628 ± 0.0226 | 0.2447 ± 0.0252 |
| Correlation 0.8 | Batch COPOD | 0.5506 ± 0.0198 | 0.0503 ± 0.0020 |
| Correlation 0.8 | Fixed-reference COPOD | 0.5507 ± 0.0199 | 0.0504 ± 0.0021 |
| Correlation 0.8 | Ledoit-Wolf distance | 0.9967 ± 0.0018 | 0.9763 ± 0.0083 |

COPOD remained weak in these dependence settings. At rho 0.8, its ROC AUC was around 0.55, while AP remained near the 0.05 prevalence reference. It therefore showed some ranking discrimination, but little separation by AP. Calling every metric exactly random would overstate the finding.

Ledoit-Wolf improved strongly as normal correlation increased, reaching mean ROC AUC 0.9967 and AP 0.9763 at rho 0.8. This demonstrates that the constructed relationship signal was recoverable by the chosen covariance-based method, even though COPOD performed weakly on it. It does not establish that COPOD can never detect a dependence anomaly.

### Null controls

Across the two controls and three methods, mean ROC AUC ranged from about 0.5015 to 0.5091 and mean AP from 0.0512 to 0.0536. These results are consistent with the absence of a deliberately introduced signal. Individual trials fluctuate, and finite-sample AP need not equal 0.05 exactly. I retained both controls separately and did not use closeness to these references as a rule for selecting seeds.

## 5. Paired comparisons

Each difference was calculated within the same seed before its mean and sample SD were computed. I did not subtract arm standard deviations or treat paired conditions as independent replicates.

For the prespecified principal comparison at rho 0.8, Ledoit-Wolf minus fixed-reference COPOD gave:

| Metric | Mean paired difference | Sample SD of differences | Seeds with a positive difference |
|---|---:|---:|---:|
| ROC AUC | +0.445921 | 0.018696 | 10 of 10 |
| AP | +0.925907 | 0.007094 | 10 of 10 |

Ledoit-Wolf also exceeded fixed-reference COPOD on both metrics in every seed of each other non-null setting. All null and non-null comparisons remain in the saved tables; the principal setting was specified before evaluation.

The two COPOD procedures produced small mean differences. Across the seven settings, fixed minus batch mean differences ranged from approximately -0.003196 to +0.000978 for ROC AUC and -0.001050 to +0.000039 for AP. Individual paired differences could be larger and had both signs. A training-only reference gave no consistent performance improvement here, and it did not resolve COPOD's weakness on the relationship settings. This is not an equivalence finding and does not remove the conceptual difference in access to test features.

Comparisons with each family's own null control tell the same story. At rho 0.8, fixed COPOD's paired mean change was +0.0430 for ROC AUC and -0.0023 for AP, while Ledoit-Wolf's changes were +0.4894 and +0.9242. Negative AP changes were retained rather than hidden. They describe these finite samples; they do not prove that dependence systematically reduces AP.

No significance tests, equivalence tests or bootstrap intervals were performed for this extension. The figures display all ten seed values and mean ± sample SD, with zero or prevalence references labelled. Different scales in the method-difference figure are explicitly identified, and SD bars are not clipped at metric boundaries.

## 6. What the results support and what remains untested

The extension supports a bounded conclusion: in the tested Gaussian settings, COPOD responded more effectively to stronger single-feature shifts than to broken relationships with unchanged individual population distributions. A training-only covariance-based comparator recovered these constructed signals much more effectively. Changing COPOD's reference distribution alone did not change that overall pattern.

The following limitations must accompany this conclusion:

- Sample size, dimensionality, positive proportion and split procedure were fixed: 10,000 rows, 20 features, 5% positives and a 60/40 split with split seed 42. Ten data seeds describe variation within that design.
- Ledoit-Wolf was deliberately selected as suitable for these Gaussian relationships. These experiments do not establish a generally best anomaly detector or robustness to arbitrary distributions.
- Training sets contained positives, so fitted references and covariance estimates could be influenced by them. A normal-only training experiment was not performed.
- The broader proposal's skewed/heavy-tailed distributions, dimensions 5 and 100, positive proportions 1% and 10%, local anomalies and small anomalous clusters remain untested. The final report must acknowledge this narrowed scope.
- The synthetic features are not calibrated to Scania APS measurements. These results do not establish direct transfer to APS or other applications; the real-data results provide separate evidence.
- Saved fit and scoring times are descriptive records, not a controlled speed benchmark. No speed ranking or scaling claim follows from this extension.

## 7. Evidence and reproducibility

The complete numerical sources are under `results/synthetic_extension/v1/summary/`: `all_metrics.csv`, `performance_summary.csv`, the method and strength paired-difference tables and their summaries, `numerical_diagnostics.csv`, `verified_inventory.csv`, `warning_log.csv` and `summary_metadata.json`.

Per-seed directories retain construction checks, annotations-related metadata, saved row splits, test labels and scores, covariance diagnostics, and source/configuration hashes. Predictor and annotation CSVs are generated under `data_new/synthetic_extension/v1/` and are excluded from Git by the existing dataset rule. The generator and recorded seed/configuration information provide their regeneration path; validation hashes connect them to the saved evaluation evidence.

The recorded evaluation environment was Python 3.12.1, NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.0, scikit-learn 1.9.0, PyOD 3.6.4 and statsmodels 0.14.6. Detailed versions, implementation hashes and random-stream metadata are retained in the manifests.

The reproducibility sequence, from the repository root in the recorded environment, is:

```bash
python scripts/generate_synthetic_extension.py --seeds 100 101 102 103 104 105 106 107 108 109
python scripts/plot_synthetic_extension_diagnostics.py
python scripts/evaluate_synthetic_extension.py --seeds 100 101 102 103 104 105 106 107 108 109
python scripts/summarise_synthetic_extension.py
python scripts/plot_synthetic_extension_results.py
```

These commands document how to generate or verify the evidence; no additional run is needed to write this note. Reuse requires matching configuration and hashes. Differing existing evidence is not silently overwritten.

The three comparative figures are under `figures/synthetic_extension/v1/results/`, each in PNG and PDF:

- `ten_seed_performance`: performance against shift magnitude and correlation.
- `ten_seed_method_differences`: the two planned method comparisons.
- `ten_seed_strength_differences`: each non-null setting compared with its own family's control.

Separate captions are in `figure_captions.md`; exact plotting values and checks are under `results/synthetic_extension/v1/plot_data/`. The plotting script verified all nine summary tables and all 210 results, then checked the plotted values and axis bounds. I reviewed all three figures. The observed font fallback changed semibold titles to bold; it did not change the data or metrics.

## 8. Implication for the final report

I can report a completed, controlled extension that adds signal-strength experiments, null controls, paired comparisons and a covariance-based method to the original pilot. Its contribution is evidence about when the tested scoring approaches succeed or struggle, supported by reproducible construction and validation. The report should present the observed advantages and weaknesses alongside the design's scope, rather than treating favourable comparator scores as a universal ranking of methods.
