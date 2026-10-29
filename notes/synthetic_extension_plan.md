# Synthetic extension: controlled anomaly-strength experiments

Written: 29 October 2026. Plan version: v1.

Status: design specified before generating or evaluating the extension datasets. I will commit this plan before those experiments. The completed five-seed synthetic pilot remains unchanged.

## 1. Purpose and connection to the pilot

The pilot compared unusual individual values with unusual relationships between otherwise ordinary values. Its two settings used a shift magnitude of 4 and within-block normal correlation of 0.8. COPOD performed better on the shift scenario and struggled on the dependence-violation scenario. Differences between its two scoring protocols were small and inconsistent across the five pilot seeds.

I will extend that work by changing the strength of the constructed signal, adding cases with no population-level difference between the labelled groups, and comparing COPOD with a method that uses covariance between features. The purpose is to explain behaviour under controlled conditions, not to create a scenario that guarantees a preferred detector will win.

This plan is informed by the completed pilot. It specifies new experiments in advance, but is not a claim that the entire project was preregistered before seeing any results. The artificial features have no assigned physical meaning. Their parameters are not estimated from APS or another real dataset.

## 2. Questions and fixed design

I will investigate:

1. How do scores perform as the magnitude of a one-feature shift increases?
2. How do scores perform as normal feature relationships become stronger while the generated positive group remains independent?
3. Does a training-only covariance-based method identify this relational signal more effectively than fixed-reference COPOD in the tested Gaussian settings?
4. How sensitive are COPOD results to using a training-only reference instead of training plus the supplied test batch?
5. Do null settings behave consistently with the absence of a deliberately introduced signal?

Each dataset will contain 10,000 rows, 20 numerical features and 500 designated positive rows. Features are arranged into four blocks of five. The seven fixed conditions are:

| Condition identifier | Family | Normal within-block correlation | Positive-group construction | Purpose |
|---|---|---:|---|---|
| `marginal_delta0` | Marginal shift | 0.8 | Same block-Gaussian process, shift 0 | Null control |
| `marginal_delta2` | Marginal shift | 0.8 | Shift one feature by +2 or -2 | Smaller shift |
| `marginal_delta4` | Marginal shift | 0.8 | Shift one feature by +4 or -4 | Pilot-strength setting |
| `marginal_delta6` | Marginal shift | 0.8 | Shift one feature by +6 or -6 | Larger shift |
| `dependence_rho0p0` | Dependence violation | 0.0 | Independent standard-normal features | Null control |
| `dependence_rho0p4` | Dependence violation | 0.4 | Independent standard-normal features | Moderate normal dependence |
| `dependence_rho0p8` | Dependence violation | 0.8 | Independent standard-normal features | Strong normal dependence |

Use ten new data seeds: **100, 101, 102, 103, 104, 105, 106, 107, 108 and 109**. These differ from pilot seeds 42-46. Seven conditions times ten seeds gives **70 dataset realizations**. Three detectors per realization gives **210 planned detector fits**. Conditions sharing a family and seed are intentionally paired, so the 70 realizations are not 70 mutually independent datasets.

Ten seeds are a fixed, manageable extension for describing simulation variation, not a power calculation or a guarantee of precise conclusions. I will not increase, reduce or select the seed set based on favourable results.

## 3. How the data will be generated

For a row and feature `j` in block `b`, construct an ordinary block-Gaussian value as:

```text
X[j] = sqrt(rho) * S[b] + sqrt(1 - rho) * E[j]
```

Here, each `S[b]` and `E[j]` is an independent standard-normal draw before sharing the block factor. Rows are generated independently. A block's five features share its `S[b]`; different blocks have independent factors and residuals. This gives each ordinary feature population mean 0 and variance 1, within-block correlation `rho`, and between-block correlation 0.

This factor construction has the same Gaussian block distribution as the pilot's covariance-matrix construction at matching settings. It uses a new generator and random streams, so it is not intended to reproduce the pilot's exact rows or metrics.

### Marginal-shift family

Generate 9,500 normal rows and 500 independent positive-origin base rows with `rho=0.8`. For each positive row, select one feature uniformly from the 20 features and independently choose a sign with equal probability. Add the sign times the condition's shift magnitude to that feature only.

The intervention changes both individual feature distributions and the positive group's correlations. With shift magnitude `delta`, each positive feature has population mean 0 and variance `1 + delta^2/20`. Within-block positive correlation becomes `0.8 / (1 + delta^2/20)`. Between-block covariance remains zero. I will therefore describe this as a single-feature shift intervention, not an experiment that changes only marginals while preserving all dependence properties.

A generated positive need not be extremely unusual in every finite realization. The recorded label identifies its construction process, not guaranteed separability.

### Dependence-violation family

Generate the 9,500 normal rows using the stated `rho`. Generate the 500 positive rows with independent standard-normal feature values. Both groups have standard-normal individual feature distributions in the population; only their joint structure differs when `rho` is positive. Finite samples will not have exactly identical marginal summaries.

Matching individual distributions does not force identical distributions of an aggregated detector score. I will not assume COPOD must achieve exactly chance performance in the non-null dependence settings.

### Null controls and labels

At shift 0, both groups follow the same correlated Gaussian distribution. At dependence `rho=0`, both groups follow the same independent Gaussian distribution. The label-1 rows in these two conditions are designated comparison groups without a deliberately introduced anomaly signal.

Labels will be 0 for the normal-origin group and 1 for the positive-origin group, including the null controls. The 500 positive labels are not assigned by examining feature values or detector scores. Null ROC AUC should fluctuate around the random-ranking reference of 0.5; AP will be interpreted relative to the 5% prevalence reference. Finite-sample AP is not required to have expectation exactly 0.05, and no individual run must equal either reference exactly.

## 4. Random streams and paired comparisons

Use NumPy `SeedSequence` with entropy `[data_seed, 20261029]` and explicit `Generator(PCG64(child_sequence))` instances. Spawn the two families in the fixed order `marginal_shift`, then `dependence_violation`. Within each family, spawn five separate streams in the order: normal latent draws, positive latent draws, shifted-feature selection, shift-sign selection and row permutation. Record entropy, spawn keys, draw order and NumPy version. The official [NumPy random-stream guidance](https://numpy.org/doc/stable/reference/random/parallel.html) documents this spawning approach.

Within each latent stream, draw the block-factor matrix first and the feature-residual matrix second. Their shapes are `(group_rows, 4)` and `(group_rows, 20)`. Normal and positive groups use separate streams. In the dependence family, only the positive residual matrix is used in the final positive features; the positive block factors are unused.

Within each family and seed, create the base random draws once and reuse them across strength settings:

- Marginal conditions share normal rows, positive base rows, selected features, signs and permutation. Only shift magnitude changes.
- Dependence conditions share normal factors/residuals, independent positive rows and permutation. Only the normal correlation parameter changes.
- Labels, source-row identities and train/test membership are identical across levels within that family and seed.

This pairing helps isolate the intervention from unrelated changes in sampled rows. Rows shared across conditions are deliberate counterfactual versions, not extra independent samples. The two families have separate random streams; their null datasets must not be pooled or treated as the same paired dataset merely because seed numbers match. All streams must be created with their fixed identities even if a command requests only some conditions.

The predictor CSV will contain only `x01` through `x20`, followed by `label`. A separate annotation file will record zero-based CSV row position, original source-row identity, family, condition, label, selected feature and applied signed shift. Use -1 and 0 for feature and shift when no feature is selected in the dependence family or in normal rows. For marginal positive rows at shift 0, retain the selected feature and sign separately even though the applied shift is zero. No annotation field enters a detector.

## 5. Shared training/test protocol

Use `train_test_split` on row indices with `test_size=0.4`, `stratify=labels` and integer `random_state=42`. Within each family and seed, split once and reuse the row membership across its conditions. This gives 6,000 training rows, including 300 designated positives, and 4,000 test rows, including 200 designated positives.

All three detectors receive exactly the same training and test features within a condition. Retain the 300 positive training rows; no method is fitted only on rows selected using known normal labels. Labels are used for stratification, construction checks and evaluation, not for fitting or selecting hyperparameters.

No imputation, scaling, clipping, feature selection or additional feature engineering will be applied. The normal/base features already have common population scales; the variance changes introduced by shifts are part of the intervention. This follows the pilot's no-preprocessing approach and differs from the real APS preprocessing pipeline.

## 6. Three fixed scoring methods

### Standard PyOD batch COPOD

Use `COPOD(contamination=0.05, n_jobs=1)` with a fresh instance for each dataset. Fit using training features and score the entire 4,000-row test set in one call. The inspected implementation uses training plus test features to calculate its empirical tails and skewness. Test labels are not used. The contamination setting determines an unused threshold; metrics use raw scores.

Record the installed version and source hash. The current project version is PyOD 3.6.4; I will not silently upgrade the implementation between planned runs.

### Modified fixed training-reference COPOD

Reuse `TrainingReferenceCOPOD` from `scripts/evaluate_synthetic_copod.py`, preserving its inclusive training-only tail counts, training skewness and lower probability bound `1/6000`. Score test features using the fixed training reference. This is a project modification, not the authors' unchanged implementation.

### Ledoit-Wolf squared Mahalanobis distance

Use `sklearn.covariance.LedoitWolf(store_precision=True, assume_centered=False, block_size=1000)`. Fit all 6,000 training feature rows without labels, then use `mahalanobis(X_test)` as the anomaly score. This returns squared distances; larger means more unusual. The training mean, covariance and shrinkage coefficient are estimated rather than supplied from the generator. The API is specified in the [official LedoitWolf documentation](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html).

This score measures distance while accounting for learned feature relationships. The comparator is deliberately suitable for the Gaussian dependence structure in this study. It is not a claim of universal superiority or robustness to arbitrary anomalies. The positive training rows can affect its estimates. No true population covariance, construction annotations or test measurements are used to fit it, and no shrinkage value is tuned against labels.

The main detector comparison is Ledoit-Wolf versus fixed-reference COPOD, because both use training-only scoring references. Batch COPOD has different information access. All three are deterministic under these settings; repeated results come from new generated datasets, not repeated identical fits.

## 7. Validation before accepting model results

The implementation will separate dataset generation/validation from detector evaluation. Save validation evidence for all 70 realizations before aggregating performance.

Require correct dimensions, ordered feature columns, exact class counts, valid annotations, finite values, disjoint train/test row indices and the expected split counts. Check exact duplicate feature profiles within each realization and across its train/test split. Investigate unexpected duplicates without silently deleting rows. Cross-condition reuse within paired families is expected and will be checked separately.

Verify that only the selected coordinate changes in each marginal positive row and that undoing its recorded shift recovers its base row within numerical tolerance. Verify the declared identities across paired conditions and reproducibility under the same seed/configuration. Confirm the two null cases use identical generating distributions for both labelled groups, without requiring identical realized rows.

Record class-specific feature means, standard deviations, quantiles and within-/between-block correlations, together with the theoretical expectations in section 3. These empirical summaries should be interpreted as random finite-sample estimates, not forced to equal their population targets. They will not be used to reject difficult seeds or redraw data until a preferred metric appears. Plot representative diagnostics for the preselected seed 100 across all seven conditions; retain numerical diagnostics for every seed.

Before accepting detector outputs:

- Compare the modified COPOD training scores with PyOD's stored fit scores using `rtol=1e-10`, `atol=1e-10`, before test scoring changes diagnostic arrays.
- Check fixed-reference COPOD scores in one batch, four consecutive chunks and the first ten test rows individually, with `rtol=0`, `atol=1e-12`.
- Check the Ledoit-Wolf score formula against a small hand-calculable example; compare its test scores across those same batching arrangements with `rtol=1e-10`, `atol=1e-12`.
- Record finite score/contribution checks, warnings, COPOD training/combined skewness signs and actual zero-tail replacements separately from probabilities already equal to the floor. Symmetric population distributions need not have exactly zero sample skewness.
- Save the covariance comparator's fitted location, covariance, precision, shrinkage, eigenvalue/conditioning diagnostics and source/version information.
- Verify that the scores and labels used for metrics refer to exactly the same saved test rows and that no identifiers or construction metadata enter any fit.

Do not suppress warnings broadly or repair non-finite final scores with `nan_to_num`. A failed integrity or implementation check requires investigation. A surprising performance score, including a null result away from its reference in one seed, is not by itself a failed run.

## 8. Metrics, summaries and comparisons

Evaluate ROC AUC and non-interpolated average precision on test rows only. Use `sklearn.metrics.roc_auc_score` and `average_precision_score`, keeping full precision for calculations. The references are ROC AUC 0.5 and AP prevalence 0.05. No thresholded accuracy, best-seed selection or outcome-dependent hyperparameter tuning is planned.

Report all 210 method-condition-seed results. For each of the 21 method-condition combinations, report the ten-seed mean and sample SD (`ddof=1`), plus individual seed values. These SDs describe variation under the specified simulator and split procedure; they are not confidence intervals or estimates of deployment uncertainty.

Prespecified paired contrasts are:

1. **Scoring reference:** fixed-reference COPOD minus batch COPOD, within each condition and seed.
2. **Detector:** Ledoit-Wolf minus fixed-reference COPOD, within each condition and seed. The strong-dependence condition `dependence_rho0p8` is the principal comparison for the relational mechanism; all seven conditions remain reportable.
3. **Intervention strength:** for each method, marginal shifts 2, 4 and 6 minus shift 0 within the same family/seed; dependence correlations 0.4 and 0.8 minus correlation 0 within the same family/seed.

Calculate each difference before summarising its ten values. Do not subtract standard deviations or treat all correlated conditions as independent replicates. Do not pool the two families' controls. No significance tests, equivalence tests or additional bootstrap intervals are planned for this bounded exploratory extension.

Show ROC AUC and AP against the varied parameter in separate family panels, using individual seed values and means with clearly labelled sample-SD bars. Use additional paired-difference tables or plots to support the predefined contrasts. Do not require a monotonic trend or a successful comparison method to accept the results.

## 9. Output locations and reproducibility

Use new scripts and versioned directories so the pilot is preserved:

- `scripts/`: extension generation/validation, evaluation, summary and plotting scripts.
- `data_new/synthetic_extension/v1/`: generated predictor/label CSVs and annotations, excluded from Git by the existing `data_new/` rule.
- `results/synthetic_extension/v1/`: configuration manifest, dataset/validation metadata, row splits, all test scores, per-run metrics, numerical checks and summaries.
- `figures/synthetic_extension/v1/`: construction diagnostics and comparative result figures in PNG/PDF.
- `notes/synthetic_extension_plan.md`: this design; a separate findings note will be written after the results.

Record the committed plan hash, configuration, seed-stream identities, generator/evaluator/scorer hashes, package versions, file checksums and row references. Require matching data and validation hashes before evaluation. Store all three scores together for each dataset's test rows. Save sufficient configuration and code to regenerate the ignored datasets; no hidden local-only changes should be needed by the marker.

Existing outputs may be reused only after their configuration and provenance are verified. Differing evidence must not be overwritten silently. If a real implementation error is discovered, document the correction and preserve the earlier evidence; use an explicit new version for affected reruns. Performance-driven changes would be separate exploratory work, not part of this fixed plan.

## 10. Relation to the original proposal and limits

The proposal outlined a broader factorial study spanning skewed and heavy-tailed marginals, dimensions 5/20/100, positive proportions 1%/5%/10%, local anomalies and small anomaly clusters. This extension deliberately narrows the scope to two interpretable Gaussian mechanisms, their strength settings, two null controls and a covariance-aware comparator.

The narrowed design makes it possible to pair underlying draws, check construction against known population properties and attribute comparisons to explicit interventions. It does not fulfil or establish results for the proposal's other dimensions. Those extensions remain untested and must be identified as scope changes or future work in the final report, not presented as completed experiments.

Other limitations include fixed sample size and positive proportion, synthetic rather than real-world feature meanings, training sets containing generated positives, and a comparator selected for its suitability to Gaussian relationships. These results cannot show that one detector is best for all anomaly types, that COPOD can never detect dependence anomalies, or that the findings transfer directly to APS. The broader real-data evaluation and reproduction investigations provide separate evidence.

## 11. Execution order and stopping point

1. Commit this plan before the new data/model runs.
2. Implement and check generation/validation on seed 100 for all seven conditions. Review construction evidence before evaluation.
3. Evaluate the three methods on that seed to check execution and saved evidence. Keep its results as part of the planned ten seeds; do not use them to change parameters or choose a favourable baseline.
4. Complete seeds 101-109 with the same configuration, validating each dataset before its evaluation.
5. Verify the complete 70-dataset/210-result inventory, produce the defined summaries and figures, and document findings and scope limits.

The extension is complete when the planned inventory and checks are accounted for and the evidence is saved and reviewed. Do not add seeds or scenarios merely to strengthen the headline. If an unresolved technical failure prevents completion, report the missing or invalid cases and the reason explicitly; do not silently summarise a selected subset as the planned experiment.
