# New Data Feasibility Assessment

## Candidate 1: Bank Account Fraud Dataset Suite

### Source and provenance

- Dataset: Bank Account Fraud Dataset Suite (BAF)
- Creators: Sérgio Jesus and collaborators at Feedzai
- Research paper: *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation*
- Publication venue: NeurIPS 2022
- Official repository: https://github.com/feedzai/bank-account-fraud
- Dataset download: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
- Licence: CC BY-NC-SA 4.0
- Dataset version inspected: `Base.csv`

The dataset is synthetic and privacy-preserving but was generated to reproduce patterns found in a real bank-account-opening fraud dataset.

### Availability and structure

The dataset was successfully downloaded from Kaggle. The base version contains:

- 1,000,000 observations;
- 31 predictor variables;
- one binary target variable named `fraud_bool`;
- 988,971 legitimate applications;
- 11,029 fraudulent applications;
- a fraud rate of 1.1029%.

The five categorical predictors are:

- `payment_type`;
- `employment_status`;
- `housing_status`;
- `source`;
- `device_os`.

These require numerical encoding before COPOD can be applied.

No actual null values were detected. However, several numerical variables use `-1` to represent unavailable or not-applicable information. These variables include previous-address history, bank-account age, current-address history, session length, credit-risk score and device email counts. Their treatment must follow the dataset documentation.

### Preliminary COPOD smoke test

A preliminary smoke test was completed using the official PyOD implementation of COPOD.

Experimental setup:

- stratified sample of 20,000 observations;
- random seed 42;
- 60% training and 40% testing;
- categorical variables converted using one-hot encoding;
- constant predictors removed;
- predictors standardised;
- 12,000 training observations;
- 8,000 testing observations;
- 88 fraud cases in the test data;
- contamination rate of 1.105%.

Results:

- ROC AUC: 0.5467;
- average precision: 0.0149;
- execution time: 1.0088 seconds;
- encoded predictors used: 51.

COPOD completed without errors and was computationally fast. However, its preliminary detection performance was only slightly better than random ranking. This may indicate that BAF fraud cases are defined by complex combinations of otherwise ordinary feature values rather than extreme marginal values.

These results are preliminary and do not constitute the final evaluation. The final project would use repeated trials, uncertainty estimates, the corrected reproduction implementation and comparisons with baseline methods.

### Novelty check

BAF was not used in the original COPOD paper and does not appear in the ADBench dataset list. A targeted search combining COPOD with the dataset name did not identify a clearly documented published COPOD evaluation. The appropriate claim is therefore that no published or publicly documented COPOD evaluation was identified during the preliminary search, rather than that COPOD has certainly never been applied privately.

### Feasibility conclusion

BAF is feasible and is a strong candidate for the project because:

- it is publicly accessible and academically documented;
- it has an explicit licence;
- it contains labelled rare fraud events;
- it is substantially larger than the original COPOD benchmarks;
- it tests scalability and performance in a realistic fintech setting;
- COPOD runs successfully on a representative sample;
- its weak preliminary performance creates a meaningful generalisability question.

The dataset should be retained as a candidate for the final project.

## Candidate 2: APS Failure at Scania Trucks

### Source and provenance

- Dataset: APS Failure at Scania Trucks
- Creator: Scania CV AB
- Donors: Tony Lindgren and Jonas Biteus
- Original release: 2016
- Official repository: https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks
- DOI: https://doi.org/10.24432/C51S51
- Download size: approximately 54 MB
- Original use: Industrial Challenge 2016 at the 15th International Symposium on Intelligent Data Analysis

The dataset contains operational data collected from heavy Scania trucks. The Air Pressure System generates pressurised air for functions including braking and gear changes. The positive class represents failures involving a particular APS component, while the negative class represents truck failures caused by unrelated components.

### Licence

The current UCI page identifies the dataset licence as CC BY 4.0. The description included in the downloaded archive states GNU General Public License version 3 or later. Both permit academic use and modification under their respective conditions, but this difference between the current repository metadata and the archived documentation will be reported explicitly. Appropriate attribution will be provided.

### Availability and structure

The official archive was successfully downloaded directly from UCI. It contains:

- `aps_failure_training_set.csv`;
- `aps_failure_test_set.csv`;
- `aps_failure_description.txt`.

The first 20 lines of each CSV contain documentation, so the table must be read using `skiprows=20`. Missing values are represented by `na`.

The training dataset contains:

- 60,000 observations;
- 59,000 negative cases;
- 1,000 positive APS failures;
- a positive-class rate of 1.6667%;
- 170 numerical predictors;
- 850,015 missing predictor values;
- missing values in 169 predictors.

The test dataset contains:

- 16,000 observations;
- 15,625 negative cases;
- 375 positive APS failures;
- a positive-class rate of 2.3438%;
- 170 numerical predictors;
- 228,680 missing predictor values;
- missing values in 169 predictors.

Approximately 8.3% of all predictor values are missing overall. Several individual predictors contain very high missingness. The predictor `cd_000` is constant or completely missing and must be removed.

### Proposed preprocessing

The proposed preprocessing procedure is:

1. retain the supplied training and testing separation;
2. convert `pos` to 1 and `neg` to 0;
3. identify unusable predictors using training data only;
4. remove constant or completely missing predictors;
5. calculate median values from the training data;
6. use the training medians to impute both training and testing data;
7. standardise both sets using statistics learned from the training data;
8. conduct a sensitivity test that removes predictors with very high missingness.

This prevents information from the test dataset influencing preprocessing decisions.

### Preliminary COPOD smoke test

A preliminary smoke test was completed using the official PyOD implementation of COPOD.

Experimental setup:

- stratified sample of 20,000 supplied training observations;
- stratified sample of 8,000 supplied testing observations;
- random seed 42;
- 333 positive cases in the training sample;
- 188 positive cases in the testing sample;
- training contamination rate of 1.665%;
- training-median imputation;
- standardisation using training statistics;
- 169 predictors after removing the constant predictor.

Results:

- ROC AUC: 0.9780;
- average precision: 0.5016;
- execution time: 2.2883 seconds.

COPOD completed without errors and produced strong preliminary performance. The average precision is substantially higher than the test positive-class prevalence of approximately 2.35%.

These results remain preliminary. The final project would include repeated trials, uncertainty estimates, sensitivity to missing-value handling, the corrected reproduction implementation and baseline comparisons.

### Novelty check

APS Failure was not used in the original COPOD paper and does not appear in the ADBench dataset list. A targeted search combining COPOD with the dataset name did not identify a clearly documented published COPOD evaluation. The appropriate conclusion is that no published or publicly documented COPOD evaluation was identified during the preliminary search.

### Feasibility conclusion

APS Failure is feasible and should be retained as a strong candidate because:

- it comes from an authoritative academic repository;
- it provides real operational data and explicit labels;
- rare equipment failure is a natural anomaly-detection problem;
- it offers a domain distinct from the original benchmark collection and BAF;
- its missing-data challenge enables meaningful robustness analysis;
- COPOD executed quickly and achieved strong preliminary performance;
- it creates an informative contrast with COPOD’s weak preliminary BAF results.

The dataset should be retained as a candidate for the final project.