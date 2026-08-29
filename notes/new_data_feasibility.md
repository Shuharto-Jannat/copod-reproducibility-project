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