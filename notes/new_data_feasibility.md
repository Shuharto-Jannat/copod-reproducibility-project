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

## Candidate 3: RT-IoT2022

### Dataset identity and source

RT-IoT2022 is a labelled network-traffic dataset collected from a real-time Internet of Things infrastructure. It combines benign activity from IoT devices with traffic generated using several network-attack techniques.

Official source:

- UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/942/rt-iot2022
- Dataset DOI: https://doi.org/10.24432/C5P338
- Introductory paper: *Quantized Autoencoder (QAE) Intrusion Detection System for Anomaly Detection in Resource-Constrained IoT Devices Using RT-IoT2022 Dataset* (Cybersecurity, 2023)

The dataset is distributed through the UCI Machine Learning Repository and is directly downloadable as a CSV file. The UCI repository page provides its licensing and citation information.

### Domain and anomaly definition

Each row describes a network flow using information such as:

- transport protocol and network service;
- source and destination ports;
- packet and payload counts;
- packet rates;
- header sizes;
- TCP flag counts;
- inter-arrival times;
- active and idle durations;
- network-window sizes.

The original `Attack_type` field contains twelve activity labels. The following three labels describe benign IoT-device activity:

- `MQTT_Publish`;
- `Thing_Speak`;
- `Wipro_bulb`.

The remaining nine labels represent attacks, including denial-of-service, ARP poisoning, network scanning, Slowloris, and SSH brute-force activity. For COPOD evaluation, benign activity is assigned label 0 and every attack category is assigned label 1.

### Inspection results

The downloaded CSV contains:

- 123,117 rows;
- 85 total columns;
- 83 predictive features after excluding `id` and `Attack_type`;
- two categorical predictors: `proto` and `service`;
- no missing values;
- no infinite numeric values;
- one constant predictor, `bwd_URG_flag_count`;
- 5,195 duplicate feature-and-label rows.

After removing the row identifier and duplicate records, 117,922 unique observations remain:

- 12,015 benign observations;
- 105,907 attack observations;
- an original attack rate of 89.8111%.

The released dataset is intentionally attack-heavy. Applying COPOD directly with nearly 90% contamination would conflict with the usual unsupervised anomaly-detection assumption that anomalies form a minority.

### Proposed preprocessing and evaluation construction

The feasibility test therefore constructs a reproducible evaluation sample by:

1. removing the non-predictive `id` column;
2. removing duplicate feature-and-label rows before splitting;
3. retaining all 12,015 unique benign observations;
4. sampling 1,335 attack observations using random seed 42;
5. guaranteeing at least one observation from each of the nine attack categories;
6. producing a controlled attack contamination rate of 10%;
7. one-hot encoding `proto` and `service`;
8. removing the constant predictor;
9. applying a stratified 60% training and 40% testing split;
10. standardising predictors using training-set statistics.

This construction is necessary to create an anomaly-detection setting while retaining diversity across all attack categories. The sampling procedure and random seed will be reported explicitly so the evaluation remains reproducible.

### COPOD smoke test

The preliminary COPOD smoke test used:

- 13,350 observations;
- 12,015 benign observations;
- 1,335 attack observations;
- 92 predictors after categorical encoding and constant-feature removal;
- 8,010 training observations;
- 5,340 testing observations;
- 10% training contamination.

Results:

- ROC AUC: 0.6891;
- average precision: 0.1360;
- execution time: 0.8727 seconds.

These values are preliminary feasibility evidence rather than final experimental results. COPOD completed without technical difficulty, although its modest performance suggests that the network-attack patterns are not always identifiable as marginal distributional outliers.

### Novelty assessment

RT-IoT2022 was released after the 2020 COPOD paper and therefore could not have appeared in its original evaluation. A targeted search did not identify a published or publicly documented COPOD evaluation on RT-IoT2022. This is reported cautiously rather than as proof that COPOD has never been applied privately or in inaccessible work.

### Feasibility conclusion

RT-IoT2022 should be retained as a candidate because:

- it is available from an official and citable repository;
- it contains labelled benign and attack observations;
- its features are compatible with COPOD after minimal categorical encoding;
- it has no missing or infinite values;
- COPOD runs in less than one second on the constructed sample;
- all nine attack categories can be represented;
- its cybersecurity domain differs substantially from the original COPOD benchmarks;
- its modest preliminary performance creates a meaningful generalisability question.

The main limitation is that the released dataset is dominated by attacks. Consequently, the controlled 10% attack sample and its construction must be disclosed clearly in the proposal and final report.

## Candidate 4: PhiUSIIL Phishing URL (Website)

### Dataset identity and source

PhiUSIIL is a labelled tabular dataset containing legitimate and phishing URLs together with characteristics extracted from each URL and its corresponding webpage.

Official source:

- UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
- Introductory paper: Arvind Prasad and Shalini Chandra, *PhiUSIIL: A Diverse Security Profile Empowered Phishing URL Detection Framework Based on Similarity Index and Incremental Learning*, published in *Computers & Security*.

The official UCI archive is directly downloadable and contains one CSV file. The repository page provides the required licensing and citation information.

### Domain and anomaly definition

Phishing websites imitate legitimate services to obtain passwords, payment information, or other sensitive data. Each observation represents one URL and its corresponding webpage.

The dataset contains:

- 134,850 legitimate websites, labelled 1;
- 100,945 phishing websites, labelled 0.

For COPOD evaluation, the labels are converted so that:

- legitimate website = 0, representing a normal observation;
- phishing website = 1, representing an anomaly.

The engineered features describe properties including:

- URL and domain length;
- character, digit and special-character ratios;
- URL obfuscation;
- HTTPS use;
- webpage source-code length;
- title and domain matching;
- redirects and pop-ups;
- forms, hidden fields and password fields;
- social-network and copyright information;
- counts of images, CSS, JavaScript and references.

### Inspection results

The downloaded CSV contains:

- 235,795 rows;
- 56 total columns;
- 54 reported features after excluding the identifier and target;
- no missing values;
- no infinite numeric values;
- no constant numeric predictors;
- no duplicate feature-and-label rows.

Five non-numeric fields have high cardinality:

- `FILENAME`: 235,795 unique values;
- `URL`: 235,370 unique values;
- `Domain`: 220,086 unique values;
- `TLD`: 695 unique values;
- `Title`: 197,874 unique values.

`FILENAME` is only an identifier. Directly one-hot encoding the other raw text fields would create a very large and sparse feature space that is unsuitable for this COPOD evaluation. These five columns are therefore removed. The dataset still provides 50 engineered numeric predictors capturing relevant properties of each URL and webpage.

### Proposed preprocessing and evaluation construction

Phishing websites form 42.8105% of the released dataset. This is useful for supervised classification but does not resemble the minority-anomaly setting assumed by COPOD.

The feasibility test therefore constructs a reproducible 20,000-observation sample containing:

- 18,000 legitimate websites;
- 2,000 phishing websites;
- a controlled phishing contamination rate of 10%.

The preprocessing procedure is:

1. randomly sample both classes using seed 42;
2. remove `FILENAME`, `URL`, `Domain`, `TLD`, and `Title`;
3. convert the original label so phishing observations equal 1;
4. retain the 50 engineered numeric predictors;
5. perform a stratified 60% training and 40% testing split;
6. standardise all predictors using training-set statistics;
7. fit COPOD using the training contamination rate.

The sampling method and modified class distribution will be disclosed explicitly. The constructed 10% rate is an experimental anomaly-detection setting and is not presented as the real prevalence of phishing websites.

### COPOD smoke test

The preliminary smoke test used:

- 20,000 observations;
- 50 numeric predictors;
- 12,000 training observations;
- 8,000 testing observations;
- 1,200 phishing cases in training;
- 800 phishing cases in testing;
- 10% training contamination.

Results:

- ROC AUC: 0.9452;
- average precision: 0.7369;
- execution time: 0.6424 seconds.

These preliminary results show that COPOD strongly separates phishing websites from legitimate websites in the constructed sample. They demonstrate feasibility but are not treated as final experimental findings.

### Proposed sensitivity analysis

Two engineered variables, `URLSimilarityIndex` and `TLDLegitimateProb`, may be especially informative. A later sensitivity analysis should repeat the evaluation without these variables. This will determine whether COPOD's performance reflects broader distributional differences or relies disproportionately on features designed specifically for phishing detection.

The final experiment should also evaluate multiple random samples and contamination levels so that the conclusion does not depend on a single constructed sample.

### Novelty assessment

PhiUSIIL was released after the 2020 COPOD paper and could not have appeared in its original evaluation. A targeted search did not identify a published or publicly documented COPOD evaluation using PhiUSIIL. This is stated cautiously and does not claim that no private or inaccessible application exists.

### Feasibility conclusion

PhiUSIIL should be retained as a strong candidate because:

- it is available from an official and citable repository;
- it has explicit legitimate and phishing labels;
- it contains a large number of observations;
- its 50 engineered numeric predictors are directly compatible with COPOD;
- it has no missing, infinite, constant or duplicate numeric data problems;
- the smoke test runs in less than one second;
- preliminary ROC AUC and average precision are strong;
- it provides a recent malicious-website domain not included in the original COPOD evaluation.

Its main limitations are the artificially controlled 10% phishing rate and the possibility that specialised engineered features make the task unusually easy. Both issues can be addressed through transparent sampling and sensitivity analysis.