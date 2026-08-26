# COPOD Reproduction Log

## Project

**Paper:** COPOD: Copula-Based Outlier Detection  
**Authors:** Li, Zhao, Botta, Ionescu, and Hu  
**Conference:** IEEE International Conference on Data Mining (ICDM), 2020

## Computing environment

- Platform: GitHub Codespaces
- Operating system: Linux
- Python version: 3.12.1
- Virtual environment: `.venv`
- Original code: included as the Git submodule `original_code/COPOD`

## Initial setup

The official COPOD repository was added as a Git submodule from:

`https://github.com/winstonll/COPOD.git`

A private Python virtual environment was created with:

```bash
python -m venv .venv
```

The required Python packages were installed inside this environment.

## Initial example execution

The example program examined was:

`original_code/COPOD/results/cod_example.py`

It loads the `breastw.mat` dataset and produces feature-level anomaly explanations for observations 69 and 97.

### First issue: execution directory

Running the script from the main project directory produced:

```text
ModuleNotFoundError: No module named 'models'
```

The script assumes that it is executed from the authors' `results` directory.

### Second issue: inconsistent class name

After running the script from the expected directory, it produced:

```text
ImportError: cannot import name 'COD' from 'models.cod'
```

The example imports a class named `COD`, but `models/cod.py` defines the class as `COPOD`.

This appears to be an inconsistency in the released research code.

### Temporary compatibility correction

The example was executed without permanently modifying the authors' code by temporarily replacing:

```python
from models.cod import COD
```

with:

```python
from models.cod import COPOD as COD
```

The command used was:

```bash
sed 's/from models.cod import COD/from models.cod import COPOD as COD/' cod_example.py | PYTHONPATH=.. python -
```

### Result

The example executed successfully and printed feature-level anomaly explanations for observations 69 and 97.

For each observation:

- indices 0–8 represent the dataset's nine features;
- the first set contains that observation's feature-level anomaly scores;
- the `0.9` set contains the 90th-percentile reference scores;
- the `0.99` set contains the 99th-percentile reference scores.

This verifies that the original COPOD implementation can execute in the current environment after addressing the class-name inconsistency. It does not yet reproduce the paper's complete benchmark results.

## BreastW benchmark smoke test

A reduced benchmark was performed on `breastw.mat` to verify that the released benchmarking pipeline works in the current environment.

For this smoke test:

- dataset: `breastw.mat`;
- samples: 683;
- dimensions: 9;
- outlier percentage: 34.9927%;
- iterations: 1 rather than the paper’s full 10 iterations;
- compared methods: ABOD, CBLOF, HBOS, IForest, KNN, LODA, LOF, OCSVM, PCA, and COPOD.

The released benchmark script required several compatibility changes:

1. `COD` was replaced with `COPOD as COD` because the released model class is named `COPOD`.
2. The dataset path was changed to point to `original_code/COPOD/data/`.
3. The unavailable `cover.mat` dataset was excluded.
4. The test was temporarily restricted to `breastw.mat` and one iteration.

COPOD produced the following smoke-test results:

- ROC AUC: 0.9925
- Precision at rank n: 0.93
- Average precision: 0.9862
- Execution time: 0.0137 seconds

Among the ten evaluated methods, COPOD achieved the highest ROC AUC, precision at rank n, and average precision in this run. This confirms that the benchmark pipeline works, but this single-dataset, single-iteration smoke test is not yet a complete reproduction of the paper.

The output files were saved in:

`results/smoke_test_breastw/`

## Audit against the published experiment

The published paper was obtained from arXiv and compared with the released datasets and benchmark scripts.

The paper reports results on 30 datasets:

- 14 datasets in MAT format from ODDS;
- 16 datasets in ARFF format from DAMI;
- a 60% training and 40% testing split;
- 10 independent trials;
- ROC AUC and average precision as the primary evaluation metrics;
- COPOD compared with nine baseline methods.

All 30 datasets reported in the paper are present in the authors' repository. However, several inconsistencies were identified in the released benchmark scripts.

### MAT benchmark inconsistencies

The paper reports results for `shuttle.mat`, but the released `benchmark_mat.py` lists `cover.mat` instead. The repository contains `shuttle.mat` but does not contain `cover.mat`. The repository also contains `letter.mat`, which is not included in the paper's reported 30-dataset experiment.

Therefore, the published 14-dataset MAT experiment can be reconstructed by using `shuttle.mat` rather than the script's incorrect `cover.mat` entry.

### ARFF benchmark inconsistencies

The released `benchmark_arff.py` references seven filenames that do not match the supplied files. The affected supplied filenames contain `_norm`, while this part is omitted from the script paths.

The script also loads ARFF predictors as strings. Modern scikit-learn requires these values to be converted explicitly to numeric values before standardisation.

### Baseline-method inconsistency

The paper compares COPOD against Feature Bagging, but both released benchmark scripts evaluate PCA instead. Although `FeatureBagging` is imported, it is never included in the classifier dictionary. The reproduction scripts were corrected to evaluate Feature Bagging in place of PCA, matching the published tables.

### Environment limitation

The authors' repository provides no requirements file, environment file, package metadata, or exact dependency versions. The current working environment has therefore been recorded in:

`environment/requirements.txt`

Important installed versions include:

- Python 3.12.1;
- PyOD 3.6.4;
- scikit-learn 1.9.0;
- NumPy 2.5.2;
- SciPy 1.18.0;
- pandas 3.0.5;
- liac-arff 2.5.0;
- combo 0.1.3.

## Corrected paper-method smoke tests

### BreastW MAT test

The corrected MAT benchmark was tested on `breastw.mat` for one iteration using Feature Bagging rather than PCA.

COPOD produced:

- ROC AUC: 0.9925;
- precision at rank n: 0.93;
- average precision: 0.9862;
- execution time: 0.0098 seconds.

For comparison, the paper's 10-trial averages for COPOD on BreastW were 0.9936 ROC AUC and 0.9877 average precision. The corrected one-trial results are close to the published averages.

The corrected output was saved in:

`results/smoke_test_breastw_paper_methods/`

### Arrhythmia ARFF test

The corrected ARFF benchmark was tested on the Arrhythmia ARFF dataset for one iteration.

COPOD produced:

- ROC AUC: 0.7661;
- precision at rank n: 0.6351;
- average precision: 0.7361;
- execution time: 0.1698 seconds.

For comparison, the paper's 10-trial averages for COPOD on Arrhythmia ARFF were 0.7618 ROC AUC and 0.7519 average precision. The corrected one-trial results are reasonably close to the published averages.

The output was saved in:

`results/smoke_test_arrhythmia_arff/`

These tests indicate that reproducing the complete 30-dataset experiment is feasible after transparently correcting the released scripts. Full 10-trial reproduction has not yet been performed.

## Complete one-trial compatibility assessment

The corrected benchmark scripts were expanded to cover all datasets reported in the paper while retaining one iteration as a compatibility test.

The MAT compatibility run successfully evaluated all 14 published MAT datasets with all ten methods. The results and complete terminal log were saved in:

`results/compatibility_mat_14_one_trial/`

The ARFF compatibility run successfully evaluated all 16 published ARFF datasets with all ten methods. The results and complete terminal log were saved in:

`results/compatibility_arff_16_one_trial/`

Overall, all 30 of the paper's datasets completed successfully without runtime errors. This establishes that the complete 30-dataset experiment is technically feasible in the current environment after applying the documented corrections. These results use one trial and are compatibility evidence rather than the final 10-trial reproduction.

## Complete 10-trial reproduction and comparison

The corrected benchmark scripts were run using the experimental procedure described in the paper:

- 30 datasets;
- 14 MAT datasets and 16 ARFF datasets;
- 60% training and 40% testing;
- 10 independent trials;
- 10 anomaly-detection methods;
- ROC AUC and average precision evaluation.

All 30 datasets completed successfully. The final result tables were saved in:

- `results/reproduction_mat_14_10_trials/`
- `results/reproduction_arff_16_10_trials/`

The reproduced COPOD results were compared with Tables I and II of the paper using:

`scripts/compare_published_results.py`

The comparison outputs were saved in:

- `results/comparison/copod_published_vs_reproduced.csv`
- `results/comparison/copod_comparison_summary.csv`

### Overall comparison

The paper reported a mean COPOD ROC AUC of 0.8247, while the reproduction obtained 0.8132. The difference in the overall means was -0.0115, and the mean absolute dataset-level ROC AUC difference was 0.0177.

The paper reported a mean COPOD average precision of 0.5649, while the reproduction obtained 0.5475. The difference in the overall means was -0.0174, while the mean absolute dataset-level average precision difference was 0.0922.

The ROC AUC results reproduced closely overall. Average precision was close at the aggregate level but showed substantial dataset-level discrepancies, particularly for several ARFF datasets.

The five largest ROC AUC differences occurred for Waveform ARFF, Wine MAT, Optdigits MAT, Shuttle ARFF, and Ionosphere ARFF.

The five largest average precision differences occurred for Lymphography ARFF, SpamBase ARFF, Stamps ARFF, KDDCup99 ARFF, and Pima ARFF.

### Feasibility conclusion

The complete experiment is technically reproducible after applying documented corrections to the released scripts. The ROC AUC findings are strongly reproducible overall. The average precision findings are only partially reproducible at the individual-dataset level, despite similar overall averages.

Possible explanations include differences in software versions, dataset preprocessing or label handling, unavailable original environment details, and inconsistencies between the released scripts and published tables. These discrepancies require further investigation during the semester project but do not prevent the project from proceeding.