# MAT reproduction gaps: findings

Written: 26 October 2026.

Evidence: [project commit 49d2873](https://github.com/Shuharto-Jannat/copod-reproducibility-project/commit/49d28734f212151f336a12e242dd1c533de6e88f).

## Purpose and scope

I investigated the remaining average-precision differences for Wine, WBC and Satimage-2, three datasets from the original MAT benchmark. The aim was to check whether I could recover my historical results and examine the individual trials behind the averages.

This diagnostic completed 30 COPOD trials: three datasets, each using the original split seeds 0–9. The historical benchmark files, raw datasets and authors' implementation were preserved. The other nine detectors were outside this diagnostic.

## Data and evaluation settings

| Dataset | Total rows | Features | Total anomalies | Training / test rows per split | Test anomalies across the ten splits |
|---|---:|---:|---:|---:|---:|
| Wine | 129 | 13 | 10 | 77 / 52 | 1–5 |
| WBC | 378 | 30 | 21 | 226 / 152 | 5–13 |
| Satimage-2 | 5,803 | 36 | 71 | 3,481 / 2,322 | 19–37 |

I loaded measurements from the MAT `X` array and known labels from `y`. I retained the original non-stratified 60/40 split, using `np.random.RandomState(seed)`. All three datasets contain fewer than 10,000 rows, so the benchmark's unseeded 10,000-row sampling step does not apply to these cases.

The source-file check recorded zero duplicate feature rows in Wine, one in WBC and two in Satimage-2, counting copies beyond the first occurrence of each feature profile. I retained these rows to match the benchmark. The effect of grouping duplicate profiles when splitting was not evaluated in this diagnostic.

I used the authors' unchanged `models/cod.py`, at commit `3ca17400016386546d2c199886e81bdcc859b36b`. Scaling was fitted on training measurements and applied to the test measurements. The contamination parameter was the full-dataset anomaly fraction, matching the original benchmark; evaluation used raw scores rather than thresholded predictions. COPOD's `fit` received features without labels.

The scorer calculates its reference distribution using training features together with the entire test batch. These are therefore batch/transductive results, despite the train/test split and training-fitted scaling. ROC AUC and average precision were calculated on test rows only. Scores for training rows were retained solely to diagnose score stability.

The trial metadata records source and data hashes, row splits, package versions and numerical checks. The runs used Python 3.12.1, NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.0, scikit-learn 1.9.0, PyOD 3.6.4 and statsmodels 0.14.6. PyOD supplied preprocessing utilities; the detector was the authors' source implementation.

## Recovery of historical results

All six historical means were recovered: ROC AUC and average precision for each of the three datasets. The comparison used the original convention of rounding each trial to four decimal places before averaging. Each historical comparison matched within `1e-12` numerical tolerance.

This establishes repeatability of the saved project averages under the recorded setup. It does not independently verify the published averages or establish every detail of the original published experiments.

## Observed performance and variation

These summaries use the full-precision trial metrics. SD is the sample standard deviation across ten splits. Values below are rounded to four decimal places for readability; AP means average precision.

| Dataset | Mean ROC AUC ± SD | Mean AP ± SD | Individual-trial AP range |
|---|---:|---:|---:|
| Wine | 0.8909 ± 0.0622 | 0.4324 ± 0.1520 | 0.1667–0.6926 |
| WBC | 0.9590 ± 0.0085 | 0.6953 ± 0.1061 | 0.4651–0.8423 |
| Satimage-2 | 0.9784 ± 0.0141 | 0.7824 ± 0.0735 | 0.6463–0.8863 |

Within each dataset, every row received exactly the same anomaly score across the ten tested splits: the maximum absolute difference from the seed-0 score vector was zero. Metric variation therefore arose from which rows entered the test set, including changes in positive identities and prevalence, rather than changes in their assigned scores.

Wine illustrates the issue clearly. Seed 5 placed only one anomaly in the test set and produced AP 0.1667. Seed 3 placed five anomalies in the test set and produced AP 0.6926. This demonstrates sensitivity to test-set composition; it does not establish that anomaly count alone determines AP.

The observed score stability is consistent with the full-batch reference containing the same dataset in every split. It is an empirical finding for these files and seeds, not a claim that every dataset or possible split must behave identically.

Across the 30 runs, there were no recorded warnings, non-finite final scores or non-finite entries in the final feature-contribution matrix `O`. Thus the historical `nan_to_num` replacement did not alter the finite final scores in these runs.

## Differences from the paper remain unresolved

The table below uses paper values already recorded in the project's comparison file. The diagnostic did not independently re-extract those values from the publication. The recovered historical means use the original per-trial rounding rule described above.

| Dataset | Paper ROC AUC | Recovered historical ROC AUC | Paper AP | Recovered historical AP |
|---|---:|---:|---:|---:|
| Wine | 0.9490 | 0.89087 | 0.6082 | 0.43240 |
| WBC | 0.9747 | 0.95903 | 0.7827 | 0.69529 |
| Satimage-2 | 0.9852 | 0.97843 | 0.8599 | 0.78243 |

The AP differences, diagnostic minus paper, remain −0.17580 for Wine, −0.08741 for WBC and −0.07747 for Satimage-2. The corresponding ROC AUC differences remain −0.05813, −0.01567 and −0.00677.

I confirmed variation between test splits, but did not establish that this caused the differences from the paper. The diagnostic does not recover the publication's exact historical data versions, selected test rows or software environment. A published mean falling within a range of individual-trial values would not establish agreement between the averaged results.

The ten test sets overlap. Their SD and ranges describe observed variation across those splits; they are not confidence intervals or results from ten independent datasets. These findings also do not determine whether any implementation or data-version difference accounts for the publication gap.

## Implication for the final report

I can report that I reproduced my earlier MAT averages and investigated both numerical behaviour and test-set composition. The saved trial evidence identifies evaluation variability even when COPOD's row scores are stable. The remaining published discrepancies should be reported transparently as unresolved, with this investigation distinguished from the separate confirmed ARFF loading errors.

## Reproducibility and evidence

To run or verify the original diagnostic trials:

```bash
python scripts/diagnose_mat_reproduction_gaps.py
```

The script checks existing trial files before reuse and compares their recorded configuration. It saves each dataset under `results/audits/mat_reproduction_gaps/v1/<dataset>/`:

- `seed00/` through `seed09/`: metrics, all-row diagnostic scores, train/test row indices and metadata.
- `summary/all_metrics.csv`: the ten trial results and test anomaly counts.
- `summary/performance_summary.csv`: means, sample SD, ranges and valid-trial counts.
- `summary/historical_comparison.csv`: recovery of historical means and the remaining recorded paper gaps.
- `summary/score_stability.csv`: differences between each trial's all-row score vector and seed 0.
- `summary/summary_metadata.json`: configuration, file hashes and interpretation limits.

The original results remain in `results/reproduction_mat_14_10_trials/`. This diagnostic used only seeds 0–9; additional split-sensitivity experiments have not been performed at this stage.
