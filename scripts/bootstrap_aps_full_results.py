"""Run the frozen primary APS paired, class-stratified resampling analysis.

Run: python scripts/bootstrap_aps_full_results.py
Keeps original scores and fitted models fixed. Does not fit or score any model.
Repeating the command regenerates the deterministic draws and verifies saved files.
"""

from pathlib import Path
from importlib.metadata import version
import argparse
import hashlib
import inspect
import json
import platform
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import summarise_aps_full_results as summary


ROOT = Path(__file__).resolve().parents[1]
require = summary.require
N_RESAMPLES = 2000
RANDOM_SEED = 20261026
METHODS = ("pyod_batch", "fixed_training_reference", "fixed_minus_batch")
METRICS = ("roc_auc", "average_precision")
INTERPRETATION = {
    "interval_label": "95% percentile resampling intervals for the observed score-label pairs",
    "conditioning": "Fitted preprocessing, fitted models, original full-test score vectors, "
                    "original batch-scoring reference, and observed test prevalence are fixed.",
    "pairing": "Both procedures use the same sampled row indices in every replicate. "
               "Difference intervals use replicate-level fixed-reference-minus-batch differences.",
    "stratification": "Every replicate contains 375 positive and 15625 negative draws with replacement; "
                      "test prevalence stays at 0.0234375.",
    "assumption": "Rows are treated as independent resampling units; anonymous measurements "
                  "do not establish truck-level independence.",
    "omitted_uncertainty": "No refitting, changed-batch scoring reference, prevalence uncertainty, "
                           "or population/deployment shift is represented.",
    "claims": "Exploratory conditional intervals, not guaranteed future-performance coverage "
              "or confirmatory significance/equivalence evidence.",
    "scope": "Primary COPOD ROC AUC and non-interpolated average precision only. "
             "No bootstrap for Isolation Forest, secondary preprocessing, or Precision@375.",
}


def validate_pair(y, batch, fixed):
    y, batch = summary.evaluation.validate_scores(y, batch)
    _, fixed = summary.evaluation.validate_scores(y, fixed)
    require(np.unique(y).size == 2, "Both classes are required for paired resampling.")
    return y, batch, fixed


def paired_metrics(y, batch, fixed):
    result = {}
    for metric, function in [("roc_auc", roc_auc_score), ("average_precision", average_precision_score)]:
        standard_value, fixed_value = float(function(y, batch)), float(function(y, fixed))
        require(np.isfinite([standard_value, fixed_value]).all()
                and 0 <= standard_value <= 1 and 0 <= fixed_value <= 1, "Invalid resampled metric.")
        result["pyod_batch_" + metric] = standard_value
        result["fixed_training_reference_" + metric] = fixed_value
        result["fixed_minus_batch_" + metric] = fixed_value - standard_value
    return result


def draw_replicates(y, batch, fixed, n_resamples, seed, progress=False):
    """Explicit row draws; the same index vector is applied to both procedures."""
    y, batch, fixed = validate_pair(y, batch, fixed)
    require(isinstance(n_resamples, int) and not isinstance(n_resamples, bool) and n_resamples >= 2,
            "At least two integer resamples are required.")
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.Generator(np.random.PCG64(seed))
    records = []
    for replicate in range(1, n_resamples + 1):
        # Draw positives first, then negatives. No extra RNG calls or row shuffle.
        indices = np.concatenate([rng.choice(positive, size=len(positive), replace=True),
                                  rng.choice(negative, size=len(negative), replace=True)])
        sampled_y = y[indices]
        require(len(indices) == len(y) and int(sampled_y.sum()) == len(positive)
                and (sampled_y[:len(positive)] == 1).all() and (sampled_y[len(positive):] == 0).all(),
                "A replicate failed the fixed class-count check.")
        # Canonical little-endian int64 digest records the exact ordered draw.
        digest = hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
        records.append({"replicate_id_one_based": replicate, "draw_indices_sha256": digest,
                        **paired_metrics(sampled_y, batch[indices], fixed[indices])})
        if progress and replicate % 250 == 0:
            print(f"  Completed {replicate:,} / {n_resamples:,} paired resamples.", flush=True)
    return pd.DataFrame(records)


def check_small_examples():
    """Analytical tie/perfect-score fixtures; no model is fitted."""
    labels = np.array([0, 0, 0, 1, 1])
    tied_scores = np.array([0., 2., 1., 2., 3.])
    same = draw_replicates(labels, tied_scores, tied_scores.copy(), 8, 123)
    for metric in METRICS:
        require((same["fixed_minus_batch_" + metric] == 0).all(), "Identical methods produced nonzero paired differences.")
    # A constant score has ROC AUC 0.5 and AP equal to prevalence; perfect scores have both 1.
    example = draw_replicates(labels, np.zeros(len(labels)), labels.astype(float), 8, 123)
    np.testing.assert_allclose(example[["pyod_batch_roc_auc", "pyod_batch_average_precision"]],
                               np.tile([0.5, labels.mean()], (8, 1)), rtol=0, atol=1e-15)
    np.testing.assert_array_equal(example[["fixed_training_reference_roc_auc", "fixed_training_reference_average_precision"]],
                                  np.ones((8, 2)))
    reversed_methods = draw_replicates(labels, labels.astype(float), np.zeros(len(labels)), 8, 123)
    pd.testing.assert_series_equal(example.draw_indices_sha256, reversed_methods.draw_indices_sha256)
    for metric in METRICS:
        np.testing.assert_array_equal(example["fixed_minus_batch_" + metric],
                                      -reversed_methods["fixed_minus_batch_" + metric])
    pd.testing.assert_frame_equal(example, draw_replicates(labels, np.zeros(len(labels)), labels.astype(float), 8, 123))
    np.testing.assert_allclose(np.quantile([0., 10.], [0.025, 0.975], method="linear"),
                               [0.25, 9.75], rtol=0, atol=1e-15)
    return {"status": "passed", "checks": ["identical_scores_zero_differences", "constant_score_ties",
                                              "perfect_scores", "paired_method_swap", "same_seed_replay", "linear_quantiles"]}


def make_intervals(replicates, point_estimates):
    records = []
    for method in METHODS:
        for metric in METRICS:
            column = method + "_" + metric
            values = replicates[column].to_numpy(dtype=float)
            require(np.isfinite(values).all(), "Non-finite bootstrap distribution.")
            low, high = np.quantile(values, [0.025, 0.975], method="linear")
            records.append({"method_or_difference": method, "metric": metric,
                            "point_estimate": point_estimates[column], "bootstrap_mean": float(values.mean()),
                            "lower_2_5_percentile": float(low), "upper_97_5_percentile": float(high),
                            "n_resamples": len(replicates)})
    return pd.DataFrame(records)


def configuration(evidence):
    inputs = dict(evidence["input_sha256"])
    for path in [Path(__file__).resolve(), Path(summary.__file__).resolve()]:
        summary.remember_file(path, inputs)
    versions = {"python": platform.python_version(), **{name: version(name) for name in
                 ["numpy", "pandas", "scipy", "scikit-learn", "pyod"]}}
    source_functions = {"roc_auc_score": roc_auc_score, "average_precision_score": average_precision_score,
                        "numpy_quantile": np.quantile}
    return {"schema": "aps_primary_paired_bootstrap_v1", "condition": "primary",
            "methods": list(METHODS[:2]), "metrics": list(METRICS),
            "n_resamples": N_RESAMPLES, "random_seed": RANDOM_SEED,
            "rng": "numpy.random.Generator(numpy.random.PCG64(seed))",
            "draw_order": "Positive indices first, then negative indices; both sampled with replacement; no shuffle.",
            "test_rows": 16000, "positive_draws_per_replicate": 375, "negative_draws_per_replicate": 15625,
            "quantile_probabilities": [0.025, 0.975], "quantile_method": "linear",
            "input_sha256": inputs, "versions": versions,
            "metric_source_sha256": {name: summary.sha256(Path(inspect.getsourcefile(inspect.unwrap(function))))
                                     for name, function in source_functions.items()},
            "evaluation_versions": evidence["manifest"]["versions"]}


def save_or_verify(folder, tables, config, checks, warnings_recorded):
    summary.evaluation.check_input_hashes(config["input_sha256"])
    payloads = {name: table.to_csv(index=False).encode("utf-8") for name, table in tables.items()}
    metadata = {"configuration": config, "checks": checks, "warnings": warnings_recorded,
                "interpretation": INTERPRETATION,
                "output_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}}
    payloads["bootstrap_metadata.json"] = summary.evaluation.json_bytes(metadata)
    if folder.exists():
        require({path.name for path in folder.iterdir()} <= set(payloads), "Unexpected existing resampling files.")
    # Verify all existing files before writing any missing file. Never replace differing evidence.
    for name, content in payloads.items():
        path = folder / name
        require(not path.exists() or path.read_bytes() == content,
                f"Existing resampling output differs: {path}. Inspect the change; do not overwrite evidence.")
    folder.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        path = folder / name
        if not path.exists():
            path.write_bytes(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="v1")
    args = parser.parse_args()
    evidence = summary.load_verified_results(args.run_name)
    scores = evidence["scores"][("primary", "copod")]
    y, batch, fixed = validate_pair(evidence["labels"], scores.pyod_batch.to_numpy(),
                                    scores.fixed_training_reference.to_numpy())
    require(len(y) == 16000 and int(y.sum()) == 375, "Primary APS class counts differ from the plan.")
    config = configuration(evidence)
    destination = evidence["folder"] / "paired_bootstrap_primary"
    old_metadata = destination / "bootstrap_metadata.json"
    if old_metadata.exists():
        saved = json.loads(old_metadata.read_text(encoding="utf-8"))
        require(saved["configuration"] == config, "Saved resampling configuration or inputs differ; inspect before proceeding.")
        print("Saved configuration matches; regenerating draws to verify the saved resampling results.", flush=True)
    before = [summary.evaluation.array_sha256(array) for array in [y, batch, fixed]]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        checks = check_small_examples()
        print("Small paired-resampling, tie and quantile checks: passed.", flush=True)
        point_estimates = paired_metrics(y, batch, fixed)
        print("Running 2,000 paired class-stratified resamples of fixed primary COPOD scores...", flush=True)
        replicates = draw_replicates(y, batch, fixed, N_RESAMPLES, RANDOM_SEED, progress=True)
        require(np.array_equal(replicates.replicate_id_one_based, np.arange(1, N_RESAMPLES + 1)),
                "Incomplete resampling sequence.")
        for metric in METRICS:
            np.testing.assert_array_equal(replicates["fixed_minus_batch_" + metric],
                                          replicates["fixed_training_reference_" + metric] - replicates["pyod_batch_" + metric])
        intervals = make_intervals(replicates, point_estimates)
    after = [summary.evaluation.array_sha256(array) for array in [y, batch, fixed]]
    require(before == after, "Original label or score arrays were changed.")
    checks.update({"original_arrays_unchanged": True, "all_replicates_have_fixed_class_counts": True,
                   "shared_indices_for_both_methods": True, "paired_difference_identity": "passed",
                   "completed_replicates": N_RESAMPLES, "nonfinite_metrics": 0})
    warning_records = [{"category": item.category.__name__, "message": str(item.message)} for item in caught]
    for warning in warning_records:
        print(f"Warning [{warning['category']}]: {warning['message']}", flush=True)
    tables = {"replicate_metrics.csv": replicates, "percentile_intervals.csv": intervals}
    save_or_verify(destination, tables, config, checks, warning_records)
    print("\n95% percentile resampling intervals for the observed score-label pairs:")
    print(intervals[["method_or_difference", "metric", "point_estimate", "lower_2_5_percentile", "upper_97_5_percentile"]]
          .to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nCompleted {N_RESAMPLES:,} paired resamples. Recorded warnings: {len(warning_records)}.")
    print("Scores, fitted models, original batch reference and class prevalence stayed fixed.")
    print("These conditional intervals do not establish future performance, significance or equivalence.")
    print(f"Saved or verified two CSV files and bootstrap_metadata.json: {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
