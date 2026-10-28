"""Run the committed full APS comparison, preserving the supplied train/test split.

First condition: python scripts/evaluate_aps_full_data.py --conditions primary
Both conditions: python scripts/evaluate_aps_full_data.py

Completed bundles are verified and reused. Bootstrap summaries and presentation
figures are separate follow-up steps; this script saves their underlying scores.
"""

from pathlib import Path
from time import perf_counter
from importlib.metadata import version
import argparse
import hashlib
import inspect
import json
import os
import platform
import re
import warnings

import numpy as np
import pandas as pd
import pyod.models.copod as copod_module
import pyod.utils.stat_models as ecdf_module
from pyod.models.copod import COPOD
from scipy.stats import skew
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import audit_aps_full_data as audit
from evaluate_synthetic_copod import TrainingReferenceCOPOD


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "notes/aps_full_evaluation_plan.md"
PLAN_COMMIT = "f005e7041ab47449e2f7fe24a60c70452d6ed247"
PLAN_SHA256 = "a9e1d152c94caf7012a8a8cab732d91f0a9672fd560abba037ab020e66d7f3b7"
AUDIT_DIR = ROOT / "results/audits/aps_full_data/v1"
CONDITIONS = ("primary", "drop_high_missingness")
IF_SEEDS = (42, 43, 44, 45, 46)
IF_PARAMS = dict(n_estimators=200, max_samples=256, max_features=1.0,
                 bootstrap=False, contamination="auto", n_jobs=1, warm_start=False)
SCHEMA = "aps_full_evaluation_v1"
require, sha256 = audit.require, audit.sha256


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def array_sha256(values):
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def check_input_hashes(hashes):
    for relative, digest in hashes.items():
        path = ROOT / relative
        require(path.is_file() and sha256(path) == digest,
                f"Missing or changed protected input: {relative}")


def measured(stage, function, warning_log):
    """Time only the requested operation, then retain and display its warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = perf_counter()
        try:
            result = function()
            elapsed = perf_counter() - start
        finally:
            for warning in caught:
                item = {"stage": stage, "category": warning.category.__name__,
                        "message": str(warning.message)}
                warning_log.append(item)
                print(f"Warning [{stage}]: {item['message']}", flush=True)
    return result, elapsed


def write_bundle(folder, config, tables, details):
    require(not folder.exists(), f"Output directory already exists: {folder}")
    payloads = {name: frame.to_csv(index=False).encode("utf-8")
                for name, frame in tables.items()}
    metadata = {"schema": SCHEMA, "configuration": config, "details": details,
                "output_sha256": {name: hashlib.sha256(content).hexdigest()
                                  for name, content in payloads.items()}}
    folder.mkdir(parents=True, exist_ok=False)
    for name, content in payloads.items():
        (folder / name).write_bytes(content)
    # Written last: an interrupted bundle cannot appear complete.
    (folder / "metadata.json").write_bytes(json_bytes(metadata))
    return metadata


def read_bundle(folder, config, expected_files):
    path = folder / "metadata.json"
    require(path.is_file(), f"Incomplete bundle: {folder}; inspect it and use a new --run-name.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    require(metadata.get("schema") == SCHEMA and metadata["configuration"] == config,
            f"Saved configuration differs: {folder}; use a new --run-name.")
    require(set(metadata["output_sha256"]) == set(expected_files),
            f"Unexpected saved files: {folder}")
    tables = {}
    for name, digest in metadata["output_sha256"].items():
        require((folder / name).is_file() and sha256(folder / name) == digest,
                f"Missing or changed result: {folder / name}")
        tables[name] = pd.read_csv(folder / name, float_precision="round_trip")
    return metadata, tables


def validate_scores(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    require(y.ndim == scores.ndim == 1 and y.shape == scores.shape and len(y) > 0,
            "Labels and scores must be nonempty matching vectors.")
    require(np.isin(y, [0, 1]).all() and np.isfinite(scores).all(),
            "Invalid labels or non-finite scores.")
    return y, scores


def precision_at_k(y, scores, k):
    """Expected precision if a cutoff tie is resolved uniformly at random."""
    y, scores = validate_scores(y, scores)
    require(isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_))
            and 1 <= k <= len(y), "k must be an integer between 1 and the test size.")
    cutoff = float(np.partition(scores, len(scores) - k)[len(scores) - k])
    above, tied = scores > cutoff, scores == cutoff  # Exact equality, not isclose.
    slots, tie_size = int(k - above.sum()), int(tied.sum())
    positive_above, positive_tied = int(y[above].sum()), int(y[tied].sum())
    return {"precision_at_375": float((positive_above + slots * positive_tied / tie_size) / k),
            "precision_k": int(k), "precision_cutoff_score": cutoff,
            "cutoff_tie_size": tie_size, "cutoff_remaining_slots": slots,
            "positives_above_cutoff": positive_above, "positives_in_cutoff_tie": positive_tied}


def performance(y, scores, condition):
    y, scores = validate_scores(y, scores)
    require(np.unique(y).size == 2, "Both test classes are required for ROC AUC.")
    result = {"roc_auc": float(roc_auc_score(y, scores)),
              "average_precision": float(average_precision_score(y, scores))}
    if condition == "primary":
        result.update(precision_at_k(y, scores, int(y.sum())))
    return result


def validate_ties():
    """A small label-free fixture, including ties and both unseen tails."""
    X = np.array([[0., 0., 0.], [0., 3., 1.], [1., 3., 1.], [3., 3., 2.]])
    query = np.array([[-1., -1., -1.], [0., 0., 0.], [.5, 2., 1.],
                      [3., 3., 2.], [4., 4., 3.]])
    fixed = TrainingReferenceCOPOD().fit(X)
    standard = COPOD(contamination=0.1, n_jobs=1).fit(X)
    np.testing.assert_allclose(fixed.decision_function(X), standard.decision_scores_,
                               rtol=1e-10, atol=1e-10)
    left_counts = (X[None, :, :] <= query[:, None, :]).sum(axis=1)
    right_counts = (X[None, :, :] >= query[:, None, :]).sum(axis=1)
    left = -np.log(np.maximum(left_counts / len(X), 1 / len(X)))
    right = -np.log(np.maximum(right_counts / len(X), 1 / len(X)))
    signs = np.sign(skew(X, axis=0, bias=True))
    expected = np.maximum(left * -np.sign(signs - 1) + right * np.sign(signs + 1),
                          (left + right) / 2).sum(axis=1)
    np.testing.assert_allclose(fixed.decision_function(query), expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(left_counts[:, 0], [0, 2, 2, 4, 4])
    np.testing.assert_array_equal(right_counts[:, 0], [4, 4, 2, 1, 0])
    # Also exercise the explicitly defined fractional cutoff convention.
    tied_precision = precision_at_k([1, 1, 0, 1, 0], [5, 4, 4, 4, 1], 3)
    np.testing.assert_allclose(tied_precision["precision_at_375"], 7 / 9, rtol=0, atol=1e-15)
    return {"status": "passed", "training_rows": 4, "query_rows": 5,
            "skewness_signs": signs.astype(int).tolist(),
            "timing_note": "This tiny check calls PyOD ECDF before full-data timing; "
                           "other first-call effects are not controlled."}


def load_inputs():
    require(PLAN.is_file() and sha256(PLAN) == PLAN_SHA256,
            "The evaluation plan differs from committed f005e70. Review the plan before running.")
    audit_path = AUDIT_DIR / "audit_metadata.json"
    require(audit_path.is_file(), "Run and save the full APS audit first.")
    evidence = json.loads(audit_path.read_text(encoding="utf-8"))
    require(evidence["schema_version"] == "aps_full_raw_data_audit_v1", "Unexpected audit version.")
    expected_reports = {"feature_audit.csv", "duplicate_groups.csv", "duplicate_members.csv"}
    require(set(evidence["output_sha256"]) == expected_reports, "Unexpected audit reports.")
    for name, digest in evidence["output_sha256"].items():
        require((AUDIT_DIR / name).is_file() and sha256(AUDIT_DIR / name) == digest,
                f"Audit report missing or changed: {name}")
    check_input_hashes(evidence["input_sha256"])
    loaded = {}
    for partition, name in audit.FILES.items():
        print(f"Loading audited full {partition} file...", flush=True)
        loaded[partition] = audit.load_partition(audit.DATA / name, audit.EXPECTED_ROWS[partition])
        X, y = loaded[partition]
        require(audit.partition_summary(X, y) == evidence["partitions"][partition],
                f"Parsed {partition} data differs from the saved audit.")
        require(X.columns.tolist() == evidence["parsing"]["ordered_predictors"], "Column order differs.")
    require(loaded["train"][0].shape == (60000, 170) and int(loaded["train"][1].sum()) == 1000
            and loaded["test"][0].shape == (16000, 170) and int(loaded["test"][1].sum()) == 375,
            "Full APS partition sizes or positive counts differ from the frozen plan.")
    return loaded, evidence


def establish_manifest(folder, evidence):
    require(version("pyod") == "3.6.4", "Review any PyOD version change before this planned comparison.")
    source_paths = [PLAN, Path(__file__).resolve(), Path(inspect.getsourcefile(audit)),
                    Path(inspect.getsourcefile(TrainingReferenceCOPOD)),
                    AUDIT_DIR / "audit_metadata.json"]
    source_paths += [AUDIT_DIR / name for name in evidence["output_sha256"]]
    hashes = dict(evidence["input_sha256"])
    hashes.update({str(path.relative_to(ROOT)): sha256(path) for path in source_paths})
    implementations = {"pyod_copod_module": copod_module, "pyod_ecdf_module": ecdf_module,
                       "isolation_forest_module": IsolationForest, "simple_imputer_module": SimpleImputer,
                       "standard_scaler_module": StandardScaler, "scipy_skew_module": skew}
    manifest = {
        "schema": SCHEMA, "run_name": folder.name,
        "plan_commit": PLAN_COMMIT, "plan_sha256": PLAN_SHA256,
        "input_sha256": hashes,
        "implementation_sha256": {name: sha256(Path(inspect.getsourcefile(obj)))
                                  for name, obj in implementations.items()},
        "versions": {name: version(name) for name in
                     ["numpy", "pandas", "scipy", "scikit-learn", "pyod", "numba", "joblib"]},
        "python": platform.python_version(),
        "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                     "processor": platform.processor(), "logical_cpus": os.cpu_count()},
        "planned_conditions": list(CONDITIONS), "isolation_forest_seeds": list(IF_SEEDS),
        "isolation_forest_parameters": IF_PARAMS,
        "copod_parameters": {"contamination": 0.1, "n_jobs": 1},
        "label_use": "Descriptive counts, audit conflicts and test metrics only; never model fitting.",
        "row_reference": "Every supplied row in original order; indices zero-based after CSV header.",
        "partitions": {"train": {"rows": 60000, "first_index": 0, "last_index": 59999},
                       "test": {"rows": 16000, "first_index": 0, "last_index": 15999}},
        "pending_followups": "Primary paired bootstrap summaries and presentation figures are separate steps.",
    }
    content = json_bytes(manifest)
    path = folder / "run_manifest.json"
    require(not path.exists() or path.read_bytes() == content,
            "Existing run provenance differs. Inspect it and use a new --run-name.")
    folder.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)
    return manifest, hashlib.sha256(content).hexdigest()


def prepare_condition(folder, condition, loaded, manifest_hash):
    train, y_train = loaded["train"]
    test, y_test = loaded["test"]
    warning_log = []

    def select():
        unique = train.nunique(dropna=True)
        missing = train.isna().mean()
        reasons = pd.Series("keep", index=train.columns)
        reasons.loc[unique == 0] = "all_missing_in_training"
        reasons.loc[unique == 1] = "one_observed_value_in_training"
        if condition == "drop_high_missingness":
            reasons.loc[(unique > 1) & (missing > 0.50)] = "training_missing_fraction_gt_0.50"
        return reasons

    reasons, selection_seconds = measured("feature_selection", select, warning_log)
    columns = reasons.index[reasons.eq("keep")].tolist()
    require(len(columns) > 0, "No usable training predictors.")
    if condition == "primary":
        require(len(columns) == 169 and reasons.index[~reasons.eq("keep")].tolist() == ["cd_000"],
                "Primary feature selection differs from the audit.")
    raw_train, raw_test = train[columns], test[columns]
    imputer = SimpleImputer(strategy="median", add_indicator=False)
    (imputed_train, imputed_test), imputation_seconds = measured(
        "median_imputation", lambda: (imputer.fit_transform(raw_train), imputer.transform(raw_test)), warning_log)
    scaler = StandardScaler()
    (X_train, X_test), scaling_seconds = measured(
        "standard_scaling", lambda: (scaler.fit_transform(imputed_train), scaler.transform(imputed_test)), warning_log)
    require(X_train.shape == (len(train), len(columns)) and X_test.shape == (len(test), len(columns)),
            "Preprocessing changed the number of rows or unexpectedly dropped columns.")
    require(np.isfinite(X_train).all() and np.isfinite(X_test).all(), "Non-finite prepared measurements.")
    require((np.ptp(X_train, axis=0) > 0).all(), "Constant prepared training feature.")
    decisions = pd.DataFrame({"feature": train.columns, "train_missing": train.isna().sum().to_numpy(),
                              "test_missing": test.isna().sum().to_numpy(),
                              "train_missing_fraction": train.isna().mean().to_numpy(),
                              "train_unique_observed": train.nunique(dropna=True).to_numpy(),
                              "retained": reasons.eq("keep").to_numpy(), "reason": reasons.to_numpy()})
    parameters = pd.DataFrame({"feature": columns, "training_median": imputer.statistics_,
                               "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                               "scaler_variance": scaler.var_})
    print(f"{condition}: {len(columns)} predictors; checking profiles after preprocessing...", flush=True)
    selected_groups, selected_members, selected_summary = audit.duplicate_audit(raw_train, raw_test, y_train, y_test)
    prepared_groups, prepared_members, prepared_summary = audit.duplicate_audit(X_train, X_test, y_train, y_test)
    tables = {"feature_decisions.csv": decisions, "preprocessing_parameters.csv": parameters,
              "selected_duplicate_groups.csv": selected_groups, "selected_duplicate_members.csv": selected_members,
              "prepared_duplicate_groups.csv": prepared_groups, "prepared_duplicate_members.csv": prepared_members}
    facts = {"ordered_predictors": columns,
             "prepared_train_sha256": array_sha256(X_train), "prepared_test_sha256": array_sha256(X_test),
             "train_labels_sha256": array_sha256(y_train), "test_labels_sha256": array_sha256(y_test),
             "selected_raw_profile_audit": selected_summary, "prepared_profile_audit": prepared_summary,
             "duplicate_semantics": "Exact numeric equality of this stage's selected columns; "
                                    "NaNs and signed zeros canonicalised. All rows retained."}
    config = {"run_manifest_sha256": manifest_hash, "condition": condition,
              "missing_fraction_exclusion": None if condition == "primary" else "strictly_greater_than_0.50",
              "imputer": {"strategy": "median", "add_indicator": False},
              "scaler": "StandardScaler fitted only on imputed training measurements"}
    destination = folder / "preprocessing"
    if destination.exists():
        metadata, _ = read_bundle(destination, config, tables)
        require(metadata["details"]["facts"] == facts, "Prepared arrays/columns differ from saved evidence.")
        for name, frame in tables.items():
            require(hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()
                    == metadata["output_sha256"][name], f"Recomputed preprocessing report differs: {name}")
        require(metadata["details"]["warnings"] == warning_log, "Preprocessing warnings differ on verification.")
        print("  Verified saved preprocessing and exact prepared-array hashes.", flush=True)
    else:
        metadata = write_bundle(destination, config, tables,
                                {"facts": facts, "warnings": warning_log,
                                 "timings_seconds": {"feature_selection": selection_seconds,
                                                     "imputation": imputation_seconds, "scaling": scaling_seconds},
                                 "all_rows_retained": True, "checks": "passed"})
    shared = prepared_summary["across_supplied_split"]["shared_profiles"]
    print(f"  Shared prepared train/test profiles: {shared}; all supplied rows retained.", flush=True)
    return X_train, X_test, columns, sha256(destination / "metadata.json")


def feature_diagnostics(X_train, X_test, columns, fixed, warning_log):
    combined_signs, _ = measured("combined_skewness_diagnostic",
                                 lambda: np.sign(skew(np.vstack([X_train, X_test]), axis=0, bias=True)), warning_log)
    require(np.isfinite(combined_signs).all(), "Undefined combined-data skewness.")
    records = []
    for j, name in enumerate(columns):
        reference = fixed.sorted_train[:, j]
        left = np.searchsorted(reference, X_test[:, j], side="right")
        right = len(reference) - np.searchsorted(reference, X_test[:, j], side="left")
        below, above = X_test[:, j] < reference[0], X_test[:, j] > reference[-1]
        require(np.array_equal(left == 0, below) and np.array_equal(right == 0, above),
                "Zero-count tails do not agree with training-range checks.")
        records.append({"feature": name, "training_skewness_sign": int(fixed.skew_sign[j]),
                        "combined_skewness_sign": int(combined_signs[j]),
                        "skewness_sign_changed": bool(fixed.skew_sign[j] != combined_signs[j]),
                        "train_min_scaled": float(reference[0]), "train_max_scaled": float(reference[-1]),
                        "test_below_train_min": int(below.sum()), "test_above_train_max": int(above.sum()),
                        "left_zero_probability_replacements": int((left == 0).sum()),
                        "right_zero_probability_replacements": int((right == 0).sum()),
                        "left_already_at_floor_not_replaced": int((left == 1).sum()),
                        "right_already_at_floor_not_replaced": int((right == 1).sum())})
    return pd.DataFrame(records)


def metric_row(condition, method, seed, y_train, y_test, n_features, scores, fit_time, score_time):
    return {"condition": condition, "method": method, "model_seed": seed,
            "n_train": len(y_train), "n_test": len(y_test), "n_features": n_features,
            "train_positives": int(y_train.sum()), "test_positives": int(y_test.sum()),
            "test_prevalence": float(y_test.mean()), **performance(y_test, scores, condition),
            "fit_seconds": fit_time, "score_seconds": score_time}


def verify_model_bundle(folder, config, methods, condition, y_train, y_test, n_features, extra_files=()):
    metadata, tables = read_bundle(folder, config, ["metrics.csv", "scores.csv", *extra_files])
    scores, metrics = tables["scores.csv"], tables["metrics.csv"]
    require(scores.columns.tolist() == ["row_index_zero_based", "label", *methods], "Unexpected score columns.")
    require(np.array_equal(scores.row_index_zero_based, np.arange(len(y_test)))
            and np.array_equal(scores.label, y_test), "Saved test rows/labels differ from original source order.")
    require(len(metrics) == len(methods) and metrics.method.tolist() == methods, "Metric method order differs.")
    for _, row in metrics.iterrows():
        require(row.condition == condition and row.n_train == len(y_train) and row.n_test == len(y_test)
                and row.n_features == n_features and row.train_positives == int(y_train.sum())
                and row.test_positives == int(y_test.sum()) and row.test_prevalence == float(y_test.mean()),
                "Saved dataset counts or condition differ.")
        expected_seed = config.get("random_state")
        require(pd.isna(row.model_seed) if expected_seed is None else row.model_seed == expected_seed,
                "Saved model seed differs.")
        for name, value in performance(y_test, scores[row.method].to_numpy(), condition).items():
            require(np.isclose(row[name], value, rtol=0, atol=1e-12), f"Saved metric differs from scores: {name}")
        require(np.isfinite([row.fit_seconds, row.score_seconds]).all()
                and row.fit_seconds >= 0 and row.score_seconds >= 0, "Invalid saved timings.")
    require(metadata["details"]["checks"] == "passed", "Saved implementation checks did not pass.")
    return metadata, tables


def run_copod(folder, condition, X_train, X_test, y_train, y_test, columns, config):
    destination = folder / "copod"
    config = {**config, "methods": ["pyod_batch", "fixed_training_reference"],
              "standard_parameters": {"contamination": 0.1, "n_jobs": 1},
              "modified_probability_floor": 1 / len(X_train), "batch_call_rows": len(X_test)}
    if destination.exists():
        metadata, tables = verify_model_bundle(destination, config, config["methods"], condition,
                                               y_train, y_test, len(columns), ["feature_diagnostics.csv"])
        require(metadata["details"]["training_score_agreement"] == "passed"
                and metadata["details"]["fixed_batch_invariance"] == "passed", "Missing COPOD checks.")
        print("  Verified saved COPOD results; no model refit.", flush=True)
        return tables["metrics.csv"]
    print("  Fitting standard COPOD and the fixed training-reference scorer...", flush=True)
    warning_log = []
    standard = COPOD(contamination=0.1, n_jobs=1)  # Fresh object for this condition.
    _, standard_fit = measured("pyod_fit", lambda: standard.fit(X_train), warning_log)
    fixed = TrainingReferenceCOPOD()
    _, fixed_fit = measured("fixed_fit", lambda: fixed.fit(X_train), warning_log)
    fixed_train, _ = measured("training_score_check", lambda: fixed.decision_function(X_train), warning_log)
    require(np.isfinite(fixed_train).all() and np.isfinite(standard.decision_scores_).all(),
            "Non-finite training scores.")
    np.testing.assert_allclose(fixed_train, standard.decision_scores_, rtol=1e-10, atol=1e-10)
    agreement_difference = float(np.max(np.abs(fixed_train - standard.decision_scores_)))
    training_flags = {name: int((~np.isfinite(getattr(standard, name))).sum())
                      for name in ["U_l", "U_r", "U_skew", "O"]}
    require(not any(training_flags.values()), "Non-finite PyOD training contributions.")
    # Exactly one call on the complete supplied test batch.
    standard_scores, standard_time = measured("pyod_full_test_batch",
                                               lambda: standard.decision_function(X_test), warning_log)
    fixed_scores, fixed_time = measured("fixed_full_test_batch",
                                        lambda: fixed.decision_function(X_test), warning_log)
    require(standard.O.shape == (len(X_train) + len(X_test), len(columns)),
            "PyOD did not use the expected train-plus-test reference.")
    batch_flags = {name: int((~np.isfinite(getattr(standard, name))).sum())
                   for name in ["U_l", "U_r", "U_skew", "O"]}
    require(not any(batch_flags.values()), "Non-finite PyOD combined-data contributions.")
    require(np.array_equal(standard.O.sum(axis=1)[-len(X_test):], standard_scores),
            "Batch scores differ from PyOD's summed feature contributions.")
    chunk_scores, _ = measured("fixed_chunk_check", lambda: np.concatenate([
        fixed.decision_function(chunk) for chunk in np.array_split(X_test, 4)]), warning_log)
    single_scores, _ = measured("fixed_singleton_check", lambda: np.array([
        fixed.decision_function(X_test[i:i + 1])[0] for i in range(10)]), warning_log)
    np.testing.assert_allclose(fixed_scores, chunk_scores, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fixed_scores[:10], single_scores, rtol=0, atol=1e-12)
    diagnostics = feature_diagnostics(X_train, X_test, columns, fixed, warning_log)
    scores = pd.DataFrame({"row_index_zero_based": np.arange(len(y_test)), "label": y_test,
                           "pyod_batch": standard_scores, "fixed_training_reference": fixed_scores})
    metrics = pd.DataFrame([
        metric_row(condition, "pyod_batch", None, y_train, y_test, len(columns),
                   standard_scores, standard_fit, standard_time),
        metric_row(condition, "fixed_training_reference", None, y_train, y_test, len(columns),
                   fixed_scores, fixed_fit, fixed_time)])
    details = {"checks": "passed", "training_score_agreement": "passed",
               "training_score_max_abs_difference": agreement_difference,
               "fixed_batch_invariance": "passed",
               "fixed_chunk_max_abs_difference": float(np.max(np.abs(fixed_scores - chunk_scores))),
               "fixed_singleton_max_abs_difference": float(np.max(np.abs(fixed_scores[:10] - single_scores))),
               "nonfinite_training_contributions": training_flags,
               "nonfinite_combined_contributions": batch_flags,
               "nonfinite_final_scores": 0, "warnings": warning_log,
               "skewness_sign_changes": int(diagnostics.skewness_sign_changed.sum()),
               "zero_tail_probability_replacements": int(diagnostics[
                   ["left_zero_probability_replacements", "right_zero_probability_replacements"]].to_numpy().sum()),
               "protocols": {"pyod_batch": "training plus entire supplied test batch",
                             "fixed_training_reference": "training only; inclusive empirical tails and fixed skew signs"},
               "timing_note": "Each reported score time covers one full-test call. Training-score, "
                              "batch-invariance and feature-diagnostic checks are excluded. "
                              "Fit work differs between implementations; these are descriptive timings."}
    write_bundle(destination, config, {"metrics.csv": metrics, "scores.csv": scores,
                                       "feature_diagnostics.csv": diagnostics}, details)
    print(f"  Training-score agreement and fixed-scorer batch invariance: passed.", flush=True)
    return metrics


def run_iforest(folder, condition, seed, X_train, X_test, y_train, y_test, columns, config):
    destination = folder / f"iforest_seed{seed}"
    config = {**config, "method": "isolation_forest", "parameters": IF_PARAMS, "random_state": seed}
    if destination.exists():
        _, tables = verify_model_bundle(destination, config, ["isolation_forest"], condition,
                                         y_train, y_test, len(columns))
        print(f"  Verified saved Isolation Forest seed {seed}; no model refit.", flush=True)
        return tables["metrics.csv"]
    print(f"  Fitting Isolation Forest, seed {seed}...", flush=True)
    warning_log = []
    model = IsolationForest(**IF_PARAMS, random_state=seed)
    _, fit_time = measured(f"iforest_{seed}_fit", lambda: model.fit(X_train), warning_log)
    scores, score_time = measured(f"iforest_{seed}_score", lambda: -model.score_samples(X_test), warning_log)
    validate_scores(y_test, scores)
    metrics = pd.DataFrame([metric_row(condition, "isolation_forest", seed, y_train, y_test,
                                       len(columns), scores, fit_time, score_time)])
    table = pd.DataFrame({"row_index_zero_based": np.arange(len(y_test)), "label": y_test,
                          "isolation_forest": scores})
    write_bundle(destination, config, {"metrics.csv": metrics, "scores.csv": table},
                 {"checks": "passed", "nonfinite_final_scores": 0, "warnings": warning_log,
                  "score_direction": "negative sklearn score_samples; higher is more abnormal",
                  "reference": "training features only", "fitted_parameters": model.get_params(),
                  "effective_max_samples": int(model.max_samples_)})
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--run-name", default="v1")
    args = parser.parse_args()
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name) is not None, "Invalid run name.")
    require(len(set(args.conditions)) == len(args.conditions), "Do not repeat a condition.")
    loaded, evidence = load_inputs()
    folder = ROOT / "results/aps_full_evaluation" / args.run_name
    manifest, manifest_hash = establish_manifest(folder, evidence)
    fixture_warnings = []
    fixture_result, _ = measured("synthetic_tie_fixture", validate_ties, fixture_warnings)
    fixture_content = json_bytes({"result": fixture_result, "warnings": fixture_warnings})
    fixture_path = folder / "implementation_checks.json"
    require(not fixture_path.exists() or fixture_path.read_bytes() == fixture_content,
            "Stored small-fixture checks differ.")
    if not fixture_path.exists():
        fixture_path.write_bytes(fixture_content)
    print("Small tied-value and cutoff-precision checks: passed.", flush=True)
    y_train, y_test = loaded["train"][1], loaded["test"][1]
    for condition in CONDITIONS:
        if condition not in args.conditions:
            continue
        print(f"\nCondition: {condition}", flush=True)
        current = folder / condition
        X_train, X_test, columns, preprocessing_hash = prepare_condition(current, condition, loaded, manifest_hash)
        config = {"run_manifest_sha256": manifest_hash, "condition": condition,
                  "preprocessing_metadata_sha256": preprocessing_hash,
                  "implementation_checks_sha256": sha256(fixture_path)}
        results = [run_copod(current, condition, X_train, X_test, y_train, y_test, columns, config)]
        for seed in IF_SEEDS:
            results.append(run_iforest(current, condition, seed, X_train, X_test, y_train, y_test, columns, config))
        combined = pd.concat(results, ignore_index=True)
        require(len(combined) == 7, "Expected two COPOD rows and five Isolation Forest rows.")
        print(combined[["method", "model_seed", "roc_auc", "average_precision"]].to_string(index=False), flush=True)
        check_input_hashes(manifest["input_sha256"])
    print("\nRequested evaluation conditions completed; original data and pilot results are unchanged.")
    print("Saved/verified result bundles:", folder.relative_to(ROOT))
    print("Next: review results, then complete the planned summaries, paired resampling and figures.")


if __name__ == "__main__":
    main()
