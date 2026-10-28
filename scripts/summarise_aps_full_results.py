"""Verify and summarise both frozen APS conditions without fitting models.

Run: python scripts/summarise_aps_full_results.py
Requires the audited raw APS files and both completed evaluation conditions.
The planned paired resampling analysis and figures are separate follow-up steps.
"""

from pathlib import Path
from importlib.metadata import version
import argparse
import hashlib
import json
import platform
import re

import numpy as np
import pandas as pd

import evaluate_aps_full_data as evaluation


ROOT = Path(__file__).resolve().parents[1]
require = evaluation.require
sha256 = evaluation.sha256
METHODS = ("pyod_batch", "fixed_training_reference", "isolation_forest")
PREPROCESSING_FILES = (
    "feature_decisions.csv", "preprocessing_parameters.csv",
    "selected_duplicate_groups.csv", "selected_duplicate_members.csv",
    "prepared_duplicate_groups.csv", "prepared_duplicate_members.csv",
)
INTERPRETATION = {
    "scope": "The supplied APS split and frozen configurations; no new model fits.",
    "scoring_access": "Batch COPOD uses training plus test features; fixed-reference COPOD "
                      "and Isolation Forest establish their scoring rules from training features only.",
    "sd": "Sample SD (ddof=1) across five IF fits on the same data; not a confidence interval. "
          "Deterministic COPOD SD is not applicable and is left blank.",
    "missingness_difference": "Secondary minus primary, matching each IF seed; a fixed sensitivity check.",
    "protocol_difference": "Fixed training-reference minus standard batch COPOD; point differences only.",
    "precision": "Primary-only Precision@375, with fractional cutoff-tie credit; retrospective, "
                 "not a deployable alert threshold. Secondary values are not computed.",
    "tail_floor": "Replacement counts are feature-test-cell tail events, not distinct observations. "
                  "Probabilities already equal to the floor are counted separately.",
    "duplicates": "Exact retained measurement profiles, not verified truck identities; all rows retained.",
    "timing": "Saved descriptive timings; verification, I/O and later resampling are excluded. "
              "Fit work differs between implementations. These are not scalability benchmarks.",
    "remaining": "The planned primary paired resampling analysis and figures are not performed here.",
}


def remember_file(path, hashes):
    require(path.is_file(), f"Missing evidence: {path}")
    hashes[str(path.relative_to(ROOT))] = sha256(path)


def remember_bundle(folder, metadata, hashes):
    for name in ["metadata.json", *metadata["output_sha256"]]:
        remember_file(folder / name, hashes)


def collect_warnings(records, condition, component, warnings):
    for item in warnings:
        records.append({"condition": condition, "component": component, **item})


def check_profile_reports(groups, members, saved, y_train, y_test):
    """Cross-check saved group counts and member labels; do not rebuild profiles."""
    require(groups.profile_id.is_unique, "Repeated profile identifiers in group report.")
    require(not members.duplicated(["partition", "row_index_zero_based"]).any(),
            "A row appears in more than one duplicate group.")
    require(set(members.profile_id) == set(groups.profile_id), "Duplicate group membership differs.")
    for record in groups.to_dict("records"):
        current = members[members.profile_id.eq(record["profile_id"])]
        require(len(current) == record["total_rows"] >= 2, "Duplicate group size differs.")
        require(current.partition.isin(["train", "test"]).all(), "Invalid duplicate partition.")
        for partition, labels in [("train", y_train), ("test", y_test)]:
            subset = current[current.partition.eq(partition)]
            indices = subset.row_index_zero_based.to_numpy()
            require(np.isfinite(indices).all() and np.equal(indices, np.floor(indices)).all()
                    and ((indices >= 0) & (indices < len(labels))).all(), "Invalid duplicate row IDs.")
            expected = labels[indices.astype(int)]
            require(np.array_equal(subset.label, expected)
                    and np.array_equal(subset["class"], np.where(expected == 1, "pos", "neg")),
                    "Duplicate member labels do not match the source rows.")
            require(record[partition + "_rows"] == len(subset)
                    and record[partition + "_pos"] == int(expected.sum())
                    and record[partition + "_neg"] == int((expected == 0).sum()),
                    "Duplicate group class counts differ.")
        require(record["shared_across_split"] == bool(record["train_rows"] and record["test_rows"])
                and record["conflicting_labels"] == bool(
                    (record["train_pos"] + record["test_pos"]) and (record["train_neg"] + record["test_neg"]))
                and record["cross_split_label_disagreement"] == bool(
                    record["train_pos"] * record["test_neg"] or record["train_neg"] * record["test_pos"]),
                "Duplicate group flags differ from member labels.")
    expected_summary = {}
    for partition in ["train", "test"]:
        repeated = groups[groups[partition + "_rows"] >= 2]
        counts = repeated[partition + "_rows"]
        expected_summary[partition] = {
            "repeated_profiles": len(repeated), "rows_in_repeated_profiles": int(counts.sum()),
            "extra_copies_beyond_first": int((counts - 1).sum()),
            "conflicting_label_profiles": int(((repeated[partition + "_pos"] > 0)
                                               & (repeated[partition + "_neg"] > 0)).sum()),
        }
    shared = groups[groups.shared_across_split.eq(True)]
    expected_summary["across_supplied_split"] = {
        "shared_profiles": len(shared), "train_rows_in_shared_profiles": int(shared.train_rows.sum()),
        "test_rows_in_shared_profiles": int(shared.test_rows.sum()),
        "shared_profiles_with_cross_split_label_disagreement": int(shared.cross_split_label_disagreement.sum()),
    }
    extras = int((groups.total_rows - 1).sum())
    expected_summary["combined"] = {
        "unique_profiles": len(y_train) + len(y_test) - extras,
        "repeated_profiles": len(groups), "rows_in_repeated_profiles": len(members),
        "extra_copies_beyond_first": extras, "conflicting_label_profiles": int(groups.conflicting_labels.sum()),
    }
    require(expected_summary == saved, "Duplicate tables and saved summary disagree.")
    flat = {f"{section}_{key}": value for section, values in saved.items() for key, value in values.items()}
    flat.update({"shared_train_positive_rows": int(shared.train_pos.sum()),
                 "shared_test_positive_rows": int(shared.test_pos.sum())})
    return flat


def check_preprocessing(folder, condition, loaded, manifest_hash, hashes, warning_rows):
    train, y_train = loaded["train"]
    test, y_test = loaded["test"]
    config = {"run_manifest_sha256": manifest_hash, "condition": condition,
              "missing_fraction_exclusion": None if condition == "primary" else "strictly_greater_than_0.50",
              "imputer": {"strategy": "median", "add_indicator": False},
              "scaler": "StandardScaler fitted only on imputed training measurements"}
    metadata, tables = evaluation.read_bundle(folder, config, PREPROCESSING_FILES)
    remember_bundle(folder, metadata, hashes)
    detail, decisions, parameters = metadata["details"], tables["feature_decisions.csv"], tables["preprocessing_parameters.csv"]
    facts = detail["facts"]
    require(detail["checks"] == "passed" and detail["all_rows_retained"] is True,
            "Preprocessing checks/all-row retention were not recorded as passed.")
    require(facts["train_labels_sha256"] == evaluation.array_sha256(y_train)
            and facts["test_labels_sha256"] == evaluation.array_sha256(y_test), "Saved label hashes differ.")
    unique, missing = train.nunique(dropna=True), train.isna().sum()
    reasons = pd.Series("keep", index=train.columns)
    reasons.loc[unique == 0] = "all_missing_in_training"
    reasons.loc[unique == 1] = "one_observed_value_in_training"
    if condition == "drop_high_missingness":
        reasons.loc[(unique > 1) & (missing / len(train) > 0.50)] = "training_missing_fraction_gt_0.50"
    columns = reasons.index[reasons.eq("keep")].tolist()
    require(len(columns) == (169 if condition == "primary" else 161), "Unexpected retained feature count.")
    require(decisions.feature.tolist() == train.columns.tolist()
            and decisions.reason.tolist() == reasons.tolist()
            and np.array_equal(decisions.retained, reasons.eq("keep"))
            and np.array_equal(decisions.train_unique_observed, unique)
            and np.array_equal(decisions.train_missing, missing)
            and np.array_equal(decisions.test_missing, test.isna().sum()), "Saved feature decisions differ.")
    np.testing.assert_allclose(decisions.train_missing_fraction, missing / len(train), rtol=0, atol=1e-15)
    require(facts["ordered_predictors"] == columns == parameters.feature.tolist(), "Retained feature order differs.")
    require(np.isfinite(parameters.drop(columns="feature").to_numpy()).all()
            and (parameters.scaler_scale > 0).all() and (parameters.scaler_variance > 0).all(),
            "Invalid saved preprocessing parameters.")
    require(set(detail["timings_seconds"]) == {"feature_selection", "imputation", "scaling"}
            and all(np.isfinite(x) and x >= 0 for x in detail["timings_seconds"].values()), "Invalid preprocessing times.")
    collect_warnings(warning_rows, condition, "preprocessing", detail["warnings"])
    profile_rows = []
    for stage, key in [("selected", "selected_raw_profile_audit"), ("prepared", "prepared_profile_audit")]:
        profile_rows.append({"condition": condition, "stage": stage,
                             **check_profile_reports(tables[stage + "_duplicate_groups.csv"],
                                                     tables[stage + "_duplicate_members.csv"],
                                                     facts[key], y_train, y_test)})
    summary = {"condition": condition, "n_train": len(train), "n_test": len(test),
               "n_features": len(columns), "excluded_features": ";".join(reasons.index[reasons.ne("keep")]),
               "training_missing_cells_in_retained_features": int(missing[columns].sum()),
               "test_missing_cells_in_retained_features": int(test[columns].isna().sum().sum()),
               **{name + "_seconds": value for name, value in detail["timings_seconds"].items()},
               "warning_count": len(detail["warnings"]), "checks": "passed"}
    return columns, summary, profile_rows


def check_numerics(metadata, diagnostics, columns, condition):
    detail = metadata["details"]
    require(detail["training_score_agreement"] == detail["fixed_batch_invariance"] == "passed",
            "COPOD training agreement or batch invariance did not pass.")
    # The original allclose training check includes a relative tolerance; do not replace it with an absolute-only gate.
    training_gap = detail["training_score_max_abs_difference"]
    require(np.isfinite(training_gap) and training_gap >= 0, "Invalid recorded training difference.")
    require(all(np.isfinite(detail[key]) and 0 <= detail[key] <= 1e-12 for key in
                ["fixed_chunk_max_abs_difference", "fixed_singleton_max_abs_difference"]), "Batch invariance exceeded tolerance.")
    for key in ["nonfinite_training_contributions", "nonfinite_combined_contributions"]:
        require(set(detail[key]) == {"U_l", "U_r", "U_skew", "O"} and all(v == 0 for v in detail[key].values()),
                "Non-finite COPOD contributions were recorded.")
    require(diagnostics.feature.tolist() == columns, "Diagnostic feature order differs.")
    signs = diagnostics[["training_skewness_sign", "combined_skewness_sign"]]
    require(signs.isin([-1, 0, 1]).all().all(), "Invalid skewness signs.")
    changed = diagnostics.training_skewness_sign.ne(diagnostics.combined_skewness_sign)
    require(np.array_equal(changed, diagnostics.skewness_sign_changed)
            and int(changed.sum()) == detail["skewness_sign_changes"], "Skewness-change counts differ.")
    count_columns = ["test_below_train_min", "test_above_train_max", "left_zero_probability_replacements",
                     "right_zero_probability_replacements", "left_already_at_floor_not_replaced", "right_already_at_floor_not_replaced"]
    counts = diagnostics[count_columns].to_numpy()
    require(np.isfinite(counts).all() and (counts == np.floor(counts)).all()
            and ((counts >= 0) & (counts <= 16000)).all(), "Invalid feature-tail counts.")
    require(np.array_equal(diagnostics.test_below_train_min, diagnostics.left_zero_probability_replacements)
            and np.array_equal(diagnostics.test_above_train_max, diagnostics.right_zero_probability_replacements),
            "Out-of-range and zero-tail counts differ.")
    replacements = int(diagnostics.left_zero_probability_replacements.sum() + diagnostics.right_zero_probability_replacements.sum())
    require(replacements == detail["zero_tail_probability_replacements"], "Tail-replacement summary differs.")
    return {"condition": condition, "component": "copod_pair", "checks": "passed",
            "nonfinite_final_scores": detail["nonfinite_final_scores"], "warning_count": len(detail["warnings"]),
            "training_score_max_abs_difference": training_gap,
            "fixed_chunk_max_abs_difference": detail["fixed_chunk_max_abs_difference"],
            "fixed_singleton_max_abs_difference": detail["fixed_singleton_max_abs_difference"],
            "skewness_sign_changes": int(changed.sum()), "zero_tail_replacement_events": replacements,
            "already_at_floor_tail_events_not_replaced": int(diagnostics.left_already_at_floor_not_replaced.sum()
                                                            + diagnostics.right_already_at_floor_not_replaced.sum())}


def load_verified_results(run_name="v1"):
    """Read-only evidence loader, also reusable for the later paired resampling."""
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_name) is not None, "Invalid run name.")
    folder = ROOT / "results/aps_full_evaluation" / run_name
    manifest_path, fixture_path = folder / "run_manifest.json", folder / "implementation_checks.json"
    require(manifest_path.is_file() and fixture_path.is_file(), "Complete the full APS evaluation first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["schema"] == evaluation.SCHEMA and manifest["run_name"] == run_name
            and manifest["plan_commit"] == evaluation.PLAN_COMMIT and manifest["plan_sha256"] == evaluation.PLAN_SHA256
            and manifest["planned_conditions"] == list(evaluation.CONDITIONS)
            and manifest["isolation_forest_seeds"] == list(evaluation.IF_SEEDS)
            and manifest["isolation_forest_parameters"] == evaluation.IF_PARAMS
            and isinstance(manifest["isolation_forest_parameters"]["max_features"], float)
            and manifest["copod_parameters"] == {"contamination": 0.1, "n_jobs": 1}, "Evaluation manifest differs from frozen plan.")
    require(manifest["versions"]["pyod"] == "3.6.4", "Unexpected recorded PyOD version.")
    evaluation.check_input_hashes(manifest["input_sha256"])
    loaded, _ = evaluation.load_inputs()
    y_train, y_test = loaded["train"][1], loaded["test"][1]
    hashes = dict(manifest["input_sha256"])
    remember_file(manifest_path, hashes)
    remember_file(fixture_path, hashes)
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    require(fixture["result"]["status"] == "passed", "Small implementation checks did not pass.")
    warning_rows = []
    collect_warnings(warning_rows, "all", "implementation_fixture", fixture["warnings"])
    metric_frames, preprocessing_rows, duplicate_rows, numerical_rows, score_tables = [], [], [], [], {}
    for condition in evaluation.CONDITIONS:
        current = folder / condition
        columns, preproc, duplicates = check_preprocessing(current / "preprocessing", condition, loaded,
                                                           sha256(manifest_path), hashes, warning_rows)
        preprocessing_rows.append(preproc)
        duplicate_rows.extend(duplicates)
        base = {"run_manifest_sha256": sha256(manifest_path), "condition": condition,
                "preprocessing_metadata_sha256": sha256(current / "preprocessing/metadata.json"),
                "implementation_checks_sha256": sha256(fixture_path)}
        for name, seed in [("copod", None)] + [(f"iforest_seed{s}", s) for s in evaluation.IF_SEEDS]:
            methods = list(METHODS[:2]) if seed is None else ["isolation_forest"]
            extra = ["feature_diagnostics.csv"] if seed is None else []
            config = {**base, **({"methods": methods, "standard_parameters": {"contamination": 0.1, "n_jobs": 1},
                                  "modified_probability_floor": 1 / len(y_train), "batch_call_rows": len(y_test)}
                                 if seed is None else {"method": "isolation_forest", "parameters": evaluation.IF_PARAMS,
                                                      "random_state": seed})}
            metadata, tables = evaluation.verify_model_bundle(current / name, config, methods, condition,
                                                               y_train, y_test, len(columns), extra)
            require(metadata["details"]["nonfinite_final_scores"] == 0, "Non-finite scores were recorded.")
            remember_bundle(current / name, metadata, hashes)
            collect_warnings(warning_rows, condition, name, metadata["details"]["warnings"])
            metric_frames.append(tables["metrics.csv"])
            score_tables[(condition, name)] = tables["scores.csv"]
            if seed is None:
                numerical_rows.append(check_numerics(metadata, tables["feature_diagnostics.csv"], columns, condition))
            else:
                details = metadata["details"]
                require(all(details["fitted_parameters"][key] == value for key, value in evaluation.IF_PARAMS.items())
                        and details["fitted_parameters"]["random_state"] == seed and details["effective_max_samples"] == 256,
                        "Saved fitted Isolation Forest parameters differ.")
                numerical_rows.append({"condition": condition, "component": name, "checks": "passed",
                                       "nonfinite_final_scores": 0, "warning_count": len(details["warnings"])})
        print(f"Verified {condition}: {len(columns)} predictors, two COPOD results and five Isolation Forest seeds.", flush=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics["model_seed"] = metrics.model_seed.astype("Int64")
    require(len(metrics) == 14 and not metrics.duplicated(["condition", "method", "model_seed"]).any(),
            "Expected exactly 14 distinct model-condition results.")
    require(metrics.loc[metrics.condition.eq("drop_high_missingness"), "precision_at_375"].isna().all(),
            "Secondary Precision@375 was not part of the frozen plan.")
    return {"folder": folder, "manifest": manifest, "input_sha256": hashes, "metrics": metrics,
            "scores": score_tables, "labels": y_test,
            "preprocessing": pd.DataFrame(preprocessing_rows), "duplicates": pd.DataFrame(duplicate_rows),
            "numerics": pd.DataFrame(numerical_rows),
            "warnings": pd.DataFrame(warning_rows, columns=["condition", "component", "stage", "category", "message"])}


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    require(np.isfinite(values).all(), "Non-finite values entered a metric summary.")
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else np.nan


def build_summaries(metrics):
    performance_rows, protocol_rows, paired_rows, sensitivity_rows = [], [], [], []
    for condition in evaluation.CONDITIONS:
        current = metrics[metrics.condition.eq(condition)]
        for method in METHODS:
            group = current[current.method.eq(method)]
            expected_runs = 5 if method == "isolation_forest" else 1
            require(len(group) == expected_runs, "Unexpected method run count.")
            record = {"condition": condition, "method": method, "n_runs": len(group),
                      "n_features": int(group.n_features.iloc[0]), "test_prevalence": float(group.test_prevalence.iloc[0]),
                      "score_reference": "training_plus_test_batch" if method == "pyod_batch" else "training_only",
                      "sd_interpretation": "variation_across_five_model_seeds" if len(group) > 1 else "not_applicable_one_deterministic_run"}
            for metric in ["roc_auc", "average_precision", "precision_at_375", "fit_seconds", "score_seconds"]:
                values = [] if metric == "precision_at_375" and condition != "primary" else group[metric].to_numpy()
                record[metric + "_mean"], record[metric + "_sd"] = mean_sd(values)
            performance_rows.append(record)
        copod = current[current.method.isin(METHODS[:2])].set_index("method")
        for metric in ["roc_auc", "average_precision"] + (["precision_at_375"] if condition == "primary" else []):
            batch, fixed = float(copod.loc["pyod_batch", metric]), float(copod.loc["fixed_training_reference", metric])
            protocol_rows.append({"condition": condition, "metric": metric, "pyod_batch": batch,
                                  "fixed_training_reference": fixed, "fixed_minus_batch": fixed - batch})
    for method in METHODS:
        seeds = evaluation.IF_SEEDS if method == "isolation_forest" else [None]
        method_pairs = []
        for seed in seeds:
            selected = metrics[metrics.method.eq(method) & (metrics.model_seed.isna() if seed is None else metrics.model_seed.eq(seed))]
            require(len(selected) == 2 and set(selected.condition) == set(evaluation.CONDITIONS), "Incomplete paired conditions.")
            selected = selected.set_index("condition")
            record = {"method": method, "model_seed": seed}
            for metric in ["roc_auc", "average_precision"]:
                primary, secondary = float(selected.loc["primary", metric]), float(selected.loc["drop_high_missingness", metric])
                record[metric + "_primary"] = primary
                record[metric + "_secondary"] = secondary
                record[metric + "_secondary_minus_primary"] = secondary - primary
            method_pairs.append(record)
        paired_rows.extend(method_pairs)
        summary = {"method": method, "n_pairs": len(method_pairs)}
        for metric in ["roc_auc", "average_precision"]:
            values = [row[metric + "_secondary_minus_primary"] for row in method_pairs]
            summary[metric + "_difference_mean"], summary[metric + "_difference_sd"] = mean_sd(values)
        sensitivity_rows.append(summary)
    paired = pd.DataFrame(paired_rows)
    paired["model_seed"] = paired.model_seed.astype("Int64")
    return {"performance_summary.csv": pd.DataFrame(performance_rows),
            "copod_protocol_differences.csv": pd.DataFrame(protocol_rows),
            "missingness_paired_differences.csv": paired,
            "missingness_summary.csv": pd.DataFrame(sensitivity_rows)}


def save_or_verify_summary(folder, tables, evidence):
    inputs = dict(evidence["input_sha256"])
    remember_file(Path(__file__).resolve(), inputs)
    evaluation.check_input_hashes(inputs)
    payloads = {name: table.to_csv(index=False).encode("utf-8") for name, table in tables.items()}
    metadata = {"schema": "aps_full_summary_v1", "input_sha256": inputs,
                "evaluation_versions": evidence["manifest"]["versions"],
                "summary_versions": {"python": platform.python_version(), **{name: version(name) for name in
                                      ["numpy", "pandas", "scipy", "scikit-learn", "pyod"]}},
                "checks": "passed", "verified_model_results": 14,
                "verification_scope": "Verified input/output hashes, source labels, column decisions, saved audit consistency, "
                                      "numerical-check records, and metrics recomputed from scores. No model fitting, "
                                      "preprocessing refit or independent reconstruction of stored score contributions.",
                "interpretation": INTERPRETATION,
                "output_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}}
    payloads["summary_metadata.json"] = evaluation.json_bytes(metadata)
    if folder.exists():
        require({path.name for path in folder.iterdir()} <= set(payloads), "Unexpected existing summary files.")
    for name, content in payloads.items():
        path = folder / name
        require(not path.exists() or path.read_bytes() == content,
                f"Existing summary differs: {path}. Inspect the change; do not overwrite evidence.")
    folder.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        path = folder / name
        if not path.exists():
            path.write_bytes(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="v1")
    args = parser.parse_args()
    evidence = load_verified_results(args.run_name)
    tables = {"all_metrics.csv": evidence["metrics"], **build_summaries(evidence["metrics"]),
              "preprocessing_summary.csv": evidence["preprocessing"], "duplicate_summary.csv": evidence["duplicates"],
              "numerical_diagnostics.csv": evidence["numerics"], "warning_log.csv": evidence["warnings"]}
    destination = evidence["folder"] / "summary"
    save_or_verify_summary(destination, tables, evidence)
    print("\nVerified 14 model-condition results; no models were run.")
    print("\nPerformance: IF means and sample SD across five seeds; COPOD SD is not applicable.")
    display = ["condition", "method", "n_runs", "roc_auc_mean", "roc_auc_sd", "average_precision_mean", "average_precision_sd"]
    print(tables["performance_summary.csv"][display].to_string(index=False, float_format=lambda x: f"{x:.6f}", na_rep="--"))
    print("\nPrimary Precision@375 (fractional cutoff ties; retrospective diagnostic):")
    performance = tables["performance_summary.csv"]
    print(performance.loc[performance.condition.eq("primary"),
                          ["method", "precision_at_375_mean", "precision_at_375_sd"]].to_string(
                              index=False, float_format=lambda x: f"{x:.6f}", na_rep="--"))
    print("\nMissingness sensitivity: secondary minus primary; IF seeds paired.")
    print(tables["missingness_summary.csv"].to_string(index=False, float_format=lambda x: f"{x:.6f}", na_rep="--"))
    print(f"\nRecorded warnings: {len(evidence['warnings'])}. Review duplicate and numerical summary tables.")
    print("SD is not a confidence interval. Paired resampling and figures remain separate steps.")
    print(f"Saved or verified nine CSV files and summary_metadata.json: {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
