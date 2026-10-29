"""Verify and summarise all 70 synthetic-extension datasets and 210 results.

Run: python scripts/summarise_synthetic_extension.py
Requires all saved generation/evaluation evidence and raw CSVs for seeds 100-109.
No data generation, detector fitting, resampling or performance-based exclusions.
Means and sample SD describe data-seed variation; paired differences are computed
within each seed before summarisation. The two families' controls stay separate.
"""

import argparse
import inspect
import io
import json
import platform
import re
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_synthetic_extension as evaluation
import generate_synthetic_extension as generation


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "synthetic_extension_summary_v1"
METHODS = tuple(evaluation.METHODS)
SEEDS = tuple(generation.PLANNED_SEEDS)
CONDITIONS = tuple(generation.CONDITIONS)
METRICS = ("roc_auc", "average_precision")
require, digest = generation.require, generation.digest


def check_numerical_evidence(case, tables, details, seed):
    """Check stored diagnostics against saved arrays, without fitting any detector."""
    require(details["checks"] == details["training_score_agreement"] == "passed",
            "Missing successful numerical checks.")
    for key in ("nonfinite_pyod_training_contributions", "nonfinite_pyod_combined_contributions"):
        require(set(details[key]) == {"U_l", "U_r", "U_skew", "O"}
                and all(value == 0 for value in details[key].values()), "Recorded non-finite COPOD contributions.")
    require(details["nonfinite_fixed_test_contributions"] == 0
            and set(details["nonfinite_final_scores"]) == set(METHODS)
            and all(value == 0 for value in details["nonfinite_final_scores"].values()),
            "Recorded non-finite final contributions or scores.")
    gap = details["training_score_max_abs_difference"]
    require(np.isfinite(gap) and gap >= 0, "Invalid training-score difference.")
    # The training and LW checks used relative tolerances; a maximum absolute
    # difference alone cannot reconstruct their acceptance limits.
    for key in ("fixed_batch_invariance", "ledoit_wolf_batch_invariance"):
        record = details[key]
        require(record["status"] == "passed", "Recorded batch check failed.")
        values = [record["chunk_max_abs_difference"], record["singleton_max_abs_difference"]]
        require(np.isfinite(values).all() and min(values) >= 0, "Invalid batch differences.")
        if key == "fixed_batch_invariance":
            require(max(values) <= 1e-12, "Fixed-reference batch check exceeded its absolute tolerance.")
    X_train, X_test = case["X"][case["train_rows"]], case["X"][case["test_rows"]]
    train_sign = np.sign(evaluation.skew(X_train, axis=0, bias=True))
    combined_sign = np.sign(evaluation.skew(np.vstack([X_train, X_test]), axis=0, bias=True))
    rows, touched = [], np.zeros(4000, dtype=bool)
    for j, feature in enumerate(generation.FEATURES):
        reference = np.sort(X_train[:, j])
        left = np.searchsorted(reference, X_test[:, j], side="right")
        right = 6000 - np.searchsorted(reference, X_test[:, j], side="left")
        touched |= (left == 0) | (right == 0)
        rows.append({"feature": feature, "training_skewness_sign": int(train_sign[j]),
                     "combined_skewness_sign": int(combined_sign[j]),
                     "skewness_sign_changed": bool(train_sign[j] != combined_sign[j]),
                     "training_min": float(reference[0]), "training_max": float(reference[-1]),
                     "left_zero_tail_replacements": int((left == 0).sum()),
                     "right_zero_tail_replacements": int((right == 0).sum()),
                     "left_already_at_floor": int((left == 1).sum()),
                     "right_already_at_floor": int((right == 1).sum())})
    expected = pd.DataFrame(rows)
    evaluation.assert_table(tables["feature_diagnostics.csv"], expected, "feature diagnostics")
    zeros = int(expected[["left_zero_tail_replacements", "right_zero_tail_replacements"]].to_numpy().sum())
    floor = int(expected[["left_already_at_floor", "right_already_at_floor"]].to_numpy().sum())
    changes = int(expected.skewness_sign_changed.sum())
    require(details["fixed_zero_tail_replacement_cells"] == zeros
            and details["fixed_tail_probabilities_already_at_floor"] == floor
            and details["fixed_test_rows_with_zero_tail_replacement"] == int(touched.sum())
            and details["skewness_sign_changes"] == changes, "Recorded tail or skewness totals differ.")
    lw, eigen = details["ledoit_wolf"], tables["ledoit_wolf_eigenvalues.csv"]
    require(np.isfinite(lw["shrinkage"]) and 0 <= lw["shrinkage"] <= 1
            and np.isfinite(lw["formula_max_abs_difference"]) and lw["formula_max_abs_difference"] >= 0
            and lw["score_formula"] == "(x-location)^T precision (x-location)", "Invalid covariance diagnostics.")
    ce, pe = eigen.covariance_eigenvalue.to_numpy(), eigen.precision_eigenvalue.to_numpy()
    for key, value in {"covariance_min_eigenvalue": ce[0], "covariance_max_eigenvalue": ce[-1],
                       "covariance_spectral_condition_number": ce[-1]/ce[0],
                       "precision_spectral_condition_number": pe[-1]/pe[0]}.items():
        np.testing.assert_allclose(lw[key], value, rtol=1e-10, atol=1e-12)
    residuals = X_train - tables["ledoit_wolf_location.csv"].location.to_numpy()
    empirical = residuals.T @ residuals / len(X_train)
    expected_covariance = ((1-lw["shrinkage"])*empirical
                           + lw["shrinkage"]*np.trace(empirical)/20*np.eye(20))
    np.testing.assert_allclose(tables["ledoit_wolf_matrices.csv"].covariance.to_numpy().reshape(20, 20),
                               expected_covariance, rtol=1e-10, atol=1e-12)
    return {"data_seed": seed, **case["spec"], "checks": "passed",
            "training_score_max_abs_difference": gap,
            "fixed_chunk_max_abs_difference": details["fixed_batch_invariance"]["chunk_max_abs_difference"],
            "fixed_singleton_max_abs_difference": details["fixed_batch_invariance"]["singleton_max_abs_difference"],
            "ledoit_wolf_chunk_max_abs_difference": details["ledoit_wolf_batch_invariance"]["chunk_max_abs_difference"],
            "ledoit_wolf_singleton_max_abs_difference": details["ledoit_wolf_batch_invariance"]["singleton_max_abs_difference"],
            "nonfinite_contributions_or_scores": 0, "skewness_sign_changes": changes,
            "zero_tail_replacement_cells": zeros, "tail_probabilities_already_at_floor": floor,
            "test_rows_with_zero_tail_replacement": int(touched.sum()),
            "ledoit_wolf_shrinkage": lw["shrinkage"],
            "ledoit_wolf_covariance_condition_number": lw["covariance_spectral_condition_number"],
            "ledoit_wolf_covariance_min_eigenvalue": ce[0],
            "ledoit_wolf_covariance_max_eigenvalue": ce[-1],
            "ledoit_wolf_formula_max_abs_difference": lw["formula_max_abs_difference"],
            "model_warning_count": len(details["warnings"])}


def load_verified_results(run_name):
    """Verify the complete ten-seed evidence set without invoking any fit method."""
    base = ROOT / "results/synthetic_extension" / run_name
    hashes, warning_rows = {}, []

    def remember(path, expected=None):
        require(path.is_file(), f"Missing required evidence: {path}")
        content = path.read_bytes()
        key, actual = path.relative_to(ROOT).as_posix(), digest(content)
        require(expected is None or actual == expected, f"Evidence checksum differs: {path}")
        require(key not in hashes or hashes[key] == actual, f"Evidence changed during verification: {path}")
        hashes[key] = actual
        return content

    def collect_warnings(scope, seed, condition, records):
        require(isinstance(records, list), "Warning record must be a list.")
        for item in records:
            require(isinstance(item, dict) and "category" in item and "message" in item,
                    "Invalid saved warning record.")
            warning_rows.append({"scope": scope, "data_seed": seed, "condition": condition,
                                 "stage": item.get("stage", scope), "category": item["category"],
                                 "message": item["message"], "source_file": item.get("source_file", ""),
                                 "source_line": item.get("source_line", "")})

    manifest_path = base / "evaluation_manifest.json"
    manifest = json.loads(remember(manifest_path))
    manifest_hash = hashes[manifest_path.relative_to(ROOT).as_posix()]
    expected_fields = {"schema": evaluation.SCHEMA, "run_name": run_name,
        "plan_path": generation.PLAN_PATH, "plan_commit": generation.PLAN_COMMIT,
        "plan_sha256": generation.PLAN_SHA256, "planned_data_seeds": list(generation.PLANNED_SEEDS),
        "conditions": list(generation.CONDITIONS), "methods": list(METHODS),
        "copod_parameters": evaluation.COPOD_PARAMETERS, "ledoit_wolf_parameters": evaluation.LW_PARAMETERS,
        "fixed_reference_probability_floor": 1/6000, "preprocessing": "none", "labels_in_fit": False,
        "training_rows": 6000, "training_positives_retained": 300, "test_rows": 4000, "test_positives": 200,
        "ordered_feature_columns": generation.FEATURES, "metrics": ["roc_auc", "average_precision"],
        "metrics_use": "test rows only, raw scores", "performance_acceptance_thresholds": None,
        "null_references": {"roc_auc": .5, "average_precision_prevalence": .05},
        "scoring_references": {"pyod_batch": "training plus entire test batch in one call",
                               "fixed_training_reference": "training only",
                               "ledoit_wolf": "training only; squared Mahalanobis distances"},
        "tolerances": {"training_agreement": {"rtol": 1e-10, "atol": 1e-10},
                       "fixed_batch": {"rtol": 0, "atol": 1e-12},
                       "ledoit_wolf_batch_formula": {"rtol": 1e-10, "atol": 1e-12}}}
    require(all(manifest.get(key) == value for key, value in expected_fields.items()),
            "Evaluation manifest differs from the frozen design.")
    require(manifest["runtime"]["packages"]["pyod"] == "3.6.4", "Unexpected recorded PyOD version.")
    # Historical dependency versions are retained; verification has its own runtime record.
    sources = {"evaluator": Path(evaluation.__file__).resolve(),
               "generator": Path(generation.__file__).resolve(),
               "fixed_reference_module": Path(inspect.getsourcefile(evaluation.TrainingReferenceCOPOD)).resolve()}
    for key, path in sources.items():
        remember(path, manifest["source_sha256"][key])
    require(manifest["fixed_reference_class_sha256"] == evaluation.FIXED_CLASS_SHA256
            == digest(inspect.getsource(evaluation.TrainingReferenceCOPOD).encode("utf-8")),
            "Fixed-reference scoring class has changed.")
    remember(ROOT / generation.PLAN_PATH, generation.PLAN_SHA256)
    generation_record = json.loads(remember(base / "generation_manifest.json", manifest["generation_manifest_sha256"]))
    require(manifest["generation_versions"] == generation_record["versions"]
            and manifest["split"] == generation_record["split"], "Generation/evaluation provenance differs.")
    fixture_path = base / "implementation_checks.json"
    fixture = json.loads(remember(fixture_path))
    fixture_hash = hashes[fixture_path.relative_to(ROOT).as_posix()]
    require(fixture["schema"] == evaluation.SCHEMA and fixture["evaluation_manifest_sha256"] == manifest_hash
            and all(fixture[k] == "passed" for k in ("status", "copod_inclusive_tails_ties_and_skew",
                "ledoit_wolf_centering_shrinkage_squared_distance", "tied_auc_and_noninterpolated_ap"))
            and fixture["ledoit_wolf_fixture_expected_scores"] == [0, .6, 1.2, 1.8]
            and fixture["ledoit_wolf_fixture_expected_shrinkage"] == 4/9, "Incomplete saved implementation checks.")
    collect_warnings("implementation_checks", None, "", fixture["warnings"])
    names = [spec["condition"] for spec in generation.CONDITIONS]
    require({p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("seed")}
            == {f"seed{seed}" for seed in generation.PLANNED_SEEDS}, "Missing or unexpected seed directory.")
    all_rows, numerical_rows, inventory_rows = [], [], []
    for seed in generation.PLANNED_SEEDS:
        evidence, current_warnings = evaluation.capture_warnings(
            f"summary_verification_seed{seed}", lambda: evaluation.load_evidence(run_name, seed))
        for key, expected in evidence["input_sha256"].items():
            remember(ROOT / key, expected)
        folder = evidence["result_dir"]
        completion = json.loads(remember(folder / "evaluation_metadata.json"))
        require(completion["schema"] == evaluation.SCHEMA and completion["data_seed"] == seed
                and completion["evaluation_manifest_sha256"] == manifest_hash
                and completion["implementation_checks_sha256"] == fixture_hash
                and completion["input_sha256"] == evidence["input_sha256"]
                and completion["completed_conditions"] == names and completion["methods"] == list(METHODS)
                and completion["metric_rows"] == 21 and completion["checks"] == "passed"
                and completion["generation_warnings"] == evidence["generation_warnings"],
                f"Incomplete/mismatched seed completion record: {seed}")
        expected_paths = {folder / "evaluation_metrics.csv"}
        for name in names:
            expected_paths.update(folder / name / "evaluation" / item
                                  for item in (*evaluation.TABLE_NAMES, "metadata.json"))
        require(set(completion["output_sha256"]) == {p.relative_to(ROOT).as_posix() for p in expected_paths}
                and set(folder.glob("*/evaluation")) == {folder/name/"evaluation" for name in names},
                f"Evaluation inventory differs: seed {seed}")
        for key, expected in completion["output_sha256"].items():
            remember(ROOT / key, expected)
        collect_warnings("generation", seed, "", completion["generation_warnings"])
        collect_warnings("evaluation_input_verification", seed, "", completion["input_verification_warnings"])
        collect_warnings("summary_input_verification", seed, "", current_warnings)
        seed_rows, model_warning_count = [], 0
        for name, case in evidence["cases"].items():
            target = folder / name / "evaluation"
            config = evaluation.bundle_config(evidence, case, seed, manifest_hash, fixture_hash)
            tables, metadata = evaluation.read_bundle(target, config, case, seed)
            detail = metadata["details"]
            diagnostics, current_warnings = evaluation.capture_warnings(
                f"summary_numerics_seed{seed}/{name}", lambda: check_numerical_evidence(case, tables, detail, seed))
            numerical_rows.append(diagnostics)
            collect_warnings("model_evaluation", seed, name, detail["warnings"])
            collect_warnings("summary_numerical_verification", seed, name, current_warnings)
            model_warning_count += len(detail["warnings"])
            seed_rows.append(tables["metrics.csv"])
            inventory_rows.append({"data_seed": seed, **case["spec"], "n_methods": 3,
                                   "n_test": 4000, "test_positives": 200,
                                   "train_row_order_sha256": generation.array_digest(case["train_rows"]),
                                   "test_row_order_sha256": generation.array_digest(case["test_rows"]),
                                   "evaluation_metadata_sha256": hashes[(target/"metadata.json").relative_to(ROOT).as_posix()],
                                   "checks": "passed"})
        combined = pd.concat(seed_rows, ignore_index=True)
        saved = pd.read_csv(io.BytesIO(remember(folder / "evaluation_metrics.csv")), float_precision="round_trip")
        evaluation.assert_table(saved, combined, f"seed {seed} metrics")
        require(len(combined) == 21 and model_warning_count == completion["model_warning_count"],
                f"Seed metric/warning counts differ: {seed}")
        all_rows.append(combined)
        print(f"Verified seed {seed}: seven settings, 21 results; no models fitted.", flush=True)
    metrics = pd.concat(all_rows, ignore_index=True)
    actual = list(metrics[["data_seed", "condition", "method"]].itertuples(index=False, name=None))
    expected = {(seed, name, method) for seed in generation.PLANNED_SEEDS for name in names for method in METHODS}
    require(len(actual) == len(set(actual)) == 210 and set(actual) == expected,
            "Expected exactly 210 distinct planned seed/condition/method results.")
    for key, expected_hash in list(hashes.items()):
        remember(ROOT/key, expected_hash)
    return {"all_metrics": metrics, "numerical_diagnostics": pd.DataFrame(numerical_rows),
            "inventory": pd.DataFrame(inventory_rows),
            "warnings": pd.DataFrame(warning_rows, columns=["scope", "data_seed", "condition", "stage",
                                   "category", "message", "source_file", "source_line"]),
            "input_sha256": hashes, "manifest": manifest, "fixture": fixture}


def validate_complete_metrics(frame):
    """Reject a missing, repeated or unexpected experimental cell."""
    keys = ["data_seed", "condition", "method"]
    require(all(key in frame for key in keys), "Result keys are missing.")
    expected = {(seed, spec["condition"], method)
                for seed in SEEDS for spec in CONDITIONS for method in METHODS}
    actual = set(frame[keys].itertuples(index=False, name=None))
    require(len(frame) == 210 and not frame.duplicated(keys).any() and actual == expected,
            "Require exactly 210 unique results: ten seeds, seven conditions, three methods.")
    values = frame[list(METRICS)].to_numpy(dtype=float)
    require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), "Invalid metrics.")
    for spec in CONDITIONS:
        selected = frame[frame.condition == spec["condition"]]
        for key, value in spec.items():
            require(selected[key].isna().all() if value is None else selected[key].eq(value).all(),
                    f"Condition definition changed: {spec['condition']}/{key}")
    for key, value in {"n_train": 6000, "n_test": 4000, "n_features": 20,
                       "train_positives": 300, "test_positives": 200, "test_prevalence": .05}.items():
        require(frame[key].eq(value).all(), f"Unexpected evaluation counts: {key}")


def describe(values):
    values = np.asarray(values, dtype=float)
    require(values.shape == (10,) and np.isfinite(values).all(), "Each summary requires ten finite values.")
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def summarise_differences(pairs, keys):
    rows = []
    # Explicit matching avoids merging controls or losing NaN delta values in groupby.
    for key in pairs[keys].drop_duplicates().to_dict("records"):
        mask = np.ones(len(pairs), dtype=bool)
        for name, value in key.items():
            mask &= (pairs[name].isna() if pd.isna(value) else pairs[name].eq(value)).to_numpy()
        current = pairs[mask].sort_values("data_seed")
        require(current.data_seed.tolist() == list(SEEDS), "Paired comparison must contain each seed once.")
        row = {**key, "n_pairs": 10}
        for metric in METRICS:
            differences = current[metric+"_difference"].to_numpy()
            row.update({metric+"_difference_"+name: value for name, value in describe(differences).items()})
            row[metric+"_positive_differences"] = int((differences > 0).sum())
            row[metric+"_negative_differences"] = int((differences < 0).sum())
            row[metric+"_zero_differences"] = int((differences == 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def build_tables(all_metrics):
    validate_complete_metrics(all_metrics)
    lookup = all_metrics.set_index(["data_seed", "condition", "method"])
    canonical = [(seed, spec["condition"], method)
                 for seed in SEEDS for spec in CONDITIONS for method in METHODS]
    all_metrics = lookup.loc[canonical].reset_index()
    performance_rows, method_pairs, strength_pairs = [], [], []
    for spec in CONDITIONS:
        name = spec["condition"]
        for method in METHODS:
            rows = [lookup.loc[(seed, name, method)] for seed in SEEDS]
            row = {**spec, "method": method, "n_seeds": 10,
                   "roc_auc_random_reference": .5, "average_precision_prevalence_reference": .05}
            for metric in METRICS:
                row.update({metric+"_"+key: value for key, value in describe([r[metric] for r in rows]).items()})
            performance_rows.append(row)
        for comparison, left, right in (
                ("fixed_minus_batch", "fixed_training_reference", "pyod_batch"),
                ("ledoit_wolf_minus_fixed", "ledoit_wolf", "fixed_training_reference")):
            for seed in SEEDS:
                row = {"comparison": comparison, **spec, "data_seed": seed,
                       "left_method": left, "right_method": right,
                       "principal_relational_comparison": comparison == "ledoit_wolf_minus_fixed"
                                                          and name == "dependence_rho0p8"}
                for metric in METRICS:
                    a, b = float(lookup.loc[(seed, name, left), metric]), float(lookup.loc[(seed, name, right), metric])
                    row.update({metric+"_left": a, metric+"_right": b, metric+"_difference": a-b})
                method_pairs.append(row)
        if spec["null_control"]:
            continue
        marginal = spec["family"] == "marginal_shift"
        baseline = "marginal_delta0" if marginal else "dependence_rho0p0"
        parameter = "delta" if marginal else "rho"
        for method in METHODS:
            for seed in SEEDS:
                row = {"comparison": name+"_minus_"+baseline, "family": spec["family"],
                       "condition": name, "baseline_condition": baseline, "method": method,
                       "data_seed": seed, "varied_parameter": parameter,
                       "parameter_value": spec[parameter], "baseline_parameter_value": 0}
                for metric in METRICS:
                    a, b = float(lookup.loc[(seed, name, method), metric]), float(lookup.loc[(seed, baseline, method), metric])
                    row.update({metric+"_condition": a, metric+"_baseline": b, metric+"_difference": a-b})
                strength_pairs.append(row)
    method_pairs, strength_pairs = pd.DataFrame(method_pairs), pd.DataFrame(strength_pairs)
    method_keys = ["comparison", "condition", "family", "rho", "delta", "null_control",
                   "left_method", "right_method", "principal_relational_comparison"]
    strength_keys = ["comparison", "family", "condition", "baseline_condition", "method",
                     "varied_parameter", "parameter_value", "baseline_parameter_value"]
    tables = {"all_metrics.csv": all_metrics, "performance_summary.csv": pd.DataFrame(performance_rows),
              "method_paired_differences.csv": method_pairs,
              "method_paired_summary.csv": summarise_differences(method_pairs, method_keys),
              "strength_paired_differences.csv": strength_pairs,
              "strength_paired_summary.csv": summarise_differences(strength_pairs, strength_keys)}
    expected_sizes = {"all_metrics.csv": 210, "performance_summary.csv": 21,
                      "method_paired_differences.csv": 140, "method_paired_summary.csv": 14,
                      "strength_paired_differences.csv": 150, "strength_paired_summary.csv": 15}
    require(all(len(tables[key]) == value for key, value in expected_sizes.items()), "Summary inventory differs.")
    return tables


def check_summary_arithmetic():
    """Check pairing and SD on a deliberately correlated, hand-known example."""
    fixture = pd.DataFrame({"comparison": ["example"]*10, "data_seed": SEEDS,
                            "roc_auc_difference": [.125]*10, "average_precision_difference": [-.25]*10})
    summary = summarise_differences(fixture, ["comparison"]).iloc[0]
    require(summary.n_pairs == 10 and summary.roc_auc_difference_mean == .125
            and summary.roc_auc_difference_sd == 0
            and summary.average_precision_difference_mean == -.25
            and summary.average_precision_difference_sd == 0, "Paired-summary arithmetic failed.")
    np.testing.assert_allclose(describe(np.arange(10.))["sd"], np.sqrt(55/6), rtol=0, atol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="v1", help="Existing generation/evaluation namespace.")
    args = parser.parse_args()
    require(re.fullmatch(r"[a-z][a-z0-9_]{0,47}", args.run_name) is not None, "Invalid run name.")
    check_summary_arithmetic()
    verified, current_warnings = evaluation.capture_warnings("summary_verification", lambda: load_verified_results(args.run_name))
    tables = build_tables(verified["all_metrics"])
    tables.update({"numerical_diagnostics.csv": verified["numerical_diagnostics"],
                   "verified_inventory.csv": verified["inventory"], "warning_log.csv": verified["warnings"]})
    summary_warning_mask = verified["warnings"].scope.str.startswith("summary_")
    historical_warning_count = int((~summary_warning_mask).sum())
    summary_warning_count = int(summary_warning_mask.sum()) + len(current_warnings)
    require(len(tables["numerical_diagnostics.csv"]) == len(tables["verified_inventory.csv"]) == 70,
            "Each dataset requires numerical checks and an inventory entry.")
    destination = ROOT / "results/synthetic_extension" / args.run_name / "summary"
    payloads = {destination/name: generation.csv_bytes(frame) for name, frame in tables.items()}
    metadata = {"schema": SCHEMA, "run_name": args.run_name, "planned_data_seeds": list(SEEDS),
                "conditions": list(CONDITIONS), "methods": list(METHODS),
                "checks": "passed", "datasets_verified": 70, "results_verified": 210,
                "models_fitted": 0, "data_generated": False, "resamples_run": 0,
                "plan_commit": generation.PLAN_COMMIT, "plan_sha256": generation.PLAN_SHA256,
                "summary_script_sha256": digest(Path(__file__).read_bytes()),
                "input_sha256": verified["input_sha256"],
                "original_evaluation_runtime": verified["manifest"]["runtime"],
                "summary_runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "packages": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "pyod")}},
                "historical_warning_count": historical_warning_count,
                "summary_warning_count": summary_warning_count,
                "outer_summary_warnings": current_warnings,
                "interpretation": {
                    "performance": "Full-precision per-seed metrics; every planned result included.",
                    "sd": "Sample standard deviation (ddof=1) across ten data seeds; not a confidence interval.",
                    "method_difference": "Left method minus right method on the same condition, seed and test rows.",
                    "strength_difference": "Non-null strength minus its own family's null control, with the same seed and method.",
                    "pairing": "Compute differences per seed first, then their mean and SD; never subtract arm SDs.",
                    "sign_counts": "Descriptive exact signs of paired differences; no hypothesis tests or equivalence claims.",
                    "controls": "The two families' null controls remain separate. Individual results need not equal .5 or .05.",
                    "scoring_access": "Batch COPOD uses train and test features; fixed COPOD and Ledoit-Wolf use training features only.",
                    "principal_comparison": "Ledoit-Wolf minus fixed-reference COPOD at dependence_rho0p8, with every other condition retained.",
                    "covariance_comparator": "Chosen for these Gaussian mechanisms; no universal-superiority or real-world-transfer claim.",
                    "tail_counts": "Zero-tail cells and distinct affected rows are separate; already-at-floor probabilities are not replacements.",
                    "timing": "Original descriptive times retained in all_metrics; no speed benchmark or timing ranking.",
                    "scope": "Fixed 10,000 rows, 20 features, 5% positives and planned Gaussian settings; no additional dimensions tested."},
                "output_rows": {name: len(frame) for name, frame in tables.items()},
                "output_sha256": {path.name: digest(content) for path, content in payloads.items()}}
    evaluation.check_hashes(verified["input_sha256"])
    payloads[destination/"summary_metadata.json"] = generation.json_bytes(metadata)
    created = generation.save_or_verify(payloads)
    print("\nVerified 70 datasets and 210 results; no models were fitted.")
    print("\nPerformance across ten data seeds (SD = sample standard deviation):")
    columns = ["condition", "method", "n_seeds", "roc_auc_mean", "roc_auc_sd",
               "average_precision_mean", "average_precision_sd"]
    print(tables["performance_summary.csv"][columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nPaired method differences: left method minus right method.")
    columns = ["condition", "comparison", "n_pairs", "roc_auc_difference_mean", "roc_auc_difference_sd",
               "average_precision_difference_mean", "average_precision_difference_sd"]
    print(tables["method_paired_summary.csv"][columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSaved all 15 strength-comparison summaries and their 150 per-seed differences.")
    print(f"Recorded evidence warnings: {historical_warning_count}; current summary warnings: {summary_warning_count}.")
    print("SD describes simulation variation, not a confidence interval. No significance or equivalence tests were run.")
    print(f"Created {created} files; verified {len(payloads)-created} existing files: {destination.relative_to(ROOT)}")
    print("Nine summary CSVs and metadata saved. Original evidence remains unchanged; figures are the next step.")


if __name__ == "__main__":
    main()
