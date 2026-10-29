"""Evaluate the seven frozen synthetic-extension settings on saved datasets.

Run: python scripts/evaluate_synthetic_extension.py --seeds 100
Later, after generation: --seeds 101 102 103 104 105 106 107 108 109
All seven conditions and all three methods are always included per seed.
No data generation, preprocessing, parameter search or outcome-based rejection.
Existing complete bundles are verified without refitting dataset models.
"""

import argparse
import inspect
import io
import json
import platform
import re
import warnings
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyod.models.base as pyod_base
import pyod.models.copod as pyod_copod
import pyod.utils.stat_models as pyod_statistics
from pyod.models.copod import COPOD
from scipy.stats import skew
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import generate_synthetic_extension as generation
from evaluate_synthetic_copod import TrainingReferenceCOPOD


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "synthetic_extension_evaluation_v1"
METHODS = ("pyod_batch", "fixed_training_reference", "ledoit_wolf")
COPOD_PARAMETERS = {"contamination": 0.05, "n_jobs": 1}
LW_PARAMETERS = {"store_precision": True, "assume_centered": False, "block_size": 1000}
FIXED_CLASS_SHA256 = "22f91d4c47de51f029a0e7e059a42e74be34bf91904e6a65d7d1bbca264800da"
TABLE_NAMES = ("metrics.csv", "scores.csv", "feature_diagnostics.csv",
               "ledoit_wolf_location.csv", "ledoit_wolf_matrices.csv",
               "ledoit_wolf_eigenvalues.csv")
require, digest = generation.require, generation.digest
json_bytes, csv_bytes = generation.json_bytes, generation.csv_bytes


def assert_table(saved, recomputed, name):
    try:
        pd.testing.assert_frame_equal(saved, recomputed, check_dtype=False,
                                      check_exact=False, rtol=0, atol=2e-12)
    except AssertionError as error:
        raise ValueError(f"Saved {name} disagrees with the underlying data.") from error


def load_evidence(run_name, seed):
    """Verify the complete generation record, then read its saved arrays."""
    require(seed in generation.PLANNED_SEEDS, "Seed is outside the fixed extension plan.")
    data_dir = ROOT / "data_new/synthetic_extension" / run_name / f"seed{seed}"
    result_dir = ROOT / "results/synthetic_extension" / run_name / f"seed{seed}"
    hashes = {}

    def remember(path, expected=None):
        require(path.is_file(), f"Missing input: {path}")
        content = path.read_bytes()
        actual = digest(content)
        key = path.relative_to(ROOT).as_posix()
        require(expected is None or actual == expected, f"Input hash changed: {path}")
        require(key not in hashes or hashes[key] == actual, f"Input changed during loading: {path}")
        hashes[key] = actual
        return content

    def read_json(path, expected=None):
        return json.loads(remember(path, expected))

    def read_csv(path):
        return pd.read_csv(io.BytesIO(remember(path)), float_precision="round_trip")

    manifest_path = result_dir.parent / "generation_manifest.json"
    manifest = read_json(manifest_path)
    manifest_hash = hashes[manifest_path.relative_to(ROOT).as_posix()]
    require(manifest["schema"] == "synthetic_extension_generation_v1"
            and manifest["run_name"] == run_name
            and manifest["conditions"] == list(generation.CONDITIONS)
            and manifest["planned_seeds"] == list(generation.PLANNED_SEEDS)
            and manifest["feature_columns"] == generation.FEATURES
            and (manifest["n_rows"], manifest["n_features"], manifest["n_positive"],
                 manifest["block_size"]) == (10000, 20, 500, 5)
            and manifest["preprocessing"] == "none"
            and manifest["models_run"] is False, "Unexpected generation configuration.")
    require(manifest["plan_path"] == generation.PLAN_PATH
            and manifest["plan_commit"] == generation.PLAN_COMMIT
            and manifest["plan_sha256"] == generation.PLAN_SHA256,
            "The generation record does not identify the frozen plan.")
    remember(ROOT / generation.PLAN_PATH, generation.PLAN_SHA256)
    remember(Path(generation.__file__).resolve(), manifest["generator_sha256"])
    require(manifest["split"] == {
        "test_size": 0.4, "stratify": "label", "random_state": 42,
        "train_rows": 6000, "test_rows": 4000,
        "train_positives": 300, "test_positives": 200}, "Unexpected split protocol.")

    require(manifest["family_order"] == list(generation.FAMILIES)
            and manifest["within_family_stream_order"] == list(generation.STREAM_NAMES)
            and manifest["sample_sd_ddof"] == 1 and manifest["quantile_method"] == "linear"
            and manifest["construction_check_rtol"] == 0
            and manifest["construction_check_atol"] == 1e-12,
            "Generation pairing or validation configuration changed.")
    # Historical generation versions are provenance; they need not equal evaluator versions.
    seed_record = read_json(result_dir / "seed_metadata.json")
    names = [spec["condition"] for spec in generation.CONDITIONS]
    require(seed_record["schema"] == "synthetic_extension_seed_generation_v1"
            and seed_record["seed"] == seed
            and seed_record["generation_manifest_sha256"] == manifest_hash
            and seed_record["completed_conditions"] == names
            and seed_record["integrity_checks"] == "passed"
            and seed_record["pairing_checks"] == "passed"
            and seed_record["models_run"] is False, "Seed generation is incomplete or mismatched.")
    expected_paths = {result_dir / "validation_summary.csv", result_dir / "pairing_checks.csv"}
    for name in names:
        expected_paths.update({data_dir / f"{name}.csv", data_dir / f"{name}_annotations.csv"})
        expected_paths.update(result_dir / name / item for item in (
            "split.csv", "feature_summary.csv", "correlation_summary.csv", "metadata.json", "validation.json"))
    expected_keys = {path.relative_to(ROOT).as_posix() for path in expected_paths}
    require(set(seed_record["output_sha256"]) == expected_keys,
            "Generation inventory must include all seven datasets and their validation files.")
    for key, expected in seed_record["output_sha256"].items():
        remember(ROOT / key, expected)

    summary = read_csv(result_dir / "validation_summary.csv")
    pairing = read_csv(result_dir / "pairing_checks.csv")
    require(summary.condition.tolist() == names and (summary.seed == seed).all()
            and (summary.checks == "passed").all()
            and pairing.condition.tolist() == names
            and (pairing.required_feature_pairing == "passed").all()
            and (pairing.source_rows_labels_and_split == "identical").all(),
            "Saved generation summary or pairing table is incomplete.")
    cases, anchors = {}, {}
    for spec in generation.CONDITIONS:
        name = spec["condition"]
        target = result_dir / name
        metadata = read_json(target / "metadata.json")
        validation = read_json(target / "validation.json")
        data_path, annotation_path = data_dir / f"{name}.csv", data_dir / f"{name}_annotations.csv"
        require(metadata["schema"] == "synthetic_extension_dataset_v1"
                and metadata["seed"] == seed and metadata["specification"] == spec
                and metadata["generation_manifest_sha256"] == manifest_hash
                and metadata["feature_columns"] == generation.FEATURES
                and metadata["data_path"] == data_path.relative_to(ROOT).as_posix()
                and metadata["annotation_path"] == annotation_path.relative_to(ROOT).as_posix(),
                f"Dataset metadata differs: {name}")
        expected_artifacts = {data_path, annotation_path, target / "split.csv",
                              target / "feature_summary.csv", target / "correlation_summary.csv"}
        require(metadata["output_sha256"] == {
            path.relative_to(ROOT).as_posix(): hashes[path.relative_to(ROOT).as_posix()]
            for path in expected_artifacts}, f"Dataset hash chain differs: {name}")
        require(validation["schema"] == "synthetic_extension_validation_v1"
                and validation["condition"] == name and validation["seed"] == seed
                and validation["metadata_sha256"] == hashes[(target / "metadata.json").relative_to(ROOT).as_posix()]
                and validation["data_sha256"] == hashes[data_path.relative_to(ROOT).as_posix()]
                and validation["integrity_checks"] == "passed"
                and validation["paired_construction"] == "passed"
                and validation["models_run"] is False
                and validation["null_control"] == spec["null_control"]
                and validation["dimensions"] == [10000, 20]
                and [validation[k] for k in ("normal_rows", "positive_rows", "train_rows",
                                              "test_rows", "train_positives", "test_positives")]
                    == [9500, 500, 6000, 4000, 300, 200]
                and validation["missing_or_nonfinite_values"] == 0
                and validation["annotations_and_allowed_coordinate_changes"] == "passed"
                and not any(validation["duplicates"].values()), f"Validation differs: {name}")
        data, annotations, split = read_csv(data_path), read_csv(annotation_path), read_csv(target / "split.csv")
        require(list(data.columns) == generation.FEATURES + ["label"] and data.shape == (10000, 21),
                f"Unexpected data shape or columns: {name}")
        X, y = data[generation.FEATURES].to_numpy(dtype=float), data.label.to_numpy()
        require(list(annotations.columns) == ["row_index", "source_row_id", "family", "condition",
                "label", "selected_feature_index", "shift_sign", "applied_shift"]
                and list(split.columns) == ["row_index", "source_row_id", "label", "split", "partition_position"],
                f"Unexpected annotation or split schema: {name}")
        require(np.isfinite(X).all() and np.array_equal(np.unique(y, return_counts=True)[0], [0, 1])
                and np.array_equal(np.unique(y, return_counts=True)[1], [9500, 500]),
                f"Invalid features or labels: {name}")
        require(len(annotations) == len(split) == len(y)
                and np.array_equal(annotations.row_index, np.arange(10000))
                and np.array_equal(split.row_index, np.arange(10000))
                and np.array_equal(annotations.label, y) and np.array_equal(split.label, y)
                and np.array_equal(annotations.source_row_id, split.source_row_id)
                and np.array_equal(np.sort(annotations.source_row_id), np.arange(10000))
                and np.array_equal(y, (annotations.source_row_id >= 9500).astype(int))
                and (annotations.family == spec["family"]).all()
                and (annotations.condition == name).all(), f"Row alignment differs: {name}")
        ordered = {}
        for partition, count, positives in (("train", 6000, 300), ("test", 4000, 200)):
            part = split[split.split == partition].sort_values("partition_position")
            require(len(part) == count and part.label.sum() == positives
                    and np.array_equal(part.partition_position, np.arange(count)), f"Invalid split: {name}")
            ordered[partition] = part.row_index.to_numpy()
            require(generation.array_digest(ordered[partition])
                    == metadata["stream_provenance"][partition + "_row_order_sha256"],
                    f"Split order changed: {name}")

        require(np.array_equal(np.sort(np.r_[ordered["train"], ordered["test"]]), np.arange(10000)),
                f"Train/test rows do not form a disjoint complete partition: {name}")
        expected_train, expected_test = train_test_split(np.arange(10000), test_size=0.4,
                                                        stratify=y, random_state=42)
        require(np.array_equal(ordered["train"], expected_train)
                and np.array_equal(ordered["test"], expected_test),
                f"Saved row order differs from the specified split procedure: {name}")
        provenance = metadata["stream_provenance"]
        family_index = list(generation.FAMILIES).index(spec["family"])
        require(provenance["entropy"] == [seed, 20261029]
                and provenance["family_spawn_key"] == [family_index]
                and provenance["stream_spawn_keys"] == {
                    role: [family_index, i] for i, role in enumerate(generation.STREAM_NAMES)}
                and generation.array_digest(annotations.source_row_id.to_numpy())
                    == provenance["source_order_sha256"], f"Stream or source-row provenance differs: {name}")
        source_order = annotations.sort_values("source_row_id")
        require(generation.array_digest(source_order.selected_feature_index.to_numpy())
                    == provenance["selected_features_sha256"]
                and generation.array_digest(source_order.shift_sign.to_numpy())
                    == provenance["shift_signs_sha256"], f"Annotation source-order hashes differ: {name}")
        duplicates = generation.duplicate_checks(X, {"train": ordered["train"],
                                                     "test": ordered["test"], "labels": y})
        require(duplicates == validation["duplicates"], f"Duplicate-profile checks differ: {name}")
        features, correlations = generation.distribution_summaries(X, y, spec)
        assert_table(read_csv(target / "feature_summary.csv"), features, name + " feature summary")
        assert_table(read_csv(target / "correlation_summary.csv"), correlations, name + " correlation summary")
        anchor = anchors.setdefault(spec["family"], {"X": X, "y": y, "annotations": annotations,
                                                       "split": split, "provenance": provenance})
        pd.testing.assert_frame_equal(split, anchor["split"])
        require(provenance == anchor["provenance"], "Paired family random streams changed.")
        require(np.array_equal(annotations.source_row_id, anchor["annotations"].source_row_id)
                and np.array_equal(y, anchor["y"]), "Paired row identities changed.")
        if spec["family"] == "marginal_shift":
            selected, signs = annotations.selected_feature_index.to_numpy(), annotations.shift_sign.to_numpy()
            require((selected[y == 0] == -1).all() and (signs[y == 0] == 0).all()
                    and np.isin(selected[y == 1], np.arange(20)).all()
                    and np.isin(signs[y == 1], [-1, 1]).all()
                    and np.array_equal(selected, anchor["annotations"].selected_feature_index)
                    and np.array_equal(signs, anchor["annotations"].shift_sign)
                    and np.array_equal(annotations.applied_shift, signs * spec["delta"]),
                    "Marginal construction annotations changed.")
            expected = anchor["X"].copy()
            chosen = np.flatnonzero(y == 1)
            expected[chosen, selected[chosen]] += signs[chosen] * spec["delta"]
            np.testing.assert_allclose(X, expected, rtol=0, atol=1e-12)
        else:
            require((annotations.selected_feature_index == -1).all()
                    and (annotations.shift_sign == 0).all() and (annotations.applied_shift == 0).all()
                    and np.array_equal(X[y == 1], anchor["X"][y == 1]),
                    "Dependence pairing or annotations changed.")
        cases[name] = {"spec": spec, "X": X, "y": y.astype(np.int64),
                       "annotations": annotations, "split": split,
                       "train_rows": ordered["train"], "test_rows": ordered["test"]}
        print(f"Verified saved inputs, pairing and distribution summaries: {name}", flush=True)
    for key, expected in list(hashes.items()):
        remember(ROOT / key, expected)
    return {"cases": cases, "input_sha256": hashes, "manifest": manifest,
            "generation_warnings": seed_record["warnings"], "result_dir": result_dir}


def check_hashes(hashes):
    for relative, expected in hashes.items():
        path = ROOT / relative
        require(path.is_file() and digest(path.read_bytes()) == expected,
                f"Protected input changed: {relative}")


def capture_warnings(stage, operation):
    """Retain and display warnings; never globally hide or repair them."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = operation()
        finally:
            records = [{"stage": stage, "category": w.category.__name__,
                        "message": str(w.message), "source_file": Path(w.filename).name,
                        "source_line": w.lineno} for w in caught]
            for item in records:
                print(f"Warning [{stage}]: {item['message']}", flush=True)
    return result, records


def timed(operation):
    start = perf_counter()
    result = operation()
    return result, perf_counter() - start


def performance(y, scores):
    require(scores.shape == y.shape and np.isfinite(scores).all()
            and np.array_equal(np.unique(y), [0, 1]), "Invalid metric inputs.")
    return {"roc_auc": float(roc_auc_score(y, scores)),
            "average_precision": float(average_precision_score(y, scores))}


def implementation_checks():
    # Inclusive counts with ties, unseen extremes, and negative/zero/positive skew.
    X = np.array([[0., 0., 0.], [0., 3., 1.], [1., 3., 1.], [3., 3., 2.]])
    query = np.array([[-1., -1., -1.], [0., 0., 0.], [.5, 2., 1.],
                      [3., 3., 2.], [4., 4., 3.]])
    fixed = TrainingReferenceCOPOD().fit(X)
    standard = COPOD(**COPOD_PARAMETERS).fit(X)
    np.testing.assert_allclose(fixed.decision_function(X), standard.decision_scores_,
                               rtol=1e-10, atol=1e-10)
    left_count = (X[None, :, :] <= query[:, None, :]).sum(axis=1)
    right_count = (X[None, :, :] >= query[:, None, :]).sum(axis=1)
    np.testing.assert_array_equal(left_count[:, 0], [0, 2, 2, 4, 4])
    np.testing.assert_array_equal(right_count[:, 0], [4, 4, 2, 1, 0])
    left = -np.log(np.maximum(left_count / 4, .25))
    right = -np.log(np.maximum(right_count / 4, .25))
    signs = np.sign(skew(X, axis=0, bias=True))
    np.testing.assert_array_equal(signs, [1, -1, 0])
    expected = np.maximum(left * -np.sign(signs - 1) + right * np.sign(signs + 1),
                          (left + right) / 2).sum(axis=1)
    np.testing.assert_allclose(fixed.decision_function(query), expected, rtol=0, atol=1e-12)

    # Hand calculation: location=(10,20), empirical covariance=diag(1,4),
    # target=2.5 I, shrinkage=(1)/(9/4)=4/9. Thus covariance=diag(5/3,10/3).
    fixture = np.array([[9., 18.], [9., 22.], [11., 18.], [11., 22.]])
    lw = LedoitWolf(**LW_PARAMETERS).fit(fixture)
    q = np.array([[10., 20.], [11., 20.], [10., 22.], [11., 22.]])
    np.testing.assert_allclose(lw.location_, [10, 20], rtol=0, atol=1e-12)
    np.testing.assert_allclose(lw.shrinkage_, 4/9, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(lw.covariance_, np.diag([5/3, 10/3]), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(lw.precision_, np.diag([3/5, 3/10]), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(lw.mahalanobis(q), [0, .6, 1.2, 1.8], rtol=1e-10, atol=1e-12)
    # A tied ranking has pairwise AUC=3.5/4 and non-interpolated AP=1/2+(1/2)(2/3).
    metrics = performance(np.array([1, 0, 1, 0]), np.array([.9, .7, .7, .1]))
    np.testing.assert_allclose([metrics["roc_auc"], metrics["average_precision"]],
                               [7/8, 5/6], rtol=0, atol=1e-12)
    return {"status": "passed", "copod_inclusive_tails_ties_and_skew": "passed",
            "ledoit_wolf_centering_shrinkage_squared_distance": "passed",
            "ledoit_wolf_fixture_expected_shrinkage": 4/9,
            "ledoit_wolf_fixture_expected_scores": [0, .6, 1.2, 1.8],
            "tied_auc_and_noninterpolated_ap": "passed",
            "timing_note": "Small fixtures run before dataset timing; timings are descriptive only."}


def evaluation_manifest(run_name, evidence):
    require(version("pyod") == "3.6.4", "The frozen plan uses PyOD 3.6.4; investigate version changes.")
    class_hash = digest(inspect.getsource(TrainingReferenceCOPOD).encode("utf-8"))
    require(class_hash == FIXED_CLASS_SHA256, "The pilot's fixed-reference scorer has changed.")
    sources = {"evaluator": Path(__file__).resolve(), "generator": Path(generation.__file__).resolve(),
               "fixed_reference_module": Path(inspect.getsourcefile(TrainingReferenceCOPOD)),
               "pyod_copod": Path(pyod_copod.__file__), "pyod_base": Path(pyod_base.__file__),
               "pyod_ecdf": Path(pyod_statistics.__file__),
               "ledoit_wolf": Path(inspect.getsourcefile(LedoitWolf)),
               "mahalanobis": Path(inspect.getsourcefile(LedoitWolf.mahalanobis)),
               "scipy_skew": Path(inspect.getsourcefile(skew)),
               "metrics": Path(inspect.getsourcefile(roc_auc_score))}
    versions = {package: version(package) for package in
                ("numpy", "pandas", "scipy", "scikit-learn", "pyod", "numba", "statsmodels")}
    return {"schema": SCHEMA, "run_name": run_name,
            "plan_path": generation.PLAN_PATH, "plan_commit": generation.PLAN_COMMIT,
            "plan_sha256": generation.PLAN_SHA256,
            "generation_manifest_sha256": digest((evidence["result_dir"].parent / "generation_manifest.json").read_bytes()),
            "planned_data_seeds": list(generation.PLANNED_SEEDS),
            "conditions": list(generation.CONDITIONS), "methods": list(METHODS),
            "source_sha256": {key: digest(path.read_bytes()) for key, path in sources.items()},
            "fixed_reference_class_sha256": class_hash,
            "runtime": {"python": platform.python_version(), "platform": platform.platform(), "packages": versions},
            "generation_versions": evidence["manifest"]["versions"],
            "copod_parameters": COPOD_PARAMETERS, "ledoit_wolf_parameters": LW_PARAMETERS,
            "fixed_reference_probability_floor": 1/6000,
            "preprocessing": "none", "labels_in_fit": False, "training_rows": 6000,
            "training_positives_retained": 300, "test_rows": 4000, "test_positives": 200,
            "split": evidence["manifest"]["split"], "ordered_feature_columns": generation.FEATURES,
            "scoring_references": {"pyod_batch": "training plus entire test batch in one call",
                                   "fixed_training_reference": "training only",
                                   "ledoit_wolf": "training only; squared Mahalanobis distances"},
            "tolerances": {"training_agreement": {"rtol": 1e-10, "atol": 1e-10},
                           "fixed_batch": {"rtol": 0, "atol": 1e-12},
                           "ledoit_wolf_batch_formula": {"rtol": 1e-10, "atol": 1e-12}},
            "metrics": ["roc_auc", "average_precision"], "metrics_use": "test rows only, raw scores",
            "null_references": {"roc_auc": .5, "average_precision_prevalence": .05},
            "performance_acceptance_thresholds": None,
            "timing_policy": "One fit and one whole-test score call; diagnostic checks excluded. No speed claim."}


def fixed_diagnostics(fixed, X_train, X_test, scores):
    combined_signs = np.sign(skew(np.vstack([X_train, X_test]), axis=0, bias=True))
    require(np.isfinite(combined_signs).all(), "Non-finite combined skewness signs.")
    contributions = np.empty_like(X_test)
    records = []
    touched_rows = np.zeros(len(X_test), dtype=bool)
    for j, feature in enumerate(generation.FEATURES):
        reference = fixed.sorted_train[:, j]
        counts_l = np.searchsorted(reference, X_test[:, j], side="right")
        counts_r = len(reference) - np.searchsorted(reference, X_test[:, j], side="left")
        np.testing.assert_array_equal(counts_l == 0, X_test[:, j] < reference[0])
        np.testing.assert_array_equal(counts_r == 0, X_test[:, j] > reference[-1])
        touched_rows |= (counts_l == 0) | (counts_r == 0)
        left = -np.log(np.maximum(counts_l / len(reference), fixed.probability_floor))
        right = -np.log(np.maximum(counts_r / len(reference), fixed.probability_floor))
        sign = fixed.skew_sign[j]
        contributions[:, j] = np.maximum(left * -np.sign(sign - 1) + right * np.sign(sign + 1),
                                          (left + right) / 2)
        records.append({"feature": feature, "training_skewness_sign": int(sign),
                        "combined_skewness_sign": int(combined_signs[j]),
                        "skewness_sign_changed": bool(sign != combined_signs[j]),
                        "training_min": float(reference[0]), "training_max": float(reference[-1]),
                        "left_zero_tail_replacements": int((counts_l == 0).sum()),
                        "right_zero_tail_replacements": int((counts_r == 0).sum()),
                        "left_already_at_floor": int((counts_l == 1).sum()),
                        "right_already_at_floor": int((counts_r == 1).sum())})
    require(np.isfinite(contributions).all(), "Non-finite fixed-reference contributions.")
    np.testing.assert_allclose(contributions.sum(axis=1), scores, rtol=0, atol=1e-12)
    return pd.DataFrame(records), int(touched_rows.sum())


def batch_check(function, X, full, rtol):
    chunks = np.concatenate([function(part) for part in np.array_split(X, 4)])
    singles = np.array([function(X[i:i+1])[0] for i in range(10)])
    np.testing.assert_allclose(full, chunks, rtol=rtol, atol=1e-12)
    np.testing.assert_allclose(full[:10], singles, rtol=rtol, atol=1e-12)
    return {"status": "passed", "chunk_max_abs_difference": float(np.max(np.abs(full-chunks))),
            "singleton_max_abs_difference": float(np.max(np.abs(full[:10]-singles)))}


def contribution_flags(model, rows):
    flags = {}
    for name in ("U_l", "U_r", "U_skew", "O"):
        array = np.asarray(getattr(model, name))
        require(array.shape == (rows, 20), f"Unexpected PyOD {name} reference dimensions.")
        flags[name] = int((~np.isfinite(array)).sum())
    require(not any(flags.values()), "Non-finite PyOD contributions; no repair is permitted.")
    return flags


def fit_case(case, seed):
    X, y, train, test = case["X"], case["y"], case["train_rows"], case["test_rows"]
    X_train, X_test, y_test = X[train], X[test], y[test]
    spec = case["spec"]
    standard, t_standard = timed(lambda: COPOD(**COPOD_PARAMETERS).fit(X_train))
    fixed, t_fixed = timed(lambda: TrainingReferenceCOPOD().fit(X_train))
    fixed_train = fixed.decision_function(X_train)
    require(np.isfinite(fixed_train).all() and np.isfinite(standard.decision_scores_).all(),
            "Non-finite training scores.")
    np.testing.assert_allclose(fixed_train, standard.decision_scores_, rtol=1e-10, atol=1e-10)
    agreement = float(np.max(np.abs(fixed_train-standard.decision_scores_)))
    train_flags = contribution_flags(standard, 6000)  # Save before test call overwrites arrays.
    score_batch, s_standard = timed(lambda: standard.decision_function(X_test))
    batch_flags = contribution_flags(standard, 10000)
    np.testing.assert_array_equal(standard.O.sum(axis=1)[-4000:], score_batch)
    score_fixed, s_fixed = timed(lambda: fixed.decision_function(X_test))
    fixed_batch = batch_check(fixed.decision_function, X_test, score_fixed, rtol=0)
    features, touched_rows = fixed_diagnostics(fixed, X_train, X_test, score_fixed)

    lw, t_lw = timed(lambda: LedoitWolf(**LW_PARAMETERS).fit(X_train))
    score_lw, s_lw = timed(lambda: lw.mahalanobis(X_test))
    lw_batch = batch_check(lw.mahalanobis, X_test, score_lw, rtol=1e-10)
    require(all(np.isfinite(value).all() for value in (lw.location_, lw.covariance_, lw.precision_))
            and 0 <= lw.shrinkage_ <= 1 and (score_lw >= 0).all(), "Invalid covariance fit or distances.")
    np.testing.assert_allclose(lw.location_, X_train.mean(axis=0), rtol=1e-10, atol=1e-12)
    centered = X_test - lw.location_
    formula = np.einsum("ij,jk,ik->i", centered, lw.precision_, centered)
    np.testing.assert_allclose(score_lw, formula, rtol=1e-10, atol=1e-12)
    covariance_eigenvalues = np.linalg.eigvalsh(lw.covariance_)
    precision_eigenvalues = np.linalg.eigvalsh(lw.precision_)
    require((covariance_eigenvalues > 0).all() and (precision_eigenvalues > 0).all(),
            "Covariance/precision must be positive definite in these settings.")
    np.testing.assert_allclose(lw.covariance_ @ lw.precision_, np.eye(20), rtol=1e-10, atol=1e-10)

    score_table = pd.DataFrame({"test_position": np.arange(4000), "row_index": test,
                               "source_row_id": case["annotations"].source_row_id.to_numpy()[test],
                               "label": y_test, METHODS[0]: score_batch,
                               METHODS[1]: score_fixed, METHODS[2]: score_lw})
    metric_rows = []
    for method, fit_time, score_time in zip(METHODS, (t_standard, t_fixed, t_lw),
                                          (s_standard, s_fixed, s_lw)):
        metric_rows.append({"data_seed": seed, **spec, "method": method,
                            "n_train": 6000, "n_test": 4000, "n_features": 20,
                            "train_positives": int(y[train].sum()), "test_positives": int(y_test.sum()),
                            "test_prevalence": float(y_test.mean()),
                            **performance(y_test, score_table[method].to_numpy()),
                            "fit_seconds": fit_time, "score_seconds": score_time})
    tables = {"metrics.csv": pd.DataFrame(metric_rows), "scores.csv": score_table,
              "feature_diagnostics.csv": features,
              "ledoit_wolf_location.csv": pd.DataFrame({"feature": generation.FEATURES, "location": lw.location_}),
              "ledoit_wolf_matrices.csv": pd.DataFrame({
                  "row_feature": np.repeat(generation.FEATURES, 20),
                  "column_feature": np.tile(generation.FEATURES, 20),
                  "covariance": lw.covariance_.ravel(), "precision": lw.precision_.ravel()}),
              "ledoit_wolf_eigenvalues.csv": pd.DataFrame({"ascending_position": np.arange(20),
                  "covariance_eigenvalue": covariance_eigenvalues, "precision_eigenvalue": precision_eigenvalues})}
    details = {"checks": "passed", "training_score_agreement": "passed",
               "training_score_max_abs_difference": agreement,
               "fixed_batch_invariance": fixed_batch, "ledoit_wolf_batch_invariance": lw_batch,
               "nonfinite_pyod_training_contributions": train_flags,
               "nonfinite_pyod_combined_contributions": batch_flags,
               "nonfinite_fixed_test_contributions": 0, "nonfinite_final_scores": {m: 0 for m in METHODS},
               "fixed_test_rows_with_zero_tail_replacement": touched_rows,
               "fixed_zero_tail_replacement_cells": int(features[["left_zero_tail_replacements", "right_zero_tail_replacements"]].to_numpy().sum()),
               "fixed_tail_probabilities_already_at_floor": int(features[["left_already_at_floor", "right_already_at_floor"]].to_numpy().sum()),
               "skewness_sign_changes": int(features.skewness_sign_changed.sum()),
               "ledoit_wolf": {"shrinkage": float(lw.shrinkage_), "score_formula": "(x-location)^T precision (x-location)",
                   "formula_max_abs_difference": float(np.max(np.abs(score_lw-formula))),
                   "covariance_min_eigenvalue": float(covariance_eigenvalues[0]),
                   "covariance_max_eigenvalue": float(covariance_eigenvalues[-1]),
                   "covariance_spectral_condition_number": float(covariance_eigenvalues[-1]/covariance_eigenvalues[0]),
                   "precision_spectral_condition_number": float(precision_eigenvalues[-1]/precision_eigenvalues[0])}}
    return tables, details


def verify_tables(tables, case, seed, details):
    """Recompute metrics from persisted score/label pairs, without detector fits."""
    scores, metrics = tables["scores.csv"], tables["metrics.csv"]
    test = case["test_rows"]
    require(scores.columns.tolist() == ["test_position", "row_index", "source_row_id", "label", *METHODS]
            and len(scores) == 4000 and metrics.method.tolist() == list(METHODS), "Saved output structure differs.")
    for name, expected in {"test_position": np.arange(4000), "row_index": test,
            "source_row_id": case["annotations"].source_row_id.to_numpy()[test], "label": case["y"][test]}.items():
        np.testing.assert_array_equal(scores[name], expected)
    for _, row in metrics.iterrows():
        for key, expected in {"data_seed": seed, **case["spec"], "n_train": 6000, "n_test": 4000,
                "n_features": 20, "train_positives": 300, "test_positives": 200, "test_prevalence": .05}.items():
            require(pd.isna(row[key]) if expected is None else row[key] == expected, f"Metric metadata differs: {key}")
        for metric, value in performance(scores.label.to_numpy(), scores[row.method].to_numpy()).items():
            np.testing.assert_allclose(row[metric], value, rtol=0, atol=1e-12)
        require(np.isfinite([row.fit_seconds, row.score_seconds]).all()
                and row.fit_seconds >= 0 and row.score_seconds >= 0, "Invalid timings.")
    require(details["checks"] == details["training_score_agreement"] == "passed"
            and details["fixed_batch_invariance"]["status"] == "passed"
            and details["ledoit_wolf_batch_invariance"]["status"] == "passed", "Missing implementation checks.")
    require(tables["feature_diagnostics.csv"].feature.tolist() == generation.FEATURES,
            "Feature diagnostic order differs.")
    location, matrices = tables["ledoit_wolf_location.csv"], tables["ledoit_wolf_matrices.csv"]
    require(location.feature.tolist() == generation.FEATURES and len(matrices) == 400,
            "Covariance evidence dimensions differ.")
    np.testing.assert_array_equal(matrices.row_feature, np.repeat(generation.FEATURES, 20))
    np.testing.assert_array_equal(matrices.column_feature, np.tile(generation.FEATURES, 20))
    center = location.location.to_numpy()
    covariance, precision = [matrices[key].to_numpy().reshape(20, 20) for key in ("covariance", "precision")]
    require(all(np.isfinite(a).all() for a in (center, covariance, precision)), "Invalid saved fitted parameters.")
    np.testing.assert_allclose(center, case["X"][case["train_rows"]].mean(axis=0), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(covariance @ precision, np.eye(20), rtol=1e-10, atol=1e-10)
    centered = case["X"][test]-center
    np.testing.assert_allclose(scores.ledoit_wolf, np.einsum("ij,jk,ik->i", centered, precision, centered),
                               rtol=1e-10, atol=1e-12)
    eigenvalues = tables["ledoit_wolf_eigenvalues.csv"]
    np.testing.assert_array_equal(eigenvalues.ascending_position, np.arange(20))
    for key, matrix in (("covariance", covariance), ("precision", precision)):
        values = np.linalg.eigvalsh(matrix)
        require((values > 0).all(), "Non-positive saved eigenvalue.")
        np.testing.assert_allclose(eigenvalues[key+"_eigenvalue"], values, rtol=1e-10, atol=1e-12)


def bundle_config(evidence, case, seed, manifest_hash, checks_hash):
    return {"evaluation_manifest_sha256": manifest_hash, "implementation_checks_sha256": checks_hash,
            "data_seed": seed, "specification": case["spec"], "input_sha256": evidence["input_sha256"]}


def read_bundle(folder, config, case, seed):
    path = folder / "metadata.json"
    require(path.is_file(), f"Incomplete result bundle: {folder}. Investigate before rerunning.")
    metadata = json.loads(path.read_bytes())
    require(metadata["schema"] == SCHEMA and metadata["configuration"] == config,
            f"Saved evaluation configuration/provenance changed: {folder}")
    require(set(metadata["output_sha256"]) == set(TABLE_NAMES)
            and {p.name for p in folder.iterdir()} == {*TABLE_NAMES, "metadata.json"}, "Unexpected bundle inventory.")
    tables = {}
    for name in TABLE_NAMES:
        content = (folder / name).read_bytes()
        require(digest(content) == metadata["output_sha256"][name], f"Saved output changed: {folder/name}")
        tables[name] = pd.read_csv(io.BytesIO(content), float_precision="round_trip")
    verify_tables(tables, case, seed, metadata["details"])
    return tables, metadata


def write_bundle(folder, config, tables, details):
    require(not folder.exists(), f"Refusing to replace result bundle: {folder}")
    payloads = {name: csv_bytes(tables[name]) for name in TABLE_NAMES}
    metadata = {"schema": SCHEMA, "configuration": config, "details": details,
                "output_sha256": {name: digest(content) for name, content in payloads.items()}}
    folder.mkdir(parents=True, exist_ok=False)
    for name, content in payloads.items():
        with (folder / name).open("xb") as handle:
            handle.write(content)
    # Completion marker last. Interrupted or differing evidence is never overwritten.
    with (folder / "metadata.json").open("xb") as handle:
        handle.write(json_bytes(metadata))


def evaluate_seed(evidence, seed, manifest_hash, checks_hash):
    check_hashes(evidence["input_sha256"])
    cached = {}
    # Check every existing requested bundle before any new dataset fit.
    for name, case in evidence["cases"].items():
        folder = evidence["result_dir"] / name / "evaluation"
        config = bundle_config(evidence, case, seed, manifest_hash, checks_hash)
        if folder.exists():
            cached[name] = read_bundle(folder, config, case, seed)
    rows, output_hashes, model_warning_count = [], {}, 0
    for name, case in evidence["cases"].items():
        folder = evidence["result_dir"] / name / "evaluation"
        config = bundle_config(evidence, case, seed, manifest_hash, checks_hash)
        print(f"\nSeed {seed}, {name}: 6,000 train / 4,000 test; three methods", flush=True)
        if name in cached:
            tables, metadata = cached[name]
            print("  Verified existing scores, row alignment, metrics and fitted covariance; no dataset refit.")
        else:
            (tables, details), warning_log = capture_warnings(f"seed{seed}/{name}", lambda: fit_case(case, seed))
            details["warnings"] = warning_log
            verify_tables(tables, case, seed, details)
            check_hashes(evidence["input_sha256"])
            write_bundle(folder, config, tables, details)
            tables, metadata = read_bundle(folder, config, case, seed)
            print("  Training-score agreement, batch invariance and saved-score checks: passed.")
        model_warning_count += len(metadata["details"]["warnings"])
        print(tables["metrics.csv"][["method", "roc_auc", "average_precision"]].to_string(index=False))
        rows.append(tables["metrics.csv"])
        for path in sorted(folder.iterdir()):
            output_hashes[path.relative_to(ROOT).as_posix()] = digest(path.read_bytes())
    check_hashes(evidence["input_sha256"])
    summary = pd.concat(rows, ignore_index=True)
    require(len(summary) == 21, "Each seed must contain seven conditions and three methods.")
    content = csv_bytes(summary)
    summary_path = evidence["result_dir"] / "evaluation_metrics.csv"
    output_hashes[summary_path.relative_to(ROOT).as_posix()] = digest(content)
    completion = {"schema": SCHEMA, "data_seed": seed, "evaluation_manifest_sha256": manifest_hash,
                  "implementation_checks_sha256": checks_hash, "input_sha256": evidence["input_sha256"],
                  "completed_conditions": [s["condition"] for s in generation.CONDITIONS],
                  "methods": list(METHODS), "metric_rows": 21, "checks": "passed",
                  "generation_warnings": evidence["generation_warnings"],
                  "input_verification_warnings": evidence.get("input_verification_warnings", []),
                  "model_warning_count": model_warning_count, "output_sha256": output_hashes}
    generation.save_or_verify({summary_path: content,
        evidence["result_dir"] / "evaluation_metadata.json": json_bytes(completion)})
    print(f"Seed {seed} complete: 21 result rows; recorded model warnings: {model_warning_count}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, choices=generation.PLANNED_SEEDS, default=[100])
    parser.add_argument("--run-name", default="v1", help="Existing generation namespace; default v1.")
    args = parser.parse_args()
    require(len(set(args.seeds)) == len(args.seeds), "List each requested seed once.")
    require(re.fullmatch(r"[a-z][a-z0-9_]{0,47}", args.run_name) is not None, "Invalid run name.")
    # The generation manifest pins its own environment; do not confuse it with evaluation versions.
    loaded = {}
    for seed in sorted(args.seeds):
        print(f"Verifying saved construction and splits: seed {seed}...", flush=True)
        evidence, warning_log = capture_warnings(f"input_seed{seed}", lambda: load_evidence(args.run_name, seed))
        evidence["input_verification_warnings"] = warning_log
        loaded[seed] = evidence
    manifests = [evaluation_manifest(args.run_name, evidence) for evidence in loaded.values()]
    require(all(manifest == manifests[0] for manifest in manifests), "Seed generation provenance differs.")
    base = next(iter(loaded.values()))["result_dir"].parent
    manifest_content = json_bytes(manifests[0])
    manifest_hash = digest(manifest_content)
    # Refuse changed software/configuration before running even the small fixtures.
    manifest_path = base / "evaluation_manifest.json"
    if manifest_path.exists():
        require(manifest_path.read_bytes() == manifest_content, "Existing evaluation manifest differs; investigate changes.")
    checks, warning_log = capture_warnings("implementation_fixtures", implementation_checks)
    checks.update({"schema": SCHEMA, "evaluation_manifest_sha256": manifest_hash, "warnings": warning_log})
    checks_content = json_bytes(checks)
    generation.save_or_verify({manifest_path: manifest_content, base / "implementation_checks.json": checks_content})
    print("Small inclusive-tail, tied-metric and hand-calculated covariance checks: passed.", flush=True)
    for seed, evidence in loaded.items():
        evaluate_seed(evidence, seed, manifest_hash, digest(checks_content))
    print("\nRequested seeds completed. Each result uses the same saved split for all three methods.")
    print("No performance thresholds were used to accept seeds or settings.")
    print("Complete the planned seeds 100–109 before the final summaries and comparisons.")
    print(f"Evidence: results/synthetic_extension/{args.run_name}")


if __name__ == "__main__":
    main()
