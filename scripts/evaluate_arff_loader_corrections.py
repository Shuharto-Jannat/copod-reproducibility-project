"""Compare legacy and corrected ARFF loading using the authors' COPOD.

Example pilot:
python scripts/evaluate_arff_loader_corrections.py --datasets HeartDisease --seeds 0

The three conditions separate evaluation-label correction from predictor
correction. Scores still use the authors' train-plus-test batch reference.
"""

from pathlib import Path
from time import perf_counter
import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import subprocess
import warnings

import arff
import numpy as np
import pandas as pd
from pyod.utils.utility import standardizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import audit_arff_dataset_mapping as audit


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "original_code/COPOD/models/cod.py"
BENCHMARK = ROOT / "scripts/reproduce_benchmark_arff.py"
HELPER = Path(audit.__file__).resolve()
HISTORICAL = ROOT / "results/reproduction_arff_16_10_trials"
SCHEMA = "arff_loader_corrections_v1"
CONDITIONS = [
    ("legacy_columns_legacy_labels", "legacy", "legacy"),
    ("legacy_columns_true_labels", "legacy", "true"),
    ("corrected_columns_true_labels", "corrected", "true"),
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit(directory):
    return subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def load_dataset(path, misplaced, legacy_reader):
    legacy_X, legacy_y, _ = legacy_reader(str(path), misplaced)
    legacy_X = np.asarray(legacy_X, dtype=float)
    legacy_y = np.asarray(legacy_y)
    with path.open(encoding="utf-8") as handle:
        raw = arff.load(handle)
    names = [str(item[0]) for item in raw["attributes"]]
    values = np.asarray(raw["data"], dtype=object)
    ids = [i for i, name in enumerate(names) if name.casefold() == "id"]
    labels = [i for i, name in enumerate(names) if name.casefold() == "outlier"]
    require(len(ids) == len(labels) == 1, f"Expected one id and outlier column: {path}.")
    id_i, label_i = ids[0], labels[0]
    feature_indices = [i for i in range(len(names)) if i not in [id_i, label_i]]
    raw_labels = values[:, label_i].astype(str)
    require(set(raw_labels) == {"no", "yes"}, "Unexpected named outlier labels.")
    true_y = (raw_labels == "yes").astype(int)
    corrected_X = values[:, feature_indices].astype(float)
    require(np.isin(legacy_y, [0, 1]).all(), "Legacy labels are not binary.")
    require(len(legacy_y) == len(true_y), "Label lengths differ.")
    require(np.isfinite(legacy_X).all() and np.isfinite(corrected_X).all(),
            "Non-finite raw predictors: stop and inspect instead of imputing silently.")
    require(np.array_equal(legacy_X, values[:, :-2].astype(float)),
            "Legacy feature selection changed; revisit the mapping audit.")
    return {
        "X": {"legacy": legacy_X, "corrected": corrected_X},
        "y": {"legacy": legacy_y.astype(int), "true": true_y},
        "ids": values[:, id_i].astype(str),
        "features": {"legacy": names[:-2], "corrected": [names[i] for i in feature_indices]},
        "relation": raw["relation"],
        "label_disagreements": int(np.count_nonzero(legacy_y != true_y)),
    }


def score(model_class, X, features, train_indices, test_indices, contamination):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        train, test = standardizer(X[train_indices], X[test_indices])
        require(np.isfinite(train).all() and np.isfinite(test).all(),
                "Non-finite scaled predictors.")
        model = model_class(contamination=contamination)
        start = perf_counter()
        model.fit(train)  # Features only; no labels supplied to fit.
        scores = np.asarray(model.decision_function(test), dtype=float).ravel()
        elapsed = perf_counter() - start
    require(len(scores) == len(test_indices) and np.isfinite(scores).all(),
            "Invalid final scores: no nan_to_num replacement is allowed here.")
    # The unchanged source sums a DataFrame with skipna=True. Record any
    # non-finite contributions even when its final scores remain finite.
    nonfinite = ~np.isfinite(np.asarray(model.O, dtype=float))
    constant = np.ptp(X[train_indices], axis=0) == 0
    details = {
        "fit_and_score_seconds": elapsed,
        "training_constant_features": [name for name, flag in zip(features, constant) if flag],
        "nonfinite_intermediate_entries": int(nonfinite.sum()),
        "test_rows_with_nonfinite_contributions": int(nonfinite[-len(test_indices):].any(axis=1).sum()),
        "warnings": [{"category": item.category.__name__, "message": str(item.message)}
                     for item in captured],
    }
    return scores, details


def evaluate(labels, scores):
    # Do not change a seed or the split to obtain more convenient class counts.
    if np.unique(labels).size < 2:
        return {"roc_auc": None, "average_precision": None,
                "metric_status": "not_reported_single_class_test_set"}
    return {"roc_auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
            "metric_status": "ok"}


def show_metrics(metrics):
    print(metrics[["condition", "test_positives", "roc_auc", "average_precision", "metric_status"]]
          .to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def run_trial(name, seed, run_dir, data, model_class, base_config):
    output = run_dir / name / f"seed{seed:02d}"
    config = {**base_config, "dataset": name, "split_seed": seed,
              "feature_columns": data["features"], "relation": data["relation"]}
    if output.exists():
        manifest = output / "metadata.json"
        require(manifest.is_file(), f"Incomplete folder {output}; inspect it and use a new --run-name.")
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        require(saved["configuration"] == config,
                f"Configuration changed for {output}; use a new --run-name.")
        require(set(saved["output_sha256"]) == {"metrics.csv", "scores.csv", "split.csv"},
                "Unexpected cached output list.")
        for filename, digest in saved["output_sha256"].items():
            require(sha256(output / filename) == digest, f"Changed cached file: {output / filename}")
        print(f"\nVerified existing run: {name}, seed {seed}")
        show_metrics(pd.read_csv(output / "metrics.csv"))
        return

    n = len(data["y"]["true"])
    train_indices, test_indices = train_test_split(
        np.arange(n), test_size=0.4, random_state=np.random.RandomState(seed)
    )
    require(len(np.unique(np.concatenate([train_indices, test_indices]))) == n,
            "Split does not partition the dataset.")
    # Match the old benchmark's threshold parameter in BOTH scoring runs.
    # In the inspected source, this affects threshold_/labels_ only, not scores.
    contamination = float(data["y"]["legacy"].mean())
    score_arrays, score_details = {}, {}
    print(f"\n{name}, seed {seed}: {len(train_indices)} train / {len(test_indices)} test", flush=True)
    for selection in ["legacy", "corrected"]:
        score_arrays[selection], score_details[selection] = score(
            model_class, data["X"][selection], data["features"][selection],
            train_indices, test_indices, contamination
        )
    if np.array_equal(data["X"]["legacy"], data["X"]["corrected"]):
        require(np.array_equal(score_arrays["legacy"], score_arrays["corrected"]),
                "Unchanged predictors produced different scores.")

    rows = []
    for condition, feature_selection, label_selection in CONDITIONS:
        y = data["y"][label_selection]
        result = evaluate(y[test_indices], score_arrays[feature_selection])
        rows.append({
            "dataset": name, "split_seed": seed, "condition": condition,
            "feature_selection": feature_selection, "evaluation_labels": label_selection,
            "n_train": len(train_indices), "n_test": len(test_indices),
            "n_features": data["X"][feature_selection].shape[1],
            "train_positives": int(y[train_indices].sum()),
            "test_positives": int(y[test_indices].sum()),
            "test_prevalence": float(y[test_indices].mean()), **result,
            # The earlier benchmark rounded each trial before taking averages.
            "roc_auc_rounded_4dp": None if result["roc_auc"] is None else round(result["roc_auc"], 4),
            "average_precision_rounded_4dp": None if result["average_precision"] is None else round(result["average_precision"], 4),
        })
    metrics = pd.DataFrame(rows)
    scores = pd.DataFrame({
        "test_position": np.arange(len(test_indices)), "row_index_zero_based": test_indices,
        "record_id": data["ids"][test_indices],
        "legacy_label": data["y"]["legacy"][test_indices],
        "true_outlier_label": data["y"]["true"][test_indices],
        "legacy_columns_score": score_arrays["legacy"],
        "corrected_columns_score": score_arrays["corrected"],
    })
    split = pd.concat([
        pd.DataFrame({"partition": partition, "position_in_partition": np.arange(len(indices)),
                      "row_index_zero_based": indices, "record_id": data["ids"][indices]})
        for partition, indices in [("train", train_indices), ("test", test_indices)]
    ], ignore_index=True)
    for relative_path, digest in config["input_sha256"].items():
        require(sha256(ROOT / relative_path) == digest, f"Protected input changed: {relative_path}")

    payloads = {"metrics.csv": metrics.to_csv(index=False), "scores.csv": scores.to_csv(index=False),
                "split.csv": split.to_csv(index=False)}
    manifest = {
        "configuration": config, "contamination_parameter_both_runs": contamination,
        "full_dataset_label_disagreements": data["label_disagreements"],
        "test_label_disagreements": int(np.count_nonzero(scores["legacy_label"] != scores["true_outlier_label"])),
        "score_runs": score_details,
        "output_sha256": {name: hashlib.sha256(content.encode("utf-8")).hexdigest()
                          for name, content in payloads.items()},
    }
    output.mkdir(parents=True, exist_ok=False)
    for filename, content in payloads.items():
        (output / filename).write_bytes(content.encode("utf-8"))
    (output / "metadata.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    show_metrics(metrics)
    for selection, detail in score_details.items():
        if detail["warnings"] or detail["nonfinite_intermediate_entries"]:
            print(f"{selection}: {len(detail['warnings'])} recorded warnings; "
                  f"{detail['nonfinite_intermediate_entries']} non-finite intermediate entries "
                  "(authors' behavior retained; see metadata.json).")
    print(f"Saved: {output.relative_to(ROOT)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--run-name", default="v1")
    args = parser.parse_args()
    require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name)), "Invalid run name.")
    require(len(set(args.datasets)) == len(args.datasets), "Duplicate datasets.")
    require(len(set(args.seeds)) == len(args.seeds)
            and all(0 <= seed < 2**32 for seed in args.seeds), "Invalid or duplicate seeds.")
    tree = ast.parse(BENCHMARK.read_text(encoding="utf-8"))
    names, paths, misplaced = audit.configuration(tree)
    require(set(args.datasets).issubset(names), f"Choose datasets from: {names}")
    legacy_reader = audit.isolated_reader(tree, BENCHMARK)
    mapping = dict(zip(names, paths))
    spec = importlib.util.spec_from_file_location("authors_copod_for_loader_audit", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    versions = {package: importlib.metadata.version(package) for package in
                ["numpy", "pandas", "scipy", "scikit-learn", "pyod", "statsmodels", "liac-arff"]}
    protected = [BENCHMARK, MODEL_PATH, HELPER, Path(__file__).resolve(),
                 *[HISTORICAL / filename for filename in ["roc.csv", "ap.csv", "prc.csv", "time.csv"]]]
    fixed_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in protected}
    base_config = {
        "schema": SCHEMA, "python": platform.python_version(), "package_versions": versions,
        "authors_commit": git_commit(ROOT / "original_code/COPOD"),
        "test_fraction": 0.4, "stratified": False,
        "scaling": "PyOD standardizer, fitted on training features separately for each feature set",
        "scoring_reference": "Authors' implementation: train plus entire test batch",
        "threshold_policy": "Legacy full-dataset positive fraction in both fits; threshold and binary predictions unused",
        "conditions": [item[0] for item in CONDITIONS],
        "interpretation_limits": [
            "Label-only correction reuses the exact legacy score array; it is not a new detector.",
            "Feature correction removes ID and restores omitted measurements together.",
            "Repeated splits overlap; they are not independent datasets.",
            "Software versions may differ from earlier runs; preserve the rerun legacy baseline.",
            "This run does not reproduce other detectors or establish a published table correction.",
        ],
    }
    run_dir = ROOT / "results/loader_corrections" / args.run_name
    for name in args.datasets:
        path = mapping[name]
        config = {**base_config, "input_sha256": {**fixed_hashes, str(path.relative_to(ROOT)): sha256(path)}}
        data = load_dataset(path, misplaced, legacy_reader)
        for seed in args.seeds:
            run_trial(name, seed, run_dir, data, module.COPOD, config)
    print("\nCompleted. Historical benchmark files were not changed.")
    print("Single trials are pilots; combine the planned seeds before drawing conclusions.")


if __name__ == "__main__":
    main()
