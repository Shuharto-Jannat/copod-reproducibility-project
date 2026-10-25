"""Diagnose three MAT reproduction gaps using the original ten COPOD splits.

Pilot: python scripts/diagnose_mat_reproduction_gaps.py --datasets wine --seeds 0
All planned trials: python scripts/diagnose_mat_reproduction_gaps.py

Scores for all rows are retained for a stability diagnostic. Performance metrics
use test rows only. This is the authors' train-plus-test batch scoring protocol.
"""

from pathlib import Path
from time import perf_counter
import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import platform
import re
import subprocess
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from pyod.utils.utility import standardizer


ROOT = Path(__file__).resolve().parents[1]
AUTHORS = ROOT / "original_code/COPOD"
MODEL = AUTHORS / "models/cod.py"
BENCHMARK = ROOT / "scripts/reproduce_benchmark.py"
HISTORICAL = ROOT / "results/reproduction_mat_14_10_trials"
COMPARISON = ROOT / "results/comparison/copod_published_vs_reproduced.csv"
DATASETS = ["wine", "wbc", "satimage-2"]
SEEDS = list(range(10))
METRICS = {"roc_auc": (roc_auc_score, "roc.csv", "Paper_ROC"),
           "average_precision": (average_precision_score, "ap.csv", "Paper_AP")}
SCHEMA = "mat_reproduction_diagnostic_v1"
METRIC_TOL = 1e-12
SCORE_TOL = 1e-10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def close(left, right):
    return bool(np.isclose(left, right, rtol=0, atol=METRIC_TOL))


def check_inputs(config):
    for relative, digest in config["input_sha256"].items():
        require(sha256(ROOT / relative) == digest, f"Protected input changed: {relative}")


def calculate_metrics(labels, scores):
    if np.unique(labels).size != 2:
        return {"metric_status": "not_reported_single_class_test_set",
                **{name: None for name in METRICS},
                **{name + "_rounded_4dp": None for name in METRICS}}
    result = {"metric_status": "ok"}
    for name, (function, _, _) in METRICS.items():
        value = function(labels, scores)
        result[name] = float(value)
        # Match the original: round the metric's returned scalar before
        # converting to a JSON/CSV-friendly built-in float.
        result[name + "_rounded_4dp"] = float(round(value, 4))
    return result


def save_new_trial(folder, tables, metadata):
    payloads = {name: frame.to_csv(index=False).encode("utf-8") for name, frame in tables.items()}
    metadata["output_sha256"] = {name: hashlib.sha256(value).hexdigest()
                                for name, value in payloads.items()}
    folder.mkdir(parents=True, exist_ok=False)
    for name, content in payloads.items():
        (folder / name).write_bytes(content)
    # Written last so an interrupted write cannot look like a completed trial.
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")


def verify_trial(folder, config, y, train_indices, test_indices):
    path = folder / "metadata.json"
    require(path.is_file(), f"Incomplete trial: {folder}; inspect it and use a new --run-name.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    require(metadata["configuration"] == config,
            f"Configuration differs: {folder}; use a new --run-name.")
    require(set(metadata["output_sha256"]) == {"metrics.csv", "scores.csv", "split.csv"},
            "Unexpected saved files.")
    for name, digest in metadata["output_sha256"].items():
        require((folder / name).is_file() and sha256(folder / name) == digest,
                f"Missing/changed saved file: {folder / name}")
    metrics, scores, split = [read_csv(folder / name) for name in
                              ["metrics.csv", "scores.csv", "split.csv"]]
    require(len(metrics) == 1, "Expected one metric row per trial.")
    row = metrics.iloc[0]
    require(row.dataset == config["dataset"] and row.split_seed == config["split_seed"],
            "Metric identifiers differ.")
    require(np.array_equal(scores.row_index_zero_based, np.arange(len(y)))
            and np.array_equal(scores.label, y), "Score rows/labels differ from source data.")
    require(np.isfinite(scores.anomaly_score).all(), "Non-finite saved scores.")
    require(set(split.partition) == {"train", "test"} and len(split) == len(y), "Invalid partitions.")
    require(scores.partition.isin(["train", "test"]).all(), "Invalid score partitions.")
    for partition, indices in [("train", train_indices), ("test", test_indices)]:
        selected = split[split.partition == partition].sort_values("position_in_partition")
        require(np.array_equal(selected.position_in_partition, np.arange(len(indices)))
                and np.array_equal(selected.row_index_zero_based, indices),
                "Stored split differs from the requested original split.")
        require(np.array_equal(np.flatnonzero(scores.partition.eq(partition)), np.sort(indices)),
                "Score partition flags differ from split.")
        require(row["n_" + partition] == len(indices)
                and row[partition + "_positives"] == int(y[indices].sum()), "Saved counts differ.")
    require(close(row.test_prevalence, y[test_indices].mean()), "Test prevalence differs.")
    actual = calculate_metrics(y[test_indices], scores.anomaly_score.to_numpy()[test_indices])
    require(row.metric_status == actual["metric_status"], "Metric status differs.")
    for name, value in actual.items():
        if name == "metric_status":
            continue
        require(pd.isna(row[name]) if value is None else close(row[name], value),
                f"Recomputed metric differs: {name}, {folder}")
    return metrics, scores, metadata


def run_trial(folder, config, X, y, model_class):
    seed = config["split_seed"]
    train_indices, test_indices = train_test_split(
        np.arange(len(y)), test_size=0.4, random_state=np.random.RandomState(seed)
    )
    if folder.exists():
        result = verify_trial(folder, config, y, train_indices, test_indices)
        print(f"Verified existing run: {config['dataset']}, seed {seed}", flush=True)
        return result
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        train, test = standardizer(X[train_indices], X[test_indices])
        require(np.isfinite(train).all() and np.isfinite(test).all(), "Non-finite scaled features.")
        model = model_class(contamination=float(np.count_nonzero(y) / len(y)))
        start = perf_counter()
        model.fit(train)  # The authors' fit returns None; no labels are supplied.
        test_scores = np.asarray(model.decision_function(test), dtype=float).ravel()
        elapsed = perf_counter() - start
        contributions = np.asarray(model.O, dtype=float)
        # Same pandas summation as the unchanged source. Training-row scores
        # below are diagnostic values, not held-out evaluation predictions.
        combined_scores = np.asarray(model.O.sum(axis=1), dtype=float)
    require(contributions.shape == X.shape and len(test_scores) == len(test_indices),
            "Unexpected score/contribution shape.")
    require(np.isfinite(test_scores).all() and np.isfinite(combined_scores).all(),
            "Non-finite final scores. Stop and inspect; no nan_to_num replacement is applied.")
    require(np.array_equal(combined_scores[-len(test_indices):], test_scores),
            "Test scores do not match the original row summation.")
    combined_indices = np.concatenate([train_indices, test_indices])
    all_scores = np.empty(len(y), dtype=float)
    all_scores[combined_indices] = combined_scores
    partitions = np.full(len(y), "train", dtype=object)
    partitions[test_indices] = "test"
    stage_counts = {name: int((~np.isfinite(np.asarray(getattr(model, name), dtype=float))).sum())
                    for name in ["U_l", "U_r", "U_skew", "O"]}
    record = {
        "dataset": config["dataset"], "split_seed": seed,
        "n_train": len(train_indices), "n_test": len(test_indices), "n_features": X.shape[1],
        "train_positives": int(y[train_indices].sum()), "test_positives": int(y[test_indices].sum()),
        "test_prevalence": float(y[test_indices].mean()),
        **calculate_metrics(y[test_indices], test_scores),
        "nonfinite_intermediate_O_entries": stage_counts["O"],
        "test_rows_with_nonfinite_contributions": int((~np.isfinite(contributions[-len(test_indices):])).any(axis=1).sum()),
        "nonfinite_final_scores": 0, "recorded_warning_count": len(captured),
    }
    metrics = pd.DataFrame([record])
    scores = pd.DataFrame({"row_index_zero_based": np.arange(len(y)), "partition": partitions,
                           "label": y, "anomaly_score": all_scores})
    split = pd.concat([pd.DataFrame({"partition": part, "position_in_partition": np.arange(len(indices)),
                                    "row_index_zero_based": indices})
                       for part, indices in [("train", train_indices), ("test", test_indices)]],
                      ignore_index=True)
    metadata = {
        "configuration": config,
        "contamination_parameter": float(np.count_nonzero(y) / len(y)),
        "fit_and_score_seconds": elapsed,
        "training_constant_columns_zero_based": np.flatnonzero(np.ptp(X[train_indices], axis=0) == 0).tolist(),
        "nonfinite_intermediate_counts": stage_counts,
        "warnings": [{"category": item.category.__name__, "message": str(item.message)} for item in captured],
        "final_score_replacement": "None; the original benchmark's nan_to_num was unnecessary for finite scores.",
    }
    check_inputs(config)
    save_new_trial(folder, {"metrics.csv": metrics, "scores.csv": scores, "split.csv": split}, metadata)
    print(f"{config['dataset']}, seed {seed}: {len(train_indices)} train / {len(test_indices)} test; "
          f"{record['test_positives']} test anomalies", flush=True)
    if record["metric_status"] == "ok":
        print(f"  ROC AUC: {record['roc_auc']:.6f} | AP: {record['average_precision']:.6f}")
    else:
        print("  Metrics unavailable: test set has only one class; split retained.")
    print(f"  Numerical flags: {stage_counts['O']} non-finite O entries; {len(captured)} recorded warnings.")
    print(f"  Saved: {folder.relative_to(ROOT)}", flush=True)
    return verify_trial(folder, config, y, train_indices, test_indices)


def summarise_dataset(folder, base_config, y):
    if not all((folder / f"seed{seed:02d}" / "metadata.json").is_file() for seed in SEEDS):
        print("Historical comparison not assessed: original seeds 0-9 are not all available.\n")
        return
    frames, vectors, input_hashes = [], [], {}
    for seed in SEEDS:
        trial = folder / f"seed{seed:02d}"
        config = {**base_config, "split_seed": seed}
        train, test = train_test_split(np.arange(len(y)), test_size=0.4,
                                      random_state=np.random.RandomState(seed))
        metrics, scores, _ = verify_trial(trial, config, y, train, test)
        frames.append(metrics)
        vectors.append(scores.anomaly_score.to_numpy())
        for name in ["metadata.json", "metrics.csv", "scores.csv", "split.csv"]:
            input_hashes[str((trial / name).relative_to(ROOT))] = sha256(trial / name)
    all_metrics = pd.concat(frames, ignore_index=True)
    summary = {"dataset": base_config["dataset"], "n_planned_trials": len(SEEDS),
               "test_positives_min": int(all_metrics.test_positives.min()),
               "test_positives_max": int(all_metrics.test_positives.max()),
               "test_prevalence_mean": float(all_metrics.test_prevalence.mean())}
    comparison = read_csv(COMPARISON)
    selected = comparison[(comparison.Format == "MAT") & (comparison.Data == base_config["dataset"])]
    require(len(selected) == 1, "Missing/ambiguous recorded published comparison.")
    comparisons = []
    for metric, (_, filename, paper_column) in METRICS.items():
        values = all_metrics[metric].dropna()
        summary.update({metric + "_n_valid": len(values), metric + "_mean": values.mean(),
                        metric + "_sd": values.std(ddof=1), metric + "_min": values.min(),
                        metric + "_max": values.max()})
        old = read_csv(HISTORICAL / filename)
        historical = old[old.Data == base_config["dataset"]]
        require(len(historical) == 1, "Missing/ambiguous historical row.")
        saved = float(historical.iloc[0].COD)
        rounded_mean = float(all_metrics[metric + "_rounded_4dp"].mean())
        published = float(selected.iloc[0][paper_column])
        valid = len(values) == len(SEEDS)
        comparisons.append({
            "dataset": base_config["dataset"], "metric": metric, "n_valid_trials": len(values),
            "historical_project_mean_as_saved": saved,
            "diagnostic_full_precision_mean": values.mean(),
            "diagnostic_mean_of_trials_rounded_4dp": rounded_mean,
            "diagnostic_minus_historical": rounded_mean - saved if valid else np.nan,
            "historical_comparison_status": ("matching_within_1e-12" if close(rounded_mean, saved)
                                             else "difference_to_investigate") if valid
                                            else "not_assessed_missing_valid_trials",
            "paper_value_from_existing_comparison": published,
            "diagnostic_rounded_mean_minus_paper": rounded_mean - published if valid else np.nan,
        })
    stability = []
    for seed, vector in zip(SEEDS, vectors):
        differences = np.abs(vector - vectors[0])
        stability.append({"dataset": base_config["dataset"], "split_seed": seed,
                          "reference_split_seed": 0, "n_rows": len(y),
                          "max_absolute_score_difference": float(differences.max()),
                          "n_rows_differing_beyond_tolerance": int((differences > SCORE_TOL).sum()),
                          "absolute_tolerance": SCORE_TOL})
    tables = {"all_metrics.csv": all_metrics, "performance_summary.csv": pd.DataFrame([summary]),
              "historical_comparison.csv": pd.DataFrame(comparisons),
              "score_stability.csv": pd.DataFrame(stability)}
    payloads = {name: frame.to_csv(index=False).encode("utf-8") for name, frame in tables.items()}
    summary_meta = {"configuration": base_config, "split_seeds": SEEDS,
                    "trial_input_sha256": input_hashes,
                    "output_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()},
                    "interpretation_limits": [
                        "Only the original ten splits are analysed; no favourable seeds are selected.",
                        "SD describes variation across overlapping test sets; it is not a confidence interval.",
                        "Performance metrics evaluate test rows only. Training-row scores are diagnostic batch scores.",
                        "Score stability and metric variability alone do not identify the cause of a published gap.",
                        "Historical comparisons use four-decimal rounding per trial before averaging.",
                        "Paper values come from the existing project comparison; they are not newly verified here.",
                        "This diagnostic reruns only COPOD and does not change other detectors' results.",
                    ]}
    payloads["summary_metadata.json"] = (json.dumps(summary_meta, indent=2, allow_nan=False) + "\n").encode("utf-8")
    output = folder / "summary"
    for name, content in payloads.items():
        path = output / name
        require(not path.exists() or path.read_bytes() == content,
                f"Existing summary differs: {path}; inspect and use a new --run-name.")
    check_inputs(base_config)
    output.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        path = output / name
        if not path.exists():
            path.write_bytes(content)
    print(f"\n{base_config['dataset']}: original ten-trial summary")
    display = ["metric", "diagnostic_full_precision_mean", "historical_comparison_status",
               "diagnostic_rounded_mean_minus_paper"]
    print(tables["historical_comparison.csv"][display].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Test anomalies per split: {summary['test_positives_min']} to {summary['test_positives_max']}.")
    print(f"Largest full-row score change from seed 0: {max(s['max_absolute_score_difference'] for s in stability):.3g}.")
    print(f"Saved or verified summary: {output.relative_to(ROOT)}\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument("--run-name", default="v1")
    args = parser.parse_args()
    require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name)), "Invalid run name.")
    require(len(set(args.datasets)) == len(args.datasets) and len(set(args.seeds)) == len(args.seeds),
            "Duplicate datasets or seeds.")
    spec = importlib.util.spec_from_file_location("authors_copod_mat_diagnostic", MODEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protected = [Path(__file__).resolve(), BENCHMARK, AUTHORS / "results/benchmark_mat.py", MODEL, COMPARISON,
                 *[HISTORICAL / name for name in ["roc.csv", "ap.csv", "prc.csv", "time.csv"]]]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    common = {
        "schema": SCHEMA, "python": platform.python_version(),
        "package_versions": {p: importlib.metadata.version(p) for p in
                             ["numpy", "pandas", "scipy", "scikit-learn", "pyod", "statsmodels"]},
        "authors_commit": subprocess.check_output(["git", "-C", str(AUTHORS), "rev-parse", "HEAD"], text=True).strip(),
        "test_fraction": 0.4, "stratified": False, "subsampled": False,
        "scaling": "PyOD standardizer fitted on training features and applied to test features",
        "standardizer_source_sha256": hashlib.sha256(inspect.getsource(standardizer).encode("utf-8")).hexdigest(),
        "scoring_reference": "Authors' implementation: training features plus entire test batch",
        "threshold_policy": "Full-dataset positive fraction; raw scores evaluated, threshold predictions unused",
        "feature_names": "MAT X column indices; no semantic feature names assumed",
    }
    output = ROOT / "results/audits/mat_reproduction_gaps" / args.run_name
    for dataset in args.datasets:
        path = AUTHORS / "data" / (dataset + ".mat")
        data = loadmat(path)
        X, raw_y = np.asarray(data["X"]), np.asarray(data["y"]).ravel()
        require(X.ndim == 2 and X.shape[0] == len(raw_y), "Unexpected MAT shapes.")
        require(0 < len(X) <= 10000, "Outside this diagnostic: original benchmark's subsampling would apply.")
        require(np.isfinite(X).all() and set(np.unique(raw_y)) == {0, 1}, "Invalid features or labels.")
        y = raw_y.astype(int)
        config = {**common, "dataset": dataset, "n_rows": len(y), "n_features": X.shape[1],
                  "n_positive": int(y.sum()), "raw_X_dtype": str(X.dtype), "raw_y_dtype": str(raw_y.dtype),
                  "duplicate_feature_rows": int(len(X) - len(np.unique(X, axis=0))),
                  "full_dataset_constant_columns_zero_based": np.flatnonzero(np.ptp(X, axis=0) == 0).tolist(),
                  "input_sha256": {**hashes, str(path.relative_to(ROOT)): sha256(path)}}
        print(f"\nDataset {dataset}: {len(y)} rows, {X.shape[1]} features, {int(y.sum())} anomalies.", flush=True)
        for seed in args.seeds:
            run_trial(output / dataset / f"seed{seed:02d}", {**config, "split_seed": seed}, X, y, module.COPOD)
        summarise_dataset(output / dataset, config, y)
    print("Completed. Original scripts, datasets and historical benchmark results were not changed.")
    print("Establish historical agreement before considering further split-sensitivity experiments.")


if __name__ == "__main__":
    main()
