"""Verify and summarise saved ARFF loader-correction trials; no models are run.

Default: four affected datasets, split seeds 0--9, run directory v1.
Run from the project with: python scripts/summarise_arff_loader_corrections.py
"""

from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "results/reproduction_arff_16_10_trials"
DATASETS = ["HeartDisease", "KDDCup99", "Lymphography", "Pima"]
A = "legacy_columns_legacy_labels"
B = "legacy_columns_true_labels"
C = "corrected_columns_true_labels"
CONDITIONS = {
    A: ("legacy", "legacy", "legacy_label", "legacy_columns_score"),
    B: ("legacy", "true", "true_outlier_label", "legacy_columns_score"),
    C: ("corrected", "true", "true_outlier_label", "corrected_columns_score"),
}
COMPARISONS = {
    "label_correction_only": (A, B),
    "column_correction_with_true_labels": (B, C),
    "combined_correction": (A, C),
}
METRICS = ["roc_auc", "average_precision"]
TOL = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip", dtype={"record_id": str})


def close(left, right):
    return bool(np.isclose(left, right, rtol=0, atol=TOL))


def verify_trial(folder, dataset, seed, evidence_hashes):
    manifest_path = folder / "metadata.json"
    require(manifest_path.is_file(), f"Missing completed run: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["configuration"]
    require(config["schema"] == "arff_loader_corrections_v1", "Unexpected run schema.")
    require(config["dataset"] == dataset and config["split_seed"] == seed,
            f"Dataset/seed mismatch: {folder}")
    require(config["conditions"] == list(CONDITIONS), "Unexpected conditions.")
    require(config["test_fraction"] == 0.4 and config["stratified"] is False,
            "Unexpected splitting protocol.")
    require(set(manifest["output_sha256"]) == {"metrics.csv", "scores.csv", "split.csv"},
            "Unexpected saved output list.")
    evidence_hashes[str(manifest_path.relative_to(ROOT))] = sha256(manifest_path)
    for filename, digest in manifest["output_sha256"].items():
        path = folder / filename
        require(path.is_file() and sha256(path) == digest, f"Missing/changed file: {path}")
        evidence_hashes[str(path.relative_to(ROOT))] = digest

    metrics, scores, split = [read_csv(folder / name) for name in
                              ["metrics.csv", "scores.csv", "split.csv"]]
    require(len(metrics) == 3 and set(metrics.condition) == set(CONDITIONS),
            f"Expected exactly three conditions: {folder}")
    require(metrics.dataset.eq(dataset).all() and metrics.split_seed.eq(seed).all(),
            "Metric dataset/seed mismatch.")
    require(set(split.partition) == {"train", "test"}, "Unexpected split partitions.")
    require(np.array_equal(np.sort(split.row_index_zero_based), np.arange(len(split))),
            "Split must include every row exactly once.")
    train = split[split.partition == "train"].sort_values("position_in_partition")
    test = split[split.partition == "test"].sort_values("position_in_partition")
    for partition in [train, test]:
        require(np.array_equal(partition.position_in_partition, np.arange(len(partition))),
                "Invalid within-partition ordering.")
    require(len(test) == int(np.ceil(0.4 * len(split))), "Unexpected test size.")
    require(np.array_equal(scores.test_position, np.arange(len(test))), "Score ordering mismatch.")
    require(np.array_equal(scores.row_index_zero_based, test.row_index_zero_based)
            and np.array_equal(scores.record_id, test.record_id),
            "Saved scores do not match saved test rows.")
    require(scores[["legacy_label", "true_outlier_label"]].isin([0, 1]).all().all(),
            "Unexpected labels.")
    require(np.isfinite(scores[["legacy_columns_score", "corrected_columns_score"]]).all().all(),
            "Non-finite final scores.")
    require(int((scores.legacy_label != scores.true_outlier_label).sum())
            == manifest["test_label_disagreements"], "Label-disagreement count mismatch.")

    for row in metrics.itertuples(index=False):
        feature_set, label_set, label_column, score_column = CONDITIONS[row.condition]
        require(row.feature_selection == feature_set and row.evaluation_labels == label_set,
                "Condition meanings do not match.")
        require(row.n_train == len(train) and row.n_test == len(test), "Row count mismatch.")
        require(row.n_features == len(config["feature_columns"][feature_set]),
                "Feature count mismatch.")
        y, values = scores[label_column], scores[score_column]
        require(row.test_positives == int(y.sum()) and close(row.test_prevalence, y.mean()),
                "Test positive count/prevalence mismatch.")
        if y.nunique() < 2:
            require(row.metric_status == "not_reported_single_class_test_set"
                    and all(pd.isna(getattr(row, m)) for m in METRICS),
                    "Single-class metrics must be explicitly unavailable.")
        else:
            require(row.metric_status == "ok", "Unexpected metric status.")
            recalculated = [roc_auc_score(y, values), average_precision_score(y, values)]
            for metric, value in zip(METRICS, recalculated):
                require(close(getattr(row, metric), value), f"Recomputed {metric} differs: {folder}")
                require(close(getattr(row, metric + "_rounded_4dp"), round(value, 4)),
                        f"Rounded {metric} differs: {folder}")
    diagnostics = []
    for selection, details in manifest["score_runs"].items():
        diagnostics.append({
            "dataset": dataset, "split_seed": seed, "feature_selection": selection,
            "training_constant_features": json.dumps(details["training_constant_features"]),
            "nonfinite_intermediate_entries": details["nonfinite_intermediate_entries"],
            "test_rows_with_nonfinite_contributions": details["test_rows_with_nonfinite_contributions"],
            "recorded_warning_count": len(details["warnings"]),
            "warnings": json.dumps(details["warnings"]),
            "full_dataset_label_disagreements": manifest["full_dataset_label_disagreements"],
            "test_label_disagreements": manifest["test_label_disagreements"],
        })
    require(set(manifest["score_runs"]) == {"legacy", "corrected"}, "Missing scoring diagnostics.")
    return metrics, diagnostics, config


def summarise(metrics):
    performance = []
    for (dataset, condition), group in metrics.groupby(["dataset", "condition"], sort=False):
        row = {"dataset": dataset, "condition": condition, "n_trials": len(group),
               "test_positives_min": int(group.test_positives.min()),
               "test_positives_max": int(group.test_positives.max()),
               "test_prevalence_mean": group.test_prevalence.mean()}
        for metric in METRICS:
            values = group[metric].dropna()
            row.update({metric + "_n_valid": len(values), metric + "_mean": values.mean(),
                        metric + "_sd": values.std(ddof=1)})
        performance.append(row)

    pairs = []
    for (dataset, seed), group in metrics.groupby(["dataset", "split_seed"], sort=False):
        group = group.set_index("condition")
        for comparison, (before, after) in COMPARISONS.items():
            row = {"dataset": dataset, "split_seed": seed, "comparison": comparison,
                   "before_condition": before, "after_condition": after,
                   "before_test_prevalence": group.loc[before, "test_prevalence"],
                   "after_test_prevalence": group.loc[after, "test_prevalence"]}
            for metric in METRICS:
                row[metric + "_difference"] = group.loc[after, metric] - group.loc[before, metric]
            pairs.append(row)
    pairs = pd.DataFrame(pairs)
    paired_summary = []
    for (dataset, comparison), group in pairs.groupby(["dataset", "comparison"], sort=False):
        row = {"dataset": dataset, "comparison": comparison, "n_planned_pairs": len(group)}
        for metric in METRICS:
            differences = group[metric + "_difference"].dropna()
            row.update({metric + "_n_valid_pairs": len(differences),
                        metric + "_difference_mean": differences.mean(),
                        metric + "_difference_sd": differences.std(ddof=1),
                        metric + "_higher_count": int((differences > TOL).sum()),
                        metric + "_tied_count": int((differences.abs() <= TOL).sum()),
                        metric + "_lower_count": int((differences < -TOL).sum())})
        paired_summary.append(row)
    return pd.DataFrame(performance), pairs, pd.DataFrame(paired_summary)


def compare_historical(metrics, seeds, source_hashes, evidence_hashes):
    rows = []
    for metric, filename in [("roc_auc", "roc.csv"), ("average_precision", "ap.csv")]:
        path = HISTORICAL / filename
        relative = str(path.relative_to(ROOT))
        require(relative in source_hashes and sha256(path) == source_hashes[relative],
                f"Historical table differs from the file recorded by the trials: {path}")
        evidence_hashes[relative] = sha256(path)
        historical = read_csv(path)
        for dataset, group in metrics[metrics.condition == A].groupby("dataset", sort=False):
            matching = historical[historical.Data == dataset]
            require(len(matching) == 1, f"Ambiguous/missing historical row: {dataset}")
            saved_value = float(matching.iloc[0]["COD"])
            rounded = group[metric + "_rounded_4dp"]
            comparable = set(seeds) == set(range(10)) and rounded.notna().all()
            mean = rounded.mean()
            rows.append({
                "dataset": dataset, "metric": metric, "n_rerun_trials": len(group),
                "historical_legacy_mean_as_saved": saved_value,
                "rerun_legacy_full_precision_mean": group[metric].mean(),
                "rerun_legacy_mean_of_trials_rounded_4dp": mean,
                "same_planned_seeds_0_to_9_and_all_valid": bool(comparable),
                "rerun_minus_historical": mean - saved_value if comparable else np.nan,
                "comparison_status": ("matching_within_1e-12" if close(mean, saved_value)
                                      else "difference_to_investigate") if comparable
                                     else "not_comparable_seed_set_or_missing_metric",
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="v1")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--summary-name", default="summary")
    args = parser.parse_args()
    for name in [args.run_name, args.summary_name, *args.datasets]:
        require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", name)), f"Invalid path name: {name}")
    require(len(set(args.datasets)) == len(args.datasets), "Duplicate dataset names.")
    require(len(set(args.seeds)) == len(args.seeds) and all(0 <= s < 2**32 for s in args.seeds),
            "Invalid/duplicate seeds.")
    run_dir = ROOT / "results/loader_corrections" / args.run_name
    frames, diagnostics, evidence_hashes, source_hashes = [], [], {}, {}
    common_config = None
    for dataset in args.datasets:
        dataset_config = None
        for seed in args.seeds:
            folder = run_dir / dataset / f"seed{seed:02d}"
            frame, details, config = verify_trial(folder, dataset, seed, evidence_hashes)
            within = {k: v for k, v in config.items() if k != "split_seed"}
            require(dataset_config is None or within == dataset_config,
                    f"Mixed configurations within {dataset}.")
            dataset_config = within
            common = {k: v for k, v in within.items() if k not in
                      {"dataset", "feature_columns", "relation", "input_sha256"}}
            require(common_config is None or common == common_config,
                    "Mixed scientific protocols or software versions across datasets.")
            common_config = common
            for path, digest in config["input_sha256"].items():
                require(path not in source_hashes or source_hashes[path] == digest,
                        f"Mixed input versions: {path}")
                source_hashes[path] = digest
            frames.append(frame)
            diagnostics.extend(details)
        print(f"Verified {dataset}: {len(args.seeds)} trials.", flush=True)

    metrics = pd.concat(frames, ignore_index=True)
    require(not metrics.duplicated(["dataset", "split_seed", "condition"]).any(),
            "Duplicate result rows.")
    performance, pairs, paired_summary = summarise(metrics)
    historical = compare_historical(metrics, args.seeds, source_hashes, evidence_hashes)
    tables = {
        "all_metrics.csv": metrics,
        "performance_summary.csv": performance,
        "paired_differences.csv": pairs,
        "paired_summary.csv": paired_summary,
        "historical_comparison.csv": historical,
        "numerical_diagnostics.csv": pd.DataFrame(diagnostics),
    }
    payloads = {name: frame.to_csv(index=False).encode("utf-8") for name, frame in tables.items()}
    metadata = {
        "schema": "arff_loader_correction_summary_v1",
        "datasets": args.datasets, "split_seeds": args.seeds,
        "n_trials": len(frames), "n_metric_rows": len(metrics),
        "summary_script_sha256": sha256(Path(__file__).resolve()),
        "summary_python": platform.python_version(),
        "summary_package_versions": {p: importlib.metadata.version(p)
                                     for p in ["numpy", "pandas", "scikit-learn"]},
        "common_trial_configuration": common_config,
        "trial_recorded_input_sha256": source_hashes,
        "summary_input_sha256": evidence_hashes,
        "output_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()},
        "interpretation": [
            "Means and sample SD use valid per-trial full-precision metrics; valid counts are reported.",
            "Splits overlap: SD describes variation across splits, not a confidence interval.",
            "Paired differences are after minus before on the identical rows within each trial.",
            "Label-only correction reuses identical scores; changes are not detector improvements.",
            "Label changes affect positive identities and prevalence, including the AP reference level.",
            "Column correction with true labels compares predictors on the same labelled task.",
            "Removing ID and restoring the omitted measurement are one combined column change.",
            "Historical comparison uses recorded legacy means and rerun means of trial-rounded metrics.",
            "Numerical diagnostics retain the original scorer's non-finite intermediate contributions.",
            "These results do not correct the published table or rerun other detectors.",
        ],
    }
    payloads["summary_metadata.json"] = (json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode("utf-8")
    output = run_dir / args.summary_name
    require(not output.exists() or output.is_dir(), f"Output is not a directory: {output}")
    # Check every existing file before writing anything; never replace differing evidence.
    for name, content in payloads.items():
        path = output / name
        require(not path.exists() or path.read_bytes() == content,
                f"Existing summary differs: {path}. Use a new --summary-name.")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        path = output / name
        if not path.exists():
            path.write_bytes(content)

    print(f"\nVerified {len(frames)} trials and {len(metrics)} metric rows; no models were run.")
    print("\nPerformance means and sample SD:")
    display_columns = ["dataset", "condition", "n_trials", "roc_auc_mean", "roc_auc_sd",
                       "average_precision_mean", "average_precision_sd"]
    print(performance[display_columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    if (performance.roc_auc_n_valid != performance.n_trials).any() or \
            (performance.average_precision_n_valid != performance.n_trials).any():
        print("Some metrics are unavailable; valid trial counts are in performance_summary.csv.")
    print("\nColumn correction only, with the correct labels in both conditions:")
    columns = ["dataset", "n_planned_pairs", "roc_auc_difference_mean", "roc_auc_difference_sd",
               "average_precision_difference_mean", "average_precision_difference_sd"]
    print(paired_summary[paired_summary.comparison == "column_correction_with_true_labels"]
          [columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nHistorical legacy comparison:")
    print(historical[["dataset", "metric", "rerun_minus_historical", "comparison_status"]]
          .to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSD describes split-to-split variation, not a confidence interval.")
    print("Label corrections change the answer key; they are not detector improvements.")
    flagged = tables["numerical_diagnostics.csv"]
    flagged = flagged[(flagged.nonfinite_intermediate_entries > 0) | (flagged.recorded_warning_count > 0)]
    print(f"Scoring runs with recorded numerical flags: {len(flagged)}; see numerical_diagnostics.csv.")
    print(f"Saved or verified six CSV files and summary_metadata.json: {output.relative_to(ROOT)}")
    print("Individual trial files and historical benchmark files were not changed.")


if __name__ == "__main__":
    main()
