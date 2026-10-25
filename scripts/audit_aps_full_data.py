"""Audit the full, supplied APS train/test files without fitting any model.

Run from the project terminal:
    python scripts/audit_aps_full_data.py

The four reports are saved under results/audits/aps_full_data/v1/.
Raw files, labels and the supplied train/test split are never changed.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_new/aps_failure"
SOURCE_URL = "https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks"
SOURCE_DOI = "https://doi.org/10.24432/C51S51"
FILES = {"train": "aps_failure_training_set.csv", "test": "aps_failure_test_set.csv"}
EXPECTED_ROWS = {"train": 60000, "test": 16000}
LABELS = {"neg": 0, "pos": 1}
GROUP_COLUMNS = [
    "profile_id", "total_rows", "train_rows", "test_rows",
    "train_neg", "train_pos", "test_neg", "test_pos",
    "shared_across_split", "conflicting_labels", "cross_split_label_disagreement",
]
MEMBER_COLUMNS = ["profile_id", "partition", "row_index_zero_based", "class", "label"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_partition(path, expected_rows=None):
    """Check CSV structure before parsing; only the literal 'na' means missing."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for _ in range(20):
            require(stream.readline() != "", f"Incomplete preamble: {path.name}")
        reader = csv.reader(stream)
        columns = next(reader, [])
        require(len(columns) == 171 and columns[0] == "class",
                f"Expected class + 170 predictors after 20 preamble lines: {path.name}")
        require(len(set(columns)) == len(columns), f"Duplicate column names: {path.name}")
        n_rows = 0
        for index, row in enumerate(reader):
            require(len(row) == len(columns),
                    f"Incorrect field count in {path.name}, data row {index}")
            require(row[0] in LABELS,
                    f"Unexpected label in {path.name}, data row {index}: {row[0]!r}")
            require(all(value.strip() != "" for value in row[1:]),
                    f"Empty numeric field in {path.name}, data row {index}; expected 'na'.")
            n_rows += 1
    if expected_rows is not None:
        require(n_rows == expected_rows,
                f"Expected {expected_rows} rows in {path.name}, found {n_rows}.")
    dtypes = {column: np.float64 for column in columns[1:]}
    dtypes["class"] = str
    frame = pd.read_csv(path, skiprows=20, encoding="utf-8-sig", dtype=dtypes,
                        keep_default_na=False, na_values=["na"],
                        float_precision="round_trip")
    require(frame.columns.tolist() == columns and len(frame) == n_rows,
            f"Parsed schema/row count differs from the CSV structure: {path.name}")
    X = frame.drop(columns="class")
    values = X.to_numpy()
    require(not np.isinf(values).any(), f"Infinite predictor values: {path.name}")
    # Reject non-'na' tokens that a numeric parser could interpret as NaN.
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for _ in range(21):
            stream.readline()
        missing_tokens = sum(row.count("na") for row in csv.reader(stream))
    require(int(np.isnan(values).sum()) == missing_tokens,
            f"Unexpected missing-value tokens in {path.name}; only 'na' is allowed.")
    y = frame["class"].map(LABELS).to_numpy(dtype=np.int8)
    require(set(np.unique(y)) == {0, 1}, f"Both known classes are required: {path.name}")
    return X, y


def feature_audit(train, test):
    """Test statistics are descriptive; keep/drop decisions use training only."""
    records = []
    for column in train.columns:
        a, b = train[column], test[column]
        n_unique = int(a.nunique(dropna=True))
        reason = ("all_missing_in_training" if n_unique == 0 else
                  "one_observed_value_in_training" if n_unique == 1 else "keep")
        records.append({
            "feature": column,
            "train_missing": int(a.isna().sum()),
            "train_missing_fraction": float(a.isna().mean()),
            "test_missing": int(b.isna().sum()),
            "test_missing_fraction": float(b.isna().mean()),
            "train_unique_observed": n_unique,
            "test_unique_observed": int(b.nunique(dropna=True)),
            "train_min_observed": a.min(), "train_max_observed": a.max(),
            "test_min_observed": b.min(), "test_max_observed": b.max(),
            "recommended_keep": n_unique > 1, "training_only_reason": reason,
        })
    return pd.DataFrame(records)


def duplicate_audit(train, test, y_train, y_test):
    """Exact numeric-profile equality, with corresponding NaNs and +/-0 equal.

    Dictionary keys contain the entire canonical float64 row, not a hash digest.
    Python checks full key equality even when its internal dictionary hashes collide.
    No label, column removal, rounding, imputation or scaling enters the comparison.
    """
    values = np.concatenate([np.asarray(train, dtype=np.float64),
                             np.asarray(test, dtype=np.float64)], axis=0)
    values[np.isnan(values)] = np.nan  # Canonical representation for missing values.
    values[values == 0] = 0.0         # Normalize negative zero to positive zero.
    profiles = {}
    for index, row in enumerate(values):
        profiles.setdefault(row.tobytes(), []).append(index)
    n_train = len(train)
    y = np.concatenate([y_train, y_test])
    records, members = [], []
    for indices in profiles.values():
        if len(indices) < 2:
            continue
        profile_id = f"profile_{len(records) + 1:06d}"
        indices = np.asarray(indices, dtype=int)
        a, b = indices[indices < n_train], indices[indices >= n_train]
        a_pos, b_pos = int(y[a].sum()), int(y[b].sum())
        a_neg, b_neg = len(a) - a_pos, len(b) - b_pos
        records.append({
            "profile_id": profile_id, "total_rows": len(indices),
            "train_rows": len(a), "test_rows": len(b),
            "train_neg": a_neg, "train_pos": a_pos,
            "test_neg": b_neg, "test_pos": b_pos,
            "shared_across_split": bool(len(a) and len(b)),
            "conflicting_labels": bool(a_pos + b_pos and a_neg + b_neg),
            "cross_split_label_disagreement": bool(a_pos * b_neg or a_neg * b_pos),
        })
        for index in indices:
            partition = "train" if index < n_train else "test"
            members.append({
                "profile_id": profile_id, "partition": partition,
                "row_index_zero_based": int(index if partition == "train" else index - n_train),
                "class": "pos" if y[index] else "neg", "label": int(y[index]),
            })
    groups = pd.DataFrame(records, columns=GROUP_COLUMNS)
    membership = pd.DataFrame(members, columns=MEMBER_COLUMNS)
    summary = {}
    for partition in ["train", "test"]:
        repeated = [record for record in records if record[partition + "_rows"] >= 2]
        counts = [record[partition + "_rows"] for record in repeated]
        summary[partition] = {
            "repeated_profiles": len(repeated),
            "rows_in_repeated_profiles": sum(counts),
            "extra_copies_beyond_first": sum(count - 1 for count in counts),
            "conflicting_label_profiles": sum(bool(record[partition + "_pos"] and
                                                      record[partition + "_neg"])
                                              for record in repeated),
        }
    shared = [record for record in records if record["shared_across_split"]]
    summary["across_supplied_split"] = {
        "shared_profiles": len(shared),
        "train_rows_in_shared_profiles": sum(record["train_rows"] for record in shared),
        "test_rows_in_shared_profiles": sum(record["test_rows"] for record in shared),
        "shared_profiles_with_cross_split_label_disagreement": sum(
            record["cross_split_label_disagreement"] for record in shared),
    }
    summary["combined"] = {
        "unique_profiles": len(profiles), "repeated_profiles": len(records),
        "rows_in_repeated_profiles": len(membership),
        "extra_copies_beyond_first": len(values) - len(profiles),
        "conflicting_label_profiles": sum(record["conflicting_labels"] for record in records),
    }
    require(summary["combined"]["extra_copies_beyond_first"] ==
            summary["train"]["extra_copies_beyond_first"] +
            summary["test"]["extra_copies_beyond_first"] + len(shared),
            "Duplicate-count consistency check failed.")
    return groups, membership, summary


def partition_summary(X, y):
    return {
        "rows": len(X), "predictors": X.shape[1],
        "class_counts": {"neg": int((y == 0).sum()), "pos": int((y == 1).sum())},
        "positive_fraction": float(y.mean()),
        "missing_cells": int(X.isna().sum().sum()),
        "missing_fraction": float(X.isna().to_numpy().mean()),
        "columns_with_missing_values": int(X.isna().any(axis=0).sum()),
        "rows_with_any_missing_values": int(X.isna().any(axis=1).sum()),
        "rows_with_all_predictors_missing": int(X.isna().all(axis=1).sum()),
    }


def save_or_verify(folder, tables, metadata):
    payloads = {name: frame.to_csv(index=False).encode("utf-8")
                for name, frame in tables.items()}
    metadata["output_sha256"] = {name: hashlib.sha256(content).hexdigest()
                                 for name, content in payloads.items()}
    payloads["audit_metadata.json"] = (
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    # Check every existing output before writing anything; never replace differing evidence.
    for name, content in payloads.items():
        path = folder / name
        require(not path.exists() or path.read_bytes() == content,
                f"Existing report differs: {path}. Inspect it and use a new --run-name.")
    folder.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        path = folder / name
        if not path.exists():
            path.write_bytes(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="v1")
    args = parser.parse_args()
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name) is not None,
            "Use letters, digits, underscores or hyphens for --run-name.")
    paths = [DATA / name for name in FILES.values()]
    paths.append(DATA / "aps_failure_description.txt")
    require(all(path.is_file() for path in paths),
            "Both supplied APS CSV files and aps_failure_description.txt are required.")
    if (DATA / "aps_failure.zip").is_file():
        paths.append(DATA / "aps_failure.zip")
    paths.append(Path(__file__).resolve())
    input_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    loaded = {}
    for partition, name in FILES.items():
        print(f"Checking full {partition} file...", flush=True)
        loaded[partition] = load_partition(DATA / name, EXPECTED_ROWS[partition])
    train, y_train = loaded["train"]
    test, y_test = loaded["test"]
    require(train.columns.tolist() == test.columns.tolist(),
            "Training/test predictor names or order differ.")
    features = feature_audit(train, test)
    print("Checking exact measurement profiles across both files...", flush=True)
    groups, members, duplicates = duplicate_audit(train, test, y_train, y_test)
    metadata = {
        "schema_version": "aps_full_raw_data_audit_v1", "run_name": args.run_name,
        "source_url": SOURCE_URL, "source_doi": SOURCE_DOI,
        "input_sha256": input_hashes,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "parsing": {"preamble_lines": 20, "missing_token": "na",
                    "numeric_dtype": "float64", "float_precision": "round_trip",
                    "label_mapping": LABELS, "ordered_predictors": train.columns.tolist()},
        "partitions": {"train": partition_summary(train, y_train),
                       "test": partition_summary(test, y_test)},
        "training_only_feature_recommendation": {
            "keep": features.loc[features.recommended_keep, "feature"].tolist(),
            "exclude": features.loc[~features.recommended_keep,
                                    ["feature", "training_only_reason"]].to_dict("records"),
            "applied_to_data": False,
        },
        "duplicate_profile_summary": duplicates,
        "definitions_and_limits": [
            "Profiles use all 170 raw numeric predictors; labels are excluded from equality.",
            "Exact equality is after float64 parsing, not equality of source CSV text.",
            "Missing values match only in corresponding positions; signed zeros are equal.",
            "Full canonical row bytes are compared, not just hash digests; no tolerance is used.",
            "Repeated profiles have at least two rows within the stated file(s).",
            "Rows in repeated profiles includes the first row; extra copies excludes it.",
            "Shared profiles occur in both supplied files, even if only once in each.",
            "A conflicting-label profile contains both neg and pos labels.",
            "Cross-split label disagreement requires opposite labels on opposite sides.",
            "Membership row indices start at zero within each CSV after its header.",
            "Profile IDs follow first occurrence in training, then test file order.",
            "Shared measurements do not establish that records describe the same truck.",
            "Absence of exact duplicates does not establish statistical independence.",
            "Column removal/imputation may create additional identical profiles; check later.",
            "Feature recommendations use training data only; test statistics are descriptive.",
            "No data rows, columns, labels or supplied partitions were changed; no models ran.",
            "neg denotes failures unrelated to APS, not necessarily healthy trucks.",
        ],
    }
    for relative, digest in input_hashes.items():
        require(sha256(ROOT / relative) == digest, f"Input changed during audit: {relative}")
    folder = ROOT / "results/audits/aps_full_data" / args.run_name
    save_or_verify(folder, {"feature_audit.csv": features, "duplicate_groups.csv": groups,
                            "duplicate_members.csv": members}, metadata)
    print("\nAPS full-data audit completed; no models were run.")
    for partition, stats in metadata["partitions"].items():
        counts = stats["class_counts"]
        print(f"{partition}: {stats['rows']:,} rows, {stats['predictors']} predictors, "
              f"{counts['neg']:,} neg / {counts['pos']:,} pos, "
              f"{stats['missing_cells']:,} missing cells")
        counts = duplicates[partition]
        print(f"  Repeated profiles: {counts['repeated_profiles']}; "
              f"rows in them: {counts['rows_in_repeated_profiles']}; "
              f"extra copies: {counts['extra_copies_beyond_first']}")
    shared = duplicates["across_supplied_split"]
    print(f"Shared train/test profiles: {shared['shared_profiles']} "
          f"({shared['train_rows_in_shared_profiles']} train rows, "
          f"{shared['test_rows_in_shared_profiles']} test rows)")
    print("Conflicting-label profiles across both files:",
          duplicates["combined"]["conflicting_label_profiles"])
    excluded = metadata["training_only_feature_recommendation"]["exclude"]
    print("Training-only exclusion recommendations:", excluded)
    print("Saved or verified four audit files:", folder.relative_to(ROOT))
    print("Raw data and the supplied split are unchanged. Review this audit before evaluation.")


if __name__ == "__main__":
    main()
