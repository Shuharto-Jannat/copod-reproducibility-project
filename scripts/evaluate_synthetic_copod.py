"""Compare standard PyOD COPOD with a modified training-reference scorer."""

import argparse
import hashlib
import inspect
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from pyod.models.copod import COPOD
from scipy.stats import skew
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrainingReferenceCOPOD:
    """Modified scorer: empirical tails and skewness come only from training."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        require(X.ndim == 2 and len(X) > 1, "Invalid training dimensions.")
        require(np.isfinite(X).all(), "Training values must be finite.")
        require((np.ptp(X, axis=0) > 0).all(), "Constant training feature found.")
        self.n_train, self.n_features = X.shape
        self.sorted_train = np.sort(X, axis=0)
        self.skew_sign = np.sign(skew(X, axis=0, bias=True))
        require(np.isfinite(self.skew_sign).all(), "Undefined training skewness.")
        self.probability_floor = 1.0 / self.n_train
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        require(X.ndim == 2 and X.shape[1] == self.n_features,
                "Unexpected scoring dimensions.")
        require(np.isfinite(X).all(), "Scoring values must be finite.")
        contributions = np.empty(X.shape, dtype=float)

        for j in range(self.n_features):
            reference = self.sorted_train[:, j]

            # Inclusive tails: P(training <= x) and P(training >= x).
            left_count = np.searchsorted(reference, X[:, j], side="right")
            right_count = self.n_train - np.searchsorted(
                reference, X[:, j], side="left"
            )

            left = -np.log(np.clip(
                left_count / self.n_train, self.probability_floor, 1.0
            ))
            right = -np.log(np.clip(
                right_count / self.n_train, self.probability_floor, 1.0
            ))

            # Same skewness-based aggregation as the inspected PyOD code.
            sign = self.skew_sign[j]
            skew_tail = left * -np.sign(sign - 1) + right * np.sign(sign + 1)
            contributions[:, j] = np.maximum(skew_tail, (left + right) / 2)

        return contributions.sum(axis=1)


def evaluate(root, scenario, data_seed, split_seed):
    source_stem = scenario + "_seed" + str(data_seed)
    output_stem = source_stem + "_split" + str(split_seed)
    results_dir = root / "results" / "synthetic"

    data_path = root / "data_new" / "synthetic" / (source_stem + ".csv")
    metadata_path = results_dir / (source_stem + "_metadata.json")
    validation_path = results_dir / (source_stem + "_validation.json")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))

    # Confirm that we are evaluating the data previously inspected.
    require(
        metadata["scenario"] == scenario and metadata["seed"] == data_seed,
        "Dataset metadata mismatch."
    )
    require(
        sha256(data_path) == metadata["data_sha256"],
        "Dataset hash mismatch."
    )
    require(
        validation["metadata_sha256"] == sha256(metadata_path),
        "Validation does not match this metadata file."
    )
    require(
        validation["integrity_checks"] == "passed"
        and validation["data_sha256"] == metadata["data_sha256"],
        "Matching dataset validation is required."
    )

    data = pd.read_csv(data_path, float_precision="round_trip")
    columns = metadata["feature_columns"]
    require(list(data.columns) == columns + ["label"], "Unexpected columns.")
    require(data["label"].isin([0, 1]).all(), "Invalid labels.")

    X = data[columns].to_numpy(dtype=float)
    y = data["label"].to_numpy(dtype=int)
    require(np.isfinite(X).all(), "Non-finite predictors found.")

    # One shared split for both procedures.
    # Stratification preserves the normal/anomaly proportions.
    train_rows, test_rows = train_test_split(
        np.arange(len(data)),
        test_size=0.4,
        random_state=split_seed,
        stratify=y,
    )

    X_train, X_test = X[train_rows], X[test_rows]
    y_train, y_test = y[train_rows], y[test_rows]

    require(
        np.intersect1d(train_rows, test_rows).size == 0,
        "Training and test rows overlap."
    )

    # No labels enter either fit call. No additional preprocessing is used.
    # Contamination is fixed at the pilot's designed 5%.
    # We evaluate raw scores, so prediction thresholds are not used.
    standard = COPOD(contamination=0.05, n_jobs=1).fit(X_train)
    fixed = TrainingReferenceCOPOD().fit(X_train)

    # On training rows, our formula should match PyOD's stored fit scores.
    fixed_train_scores = fixed.decision_function(X_train)
    np.testing.assert_allclose(
        fixed_train_scores,
        standard.decision_scores_,
        rtol=1e-10,
        atol=1e-10,
    )

    # Standard PyOD receives the entire test batch in one call.
    standard_scores = standard.decision_function(X_test)
    fixed_scores = fixed.decision_function(X_test)

    # Check the modified scorer using separate batches and individual rows.
    chunk_scores = np.concatenate([
        fixed.decision_function(chunk)
        for chunk in np.array_split(X_test, 4)
    ])

    single_scores = np.array([
        fixed.decision_function(X_test[i:i + 1])[0]
        for i in range(min(10, len(X_test)))
    ])

    np.testing.assert_allclose(
        fixed_scores, chunk_scores, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        fixed_scores[:len(single_scores)],
        single_scores,
        rtol=0,
        atol=1e-12,
    )

    rows = []
    for method, scores in [
        ("pyod_batch", standard_scores),
        ("fixed_training_reference", fixed_scores),
    ]:
        require(
            scores.shape == y_test.shape and np.isfinite(scores).all(),
            "Invalid anomaly scores."
        )

        rows.append({
            "scenario": scenario,
            "method": method,
            "data_seed": data_seed,
            "split_seed": split_seed,
            "n_features": X.shape[1],
            "n_train": len(train_rows),
            "n_test": len(test_rows),
            "train_anomalies": int(y_train.sum()),
            "test_anomalies": int(y_test.sum()),
            "test_prevalence": float(y_test.mean()),
            "roc_auc": float(roc_auc_score(y_test, scores)),
            "average_precision": float(
                average_precision_score(y_test, scores)
            ),
        })

    combined_sign = np.sign(
        skew(np.vstack([X_train, X_test]), axis=0, bias=True)
    )

    report = {
        "scenario": scenario,
        "data_seed": data_seed,
        "split_seed": split_seed,
        "split": "stratified 60/40; identical rows for both methods",
        "preprocessing": "none",
        "feature_columns": columns,
        "pyod_parameters": {"contamination": 0.05, "n_jobs": 1},
        "thresholds_used_for_metrics": False,
        "standard_protocol": (
            "PyOD scoring on combined training and full test batch"
        ),
        "modified_protocol": (
            "Fixed training ECDFs and training skewness signs"
        ),
        "modified_probability_floor": fixed.probability_floor,
        "modified_tail_convention": (
            "inclusive <= and >= counts divided by n_train"
        ),
        "training_score_agreement": "passed",
        "training_score_max_abs_difference": float(np.max(
            np.abs(fixed_train_scores - standard.decision_scores_)
        )),
        "modified_batch_invariance": (
            "passed: four chunks and first ten rows singly"
        ),
        "modified_chunk_max_abs_difference": float(np.max(
            np.abs(fixed_scores - chunk_scores)
        )),
        "training_skewness_signs": fixed.skew_sign.tolist(),
        "combined_skewness_signs": combined_sign.tolist(),
        "skewness_sign_changes": int(
            (fixed.skew_sign != combined_sign).sum()
        ),
        "data_sha256": sha256(data_path),
        "validation_sha256": sha256(validation_path),
        "evaluation_script_sha256": sha256(Path(__file__)),
        "pyod_scoring_source_sha256": hashlib.sha256(
            inspect.getsource(COPOD.decision_function).encode("utf-8")
        ).hexdigest(),
        "versions": {
            name: version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "pyod"]
        },
        "python": platform.python_version(),
        "metrics": rows,
    }

    # Save every row's split membership.
    membership = np.full(len(data), "train", dtype="<U5")
    membership[test_rows] = "test"

    pd.DataFrame({
        "row_index": np.arange(len(data)),
        "split": membership,
    }).to_csv(
        results_dir / (output_stem + "_split.csv"),
        index=False,
    )

    # Save both scores for exactly the same test observations.
    pd.DataFrame({
        "row_index": test_rows,
        "label": y_test,
        "pyod_batch_score": standard_scores,
        "fixed_training_reference_score": fixed_scores,
    }).to_csv(
        results_dir / (output_stem + "_scores.csv"),
        index=False,
    )

    (results_dir / (output_stem + "_evaluation.json")).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nScenario:", scenario)
    print("Training/test rows:", len(train_rows), "/", len(test_rows))
    print(
        "Training/test anomalies:",
        int(y_train.sum()), "/", int(y_test.sum())
    )
    print("Training-score agreement: passed")
    print("Modified-scorer batch invariance: passed")

    for row in rows:
        print(
            row["method"],
            "| ROC AUC:", round(row["roc_auc"], 4),
            "| Average precision:", round(row["average_precision"], 4),
        )

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    rows = []

    for scenario in ["marginal_shift", "dependence_violation"]:
        rows.extend(evaluate(root, scenario, args.seed, args.split_seed))

    name = "evaluation_seed" + str(args.seed) + "_split" + str(args.split_seed)
    output = root / "results" / "synthetic" / (name + "_metrics.csv")
    pd.DataFrame(rows).to_csv(output, index=False)

    print("\nSaved metrics:", output.relative_to(root))
    print("Single-seed pilot results; repeated runs are still required.")


if __name__ == "__main__":
    main()