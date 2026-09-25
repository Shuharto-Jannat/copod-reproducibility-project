"""Generate the two controlled COPOD pilot scenarios. No models are fitted."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


CONFIG = {
    "n_rows": 10000,
    "n_features": 20,
    "n_anomalies": 500,
    "block_size": 5,
    "within_block_correlation": 0.8,
    "shift_magnitude": 4.0,
}
SCENARIOS = ("marginal_shift", "dependence_violation")


def make_dataset(scenario, seed_sequence):
    """Return predictor/label data and separate construction annotations."""
    normal_seed, anomaly_seed, shuffle_seed = seed_sequence.spawn(3)
    normal_rng = np.random.default_rng(normal_seed)
    anomaly_rng = np.random.default_rng(anomaly_seed)
    shuffle_rng = np.random.default_rng(shuffle_seed)

    n = CONFIG["n_rows"]
    d = CONFIG["n_features"]
    n_anomalies = CONFIG["n_anomalies"]
    n_normal = n - n_anomalies
    block_size = CONFIG["block_size"]
    rho = CONFIG["within_block_correlation"]

    # Ones on the diagonal, rho within blocks, zero between blocks.
    block = np.full((block_size, block_size), rho)
    np.fill_diagonal(block, 1.0)
    covariance = np.kron(np.eye(d // block_size), block)
    normal = normal_rng.multivariate_normal(
        np.zeros(d), covariance, size=n_normal, method="cholesky"
    )

    # -1 means no feature was deliberately shifted.
    shifted_feature = np.full(n, -1, dtype=int)
    applied_shift = np.zeros(n)

    if scenario == "marginal_shift":
        anomalies = anomaly_rng.multivariate_normal(
            np.zeros(d), covariance, size=n_anomalies, method="cholesky"
        )
        selected = anomaly_rng.integers(0, d, size=n_anomalies)
        shifts = anomaly_rng.choice([-1.0, 1.0], size=n_anomalies)
        shifts = shifts * CONFIG["shift_magnitude"]
        anomalies[np.arange(n_anomalies), selected] += shifts
        shifted_feature[n_normal:] = selected
        applied_shift[n_normal:] = shifts

    elif scenario == "dependence_violation":
        anomalies = anomaly_rng.normal(size=(n_anomalies, d))

    else:
        raise ValueError("Unknown scenario: " + scenario)

    features = np.vstack([normal, anomalies])
    labels = np.concatenate([
        np.zeros(n_normal, dtype=int), np.ones(n_anomalies, dtype=int)
    ])
    feature_columns = ["x" + str(i + 1).zfill(2) for i in range(d)]
    order = shuffle_rng.permutation(n)

    data = pd.DataFrame(features[order], columns=feature_columns)
    data["label"] = labels[order]

    # Join annotations to data using its zero-based CSV row position.
    annotations = pd.DataFrame({
        "row_index": np.arange(n),
        "label": labels[order],
        "anomaly_type": np.where(labels[order] == 1, scenario, "normal"),
        "shifted_feature_index": shifted_feature[order],
        "applied_shift": applied_shift[order],
    })

    if data.shape != (n, d + 1):
        raise ValueError("Unexpected dataset dimensions.")

    if not np.isfinite(data[feature_columns].to_numpy()).all():
        raise ValueError("Missing or non-finite feature values found.")

    if int(data["label"].sum()) != n_anomalies:
        raise ValueError("Unexpected anomaly count.")

    return data, annotations, feature_columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.seed < 0:
        parser.error("--seed must be non-negative")

    # Resolve locations relative to this script, not the terminal location.
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data_new" / "synthetic"
    results_dir = root / "results" / "synthetic"

    # Each scenario has separate random streams.
    child_seeds = np.random.SeedSequence(args.seed).spawn(len(SCENARIOS))
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    files = {}

    for scenario, child_seed in zip(SCENARIOS, child_seeds):
        data, annotations, feature_columns = make_dataset(scenario, child_seed)
        stem = scenario + "_seed" + str(args.seed)
        data_path = data_dir / (stem + ".csv")
        annotation_path = data_dir / (stem + "_annotations.csv")

        data_bytes = data.to_csv(
            index=False, lineterminator="\n"
        ).encode("utf-8")

        annotation_bytes = annotations.to_csv(
            index=False, lineterminator="\n"
        ).encode("utf-8")

        metadata = {
            "generator_version": "1.0",
            "scenario": scenario,
            "seed": args.seed,
            "scenario_spawn_key": list(child_seed.spawn_key),
            "random_stream_order": ["normal", "anomaly", "shuffle"],
            "bit_generator": "PCG64",
            "config": CONFIG,
            "feature_columns": feature_columns,
            "label_meaning": {
                "0": "normal process",
                "1": "anomaly process",
            },
            "normal_count": int((data["label"] == 0).sum()),
            "anomaly_count": int((data["label"] == 1).sum()),
            "missing_values": int(data.isna().sum().sum()),
            "duplicate_feature_rows": int(
                data[feature_columns].duplicated().sum()
            ),
            "data_path": data_path.relative_to(root).as_posix(),
            "annotation_path": annotation_path.relative_to(root).as_posix(),
            "annotation_join": (
                "row_index = zero-based data row, excluding header"
            ),
            "shift_index_convention": (
                "0=x01, ..., 19=x20; -1=no shifted feature"
            ),
            "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
            "annotations_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
            "generator_sha256": source_hash,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }

        files[data_path] = data_bytes
        files[annotation_path] = annotation_bytes
        files[results_dir / (stem + "_metadata.json")] = (
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")

    # Allow identical reruns; protect existing outputs that differ.
    for path, content in files.items():
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(
                "Existing output differs; review before replacing: " + str(path)
            )

    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
        print("Saved or verified:", path.relative_to(root))

    print("Each scenario: 10,000 rows, 20 predictors, 9,500 normal, 500 anomalies.")
    print("Generation complete. Distribution checks and model evaluation come next.")


if __name__ == "__main__":
    main()