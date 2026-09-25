"""Check generated pilot datasets and save diagnostics without fitting models."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def require(condition, message):
    """Stop with an explanation if an integrity check fails."""
    if not condition:
        raise ValueError(message)


def group_summary(values, block_size):
    """Summarise feature distributions and relationships."""
    d = values.shape[1]
    block_id = np.arange(d) // block_size
    same_block = block_id[:, None] == block_id[None, :]
    within = same_block & ~np.eye(d, dtype=bool)
    between = ~same_block

    corr = np.corrcoef(values, rowvar=False)
    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=1)

    return {
        "rows": int(len(values)),
        "feature_mean_min": float(means.min()),
        "feature_mean_max": float(means.max()),
        "feature_std_min": float(stds.min()),
        "feature_std_max": float(stds.max()),
        "mean_within_block_correlation": float(corr[within].mean()),
        "mean_between_block_correlation": float(corr[between].mean()),
        "mean_absolute_between_block_correlation": float(
            np.abs(corr[between]).mean()
        ),
    }


def inspect(root, scenario, seed):
    stem = scenario + "_seed" + str(seed)
    data_dir = root / "data_new" / "synthetic"
    result_dir = root / "results" / "synthetic"
    figure_dir = root / "figures" / "synthetic"

    metadata_path = result_dir / (stem + "_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = metadata["config"]
    columns = metadata["feature_columns"]

    require(metadata["scenario"] == scenario, "Scenario mismatch.")
    require(metadata["seed"] == seed, "Seed mismatch.")

    data_path = data_dir / (stem + ".csv")
    annotation_path = data_dir / (stem + "_annotations.csv")

    # Verify that the CSV files match their saved fingerprints.
    for path, key in [
        (data_path, "data_sha256"),
        (annotation_path, "annotations_sha256"),
    ]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(
            digest == metadata[key],
            "File fingerprint mismatch: " + str(path),
        )

    data = pd.read_csv(data_path, float_precision="round_trip")
    annotations = pd.read_csv(annotation_path, float_precision="round_trip")

    n, d = config["n_rows"], config["n_features"]
    n_anomalies = config["n_anomalies"]
    block_size = config["block_size"]

    # Check dimensions, labels and annotation alignment.
    require(data.shape == (n, d + 1), "Unexpected dataset dimensions.")
    require(list(data.columns) == columns + ["label"], "Unexpected columns.")
    require(not data.isna().any().any(), "Missing values found.")
    require(data["label"].isin([0, 1]).all(), "Invalid labels.")
    require(
        int(data["label"].sum()) == n_anomalies,
        "Incorrect class counts.",
    )
    require(len(annotations) == n, "Annotation row count mismatch.")
    require(not annotations.isna().any().any(), "Missing annotations.")
    require(
        np.array_equal(annotations["row_index"], np.arange(n)),
        "Annotation row positions are incorrect.",
    )
    require(
        np.array_equal(annotations["label"], data["label"]),
        "Annotation labels do not match dataset labels.",
    )

    X = data[columns].to_numpy(dtype=float)
    y = data["label"].to_numpy(dtype=int)
    require(np.isfinite(X).all(), "Non-finite feature values found.")

    normal, anomaly = X[y == 0], X[y == 1]
    expected_types = np.where(y == 1, scenario, "normal")
    require(
        np.array_equal(annotations["anomaly_type"], expected_types),
        "Anomaly type annotations are inconsistent.",
    )

    normal_audit = annotations.loc[y == 0]
    require(
        (normal_audit["shifted_feature_index"] == -1).all()
        and (normal_audit["applied_shift"] == 0).all(),
        "Normal observations have unexpected shift annotations.",
    )

    summary = {
        "scenario": scenario,
        "seed": seed,
        "integrity_checks": "passed",
        "metadata_sha256": hashlib.sha256(
            metadata_path.read_bytes()
        ).hexdigest(),
        "data_sha256": metadata["data_sha256"],
        "annotations_sha256": metadata["annotations_sha256"],
        "config": config,
        "missing_values": int(data.isna().sum().sum()),
        "duplicate_feature_rows": int(
            data[columns].duplicated().sum()
        ),
        "normal": group_summary(normal, block_size),
        "anomaly": group_summary(anomaly, block_size),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }

    if scenario == "marginal_shift":
        audit = annotations.loc[y == 1]
        selected = audit["shifted_feature_index"].to_numpy()
        shifts = audit["applied_shift"].to_numpy()

        require(
            np.isin(selected, np.arange(d)).all(),
            "Invalid shifted feature index.",
        )
        require(
            (np.abs(shifts) == config["shift_magnitude"]).all(),
            "Unexpected shift magnitude.",
        )

        # Undo shifts in a temporary copy to inspect the original structure.
        restored = anomaly.copy()
        restored[np.arange(n_anomalies), selected.astype(int)] -= shifts
        summary["anomaly_after_undoing_recorded_shifts"] = group_summary(
            restored, block_size
        )

    else:
        require(
            (annotations["shifted_feature_index"] == -1).all()
            and (annotations["applied_shift"] == 0).all(),
            "Dependence scenario has unexpected shifts.",
        )

    # Save per-feature statistics separately for each class.
    tables = []
    for label, name in [(0, "normal"), (1, "anomaly")]:
        table = data.loc[y == label, columns].describe(
            percentiles=[0.01, 0.5, 0.99]
        ).T
        table.index.name = "feature"
        table = table.reset_index()
        table.insert(0, "class", name)
        tables.append(table)

    # Create four diagnostic panels.
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 9), constrained_layout=True
    )

    # Panel 1: two features, showing all observations.
    ax = axes[0, 0]
    ax.scatter(
        normal[:, 0], normal[:, 1],
        s=5, alpha=0.15, color="steelblue", label="Normal",
    )
    ax.scatter(
        anomaly[:, 0], anomaly[:, 1],
        s=12, alpha=0.5, color="darkorange", label="Anomaly",
    )
    ax.set(
        xlabel=columns[0],
        ylabel=columns[1],
        title="First two features only; all rows shown",
    )
    ax.legend()

    # Panel 2: the most extreme feature value in each observation.
    normal_max = np.abs(normal).max(axis=1)
    anomaly_max = np.abs(anomaly).max(axis=1)
    bins = np.linspace(
        0, max(normal_max.max(), anomaly_max.max()) + 0.1, 35
    )

    ax = axes[0, 1]
    ax.hist(
        normal_max, bins=bins, density=True,
        alpha=0.55, color="steelblue", label="Normal",
    )
    ax.hist(
        anomaly_max, bins=bins, density=True,
        alpha=0.55, color="darkorange", label="Anomaly",
    )
    ax.set(
        xlabel="Largest absolute feature value in each row",
        ylabel="Density within class",
        title="Extremeness across all 20 features",
    )
    ax.legend()

    # Panels 3 and 4: feature correlations for each class.
    ticks = np.arange(0, d, block_size)
    for ax, values, name in [
        (axes[1, 0], normal, "Normal"),
        (axes[1, 1], anomaly, "Anomaly"),
    ]:
        plot = ax.imshow(
            np.corrcoef(values, rowvar=False),
            vmin=-1, vmax=1, cmap="coolwarm",
        )
        ax.set_title(name + " feature correlations")
        ax.set_xticks(
            ticks, [columns[i] for i in ticks], rotation=45
        )
        ax.set_yticks(ticks, [columns[i] for i in ticks])
        fig.colorbar(
            plot, ax=ax, label="Pearson correlation", shrink=0.8
        )

    fig.suptitle(
        scenario.replace("_", " ").title() + " | seed " + str(seed)
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    pd.concat(tables, ignore_index=True).to_csv(
        result_dir / (stem + "_feature_summary.csv"), index=False
    )
    (result_dir / (stem + "_validation.json")).write_text(
        json.dumps(
            summary, indent=2, sort_keys=True, allow_nan=False
        ) + "\n",
        encoding="utf-8",
    )
    fig.savefig(
        figure_dir / (stem + "_diagnostics.png"), dpi=160
    )
    plt.close(fig)

    print("\nScenario:", scenario)
    print("Integrity checks: passed")
    print("Normal / anomaly rows:", len(normal), "/", len(anomaly))
    print("Missing values:", summary["missing_values"])
    print("Duplicate feature rows:", summary["duplicate_feature_rows"])

    for key in [
        "normal",
        "anomaly",
        "anomaly_after_undoing_recorded_shifts",
    ]:
        if key in summary:
            stats = summary[key]
            print(
                key,
                "| mean within-block correlation:",
                round(stats["mean_within_block_correlation"], 4),
                "| mean between-block correlation:",
                round(stats["mean_between_block_correlation"], 4),
            )

    print("Saved validation JSON, feature summary CSV and diagnostic PNG.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    for scenario in ["marginal_shift", "dependence_violation"]:
        inspect(root, scenario, args.seed)

    print(
        "\nFinished. Review the statistical summaries and plots before modelling."
    )


if __name__ == "__main__":
    main()