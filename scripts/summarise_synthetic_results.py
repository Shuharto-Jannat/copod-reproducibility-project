"""Summarise the five synthetic experiments without rerunning models."""

from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = [42, 43, 44, 45, 46]
SPLIT_SEED = 42
SCENARIOS = ["marginal_shift", "dependence_violation"]
METHODS = ["pyod_batch", "fixed_training_reference"]
METRICS = ["roc_auc", "average_precision"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_results(folder):
    frames = []
    expected_pairs = {(s, m) for s in SCENARIOS for m in METHODS}
    expected_counts = {
        "n_features": 20,
        "n_train": 6000,
        "n_test": 4000,
        "train_anomalies": 300,
        "test_anomalies": 200,
    }

    for seed in SEEDS:
        name = "evaluation_seed" + str(seed) + "_split" + str(SPLIT_SEED)
        path = folder / (name + "_metrics.csv")
        require(path.is_file(), "Missing results: " + str(path))

        data = pd.read_csv(path)
        required = {
            "scenario", "method", "data_seed", "split_seed", "test_prevalence"
        } | set(METRICS) | set(expected_counts)

        require(
            required.issubset(data.columns),
            "Missing columns: " + path.name
        )
        require(
            len(data) == 4,
            "Expected four result rows: " + path.name
        )
        require(
            set(zip(data["scenario"], data["method"])) == expected_pairs,
            "Missing or repeated scenario/method pair: " + path.name
        )
        require(
            data["data_seed"].eq(seed).all()
            and data["split_seed"].eq(SPLIT_SEED).all(),
            "Unexpected seed: " + path.name
        )

        for column, expected in expected_counts.items():
            require(
                data[column].eq(expected).all(),
                "Unexpected " + column + ": " + path.name
            )

        require(
            np.allclose(
                data["test_prevalence"], 0.05, rtol=0, atol=1e-12
            ),
            "Unexpected anomaly prevalence: " + path.name
        )

        values = data[METRICS].to_numpy(dtype=float)
        require(
            np.isfinite(values).all()
            and ((values >= 0) & (values <= 1)).all(),
            "Invalid metric values: " + path.name
        )

        frames.append(data)

    return pd.concat(frames, ignore_index=True)


def build_summaries(data):
    # Sample standard deviation across the five data seeds.
    summary = data.groupby(
        ["scenario", "method"], as_index=False
    ).agg(
        n_seeds=("data_seed", "nunique"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_sd=("roc_auc", "std"),
        average_precision_mean=("average_precision", "mean"),
        average_precision_sd=("average_precision", "std"),
    )

    # Match methods using the same scenario, data seed and split seed.
    keys = ["scenario", "data_seed", "split_seed"]

    standard = data.loc[
        data["method"].eq("pyod_batch"), keys + METRICS
    ]
    fixed = data.loc[
        data["method"].eq("fixed_training_reference"), keys + METRICS
    ]

    paired = standard.merge(
        fixed,
        on=keys,
        validate="one_to_one",
        suffixes=("_standard", "_fixed"),
    )

    # Positive differences favour the modified training-reference scorer.
    for metric in METRICS:
        paired[metric + "_difference"] = (
            paired[metric + "_fixed"] - paired[metric + "_standard"]
        )

    differences = paired.groupby(
        "scenario", as_index=False
    ).agg(
        n_pairs=("data_seed", "count"),
        roc_auc_difference_mean=("roc_auc_difference", "mean"),
        roc_auc_difference_sd=("roc_auc_difference", "std"),
        average_precision_difference_mean=(
            "average_precision_difference", "mean"
        ),
        average_precision_difference_sd=(
            "average_precision_difference", "std"
        ),
    )

    return summary, paired, differences


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "results" / "synthetic"

    data = load_results(folder)
    summary, paired, differences = build_summaries(data)

    outputs = {
        "five_seed_all_metrics.csv": data,
        "five_seed_summary.csv": summary,
        "five_seed_paired_comparisons.csv": paired,
        "five_seed_paired_summary.csv": differences,
    }

    # Save full precision; round only the terminal display.
    for name, table in outputs.items():
        table.to_csv(folder / name, index=False)

    print(
        "Verified: five seeds, two scenarios and two methods "
        "(20 result rows)."
    )
    print("\nPerformance across seeds (SD = sample standard deviation):")
    print(summary.round(4).to_string(index=False))

    print("\nPaired differences: modified scorer minus standard PyOD.")
    print("Positive = modified higher; negative = standard higher.")
    print(differences.round(4).to_string(index=False))

    print(
        "\nSD describes variation across seeds; "
        "it is not a confidence interval."
    )
    print("These results apply to the tested synthetic settings.")
    print("\nSaved four summary CSV files under results/synthetic/.")


if __name__ == "__main__":
    main()