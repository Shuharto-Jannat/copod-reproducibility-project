from pathlib import Path

import pandas as pd


data_directory = Path("data_new/aps_failure")

training_path = (
    data_directory / "aps_failure_training_set.csv"
)

testing_path = (
    data_directory / "aps_failure_test_set.csv"
)


def inspect_dataset(file_path, dataset_name):
    data = pd.read_csv(
        file_path,
        skiprows=20,
        na_values="na"
    )

    predictors = data.drop(columns=["class"])

    class_counts = data["class"].value_counts()
    positive_cases = int(class_counts.get("pos", 0))
    positive_rate = positive_cases / len(data)

    missing_counts = predictors.isna().sum()
    columns_with_missing = missing_counts[
        missing_counts > 0
    ]

    categorical_columns = predictors.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    constant_columns = predictors.columns[
        predictors.nunique(dropna=True) <= 1
    ].tolist()

    print(dataset_name)
    print("Rows:", len(data))
    print("Total columns:", len(data.columns))
    print("Predictors:", len(predictors.columns))
    print()
    print("Class counts:")
    print(class_counts)
    print()
    print(
        "Positive-class rate:",
        round(positive_rate * 100, 4),
        "%"
    )
    print(
        "Total missing values:",
        int(predictors.isna().sum().sum())
    )
    print(
        "Columns containing missing values:",
        len(columns_with_missing)
    )
    print(
        "Categorical predictor columns:",
        categorical_columns
    )
    print(
        "Constant or completely missing columns:",
        constant_columns
    )
    print()
    print("Five columns with the most missing values:")
    print(
        columns_with_missing.sort_values(
            ascending=False
        ).head()
    )
    print("-" * 60)


inspect_dataset(
    training_path,
    "APS training dataset"
)

inspect_dataset(
    testing_path,
    "APS testing dataset"
)