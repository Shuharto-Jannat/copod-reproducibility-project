from pathlib import Path

import numpy as np
import pandas as pd


DATASET_PATH = Path(
    "data_new/rt_iot2022/RT_IOT2022.csv"
)

BENIGN_LABELS = {
    "MQTT_Publish",
    "Thing_Speak",
    "Wipro_bulb",
}


df = pd.read_csv(DATASET_PATH)

print("Dataset path:", DATASET_PATH)
print("Total rows:", len(df))
print("Total columns:", df.shape[1])
print("Predictor columns:", df.shape[1] - 2)

print("\nAttack-type counts:")
print(df["Attack_type"].value_counts())

binary_label = (~df["Attack_type"].isin(BENIGN_LABELS)).astype(int)

print("\nBinary class counts:")
print(
    binary_label.value_counts()
    .rename(index={0: "benign", 1: "attack"})
)

print(
    "\nAttack rate:",
    round(binary_label.mean() * 100, 4),
    "%"
)

predictors = df.drop(
    columns=["id", "Attack_type"]
)

categorical_columns = predictors.select_dtypes(
    include=["object", "string", "str"]
).columns.tolist()

print("\nCategorical predictor columns:")
print(categorical_columns)

print(
    "\nTotal missing values:",
    int(predictors.isna().sum().sum())
)

numeric_predictors = predictors.select_dtypes(
    include=[np.number]
)

infinite_counts = np.isinf(
    numeric_predictors
).sum()

print("\nNumeric columns containing infinite values:")
print(infinite_counts[infinite_counts > 0])

constant_columns = [
    column
    for column in predictors.columns
    if predictors[column].nunique(dropna=False) <= 1
]

print("\nConstant columns:")
print(constant_columns)

duplicate_count = df.drop(
    columns=["id"]
).duplicated().sum()

print(
    "\nDuplicate rows excluding id:",
    duplicate_count
)

deduplicated_df = df.drop(
    columns=["id"]
).drop_duplicates()

deduplicated_label = (
    ~deduplicated_df["Attack_type"].isin(BENIGN_LABELS)
).astype(int)

print(
    "Rows after removing duplicates:",
    len(deduplicated_df)
)

print("\nClass counts after removing duplicates:")
print(
    deduplicated_label.value_counts()
    .rename(index={0: "benign", 1: "attack"})
)

print(
    "\nAttack rate after removing duplicates:",
    round(deduplicated_label.mean() * 100, 4),
    "%"
)