from pathlib import Path

import numpy as np
import pandas as pd


DATASET_PATH = Path(
    "data_new/phiusiil/"
    "PhiUSIIL_Phishing_URL_Dataset.csv"
)

NON_NUMERIC_COLUMNS = [
    "FILENAME",
    "URL",
    "Domain",
    "TLD",
    "Title",
]


df = pd.read_csv(DATASET_PATH)

print("Dataset path:", DATASET_PATH)
print("Total rows:", len(df))
print("Total columns:", df.shape[1])
print("Reported feature columns:", df.shape[1] - 2)

print("\nOriginal label counts:")
print(df["label"].value_counts())

print(
    "\nPhishing rate:",
    round((df["label"] == 0).mean() * 100, 4),
    "%"
)

print("\nNon-numeric column cardinalities:")
print(df[NON_NUMERIC_COLUMNS].nunique())

print(
    "\nTotal missing values:",
    int(df.isna().sum().sum())
)

duplicate_count = df.drop(
    columns=["FILENAME"]
).duplicated().sum()

print(
    "\nDuplicate rows excluding FILENAME:",
    duplicate_count
)

predictors = df.drop(
    columns=NON_NUMERIC_COLUMNS + ["label"]
)

print(
    "\nNumeric predictors retained for COPOD:",
    predictors.shape[1]
)

unexpected_non_numeric = predictors.select_dtypes(
    exclude=[np.number]
).columns.tolist()

print(
    "Unexpected non-numeric predictors:",
    unexpected_non_numeric
)

infinite_counts = np.isinf(
    predictors.select_dtypes(include=[np.number])
).sum()

print("\nNumeric columns containing infinite values:")
print(infinite_counts[infinite_counts > 0])

constant_columns = [
    column
    for column in predictors.columns
    if predictors[column].nunique(dropna=False) <= 1
]

print("\nConstant numeric columns:")
print(constant_columns)

print("\nNumeric predictor summary:")
print(
    predictors.describe()
    .transpose()[["min", "max", "mean", "std"]]
)