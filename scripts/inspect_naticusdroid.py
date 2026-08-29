from pathlib import Path

import pandas as pd


DATASET_PATH = Path(
    "data_new/naticusdroid/data.csv"
)

TARGET_COLUMN = "Result"


df = pd.read_csv(DATASET_PATH)

feature_columns = [
    column
    for column in df.columns
    if column != TARGET_COLUMN
]

print("Dataset path:", DATASET_PATH)
print("Total rows:", len(df))
print("Total columns:", df.shape[1])
print("Permission features:", len(feature_columns))

print("\nOriginal class counts:")
print(df[TARGET_COLUMN].value_counts())

print(
    "\nOriginal malware rate:",
    round(df[TARGET_COLUMN].mean() * 100, 4),
    "%"
)

print(
    "\nTotal missing values:",
    int(df.isna().sum().sum())
)

feature_values = sorted(
    pd.unique(
        df[feature_columns].to_numpy().ravel()
    ).tolist()
)

print("\nUnique permission values:")
print(feature_values)

exact_duplicate_count = df.duplicated().sum()

print(
    "\nExact duplicate rows:",
    exact_duplicate_count
)

profile_groups = df.groupby(
    feature_columns,
    dropna=False,
)[TARGET_COLUMN]

print(
    "Unique permission profiles:",
    profile_groups.ngroups
)

conflicting_profile_count = (
    profile_groups.nunique() > 1
).sum()

print(
    "Profiles with conflicting labels:",
    int(conflicting_profile_count)
)

profile_label_counts = profile_groups.transform(
    "nunique"
)

conflicting_row_count = (
    profile_label_counts > 1
).sum()

print(
    "Rows belonging to conflicting profiles:",
    int(conflicting_row_count)
)

clean_df = (
    df[profile_label_counts == 1]
    .drop_duplicates(subset=feature_columns)
    .reset_index(drop=True)
)

print(
    "\nUnambiguous unique profiles:",
    len(clean_df)
)

print("\nCleaned class counts:")
print(clean_df[TARGET_COLUMN].value_counts())

print("\nCleaned class percentages:")
print(
    (
        clean_df[TARGET_COLUMN]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)

constant_columns = [
    column
    for column in feature_columns
    if clean_df[column].nunique(dropna=False) <= 1
]

print("\nConstant features after cleaning:")
print(constant_columns)