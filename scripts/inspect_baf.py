from pathlib import Path

import pandas as pd


data_path = Path("data_new/baf/Base.csv")
chunk_size = 100000

total_rows = 0
fraud_counts = pd.Series(dtype="int64")
missing_counts = None
minus_one_counts = None
column_types = None

for chunk in pd.read_csv(data_path, chunksize=chunk_size):
    total_rows = total_rows + len(chunk)

    current_fraud_counts = chunk["fraud_bool"].value_counts()
    fraud_counts = fraud_counts.add(
        current_fraud_counts,
        fill_value=0
    )

    current_missing_counts = chunk.isna().sum()
    current_minus_one_counts = chunk.select_dtypes(
        include="number"
    ).eq(-1).sum()

    if missing_counts is None:
        missing_counts = current_missing_counts
        minus_one_counts = current_minus_one_counts
        column_types = chunk.dtypes
    else:
        missing_counts = missing_counts.add(
            current_missing_counts,
            fill_value=0
        )
        minus_one_counts = minus_one_counts.add(
            current_minus_one_counts,
            fill_value=0
        )

fraud_counts = fraud_counts.astype("int64")
fraud_rows = int(fraud_counts.get(1, 0))
fraud_rate = fraud_rows / total_rows

categorical_columns = [
    column
    for column, data_type in column_types.items()
    if str(data_type) in ["object", "str", "string"]
]

print("Dataset path:", data_path)
print("Total rows:", total_rows)
print("Total columns:", len(column_types))
print("Predictor columns:", len(column_types) - 1)
print()
print("Fraud label counts:")
print(fraud_counts.sort_index())
print()
print("Fraud rate:", round(fraud_rate * 100, 4), "%")
print()
print("Categorical columns:")
print(categorical_columns)
print()
print("Actual missing values:")
print(missing_counts[missing_counts > 0].sort_values(ascending=False))
print()
print("Numeric columns containing -1:")
print(minus_one_counts[minus_one_counts > 0].sort_values(ascending=False))