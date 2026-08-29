from pathlib import Path
from time import perf_counter

import pandas as pd
from pyod.models.copod import COPOD
from pyod.utils.utility import standardizer
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


DATASET_PATH = Path(
    "data_new/naticusdroid/data.csv"
)

TARGET_COLUMN = "Result"
RANDOM_STATE = 42
TARGET_MALWARE_RATE = 0.10


print("Loading NATICUSdroid dataset...")

df = pd.read_csv(DATASET_PATH)

feature_columns = [
    column
    for column in df.columns
    if column != TARGET_COLUMN
]

# Identify permission profiles associated with more than one
# label. These profiles cannot be distinguished using the
# available features and are excluded before evaluation.
profile_label_counts = df.groupby(
    feature_columns,
    dropna=False,
)[TARGET_COLUMN].transform("nunique")

clean_df = (
    df[profile_label_counts == 1]
    .drop_duplicates(subset=feature_columns)
    .reset_index(drop=True)
)

benign_df = clean_df[
    clean_df[TARGET_COLUMN] == 0
].copy()

malware_df = clean_df[
    clean_df[TARGET_COLUMN] == 1
].copy()

number_of_malware_profiles = round(
    len(benign_df)
    * TARGET_MALWARE_RATE
    / (1 - TARGET_MALWARE_RATE)
)

sampled_malware = malware_df.sample(
    n=number_of_malware_profiles,
    random_state=RANDOM_STATE,
)

evaluation_df = pd.concat(
    [benign_df, sampled_malware],
    ignore_index=True,
).sample(
    frac=1,
    random_state=RANDOM_STATE,
).reset_index(drop=True)

X = evaluation_df[feature_columns]
y = evaluation_df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.40,
    stratify=y,
    random_state=RANDOM_STATE,
)

X_train_norm, X_test_norm = standardizer(
    X_train,
    X_test,
)

contamination = y_train.mean()

model = COPOD(
    contamination=contamination,
    n_jobs=1,
)

start_time = perf_counter()

model.fit(X_train_norm)
test_scores = model.decision_function(X_test_norm)

execution_time = perf_counter() - start_time

roc_auc = roc_auc_score(
    y_test,
    test_scores,
)

average_precision = average_precision_score(
    y_test,
    test_scores,
)

print("\nNATICUSdroid COPOD smoke test completed")
print("Original rows:", len(df))
print("Unambiguous unique profiles:", len(clean_df))
print("Benign profiles retained:", len(benign_df))
print("Malware profiles sampled:", len(sampled_malware))
print("Evaluation sample:", len(evaluation_df))
print("Permission predictors used:", X.shape[1])
print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))
print("Training malware cases:", int(y_train.sum()))
print("Testing malware cases:", int(y_test.sum()))
print(
    "Training contamination:",
    round(contamination * 100, 4),
    "%",
)
print("ROC AUC:", round(roc_auc, 4))
print(
    "Average precision:",
    round(average_precision, 4),
)
print(
    "Execution time:",
    round(execution_time, 4),
    "seconds",
)