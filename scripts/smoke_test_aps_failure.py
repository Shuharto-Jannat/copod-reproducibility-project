from pathlib import Path
from time import time

import pandas as pd
from pyod.models.copod import COPOD
from pyod.utils.utility import standardizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


data_directory = Path("data_new/aps_failure")

training_path = (
    data_directory / "aps_failure_training_set.csv"
)

testing_path = (
    data_directory / "aps_failure_test_set.csv"
)

training_sample_size = 20000
testing_sample_size = 8000
random_seed = 42

print("Loading APS datasets...")

training_data = pd.read_csv(
    training_path,
    skiprows=20,
    na_values="na"
)

testing_data = pd.read_csv(
    testing_path,
    skiprows=20,
    na_values="na"
)

X_training_full = training_data.drop(columns=["class"])
y_training_full = training_data["class"].map(
    {"neg": 0, "pos": 1}
)

X_testing_full = testing_data.drop(columns=["class"])
y_testing_full = testing_data["class"].map(
    {"neg": 0, "pos": 1}
)

X_train, _, y_train, _ = train_test_split(
    X_training_full,
    y_training_full,
    train_size=training_sample_size,
    stratify=y_training_full,
    random_state=random_seed
)

X_test, _, y_test, _ = train_test_split(
    X_testing_full,
    y_testing_full,
    train_size=testing_sample_size,
    stratify=y_testing_full,
    random_state=random_seed
)

usable_columns = X_train.columns[
    X_train.nunique(dropna=True) > 1
]

X_train = X_train[usable_columns]
X_test = X_test[usable_columns]

imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

X_train_norm, X_test_norm = standardizer(
    X_train_imputed,
    X_test_imputed
)

contamination = y_train.mean()

model = COPOD(
    contamination=contamination,
    n_jobs=1
)

start_time = time()

model.fit(X_train_norm)
test_scores = model.decision_function(X_test_norm)

execution_time = time() - start_time

roc_auc = roc_auc_score(y_test, test_scores)
average_precision = average_precision_score(
    y_test,
    test_scores
)

print()
print("APS COPOD smoke test completed")
print("Training sample:", len(X_train))
print("Testing sample:", len(X_test))
print("Predictors used:", len(usable_columns))
print("Training positive cases:", int(y_train.sum()))
print("Testing positive cases:", int(y_test.sum()))
print(
    "Training contamination:",
    round(contamination * 100, 4),
    "%"
)
print("ROC AUC:", round(roc_auc, 4))
print(
    "Average precision:",
    round(average_precision, 4)
)
print(
    "Execution time:",
    round(execution_time, 4),
    "seconds"
)