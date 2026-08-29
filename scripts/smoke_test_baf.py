from pathlib import Path
from time import time

import pandas as pd
from pyod.models.copod import COPOD
from pyod.utils.utility import standardizer
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


data_path = Path("data_new/baf/Base.csv")
sample_size = 20000
random_seed = 42

print("Loading BAF dataset...")
data = pd.read_csv(data_path)

X = data.drop(columns=["fraud_bool"])
y = data["fraud_bool"]

X_sample, _, y_sample, _ = train_test_split(
    X,
    y,
    train_size=sample_size,
    stratify=y,
    random_state=random_seed
)

categorical_columns = X_sample.select_dtypes(
    include=["object", "string"]
).columns.tolist()

X_sample = pd.get_dummies(
    X_sample,
    columns=categorical_columns,
    drop_first=False,
    dtype=float
)

nonconstant_columns = X_sample.columns[
    X_sample.nunique() > 1
]

X_sample = X_sample[nonconstant_columns]

X_train, X_test, y_train, y_test = train_test_split(
    X_sample,
    y_sample,
    test_size=0.4,
    stratify=y_sample,
    random_state=random_seed
)

X_train_norm, X_test_norm = standardizer(
    X_train,
    X_test
)

contamination = y_sample.mean()

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
print("BAF COPOD smoke test completed")
print("Sample size:", len(X_sample))
print("Original categorical predictors:", categorical_columns)
print("Encoded predictors used:", len(nonconstant_columns))
print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))
print("Testing fraud cases:", int(y_test.sum()))
print("Contamination:", round(contamination * 100, 4), "%")
print("ROC AUC:", round(roc_auc, 4))
print("Average precision:", round(average_precision, 4))
print("Execution time:", round(execution_time, 4), "seconds")