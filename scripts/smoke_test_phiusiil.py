from pathlib import Path
from time import perf_counter

import pandas as pd
from pyod.models.copod import COPOD
from pyod.utils.utility import standardizer
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


DATASET_PATH = Path(
    "data_new/phiusiil/"
    "PhiUSIIL_Phishing_URL_Dataset.csv"
)

RANDOM_STATE = 42
SAMPLE_SIZE = 20_000
TARGET_PHISHING_RATE = 0.10

NON_NUMERIC_COLUMNS = [
    "FILENAME",
    "URL",
    "Domain",
    "TLD",
    "Title",
]


print("Loading PhiUSIIL dataset...")

df = pd.read_csv(DATASET_PATH)

phishing_sample_size = round(
    SAMPLE_SIZE * TARGET_PHISHING_RATE
)

legitimate_sample_size = (
    SAMPLE_SIZE - phishing_sample_size
)

legitimate_df = df[
    df["label"] == 1
].sample(
    n=legitimate_sample_size,
    random_state=RANDOM_STATE,
)

phishing_df = df[
    df["label"] == 0
].sample(
    n=phishing_sample_size,
    random_state=RANDOM_STATE,
)

evaluation_df = pd.concat(
    [legitimate_df, phishing_df],
    ignore_index=True,
).sample(
    frac=1,
    random_state=RANDOM_STATE,
).reset_index(drop=True)

# In the original dataset, label 1 means legitimate and
# label 0 means phishing. PyOD requires anomalies to be 1.
y = (
    evaluation_df["label"] == 0
).astype(int)

X = evaluation_df.drop(
    columns=NON_NUMERIC_COLUMNS + ["label"]
)

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

print("\nPhiUSIIL COPOD smoke test completed")
print("Original dataset rows:", len(df))
print("Evaluation sample:", len(evaluation_df))
print("Numeric predictors used:", X.shape[1])
print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))
print("Training phishing cases:", int(y_train.sum()))
print("Testing phishing cases:", int(y_test.sum()))
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