from pathlib import Path
from time import perf_counter

import pandas as pd
from pyod.models.copod import COPOD
from pyod.utils.utility import standardizer
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


DATASET_PATH = Path(
    "data_new/rt_iot2022/RT_IOT2022.csv"
)

RANDOM_STATE = 42
TARGET_ATTACK_RATE = 0.10

BENIGN_LABELS = {
    "MQTT_Publish",
    "Thing_Speak",
    "Wipro_bulb",
}


print("Loading RT-IoT2022 dataset...")

df = pd.read_csv(DATASET_PATH)

# The id column is only a row identifier and must not be used
# as a predictor. Identical feature-and-label rows are removed
# before the training and testing split.
df = (
    df.drop(columns=["id"])
    .drop_duplicates()
    .reset_index(drop=True)
)

benign_df = df[
    df["Attack_type"].isin(BENIGN_LABELS)
].copy()

attack_df = df[
    ~df["Attack_type"].isin(BENIGN_LABELS)
].copy()

# Determine how many attacks produce the target contamination
# while retaining every available benign observation.
number_of_attacks = round(
    len(benign_df)
    * TARGET_ATTACK_RATE
    / (1 - TARGET_ATTACK_RATE)
)

# Retain at least one observation from every attack category.
mandatory_attacks = (
    attack_df.groupby(
        "Attack_type",
        group_keys=False,
    )
    .sample(
        n=1,
        random_state=RANDOM_STATE,
    )
)

remaining_attack_pool = attack_df.drop(
    index=mandatory_attacks.index
)

remaining_attack_count = (
    number_of_attacks - len(mandatory_attacks)
)

additional_attacks, _ = train_test_split(
    remaining_attack_pool,
    train_size=remaining_attack_count,
    stratify=remaining_attack_pool["Attack_type"],
    random_state=RANDOM_STATE,
)

sampled_attacks = pd.concat(
    [mandatory_attacks, additional_attacks],
    ignore_index=True,
).sample(
    frac=1,
    random_state=RANDOM_STATE,
).reset_index(drop=True)

evaluation_df = pd.concat(
    [benign_df, sampled_attacks],
    ignore_index=True,
).sample(
    frac=1,
    random_state=RANDOM_STATE,
).reset_index(drop=True)

y = (
    ~evaluation_df["Attack_type"].isin(BENIGN_LABELS)
).astype(int)

X = evaluation_df.drop(columns=["Attack_type"])

categorical_columns = X.select_dtypes(
    include=["object", "string", "str"]
).columns.tolist()

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=False,
    dtype=float,
)

constant_columns = [
    column
    for column in X.columns
    if X[column].nunique(dropna=False) <= 1
]

X = X.drop(columns=constant_columns)

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

print("\nRT-IoT2022 COPOD smoke test completed")
print("Rows after duplicate removal:", len(df))
print("Benign records retained:", len(benign_df))
print("Attack records sampled:", len(sampled_attacks))
print("Evaluation sample:", len(evaluation_df))
print("Original categorical predictors:", categorical_columns)
print("Constant predictors removed:", constant_columns)
print("Encoded predictors used:", X.shape[1])
print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))
print("Training attack cases:", int(y_train.sum()))
print("Testing attack cases:", int(y_test.sum()))
print(
    "Training contamination:",
    round(contamination * 100, 4),
    "%",
)

print("\nSampled attack categories:")
print(sampled_attacks["Attack_type"].value_counts())

print("\nROC AUC:", round(roc_auc, 4))
print(
    "Average precision:",
    round(average_precision, 4),
)
print(
    "Execution time:",
    round(execution_time, 4),
    "seconds",
)