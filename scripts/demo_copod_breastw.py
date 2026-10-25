from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from pyod.utils.utility import standardizer

# Find the project and import the authors' COPOD implementation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "original_code" / "COPOD"))
from models.cod import COPOD

# Load measurements (X) and known labels (y).
data = loadmat(ROOT / "original_code/COPOD/data/breastw.mat")
X = data["X"]
y = data["y"].ravel()

# Follow the first benchmark trial: seed 0 and a 60/40 split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.4,
    random_state=np.random.RandomState(0)
)

# Calculate scaling settings using training data.
X_train_scaled, X_test_scaled = standardizer(X_train, X_test)

# Match the benchmark's contamination setting.
# This sets the classification threshold; we evaluate raw scores.
model = COPOD(contamination=float(np.count_nonzero(y) / len(y)))

print("COPOD demonstration: BreastW", flush=True)
print("Implementation: authors' original models/cod.py")
print(f"Dataset: {len(X)} rows, {X.shape[1]} features")
print(f"Training rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")
print("Scoring reference: training plus test features")
print("Running COPOD...", flush=True)

start = perf_counter()

# Fit receives features only, without labels.
model.fit(X_train_scaled)
scores = model.decision_function(X_test_scaled)

elapsed = perf_counter() - start

if not np.isfinite(scores).all():
    raise ValueError("Non-finite anomaly scores detected.")

# Use the known test labels to evaluate the scores.
print(f"ROC AUC: {roc_auc_score(y_test, scores):.4f}")
print(f"Average precision: {average_precision_score(y_test, scores):.4f}")
print(f"Model execution time: {elapsed:.3f} seconds")
print("Complete. Higher scores mean more unusual observations.")
