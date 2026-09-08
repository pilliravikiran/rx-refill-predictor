"""
STEP 7 - Cross-validation: is our score real, or were we lucky?

So far we trained once and tested once, on 2026. That gives ONE number.
If 2026 happened to be an easy year, we would never know.

Cross-validation tests several times, on several slices of time, and gives
us an average AND a spread. The spread is the part people forget.

Run it:   python 07_cross_validate.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBClassifier


# --- PIECE 1: set up, same as before -------------------------------------

t = pd.read_csv("data/training_table.csv", parse_dates=["runout_dt"])
t = t.sort_values("runout_dt")            # cross-validation needs time order

train = t[t["runout_dt"] < "2026-01-01"]  # we do all of this INSIDE the training data

FEATURES = ["patient_age", "drug_name", "is_branded", "supply_days",
            "fill_number", "prev_gap", "was_late", "avg_gap", "pct_late"]

X = pd.get_dummies(train[FEATURES], columns=["drug_name"]).astype(float)
y = train["came_back"]

print("Rows used for cross-validation:", len(X))


# --- PIECE 2: five folds, five scores ------------------------------------

# TimeSeriesSplit keeps every test slice AFTER its training slice:
#
#   fold 1   train [.....]                test [...]
#   fold 2   train [.........]            test [...]
#   fold 3   train [.............]        test [...]
#
# The ordinary KFold would cut randomly, and train on June to predict March.
# For anything with dates, that is cheating.
cv = TimeSeriesSplit(n_splits=5)

model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                      eval_metric="logloss", random_state=42)

# cross_val_score does the whole loop: fit, score, move to the next fold.
scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

print()
print("Score on each fold:", np.round(scores, 3))
print(f"Average : {scores.mean():.3f}")
print(f"Spread  : {scores.std():.3f}")
print()
print(f"So the honest claim is {scores.mean():.3f} give or take {scores.std():.3f},")
print("not one number from one lucky slice of time.")
