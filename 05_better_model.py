"""
STEP 5 - Try a stronger model, and see if it actually helps.

Logistic Regression draws one straight line through the data.
XGBoost asks lots of yes/no questions instead. We try it and compare.

Run it:   python 05_better_model.py
"""

import pandas as pd


# --- PIECE 1: same setup as file 4 ----------------------------------------

# You have seen all of this before. Split by date, features and answer apart,
# words turned into 0/1 columns.

t = pd.read_csv("data/training_table.csv", parse_dates=["runout_dt"])

train = t[t["runout_dt"] <  "2026-01-01"]
test  = t[t["runout_dt"] >= "2026-01-01"]

FEATURES = ["patient_age", "drug_name", "is_branded", "supply_days",
            "fill_number", "prev_gap", "was_late", "avg_gap", "pct_late"]
LABEL = "came_back"

X_train = pd.get_dummies(train[FEATURES], columns=["drug_name"]).astype(float)
X_test  = pd.get_dummies(test[FEATURES],  columns=["drug_name"])
X_test  = X_test.reindex(columns=X_train.columns, fill_value=0).astype(float)

y_train, y_test = train[LABEL], test[LABEL]

print("Train:", X_train.shape, " Test:", X_test.shape)


# --- PIECE 2: train both models and compare -------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

# The old model, so we have something to compare against.
simple = LogisticRegression(max_iter=1000)
simple.fit(X_train, y_train)
guess_simple = simple.predict_proba(X_test)[:, 1]

# The new one.
#   n_estimators  = how many small trees to build
#   max_depth     = how many questions deep each tree may go
#   learning_rate = how big a correction each tree is allowed to make
#                   (small = slow and careful, less likely to over-memorise)
strong = XGBClassifier(n_estimators=300,
                       max_depth=4,
                       learning_rate=0.05,
                       eval_metric="logloss",
                       random_state=42)
strong.fit(X_train, y_train)
guess_strong = strong.predict_proba(X_test)[:, 1]

lazy = (y_test == 1).mean()

print()
print(f"{'model':<22}{'accuracy':>10}{'roc auc':>10}")
print(f"{'lazy (always yes)':<22}{lazy:>10.3f}{'-':>10}")
for name, g in [("logistic regression", guess_simple), ("xgboost", guess_strong)]:
    acc = accuracy_score(y_test, (g >= 0.5).astype(int))
    auc = roc_auc_score(y_test, g)
    print(f"{name:<22}{acc:>10.3f}{auc:>10.3f}")


# --- PIECE 3: which features did the model actually use? ------------------

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# XGBoost keeps a score for each feature: how much it helped when the model
# used it to split. Bigger = more useful. They all add up to 1.
importance = pd.Series(strong.feature_importances_, index=X_train.columns)
importance = importance.sort_values(ascending=False)

print()
print("Top features the model relied on:")
print(importance.head(8).round(3).to_string())

os.makedirs("reports", exist_ok=True)
importance.head(10).sort_values().plot(kind="barh", figsize=(8, 5))
plt.title("What the model pays attention to")
plt.xlabel("importance")
plt.tight_layout()
plt.savefig("reports/feature_importance.png", dpi=150)
print()
print("Chart saved -> reports/feature_importance.png")


# --- Save the better model -------------------------------------------------

import joblib

os.makedirs("models/experiments", exist_ok=True)
joblib.dump({"model": strong, "columns": list(X_train.columns)},
            "models/experiments/xgboost_guessed.pkl")
print("Saved  -> models/experiments/xgboost_guessed.pkl")
