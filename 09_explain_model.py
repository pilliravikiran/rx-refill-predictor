"""
STEP 9 - SHAP: why did the model say THAT about THIS patient?

File 05 told us pct_late matters most across everyone.
That does not help a pharmacist looking at one patient.

SHAP splits a single prediction into contributions:
  start at the average, then each feature pushes it up or down.

In healthcare "the computer said so" is not an acceptable answer.
"They have been late on 8 of their last 10 refills" is.

Run it:   python 09_explain_model.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier


# --- PIECE 1: train the tuned model --------------------------------------

t = pd.read_csv("data/training_table.csv", parse_dates=["runout_dt"]).sort_values("runout_dt")
train = t[t["runout_dt"] <  "2026-01-01"]
test  = t[t["runout_dt"] >= "2026-01-01"]

FEATURES = ["patient_age", "drug_name", "is_branded", "supply_days",
            "fill_number", "prev_gap", "was_late", "avg_gap", "pct_late"]

X_train = pd.get_dummies(train[FEATURES], columns=["drug_name"]).astype(float)
X_test  = pd.get_dummies(test[FEATURES],  columns=["drug_name"])
X_test  = X_test.reindex(columns=X_train.columns, fill_value=0).astype(float)

# The settings file 08 chose.
model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.03,
                      eval_metric="logloss", random_state=42)
model.fit(X_train, train["came_back"])


# --- PIECE 2: work out the contributions ---------------------------------

sample = X_test.sample(800, random_state=42)      # 800 unseen rows is plenty and stays fast

explainer   = shap.TreeExplainer(model)           # fast exact SHAP for tree models
shap_values = explainer.shap_values(sample)       # one number per feature, per row

print("Average starting point:", round(float(explainer.expected_value), 3))


# --- PIECE 3: explain ONE patient ----------------------------------------

i = 0
patient = sample.iloc[i]
probability = float(model.predict_proba(sample.iloc[[i]])[0, 1])

print()
print("One patient:")
for field in ["pct_late", "prev_gap", "avg_gap", "patient_age", "fill_number"]:
    print(f"  {field:<14}{patient[field]:>8.2f}")
print(f"  model says {probability:.0%} chance they refill on time")

contributions = (pd.Series(shap_values[i], index=sample.columns)
                   .sort_values(key=abs, ascending=False).head(6))

print()
print("Why (minus pushes toward 'will not refill'):")
print(contributions.round(3).to_string())


# --- PIECE 4: the whole picture ------------------------------------------

# Average size of each feature's push, ignoring direction.
# This is feature importance, but measured on actual predictions.
average_impact = (pd.Series(np.abs(shap_values).mean(axis=0), index=sample.columns)
                    .sort_values(ascending=False).head(6))

print()
print("Average impact across all 800 patients:")
print(average_impact.round(3).to_string())

os.makedirs("reports", exist_ok=True)
shap.summary_plot(shap_values, sample, show=False, max_display=10)
plt.tight_layout()
plt.savefig("reports/shap.png", dpi=150)
plt.close()
print()
print("Chart saved -> reports/shap.png")
