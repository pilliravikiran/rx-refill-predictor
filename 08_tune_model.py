"""
STEP 8 - Hyperparameter tuning: stop guessing the settings.

n_estimators, max_depth, learning_rate - I chose those by guessing.
Here we try many combinations and let cross-validation pick the winner.

THE RULE: choose the settings using cross-validation on the TRAINING data.
The 2026 test set stays sealed. If you pick settings by looking at the test
score, the test set has helped you choose, and it is no longer an honest test.

Run it:   python 08_tune_model.py     (takes about a minute)
"""

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBClassifier


# --- PIECE 1: data, split by time ----------------------------------------

t = pd.read_csv("data/training_table.csv", parse_dates=["runout_dt"]).sort_values("runout_dt")

train = t[t["runout_dt"] <  "2026-01-01"]
test  = t[t["runout_dt"] >= "2026-01-01"]

FEATURES = ["patient_age", "drug_name", "is_branded", "supply_days",
            "fill_number", "prev_gap", "was_late", "avg_gap", "pct_late"]

X_train = pd.get_dummies(train[FEATURES], columns=["drug_name"]).astype(float)
X_test  = pd.get_dummies(test[FEATURES],  columns=["drug_name"])
X_test  = X_test.reindex(columns=X_train.columns, fill_value=0).astype(float)
y_train, y_test = train["came_back"], test["came_back"]


# --- PIECE 2: the search grid --------------------------------------------

# Every combination of these gets tried: 2 x 3 x 2 = 12 settings.
# Each one is scored on 5 folds, so 60 models get trained. That is why it
# takes a minute. Keep grids small - they multiply out fast.
grid = {
    "n_estimators":  [200, 400],       # how many small trees
    "max_depth":     [3, 4, 6],        # how many questions deep each tree may go
    "learning_rate": [0.03, 0.1],      # how big a correction each tree makes
}

search = GridSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42),
    grid,
    cv=TimeSeriesSplit(n_splits=5),    # same honest splitting as file 07
    scoring="roc_auc",
    n_jobs=-1,                         # use every CPU core
)

search.fit(X_train, y_train)           # the test set is NOT involved here

print("Best settings :", search.best_params_)
print(f"Best CV score : {search.best_score_:.3f}")


# --- PIECE 3: the leaderboard --------------------------------------------

# Do not just take the winner. Look at how close the others were.
results = (pd.DataFrame(search.cv_results_)
             [["param_max_depth", "param_n_estimators",
               "param_learning_rate", "mean_test_score", "std_test_score"]]
             .sort_values("mean_test_score", ascending=False))

print()
print("Top settings, best first:")
print(results.head(6).round(3).to_string(index=False))


# --- PIECE 4: NOW open the test set --------------------------------------

# Only at the very end, and only once.
default_model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              eval_metric="logloss", random_state=42)
default_model.fit(X_train, y_train)

default_auc = roc_auc_score(y_test, default_model.predict_proba(X_test)[:, 1])
tuned_auc   = roc_auc_score(y_test, search.best_estimator_.predict_proba(X_test)[:, 1])

print()
print(f"My guessed settings, test ROC-AUC : {default_auc:.3f}")
print(f"Tuned settings,      test ROC-AUC : {tuned_auc:.3f}")
print(f"Gain                              : {tuned_auc - default_auc:+.3f}")
print()
print("Compare that gain to the +/- 0.025 spread from file 07 before")
print("calling it an improvement.")
