"""
STEP 10 - MLflow: remember every model you ever trained.

Right now models/refill_model_xgb.pkl is a mystery file. Which settings
made it? What did it score? When? Nobody knows.

MLflow records every training run automatically: the settings, the scores,
and the model file itself. Then a web page lets you sort them.

Think of it as git for experiments. Git tracks your code. MLflow tracks
what each version of that code actually produced.

Run it:   python 10_track_experiments.py
Then:     mlflow ui        and open http://127.0.0.1:5000
"""

import os

import joblib
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# --- PIECE 1: the data, same split as always -----------------------------

t = pd.read_csv("data/training_table.csv", parse_dates=["runout_dt"]).sort_values("runout_dt")
train = t[t["runout_dt"] <  "2026-01-01"]
test  = t[t["runout_dt"] >= "2026-01-01"]

FEATURES = ["patient_age", "drug_name", "is_branded", "supply_days",
            "fill_number", "prev_gap", "was_late", "avg_gap", "pct_late"]

X_train = pd.get_dummies(train[FEATURES], columns=["drug_name"]).astype(float)
X_test  = pd.get_dummies(test[FEATURES],  columns=["drug_name"])
X_test  = X_test.reindex(columns=X_train.columns, fill_value=0).astype(float)
y_train, y_test = train["came_back"], test["came_back"]

os.makedirs("models/experiments", exist_ok=True)

# An "experiment" is a folder for one problem. All our runs go in this one,
# so they can be compared side by side.
mlflow.set_experiment("refill-prediction")


# --- PIECE 2: one function that trains AND records ------------------------

def train_and_log(run_name, model, params):
    """Train one model and write everything about it into MLflow."""

    # Everything inside this block belongs to one run.
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)

        probability = model.predict_proba(X_test)[:, 1]
        prediction  = (probability >= 0.5).astype(int)

        scores = {
            "test_roc_auc":  roc_auc_score(y_test, probability),
            "test_accuracy": accuracy_score(y_test, prediction),
            "test_f1":       f1_score(y_test, prediction),
        }

        mlflow.log_params(params)          # the settings YOU chose
        mlflow.log_metrics(scores)         # how well it did
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_rows", len(X_train))

        # Save the model file and attach it to this run, so the run and the
        # file can never drift apart.
        path = f"models/experiments/{run_name}.pkl"
        joblib.dump({"model": model, "columns": list(X_train.columns)}, path)
        mlflow.log_artifact(path)

        print(f"{run_name:<22} ROC-AUC {scores['test_roc_auc']:.3f}"
              f"   accuracy {scores['test_accuracy']:.3f}")
        return scores["test_roc_auc"], model


# --- PIECE 3: log the three models we have built so far ------------------

print("Training and logging three models...")
print()

train_and_log(
    "logistic-baseline",
    Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]),
    {"model_type": "LogisticRegression"},
)

train_and_log(
    "xgboost-guessed",
    XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                  eval_metric="logloss", random_state=42),
    {"model_type": "XGBoost", "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
)

best_auc, best_model = train_and_log(
    "xgboost-tuned",
    XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.03,
                  eval_metric="logloss", random_state=42),
    {"model_type": "XGBoost", "n_estimators": 200, "max_depth": 3, "learning_rate": 0.03,
     "chosen_by": "GridSearchCV, file 08"},
)


# --- PIECE 4: promote the winner to the model registry -------------------

# The registry gives a model a NAME and a VERSION NUMBER, separate from the
# runs. Your API asks for "refill-risk, Production" and never has to know
# which experiment produced it.
with mlflow.start_run(run_name="xgboost-tuned-registered"):
    mlflow.log_metric("test_roc_auc", best_auc)
    mlflow.xgboost.log_model(best_model, name="model",
                             registered_model_name="refill-risk")

# --- PIECE 5: the winner becomes THE model the API serves ----------------

# One rule for the whole project:
#   models/refill_model.pkl    <- the current best. app.py loads this one.
#   models/forecast_model.pkl  <- the demand model. app.py loads this one.
#   models/experiments/*.pkl   <- every other run. Never served.
#
# Without a rule like this you end up with five .pkl files and no idea
# which one is live. That is a real production incident, not a tidiness issue.
joblib.dump({"model": best_model, "columns": list(X_train.columns)},
            "models/refill_model.pkl")

print()
print("Registered 'refill-risk' in the model registry.")
print("Promoted the tuned model -> models/refill_model.pkl  (this is what app.py serves)")
print()
print("Now run:  mlflow ui      and open http://127.0.0.1:5000")
