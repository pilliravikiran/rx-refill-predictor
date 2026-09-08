"""
STEP 17 - Does reading a patient's history IN ORDER help?

Model A used hand-made summaries: avg_gap, pct_late, prev_gap.
Those throw the ORDER away. "Late, late, on time" and "on time, late, late"
give the same summary.

An LSTM reads the gaps in order, so it could in principle notice that a
patient is getting worse rather than better.

So: does the order carry any extra information? This file answers it by
running three models on exactly the same rows.

Run:  python 17_lstm_refill.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

tf.random.set_seed(42)
np.random.seed(42)


# --- PIECE 1: the label and the gaps, as in file 03 ----------------------

f = pd.read_csv("data/fills.csv", parse_dates=["fill_dt"])
f = f.sort_values(["patient_id", "drug_id", "fill_dt"])
g = f.groupby(["patient_id", "drug_id"])

f["next_fill"] = g["fill_dt"].shift(-1)
f["runout"]    = f["fill_dt"] + pd.to_timedelta(f["supply_days"], unit="D")
f["came_back"] = ((f["next_fill"] - f["runout"]).dt.days <= 14).astype(int)

f["gap"]   = g["fill_dt"].diff().dt.days
f["ratio"] = f["gap"] / f["supply_days"]     # 1.0 = on time, 2.0 = twice as slow

LAST = pd.Timestamp("2026-06-30")
f = f[f["runout"] <= LAST - pd.Timedelta(days=14)]      # censoring, as always


# --- PIECE 2: turn each patient's history into a window ------------------

# Same sliding-window idea as file 16, but the window is a patient's last 6
# gaps instead of 28 days of demand.
#
#   [1.0, 1.1, 0.9, 1.0, 1.2, 1.0]  ->  did they come back on time next?
#   [2.4, 1.8, 2.0, 1.9, 1.7, 2.1]  ->  ?

SEQ = 6

X, y, dates = [], [], []
for (_patient, _drug), sub in f.groupby(["patient_id", "drug_id"]):
    ratios, labels, runouts = sub["ratio"].values, sub["came_back"].values, sub["runout"].values
    for i in range(SEQ, len(sub)):
        window = ratios[i - SEQ + 1:i + 1]
        if np.isnan(window).any():           # the very first fill has no gap
            continue
        X.append(window); y.append(labels[i]); dates.append(runouts[i])

X     = np.array(X, "float32")
y     = np.array(y, "float32")
dates = pd.to_datetime(pd.Series(dates))
train = dates < "2026-01-01"

print("Rows:", len(X), " train:", train.sum(), " test:", (~train).sum())


# --- PIECE 3: three models, same rows, same split ------------------------

# 1. LSTM. Reads the 6 gaps IN ORDER.
lstm = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQ, 1)),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(1, activation="sigmoid"),   # sigmoid squashes to 0-1
])
lstm.compile(optimizer="adam", loss="binary_crossentropy")
lstm.fit(X[train][..., None], y[train], epochs=20, batch_size=64,
         validation_split=0.1, verbose=0)
lstm_pred = lstm.predict(X[~train][..., None], verbose=0).ravel()

# 2. XGBoost on the SAME 6 numbers, but as plain columns. It has no idea
#    they are in order - column 1 and column 6 are just two columns.
xgb_raw = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.03,
                        eval_metric="logloss", random_state=42)
xgb_raw.fit(X[train], y[train])
raw_pred = xgb_raw.predict_proba(X[~train])[:, 1]

# 3. XGBoost on hand-made summaries - the file 03 approach, order thrown away.
summary = np.column_stack([
    X.mean(axis=1),          # avg_gap
    X.std(axis=1),           # how erratic
    X[:, -1],                # prev_gap
    (X > 1.4).mean(axis=1),  # pct_late
])
xgb_sum = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.03,
                        eval_metric="logloss", random_state=42)
xgb_sum.fit(summary[train], y[train])
sum_pred = xgb_sum.predict_proba(summary[~train])[:, 1]

print()
print(f"{'model':<34}{'ROC-AUC':>9}")
print(f"{'LSTM, reads the order':<34}{roc_auc_score(y[~train], lstm_pred):>9.3f}")
print(f"{'XGBoost, same 6 numbers':<34}{roc_auc_score(y[~train], raw_pred):>9.3f}")
print(f"{'XGBoost, hand-made summaries':<34}{roc_auc_score(y[~train], sum_pred):>9.3f}")
print()
print("File 07 measured the noise in this data at +/- 0.025.")
print("Compare the gaps above to that number before calling anything a winner.")
