"""
STEP 16 - Forecast demand with an LSTM, and compare it to XGBoost.

An LSTM reads a sequence one step at a time, keeping a memory as it goes.
Like reading a sentence: by the end you remember the start.

Here the "sentence" is 28 days of dispensing. The LSTM reads day 1, day 2,
day 3... and by day 28 it has an impression of how busy things have been.
Then it guesses the next 7 days.

XGBoost cannot do that. It sees 28 numbers with no idea they are in order -
day 3 and day 27 are just two columns to it. So this is a fair fight on a
problem that should suit the LSTM.

Run:  python 16_lstm_forecast.py      (needs: pip install tensorflow-cpu)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

tf.random.set_seed(42)
np.random.seed(42)


# --- PIECE 1: one row per drug per day -----------------------------------

fills = pd.read_csv("data/fills.csv", parse_dates=["fill_dt"])

daily = fills.groupby(["drug_id", "fill_dt"])["disp_qty"].sum().reset_index()
all_days = pd.date_range(fills["fill_dt"].min(), fills["fill_dt"].max(), freq="D")

# For each drug, one long list of daily units. Quiet days become 0.
series = {}
for drug_id, g in daily.groupby("drug_id"):
    series[drug_id] = g.set_index("fill_dt")["disp_qty"].reindex(all_days, fill_value=0).values

print("Drugs:", len(series), " Days:", len(all_days))


# --- PIECE 2: cut the history into windows -------------------------------

# THIS IS THE NEW IDEA.
#
# XGBoost wanted a flat table: one row, some columns.
# An LSTM wants a SEQUENCE: 28 numbers in order.
#
# So we slide a window along each drug's history:
#
#   days 1-28   -> what happened on days 29-35
#   days 2-29   -> what happened on days 30-36
#   days 3-30   -> what happened on days 31-37
#
# Each slide gives one training example. Thousands of them from one drug.

LOOK_BACK = 28      # how many days the model reads
HORIZON   = 7       # how many days ahead we add up

SPLIT = list(all_days).index(pd.Timestamp("2026-01-01"))    # train before, test after

X_train, y_train = [], []
X_test,  y_test, test_scale = [], [], []

for drug_id, values in series.items():
    # Each drug runs at a different level - Metformin dispenses far more
    # than Insulin. Divide by the drug's own normal level so the model sees
    # "1.2x busier than usual" instead of raw numbers it cannot compare.
    scale = values[:SPLIT].mean()

    for t in range(LOOK_BACK, len(values) - HORIZON):
        window = values[t - LOOK_BACK:t] / scale        # the 28 days it reads
        target = values[t:t + HORIZON].sum() / scale    # the next 7 days added up

        if t < SPLIT:
            X_train.append(window); y_train.append(target)
        else:
            X_test.append(window);  y_test.append(target); test_scale.append(scale)

# An LSTM needs a 3D shape: (how many examples, how many steps, how many
# numbers per step). We have 1 number per day, so [..., None] adds that last 1.
X_train = np.array(X_train, "float32")[..., None]
X_test  = np.array(X_test,  "float32")[..., None]
y_train = np.array(y_train, "float32")
y_test  = np.array(y_test,  "float32")
test_scale = np.array(test_scale)

print("Training windows:", X_train.shape, " Test windows:", X_test.shape)
print("  shape means: (examples, days per window, numbers per day)")


# --- PIECE 3: build the LSTM ---------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(LOOK_BACK, 1)),

    # The LSTM reads the 28 days one at a time, carrying a memory forward.
    # 32 = the size of that memory. Bigger remembers more, and overfits sooner.
    # It outputs one summary of the whole window.
    tf.keras.layers.LSTM(32),

    # Turn that summary into a single number: the next 7 days.
    tf.keras.layers.Dense(1),
])

# loss="mae" = average size of the mistake. We use MAE rather than MSE so one
# freak week does not dominate the training.
model.compile(optimizer="adam", loss="mae")

model.fit(X_train, y_train, epochs=25, batch_size=64,
          validation_split=0.1, verbose=0)

# Multiply back by the drug's scale to get real units again.
lstm_pred = model.predict(X_test, verbose=0).ravel() * test_scale
truth     = y_test * test_scale


# --- PIECE 4: the two things it has to beat ------------------------------

# The obvious guess: next week looks like last week.
# X_test[:, -7:, 0] is the last 7 days of each window.
naive_pred = X_test[:, -7:, 0].sum(axis=1) * test_scale

# XGBoost on exactly the same data, flattened from 28x1 into 28 columns.
# reshape(len, -1) means "keep the rows, squash everything else into one row".
xgb = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, random_state=42)
xgb.fit(X_train.reshape(len(X_train), -1), y_train)
xgb_pred = xgb.predict(X_test.reshape(len(X_test), -1)) * test_scale

print()
print(f"Average real demand over 7 days: {truth.mean():.0f} units")
print()
print(f"{'method':<24}{'MAE (units)':>12}")
print(f"{'next week = last week':<24}{mean_absolute_error(truth, naive_pred):>12.0f}")
print(f"{'LSTM':<24}{mean_absolute_error(truth, lstm_pred):>12.0f}")
print(f"{'XGBoost':<24}{mean_absolute_error(truth, xgb_pred):>12.0f}")
print()
print("The LSTM beats the obvious guess but loses to XGBoost - on a problem")
print("that should have suited it. See the README for why.")
