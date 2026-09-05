"""
STEP 6 - Model B: how much stock will we need?

Model A answered a yes/no question about one patient.
Model B answers a NUMBER question about the whole pharmacy:

    "How many units of each drug will we dispense over the next 7 days?"

That is what an owner needs to decide how much to order.

Two things are new here:
  1. We predict a number, not yes/no. That is called REGRESSION.
  2. Time is the main thing. That is called a TIME SERIES.

Run it:   python 06_forecast_demand.py
"""

import pandas as pd


# --- PIECE 1: turn pickups into a daily calendar --------------------------

fills = pd.read_csv("data/fills.csv", parse_dates=["fill_dt"])

# Add up everything dispensed for each drug on each day.
daily = fills.groupby(["drug_id", "fill_dt"])["disp_qty"].sum().reset_index()

# PROBLEM: days where nothing was dispensed have NO ROW at all.
# A forecast needs a full calendar. A quiet day is a real 0, not a missing day.
# If we skip them, every average we calculate later comes out too high.
all_days = pd.date_range(fills["fill_dt"].min(), fills["fill_dt"].max(), freq="D")

frames = []
for drug, g in daily.groupby("drug_id"):
    # reindex forces this drug onto the full calendar.
    # Any day that was not there gets filled with 0.
    s = g.set_index("fill_dt")["disp_qty"].reindex(all_days, fill_value=0)
    frames.append(pd.DataFrame({"date": all_days, "drug_id": drug, "units": s.values}))

ts = pd.concat(frames).sort_values(["drug_id", "date"]).reset_index(drop=True)

print("Days in the calendar:", len(all_days))
print("Drugs               :", ts["drug_id"].nunique())
print("Rows (days x drugs) :", len(ts))
print("Quiet days (0 units):", (ts["units"] == 0).sum())
print()
print("Drug 1, first 8 days:")
print(ts[ts["drug_id"] == 1].head(8).to_string(index=False))


# --- PIECE 2: features. A forecast can only look BACKWARDS ---------------

by = ts.groupby("drug_id")

# LAGS: "how many units did we dispense N days ago?"
# shift(7) copies the value from 7 rows above into this row.
# Same idea as prev_gap in file 3, just at fixed distances.
for lag in [1, 7, 14, 28]:
    ts[f"lag_{lag}"] = by["units"].shift(lag)

# ROLLING AVERAGES: "what was our normal level over the last week / month?"
# One day is noisy. An average smooths it out.
#
# Read the .shift(1).rolling(7).mean() carefully:
#   .shift(1)     first move everything down one row, so TODAY is excluded
#   .rolling(7)   then take a 7 row window
#   .mean()       average it
#
# Without that .shift(1), today's own units would be inside the average.
# We would be using the answer to predict the answer. Leakage again.
ts["roll7"]  = by["units"].transform(lambda s: s.shift(1).rolling(7).mean())
ts["roll28"] = by["units"].transform(lambda s: s.shift(1).rolling(28).mean())

# Calendar features. These are safe - you always know what day tomorrow is.
ts["dow"]   = ts["date"].dt.dayofweek     # 0 = Monday ... 6 = Sunday
ts["month"] = ts["date"].dt.month

print()
print("Drug 1, once the history has built up:")
cols = ["date", "units", "lag_1", "lag_7", "roll7", "roll28", "dow"]
print(ts[ts["drug_id"] == 1][cols].iloc[28:35].round(1).to_string(index=False))


# --- PIECE 3: the answer - total units over the NEXT 7 days ---------------

by = ts.groupby("drug_id")          # rebuild it, the table has new columns now

# We want, for every day: the sum of the 7 days AFTER it.
# Pandas has no "look forward and add" function, so we use a small trick:
#
#   .rolling(7).sum()   on day i = the sum of days i-6 ... i    (looks BACK)
#   .shift(-7)          slides that value 7 rows UP
#
#   result on day i     = the sum of days i+1 ... i+7            (looks FORWARD)
#
# Tiny example. units = 10, 12, 8, 15, 20, 5, 9, 11 ...
#   day 1 target = 12+8+15+20+5+9+11 = 80   <- days 2 to 8. Correct.
ts["target"] = by["units"].transform(lambda s: s.rolling(7).sum().shift(-7))

# Rows at the very start have no history yet (roll28 is empty for 28 days).
# Rows at the very end have no future yet (target is empty for the last 7).
# Neither can be used. dropna() removes both.
before = len(ts)
data = ts.dropna().copy()

print()
print("Rows before dropping incomplete ones:", before)
print("Rows we can actually use            :", len(data))
print()
print("Drug 1 - what we know, and what we want to predict:")
cols = ["date", "units", "roll7", "roll28", "target"]
print(data[data["drug_id"] == 1][cols].head(6).round(1).to_string(index=False))


# --- PIECE 4: train it, and check it beats the obvious guess -------------

from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# drug_id is a label, not a quantity. Drug 12 is not "twice" drug 6.
# So one hot encode it, same as drug_name in file 4.
data = pd.get_dummies(data, columns=["drug_id"])

# Split by date again. Learn from the past, get tested on the future.
train = data[data["date"] <  "2026-01-01"]
test  = data[data["date"] >= "2026-01-01"]

DROP = ["date", "target"]           # date is not a feature, target is the answer

# XGBRegressor, not XGBClassifier. Same idea, but it predicts a NUMBER.
model = XGBRegressor(n_estimators=400, max_depth=4,
                     learning_rate=0.05, random_state=42)
model.fit(train.drop(columns=DROP), train["target"])

pred = model.predict(test.drop(columns=DROP))

# THE OBVIOUS GUESS: "next week will look like last week."
# roll7 is the average day last week, times 7 days. No model, no code.
# This is the forecasting version of the lazy model from file 4.
naive = test["roll7"] * 7

# MAE = mean absolute error. On average, how many units are we off by?
# Same unit as the thing we predict, so it is easy to explain.
mae_model = mean_absolute_error(test["target"], pred)
mae_naive = mean_absolute_error(test["target"], naive)

# MAPE = the same error as a PERCENTAGE of the real value.
# clip(lower=1) avoids dividing by zero on very quiet drugs.
mape_model = (abs(test["target"] - pred)  / test["target"].clip(lower=1)).mean() * 100
mape_naive = (abs(test["target"] - naive) / test["target"].clip(lower=1)).mean() * 100

print()
print("Average real demand over 7 days:", round(test["target"].mean()), "units")
print()
print(f"{'':<18}{'MAE (units)':>13}{'MAPE':>9}")
print(f"{'next = last week':<18}{mae_naive:>13.0f}{mape_naive:>8.1f}%")
print(f"{'our model':<18}{mae_model:>13.0f}{mape_model:>8.1f}%")


# --- PIECE 5: draw it -----------------------------------------------------

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("reports", exist_ok=True)

# Pick one drug so the chart is readable. drug_id 1 = Metformin.
# After get_dummies the column is called drug_id_1 and holds True / False.
one = test[test["drug_id_1"] == True].copy()
forecast = model.predict(one.drop(columns=DROP))

plt.figure(figsize=(11, 4))
plt.plot(one["date"], one["target"], label="actual demand (next 7 days)")
plt.plot(one["date"], forecast,      label="our forecast")
plt.title("Metformin - 7 day demand forecast on unseen 2026 data")
plt.ylabel("tablets")
plt.legend()
plt.tight_layout()
plt.savefig("reports/forecast.png", dpi=150)

print()
print("Chart saved -> reports/forecast.png")
