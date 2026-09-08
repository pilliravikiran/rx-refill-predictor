"""
STEP 11 - Drift monitoring: has the world moved away from what we trained on?

Models rot. Not because the code breaks, but because the world changes.
Nothing crashes. The API keeps returning 200. The answers are just wrong.

PSI (Population Stability Index) compares the SHAPE of new data against the
shape of the training data, feature by feature.

Run it:   python 11_monitor_drift.py
"""

import numpy as np
import pandas as pd


# --- PIECE 1: the PSI calculation ----------------------------------------

def psi(expected, actual, bins=10):
    """
    How different is `actual` from `expected`?

    1. Cut the TRAINING data into 10 equal-size buckets.
    2. See what share of the NEW data lands in each of those same buckets.
    3. If the shares match, the answer is near 0. If the shape moved, it grows.

    under 0.10   fine
    0.10 - 0.25  watch it
    over 0.25    retrain
    """
    # np.quantile finds the cut points that split training data into equal groups.
    cuts = np.quantile(expected, np.linspace(0, 1, bins + 1))
    cuts = np.unique(cuts)                  # a feature like was_late may have few values
    cuts[0], cuts[-1] = -np.inf, np.inf     # open the ends so nothing falls outside

    share_expected = np.histogram(expected, bins=cuts)[0] / len(expected)
    share_actual   = np.histogram(actual,   bins=cuts)[0] / len(actual)

    # Never divide by zero. A bucket with nothing in it gets a tiny number instead.
    share_expected = np.clip(share_expected, 1e-6, None)
    share_actual   = np.clip(share_actual,   1e-6, None)

    return float(np.sum((share_actual - share_expected)
                        * np.log(share_actual / share_expected)))


# --- PIECE 2: compare training data against live data --------------------

t = pd.read_csv("data/training_table.csv", parse_dates=["runout_dt"])

training = t[t["runout_dt"] <  "2026-01-01"]   # what the model learned from
live     = t[t["runout_dt"] >= "2026-01-01"]   # what it is seeing now

NUMERIC = ["patient_age", "supply_days", "fill_number",
           "prev_gap", "avg_gap", "was_late", "pct_late"]

report = pd.DataFrame({
    "feature": NUMERIC,
    "psi": [round(psi(training[c], live[c]), 3) for c in NUMERIC],
})
report["verdict"] = pd.cut(report["psi"], [-1, 0.1, 0.25, 99],
                           labels=["stable", "watch", "RETRAIN"])

print("Drift report")
print(report.sort_values("psi", ascending=False).to_string(index=False))


# --- PIECE 3: do NOT blindly obey your own alert -------------------------

# fill_number will always look terrible. Later in time means patients have
# collected more times. It drifts BY CONSTRUCTION, not because anything is
# wrong. Alerting on it would page someone every single month for no reason.
#
# So split the features into two groups and only alert on the second.
DRIFTS_BY_DESIGN = ["fill_number"]      # grows with time no matter what

behavioural = report[~report["feature"].isin(DRIFTS_BY_DESIGN)]
alerting    = behavioural[behavioural["psi"] > 0.25]

print()
print("Ignoring features that drift by design:", DRIFTS_BY_DESIGN)

if len(alerting):
    print()
    print("REAL DRIFT - these are behaviour changes, not artefacts:")
    print(alerting.to_string(index=False))
    print()
    print("Action: retrain on recent data, then compare the new run in MLflow")
    print("against the current registered model before promoting it.")
else:
    print()
    print("No material drift. The model is safe to keep serving.")
