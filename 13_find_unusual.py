"""
STEP 13 - Find the odd ones out.

File 12 asked "which pile does this patient belong to?"
This asks "which patients do not look like anybody else?"

For a pharmacy that means controlled substances. Somebody collecting a
30 day supply every 12 days is either in serious trouble or selling it.
Catching that is what PDMP reporting exists for.

We planted 20 fake diverters in file 01b. Now we see how many we catch.

Run:  python 01b_plant_diversion.py   (first, once)
      python 13_find_unusual.py
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# --- PIECE 1: look only at controlled drugs ------------------------------

fills = pd.read_csv("data/fills_with_diversion.csv", parse_dates=["fill_dt"])
drugs = pd.read_csv("data/drugs.csv")
known = set(pd.read_csv("data/known_diverters.csv")["patient_id"])   # the answer key

controlled = drugs[drugs["drug_class"] > 0]["drug_id"].tolist()

c = fills[fills["drug_id"].isin(controlled)].copy()
c = c.sort_values(["patient_id", "drug_id", "fill_dt"])


# --- PIECE 2: build features that describe "coming back too soon" --------

c["gap"] = c.groupby(["patient_id", "drug_id"])["fill_dt"].diff().dt.days

# How many days of medication they still had left when they came back.
# 30 day supply, back after 12 days -> 18 days early.
c["early_days"] = (c["supply_days"] - c["gap"]).clip(lower=0)

# "Early" = came back before three quarters of the supply was used.
c["is_early"] = c["gap"] < c["supply_days"] * 0.75

profile = c.groupby("patient_id").agg(
    fills          = ("fill_id",    "count"),   # how many pickups
    avg_gap        = ("gap",        "mean"),    # typical days between them
    min_gap        = ("gap",        "min"),     # the shortest gap ever
    pct_early      = ("is_early",   "mean"),    # share of pickups that were early
    avg_early_days = ("early_days", "mean"),    # how early, on average
    total_qty      = ("disp_qty",   "sum"),     # total medication collected
).dropna()

print("Patients on a controlled drug:", len(profile))

X = StandardScaler().fit_transform(profile)     # same scaling reason as file 12


# --- PIECE 3: how the detector works -------------------------------------

# Isolation Forest. The idea is simple and rather clever.
#
# It repeatedly splits the patients at random. An ordinary patient sits in
# the middle of the crowd, so it takes many splits to separate them from
# everyone else. An odd one sits out on its own and gets separated after
# only a few splits.
#
# "How few splits did it take to isolate this patient?" is the score.
# Few splits = unusual.
#
# contamination = what share of patients you expect to be odd. It is really
# a REVIEW BUDGET: how many cases can a pharmacist actually look at?

print()
print(f"{'review budget':>14}{'flagged':>9}{'caught':>8}{'of 20':>7}{'false alarms':>14}")

for budget in [0.01, 0.02, 0.03, 0.05, 0.08]:
    detector = IsolationForest(contamination=budget, random_state=42).fit(X)
    flagged  = set(profile.index[detector.predict(X) == -1])
    caught   = len(flagged & known)
    print(f"{budget*100:>13.0f}%{len(flagged):>9}{caught:>8}{len(known):>7}"
          f"{len(flagged - known):>14}")


# --- PIECE 4: the settings we would actually use -------------------------

detector = IsolationForest(contamination=0.05, random_state=42).fit(X)

# decision_function gives a score. The LOWER it is, the odder the patient.
profile["oddness"] = detector.decision_function(X)
profile["flagged"] = detector.predict(X) == -1
profile["was_planted"] = profile.index.isin(known)

flagged = profile[profile["flagged"]].sort_values("oddness")

print()
print("The 8 oddest patients:")
print(flagged.head(8).round(2).to_string())

profile.sort_values("oddness").to_csv("data/controlled_review_queue.csv")
print()
print("Saved -> data/controlled_review_queue.csv  (a work queue, oddest first)")
