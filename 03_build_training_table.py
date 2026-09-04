"""
STEP 3 - Build the table the model will learn from.

In file 2 we looked at the past.
Now we need two things for every row:

  1. What we KNOW today          (these are called FEATURES)
  2. What HAPPENS next           (this is called the LABEL, the answer)

The model looks at 1 and tries to guess 2.

We build this file one piece at a time.

Run it:   python 03_build_training_table.py
"""

import pandas as pd


# --- PIECE 1: when did they come back next? --------------------------------

fills = pd.read_csv("data/fills.csv", parse_dates=["fill_dt"])
fills = fills.sort_values(["patient_id", "drug_id", "fill_dt"])

# In file 2 we used .diff(), which looks BACKWARD at the row above.
# Now we need to look FORWARD, at the row below.
# shift(-1) does that. It copies the next row's value up into this row.
fills["next_fill_dt"] = fills.groupby(["patient_id", "drug_id"])["fill_dt"].shift(-1)

one = fills[(fills["patient_id"] == 138) & (fills["drug_id"] == 12)]

print("Patient 138 - this pickup, and the next one:")
print(one[["fill_dt", "supply_days", "next_fill_dt"]].head(8).to_string(index=False))


# --- PIECE 2: the answer we want the model to learn (the LABEL) ------------

# The day their pills run out = fill date + days supply.
# pd.to_timedelta turns the number 30 into "30 days" so we can add it to a date.
fills["runout_dt"] = fills["fill_dt"] + pd.to_timedelta(fills["supply_days"], unit="D")

# How many days after running out did they come back?
#   0 or less = they came back early, before the pills ran out
#   +5        = they were out of medicine for 5 days
fills["days_late"] = (fills["next_fill_dt"] - fills["runout_dt"]).dt.days

# THE ANSWER. 1 = came back in time. 0 = did not.
# We allow a 14 day grace period.
# .astype(int) turns True into 1 and False into 0. Models want numbers.
# If they never came back at all, days_late is empty, the test is False, so 0.
# That is correct - never coming back is the worst case, not missing data.
fills["came_back"] = (fills["days_late"] <= 14).astype(int)

show = ["fill_dt", "supply_days", "runout_dt", "next_fill_dt", "days_late", "came_back"]

print()
print("Patient 138 - a reliable patient:")
print(fills[(fills["patient_id"] == 138) & (fills["drug_id"] == 12)][show].head(6).to_string(index=False))

print()
print("Patient 19 - an unreliable patient:")
print(fills[(fills["patient_id"] == 19) & (fills["drug_id"] == 5)][show].head(6).to_string(index=False))

print()
print("Out of all rows, how many came back in time?")
print(fills["came_back"].value_counts().to_string())


# --- PIECE 3: what we knew BEFORE the answer (the FEATURES) ---------------

# THE GOLDEN RULE
# A feature must be something we could know ON the day we make the guess.
# If a column can only be filled in AFTER the patient comes back, it cannot
# be a feature. Using one by mistake is called LEAKAGE. The model scores
# brilliantly in testing and is useless in real life.
#
# So: next_fill_dt, days_late and came_back can NEVER be features.
# They are the answer. We only use them to check if the guess was right.

by = fills.groupby(["patient_id", "drug_id"])

# Feature 1: how many days since their PREVIOUS pickup.
# This is .diff() again - looking backward, which is safe.
fills["prev_gap"] = by["fill_dt"].diff().dt.days

# Feature 2: is this their 1st pickup, 2nd, 3rd...?
# cumcount() counts rows inside each group, starting at 0. We add 1.
fills["fill_number"] = by.cumcount() + 1

# The very first pickup has no previous one, so prev_gap is empty.
# We fill it with their normal supply length - a sensible "no news" value.
fills["prev_gap"] = fills["prev_gap"].fillna(fills["supply_days"])

show = ["fill_number", "fill_dt", "supply_days", "prev_gap", "came_back"]

print()
print("Patient 19 - what we know, and what happened:")
print(fills[(fills["patient_id"] == 19) & (fills["drug_id"] == 5)][show].head(6).to_string(index=False))


# --- PIECE 4: their whole history so far ----------------------------------

# prev_gap only remembers the LAST pickup. One late month might be a holiday.
# Being late 8 times out of 10 is a habit. We want the habit.

# Was the PREVIOUS pickup late? 1 = yes, 0 = no.
fills["was_late"] = (fills["prev_gap"] > fills["supply_days"] + 14).astype(int)

by = fills.groupby(["patient_id", "drug_id"])

# expanding() means "everything from the start up to and including this row,
# and nothing after it". So each row sees only its own past. That keeps it legal.
#
#   prev_gap:  30   71   54   60
#   avg_gap:   30   50   51   53     <- average grows one row at a time
#
# If we used a plain .mean() instead, every row would get the average of the
# WHOLE history, including future rows. That is leakage. Same word, again.
fills["avg_gap"]  = by["prev_gap"].transform(lambda s: s.expanding().mean())
fills["pct_late"] = by["was_late"].transform(lambda s: s.expanding().mean())

show = ["fill_number", "prev_gap", "was_late", "avg_gap", "pct_late", "came_back"]

print()
print("Patient 19 - history building up row by row:")
print(fills[(fills["patient_id"] == 19) & (fills["drug_id"] == 5)][show].head(6).round(2).to_string(index=False))

print()
print("Patient 138 - the reliable one:")
print(fills[(fills["patient_id"] == 138) & (fills["drug_id"] == 12)][show].head(6).round(2).to_string(index=False))


# --- PIECE 5: throw away rows we are not allowed to judge ------------------

# Our data stops on a certain day. Think about a pickup that runs out 3 days
# before that day. The patient still has 11 days of grace left to come back.
# We simply do not know the answer yet.
#
# But our code already marked it came_back = 0, because there is no next row.
# That is a LIE. If we keep it, we teach the model "recent = never comes back".
#
# The fix: drop any row whose runout day is inside the last 14 days.
# This is called CENSORING.

LAST_DAY = fills["fill_dt"].max()
CUT_OFF  = LAST_DAY - pd.Timedelta(days=14)

print()
print("Last day in our data :", LAST_DAY.date())
print("Cut off at           :", CUT_OFF.date())

before = len(fills)
fills = fills[fills["runout_dt"] <= CUT_OFF]

print("Rows before cut      :", before)
print("Rows after cut       :", len(fills))
print("Dropped              :", before - len(fills))


# --- PIECE 6: add patient and drug details, then save ---------------------

# So far every feature came from the fill dates. Let us also give the model
# the patient's age and which drug it is. Those come from the other files.

patients = pd.read_csv("data/patients.csv", parse_dates=["birth_dt"])
drugs    = pd.read_csv("data/drugs.csv")

# merge() is a JOIN. on= is the matching column.
# We only pull the few columns we need, not the whole table.
t = fills.merge(patients[["patient_id", "birth_dt"]], on="patient_id")
t = t.merge(drugs[["drug_id", "drug_name", "is_branded"]], on="drug_id")

# Age on the day of the pickup.
t["patient_age"] = ((t["fill_dt"] - t["birth_dt"]).dt.days / 365).astype(int)

# Now choose exactly what the model is allowed to see.
FEATURES = [
    "patient_age",     # who they are
    "drug_name",       # what they take
    "is_branded",
    "supply_days",     # how long the pills last
    "fill_number",     # how far into their treatment they are
    "prev_gap",        # last time, how long did they take
    "was_late",        # was last time late
    "avg_gap",         # their average so far
    "pct_late",        # how often they are late
]

LABEL = "came_back"    # the answer

# runout_dt is not a feature. We keep it only to split the data by date later.
t = t[FEATURES + ["runout_dt", LABEL]]

t.to_csv("data/training_table.csv", index=False)

print()
print("Saved -> data/training_table.csv")
print("Rows:", len(t), " Columns:", len(t.columns))
print("Came back in time:", round(t[LABEL].mean() * 100, 1), "%")
print()
print(t.head(5).round(2).to_string(index=False))
