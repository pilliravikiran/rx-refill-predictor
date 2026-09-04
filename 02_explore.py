"""
STEP 2 - Look at the data.

We build this file in small pieces. This is PIECE 1: just open the file
and see what is inside it.

Run it:   python 02_explore.py
"""

import pandas as pd                 # pandas is the library for working with tables


# --- PIECE 1: open data/fills.csv -----------------------------------------

# read_csv reads the file into a table (pandas calls a table a "DataFrame").
# parse_dates tells it: the fill_dt column is a real date, not just text.
fills = pd.read_csv("data/fills.csv", parse_dates=["fill_dt"])

print("Rows and columns:", fills.shape)     # .shape gives (how many rows, how many columns)

print()
print("First 5 rows:")
print(fills.head())                          # .head() shows the first 5 rows


# --- PIECE 2: look at ONE patient -----------------------------------------

# 105,549 rows is too many to look at. Zoom in on one person and one drug.
# Inside the square brackets we write a condition. Pandas keeps only the rows
# where the condition is True.
#   ==  means "is equal to"
#   &   means "and"        (each condition needs its own brackets)
one = fills[(fills["patient_id"] == 138) & (fills["drug_id"] == 12)]

# Put them oldest first. Without this the dates could be in any order,
# and the day-counting we do next would be nonsense.
one = one.sort_values("fill_dt")

print()
print("Patient 138, drug 12 - every time they picked it up:")

# Inside the brackets, a LIST of column names picks just those columns.
print(one[["fill_dt", "supply_days", "disp_qty"]].to_string(index=False))


# --- PIECE 3: count the days between pickups -------------------------------

# .copy() makes a real, separate table. Without it pandas warns you, because
# `one` is only a VIEW of `fills` and adding a column to a view is ambiguous.
one = one.copy()

# .diff() subtracts the previous row from the current row.
# On dates that gives a length of time, and .dt.days turns it into a number.
one["gap"] = one["fill_dt"].diff().dt.days

print()
print("Patient 138 - days between pickups:")
print(one[["fill_dt", "supply_days", "gap"]].to_string(index=False))

# Now the same thing for a different patient, so you can see the difference.
two = fills[(fills["patient_id"] == 19) & (fills["drug_id"] == 5)].sort_values("fill_dt").copy()
two["gap"] = two["fill_dt"].diff().dt.days

print()
print("Patient 19 - days between pickups:")
print(two[["fill_dt", "supply_days", "gap"]].head(8).to_string(index=False))


# --- PIECE 4: do it for all 3,000 patients at once -------------------------

# Sort first. .diff() trusts the row order blindly.
f = fills.sort_values(["patient_id", "drug_id", "fill_dt"]).copy()

# groupby() splits the table into little groups - one per patient per drug -
# runs .diff() inside each group, then puts the answers back together.
# WITHOUT groupby, .diff() would subtract the last row of one patient from
# the first row of the next patient, and invent a gap that never happened.
f["gap"] = f.groupby(["patient_id", "drug_id"])["fill_dt"].diff().dt.days

print()
print("Days between refills, across the whole pharmacy:")
print(f["gap"].describe().round(1).to_string())


# --- PIECE 5: was this refill late? ---------------------------------------

# The first pickup of each course has gap = NaN. We cannot say whether it was
# late, because there is nothing to compare it to. notna() keeps only the rows
# where gap has a real value.
g = f[f["gap"].notna()].copy()

# LATE = they took longer than their supply, plus a 14 day grace period.
# 30 day supply -> anything over 44 days is late.
# The result is True or False for every single row.
g["late"] = g["gap"] > g["supply_days"] + 14

print()
print("Every refill, late or not:")
print(g["late"].value_counts().to_string())
print("Share late:", round(g["late"].mean() * 100, 1), "%")

# Now per patient. mean() on True/False gives the SHARE that are True,
# because True counts as 1 and False counts as 0.
late_rate = g.groupby("patient_id")["late"].mean()

print()
print("Each patient's personal late rate:")
print(late_rate.describe().round(2).to_string())

print()
print("Patients never late      :", round((late_rate == 0).mean() * 100, 1), "%")
print("Patients late over half  :", round((late_rate > 0.5).mean() * 100, 1), "%")


# --- PIECE 6: draw it ------------------------------------------------------

# Normally imports go at the very top of a file. They are here only because we
# built this file piece by piece - move them up once you are done.
import os
import matplotlib
matplotlib.use("Agg")            # save charts to a file instead of opening a window
import matplotlib.pyplot as plt

os.makedirs("reports", exist_ok=True)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))     # one picture, two charts side by side

# Left chart: how long people take between refills.
# .loc[rows, column] picks rows AND a column. We drop the long tail over 200
# days so the interesting part is not squashed into the left edge.
g.loc[g["gap"] < 200, "gap"].hist(bins=60, ax=ax[0])
ax[0].set_title("Days between refills")
ax[0].set_xlabel("days")

# Right chart: each patient's late rate, from piece 5.
late_rate.hist(bins=30, ax=ax[1])
ax[1].set_title("Each patient's late rate")
ax[1].set_xlabel("share of their refills that were late")

plt.tight_layout()                                # stop the labels overlapping
plt.savefig("reports/eda.png", dpi=150)           # write the picture to disk
print()
print("Chart saved -> reports/eda.png")
