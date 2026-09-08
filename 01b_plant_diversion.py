"""
STEP 1b - Plant some fake drug diversion, so we can test whether we catch it.

We are about to build a detector for suspicious dispensing. But our data has
nothing suspicious in it, so there would be no way to know if the detector
works.

So we do the same trick as the hidden adherence score in file 01: plant
something, keep a list of what we planted, then check later whether we
found it.

This writes a SEPARATE copy. data/fills.csv is left untouched, so files
03 to 12 keep giving the same numbers as before.

Run it:   python 01b_plant_diversion.py
"""

from datetime import timedelta

import numpy as np
import pandas as pd

rng = np.random.default_rng(99)

fills = pd.read_csv("data/fills.csv", parse_dates=["fill_dt"])
drugs = pd.read_csv("data/drugs.csv")

# drug_class 0 = not controlled. Anything above 0 is a controlled substance.
# Only Gabapentin is controlled in our list.
controlled = drugs[drugs["drug_class"] > 0]["drug_id"].tolist()
print("Controlled drug ids:", controlled)

# Pick 20 patients who already take one, and have enough history to look normal.
counts     = fills[fills["drug_id"].isin(controlled)].groupby("patient_id").size()
candidates = counts[counts >= 8].index.tolist()
diverters  = list(rng.choice(candidates, size=20, replace=False))

# What diversion looks like: coming back for a 30 day supply after ~12 days,
# over and over. They are getting far more medication than prescribed.
new_rows = []
next_id  = fills["fill_id"].max() + 1

for patient_id in diverters:
    history = fills[(fills["patient_id"] == patient_id)
                    & (fills["drug_id"].isin(controlled))].sort_values("fill_dt")
    first = history.iloc[0]
    day   = first["fill_dt"]

    for _ in range(14):                                  # a run of 14 early pickups
        day = day + timedelta(days=int(rng.integers(8, 16)))
        new_rows.append({
            "fill_id": next_id, "presc_id": first["presc_id"],
            "patient_id": patient_id, "drug_id": first["drug_id"],
            "refill_num": 0, "fill_dt": day.date().isoformat(),
            "supply_days": first["supply_days"], "disp_qty": first["disp_qty"],
            "daw_id": 0, "price": first["price"], "copay": first["copay"],
            "insurance_id": first["insurance_id"], "status": "Filled",
        })
        next_id += 1

extra = pd.DataFrame(new_rows)
out   = pd.concat([fills, extra], ignore_index=True)
out["fill_dt"] = pd.to_datetime(out["fill_dt"])
out = out.sort_values(["patient_id", "presc_id", "fill_dt"])

out.to_csv("data/fills_with_diversion.csv", index=False)
pd.DataFrame({"patient_id": diverters}).to_csv("data/known_diverters.csv", index=False)

print("Planted diversion for", len(diverters), "patients")
print("Extra pickups added :", len(extra))
print("Total pickups now   :", len(out))
print()
print("Saved -> data/fills_with_diversion.csv")
print("Saved -> data/known_diverters.csv   (the answer key)")
