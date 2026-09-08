"""
STEP 15 - "Patients on this drug are usually also on that one."

Same idea as Amazon's "people who bought this also bought".

Pharmacy version: if a patient is on Metformin and NOT on a blood pressure
tablet, that might be a gap worth showing a pharmacist. Diabetes and blood
pressure travel together.

IMPORTANT: this only works because file 01 was changed to give patients a
CONDITION. When drugs were picked at random, every pair came out at 13-14%
and there was nothing to find. A recommender cannot invent a pattern that
is not in the data.

Run it:   python 15_recommend.py
"""

import itertools

import pandas as pd


# --- PIECE 1: who takes what ---------------------------------------------

fills = pd.read_csv("data/fills.csv")
drugs = pd.read_csv("data/drugs.csv")
name  = dict(zip(drugs["drug_id"], drugs["drug_name"]))
ids   = sorted(drugs["drug_id"])

# One row per patient per drug. drop_duplicates because a patient has many
# pickups of the same drug and we only care THAT they take it.
pairs   = fills[["patient_id", "drug_id"]].drop_duplicates()
baskets = pairs.groupby("patient_id")["drug_id"].apply(set)     # patient -> {drugs}

n_patients = len(baskets)
on_drug    = pairs.groupby("drug_id").size()                    # how many take each drug

print("Patients:", n_patients)


# --- PIECE 2: count every pair -------------------------------------------

# together.loc[a, b] = how many patients take BOTH a and b.
together = pd.DataFrame(0, index=ids, columns=ids)

for basket in baskets:
    # permutations gives every ordered pair from this patient's drugs:
    # {1,2,3} -> (1,2) (1,3) (2,1) (2,3) (3,1) (3,2)
    for a, b in itertools.permutations(basket, 2):
        together.loc[a, b] += 1


# --- PIECE 3: share, baseline, and LIFT ----------------------------------

# THE ONE IDEA THAT MATTERS HERE.
#
# "43% of Metformin patients also take Atorvastatin" sounds impressive.
# But 26% of ALL patients take Atorvastatin - it is just a popular drug.
# So the real signal is only 43 / 26 = 1.6x, not 43x.
#
# Without this correction a recommender just suggests the most popular item
# to everybody, which is useless. Lift is the fix.
#
#   lift = 1    no connection, it is just popular
#   lift = 2    twice as likely as normal
#   lift = 4    strongly connected

baseline = on_drug / n_patients        # share of ALL patients on each drug

rows = []
for a in ids:
    for b in ids:
        if a == b:
            continue
        share = together.loc[a, b] / on_drug[a]     # share of A patients also on B
        rows.append({
            "if_on":    name[a],
            "also_on":  name[b],
            "patients": together.loc[a, b],
            "share":    round(share, 3),
            "baseline": round(baseline[b], 3),
            "lift":     round(share / baseline[b], 2),
        })

table = pd.DataFrame(rows).sort_values("lift", ascending=False)

print()
print("Strongest connections:")
print(table.head(8).to_string(index=False))


# --- PIECE 4: recommend for one patient ----------------------------------

def recommend(patient_id, min_lift=1.5, min_share=0.20):
    """What is this patient NOT on, that patients like them usually are?"""
    theirs = baskets.loc[patient_id]

    suggestions = []
    for other in ids:
        if other in theirs:
            continue                       # they already have it

        # Take the strongest link from any drug they DO take.
        best_lift, best_share, best_from = 0, 0, None
        for mine in theirs:
            share = together.loc[mine, other] / on_drug[mine]
            lift  = share / baseline[other]
            if lift > best_lift:
                best_lift, best_share, best_from = lift, share, name[mine]

        if best_lift >= min_lift and best_share >= min_share:
            suggestions.append({
                "suggest": name[other],
                "because_of": best_from,
                "share": round(best_share, 2),
                "lift": round(best_lift, 2),
            })

    return pd.DataFrame(suggestions).sort_values("lift", ascending=False)


# Find a patient on Metformin who is NOT on a blood pressure tablet.
metformin = drugs.loc[drugs["drug_name"] == "Metformin", "drug_id"].iloc[0]
lisinopril = drugs.loc[drugs["drug_name"] == "Lisinopril", "drug_id"].iloc[0]

example = next(pid for pid, b in baskets.items()
               if metformin in b and lisinopril not in b)

print()
print(f"Patient {example} currently takes:",
      ", ".join(name[i] for i in sorted(baskets.loc[example])))
print()
print("Worth asking the pharmacist about:")
print(recommend(example).to_string(index=False))

print()
print("This is a PROMPT for a human, not an instruction. Nobody gets a drug")
print("because a computer noticed a pattern. A pharmacist decides.")
