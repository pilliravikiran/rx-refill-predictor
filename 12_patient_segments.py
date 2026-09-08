"""
STEP 12 - Clustering: group patients without being told the answer.

Everything so far was SUPERVISED - we had the right answer for every row.
This is UNSUPERVISED. Nobody labels the patients. We hand the computer their
behaviour and it works out which ones are alike.

Why a pharmacy wants it: you cannot run one adherence programme for 3,000
people. You can run four, one per segment.

Run it:   python 12_patient_segments.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# --- PIECE 1: one row per PATIENT ----------------------------------------

# So far a row was one pickup. To group patients we need a row per patient:
# their whole behaviour summed up into a few numbers.

fills = pd.read_csv("data/fills.csv", parse_dates=["fill_dt"])
fills = fills.sort_values(["patient_id", "drug_id", "fill_dt"])

fills["gap"]        = fills.groupby(["patient_id", "drug_id"])["fill_dt"].diff().dt.days
fills["late_ratio"] = fills["gap"] / fills["supply_days"]        # 1.0 = on time, 2.0 = twice as slow
fills["late"]       = fills["gap"] > fills["supply_days"] + 14

patients = pd.read_csv("data/patients.csv", parse_dates=["birth_dt"])

profile = fills.groupby("patient_id").agg(
    pct_late       = ("late",       "mean"),   # share of refills that were late
    avg_late_ratio = ("late_ratio", "mean"),   # how slow they are on average
    n_fills        = ("fill_id",    "count"),  # how much they use the pharmacy
    n_drugs        = ("drug_id",    "nunique"),# how many medications
    avg_copay      = ("copay",      "mean"),   # what they pay
).dropna()

profile = profile.join(patients.set_index("patient_id")[["birth_dt"]])
profile["age"] = ((pd.Timestamp("2026-06-30") - profile["birth_dt"]).dt.days / 365).astype(int)
profile = profile.drop(columns=["birth_dt"])

print("Patients:", len(profile))
print("Features:", list(profile.columns))


# --- PIECE 2: put every feature on the same scale ------------------------

# K-Means measures DISTANCE between patients. n_fills goes up to 100,
# pct_late only to 1. Without scaling, n_fills would drown out everything
# else purely because its numbers are bigger. Scaling is not optional here.
X = StandardScaler().fit_transform(profile)


# --- PIECE 3: how many groups? -------------------------------------------

# You have to tell K-Means how many groups to find. It cannot work that out.
# Two ways to choose:
#
#   inertia    - how tightly packed the groups are. Always falls as k rises,
#                so look for the "elbow" where it stops falling quickly.
#   silhouette - are patients closer to their own group than to the next one?
#                Runs from -1 to +1. Higher is better. This one has a peak,
#                which makes it more useful than the elbow.
print()
print(f"{'k':>2}{'inertia':>11}{'silhouette':>13}")
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    score = silhouette_score(X, km.labels_, sample_size=2000, random_state=42)
    print(f"{k:>2}{km.inertia_:>11.0f}{score:>13.3f}")


# --- PIECE 4: build the segments -----------------------------------------

K = 4                                   # the silhouette peaked here
kmeans = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
profile["segment"] = kmeans.labels_

summary = profile.groupby("segment").agg(
    patients   = ("pct_late",       "size"),
    pct_late   = ("pct_late",       "mean"),
    late_ratio = ("avg_late_ratio", "mean"),
    fills      = ("n_fills",        "mean"),
    drugs      = ("n_drugs",        "mean"),
    copay      = ("avg_copay",      "mean"),
    age        = ("age",            "mean"),
).round(2)

print()
print("The four segments:")
print(summary.to_string())

# A cluster number means nothing on its own. YOU have to read the numbers
# and give it a name a pharmacist would understand.
print()
print("Read the table and name each segment yourself - that is the actual work.")


# --- PIECE 5: PCA, so we can draw it -------------------------------------

# We have 6 features. A screen has 2 axes. PCA squashes 6 into 2, keeping as
# much of the spread as it can. The 2 new axes are not any original feature -
# they are blends. You use them to LOOK, not to explain.
pca = PCA(n_components=2)
coords = pca.fit_transform(X)

kept = pca.explained_variance_ratio_
print()
print(f"2 components keep {kept.sum() * 100:.1f}% of the spread "
      f"({kept[0]*100:.1f}% + {kept[1]*100:.1f}%)")

os.makedirs("reports", exist_ok=True)
plt.figure(figsize=(8, 6))
for segment in range(K):
    mask = profile["segment"] == segment
    plt.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6,
                label=f"segment {segment} (n={mask.sum()})")
plt.title("Patient segments, squashed to 2 dimensions with PCA")
plt.xlabel("component 1")
plt.ylabel("component 2")
plt.legend()
plt.tight_layout()
plt.savefig("reports/segments.png", dpi=150)
print("Chart saved -> reports/segments.png")

profile.to_csv("data/patient_segments.csv")
print("Saved   -> data/patient_segments.csv")
