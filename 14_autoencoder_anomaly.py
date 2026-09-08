"""
STEP 14 - The same job again, with a neural network. And it loses.

An AUTOENCODER is a squeeze-and-rebuild network:

    6 numbers in  ->  squeeze to 2  ->  rebuild back to 6

It is forced through a narrow middle, so it cannot memorise. It has to learn
what a TYPICAL patient looks like.

Then: show it a patient and ask it to rebuild them.
  rebuilt well  -> they look typical
  rebuilt badly -> they do not look like anybody. Odd.

Like someone who has only ever seen normal prescriptions being asked to
redraw one from memory. Normal ones they get right. A strange one they mangle.

Run:  python 14_autoencoder_anomaly.py     (needs: pip install tensorflow-cpu)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # hide TensorFlow's startup noise

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

tf.random.set_seed(42)
np.random.seed(42)


# --- PIECE 1: exactly the same features as file 13 ------------------------

fills = pd.read_csv("data/fills_with_diversion.csv", parse_dates=["fill_dt"])
drugs = pd.read_csv("data/drugs.csv")
known = set(pd.read_csv("data/known_diverters.csv")["patient_id"])

controlled = drugs[drugs["drug_class"] > 0]["drug_id"].tolist()
c = fills[fills["drug_id"].isin(controlled)].sort_values(
    ["patient_id", "drug_id", "fill_dt"]).copy()

c["gap"]        = c.groupby(["patient_id", "drug_id"])["fill_dt"].diff().dt.days
c["early_days"] = (c["supply_days"] - c["gap"]).clip(lower=0)
c["is_early"]   = c["gap"] < c["supply_days"] * 0.75

profile = c.groupby("patient_id").agg(
    fills          = ("fill_id",    "count"),
    avg_gap        = ("gap",        "mean"),
    min_gap        = ("gap",        "min"),
    pct_early      = ("is_early",   "mean"),
    avg_early_days = ("early_days", "mean"),
    total_qty      = ("disp_qty",   "sum"),
).dropna()

X = StandardScaler().fit_transform(profile).astype("float32")
n_features = X.shape[1]


# --- PIECE 2: build the squeeze-and-rebuild network ----------------------

# Each Dense layer is a set of numbers the network learns. Read the sizes
# and you can see the squeeze:  6 -> 4 -> 2 -> 4 -> 6
#
# The 2 in the middle is the bottleneck. Everything about a patient has to
# fit through it, so the network is forced to keep only what matters.
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(n_features,)),
    tf.keras.layers.Dense(4, activation="relu"),   # squeeze
    tf.keras.layers.Dense(2, activation="relu"),   # the bottleneck
    tf.keras.layers.Dense(4, activation="relu"),   # rebuild
    tf.keras.layers.Dense(n_features),             # back to 6
])

# loss="mse" = how wrong the rebuild is. The network trains itself to
# make that small.
model.compile(optimizer="adam", loss="mse")

# Note the fit(X, X). Input and answer are THE SAME THING.
# We are not predicting anything. We are asking it to copy - through a
# narrow gap, which is what forces it to learn.
history = model.fit(X, X, epochs=60, batch_size=32, verbose=0, validation_split=0.1)

print(f"rebuild error, training: {history.history['loss'][-1]:.4f}")
print(f"rebuild error, held out: {history.history['val_loss'][-1]:.4f}")


# --- PIECE 3: how badly was each patient rebuilt? ------------------------

rebuilt = model.predict(X, verbose=0)
profile["error"] = np.mean((X - rebuilt) ** 2, axis=1)     # big error = odd

print()
print(f"{'review budget':>14}{'flagged':>9}{'caught':>8}{'of 20':>7}{'false alarms':>14}")
for budget in [0.01, 0.02, 0.03, 0.05, 0.08]:
    k       = int(len(profile) * budget)
    flagged = set(profile.nlargest(k, "error").index)
    print(f"{budget*100:>13.0f}%{len(flagged):>9}{len(flagged & known):>8}"
          f"{len(known):>7}{len(flagged - known):>14}")

print()
print("The 6 it found strangest:")
print(profile.nlargest(6, "error").round(2).to_string())

print()
print("Compare these numbers with file 13. Isolation Forest wins, clearly.")
print("Read the section in the README before assuming the neural network")
print("must be better because it is a neural network.")
