"""
STEP 4 - Train the model.

The hard work is already done. The table is ready. Now we teach a model
to look at the features and guess the answer.

We build this file one piece at a time.

Run it:   python 04_train_model.py
"""

import pandas as pd


# --- PIECE 1: split the data into TRAIN and TEST --------------------------

t = pd.read_csv("data/training_table.csv", parse_dates=["runout_dt"])

# Why split at all?
# If you teach a student using an exam paper, and then test them with the
# SAME paper, they score 100%. It proves nothing. They memorised it.
# So we hide some rows from the model, and use those to test it honestly.
#
# Why split by DATE and not randomly?
# In real life you learn from the past and predict the future. A random split
# would let the model study March and then be tested on February. Easy, and
# nothing like the real job.
#
# So: learn from everything before 2026. Get tested on 2026.

train = t[t["runout_dt"] <  "2026-01-01"]
test  = t[t["runout_dt"] >= "2026-01-01"]

print("Train rows (the past)  :", len(train))
print("Test rows  (the future):", len(test))
print()
print("Came back in time, train:", round(train["came_back"].mean() * 100, 1), "%")
print("Came back in time, test :", round(test["came_back"].mean() * 100, 1), "%")


# --- PIECE 2: get the data into the shape a model needs -------------------

# A model needs two things, kept separate:
#   X = the questions (all the features)
#   y = the answers   (the label)
# X and y are just the usual names people use. Nothing clever about them.

FEATURES = ["patient_age", "drug_name", "is_branded", "supply_days",
            "fill_number", "prev_gap", "was_late", "avg_gap", "pct_late"]
LABEL = "came_back"

X_train, y_train = train[FEATURES], train[LABEL]
X_test,  y_test  = test[FEATURES],  test[LABEL]

# Problem: drug_name is WORDS. A model can only do maths, so it needs numbers.
#
# get_dummies fixes this. It replaces one word column with many 0/1 columns:
#
#   drug_name                  drug_name_Metformin   drug_name_Losartan
#   Metformin        ->                1                     0
#   Losartan         ->                0                     1
#
# This is called ONE HOT ENCODING. One column per drug, a 1 in the right one.
X_train = pd.get_dummies(X_train, columns=["drug_name"])
X_test  = pd.get_dummies(X_test,  columns=["drug_name"])

# Safety step. If a drug never appears in the test rows, that column would be
# missing there, and the model would break. reindex forces the test data to
# have exactly the same columns, in the same order, filling gaps with 0.
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# True/False -> 1/0 so everything is a number.
X_train = X_train.astype(float)
X_test  = X_test.astype(float)

print()
print("X_train shape:", X_train.shape, " (rows, features)")
print("X_test  shape:", X_test.shape)
print()
print("The features the model will see:")
print(list(X_train.columns))


# --- PIECE 3: train the model ---------------------------------------------

from sklearn.linear_model import LogisticRegression

# We start with the simplest model that works: Logistic Regression.
# Not because it is the best, but because every later model has to BEAT
# something. A simple model you understand is worth more than a fancy one
# you cannot explain.
#
# What it does, in one sentence: it finds a weight for each feature, so that
# feature x weight, all added up, lines up with the answer as often as possible.

model = LogisticRegression(max_iter=1000)

# .fit() is the learning. This one line is the whole "training".
model.fit(X_train, y_train)

# .predict_proba gives a probability between 0 and 1 for each test row.
# [:, 1] takes the second column, which is the chance of the answer being 1.
guess = model.predict_proba(X_test)[:, 1]

print()
print("Model trained on", len(X_train), "rows.")
print()
print("A few guesses on rows the model has never seen:")

look = test[["pct_late", "prev_gap", "supply_days", "came_back"]].head(8).copy()
look["model_guess"] = guess[:8].round(2)
print(look.round(2).to_string(index=False))


# --- PIECE 4: is it any good? ---------------------------------------------

from sklearn.metrics import accuracy_score

# The model gives a probability. To count right and wrong we need a yes/no.
# Anything 0.5 or above -> we say "yes, they will come back".
answer = (guess >= 0.5).astype(int)

model_score = accuracy_score(y_test, answer)

# THE LAZY MODEL.
# Imagine someone who does no work at all and just says "yes" to everybody.
# Because 55.5% of test rows really are 1, that person is right 55.5% of the
# time. Our model MUST beat this, or it has learned nothing.
lazy_score = (y_test == 1).mean()

print()
print("Lazy guess (always yes):", round(lazy_score * 100, 1), "%")
print("Our model              :", round(model_score * 100, 1), "%")
print("Improvement            :", round((model_score - lazy_score) * 100, 1), "points")


# --- PIECE 5: what KIND of mistakes is it making? -------------------------

from sklearn.metrics import confusion_matrix, classification_report

# A confusion matrix splits every prediction into four boxes.
# For us, the patients we care about are the 0s - the ones who do NOT come back.
print()
print("Confusion matrix:")
print(confusion_matrix(y_test, answer))

print()
print(classification_report(y_test, answer, digits=3,
                            target_names=["0 = did NOT come back", "1 = came back"]))


# --- PIECE 6: choose the cut-off, then save the model ---------------------

# The 0.5 line was our choice, not a rule. Move it and everything changes.
# Below the cut-off we say "this patient is at risk - call them".
# Lower cut-off  = fewer calls, but we miss more people.
# Higher cut-off = we catch more people, but waste more calls.

y = y_test.values

print()
print(f"{'cutoff':>7}{'we call':>9}{'caught':>9}{'missed':>9}{'wasted':>9}")
for cut in [0.4, 0.5, 0.6, 0.7, 0.8]:
    at_risk = guess < cut                      # rows we would phone
    caught  = (at_risk & (y == 0)).sum()       # really at risk, we called them  GOOD
    wasted  = (at_risk & (y == 1)).sum()       # were fine, we called anyway     waste
    missed  = ((~at_risk) & (y == 0)).sum()    # really at risk, we did nothing  BAD
    print(f"{cut:>7}{at_risk.sum():>9}{caught:>9}{missed:>9}{wasted:>9}")

print()
print("Pick the cut-off from the business, not from the maths.")


# --- Save the model so other files can use it ------------------------------

import joblib
import os

os.makedirs("models/experiments", exist_ok=True)

# We save the model AND the exact column list. If the columns are in a
# different order later, the model gives nonsense without any error.
joblib.dump({"model": model, "columns": list(X_train.columns)},
            "models/experiments/logistic_baseline.pkl")

print()
print("Saved -> models/experiments/logistic_baseline.pkl")
