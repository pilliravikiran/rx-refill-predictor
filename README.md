# Pharmacy Refill Risk Prediction

Predicts whether a patient will pick up their next prescription refill on time,
so a pharmacy can reach out to the ones who won't — before they run out of medication.

Built end to end: synthetic data → exploration → feature engineering → model → REST API.

---

## The problem

A pharmacy dispenses a 30 day supply. A patient taking their medication properly comes
back in about 30 days. A patient who comes back after 60 days spent a month with no medicine.

The pharmacy wants to call those patients before it happens. With thousands of patients
they cannot call everyone, so they need to know **who** to call.

**Prediction question:** for each fill, on the day the medication runs out
(`fill_date + days_supply`), will the patient return within **14 days**?

- `1` = returned in time
- `0` = returned late, or never returned

Never returning counts as `0`, not as missing data. Those are the patients that matter most.

---

## The data

**Synthetic.** Real dispensing records are protected health information.

The field shapes mirror a production pharmacy management system — `Person`, `Patient`,
`Prescriber`, `Drug`, `Insurance`, `Prescription`, `PrescReFill`. Identifiers used
(NDC-11, NPI, DEA, DAW, BIN/PCN) are public NCPDP and FDA standards.

| File | Rows | What it holds |
|---|---:|---|
| `data/drugs.csv` | 12 | NDC, strength, form, DEA class, AWP, pack size |
| `data/insurances.csv` | 6 | BIN/PCN, brand and generic copay, cash flag |
| `data/prescribers.csv` | 60 | NPI, DEA number, specialty |
| `data/patients.csv` | 3,000 | DOB, gender, address, language, primary insurance |
| `data/prescriptions.csv` | 17,096 | Rx header — drug, qty, days, refills authorized |
| `data/fills.csv` | 105,549 | one row per pickup — the fact table |

**How the signal was planted.** Each patient is assigned a hidden `adherence` score
between 0 and 1 that drives how long they wait before returning. That score is never
written to any file. The model has to recover it from the dates alone — which makes it
possible to verify the pipeline actually works.

Prescriptions renew when refills run out, producing a new Rx number and resetting the
refill counter, the same as a real pharmacy. Verified: no row has
`refill_num > refills_authorized`.

---

## Results

Test set is every fill running out in 2026. Training set is everything before it.

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Always guess "will return" | 0.555 | — |
| Logistic Regression | 0.657 | 0.718 |
| XGBoost | **0.671** | 0.718 |

**XGBoost beat logistic regression by 1.4 points of accuracy and not at all on AUC.**

That is a real finding, not a disappointment. `pct_late` — how often the patient has been
late before — carries 63% of the model's total feature importance, and its relationship to
the outcome is close to linear. A straight line already captures it, so trees have little
left to add.

The ceiling here is set by the data, not the algorithm. Improving it needs more
information (copay amount, auto-refill enrolment, distance to pharmacy, total medication
count), not a bigger model.

### Error breakdown, XGBoost at the 0.5 cut-off

|  | Predicted: won't return | Predicted: will return |
|---|---:|---:|
| **Actually didn't return** | 3,137 | 2,072 |
| **Actually returned** | 1,781 | 4,724 |

Precision on the at-risk class 0.64, recall 0.60.

### The cut-off is a business decision

The 0.5 threshold is a choice, not a rule. Moving it changes the whole operation:

| Cut-off | Calls made | At-risk caught | At-risk missed | Wasted calls |
|---:|---:|---:|---:|---:|
| 0.4 | 1,424 | 1,031 | 4,178 | 393 |
| 0.5 | 3,182 | 2,185 | 3,024 | 997 |
| 0.6 | 5,175 | 3,246 | 1,963 | 1,929 |
| 0.7 | 7,395 | 4,157 | 1,052 | 3,238 |
| 0.8 | 9,895 | 4,898 | 311 | 4,997 |

*(from the logistic model; the shape is the same for XGBoost)*

At 0.7 the pharmacy catches 80% of at-risk patients for about 7,400 calls. At 0.5 it makes
half the calls and catches 42%. Which is correct depends on call capacity and on how much
a missed refill costs — not on anything in the model.

---

## Avoiding leakage

Three deliberate decisions, each one a way this project could have silently produced a
useless model that scored well:

**Time-based split.** Train on everything before 2026-01-01, test on 2026. A random split
would let the model train on March and be tested on February.

**Expanding-window features.** `avg_gap` and `pct_late` use `.expanding().mean()`, so each
row sees only its own past. A plain `.mean()` over each group would leak future rows into
past ones.

**Censoring.** Any fill whose run-out date falls in the final 14 days of the data is
dropped — 3,603 rows. Those patients still had time to return; the pipeline had marked
them `0` because no next row existed. Keeping them would teach the model that recent
prescriptions never get refilled.

---

## Feature importance

![feature importance](reports/feature_importance.png)

| Feature | Importance |
|---|---:|
| `pct_late` | 0.627 |
| `fill_number` | 0.103 |
| `was_late` | 0.041 |
| `avg_gap` | 0.021 |
| `prev_gap` | 0.019 |

`pct_late` dominating is the confirmation that the pipeline is correct end to end — it is
the visible trace of the hidden adherence score from the data generator.

This also matters operationally. In healthcare "the model said so" is not an acceptable
answer. "This patient has been late on 8 of their last 10 refills" is one a pharmacist
can check and disagree with.

---

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python 01_generate_data.py         # writes data/
python 02_explore.py               # writes reports/eda.png
python 03_build_training_table.py  # writes data/training_table.csv
python 04_train_model.py           # baseline + threshold analysis
python 05_better_model.py          # XGBoost + feature importance

uvicorn app:app --reload           # http://127.0.0.1:8000/docs
```

### API

`GET /health` — liveness check.

`POST /predict`

```json
{
  "patient_age": 71, "drug_name": "Metformin", "is_branded": false,
  "supply_days": 90, "fill_number": 9, "prev_gap": 92,
  "was_late": 0, "avg_gap": 91, "pct_late": 0.05
}
```

```json
{ "refill_probability": 0.781, "risk": "low", "action": "no action" }
```

Change `pct_late` to `0.85`, `prev_gap` to `150`, `was_late` to `1`:

```json
{ "refill_probability": 0.178, "risk": "high", "action": "pharmacist call" }
```

The endpoint returns a risk band and an action, not a bare probability — a pharmacy can
act on "pharmacist call", not on `0.178`. The band cut-offs come from the trade-off table
above.

The saved model bundle stores the trained model **and** the exact column order it was
trained on. One-hot encoding a single incoming patient produces one drug column where the
model expects twelve; `reindex` against that stored order fills the rest. Without it the
model reads the wrong value from the wrong column and returns confident nonsense with no
error.

---

## Files

```
01_generate_data.py          synthetic pharmacy data -> 6 CSVs
02_explore.py                EDA: is this problem solvable at all?
03_build_training_table.py   label + leak-free features -> training_table.csv
04_train_model.py            logistic baseline, error analysis, threshold trade-off
05_better_model.py           XGBoost, model comparison, feature importance
app.py                       FastAPI service
```

`data/`, `models/` and `reports/` are gitignored — every one of them is rebuilt by
running the scripts in order.

---

## What I'd do next

- **Calibration.** Check whether a predicted 0.7 really does return 70% of the time.
  A probability that drives a phone-call budget should be calibrated, not just ranked well.
- **More features.** Copay amount, auto-refill enrolment, distance to pharmacy, total
  active medication count. The 1.4-point gap between models says the limit is information,
  not algorithm.
- **Drift monitoring.** Track feature distributions against training. Note that
  `fill_number` drifts upward by construction as time passes, so it needs excluding from
  any naive alert.
- **Scheduled retraining** with the run and its metrics recorded per model version.

---

*Portfolio project. The data is synthetic and generated by `01_generate_data.py`;
it is not from any production system.*
