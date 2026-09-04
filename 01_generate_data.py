"""
STEP 1 - Make the data.

We are building a model that answers: "will this patient come back for their
refill on time?"  Real pharmacy records are protected health information, so we
invent our own - but we shape them like the real ePrimeRx tables so nothing
here is a toy.

Table we copy the shape of      ->  file we write
--------------------------------------------------------------
Person + Patient                ->  data/patients.csv
Person + Prescriber             ->  data/prescribers.csv
Drug                            ->  data/drugs.csv
Insurance + Insurer             ->  data/insurances.csv
Prescription  (the Rx header)   ->  data/prescriptions.csv
PrescReFill   (each pickup)     ->  data/fills.csv

Only the field SHAPE is borrowed - NDC, NPI, DEA, DAW, BIN/PCN are public
NCPDP/FDA standards. No real data, no company logic.

THE HIDDEN TRUTH: every patient gets an "adherence" score - how reliably they
come back. It is never written to any file. Later the model has to rediscover
it from the dates alone. That is what machine learning actually is.

Run it:   python 01_generate_data.py
"""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)      # fixed seed -> you get the same data every run

N_PATIENTS    = 3000
N_PRESCRIBERS = 60
START = date(2023, 1, 1)             # first day this fake pharmacy has records
END   = date(2026, 6, 30)            # last day

os.makedirs("data", exist_ok=True)


# ===========================================================================
# 1. DRUGS      (shape of the Drug table)
# ===========================================================================
# drug_class follows the DEA schedule: 0 = not controlled, 2 = CII ... 5 = CV.
# qty_pack is the bottle size. default_days is a typical supply length.
DRUGS = [
    # ndc,          name,            generic,         brand,      strength,  form,     class, branded, therapeutic,      qty_pack, days, awp_unit
    ("00093105301", "Metformin",     "Metformin HCl", "Glucophage", "500 MG", "TABLET",   0, False, "Antidiabetic",        180, 90, 0.09),
    ("00378181501", "Lisinopril",    "Lisinopril",    "Zestril",    "10 MG",  "TABLET",   0, False, "ACE Inhibitor",        90, 30, 0.06),
    ("00071015623", "Atorvastatin",  "Atorvastatin",  "Lipitor",    "20 MG",  "TABLET",   0, False, "Statin",               90, 30, 0.11),
    ("00074433290", "Levothyroxine", "Levothyroxine", "Synthroid",  "75 MCG", "TABLET",   0, False, "Thyroid Hormone",     180, 90, 0.21),
    ("00069153041", "Amlodipine",    "Amlodipine",    "Norvasc",    "5 MG",   "TABLET",   0, False, "Calcium Blocker",      90, 30, 0.05),
    ("00186502031", "Omeprazole",    "Omeprazole",    "Prilosec",   "20 MG",  "CAPSULE",  0, False, "Proton Pump Inhib",    90, 30, 0.14),
    ("00006095054", "Losartan",      "Losartan Pot",  "Cozaar",     "50 MG",  "TABLET",   0, False, "ARB",                 180, 90, 0.08),
    ("00006074031", "Simvastatin",   "Simvastatin",   "Zocor",      "20 MG",  "TABLET",   0, False, "Statin",               90, 30, 0.07),
    ("00002323330", "Insulin Glarg", "Insulin Glarg", "Lantus",     "100 U/ML","SOLUTION",0, True,  "Insulin",              15, 30, 32.50),
    ("50458057901", "Gabapentin",    "Gabapentin",    "Neurontin",  "300 MG", "CAPSULE",  5, False, "Anticonvulsant",       90, 30, 0.13),
    ("00093721410", "Sertraline",    "Sertraline",    "Zoloft",     "50 MG",  "TABLET",   0, False, "SSRI",                 90, 30, 0.10),
    ("00378411205", "Montelukast",   "Montelukast",   "Singulair",  "10 MG",  "TABLET",   0, False, "Leukotriene Mod",      90, 30, 0.12),
]

drugs = pd.DataFrame(DRUGS, columns=[
    "ndc", "drug_name", "generic_name", "brand_name", "strength", "dosage_form",
    "drug_class", "is_branded", "therapeutic_class", "qty_pack", "default_days", "unit_price_awp",
])
drugs.insert(0, "drug_id", range(1, len(drugs) + 1))     # a simple surrogate key, like Drug.Id
drugs["sig"] = "TAKE 1 TABLET BY MOUTH DAILY"            # the Instructions column on Drug
drugs.to_csv("data/drugs.csv", index=False)


# ===========================================================================
# 2. INSURANCES  (shape of Insurance + Insurer)
# ===========================================================================
# BIN and PCN are how a claim is routed to the right payer. Cash = no insurance.
INSURERS = [
    # name,                bin,      pcn,        copay_generic, copay_brand, is_cash
    ("Medicare Part D",    "610502", "MEDDPRIME",  1.50,  8.00, False),
    ("Medicaid State",     "610011", "MCAIDRX",    0.00,  3.00, False),
    ("Express Scripts",    "003858", "A4",         5.00, 35.00, False),
    ("CVS Caremark",       "004336", "ADV",       10.00, 45.00, False),
    ("OptumRx",            "610279", "9999",       8.00, 40.00, False),
    ("CASH",               "",       "",           0.00,  0.00, True),
]

insurances = pd.DataFrame(INSURERS, columns=[
    "insurer_name", "bin", "pcn", "copay_generic", "copay_brand", "is_cash",
])
insurances.insert(0, "insurance_id", range(1, len(insurances) + 1))
insurances.to_csv("data/insurances.csv", index=False)


# ===========================================================================
# 3. PRESCRIBERS  (shape of Person + Prescriber)
# ===========================================================================
FIRST_NAMES = ["James", "Mary", "Robert", "Patricia", "Michael", "Jennifer", "David",
               "Linda", "William", "Elizabeth", "Richard", "Barbara", "Joseph", "Susan",
               "Thomas", "Jessica", "Charles", "Sarah", "Daniel", "Karen", "Priya",
               "Wei", "Ahmed", "Sofia", "Rajesh", "Aisha", "Diego", "Yuki"]
LAST_NAMES  = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
               "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson",
               "Anderson", "Taylor", "Thomas", "Patel", "Nguyen", "Kim", "Chen",
               "Okafor", "Rossi", "Kowalski", "Silva"]
SPECIALTIES = ["Family Medicine", "Internal Medicine", "Cardiology", "Endocrinology",
               "Nephrology", "Psychiatry", "Nurse Practitioner"]

prescriber_rows = []
for prescriber_id in range(1, N_PRESCRIBERS + 1):
    prescriber_rows.append({
        "prescriber_id": prescriber_id,
        "first_name":  rng.choice(FIRST_NAMES),
        "last_name":   rng.choice(LAST_NAMES),
        # NPI is a real 10-digit national identifier. DEA is 2 letters + 7 digits.
        "npi":         f"1{rng.integers(100000000, 999999999)}",
        "dea_num":     f"{chr(rng.integers(65, 91))}{chr(rng.integers(65, 91))}{rng.integers(1000000, 9999999)}",
        "specialty":   rng.choice(SPECIALTIES),
        "is_active":   True,
    })
prescribers = pd.DataFrame(prescriber_rows)
prescribers.to_csv("data/prescribers.csv", index=False)


# ===========================================================================
# 4. PATIENTS  (shape of Person + Patient)
# ===========================================================================
CITIES = [("Newark", "NJ", "07102"), ("Jersey City", "NJ", "07302"),
          ("Paterson", "NJ", "07501"), ("Edison", "NJ", "08817"),
          ("Trenton", "NJ", "08608"), ("Clifton", "NJ", "07011")]
LANGUAGES = ["English", "Spanish", "Hindi", "Mandarin", "Portuguese"]

TODAY = date(2026, 6, 30)                      # "today" for calculating age from birth date

patient_rows  = []
adherence_map = {}                             # patient_id -> hidden score. NEVER saved.

for patient_id in range(1, N_PATIENTS + 1):
    age = int(np.clip(rng.normal(62, 15), 18, 95))
    birth_dt = TODAY - timedelta(days=age * 365 + int(rng.integers(0, 365)))
    city, state, zip_code = CITIES[int(rng.integers(0, len(CITIES)))]

    # THE HIDDEN TRUTH. 0 = never on time, 1 = always on time.
    adherence_map[patient_id] = float(np.clip(rng.beta(1.5, 1.5) * 0.95 + 0.03, 0.02, 0.99))

    patient_rows.append({
        "patient_id":   patient_id,
        "first_name":   rng.choice(FIRST_NAMES),
        "last_name":    rng.choice(LAST_NAMES),
        "birth_dt":     birth_dt.isoformat(),
        "gender":       rng.choice(["M", "F"]),
        "city":         city,
        "state":        state,
        "zip":          zip_code,
        "language":     rng.choice(LANGUAGES, p=[0.72, 0.14, 0.06, 0.04, 0.04]),
        # PrimeInsuId on the Patient table: their main insurance
        "insurance_id": int(rng.choice(insurances["insurance_id"],
                                       p=[0.34, 0.14, 0.18, 0.16, 0.12, 0.06])),
        "is_active":    True,
    })
patients = pd.DataFrame(patient_rows)
patients.to_csv("data/patients.csv", index=False)


# ===========================================================================
# 5. PRESCRIPTIONS + FILLS
#    Prescription = the order the doctor wrote (one row, written once).
#    PrescReFill  = one row EVERY TIME the patient actually picks it up.
#    The gap between those pickups is what the model will learn from.
# ===========================================================================
DAW_CODES    = [0, 0, 0, 0, 1, 2]                       # 0 = no product selection indicated
RX_ORIGINS   = ["Electronic", "Written", "Telephone", "Fax"]

presc_rows = []
fill_rows  = []
presc_id   = 0
fill_id    = 0
presc_num  = 1000000                                    # the Rx number printed on the label


def add_prescription(patient_id, drug, written_dt, prescriber_id):
    """Write one Prescription header row. Returns its id and how many refills it allows."""
    global presc_id, presc_num                          # 'global' = update the counters outside
    presc_id  += 1
    presc_num += 1
    refills = int(rng.choice([0, 3, 5, 11], p=[0.05, 0.25, 0.30, 0.40]))
    presc_rows.append({
        "presc_id":           presc_id,
        "presc_num":          presc_num,
        "patient_id":         patient_id,
        "prescriber_id":      prescriber_id,
        "drug_id":            int(drug["drug_id"]),
        "qty":                float(drug["default_days"]),
        "days_supply":        int(drug["default_days"]),
        "refills_authorized": refills,
        "rx_receipt_dt":      written_dt.isoformat(),
        "rx_origin":          rng.choice(RX_ORIGINS, p=[0.62, 0.18, 0.12, 0.08]),
        "is_discontinued":    False,
    })
    return presc_id, refills


for patient_id in range(1, N_PATIENTS + 1):
    adherence = adherence_map[patient_id]
    insurance = insurances.loc[patients.loc[patient_id - 1, "insurance_id"] - 1]

    n_drugs  = int(rng.integers(1, 4))                          # 1-3 medications per patient
    drug_idx = rng.choice(len(drugs), size=n_drugs, replace=False)

    for i in drug_idx:
        drug = drugs.iloc[i]

        supply_days = int(drug["default_days"])
        qty         = float(supply_days)                        # 1 unit per day, keeps it simple
        written_dt  = START + timedelta(days=int(rng.integers(0, 120)))
        prescriber  = int(rng.integers(1, N_PRESCRIBERS + 1))   # same doctor for this whole course

        this_presc, refills_left = add_prescription(patient_id, drug, written_dt, prescriber)

        # ---- now every pickup against that prescription ----
        current    = written_dt
        refill_num = 0
        while current <= END:
            fill_id += 1

            # what the patient pays: generic vs brand copay, or full price if cash
            if bool(insurance["is_cash"]):
                copay = round(float(drug["unit_price_awp"]) * qty * 0.85 + 4.99, 2)
            elif bool(drug["is_branded"]):
                copay = float(insurance["copay_brand"])
            else:
                copay = float(insurance["copay_generic"])

            price = round(float(drug["unit_price_awp"]) * qty + 3.50, 2)   # AWP + dispensing fee

            fill_rows.append({
                "fill_id":       fill_id,
                "presc_id":      this_presc,
                "patient_id":    patient_id,
                "drug_id":       int(drug["drug_id"]),
                "refill_num":    refill_num,
                "fill_dt":       current.isoformat(),
                "supply_days":   supply_days,
                "disp_qty":      qty,
                "daw_id":        int(rng.choice(DAW_CODES)),
                "price":         price,
                "copay":         copay,
                "insurance_id":  int(insurance["insurance_id"]),
                "status":        "Filled",
            })

            # How long until they come back? This is where adherence shows up.
            if rng.random() > adherence:                    # unreliable this time -> late
                gap = supply_days + int(rng.integers(10, 75))
            else:                                           # reliable -> right on schedule
                gap = supply_days + int(rng.normal(1, 3))

            if rng.random() < 0.02:                         # 2% chance they stop the drug for good
                break

            current = current + timedelta(days=max(gap, 5))
            if current > END:
                break

            if refill_num >= refills_left:
                # Refills used up. In a real pharmacy the prescriber renews it,
                # which creates a NEW Rx number and resets the refill counter.
                this_presc, refills_left = add_prescription(patient_id, drug, current, prescriber)
                refill_num = 0
            else:
                refill_num += 1

prescriptions = pd.DataFrame(presc_rows)
fills         = pd.DataFrame(fill_rows)

prescriptions.to_csv("data/prescriptions.csv", index=False)
fills.sort_values(["patient_id", "presc_id", "fill_dt"]).to_csv("data/fills.csv", index=False)


# ===========================================================================
# 6. WHAT WE MADE
# ===========================================================================
print("Files written to data/")
for name, table in [("drugs", drugs), ("insurances", insurances),
                    ("prescribers", prescribers), ("patients", patients),
                    ("prescriptions", prescriptions), ("fills", fills)]:
    print(f"  {name+'.csv':<20} {len(table):>7,} rows   {len(table.columns)} columns")

print("\nOne prescription, and every pickup against it:")
sample = prescriptions.iloc[0]
print(f"  Rx #{sample['presc_num']}  patient {sample['patient_id']}  "
      f"{drugs.loc[sample['drug_id'] - 1, 'drug_name']} "
      f"{drugs.loc[sample['drug_id'] - 1, 'strength']}  "
      f"qty {sample['qty']:.0f}  {sample['days_supply']} days  "
      f"{sample['refills_authorized']} refills authorized")
print()
print(fills[fills["presc_id"] == sample["presc_id"]]
      [["refill_num", "fill_dt", "supply_days", "disp_qty", "copay"]]
      .to_string(index=False))
