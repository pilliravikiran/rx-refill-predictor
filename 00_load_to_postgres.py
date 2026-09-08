"""
STEP 0 - Put the pharmacy data in a real database.

CSV files are fine for learning. No pharmacy runs on them.
A database gives you: many users at once, indexes, joins, types, and
permissions - none of which a CSV has.

Start the database first (needs Docker running):

    docker compose up -d

Then:   python 00_load_to_postgres.py
Stop it later with:  docker compose down
"""

import os

import pandas as pd
from sqlalchemy import create_engine, text


# --- PIECE 1: connect ------------------------------------------------------

# The connection string, piece by piece:
#   postgresql+psycopg2   which database, and which python driver
#   rx:rxpass             username : password
#   localhost:5432        where it is listening
#   /pharmacy             which database on that server
#
# Read from an environment variable first, so a real deployment can point
# somewhere else without touching the code. Never hard-code a real password.
DB_URL = os.getenv(
    "PHARMACY_DB_URL",
    "postgresql+psycopg2://rx:rxpass@localhost:5432/pharmacy",
)

engine = create_engine(DB_URL)      # a pool of connections, not one connection

with engine.connect() as conn:
    print("Connected to:", conn.execute(text("SELECT current_database()")).scalar())


# --- PIECE 2: load each CSV into its own table ----------------------------

TABLES = {
    "drugs":         "data/drugs.csv",
    "insurances":    "data/insurances.csv",
    "prescribers":   "data/prescribers.csv",
    "patients":      "data/patients.csv",
    "prescriptions": "data/prescriptions.csv",
    "fills":         "data/fills.csv",
}

print()
print("Loading tables:")
for table, path in TABLES.items():
    df = pd.read_csv(path)

    # to_sql creates the table and inserts the rows.
    #   if_exists="replace"  drop and rebuild, so re-running is safe
    #   index=False          do not add pandas' row number as a column
    #   chunksize=5000       insert in batches, not 105,000 rows in one go
    df.to_sql(table, engine, if_exists="replace", index=False, chunksize=5000)
    print(f"  {table:<15}{len(df):>8,} rows")


# --- PIECE 3: indexes -----------------------------------------------------

# An index is a lookup shortcut. Without one, "find this patient's fills"
# reads all 105,549 rows. With one, it jumps straight there.
# Index the columns you filter and join on.
INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_fills_patient ON fills (patient_id)",
    "CREATE INDEX IF NOT EXISTS ix_fills_drug    ON fills (drug_id)",
    "CREATE INDEX IF NOT EXISTS ix_fills_date    ON fills (fill_dt)",
    "CREATE INDEX IF NOT EXISTS ix_presc_patient ON prescriptions (patient_id)",
]

with engine.connect() as conn:
    for statement in INDEXES:
        conn.execute(text(statement))
    conn.commit()            # nothing is saved until you commit

print()
print("Indexes created.")
