"""
STEP 3b - Build the SAME training table, but in SQL instead of pandas.

File 03 did all the work in Python. This does it in one query, inside the
database. Same 101,946 rows, same 55.9% label rate - I checked.

Why bother:
  - the database is where the data already is. Moving 105,000 rows into
    Python to compute an average is wasted movement.
  - at 105 million rows the pandas version runs out of memory. This does not.
  - every step here is a window function, which is the SQL you get asked
    about in interviews.

Needs the database running:  docker compose up -d
Run it:   python 03b_build_training_table_sql.py
"""

import os

import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(os.getenv(
    "PHARMACY_DB_URL",
    "postgresql+psycopg2://rx:rxpass@localhost:5432/pharmacy",
))


# --- The query ------------------------------------------------------------
#
# Every pandas step from file 03 has a SQL twin:
#
#   pandas                                SQL
#   -------------------------------       ------------------------------
#   groupby([...]).shift(-1)              LEAD(...)  OVER w
#   groupby([...]).diff()                 x - LAG(x) OVER w
#   groupby([...]).cumcount() + 1         ROW_NUMBER() OVER w
#   groupby([...]).expanding().mean()     AVG(x) OVER h
#   merge(patients, on=...)               JOIN patients ON ...
#
# The magic word is OVER. It means "look at other rows, but still give me
# one answer per row". PARTITION BY is groupby. ORDER BY is sort_values.

SQL = """
WITH base AS (
    SELECT  f.patient_id,
            f.drug_id,
            f.fill_dt::date                                  AS fill_dt,
            f.supply_days,
            d.drug_name,
            d.is_branded,
            p.birth_dt::date                                 AS birth_dt,

            -- shift(-1): the NEXT pickup for this patient and drug
            LEAD(f.fill_dt::date) OVER w                     AS next_fill_dt,

            -- diff(): today minus the PREVIOUS pickup, in days
            (f.fill_dt::date - LAG(f.fill_dt::date) OVER w)  AS prev_gap_raw,

            -- cumcount()+1: is this their 1st pickup, 2nd, 3rd...
            ROW_NUMBER() OVER w                              AS fill_number
    FROM    fills f
    JOIN    drugs    d ON d.drug_id    = f.drug_id
    JOIN    patients p ON p.patient_id = f.patient_id

    -- Define the window once and reuse it, instead of repeating it 3 times.
    WINDOW  w AS (PARTITION BY f.patient_id, f.drug_id ORDER BY f.fill_dt::date)
),
gaps AS (
    SELECT  *,
            COALESCE(prev_gap_raw, supply_days)   AS prev_gap,   -- fillna()
            fill_dt + supply_days::int            AS runout_dt   -- day the pills run out
    FROM    base
),
flags AS (
    SELECT  *,
            CASE WHEN prev_gap > supply_days + 14 THEN 1 ELSE 0 END AS was_late
    FROM    gaps
),
history AS (
    SELECT  *,
            AVG(prev_gap) OVER h  AS avg_gap,
            AVG(was_late) OVER h  AS pct_late
    FROM    flags

    -- ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW is .expanding().
    -- "everything from the start up to this row, and nothing after it".
    -- Leave it out and SQL would average the WHOLE group - future included.
    -- That is the leakage trap from file 03, in a different language.
    WINDOW  h AS (PARTITION BY patient_id, drug_id ORDER BY fill_dt
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
)
SELECT  EXTRACT(YEAR FROM AGE(fill_dt, birth_dt))::int  AS patient_age,
        drug_name,
        is_branded,
        supply_days,
        fill_number,
        prev_gap,
        was_late,
        ROUND(avg_gap::numeric,  4) AS avg_gap,
        ROUND(pct_late::numeric, 4) AS pct_late,
        runout_dt,

        -- the label: did they come back within 14 days of running out?
        CASE WHEN next_fill_dt IS NOT NULL
              AND (next_fill_dt - runout_dt) <= 14 THEN 1 ELSE 0 END AS came_back
FROM    history

-- Censoring, as one line. Drop rows whose 14-day window runs past our data.
WHERE   runout_dt <= (SELECT MAX(fill_dt::date) - 14 FROM fills)
ORDER BY runout_dt
"""

table = pd.read_sql(SQL, engine, parse_dates=["runout_dt"])

table.to_csv("data/training_table_sql.csv", index=False)

print("Rows :", len(table))
print("Cols :", len(table.columns))
print("Came back in time:", round(table["came_back"].mean() * 100, 1), "%")
print()
print(table.head(5).to_string(index=False))
print()
print("Saved -> data/training_table_sql.csv")
print()
print("Compare with data/training_table.csv from file 03. They should match.")
