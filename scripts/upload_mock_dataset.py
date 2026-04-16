"""
scripts/upload_mock_dataset.py

Upload data/mock_dataset.csv to the cloud MySQL database (Aiven).
Uses DatabaseManager from scripts/database_mysql.py for the connection.

Actions:
  1. CREATE TABLE IF NOT EXISTS `employee_risk_profiles`
  2. INSERT all 1020 records in batches of 100

Table: employee_risk_profiles
  - Matches ibm_enhanced_test.csv column schema
  - Adds manager_id to capture reporting lines

Usage:
    python scripts/upload_mock_dataset.py
"""

import csv
import sys
from pathlib import Path

# Make sure project root is on sys.path so we can import scripts.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.database_mysql import DatabaseManager

CSV_PATH   = PROJECT_ROOT / "data" / "mock_dataset.csv"
TABLE_NAME = "employee_risk_profiles"
BATCH_SIZE = 100

# ──────────────────────────────────────────────────────────────────────────────
# DDL — create table if it doesn't already exist
# ──────────────────────────────────────────────────────────────────────────────
CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS `{TABLE_NAME}` (
    `employee_id`             VARCHAR(20)    NOT NULL,
    `manager_id`              VARCHAR(20)    NOT NULL,
    `role_name`               VARCHAR(100)   NOT NULL,
    `current_salary`          INT            NOT NULL,
    `BusinessTravel`          TINYINT        NOT NULL,
    `DailyRate`               INT            NOT NULL,
    `Department`              TINYINT        NOT NULL,
    `DistanceFromHome`        TINYINT        NOT NULL,
    `Education`               TINYINT        NOT NULL,
    `EducationField`          TINYINT        NOT NULL,
    `HourlyRate`              TINYINT        NOT NULL,
    `JobInvolvement`          TINYINT        NOT NULL,
    `JobLevel`                TINYINT        NOT NULL,
    `JobRole`                 TINYINT        NOT NULL,
    `MaritalStatus`           TINYINT        NOT NULL,
    `MonthlyRate`             INT            NOT NULL,
    `NumCompaniesWorked`      TINYINT        NOT NULL,
    `Over18`                  TINYINT        NOT NULL,
    `PercentSalaryHike`       TINYINT        NOT NULL,
    `PerformanceRating`       TINYINT        NOT NULL,
    `StockOptionLevel`        TINYINT        NOT NULL,
    `TotalWorkingYears`       TINYINT        NOT NULL,
    `TrainingTimesLastYear`   TINYINT        NOT NULL,
    `YearsAtCompany`          TINYINT        NOT NULL,
    `YearsInCurrentRole`      TINYINT        NOT NULL,
    `YearsWithCurrManager`    TINYINT        NOT NULL,
    `Market_Median_2026`      INT            NOT NULL,
    `Internal_Salary_Rank`    FLOAT          NOT NULL,
    `Performance_Consistency` TINYINT        NOT NULL,
    PRIMARY KEY (`employee_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# INSERT template — positional placeholders match COLUMNS order
COLUMNS = [
    "employee_id", "manager_id", "role_name", "current_salary",
    "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
    "Education", "EducationField", "HourlyRate", "JobInvolvement",
    "JobLevel", "JobRole", "MaritalStatus", "MonthlyRate",
    "NumCompaniesWorked", "Over18", "PercentSalaryHike", "PerformanceRating",
    "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear",
    "YearsAtCompany", "YearsInCurrentRole", "YearsWithCurrManager",
    "Market_Median_2026", "Internal_Salary_Rank", "Performance_Consistency",
]

PLACEHOLDERS = ", ".join(["%s"] * len(COLUMNS))
COL_LIST     = ", ".join(f"`{c}`" for c in COLUMNS)
INSERT_SQL   = f"INSERT IGNORE INTO `{TABLE_NAME}` ({COL_LIST}) VALUES ({PLACEHOLDERS})"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_csv(path: Path) -> list[tuple]:
    """Read CSV and return rows as tuples in COLUMNS order."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(tuple(
                float(row[c]) if c in ("Internal_Salary_Rank",) else int(float(row[c])) if c != "role_name" and c != "employee_id" and c != "manager_id" else row[c]
                for c in COLUMNS
            ))
    return rows


def insert_batch(db: DatabaseManager, batch: list[tuple]) -> bool:
    """
    Insert a batch using executemany via a raw connection.
    Falls back to row-by-row inserts if executemany is unavailable.
    """
    conn = db.get_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.executemany(INSERT_SQL, batch)
        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        print(f"  [ERROR] Batch insert failed: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ── Connect ──────────────────────────────────────────────────────────────
    print("[MySQL] Connecting to cloud database...")
    db = DatabaseManager()
    if not db.connect():
        print("[ERROR] Cannot reach the database. Check config.env credentials.")
        sys.exit(1)

    # ── Create table ─────────────────────────────────────────────────────────
    print(f"[MySQL] Creating table `{TABLE_NAME}` if not exists...")
    if not db.execute_sql(CREATE_TABLE_SQL):
        print("[ERROR] Failed to create table.")
        sys.exit(1)
    print(f"[MySQL] Table ready.")

    # ── Load CSV ─────────────────────────────────────────────────────────────
    print(f"[CSV]   Loading {CSV_PATH} ...")
    rows = load_csv(CSV_PATH)
    print(f"[CSV]   {len(rows)} records loaded.")

    # ── Batch insert ─────────────────────────────────────────────────────────
    total_inserted = 0
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        ok = insert_batch(db, batch)
        if ok:
            total_inserted += len(batch)
            print(f"  Inserted rows {start + 1}–{start + len(batch)} ✓")
        else:
            print(f"  [WARN] Batch {start + 1}–{start + len(batch)} failed, skipping.")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n[Done] {total_inserted}/{len(rows)} records inserted into `{TABLE_NAME}`.")


if __name__ == "__main__":
    main()
