"""
scripts/generate_mock_dataset.py

Generate a mock HR dataset:
  - 1000 regular employees  (distributed unevenly across 10 managers)
  -   10 managers           (JobLevel 4-5, JobRole=3)
  -   10 HR staff           (~1:100 HR ratio; Department=3)
  Total: 1020 records

Output schema matches ibm_enhanced_test.csv, plus one extra column
`manager_id` to capture reporting lines.

Encoding (matches Attrition.csv / ibm_enhanced_test.csv):
  BusinessTravel  : 0=Non-Travel, 1=Travel_Rarely, 2=Travel_Frequently
  Department      : 1=R&D, 2=Sales, 3=Human Resources
  EducationField  : 1=Life Sciences, 2=Other, 3=Medical, 4=Marketing,
                    5=Technical Degree, 6=Human Resources
  JobRole         : 0=Healthcare Rep, 1=Human Resources, 2=Lab Technician,
                    3=Manager, 4=Manufacturing Director, 5=Research Director,
                    6=Research Scientist, 7=Sales Executive, 8=Sales Rep
  MaritalStatus   : 0=Divorced, 1=Single, 2=Married
  Over18          : 1=Y (always)

Output: data/mock_dataset.csv
"""

import csv
import random
from collections import Counter
from pathlib import Path

random.seed(42)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "mock_dataset.csv"

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

CITIES = ["Seattle", "Austin", "NYC", "Chicago", "Boston", "Denver", "Atlanta", "Miami"]

# ──────────────────────────────────────────────────────────────────────────────
# Role catalogues
# Tuple shape:
#   employee / HR : (role_prefix, dept, jobrole, sal_min, sal_max, market_median, level_range)
#   manager       : (role_prefix, dept, jobrole, sal_min, sal_max, market_median)
# ──────────────────────────────────────────────────────────────────────────────
EMPLOYEE_ROLES = [
    ("Software_Developer",        1, 7, 95_000,  190_000, 155_000, (1, 3)),
    ("Data_Scientist",            1, 6, 115_000, 205_000, 170_000, (2, 4)),
    ("Research_Scientist",        1, 5, 85_000,  155_000, 120_000, (1, 3)),
    ("Laboratory_Technician",     1, 2, 55_000,  100_000,  75_000, (1, 2)),
    ("Sales_Executive",           2, 7, 75_000,  145_000, 110_000, (2, 3)),
    ("Sales_Representative",      2, 8, 50_000,   92_000,  70_000, (1, 2)),
    ("Healthcare_Representative", 1, 0, 60_000,  115_000,  85_000, (1, 2)),
    ("Manufacturing_Director",    1, 4, 95_000,  165_000, 130_000, (3, 4)),
    ("Business_Analyst",          1, 2, 78_000,  135_000, 105_000, (2, 3)),
]

# 10 manager templates — cycled if needed
MANAGER_ROLES = [
    ("Engineering_Manager",  1, 3, 145_000, 235_000, 190_000),
    ("Sales_Manager",        2, 3, 125_000, 205_000, 165_000),
    ("Research_Manager",     1, 3, 135_000, 215_000, 175_000),
    ("Product_Manager",      1, 3, 140_000, 225_000, 180_000),
    ("Operations_Manager",   1, 3, 125_000, 200_000, 160_000),
    ("Engineering_Manager",  1, 3, 150_000, 240_000, 192_000),
    ("Sales_Manager",        2, 3, 130_000, 210_000, 168_000),
    ("Research_Manager",     1, 3, 138_000, 218_000, 178_000),
    ("Product_Manager",      1, 3, 143_000, 228_000, 182_000),
    ("Operations_Manager",   1, 3, 128_000, 202_000, 162_000),
]

HR_ROLES = [
    ("HR_Business_Partner", 3, 1, 80_000, 135_000, 105_000, (3, 4)),
    ("HR_Specialist",       3, 1, 58_000, 100_000,  80_000, (2, 3)),
    ("HR_Coordinator",      3, 1, 45_000,  80_000,  65_000, (1, 2)),
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def rand_salary(lo: int, hi: int) -> int:
    return round(random.uniform(lo, hi) / 1_000) * 1_000


def rand_education_field(dept: int) -> int:
    if dept == 3:       # HR
        weights = [10, 10, 10, 15, 10, 45]
    elif dept == 2:     # Sales
        weights = [20, 15, 10, 30, 20,  5]
    else:               # R&D
        weights = [30, 15, 15, 10, 25,  5]
    return random.choices([1, 2, 3, 4, 5, 6], weights=weights)[0]


def build_record(
    emp_id: str,
    manager_id: str,
    role_prefix: str,
    city: str,
    dept: int,
    jobrole: int,
    sal_min: int,
    sal_max: int,
    market_median: int,
    level_range: tuple,
) -> dict:
    salary       = rand_salary(sal_min, sal_max)
    total_yrs    = random.randint(1, 38)
    yrs_company  = random.randint(0, min(total_yrs, 30))
    yrs_role     = random.randint(0, min(yrs_company, 15))
    yrs_mgr      = random.randint(0, min(yrs_company, 14))
    salary_rank  = round(min(max(random.gauss(0.5, 0.22), 0.0), 1.0), 2)

    return {
        "employee_id":             emp_id,
        "manager_id":              manager_id,
        "role_name":               f"{role_prefix}_{city}",
        "current_salary":          salary,
        "BusinessTravel":          random.choices([0, 1, 2], weights=[10, 65, 25])[0],
        "DailyRate":               random.randint(100, 1_499),
        "Department":              dept,
        "DistanceFromHome":        random.randint(1, 29),
        "Education":               random.choices([1, 2, 3, 4, 5], weights=[8, 18, 36, 28, 10])[0],
        "EducationField":          rand_education_field(dept),
        "HourlyRate":              random.randint(30, 100),
        "JobInvolvement":          random.choices([1, 2, 3, 4], weights=[10, 25, 45, 20])[0],
        "JobLevel":                random.randint(*level_range),
        "JobRole":                 jobrole,
        "MaritalStatus":           random.choices([0, 1, 2], weights=[18, 34, 48])[0],
        "MonthlyRate":             random.randint(2_094, 26_999),
        "NumCompaniesWorked":      random.randint(0, 9),
        "Over18":                  1,
        "PercentSalaryHike":       random.randint(11, 25),
        "PerformanceRating":       random.choices([3, 4], weights=[85, 15])[0],
        "StockOptionLevel":        random.choices([0, 1, 2, 3], weights=[40, 35, 15, 10])[0],
        "TotalWorkingYears":       total_yrs,
        "TrainingTimesLastYear":   random.randint(0, 6),
        "YearsAtCompany":          yrs_company,
        "YearsInCurrentRole":      yrs_role,
        "YearsWithCurrManager":    yrs_mgr,
        "Market_Median_2026":      market_median,
        "Internal_Salary_Rank":    salary_rank,
        "Performance_Consistency": random.choices([1, 2, 3, 4], weights=[5, 20, 55, 20])[0],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dataset generator
# ──────────────────────────────────────────────────────────────────────────────
def generate_dataset() -> list[dict]:
    records: list[dict] = []
    counter = 1

    def next_id() -> str:
        nonlocal counter
        eid = f"EMP{counter:04d}"
        counter += 1
        return eid

    # ── 1. Managers (10) ──────────────────────────────────────────────────────
    manager_ids: list[str] = []
    for i in range(10):
        prefix, dept, jobrole, sal_min, sal_max, market_median = MANAGER_ROLES[i]
        emp_id = next_id()
        rec = build_record(
            emp_id=emp_id,
            manager_id="EXEC",
            role_prefix=prefix,
            city=random.choice(CITIES),
            dept=dept,
            jobrole=jobrole,
            sal_min=sal_min,
            sal_max=sal_max,
            market_median=market_median,
            level_range=(4, 5),
        )
        records.append(rec)
        manager_ids.append(emp_id)

    # ── 2. HR staff (10) ──────────────────────────────────────────────────────
    # HR reports to a single dedicated HR manager (the last manager in the list)
    hr_manager_id = manager_ids[-1]
    for _ in range(10):
        prefix, dept, jobrole, sal_min, sal_max, market_median, level_range = random.choice(HR_ROLES)
        emp_id = next_id()
        rec = build_record(
            emp_id=emp_id,
            manager_id=hr_manager_id,
            role_prefix=prefix,
            city=random.choice(CITIES),
            dept=dept,
            jobrole=jobrole,
            sal_min=sal_min,
            sal_max=sal_max,
            market_median=market_median,
            level_range=level_range,
        )
        records.append(rec)

    # ── 3. Employees (1000) — unevenly distributed across managers ────────────
    # Random weights give each manager a different team size (none are equal)
    raw_weights = [random.uniform(0.4, 2.8) for _ in manager_ids]
    total_w = sum(raw_weights)
    team_sizes: list[int] = []
    allocated = 0
    for i, w in enumerate(raw_weights):
        if i == len(raw_weights) - 1:
            team_sizes.append(1_000 - allocated)   # absorb rounding remainder
        else:
            n = max(1, round(1_000 * w / total_w))
            team_sizes.append(n)
            allocated += n

    for mgr_id, team_size in zip(manager_ids, team_sizes):
        for _ in range(team_size):
            prefix, dept, jobrole, sal_min, sal_max, market_median, level_range = random.choice(EMPLOYEE_ROLES)
            emp_id = next_id()
            rec = build_record(
                emp_id=emp_id,
                manager_id=mgr_id,
                role_prefix=prefix,
                city=random.choice(CITIES),
                dept=dept,
                jobrole=jobrole,
                sal_min=sal_min,
                sal_max=sal_max,
                market_median=market_median,
                level_range=level_range,
            )
            records.append(rec)

    return records, team_sizes, manager_ids


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("Generating mock HR dataset...")
    records, team_sizes, manager_ids = generate_dataset()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    managers  = records[:10]
    hr_staff  = records[10:20]
    employees = records[20:]

    print(f"\nWritten {len(records)} records → {OUTPUT_PATH}")
    print(f"  Managers : {len(managers)}")
    print(f"  HR staff : {len(hr_staff)}")
    print(f"  Employees: {len(employees)}")

    print("\nManager team sizes (employees only):")
    for mgr_id, size in zip(manager_ids, team_sizes):
        role = next(r["role_name"] for r in managers if r["employee_id"] == mgr_id)
        print(f"  {mgr_id}  {role:<40}  {size:>4} reports")


if __name__ == "__main__":
    main()
