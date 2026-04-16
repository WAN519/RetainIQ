"""
scripts/MockData.py

Seed script for populating the MySQL database with realistic demo data.

Generates and inserts:
  - 1,000 synthetic employees across 10 departments and 10 job roles.
    ~15% of employees are deliberately created as "high-risk" (low pay,
    poor work-life balance, frequent overtime) to ensure the dashboard
    demonstrates meaningful at-risk flagging.
  - Market salary reference data for each role.
  - 10 Manager accounts (one per department) and 3 HR Admin accounts.
    All demo accounts share the password: mlh2026

Tables written (MySQL):
    market_data     — role-level average market salaries
    employee_stats  — individual employee attributes used by the retention model
    users           — login credentials for managers and HR admins

Usage:
    python scripts/MockData.py
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import bcrypt
import random

# Update this connection string to match your local or cloud MySQL instance
DB_URL = "mysql+pymysql://root:password@localhost/retention_agent_db"
engine = create_engine(DB_URL)


def hash_password(password: str) -> str:
    """
    Hash a plaintext password using bcrypt.

    Args:
        password (str): Plaintext password string.

    Returns:
        str: bcrypt-hashed password string, safe for database storage.
    """
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def generate_bulk_data():
    """
    Generate and insert all seed data into the MySQL database.

    High-risk employees (15% of total) are characterized by:
      - Monthly income in the lower $3,000–$6,000 band
      - Work-life balance score of 1 or 2 (out of 4)
      - Overtime always set to 1
      - 3–6 years since last promotion (stagnant career)
    """
    print("Generating demo data for MLH submission...")

    depts = [
        'Engineering', 'Sales', 'Marketing', 'R&D', 'Finance',
        'Human Resources', 'Legal', 'Support', 'Product', 'Operations'
    ]
    roles = [
        'Software Engineer', 'Sales Rep', 'Marketing Specialist', 'Research Scientist',
        'Analyst', 'HRBP', 'Legal Counsel', 'Support Tech', 'Product Manager', 'Coordinator'
    ]

    # --- Market salary reference data ---
    market_data = []
    for role in roles:
        market_data.append({
            "job_role": role,
            "avg_market_salary": random.randint(5000, 12000),
            "cpi_index_ref": 102.5
        })
    pd.DataFrame(market_data).to_sql('market_data', engine, if_exists='replace', index=False)

    # --- Employee records ---
    employee_list = []
    for i in range(1, 1001):
        dept = random.choice(depts)
        role = random.choice(roles)

        # Flag ~15% of employees as high-risk to surface meaningful retention alerts
        is_high_risk = random.random() < 0.15

        employee_list.append({
            "employee_id":                1000 + i,
            "name":                       f"Employee_{i}",
            "department":                 dept,
            "job_role":                   role,
            "monthly_income":             random.randint(3000, 6000) if is_high_risk else random.randint(7000, 15000),
            "years_at_company":           random.randint(1, 10),
            "work_life_balance":          random.randint(1, 2) if is_high_risk else random.randint(3, 4),
            "overtime_status":            1 if is_high_risk else random.choice([0, 1]),
            "years_since_last_promotion": random.randint(3, 6) if is_high_risk else random.randint(0, 2),
        })
    pd.DataFrame(employee_list).to_sql('employee_stats', engine, if_exists='replace', index=False)

    # --- User accounts ---
    user_list = []
    # All demo accounts use the same password for convenience during judging
    common_pwd = hash_password("mlh2026")

    # One manager per department, linked to the first employee of each dept index
    for i in range(10):
        dept_name = depts[i]
        user_list.append({
            "username":        f"manager_{dept_name.lower()}",
            "password_hash":   common_pwd,
            "role":            "Manager",
            "managed_dept":    dept_name,
            "self_employee_id": 1000 + (i + 1),
        })

    # Three HR admin accounts with access to all departments
    for i in range(1, 4):
        user_list.append({
            "username":        f"hr_admin_{i}",
            "password_hash":   common_pwd,
            "role":            "HR",
            "managed_dept":    None,
            "self_employee_id": 1000 + (10 + i),
        })

    pd.DataFrame(user_list).to_sql('users', engine, if_exists='replace', index=False)
    print("Done: 1,000 employees, 10 managers, and 3 HR admins inserted into MySQL.")


if __name__ == "__main__":
    generate_bulk_data()