"""
tests/conftest.py — Shared pytest fixtures for RetentionAgent test suite.
"""

import pytest
import pandas as pd


@pytest.fixture
def sample_hr_df(tmp_path):
    """Minimal HR CSV with one normal and one high-risk employee."""
    csv = tmp_path / "hr.csv"
    pd.DataFrame([
        {
            "EmployeeNumber": 1,
            "OverTime": "Yes",
            "YearsSinceLastPromotion": 3,
            "YearsAtCompany": 5,
            "MonthlyIncome": 4000,
            "JobSatisfaction": 2,
            "WorkLifeBalance": 1,
            "Attrition": "Yes",
            "Department": "Engineering",
            "JobRole": "Software Engineer",
        },
        {
            "EmployeeNumber": 2,
            "OverTime": "No",
            "YearsSinceLastPromotion": 1,
            "YearsAtCompany": 3,
            "MonthlyIncome": 9000,
            "JobSatisfaction": 4,
            "WorkLifeBalance": 3,
            "Attrition": "No",
            "Department": "Sales",
            "JobRole": "Sales Rep",
        },
    ]).to_csv(csv, index=False)
    return str(csv)


@pytest.fixture
def sample_review_df():
    """Small Glassdoor review DataFrame for NLP pipeline tests."""
    return pd.DataFrame([
        {"firm": "Acme", "cons": "Management is terrible and pay is low.", "recommend": "x", "current": "Former Employee"},
        {"firm": "Acme", "cons": "Great work-life balance and good pay.", "recommend": "v", "current": "Current Employee"},
        {"firm": "Acme", "cons": "",  "recommend": "o", "current": "Current Employee"},
    ])
