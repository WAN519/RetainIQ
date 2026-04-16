"""
tests/agents/test_retention_agent.py

Unit tests for RetentionAgentV1 — Cox survival model attrition scorer.
Model loading and MongoDB are mocked; CSV processing uses tmp_path fixtures.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from agents.retention.retention_agent_v1 import RetentionAgentV1, ensure_columns


# ---------------------------------------------------------------------------
# ensure_columns (pure function)
# ---------------------------------------------------------------------------

def test_ensure_columns_adds_missing():
    df = pd.DataFrame({"a": [1, 2]})
    result = ensure_columns(df, ["a", "b", "c"], fill_value=0)
    assert "b" in result.columns
    assert "c" in result.columns
    assert result["b"].tolist() == [0, 0]


def test_ensure_columns_does_not_overwrite_existing():
    df = pd.DataFrame({"a": [1, 2]})
    result = ensure_columns(df, ["a"], fill_value=99)
    assert result["a"].tolist() == [1, 2]


# ---------------------------------------------------------------------------
# Fixture: agent with mocked model and DB
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    a = RetentionAgentV1.__new__(RetentionAgentV1)
    a.feature_columns = [
        "OverTime_flag", "MonthsSincePromotion", "YearsSinceLastPromotion",
        "MonthlyIncome", "MonthlyIncome_k", "duration", "event",
    ]
    a.cph = MagicMock()
    a.db = MagicMock()
    a.db.is_connected = True
    return a


# ---------------------------------------------------------------------------
# _prepare_hr_data
# ---------------------------------------------------------------------------

def test_prepare_hr_data_overtime_flag(agent, sample_hr_df):
    df = agent._prepare_hr_data(sample_hr_df)
    # Employee 1 has OverTime=Yes → flag should be 1
    assert df.loc[df["employee_id"] == 1, "OverTime_flag"].iloc[0] == 1
    # Employee 2 has OverTime=No → flag should be 0
    assert df.loc[df["employee_id"] == 2, "OverTime_flag"].iloc[0] == 0


def test_prepare_hr_data_months_since_promotion(agent, sample_hr_df):
    df = agent._prepare_hr_data(sample_hr_df)
    # Employee 1: YearsSinceLastPromotion=3 → MonthsSincePromotion=36
    assert df.loc[df["employee_id"] == 1, "MonthsSincePromotion"].iloc[0] == pytest.approx(36.0)


def test_prepare_hr_data_attrition_event(agent, sample_hr_df):
    df = agent._prepare_hr_data(sample_hr_df)
    assert df.loc[df["employee_id"] == 1, "event"].iloc[0] == 1  # Attrition=Yes
    assert df.loc[df["employee_id"] == 2, "event"].iloc[0] == 0  # Attrition=No


def test_prepare_hr_data_raises_on_missing_file(agent):
    with pytest.raises(FileNotFoundError):
        agent._prepare_hr_data("nonexistent_file.csv")


def test_prepare_hr_data_duration_replaces_zero(agent, tmp_path):
    """Employees with 0 years at company should get duration=0.5 to avoid model degenerate behavior."""
    csv = tmp_path / "hr_zero.csv"
    pd.DataFrame([{
        "EmployeeNumber": 99, "OverTime": "No", "YearsSinceLastPromotion": 0,
        "YearsAtCompany": 0, "MonthlyIncome": 5000, "Attrition": "No",
    }]).to_csv(csv, index=False)

    df = agent._prepare_hr_data(str(csv))
    assert df["duration"].iloc[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _build_model_input
# ---------------------------------------------------------------------------

def test_build_model_input_column_alignment(agent, sample_hr_df):
    hr = agent._prepare_hr_data(sample_hr_df)
    X = agent._build_model_input(hr)
    assert list(X.columns) == agent.feature_columns


def test_build_model_input_no_nans(agent, sample_hr_df):
    hr = agent._prepare_hr_data(sample_hr_df)
    X = agent._build_model_input(hr)
    assert not X.isnull().any().any()


# ---------------------------------------------------------------------------
# _load_equity_predictions — nested dict format
# ---------------------------------------------------------------------------

def test_load_equity_predictions_unpacks_nested_gaps(agent):
    agent.db.db = {
        "Equity_Predictions": MagicMock()
    }
    agent.db.db["Equity_Predictions"].find.return_value = [
        {"employee_id": 1, "equity_gaps": {"external_gap_pct": -15.0}}
    ]

    df = agent._load_equity_predictions()
    assert df["pay_gap"].iloc[0] == pytest.approx(-0.15)


def test_load_equity_predictions_empty_collection(agent):
    agent.db.db = {"Equity_Predictions": MagicMock()}
    agent.db.db["Equity_Predictions"].find.return_value = []

    df = agent._load_equity_predictions()
    assert df.empty