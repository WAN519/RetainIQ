"""
tests/agents/test_market_db.py

Unit tests for MarketDB — MongoDB connection wrapper.
All tests mock out MongoClient so no real database is needed.
"""

import pytest
from unittest.mock import MagicMock, patch
from tools.mongoDB import MarketDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _disconnected_db() -> MarketDB:
    """Return a MarketDB instance that skipped __init__ and is marked disconnected."""
    db = MarketDB.__new__(MarketDB)
    db.client = None
    db.is_connected = False
    return db


def _connected_db(mock_collection) -> MarketDB:
    """Return a MarketDB instance wired to a mock collection."""
    db = MarketDB.__new__(MarketDB)
    db.client = MagicMock()
    db.db = MagicMock()
    db.collection = mock_collection
    db.is_connected = True
    return db


# ---------------------------------------------------------------------------
# get_benchmark
# ---------------------------------------------------------------------------

def test_get_benchmark_returns_none_when_disconnected():
    db = _disconnected_db()
    assert db.get_benchmark("Software_Developer_Seattle") is None


def test_get_benchmark_returns_document():
    expected = {"role_name": "Software_Developer_Seattle", "predicted_salary_2026": 145000.0}
    mock_col = MagicMock()
    mock_col.find_one.return_value = expected

    db = _connected_db(mock_col)
    result = db.get_benchmark("Software_Developer_Seattle")

    assert result == expected
    mock_col.find_one.assert_called_once_with({"role_name": "Software_Developer_Seattle"})


def test_get_benchmark_returns_none_for_unknown_role():
    mock_col = MagicMock()
    mock_col.find_one.return_value = None

    db = _connected_db(mock_col)
    assert db.get_benchmark("Unknown_Role") is None


# ---------------------------------------------------------------------------
# upsert_benchmark
# ---------------------------------------------------------------------------

def test_upsert_benchmark_returns_false_when_disconnected():
    db = _disconnected_db()
    assert db.upsert_benchmark({"role_name": "x"}) is False


def test_upsert_benchmark_calls_update_one():
    mock_col = MagicMock()
    mock_col.update_one.return_value = MagicMock(upserted_id=None)

    db = _connected_db(mock_col)
    result = db.upsert_benchmark({"role_name": "Data_Scientist_Seattle", "predicted_salary_2026": 130000})

    assert result is True
    mock_col.update_one.assert_called_once()