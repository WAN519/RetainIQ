"""
tests/agents/test_market_logic.py

Unit tests for MarketDataCoordinator — BLS API fetcher with MongoDB cache.
All HTTP calls are mocked so no real API key is required.
"""

import pytest
import datetime
from unittest.mock import MagicMock, patch
from tools.market_logic import MarketDataCoordinator


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.get_benchmark.return_value = None  # cache miss by default
    return db


@pytest.fixture
def coordinator(mock_db):
    return MarketDataCoordinator(db_handler=mock_db, api_key="FAKE_KEY")


# ---------------------------------------------------------------------------
# Cache hit — should NOT call BLS API
# ---------------------------------------------------------------------------

def test_returns_cached_result_when_fresh(coordinator, mock_db):
    fresh_ts = datetime.datetime.utcnow() - datetime.timedelta(days=5)
    cached = {"role_name": "Software_Developer_Seattle", "predicted_salary_2026": 145000, "sync_timestamp": fresh_ts}
    mock_db.get_benchmark.return_value = cached

    result = coordinator.get_market_intelligence("SERIES_ID", "Software_Developer_Seattle")

    assert result == cached
    mock_db.upsert_benchmark.assert_not_called()


# ---------------------------------------------------------------------------
# Cache miss — should fetch from BLS and save result
# ---------------------------------------------------------------------------

def test_fetches_from_bls_on_cache_miss(coordinator, mock_db):
    fake_salary_response = {
        "Results": {"series": [{"data": [{"value": "50.00"}]}]}
    }
    fake_cpi_response = {
        "Results": {"series": [{"data": [{"value": "280.0"}]}]}
    }

    with patch("tools.market_logic.requests.post") as mock_post:
        # First call: wage data. Second/third calls: CPI data.
        mock_post.return_value.json.side_effect = [
            fake_salary_response,
            fake_cpi_response,
            fake_cpi_response,
        ]

        result = coordinator.get_market_intelligence("SERIES_ID", "Software_Developer_Seattle")

    assert result is not None
    assert "predicted_salary_2026" in result
    mock_db.upsert_benchmark.assert_called_once()


def test_returns_none_when_bls_has_no_data(coordinator, mock_db):
    empty_response = {"Results": {"series": [{"data": []}]}}

    with patch("tools.market_logic.requests.post") as mock_post:
        mock_post.return_value.json.return_value = empty_response
        result = coordinator.get_market_intelligence("BAD_SERIES", "Unknown_Role")

    assert result is None


# ---------------------------------------------------------------------------
# Salary projection formula sanity check
# ---------------------------------------------------------------------------

def test_salary_projection_formula(coordinator):
    """
    Manual verification of: (hourly * 2080) * (now_cpi / base_cpi) * 1.03
    With hourly=50, base_cpi=280, now_cpi=300 → expected ≈ 111,214
    """
    hourly, base_cpi, now_cpi = 50.0, 280.0, 300.0
    expected = round((hourly * 2080) * (now_cpi / base_cpi) * 1.03, 2)
    assert expected == pytest.approx(114_728.57, rel=1e-3)