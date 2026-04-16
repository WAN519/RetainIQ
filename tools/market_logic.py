"""
tools/market_logic.py

Fetches and inflation-adjusts market salary data from the U.S. Bureau of
Labor Statistics (BLS) public API, then caches results in MongoDB.

Salary projection formula:
    predicted = (hourly_wage * 2080) * (current_CPI / base_CPI) * 1.03

    - 2080  = standard full-time work hours per year (52 weeks * 40 hours)
    - CPI ratio adjusts the historical wage to today's purchasing power
    - 1.03  = 3% technology sector premium over the general CPI adjustment

BLS series IDs referenced:
    - Occupational wage series: OEUM* (from the Occupational Employment Statistics)
    - CPI series: CUUR0000SA0 (All Urban Consumers, All Items)
"""

import datetime
import requests


class MarketDataCoordinator:
    """
    Orchestrates market salary lookups with a 30-day MongoDB cache.

    On each call to `get_market_intelligence`:
      1. Check MongoDB for a cached result less than 30 days old.
      2. If stale or missing, fetch live data from the BLS API and update the cache.

    Attributes:
        db (MarketDB): Connected database handler used for caching.
        api_key (str): BLS API registration key (required for higher rate limits).
        url (str): BLS v2 timeseries API endpoint.
    """

    def __init__(self, db_handler, api_key: str):
        """
        Args:
            db_handler (MarketDB): An already-connected MarketDB instance.
            api_key (str): BLS API key from environment (BLS_API_KEY).
        """
        self.db = db_handler
        self.api_key = api_key
        self.url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'

    def get_market_intelligence(self, series_id: str, role_name: str) -> dict | None:
        """
        Return the inflation-adjusted market salary for a role, using cache when fresh.

        Args:
            series_id (str): BLS Occupational Employment Statistics series ID for the role.
            role_name (str): Human-readable role label used as the MongoDB cache key.

        Returns:
            dict | None: Benchmark document with `predicted_salary_2026`, or None on failure.
        """
        # Return cached data if it was synced within the last 30 days
        cached = self.db.get_benchmark(role_name)
        if cached and 'sync_timestamp' in cached:
            if (datetime.datetime.utcnow() - cached['sync_timestamp']).days < 30:
                print(f"Cache hit: {role_name}")
                return cached

        # Cache is stale or missing — fetch fresh data from BLS
        print(f"Syncing {role_name} from BLS API...")
        data = self._fetch_and_calculate(series_id, role_name)

        if data:
            data['sync_timestamp'] = datetime.datetime.utcnow()
            self.db.upsert_benchmark(data)
            return data
        return None

    def _fetch_and_calculate(self, series_id: str, role_name: str) -> dict | None:
        """
        Pull BLS wage data, find the most recent available year, then project
        the salary forward to the current year using CPI inflation adjustment.

        Args:
            series_id (str): BLS series ID for the target occupation.
            role_name (str): Label to store alongside the result.

        Returns:
            dict | None: Document containing the projected annual salary, or None
                         if no BLS data could be retrieved.
        """
        curr_year = datetime.datetime.now().year
        anchor_year, base_h = None, 0.0

        # Walk back up to 4 years to find the most recent year with published data
        for y in range(curr_year - 1, curr_year - 5, -1):
            res = requests.post(self.url, json={
                "seriesid": [series_id],
                "startyear": str(y),
                "endyear": str(y),
                "registrationkey": self.api_key
            }).json()
            pts = res.get('Results', {}).get('series', [{}])[0].get('data', [])
            if pts:
                anchor_year = str(y)
                base_h = float(pts[0]['value'])  # hourly wage in USD
                break

        if not anchor_year:
            return None

        # Fetch the CPI value for the anchor year (baseline for inflation calculation)
        cpi_id = "CUUR0000SA0"
        b_res = requests.post(self.url, json={
            "seriesid": [cpi_id],
            "startyear": anchor_year,
            "endyear": anchor_year,
            "registrationkey": self.api_key
        }).json()
        base_cpi = float(b_res['Results']['series'][0]['data'][0]['value'])

        # Fetch the most recent CPI value for inflation adjustment to today
        n_res = requests.post(self.url, json={
            "seriesid": [cpi_id],
            "startyear": str(curr_year - 1),
            "endyear": str(curr_year),
            "registrationkey": self.api_key
        }).json()
        now_cpi = float(n_res['Results']['series'][0]['data'][0]['value'])

        # Project salary: annualize hourly wage, adjust for inflation, add tech premium
        predicted = (base_h * 2080) * (now_cpi / base_cpi) * 1.03

        return {
            "role_name": role_name,
            "series_id": series_id,
            "anchor_year": anchor_year,
            "hourly_base": base_h,
            "predicted_salary_2026": round(predicted, 2)
        }