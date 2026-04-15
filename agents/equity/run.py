"""
agents/equity/run.py

Entry point for syncing market salary benchmarks from the BLS API into MongoDB.

Run this script once before executing the Equity Agent to ensure the benchmark
collection is populated with up-to-date CPI-adjusted salary data.

Usage:
    python -m agents.equity.run

Environment variables required (config.env):
    BLS_API_KEY   - BLS public API registration key
    MONGODB_URI   - MongoDB connection string
    MONGODB_NAME  - Target database name
    COLLECTION_NAME - Collection for market benchmarks
"""

import os
from tools.mongoDB import MarketDB
from tools.market_logic import MarketDataCoordinator


def main():
    # Connect to MongoDB — abort early if the connection fails
    db = MarketDB()
    if not db.is_connected:
        return

    api_key = os.getenv("BLS_API_KEY")
    coordinator = MarketDataCoordinator(db, api_key)

    # BLS series IDs for Seattle-area tech roles (Occupational Employment Statistics)
    targets = [
        ("OEUM004266000000015125203", "Software_Developer_Seattle"),
        ("OEUM004266000000015205103", "Data_Scientist_Seattle"),
    ]

    print("\n--- RetentionAgent: Market Benchmark Sync ---")
    for series_id, role_name in targets:
        result = coordinator.get_market_intelligence(series_id, role_name)
        if result:
            print(f"  {role_name}: ${result['predicted_salary_2026']:,.0f} (projected 2026)")

    db.close()
    print("--- Sync complete ---")


if __name__ == "__main__":
    main()