"""
tools/mongoDB.py

MongoDB connection handler for the RetentionAgent system.
Manages read/write access to the market salary benchmark collection,
which is shared across the Equity Agent and Retention Agent.

Environment variables required (config.env):
    MONGODB_URI      - MongoDB Atlas connection string
    MONGODB_NAME     - Target database name
    COLLECTION_NAME  - Collection storing market salary benchmarks
"""
from __future__ import annotations

import os
import sys
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi
from pymongo.errors import PyMongoError

load_dotenv("config.env")


class MarketDB:
    """
    Thin wrapper around a MongoDB connection for market salary data.

    Attributes:
        uri (str): MongoDB connection URI loaded from environment.
        db_name (str): Target database name.
        collection_name (str): Name of the benchmark collection.
        client (MongoClient | None): Active PyMongo client, or None if connection failed.
        db: PyMongo database handle.
        collection: PyMongo collection handle for salary benchmarks.
        is_connected (bool): True if the connection ping succeeded.
    """

    def __init__(self):
        self.uri = os.getenv("MONGODB_URI")
        self.db_name = os.getenv("MONGODB_NAME")
        self.collection_name = os.getenv("COLLECTION_NAME")
        self.client = None
        self.is_connected = False

        try:
            if not self.uri:
                print("CONFIG ERROR: MONGODB_URI is missing in config.env")
                return

            # Use certifi's CA bundle to satisfy Atlas TLS requirements
            self.client = MongoClient(
                self.uri,
                serverSelectionTimeoutMS=5000,
                tz_aware=True,
                tlsCAFile=certifi.where()
            )

            # Ping the server to verify the connection before proceeding
            self.client.admin.command('hello')

            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            self.is_connected = True
            print(f"MongoDB connected: {self.db_name}.{self.collection_name}")

        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            self.is_connected = False

    def upsert_benchmark(self, doc: dict) -> bool:
        """
        Insert or update a market salary benchmark document.

        Matches on `role_name`. If the role already exists, its fields are
        updated in place; otherwise a new document is created.

        Args:
            doc (dict): Benchmark data. Must contain a `role_name` key.

        Returns:
            bool: True on success, False on database error or no connection.
        """
        if not self.is_connected:
            return False
        try:
            result = self.collection.update_one(
                {"role_name": doc["role_name"]},
                {"$set": doc},
                upsert=True
            )
            if result.upserted_id:
                print(f"New benchmark created: {doc['role_name']}")
            else:
                print(f"Benchmark updated: {doc['role_name']}")
            return True
        except PyMongoError as e:
            print(f"Upsert failed: {e}")
            return False

    def get_benchmark(self, role_name: str) -> dict | None:
        """
        Retrieve the market salary benchmark for a given role.

        Args:
            role_name (str): The job role to look up (e.g. "Software_Developer_Seattle").

        Returns:
            dict | None: The matching MongoDB document, or None if not found or on error.
        """
        if not self.is_connected:
            return None
        try:
            return self.collection.find_one({"role_name": role_name})
        except PyMongoError as e:
            print(f"Query failed: {e}")
            return None

    def close(self):
        """Close the MongoDB client connection."""
        if self.client:
            self.client.close()
            print("MongoDB connection closed.")