"""
scripts/database_mysql.py

MySQL database manager for the RetentionAgent web backend.

Provides a connection pool backed by `mysql-connector-python` and thin
wrapper methods for executing writes and reads. Each public method opens a
fresh connection from the pool, runs the query, commits (or rolls back on
error), and returns the connection to the pool — no persistent connection
state is held between calls.

SSL/TLS support is included for Aiven-hosted MySQL (requires the CA
certificate content to be supplied as an environment variable).

Environment variables required (config.env):
    DB_HOST              - MySQL hostname
    DB_DATABASE          - Database name
    DB_USER              - Database username
    DB_PASSWORD          - Database password
    DB_PORT              - Port number (default: 5432)
    DB_SSL_CA_CONTENT    - PEM-encoded CA certificate content for TLS
    DB_SSL_VERIFY_CERT   - Whether to verify the server's TLS certificate
"""

import mysql.connector
from mysql.connector import Error
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DatabaseManager:
    """
    Connection-pooled MySQL client.

    A new connection pool is created at initialization and reused across
    all method calls. The pool size is set to 3, which is sufficient for
    the low-concurrency web backend.

    Attributes:
        config (dict): mysql-connector connection parameters, including
                       pool name/size and TLS settings.
    """

    def __init__(self):
        """
        Load database credentials from config.env and initialize the connection pool.

        The SSL CA certificate is written to a temporary file because
        mysql-connector requires a file path rather than inline PEM content.
        """
        load_dotenv(_PROJECT_ROOT / "config.env")

        ssl_ca_content = os.environ.get('DB_SSL_CA_CONTENT')
        ssl_ca_path = None
        if ssl_ca_content:
            # Write inline certificate content to a temp file for the connector
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False)
            # Ensure PEM ends with a newline — required by OpenSSL's parser
            temp_file.write(ssl_ca_content if ssl_ca_content.endswith('\n') else ssl_ca_content + '\n')
            temp_file.close()
            ssl_ca_path = temp_file.name

        DB_CONFIG = {
            'host':             os.environ.get('DB_HOST'),
            'database':         os.environ.get('DB_DATABASE'),
            'user':             os.environ.get('DB_USER'),
            'password':         os.environ.get('DB_PASSWORD'),
            'port':             int(os.environ.get('DB_PORT', 5432)),
            'ssl_ca':           ssl_ca_path,
            'ssl_verify_cert':  bool(os.environ.get('DB_SSL_VERIFY_CERT')) == True,
        }

        self.config = DB_CONFIG.copy()
        # Additional settings required for Aiven MySQL compatibility
        self.config.update({
            'use_pure':   True,        # Pure Python driver (avoids C extension issues)
            'autocommit': False,       # Explicit commit for all writes
            'pool_name':  'mypool',
            'pool_size':  3,
        })

    def get_connection(self):
        """
        Acquire a connection from the pool.

        Returns:
            mysql.connector.connection.MySQLConnection | None:
                An active connection, or None if the pool is exhausted or the
                host is unreachable.
        """
        try:
            return mysql.connector.connect(**self.config)
        except Error as e:
            print(f"Connection failed: {e}")
            return None

    def connect(self) -> bool:
        """
        Test the database connection and immediately release it.

        Returns:
            bool: True if a connection could be established, False otherwise.
        """
        conn = self.get_connection()
        if conn and conn.is_connected():
            print("Database connection successful.")
            conn.close()
            return True
        return False

    def execute_sql(self, sql: str, params: tuple = None) -> bool:
        """
        Execute a write statement (INSERT, UPDATE, DELETE, DDL) with auto-commit.

        Rolls back the transaction automatically on error.

        Args:
            sql (str): SQL statement to execute.
            params (tuple | None): Parameterized query values (prevents SQL injection).

        Returns:
            bool: True on successful commit, False on any error.
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            if not conn:
                return False

            cursor = conn.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)

            conn.commit()
            return True

        except Error as e:
            print(f"Execute failed: {e}")
            print(f"SQL (truncated): {sql[:100]}...")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return False

        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def fetch_all(self, query: str, params: tuple = None) -> list | None:
        """
        Execute a SELECT query and return all matching rows.

        Args:
            query (str): SQL SELECT statement.
            params (tuple | None): Parameterized query values.

        Returns:
            list | None: List of row tuples, or None on error.
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            if not conn:
                return None

            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            return cursor.fetchall()

        except Error as e:
            print(f"Query failed: {e}")
            return None

        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def fetch_one(self, query: str, params: tuple = None) -> tuple | None:
        """
        Execute a SELECT query and return the first matching row.

        Args:
            query (str): SQL SELECT statement.
            params (tuple | None): Parameterized query values.

        Returns:
            tuple | None: First result row, or None if no rows matched or on error.
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            if not conn:
                return None

            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            return cursor.fetchone()

        except Error as e:
            print(f"Query failed: {e}")
            return None

        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def close(self):
        """Log a completion message (pool connections are managed per-call)."""
        print("Database operations completed.")