"""Recreate the ``ajustes`` table and seed one row; standard library only."""

import os
import sqlite3
import sys

PREFIX = "sqlite:///"


def main():
    """Reset the database named by ``DB_URL`` and return the exit code."""
    url = os.environ.get("DB_URL", "")
    if not url.startswith(PREFIX):
        print("reset_db.py: DB_URL must start with sqlite:///", file=sys.stderr)
        return 2
    connection = sqlite3.connect(url[len(PREFIX) :])
    try:
        with connection:
            connection.execute("DROP TABLE IF EXISTS ajustes")
            connection.execute(
                "CREATE TABLE ajustes (id INTEGER PRIMARY KEY, tenant TEXT NOT NULL, "
                "cuenta TEXT NOT NULL, importe NUMERIC NOT NULL)"
            )
            connection.execute(
                "INSERT INTO ajustes (tenant, cuenta, importe) VALUES ('A', '001', 5)"
            )
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
