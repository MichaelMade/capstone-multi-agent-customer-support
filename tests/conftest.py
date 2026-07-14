"""Shared pytest fixtures.

The Chinook database is loaded once per test session into the in-memory
SQLite engine. No API key is required: tests cover the deterministic
database and tool layers only.
"""

import pytest

from src.database import init_database


@pytest.fixture(scope="session", autouse=True)
def database():
    """Initialize the in-memory Chinook database once for all tests."""
    return init_database()
