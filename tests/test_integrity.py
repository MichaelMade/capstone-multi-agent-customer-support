"""Tests for the Chinook SQL integrity pin (no API key required)."""

import hashlib

import pytest

from src.database import verify_sql_integrity


class TestSqlIntegrity:
    def test_matching_hash_passes(self):
        script = "SELECT 1;"
        digest = hashlib.sha256(script.encode("utf-8")).hexdigest()
        verify_sql_integrity(script, expected_sha256=digest)  # no raise

    def test_mismatch_fails_closed(self):
        with pytest.raises(RuntimeError, match="integrity check failed"):
            verify_sql_integrity("SELECT 1;", expected_sha256="0" * 64)

    def test_skip_sentinel_disables_check(self):
        verify_sql_integrity("anything", expected_sha256="skip")  # no raise
