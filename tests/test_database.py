"""Tests for the database layer: safe queries, phone normalization, health.

No tests require an API key.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import OperationalError

import src.database as database_module

from src.database import (
    find_customer_by_email,
    find_customer_by_id,
    find_customer_by_phone,
    normalize_phone,
    run_query_safe,
    verify_database,
)


class TestRunQuerySafe:
    def test_basic_select_customer_count(self):
        """The Chinook dataset must load all 59 customers."""
        result = run_query_safe("SELECT COUNT(*) AS count FROM Customer")
        rows = json.loads(result)
        assert rows[0]["count"] == 59

    def test_parameterized_query_filters_correctly(self):
        """Named bind parameters must filter results."""
        result = run_query_safe(
            "SELECT CustomerId, FirstName FROM Customer WHERE CustomerId = :cid",
            {"cid": 1},
        )
        rows = json.loads(result)
        assert len(rows) == 1
        assert rows[0]["CustomerId"] == 1
        assert rows[0]["FirstName"] == "Luís"

    def test_empty_results_return_empty_json_array(self):
        """No matches must return '[]', not an error."""
        result = run_query_safe(
            "SELECT * FROM Customer WHERE CustomerId = :cid", {"cid": -1}
        )
        assert result == "[]"

    def test_output_is_valid_json(self):
        """Any query output must be parseable JSON."""
        result = run_query_safe("SELECT * FROM Genre ORDER BY GenreId LIMIT 5")
        rows = json.loads(result)
        assert isinstance(rows, list)
        assert len(rows) == 5
        assert "Name" in rows[0]

    def test_injection_attempt_is_treated_as_data(self):
        """A classic injection payload passed as a parameter must be inert."""
        payload = "1; DROP TABLE Customer; --"
        result = run_query_safe(
            "SELECT * FROM Customer WHERE FirstName = :name", {"name": payload}
        )
        assert result == "[]"
        # Table must still exist and be intact.
        count = json.loads(run_query_safe("SELECT COUNT(*) AS c FROM Customer"))
        assert count[0]["c"] == 59

    @pytest.mark.parametrize(
        "statement",
        [
            "DROP TABLE Customer",
            "DELETE FROM Customer",
            "PRAGMA query_only = OFF",
        ],
    )
    def test_mutating_or_pragma_statements_are_refused(self, statement):
        with pytest.raises(ValueError, match="read-only SELECT"):
            run_query_safe(statement)
        # A refused statement must not alter the shared database.
        count = json.loads(run_query_safe("SELECT COUNT(*) AS c FROM Customer"))
        assert count[0]["c"] == 59

    def test_leading_sql_comments_are_allowed_for_selects(self):
        result = run_query_safe("/* health probe */\nSELECT 1 AS ok")
        assert json.loads(result) == [{"ok": 1}]

    def test_write_hidden_behind_cte_is_blocked_by_sqlite(self):
        statement = (
            "WITH doomed AS (SELECT CustomerId FROM Customer WHERE CustomerId = 1) "
            "DELETE FROM Customer WHERE CustomerId IN "
            "(SELECT CustomerId FROM doomed)"
        )
        with pytest.raises(OperationalError, match="readonly"):
            run_query_safe(statement)
        count = json.loads(run_query_safe("SELECT COUNT(*) AS c FROM Customer"))
        assert count[0]["c"] == 59

    def test_static_pool_access_is_serialized_across_threads(self):
        """Concurrent callers must never corrupt the shared cursor state."""

        def count_tracks(_):
            return json.loads(run_query_safe("SELECT COUNT(*) AS count FROM Track"))[0][
                "count"
            ]

        with ThreadPoolExecutor(max_workers=16) as pool:
            counts = list(pool.map(count_tracks, range(1_000)))
        assert counts == [3503] * 1_000


class TestSqlCache:
    def test_filename_only_cache_path_is_written_atomically(
        self, monkeypatch, tmp_path
    ):
        """A relative cache filename has no explicit parent directory."""
        script = "SELECT 1;"
        response = SimpleNamespace(
            text=script,
            raise_for_status=lambda: None,
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            database_module,
            "get_settings",
            lambda: SimpleNamespace(
                chinook_cache_path="chinook.sql",
                chinook_url="https://example.invalid/chinook.sql",
            ),
        )
        monkeypatch.setattr(database_module.requests, "get", lambda *a, **k: response)
        monkeypatch.setattr(
            database_module, "verify_sql_integrity", lambda *a, **k: None
        )

        assert database_module._download_chinook_sql() == script
        assert (tmp_path / "chinook.sql").read_text(encoding="utf-8") == script
        assert not list(tmp_path.glob(".*.tmp"))


class TestNormalizePhone:
    def test_international_format(self):
        assert normalize_phone("+1 (555) 123-4567") == "+15551234567"

    def test_domestic_with_dashes(self):
        assert normalize_phone("555-123-4567") == "5551234567"

    def test_empty_string(self):
        assert normalize_phone("") == ""

    def test_none_input(self):
        assert normalize_phone(None) == ""

    def test_plus_only_preserved_at_start(self):
        """A '+' is only meaningful as a leading character."""
        assert normalize_phone("555+123+4567") == "5551234567"
        assert normalize_phone("+55 (12) 3923-5555") == "+551239235555"


class TestVerifyDatabase:
    def test_health_check_reports_healthy(self):
        health = verify_database()
        assert health["status"] == "healthy"
        assert health["table_count"] >= 11
        assert health["customer_count"] == 59
        assert health["missing_tables"] == []


class TestCustomerLookups:
    def test_find_by_id(self):
        customer = find_customer_by_id(5)
        assert customer is not None
        assert customer["CustomerId"] == 5

    def test_find_by_email_is_case_insensitive(self):
        # Customer 1: luisg@embraer.com.br
        customer = find_customer_by_email("LUISG@EMBRAER.COM.BR")
        assert customer is not None
        assert customer["CustomerId"] == 1

    def test_find_by_phone_with_formatting_differences(self):
        # Customer 1 phone stored as +55 (12) 3923-5555
        customer = find_customer_by_phone("+55 12 3923 5555")
        assert customer is not None
        assert customer["CustomerId"] == 1
