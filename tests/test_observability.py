"""Privacy-safe logging contracts for tools and database access."""

import logging

from src.database import run_query_safe
from src.tools import get_albums_by_artist


def test_tool_audit_log_records_outcome_without_recording_arguments(caplog):
    private_search = "PrivateCustomerSearchValue"

    with caplog.at_level(logging.INFO, logger="src.tools.common"):
        get_albums_by_artist.invoke({"artist_name": private_search})

    log_text = caplog.text
    assert "name=get_albums_by_artist" in log_text
    assert "outcome=not_found" in log_text
    assert private_search not in log_text


def test_database_log_records_bind_names_without_recording_values(caplog):
    private_email = "private.customer@example.invalid"

    with caplog.at_level(logging.DEBUG, logger="src.database"):
        run_query_safe(
            "SELECT CustomerId FROM Customer WHERE Email = :email",
            {"email": private_email},
        )

    log_text = caplog.text
    assert "type=SELECT" in log_text
    assert "bind_names=['email']" in log_text
    assert "rows=0" in log_text
    assert private_email not in log_text
