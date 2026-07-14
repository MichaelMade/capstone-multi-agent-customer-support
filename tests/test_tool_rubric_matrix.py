"""Rubric-level outcome and JSON contracts for every agent tool.

Each of the five catalog tools and four invoice tools is exercised through
its public ``invoke`` interface for a successful lookup, an empty/not-found
lookup, and invalid input.  The Chinook identifiers used for successful
lookups are stable fixtures from the bundled database.
"""

from dataclasses import dataclass
import json
from typing import Literal

from langchain_core.tools import BaseTool
import pytest

from src.tools import (
    ALL_TOOLS,
    get_albums_by_artist,
    get_invoice_line_items,
    get_invoices_by_customer_sorted_by_date,
    get_purchased_tracks_by_customer,
    get_songs_by_genre,
    get_support_rep_by_invoice,
    get_track_details,
    get_tracks_by_artist,
    search_songs_by_title,
)


Outcome = Literal["success", "not_found", "invalid"]
OUTCOMES: tuple[Outcome, ...] = ("success", "not_found", "invalid")


@dataclass(frozen=True)
class ToolOutcomeCase:
    """One deterministic tool invocation and its expected response shape."""

    tool: BaseTool
    outcome: Outcome
    arguments: dict[str, object]
    records_key: str | None = None
    top_level_fields: frozenset[str] = frozenset()
    record_fields: frozenset[str] = frozenset()
    response_key: str | None = None
    expected_fragment: str = ""


def _success(
    tool: BaseTool,
    arguments: dict[str, object],
    *,
    record_fields: set[str],
    records_key: str | None = None,
    top_level_fields: set[str] | None = None,
) -> ToolOutcomeCase:
    return ToolOutcomeCase(
        tool=tool,
        outcome="success",
        arguments=arguments,
        records_key=records_key,
        top_level_fields=frozenset(top_level_fields or set()),
        record_fields=frozenset(record_fields),
    )


def _not_found(
    tool: BaseTool,
    arguments: dict[str, object],
    expected_fragment: str,
) -> ToolOutcomeCase:
    return ToolOutcomeCase(
        tool=tool,
        outcome="not_found",
        arguments=arguments,
        response_key="message",
        expected_fragment=expected_fragment,
    )


def _invalid(
    tool: BaseTool,
    arguments: dict[str, object],
    expected_fragment: str,
) -> ToolOutcomeCase:
    return ToolOutcomeCase(
        tool=tool,
        outcome="invalid",
        arguments=arguments,
        response_key="error",
        expected_fragment=expected_fragment,
    )


TOOL_OUTCOME_CASES = [
    # Music catalog tools.
    _success(
        get_albums_by_artist,
        {"artist_name": "AC/DC"},
        record_fields={"AlbumId", "AlbumTitle", "ArtistName"},
    ),
    _not_found(
        get_albums_by_artist,
        {"artist_name": "NoSuchArtist_Rubric_Matrix"},
        "No albums found",
    ),
    _invalid(get_albums_by_artist, {"artist_name": "   "}, "cannot be blank"),
    _success(
        get_tracks_by_artist,
        {"artist_name": "AC/DC"},
        records_key="tracks",
        top_level_fields={"total_tracks", "sample_size", "tracks"},
        record_fields={"TrackId", "TrackName", "AlbumTitle", "ArtistName"},
    ),
    _not_found(
        get_tracks_by_artist,
        {"artist_name": "NoSuchArtist_Rubric_Matrix"},
        "No tracks found",
    ),
    _invalid(get_tracks_by_artist, {"artist_name": ""}, "cannot be blank"),
    _success(
        get_songs_by_genre,
        {"genre_name": "Rock"},
        records_key="tracks",
        top_level_fields={
            "total_tracks_in_genre",
            "total_artists_in_genre",
            "sample_size",
            "tracks",
        },
        record_fields={"TrackId", "TrackName", "ArtistName", "GenreName"},
    ),
    _not_found(
        get_songs_by_genre,
        {"genre_name": "NoSuchGenre_Rubric_Matrix"},
        "No songs found",
    ),
    _invalid(get_songs_by_genre, {"genre_name": "\t"}, "cannot be blank"),
    _success(
        search_songs_by_title,
        {"title": "Thunder"},
        records_key="tracks",
        top_level_fields={"total_matches", "sample_size", "tracks"},
        record_fields={"TrackId", "TrackName", "ArtistName", "AlbumTitle"},
    ),
    _not_found(
        search_songs_by_title,
        {"title": "NoSuchSong_Rubric_Matrix"},
        "No songs found",
    ),
    _invalid(search_songs_by_title, {"title": " "}, "cannot be blank"),
    _success(
        get_track_details,
        {"track_id": "1"},
        record_fields={"TrackId", "TrackName", "AlbumTitle", "ArtistName"},
    ),
    _not_found(
        get_track_details,
        {"track_id": "999999"},
        "No track found",
    ),
    _invalid(get_track_details, {"track_id": "abc"}, "Invalid track ID"),
    # Invoice tools. Invoice 1 belongs to customer 2 in Chinook.
    _success(
        get_invoices_by_customer_sorted_by_date,
        {"customer_id": "1"},
        record_fields={"InvoiceId", "InvoiceDate", "BillingCountry", "Total"},
    ),
    _not_found(
        get_invoices_by_customer_sorted_by_date,
        {"customer_id": "99999"},
        "No invoices found",
    ),
    _invalid(
        get_invoices_by_customer_sorted_by_date,
        {"customer_id": "abc"},
        "Invalid customer ID",
    ),
    _success(
        get_purchased_tracks_by_customer,
        {"customer_id": "1"},
        record_fields={"TrackName", "ArtistName", "UnitPrice", "InvoiceId"},
    ),
    _not_found(
        get_purchased_tracks_by_customer,
        {"customer_id": "99999"},
        "No purchased tracks found",
    ),
    _invalid(
        get_purchased_tracks_by_customer,
        {"customer_id": "abc"},
        "Invalid customer ID",
    ),
    _success(
        get_support_rep_by_invoice,
        {"invoice_id": "1", "customer_id": "2"},
        record_fields={"InvoiceId", "CustomerId", "SupportRepName"},
    ),
    _not_found(
        get_support_rep_by_invoice,
        {"invoice_id": "99999", "customer_id": "2"},
        "No invoice with ID 99999 found",
    ),
    _invalid(
        get_support_rep_by_invoice,
        {"invoice_id": "abc", "customer_id": "2"},
        "Invalid invoice ID",
    ),
    _success(
        get_invoice_line_items,
        {"invoice_id": "1", "customer_id": "2"},
        record_fields={"TrackName", "ArtistName", "UnitPrice", "Quantity"},
    ),
    _not_found(
        get_invoice_line_items,
        {"invoice_id": "99999", "customer_id": "2"},
        "No line items found",
    ),
    _invalid(
        get_invoice_line_items,
        {"invoice_id": "abc", "customer_id": "2"},
        "Invalid invoice ID",
    ),
]


def test_matrix_has_each_outcome_exactly_once_for_every_registered_tool():
    """Prevent an apparently broad matrix from silently omitting a tool case."""
    expected = {(tool.name, outcome) for tool in ALL_TOOLS for outcome in OUTCOMES}
    observed = [(case.tool.name, case.outcome) for case in TOOL_OUTCOME_CASES]

    assert len(observed) == len(set(observed)), "duplicate tool/outcome matrix case"
    assert set(observed) == expected


@pytest.mark.parametrize(
    "case",
    TOOL_OUTCOME_CASES,
    ids=lambda case: f"{case.tool.name}-{case.outcome}",
)
def test_tool_outcome_is_valid_json_with_expected_contract(case: ToolOutcomeCase):
    raw_result = case.tool.invoke(case.arguments)

    assert isinstance(raw_result, str)
    payload = json.loads(raw_result)

    if case.outcome == "success":
        if case.records_key is None:
            records = payload
        else:
            assert isinstance(payload, dict)
            assert case.top_level_fields <= payload.keys()
            records = payload[case.records_key]

        assert isinstance(records, list)
        assert records, "success fixture must return at least one database row"
        assert isinstance(records[0], dict)
        assert case.record_fields <= records[0].keys()
        return

    assert isinstance(payload, dict)
    assert case.response_key is not None
    response = payload.get(case.response_key)
    assert isinstance(response, str)
    assert case.expected_fragment.casefold() in response.casefold()
