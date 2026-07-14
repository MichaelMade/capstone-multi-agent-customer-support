"""Tests for all 9 agent tools: found, not-found, and error cases,
plus determinism, ordering, diversity, and JSON validity.

No tests require an API key.
"""

import json

import pytest

from src.tools import (
    ALL_TOOLS,
    INVOICE_TOOLS,
    MUSIC_TOOLS,
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


# ---------------------------------------------------------------------------
# Music tools
# ---------------------------------------------------------------------------


class TestGetAlbumsByArtist:
    def test_found(self):
        result = get_albums_by_artist.invoke({"artist_name": "AC/DC"})
        rows = json.loads(result)
        assert isinstance(rows, list) and len(rows) >= 1
        titles = [r["AlbumTitle"] for r in rows]
        assert "For Those About To Rock We Salute You" in titles

    def test_fuzzy_partial_match(self):
        """Partial, differently-cased input must still match."""
        result = get_albums_by_artist.invoke({"artist_name": "zeppelin"})
        rows = json.loads(result)
        assert isinstance(rows, list) and len(rows) >= 1
        assert all("Zeppelin" in r["ArtistName"] for r in rows)
        assert any(r["ArtistName"] == "Led Zeppelin" for r in rows)

    def test_not_found(self):
        result = get_albums_by_artist.invoke({"artist_name": "NonexistentArtistXYZ"})
        payload = json.loads(result)
        assert "message" in payload
        assert "No albums found" in payload["message"]

    def test_blank_artist_is_rejected(self):
        payload = json.loads(get_albums_by_artist.invoke({"artist_name": "   "}))
        assert "cannot be blank" in payload["error"]

    def test_like_wildcards_are_literal(self):
        """A '%' in user input must not act as a catalog-wide wildcard."""
        payload = json.loads(get_albums_by_artist.invoke({"artist_name": "AC%DC"}))
        assert "No albums found" in payload["message"]


class TestGetTracksByArtist:
    def test_found_with_count_and_sample(self):
        result = get_tracks_by_artist.invoke({"artist_name": "AC/DC"})
        payload = json.loads(result)
        assert payload["total_tracks"] == 18
        assert 1 <= payload["sample_size"] <= 20
        assert len(payload["tracks"]) == payload["sample_size"]

    def test_sample_capped_at_20_with_truncation_note(self):
        result = get_tracks_by_artist.invoke({"artist_name": "Iron Maiden"})
        payload = json.loads(result)
        assert payload["total_tracks"] > 20
        assert payload["sample_size"] == 20
        assert "Showing 20 of" in payload["note"]

    def test_not_found(self):
        result = get_tracks_by_artist.invoke({"artist_name": "NonexistentArtistXYZ"})
        payload = json.loads(result)
        assert "No tracks found" in payload["message"]

    def test_blank_artist_is_rejected(self):
        payload = json.loads(get_tracks_by_artist.invoke({"artist_name": ""}))
        assert "cannot be blank" in payload["error"]


class TestGetSongsByGenre:
    def test_found(self):
        result = get_songs_by_genre.invoke({"genre_name": "Rock"})
        payload = json.loads(result)
        assert payload["total_tracks_in_genre"] > 0
        assert len(payload["tracks"]) >= 1

    def test_reports_artist_and_track_totals_in_matching_units(self):
        payload = json.loads(get_songs_by_genre.invoke({"genre_name": "Rock"}))
        assert payload["total_tracks_in_genre"] == 1309
        assert payload["total_artists_in_genre"] == 52
        assert "10 of 52 matching artists" in payload["note"]
        assert "1309 tracks total" in payload["note"]

    def test_deterministic_results(self):
        """Repeated calls must return byte-identical output."""
        first = get_songs_by_genre.invoke({"genre_name": "Rock"})
        second = get_songs_by_genre.invoke({"genre_name": "Rock"})
        assert first == second

    def test_artist_diversity(self):
        """The sample must span different artists, one track per artist."""
        payload = json.loads(get_songs_by_genre.invoke({"genre_name": "Rock"}))
        artists = [t["ArtistName"] for t in payload["tracks"]]
        assert len(artists) == len(set(artists)), "each artist at most once"
        assert len(set(artists)) >= 2, "sample must span multiple artists"

    def test_not_found(self):
        result = get_songs_by_genre.invoke({"genre_name": "Polka"})
        payload = json.loads(result)
        assert "No songs found" in payload["message"]

    def test_blank_genre_is_rejected(self):
        payload = json.loads(get_songs_by_genre.invoke({"genre_name": " \t "}))
        assert "cannot be blank" in payload["error"]


class TestSearchSongsByTitle:
    def test_found_fuzzy(self):
        result = search_songs_by_title.invoke({"title": "thunder"})
        payload = json.loads(result)
        assert payload["total_matches"] >= 1
        names = [t["TrackName"] for t in payload["tracks"]]
        assert any("Thunder" in n for n in names)

    def test_sample_capped_at_10(self):
        result = search_songs_by_title.invoke({"title": "love"})
        payload = json.loads(result)
        assert payload["total_matches"] > 10
        assert payload["sample_size"] == 10

    def test_not_found(self):
        result = search_songs_by_title.invoke({"title": "zzzznonexistentsongzzz"})
        payload = json.loads(result)
        assert "No songs found" in payload["message"]

    def test_blank_title_is_rejected(self):
        payload = json.loads(search_songs_by_title.invoke({"title": ""}))
        assert "cannot be blank" in payload["error"]

    def test_like_wildcard_does_not_match_every_title(self):
        payload = json.loads(search_songs_by_title.invoke({"title": "%"}))
        assert payload["total_matches"] == 2
        assert all("%" in track["TrackName"] for track in payload["tracks"])


class TestGetTrackDetails:
    def test_found(self):
        result = get_track_details.invoke({"track_id": "1"})
        rows = json.loads(result)
        assert rows[0]["TrackId"] == 1
        assert rows[0]["TrackName"] == "For Those About To Rock (We Salute You)"
        assert rows[0]["ArtistName"] == "AC/DC"

    def test_invalid_non_numeric_id(self):
        """Non-numeric input must return an error message, not crash."""
        result = get_track_details.invoke({"track_id": "abc"})
        payload = json.loads(result)
        assert "error" in payload
        assert "Invalid track ID" in payload["error"]

    def test_nonexistent_id(self):
        result = get_track_details.invoke({"track_id": "999999"})
        payload = json.loads(result)
        assert "No track found" in payload["message"]

    @pytest.mark.parametrize(
        "track_id",
        ["0", "-1", "+1", "01", "1_0", "9223372036854775808", "9" * 100],
    )
    def test_noncanonical_or_out_of_range_id_is_rejected(self, track_id):
        result = get_track_details.invoke({"track_id": track_id})
        payload = json.loads(result)
        assert "Invalid track ID" in payload["error"]

    def test_maximum_sqlite_integer_does_not_overflow(self):
        result = get_track_details.invoke({"track_id": "9223372036854775807"})
        payload = json.loads(result)
        assert "No track found" in payload["message"]


# ---------------------------------------------------------------------------
# Invoice tools
# ---------------------------------------------------------------------------


class TestGetInvoicesByCustomer:
    def test_found_sorted_by_date_desc(self):
        result = get_invoices_by_customer_sorted_by_date.invoke({"customer_id": "1"})
        rows = json.loads(result)
        assert len(rows) == 7  # customer 1 has 7 invoices in Chinook
        dates = [r["InvoiceDate"] for r in rows]
        assert dates == sorted(dates, reverse=True)

    def test_invalid_non_numeric_id(self):
        result = get_invoices_by_customer_sorted_by_date.invoke({"customer_id": "abc"})
        payload = json.loads(result)
        assert "Invalid customer ID" in payload["error"]

    def test_nonexistent_customer(self):
        result = get_invoices_by_customer_sorted_by_date.invoke(
            {"customer_id": "99999"}
        )
        payload = json.loads(result)
        assert "No invoices found" in payload["message"]


class TestGetPurchasedTracks:
    def test_found_sorted_by_price_desc(self):
        result = get_purchased_tracks_by_customer.invoke({"customer_id": "1"})
        rows = json.loads(result)
        assert len(rows) >= 1
        prices = [r["UnitPrice"] for r in rows]
        assert prices == sorted(prices, reverse=True)
        # Each row is one purchased track, not an invoice.
        assert {"TrackName", "UnitPrice", "InvoiceId"} <= set(rows[0].keys())

    def test_invalid_non_numeric_id(self):
        result = get_purchased_tracks_by_customer.invoke({"customer_id": "x!"})
        payload = json.loads(result)
        assert "Invalid customer ID" in payload["error"]


class TestGetSupportRepByInvoice:
    # Invoice 1 belongs to customer 2 in Chinook.
    def test_found_for_owner(self):
        result = get_support_rep_by_invoice.invoke(
            {"invoice_id": "1", "customer_id": "2"}
        )
        rows = json.loads(result)
        assert rows[0]["InvoiceId"] == 1
        assert rows[0]["SupportRepName"]  # a rep is assigned
        assert "SupportRepEmail" in rows[0]

    def test_nonexistent_invoice(self):
        result = get_support_rep_by_invoice.invoke(
            {"invoice_id": "99999", "customer_id": "2"}
        )
        payload = json.loads(result)
        assert "No invoice" in payload["message"]

    def test_invalid_non_numeric_id(self):
        result = get_support_rep_by_invoice.invoke(
            {"invoice_id": "abc", "customer_id": "2"}
        )
        payload = json.loads(result)
        assert "Invalid invoice ID" in payload["error"]


class TestGetInvoiceLineItems:
    def test_found_for_owner(self):
        result = get_invoice_line_items.invoke({"invoice_id": "1", "customer_id": "2"})
        rows = json.loads(result)
        assert len(rows) == 2  # invoice 1 has 2 line items in Chinook
        assert {"TrackName", "UnitPrice", "Quantity"} <= set(rows[0].keys())

    def test_nonexistent_invoice(self):
        result = get_invoice_line_items.invoke(
            {"invoice_id": "99999", "customer_id": "2"}
        )
        payload = json.loads(result)
        assert "No line items found" in payload["message"]


# ---------------------------------------------------------------------------
# Cross-cutting: every tool's output must be valid JSON
# ---------------------------------------------------------------------------

TOOL_SAMPLE_ARGS = [
    (get_albums_by_artist, {"artist_name": "AC/DC"}),
    (get_tracks_by_artist, {"artist_name": "Queen"}),
    (get_songs_by_genre, {"genre_name": "Jazz"}),
    (search_songs_by_title, {"title": "rock"}),
    (get_track_details, {"track_id": "10"}),
    (get_invoices_by_customer_sorted_by_date, {"customer_id": "5"}),
    (get_purchased_tracks_by_customer, {"customer_id": "5"}),
    # Invoice 5 belongs to customer 23 in Chinook.
    (get_support_rep_by_invoice, {"invoice_id": "5", "customer_id": "23"}),
    (get_invoice_line_items, {"invoice_id": "5", "customer_id": "23"}),
]


@pytest.mark.parametrize(
    "tool_fn,args",
    TOOL_SAMPLE_ARGS,
    ids=[t[0].name for t in TOOL_SAMPLE_ARGS],
)
def test_all_tool_outputs_are_valid_json(tool_fn, args):
    """All 9 tools must always return parseable JSON strings."""
    result = tool_fn.invoke(args)
    assert isinstance(result, str)
    json.loads(result)  # raises on invalid JSON


def test_all_nine_tools_registered():
    """The agent toolkits must expose exactly 5 music + 4 invoice tools."""
    assert len(MUSIC_TOOLS) == 5
    assert len(INVOICE_TOOLS) == 4
    assert len(ALL_TOOLS) == 9
    assert ALL_TOOLS == MUSIC_TOOLS + INVOICE_TOOLS
