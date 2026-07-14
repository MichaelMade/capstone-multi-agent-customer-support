"""Music catalog tools (5).

Every tool: parameterized SQL only, JSON string output, friendly
not-found messages, validated numeric inputs, deterministic ordering.
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

from src.database import run_query_safe
from src.tools.common import audit_tool_call, json_error, json_message, safe_int

logger = logging.getLogger(__name__)

SAMPLE_LIMIT_TRACKS = 20
SAMPLE_LIMIT_TITLE = 10
SAMPLE_LIMIT_GENRE_ARTISTS = 10


def _literal_search_pattern(value: str, field_name: str) -> tuple[str, str]:
    """Return a trimmed term and escaped SQLite LIKE substring pattern."""
    term = value.strip()
    if not term:
        raise ValueError(f"{field_name} cannot be blank.")
    # ``!`` is our explicit LIKE escape character.  Escape it first, then
    # make user-supplied '%' and '_' ordinary literal characters.
    escaped = term.replace("!", "!!").replace("%", "!%").replace("_", "!_")
    return term, f"%{escaped}%"


@tool
@audit_tool_call
def get_albums_by_artist(artist_name: str) -> str:
    """Search for albums by artist name using fuzzy (substring) matching.

    Args:
        artist_name: Full or partial artist name, e.g. "AC/DC" or "zeppelin".

    Returns:
        JSON array of albums with AlbumId, Title, and ArtistName, or a
        not-found message.
    """
    try:
        artist, pattern = _literal_search_pattern(artist_name, "Artist name")
    except ValueError as exc:
        return json_error(str(exc))

    query = """
        SELECT al.AlbumId, al.Title AS AlbumTitle, ar.Name AS ArtistName
        FROM Album al
        JOIN Artist ar ON ar.ArtistId = al.ArtistId
        WHERE ar.Name LIKE :artist ESCAPE '!'
        ORDER BY ar.Name, al.AlbumId
    """
    result = run_query_safe(query, {"artist": pattern})
    if result == "[]":
        return json_message(f"No albums found for artist: {artist}")
    return result


@tool
@audit_tool_call
def get_tracks_by_artist(artist_name: str) -> str:
    """Search for tracks by artist name with a total count and a sample.

    Args:
        artist_name: Full or partial artist name (fuzzy matched).

    Returns:
        JSON object with the total track count and a sample of up to 20
        tracks (name, album, unit price), or a not-found message.
    """
    try:
        artist, pattern = _literal_search_pattern(artist_name, "Artist name")
    except ValueError as exc:
        return json_error(str(exc))

    count_rows = json.loads(
        run_query_safe(
            """
            SELECT COUNT(*) AS total
            FROM Track t
            JOIN Album al ON al.AlbumId = t.AlbumId
            JOIN Artist ar ON ar.ArtistId = al.ArtistId
            WHERE ar.Name LIKE :artist ESCAPE '!'
            """,
            {"artist": pattern},
        )
    )
    total = count_rows[0]["total"] if count_rows else 0
    if total == 0:
        return json_message(f"No tracks found for artist: {artist_name}")

    sample = json.loads(
        run_query_safe(
            """
            SELECT t.TrackId, t.Name AS TrackName, al.Title AS AlbumTitle,
                   ar.Name AS ArtistName, t.UnitPrice
            FROM Track t
            JOIN Album al ON al.AlbumId = t.AlbumId
            JOIN Artist ar ON ar.ArtistId = al.ArtistId
            WHERE ar.Name LIKE :artist ESCAPE '!'
            ORDER BY t.TrackId
            LIMIT :limit
            """,
            {"artist": pattern, "limit": SAMPLE_LIMIT_TRACKS},
        )
    )
    return json.dumps(
        {
            "artist_query": artist,
            "total_tracks": total,
            "sample_size": len(sample),
            "note": (
                f"Showing {len(sample)} of {total} tracks."
                if total > len(sample)
                else "All matching tracks shown."
            ),
            "tracks": sample,
        },
        ensure_ascii=False,
    )


@tool
@audit_tool_call
def get_songs_by_genre(genre_name: str) -> str:
    """Browse songs in a genre with a diverse, deterministic artist sample.

    Returns one representative track per artist (up to 10 artists) so the
    sample spans different artists rather than one artist's catalog.
    Results are deterministic: repeated calls return identical output.

    Args:
        genre_name: Full or partial genre name, e.g. "Rock" or "jazz".

    Returns:
        JSON object with the total track count for the genre and the
        representative sample, or a not-found message.
    """
    try:
        genre, pattern = _literal_search_pattern(genre_name, "Genre name")
    except ValueError as exc:
        return json_error(str(exc))

    count_rows = json.loads(
        run_query_safe(
            """
            SELECT COUNT(*) AS total,
                   COUNT(DISTINCT ar.ArtistId) AS total_artists
            FROM Track t
            JOIN Genre g ON g.GenreId = t.GenreId
            LEFT JOIN Album al ON al.AlbumId = t.AlbumId
            LEFT JOIN Artist ar ON ar.ArtistId = al.ArtistId
            WHERE g.Name LIKE :genre ESCAPE '!'
            """,
            {"genre": pattern},
        )
    )
    total = count_rows[0]["total"] if count_rows else 0
    total_artists = count_rows[0]["total_artists"] if count_rows else 0
    if total == 0:
        return json_message(f"No songs found for genre: {genre_name}")

    sample = json.loads(
        run_query_safe(
            """
            WITH ranked AS (
                SELECT t.TrackId, t.Name AS TrackName, ar.Name AS ArtistName,
                       g.Name AS GenreName, t.UnitPrice,
                       ROW_NUMBER() OVER (
                           PARTITION BY ar.ArtistId ORDER BY t.TrackId
                       ) AS rn
                FROM Track t
                JOIN Genre g ON g.GenreId = t.GenreId
                JOIN Album al ON al.AlbumId = t.AlbumId
                JOIN Artist ar ON ar.ArtistId = al.ArtistId
                WHERE g.Name LIKE :genre ESCAPE '!'
            )
            SELECT TrackId, TrackName, ArtistName, GenreName, UnitPrice
            FROM ranked
            WHERE rn = 1
            ORDER BY TrackId
            LIMIT :limit
            """,
            {"genre": pattern, "limit": SAMPLE_LIMIT_GENRE_ARTISTS},
        )
    )
    return json.dumps(
        {
            "genre_query": genre,
            "total_tracks_in_genre": total,
            "total_artists_in_genre": total_artists,
            "sample_size": len(sample),
            "note": (
                "Representative sample: one track per artist; "
                f"showing {len(sample)} of {total_artists} matching artists "
                f"({total} tracks total)."
            ),
            "tracks": sample,
        },
        ensure_ascii=False,
    )


@tool
@audit_tool_call
def search_songs_by_title(title: str) -> str:
    """Search for songs by title using fuzzy (substring) matching.

    Args:
        title: Full or partial song title, e.g. "Thunderstruck" or "love".

    Returns:
        JSON object with the total match count and up to 10 matching
        tracks (title, artist, album), or a not-found message.
    """
    try:
        q, pattern = _literal_search_pattern(title, "Song title")
    except ValueError as exc:
        return json_error(str(exc))

    count_rows = json.loads(
        run_query_safe(
            "SELECT COUNT(*) AS total FROM Track WHERE Name LIKE :title ESCAPE '!'",
            {"title": pattern},
        )
    )
    total = count_rows[0]["total"] if count_rows else 0
    if total == 0:
        return json_message(f"No songs found matching title: {title}")

    sample = json.loads(
        run_query_safe(
            """
            SELECT t.TrackId, t.Name AS TrackName, ar.Name AS ArtistName,
                   al.Title AS AlbumTitle, t.UnitPrice
            FROM Track t
            LEFT JOIN Album al ON al.AlbumId = t.AlbumId
            LEFT JOIN Artist ar ON ar.ArtistId = al.ArtistId
            WHERE t.Name LIKE :title ESCAPE '!'
            ORDER BY t.TrackId
            LIMIT :limit
            """,
            {"title": pattern, "limit": SAMPLE_LIMIT_TITLE},
        )
    )
    return json.dumps(
        {
            "title_query": q,
            "total_matches": total,
            "sample_size": len(sample),
            "note": (
                f"Showing {len(sample)} of {total} matches."
                if total > len(sample)
                else "All matches shown."
            ),
            "tracks": sample,
        },
        ensure_ascii=False,
    )


@tool
@audit_tool_call
def get_track_details(track_id: str) -> str:
    """Get complete details for a specific track by its numeric TrackId.

    Args:
        track_id: The numeric track ID as a string, e.g. "1".

    Returns:
        JSON object with full track details (name, album, artist, genre,
        media type, composer, duration, size, price), a not-found message,
        or a clear error for non-numeric input.
    """
    try:
        tid = safe_int(track_id, "track ID")
    except ValueError as exc:
        return json_error(str(exc))

    result = run_query_safe(
        """
        SELECT t.TrackId, t.Name AS TrackName, al.Title AS AlbumTitle,
               ar.Name AS ArtistName, g.Name AS GenreName,
               mt.Name AS MediaType, t.Composer,
               t.Milliseconds, t.Bytes, t.UnitPrice
        FROM Track t
        LEFT JOIN Album al ON al.AlbumId = t.AlbumId
        LEFT JOIN Artist ar ON ar.ArtistId = al.ArtistId
        LEFT JOIN Genre g ON g.GenreId = t.GenreId
        LEFT JOIN MediaType mt ON mt.MediaTypeId = t.MediaTypeId
        WHERE t.TrackId = :tid
        """,
        {"tid": tid},
    )
    if result == "[]":
        return json_message(f"No track found with ID: {tid}")
    return result
