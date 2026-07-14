"""Agent tool functions: 5 music catalog tools + 4 invoice tools."""

from src.tools.invoice_tools import (
    get_invoice_line_items,
    get_invoices_by_customer_sorted_by_date,
    get_purchased_tracks_by_customer,
    get_support_rep_by_invoice,
)
from src.tools.music_tools import (
    get_albums_by_artist,
    get_songs_by_genre,
    get_track_details,
    get_tracks_by_artist,
    search_songs_by_title,
)

MUSIC_TOOLS = [
    get_albums_by_artist,
    get_tracks_by_artist,
    get_songs_by_genre,
    search_songs_by_title,
    get_track_details,
]

INVOICE_TOOLS = [
    get_invoices_by_customer_sorted_by_date,
    get_purchased_tracks_by_customer,
    get_support_rep_by_invoice,
    get_invoice_line_items,
]

ALL_TOOLS = MUSIC_TOOLS + INVOICE_TOOLS

__all__ = [
    "MUSIC_TOOLS",
    "INVOICE_TOOLS",
    "ALL_TOOLS",
    "get_albums_by_artist",
    "get_tracks_by_artist",
    "get_songs_by_genre",
    "search_songs_by_title",
    "get_track_details",
    "get_invoices_by_customer_sorted_by_date",
    "get_purchased_tracks_by_customer",
    "get_support_rep_by_invoice",
    "get_invoice_line_items",
]
