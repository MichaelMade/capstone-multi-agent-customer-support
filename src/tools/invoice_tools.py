"""Invoice tools (4) — identity enforced at the tool layer.

Security model (defense in depth):
- ``customer_id`` is NOT an LLM-facing parameter. It is injected from
  graph state via ``InjectedState`` — the model cannot pass, alter, or
  hallucinate it, and prompt injection cannot override it. Only the
  verification node writes that state field.
- If no verified customer is in state, every tool refuses.
- Invoice-scoped lookups (support rep, line items) are ownership-
  filtered: ``Invoice.CustomerId`` must match the verified customer, so
  one customer can never read another customer's invoice data (no IDOR).
- All queries are parameterized; numeric inputs validated by safe_int.
"""

from __future__ import annotations

import logging
from typing import Annotated, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from src.database import run_query_safe
from src.tools.common import audit_tool_call, json_error, json_message, safe_int

logger = logging.getLogger(__name__)

NOT_VERIFIED_MSG = (
    "Identity not verified. The customer must verify their identity "
    "(Customer ID, email, or phone) before account data can be accessed."
)


def _verified_cid(customer_id: Optional[str]) -> int:
    """Validate the state-injected customer id, or raise ValueError."""
    if not customer_id:
        raise ValueError(NOT_VERIFIED_MSG)
    return safe_int(customer_id, "customer ID")


@tool
@audit_tool_call
def get_invoices_by_customer_sorted_by_date(
    customer_id: Annotated[Optional[str], InjectedState("customer_id")],
) -> str:
    """Get all invoices for the verified customer, most recent first.

    Takes no arguments: the verified customer's identity is applied
    automatically and securely by the system.

    Returns:
        JSON array of the customer's invoices (InvoiceId, InvoiceDate,
        BillingCity, BillingCountry, Total) sorted by date descending,
        or a not-found / not-verified message.
    """
    try:
        cid = _verified_cid(customer_id)
    except ValueError as exc:
        return json_error(str(exc))

    result = run_query_safe(
        """
        SELECT InvoiceId, InvoiceDate, BillingCity, BillingCountry, Total
        FROM Invoice
        WHERE CustomerId = :cid
        ORDER BY InvoiceDate DESC, InvoiceId DESC
        """,
        {"cid": cid},
    )
    if result == "[]":
        return json_message(f"No invoices found for customer ID: {cid}")
    return result


@tool
@audit_tool_call
def get_purchased_tracks_by_customer(
    customer_id: Annotated[Optional[str], InjectedState("customer_id")],
) -> str:
    """Get every track the verified customer has purchased, sorted by
    price (highest first). Each row is one purchased track (an invoice
    line item), not a full invoice.

    Takes no arguments: the verified customer's identity is applied
    automatically and securely by the system.

    Returns:
        JSON array of purchased tracks (track name, artist, unit price,
        quantity, invoice ID and date) sorted by unit price descending,
        or a not-found / not-verified message.
    """
    try:
        cid = _verified_cid(customer_id)
    except ValueError as exc:
        return json_error(str(exc))

    result = run_query_safe(
        """
        SELECT il.InvoiceLineId, t.Name AS TrackName, ar.Name AS ArtistName,
               il.UnitPrice, il.Quantity, i.InvoiceId, i.InvoiceDate
        FROM InvoiceLine il
        JOIN Invoice i ON i.InvoiceId = il.InvoiceId
        JOIN Track t ON t.TrackId = il.TrackId
        LEFT JOIN Album al ON al.AlbumId = t.AlbumId
        LEFT JOIN Artist ar ON ar.ArtistId = al.ArtistId
        WHERE i.CustomerId = :cid
        ORDER BY il.UnitPrice DESC, il.InvoiceLineId ASC
        """,
        {"cid": cid},
    )
    if result == "[]":
        return json_message(f"No purchased tracks found for customer ID: {cid}")
    return result


@tool
@audit_tool_call
def get_support_rep_by_invoice(
    invoice_id: str,
    customer_id: Annotated[Optional[str], InjectedState("customer_id")],
) -> str:
    """Find the support representative for one of the verified
    customer's own invoices.

    Args:
        invoice_id: The numeric invoice ID as a string. Must belong to
            the verified customer.

    Returns:
        JSON object with the support rep's name, title, and email for
        that invoice, a not-found message (also returned when the
        invoice belongs to a different customer), or a not-verified /
        invalid-input error.
    """
    try:
        cid = _verified_cid(customer_id)
        iid = safe_int(invoice_id, "invoice ID")
    except ValueError as exc:
        return json_error(str(exc))

    result = run_query_safe(
        """
        SELECT i.InvoiceId, c.CustomerId,
               c.FirstName || ' ' || c.LastName AS CustomerName,
               e.FirstName || ' ' || e.LastName AS SupportRepName,
               e.Title AS SupportRepTitle, e.Email AS SupportRepEmail
        FROM Invoice i
        JOIN Customer c ON c.CustomerId = i.CustomerId
        LEFT JOIN Employee e ON e.EmployeeId = c.SupportRepId
        WHERE i.InvoiceId = :iid AND i.CustomerId = :cid
        """,
        {"iid": iid, "cid": cid},
    )
    if result == "[]":
        return json_message(
            f"No invoice with ID {iid} found on your account. Please "
            "check the invoice number."
        )
    return result


@tool
@audit_tool_call
def get_invoice_line_items(
    invoice_id: str,
    customer_id: Annotated[Optional[str], InjectedState("customer_id")],
) -> str:
    """Get the detailed line items (purchased tracks) for one of the
    verified customer's own invoices.

    Args:
        invoice_id: The numeric invoice ID as a string. Must belong to
            the verified customer.

    Returns:
        JSON array of line items (track name, artist, album, unit price,
        quantity), a not-found message (also returned when the invoice
        belongs to a different customer), or a not-verified /
        invalid-input error.
    """
    try:
        cid = _verified_cid(customer_id)
        iid = safe_int(invoice_id, "invoice ID")
    except ValueError as exc:
        return json_error(str(exc))

    result = run_query_safe(
        """
        SELECT il.InvoiceLineId, t.Name AS TrackName, ar.Name AS ArtistName,
               al.Title AS AlbumTitle, il.UnitPrice, il.Quantity
        FROM InvoiceLine il
        JOIN Invoice i ON i.InvoiceId = il.InvoiceId
        JOIN Track t ON t.TrackId = il.TrackId
        LEFT JOIN Album al ON al.AlbumId = t.AlbumId
        LEFT JOIN Artist ar ON ar.ArtistId = al.ArtistId
        WHERE il.InvoiceId = :iid AND i.CustomerId = :cid
        ORDER BY il.InvoiceLineId ASC
        """,
        {"iid": iid, "cid": cid},
    )
    if result == "[]":
        return json_message(
            f"No line items found for invoice ID {iid} on your account. "
            "Please check the invoice number."
        )
    return result
