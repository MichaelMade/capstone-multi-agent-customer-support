"""Pydantic v2 schemas for structured LLM output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VerificationAssessment(BaseModel):
    """Schema combining intent classification and identifier extraction.

    Used by the verification node to decide, in a single LLM call,
    whether a query needs account access and whether the user supplied
    an identifier.
    """

    needs_account_access: bool = Field(
        description=(
            "True if answering requires the customer's PERSONAL account "
            "data: invoices, purchases, spending, billing, order history, "
            "or their support representative. False for general music "
            "catalog browsing (songs, albums, artists, genres), greetings, "
            "and off-topic questions."
        ),
    )
    identifier: str = Field(
        default="",
        description=(
            "A single account identifier if present in the message: a "
            "numeric customer ID, an email address, or a phone number. "
            "Empty string if none. Track IDs, invoice numbers, prices, "
            "and dates are NOT account identifiers unless presented as "
            "the customer's own ID."
        ),
    )


class PreferenceExtraction(BaseModel):
    """Music preferences extracted from one customer-authored turn."""

    music_preferences: list[str] = Field(
        default_factory=list,
        description=(
            "Music genres or artists the customer has EXPLICITLY stated they "
            "like (e.g. 'I love jazz', 'AC/DC is my favorite'). Questions "
            "such as 'do you have rock?' are NOT preferences and must be "
            "excluded. Empty list if no explicit preferences were stated."
        ),
    )


class UserProfile(BaseModel):
    """Schema for a customer's persisted music preference profile."""

    customer_id: str = Field(description="The verified customer ID.")
    music_preferences: list[str] = Field(
        default_factory=list,
        description=(
            "Music genres or artists the customer has EXPLICITLY stated they "
            "like (e.g. 'I love jazz', 'AC/DC is my favorite'). Questions "
            "such as 'do you have rock?' are NOT preferences and must be "
            "excluded. Empty list if no explicit preferences were stated."
        ),
    )
