"""Identity verification node with human-in-the-loop interrupt.

Browse-first policy: catalog questions, greetings, and off-topic
messages flow to the supervisor without verification. Only queries
needing PERSONAL account data (invoices, purchases, support rep) gate
on identity. One structured LLM call per unverified turn classifies the
intent AND extracts any identifier the user offered.

Verification outcome is written to graph state:
- success  -> ``customer_id`` set (also announced via SystemMessage);
- browse   -> ``customer_id`` explicitly None (unverified browsing OK);
- required -> graph pauses at ``human_input`` via ``interrupt()`` until
  the user supplies a valid identifier.

The state field is the ONLY identity source downstream: invoice tools
receive it via ``InjectedState``, so even a confused or prompt-injected
model cannot query another customer's data.
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.types import interrupt

from src.agents.prompts import (
    VERIFICATION_ASK_MESSAGE,
    VERIFICATION_ASSESS_PROMPT,
    VERIFICATION_RETRY_MESSAGE,
)
from src.database import (
    find_customer_by_email,
    find_customer_by_id,
    find_customer_by_phone,
)
from src.llm import get_llm
from src.models import VerificationAssessment
from src.state import State
from src.tools.common import safe_int

logger = logging.getLogger(__name__)

_EMAIL_PATTERN = (
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
)
_EMAIL_RE = re.compile(rf"^{_EMAIL_PATTERN}$")
_EMAIL_SEARCH_RE = re.compile(_EMAIL_PATTERN)
_PHONE_LIKE_RE = re.compile(r"[+()\d\s\-.]{7,}")


def _deterministic_identifier(text: str) -> str:
    """Cheap, deterministic identifier detection — no LLM required.

    Catches the common resume case where the user answers the
    verification prompt with just an identifier ("5", an email, or a
    phone number), plus emails embedded anywhere in a sentence.
    """
    t = (text or "").strip()
    if not t:
        return ""
    if re.fullmatch(r"\d{1,6}", t):
        return t  # bare customer ID
    if _EMAIL_RE.fullmatch(t):
        return t  # bare email
    if _PHONE_LIKE_RE.fullmatch(t) and sum(c.isdigit() for c in t) >= 7:
        return t  # bare phone number
    found = _EMAIL_SEARCH_RE.search(t)
    if found:
        return found.group(0)  # email mentioned inside a sentence
    return ""


def _lookup_customer(identifier: str) -> dict | None:
    """Resolve an identifier to a customer row (ID, email, or phone)."""
    identifier = identifier.strip()
    if not identifier:
        return None
    if identifier.isdigit():
        try:
            customer_id = safe_int(identifier, "customer ID")
        except ValueError:
            return None
        return find_customer_by_id(customer_id)
    if _EMAIL_RE.fullmatch(identifier):
        return find_customer_by_email(identifier)
    # Fall back to phone matching (handles +, dashes, parens, spaces).
    return find_customer_by_phone(identifier)


def _last_human_message(state: State) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _verified_update(customer: dict) -> dict:
    customer_id = str(customer["CustomerId"])
    logger.info("Customer verified: id=%s", customer_id)
    return {
        "customer_id": customer_id,
        "needs_verification": False,
        "verification_message": "",
        "messages": [
            SystemMessage(
                content=(
                    f"Identity verified. customer_id={customer_id} "
                    f"({customer['FirstName']} {customer['LastName']}). "
                    "Downstream agents must use this verified "
                    "customer_id only."
                )
            )
        ],
    }


def verify_info(state: State) -> dict:
    """Assess the latest message and verify identity when required.

    Already-verified sessions pass straight through. Otherwise one
    structured LLM call classifies whether the query needs account data
    and extracts any offered identifier. Identifiers are verified
    opportunistically (even for catalog questions); verification is only
    REQUIRED — pausing the graph — for account-data queries.

    Always writes ``customer_id`` (value or None) so the state field
    exists for InjectedState-based tools.
    """
    if state.get("customer_id"):
        return {"needs_verification": False}

    user_text = _last_human_message(state)
    pending = bool(state.get("needs_verification"))

    # Deterministic fast-path: bare IDs/emails/phones (typical when the
    # user is answering the verification prompt) skip the LLM entirely.
    identifier = _deterministic_identifier(user_text)
    needs_account = pending

    # A bare identifier is unambiguous and can bypass intent classification on
    # both fresh and resumed turns. Embedded identifiers still require the LLM
    # assessment; otherwise an unknown email inside an account query could
    # accidentally skip the verification gate.
    bare_identifier = bool(identifier and user_text.strip() == identifier)
    if bare_identifier:
        # Supplying only an identifier is an explicit verification attempt,
        # including on a fresh thread.  If it does not resolve, keep the graph
        # at the human-in-the-loop gate so the user receives the required retry
        # prompt instead of letting the unexplained value reach the supervisor.
        needs_account = True
    elif user_text:
        try:
            assessor = get_llm().with_structured_output(VerificationAssessment)
            assessment = assessor.invoke(
                [
                    SystemMessage(content=VERIFICATION_ASSESS_PROMPT),
                    HumanMessage(content=user_text),
                ]
            )
            # A pending verification is released only if the user has
            # moved on to a non-account question; account questions
            # keep the gate closed.
            assessed_needs_account = bool(assessment.needs_account_access)
            assessed_identifier = (assessment.identifier or "").strip()
            identifier = identifier or assessed_identifier
            # While an interrupt is pending, an attempted identifier keeps the
            # gate closed even if the classifier treats the identifier-only
            # response as a non-account query. A genuinely new catalog query
            # without an identifier can still release the pending gate.
            needs_account = assessed_needs_account or (pending and bool(identifier))
        except Exception:  # pragma: no cover - LLM/API failure fallback
            # Fail closed: if we cannot assess, require verification so
            # account data is never exposed by accident.
            logger.exception("Verification assessment failed; failing closed")
            needs_account = True

    if identifier:
        customer = _lookup_customer(identifier)
        if customer:
            return _verified_update(customer)
        # Identifiers can be emails or phone numbers, so never write the raw
        # value to application logs.
        logger.info("Verification failed for supplied identifier")
        if needs_account:
            return {
                "customer_id": None,
                "needs_verification": True,
                "verification_message": VERIFICATION_RETRY_MESSAGE.format(
                    identifier=identifier
                ),
                "pending_memory_text": (state.get("pending_memory_text") or user_text),
            }

    if needs_account:
        return {
            "customer_id": None,
            "needs_verification": True,
            "verification_message": VERIFICATION_ASK_MESSAGE,
            "pending_memory_text": state.get("pending_memory_text") or user_text,
        }

    # Catalog / greeting / off-topic: proceed unverified. Invoice tools
    # remain hard-locked via InjectedState until verification happens.
    return {"customer_id": None, "needs_verification": False}


def route_after_verification(state: State) -> str:
    """Conditional edge: pause only when account access requires identity."""
    if state.get("customer_id"):
        return "continue"
    return "unverified" if state.get("needs_verification") else "continue"


def human_input(state: State) -> dict:
    """Pause the graph and wait for the user to supply an identifier."""
    question = state.get("verification_message") or VERIFICATION_ASK_MESSAGE
    answer = interrupt(question)
    return {"messages": [HumanMessage(content=str(answer))]}
