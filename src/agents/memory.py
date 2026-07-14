"""Preference memory nodes: load at start, extract & merge after replies.

Storage: LangGraph InMemoryStore, namespace ``("memory_profile", <cid>)``,
key ``"user_memory"``. Merging is set-union — existing preferences are
never removed, and an empty extraction never overwrites saved data.
Note: in-memory only; data is lost on restart (documented limitation).
"""

from __future__ import annotations

import json
import logging
import re
import threading

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_store

from src.agents.prompts import MEMORY_EXTRACTION_PROMPT
from src.llm import get_llm
from src.database import run_query_safe
from src.models import PreferenceExtraction, UserProfile
from src.state import State

logger = logging.getLogger(__name__)

# The store itself supports concurrent access, but create_memory performs a
# read/merge/write sequence. Serialize that final sequence so two browser
# sessions verified as the same customer cannot overwrite each other's newly
# extracted preferences.
_memory_write_lock = threading.RLock()

MEMORY_KEY = "user_memory"
MAX_PREFERENCES = 20
MAX_PREFERENCE_LENGTH = 60

# Deterministic gate: preferences are only extracted when a declarative
# sentence in the latest turn uses explicit preference language. Question
# sentences are excluded before the LLM is called. Notes:
# - "I like to see my invoices" is excluded (like + infinitive).
# - "Do I like jazz?" is excluded (question form).
_PREFERENCE_SIGNAL_RE = re.compile(
    r"\b("
    r"i\s+(really\s+|absolutely\s+)?(love|enjoy|adore|prefer)\b"
    r"|(?<!\bdo\s)i\s+(really\s+)?like\b(?!\s+to\b)"
    r"|my\s+favou?rites?\b"
    r"|favou?rite\s+(band|artist|genre|music|song)"
    r"|(big\s+)?fan\s+of"
    r"|i'?m\s+(really\s+)?into"
    r")\b",
    re.IGNORECASE,
)

# Persisted preferences are rendered as untrusted JSON labels in the music
# prompt, but reject obvious instruction-shaped values as an additional guard.
# Known catalog names are canonicalized before this check, so artists such as
# "Tool" or "System Of A Down" are still preserved when present in Chinook.
_INSTRUCTION_LIKE_PREFERENCE_RE = re.compile(
    r"\b(?:ignore|disregard|forget|override|instructions?|prompt|assistant|"
    r"developer|execute|reveal|secrets?|password|api\s*key)\b",
    re.IGNORECASE,
)


def _namespace(customer_id: str) -> tuple[str, str]:
    return ("memory_profile", str(customer_id))


def get_saved_preferences(customer_id: str) -> list[str]:
    """Read a customer's saved preferences from the store ([] if none)."""
    store = get_store()
    item = store.get(_namespace(customer_id), MEMORY_KEY)
    if item and isinstance(item.value, dict):
        try:
            profile = UserProfile.model_validate(item.value)
        except (TypeError, ValueError):
            logger.warning("Ignoring an invalid saved preference profile")
            return []
        if profile.customer_id != str(customer_id):
            logger.warning("Ignoring a saved profile with a mismatched customer ID")
            return []
        return [
            canonical
            for value in profile.music_preferences[:MAX_PREFERENCES]
            if (canonical := _canonical_catalog_preference(value))
        ]
    return []


def load_memory(state: State) -> dict:
    """Load saved preferences for the verified customer into state."""
    customer_id = state.get("customer_id")
    if not customer_id:
        return {"loaded_memory": ""}
    preferences = get_saved_preferences(customer_id)
    if preferences:
        logger.info(
            "Loaded %d saved preferences for customer %s",
            len(preferences),
            customer_id,
        )
    # JSON keeps each catalog label visibly data-shaped when it is embedded in
    # the music agent's system prompt.
    return {
        "loaded_memory": json.dumps(preferences, ensure_ascii=False)
        if preferences
        else ""
    }


def _sanitize_preference(pref: str) -> str:
    """Normalize a preference string before storage.

    Strips control characters and newlines and caps the length. Stored
    preferences are later interpolated into the music agent's system
    prompt, so this blunts indirect prompt injection via memory.
    """
    cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", str(pref))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:MAX_PREFERENCE_LENGTH]


def _merge_preferences(existing: list[str], new: list[str]) -> list[str]:
    """Set-union merge preserving order; case-insensitive de-duplication.

    Existing preferences always survive; new ones are appended
    (sanitized) up to a total cap of MAX_PREFERENCES.
    """
    merged = list(existing)
    seen = {str(p).strip().lower() for p in existing}
    for pref in new:
        cleaned = _sanitize_preference(pref)
        if cleaned and cleaned.lower() not in seen and len(merged) < MAX_PREFERENCES:
            merged.append(cleaned)
            seen.add(cleaned.lower())
    return merged


def _canonical_catalog_preference(pref: str) -> str:
    """Normalize a genre/artist preference, canonicalizing catalog matches.

    Preference text is later inserted into a system prompt. Restricting saved
    values to short, sanitized labels and rejecting instruction-shaped text
    prevents arbitrary model output from becoming persistent prompt data.
    Explicit out-of-catalog tastes are still retained because preference memory
    describes the customer, not only the store's current inventory.
    """
    cleaned = _sanitize_preference(pref)
    if not cleaned:
        return ""
    # Models sometimes return a genre as "classical music" even though the
    # catalog label is simply "Classical".
    candidate = re.sub(r"\s+music$", "", cleaned, flags=re.IGNORECASE).strip()
    rows = json.loads(
        run_query_safe(
            """
            SELECT Name
            FROM (
                SELECT Name, 0 AS preference_order FROM Genre
                UNION ALL
                SELECT Name, 1 AS preference_order FROM Artist
            )
            WHERE Name = :preference COLLATE NOCASE
            ORDER BY preference_order, Name
            LIMIT 1
            """,
            {"preference": candidate},
        )
    )
    if rows:
        return str(rows[0]["Name"])
    if _INSTRUCTION_LIKE_PREFERENCE_RE.search(candidate):
        return ""
    return candidate


def _latest_customer_text(state: State) -> str:
    """The customer's most recent message, and only that.

    Turn-scoped by design: create_memory runs once per turn, so each
    customer message is assessed exactly once, in the turn it was sent.
    Earlier preference statements were already extracted in their own
    turns, and assistant replies (which enumerate genres/artists) are
    excluded entirely so they can never bias extraction.
    """
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage) and str(message.content).strip():
            return f"Customer: {message.content}"
    return ""


def _memory_source_text(state: State) -> str:
    """Return the user turn to assess, including one held across verification.

    A verification resume appends the supplied identifier as the latest human
    message.  The original account query is therefore carried separately until
    this node has had a chance to extract any explicit preference from it.
    """
    pending = str(state.get("pending_memory_text") or "").strip()
    if pending:
        return f"Customer: {pending}"
    return _latest_customer_text(state)


def has_preference_signal(text: str) -> bool:
    """True if a declarative clause contains preference language.

    Semicolons and commas often join a preference to a follow-up question
    ("I love jazz; what albums do you have?").  Evaluate those clauses
    independently so the trailing question does not erase the explicit
    preference, while still excluding genuinely interrogative clauses.
    """
    for clause in re.split(r"[;,\n]|(?<=[.!?])\s+", text or ""):
        clause = clause.strip()
        if not clause or clause.endswith("?"):
            continue
        if _PREFERENCE_SIGNAL_RE.search(clause):
            return True
    return False


def create_memory(state: State) -> dict:
    """Extract explicit preferences from the latest turn and merge-save."""
    # Clear a held turn on every exit so it cannot be reprocessed later.
    update = {"pending_memory_text": ""}
    customer_id = state.get("customer_id")
    if not customer_id:
        return update

    existing = get_saved_preferences(customer_id)
    conversation = _memory_source_text(state)
    if not conversation:
        return update
    if not has_preference_signal(conversation):
        # No explicit preference language in the latest turn: skip
        # extraction entirely (questions are never preferences, and
        # already-processed turns are never re-extracted).
        return update

    try:
        extractor = get_llm().with_structured_output(PreferenceExtraction)
        profile = extractor.invoke(
            [
                SystemMessage(
                    content=MEMORY_EXTRACTION_PROMPT.format(
                        existing=", ".join(existing) if existing else "(none)"
                    )
                ),
                HumanMessage(content=conversation),
            ]
        )
        new_preferences = [
            canonical
            for preference in profile.music_preferences[:MAX_PREFERENCES]
            if (canonical := _canonical_catalog_preference(preference))
        ]
    except Exception:  # pragma: no cover - LLM/API failure fallback
        logger.exception("Preference extraction failed; keeping existing profile")
        return update

    if not new_preferences:
        # Never wipe saved preferences with an empty extraction.
        return update

    with _memory_write_lock:
        # Re-read inside the critical section: another session may have saved a
        # preference while this turn's LLM extraction was in flight.
        latest_existing = get_saved_preferences(customer_id)
        merged = _merge_preferences(latest_existing, new_preferences)
        if merged == latest_existing:
            return update

        store = get_store()
        saved_profile = UserProfile(
            customer_id=str(customer_id), music_preferences=merged
        )
        store.put(
            _namespace(customer_id),
            MEMORY_KEY,
            saved_profile.model_dump(),
        )
    logger.info("Saved %d preferences for customer %s", len(merged), customer_id)
    return update
