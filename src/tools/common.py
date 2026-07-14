"""Shared helpers for tool functions."""

from __future__ import annotations

import functools
import json
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

_ToolFunction = TypeVar("_ToolFunction", bound=Callable[..., str])

SQLITE_INT64_MAX = 9_223_372_036_854_775_807
_SQLITE_INT64_MAX_TEXT = str(SQLITE_INT64_MAX)


def safe_int(value: str | int, field_name: str) -> int:
    """Validate and convert a canonical positive SQLite integer ID.

    Tools receive all parameters as strings from the LLM; numeric IDs must
    be validated before ever reaching a SQL query.  Reject Python-specific
    spellings (``1_000``), signs, leading zeroes, zero/negative values, and
    values outside SQLite's signed 64-bit integer range.
    """
    candidate = str(value).strip()
    valid_digits = (
        bool(candidate)
        and candidate.isascii()
        and candidate.isdecimal()
        and candidate[0] != "0"
    )
    in_range = valid_digits and (
        len(candidate) < len(_SQLITE_INT64_MAX_TEXT)
        or (
            len(candidate) == len(_SQLITE_INT64_MAX_TEXT)
            and candidate <= _SQLITE_INT64_MAX_TEXT
        )
    )
    if not in_range:
        raise ValueError(
            f"Invalid {field_name}: {value!r}. Please provide a positive "
            f"whole number from 1 to {SQLITE_INT64_MAX}."
        )
    return int(candidate)


def json_message(message: str, **extra) -> str:
    """Return a human-readable message wrapped in a valid JSON object."""
    payload = {"message": message}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def json_error(message: str) -> str:
    """Return an error message wrapped in a valid JSON object."""
    return json.dumps({"error": message}, ensure_ascii=False)


def audit_tool_call(function: _ToolFunction) -> _ToolFunction:
    """Log a tool's lifecycle without recording its customer-supplied inputs.

    Tool arguments can contain names, invoice numbers, or other customer data,
    so audit records intentionally include only the tool name, outcome shape,
    and elapsed time. ``functools.wraps`` preserves the signature used by
    LangChain to create each tool's input schema.
    """

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        started = time.perf_counter()
        logger.info("Tool call started: name=%s", function.__name__)
        try:
            result = function(*args, **kwargs)
        except Exception:
            logger.exception("Tool call failed: name=%s", function.__name__)
            raise

        outcome = "success"
        item_count: int | None = None
        try:
            payload = json.loads(result)
            if isinstance(payload, list):
                item_count = len(payload)
                outcome = "success" if payload else "empty"
            elif isinstance(payload, dict):
                if "error" in payload:
                    outcome = "error"
                elif "message" in payload:
                    outcome = "not_found"
                for key in ("tracks", "items", "results"):
                    if isinstance(payload.get(key), list):
                        item_count = len(payload[key])
                        break
        except (TypeError, ValueError):
            # Returning non-JSON is a contract violation worth surfacing. The
            # original value is never logged because it may contain user data.
            outcome = "invalid_json"

        logger.info(
            "Tool call completed: name=%s outcome=%s item_count=%s duration_ms=%.1f",
            function.__name__,
            outcome,
            item_count if item_count is not None else "n/a",
            (time.perf_counter() - started) * 1_000,
        )
        return result

    return wrapper  # type: ignore[return-value]
