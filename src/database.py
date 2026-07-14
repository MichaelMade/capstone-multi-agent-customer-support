"""Database layer: Chinook loading, safe query execution, and utilities.

The Chinook SQL script is downloaded once and cached on disk; on every
startup it is executed into a fresh in-memory SQLite database shared by
all threads via a StaticPool single-connection engine.

Every query in this project flows through :func:`run_query_safe`, which
serializes access to the shared connection and only accepts read-only
SELECT/CTE statements with parameterized bindings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import tempfile
import threading
import time

import requests
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from src.config import get_settings

logger = logging.getLogger(__name__)

# SHA-256 of the pinned Chinook_Sqlite.sql artifact (v1.4.5). The
# downloaded script is executed with executescript(), so its integrity
# is verified before execution and startup fails closed on mismatch.
# Override with CHINOOK_SHA256=<new-hash> if upstream legitimately
# changes, or CHINOOK_SHA256=skip to disable (not recommended).
CHINOOK_SQL_SHA256 = "caf31d698a4a79c628215b552dfe6575e71be052ae02b8f18e763498f55f5d44"

_engine: Engine | None = None
# StaticPool deliberately shares one sqlite3 connection across every worker.
# Even with ``check_same_thread=False``, concurrent cursor use on that single
# connection corrupts result state (empty rows / IndexError).  One re-entrant
# lock therefore protects initialization and every connection operation.
_engine_lock = threading.RLock()

_READ_ONLY_SQL_RE = re.compile(
    r"\A(?:\s|--[^\r\n]*(?:\r?\n|\Z)|/\*.*?\*/)*(SELECT|WITH)\b",
    re.IGNORECASE | re.DOTALL,
)

EXPECTED_TABLES = {
    "Artist",
    "Album",
    "Track",
    "Genre",
    "MediaType",
    "Customer",
    "Employee",
    "Invoice",
    "InvoiceLine",
    "Playlist",
    "PlaylistTrack",
}


def verify_sql_integrity(sql_script: str, expected_sha256: str | None = None) -> None:
    """Verify the SQL script against the pinned SHA-256; fail closed.

    Raises RuntimeError on mismatch so a compromised download or cache
    can never reach ``executescript``.
    """
    expected = (
        expected_sha256
        if expected_sha256 is not None
        else os.getenv("CHINOOK_SHA256", CHINOOK_SQL_SHA256)
    )
    if expected.lower() == "skip":
        logger.warning("Chinook SQL integrity check DISABLED via CHINOOK_SHA256=skip")
        return
    actual = hashlib.sha256(sql_script.encode("utf-8")).hexdigest()
    if actual != expected.lower():
        raise RuntimeError(
            "Chinook SQL integrity check failed.\n"
            f"  expected sha256: {expected}\n"
            f"  actual sha256:   {actual}\n"
            "The download or cache may be corrupted or tampered with. "
            "Delete the cached file and retry; if upstream legitimately "
            "changed, set CHINOOK_SHA256 to the new hash after reviewing "
            "the file."
        )


def _download_chinook_sql() -> str:
    """Return the Chinook SQL script (integrity-verified), downloading
    and caching it if needed. A cached file that fails verification is
    discarded and re-downloaded once."""
    settings = get_settings()
    cache_path = settings.chinook_cache_path
    try:
        with open(cache_path, "r", encoding="utf-8") as fh:
            cached = fh.read()
    except FileNotFoundError:
        cached = ""

    if cached:
        try:
            verify_sql_integrity(cached)
            logger.info("Using cached Chinook SQL at %s", cache_path)
            return cached
        except RuntimeError:
            logger.warning("Cached Chinook SQL failed integrity check; re-downloading")
            try:
                os.remove(cache_path)
            except FileNotFoundError:
                # Another process may have replaced the same bad cache.
                pass

    logger.info("Downloading Chinook SQL from %s", settings.chinook_url)
    response = requests.get(settings.chinook_url, timeout=60)
    response.raise_for_status()
    sql_script = response.text
    verify_sql_integrity(sql_script)

    # Write beside the destination and atomically replace it.  Readers can
    # only observe a complete, integrity-verified file, even when multiple
    # app processes start at once.  Using the absolute parent also supports a
    # filename-only relative path such as CHINOOK_CACHE_PATH=chinook.sql.
    cache_dir = os.path.dirname(os.path.abspath(cache_path))
    os.makedirs(cache_dir, exist_ok=True)
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_dir,
            prefix=f".{os.path.basename(cache_path)}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            temp_path = fh.name
            fh.write(sql_script)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, cache_path)
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass
    logger.info("Cached Chinook SQL to %s (%d bytes)", cache_path, len(sql_script))
    return sql_script


def init_database(force: bool = False) -> Engine:
    """Create (once) and return the shared in-memory SQLite engine.

    Uses a StaticPool so every SQLAlchemy connection maps to the same
    underlying sqlite3 connection — required for an in-memory database
    shared across threads (Gradio handlers run in worker threads).
    """
    global _engine
    with _engine_lock:
        if _engine is not None and not force:
            return _engine

        sql_script = _download_chinook_sql()
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        raw = engine.raw_connection()
        try:
            driver: sqlite3.Connection = raw.driver_connection  # type: ignore[assignment]
            driver.executescript(sql_script)
            driver.commit()
            # Defense in depth: all runtime access is read-only.  The SQL
            # helper also rejects non-SELECT statements before execution, so
            # callers cannot turn this pragma off through run_query_safe().
            driver.execute("PRAGMA query_only = ON")
        finally:
            raw.close()

        old_engine = _engine
        _engine = engine
        if old_engine is not None:
            old_engine.dispose()
        logger.info("Chinook database loaded into in-memory SQLite")
        return _engine


def get_engine() -> Engine:
    """Return the initialized engine, initializing it on first use."""
    if _engine is None:
        return init_database()
    return _engine


def run_query_safe(sql: str, params: dict | None = None) -> str:
    """Execute a read-only parameterized query and return a JSON string.

    Args:
        sql: SELECT or CTE statement using named bind parameters (e.g. ``:cid``).
        params: Mapping of bind parameter names to values. User input must
            always be passed here — never interpolated into ``sql``.

    Returns:
        JSON array string; each row is an object keyed by column name.
        Empty result sets return ``"[]"``.
    """
    match = _READ_ONLY_SQL_RE.match(sql) if isinstance(sql, str) else None
    if not match:
        raise ValueError("Only read-only SELECT queries are allowed.")

    started = time.perf_counter()
    statement_type = match.group(1).upper()
    bind_names = sorted((params or {}).keys())

    # StaticPool returns the same underlying sqlite3 connection to every
    # caller, so execution and result consumption must be one critical
    # section.  Releasing the lock before ``all()`` is enough to corrupt a
    # concurrent cursor's result metadata.
    try:
        with _engine_lock:
            engine = get_engine()
            with engine.connect() as conn:
                result = conn.execute(text(sql), params or {})
                if not result.returns_rows:  # defensive; query_only should deny it
                    raise ValueError("Only read-only SELECT queries are allowed.")
                rows = [dict(row) for row in result.mappings().all()]
    except Exception:
        logger.exception(
            "Database query failed: type=%s bind_names=%s duration_ms=%.1f",
            statement_type,
            bind_names,
            (time.perf_counter() - started) * 1_000,
        )
        raise
    logger.debug(
        "Database query completed: type=%s bind_names=%s rows=%d duration_ms=%.1f",
        statement_type,
        bind_names,
        len(rows),
        (time.perf_counter() - started) * 1_000,
    )
    return json.dumps(rows, ensure_ascii=False, default=str)


def normalize_phone(phone: str | None) -> str:
    """Normalize a phone number for comparison.

    Strips every non-digit character while preserving a single leading
    ``+`` for international numbers.

    Examples:
        ``"+1 (555) 123-4567"`` -> ``"+15551234567"``
        ``"555-123-4567"`` -> ``"5551234567"``
        ``None`` / ``""`` -> ``""``
    """
    if not phone:
        return ""
    phone = phone.strip()
    plus = "+" if phone.startswith("+") else ""
    digits = re.sub(r"\D", "", phone)
    if not digits:
        return ""
    return plus + digits


def verify_database() -> dict:
    """Health check: confirm expected tables exist and data is queryable."""
    try:
        with _engine_lock:
            engine = get_engine()
            with engine.connect() as conn:
                table_rows = conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type = 'table'")
                ).fetchall()
                tables = {row[0] for row in table_rows}
                missing = EXPECTED_TABLES - tables
                customer_count = conn.execute(
                    text("SELECT COUNT(*) FROM Customer")
                ).scalar_one()
        healthy = not missing and customer_count > 0
        return {
            "status": "healthy" if healthy else "unhealthy",
            "table_count": len(tables),
            "missing_tables": sorted(missing),
            "customer_count": customer_count,
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Database health check failed")
        return {"status": "unhealthy", "error": str(exc)}


# --------------------------------------------------------------------------
# Customer lookup helpers (used by the identity verification node)
# --------------------------------------------------------------------------


def find_customer_by_id(customer_id: int) -> dict | None:
    """Look up a customer by numeric CustomerId. Returns None if not found."""
    rows = json.loads(
        run_query_safe(
            "SELECT CustomerId, FirstName, LastName, Email "
            "FROM Customer WHERE CustomerId = :cid",
            {"cid": customer_id},
        )
    )
    return rows[0] if rows else None


def find_customer_by_email(email: str) -> dict | None:
    """Look up a customer by email (case-insensitive). None if not found."""
    rows = json.loads(
        run_query_safe(
            "SELECT CustomerId, FirstName, LastName, Email "
            "FROM Customer WHERE LOWER(Email) = LOWER(:email)",
            {"email": email.strip()},
        )
    )
    return rows[0] if rows else None


def find_customer_by_phone(phone: str) -> dict | None:
    """Look up a customer by phone number after normalizing both sides.

    Chinook stores phones in inconsistent formats, so the comparison is
    done in Python on normalized values (59 rows — negligible cost).
    """
    target = normalize_phone(phone)
    if not target:
        return None
    rows = json.loads(
        run_query_safe(
            "SELECT CustomerId, FirstName, LastName, Email, Phone FROM Customer",
        )
    )
    for row in rows:
        stored = normalize_phone(row.get("Phone"))
        if stored and (stored == target or stored.lstrip("+") == target.lstrip("+")):
            return row
    return None
