"""Environment-driven configuration.

All runtime settings come from environment variables (loaded from a .env
file if present) and are validated before application startup.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

CHINOOK_SQL_URL = (
    "https://raw.githubusercontent.com/lerocha/chinook-database/master/"
    "ChinookDatabase/DataSources/Chinook_Sqlite.sql"
)


class ConfigurationError(ValueError):
    """Raised when an environment variable contains an invalid value."""


def _required_text(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise ConfigurationError(f"{name} cannot be blank")
    return value


def _optional_text(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _parse_temperature() -> float:
    raw = os.getenv("TEMPERATURE", "0.0").strip()
    if "_" in raw:
        raise ConfigurationError(
            f"TEMPERATURE must be a number from 0.0 to 2.0; got {raw!r}"
        )
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"TEMPERATURE must be a number from 0.0 to 2.0; got {raw!r}"
        ) from exc
    if not math.isfinite(value) or not 0.0 <= value <= 2.0:
        raise ConfigurationError(
            f"TEMPERATURE must be a finite number from 0.0 to 2.0; got {raw!r}"
        )
    return value


def _parse_port() -> int:
    raw = os.getenv("PORT", "7860").strip()
    if not raw.isascii() or not raw.isdecimal():
        raise ConfigurationError(
            f"PORT must be an integer from 1 to 65535; got {raw!r}"
        )
    value = int(raw)
    if not 1 <= value <= 65535:
        raise ConfigurationError(
            f"PORT must be an integer from 1 to 65535; got {raw!r}"
        )
    return value


def _parse_log_level() -> str:
    value = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    if value not in VALID_LOG_LEVELS:
        allowed = ", ".join(sorted(VALID_LOG_LEVELS))
        raise ConfigurationError(f"LOG_LEVEL must be one of {allowed}; got {value!r}")
    return value


def configure_logging(level: str) -> None:
    """Configure application logging after settings have been validated."""
    normalized = level.strip().upper()
    if normalized not in VALID_LOG_LEVELS:
        allowed = ", ".join(sorted(VALID_LOG_LEVELS))
        raise ConfigurationError(
            f"LOG_LEVEL must be one of {allowed}; got {normalized!r}"
        )
    logging.basicConfig(level=normalized, format=LOG_FORMAT, force=True)


@dataclass(frozen=True)
class Settings:
    """Immutable application settings resolved from the environment."""

    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "").strip(),
        repr=False,
    )
    model_name: str = field(
        default_factory=lambda: _required_text("MODEL_NAME", "gpt-4o-mini")
    )
    temperature: float = field(default_factory=_parse_temperature)
    openai_api_base: str | None = field(
        default_factory=lambda: _optional_text("OPENAI_API_BASE")
    )
    port: int = field(default_factory=_parse_port)
    # Bind to loopback by default; public binds (e.g. 0.0.0.0 inside
    # Docker) must be an explicit opt-in via the HOST variable. The chat
    # has no authentication layer, so never expose it publicly as-is.
    host: str = field(default_factory=lambda: _required_text("HOST", "127.0.0.1"))
    chinook_url: str = field(
        default_factory=lambda: _required_text("CHINOOK_SQL_URL", CHINOOK_SQL_URL)
    )
    chinook_cache_path: str = field(
        default_factory=lambda: _required_text(
            "CHINOOK_CACHE_PATH",
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data",
                "Chinook_Sqlite.sql",
            ),
        )
    )
    log_level: str = field(default_factory=_parse_log_level)


def get_settings() -> Settings:
    """Return a fresh Settings snapshot from the current environment."""
    return Settings()
