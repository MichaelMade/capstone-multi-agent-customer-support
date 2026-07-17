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


def _parse_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, str(default)).strip().casefold()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(
        f"{name} must be true or false; got {os.getenv(name, '')!r}"
    )


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
    app_username: str | None = field(
        default_factory=lambda: _optional_text("APP_USERNAME")
    )
    app_password: str | None = field(
        default_factory=lambda: _optional_text("APP_PASSWORD"),
        repr=False,
    )
    require_app_auth: bool = field(
        default_factory=lambda: _parse_bool("REQUIRE_APP_AUTH")
    )
    port: int = field(default_factory=_parse_port)
    # Bind to loopback by default; public binds (e.g. 0.0.0.0 inside
    # Docker) must be an explicit opt-in via the HOST variable.
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

    def __post_init__(self) -> None:
        has_username = self.app_username is not None
        has_password = self.app_password is not None
        if has_username != has_password:
            raise ConfigurationError(
                "APP_USERNAME and APP_PASSWORD must be set together"
            )
        if self.app_password is not None and len(self.app_password) < 12:
            raise ConfigurationError("APP_PASSWORD must be at least 12 characters")
        if self.require_app_auth and not (has_username and has_password):
            raise ConfigurationError(
                "APP_USERNAME and APP_PASSWORD are required when REQUIRE_APP_AUTH=true"
            )

    @property
    def app_auth(self) -> tuple[str, str] | None:
        """Return optional Gradio credentials without exposing the password."""
        if self.app_username is None or self.app_password is None:
            return None
        return self.app_username, self.app_password


def get_settings() -> Settings:
    """Return a fresh Settings snapshot from the current environment."""
    return Settings()
