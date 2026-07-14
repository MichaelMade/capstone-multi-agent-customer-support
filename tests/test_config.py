"""Tests for environment validation and startup configuration handling."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import app
from src import config


CONFIG_ENV_VARS = (
    "OPENAI_API_KEY",
    "MODEL_NAME",
    "TEMPERATURE",
    "OPENAI_API_BASE",
    "PORT",
    "HOST",
    "CHINOOK_SQL_URL",
    "CHINOOK_CACHE_PATH",
    "LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def clean_config_environment(monkeypatch):
    for name in CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_default_settings_are_valid():
    settings = config.get_settings()

    assert settings.temperature == 0.0
    assert settings.port == 7860
    assert settings.host == "127.0.0.1"
    assert settings.log_level == "INFO"


@pytest.mark.parametrize("value", ["invalid", "nan", "inf", "0_1", "-0.1", "2.1"])
def test_temperature_rejects_non_finite_or_out_of_range_values(monkeypatch, value):
    monkeypatch.setenv("TEMPERATURE", value)

    with pytest.raises(config.ConfigurationError, match="TEMPERATURE"):
        config.get_settings()


@pytest.mark.parametrize("value, expected", [("0", 0.0), ("2", 2.0), ("0.7", 0.7)])
def test_temperature_accepts_finite_values_in_range(monkeypatch, value, expected):
    monkeypatch.setenv("TEMPERATURE", value)
    assert config.get_settings().temperature == expected


@pytest.mark.parametrize("value", ["invalid", "0", "-1", "+1", "1_0", "65536", "1.5"])
def test_port_rejects_non_integer_or_out_of_range_values(monkeypatch, value):
    monkeypatch.setenv("PORT", value)

    with pytest.raises(config.ConfigurationError, match="PORT"):
        config.get_settings()


@pytest.mark.parametrize("value", ["1", "7860", "65535"])
def test_port_accepts_valid_range(monkeypatch, value):
    monkeypatch.setenv("PORT", value)
    assert config.get_settings().port == int(value)


def test_log_level_is_normalized(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "warning")
    assert config.get_settings().log_level == "WARNING"


def test_invalid_log_level_has_actionable_error(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "verbose")

    with pytest.raises(config.ConfigurationError, match="LOG_LEVEL must be one of"):
        config.get_settings()


def test_api_key_is_trimmed_and_hidden_from_settings_repr(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "  super-secret  ")

    settings = config.get_settings()

    assert settings.openai_api_key == "super-secret"
    assert "super-secret" not in repr(settings)


@pytest.mark.parametrize(
    "name",
    ["MODEL_NAME", "HOST", "CHINOOK_SQL_URL", "CHINOOK_CACHE_PATH"],
)
def test_required_text_settings_reject_blank_values(monkeypatch, name):
    monkeypatch.setenv(name, "   ")

    with pytest.raises(config.ConfigurationError, match=name):
        config.get_settings()


def test_optional_api_base_is_trimmed(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "  https://example.test/v1  ")
    assert config.get_settings().openai_api_base == "https://example.test/v1"


def test_configure_logging_uses_validated_level(monkeypatch):
    basic_config = Mock()
    monkeypatch.setattr(config.logging, "basicConfig", basic_config)

    config.configure_logging("DEBUG")

    basic_config.assert_called_once_with(
        level="DEBUG", format=config.LOG_FORMAT, force=True
    )


def test_main_reports_configuration_error_concisely(monkeypatch):
    error = config.ConfigurationError("PORT must be valid")
    configure_logging = Mock()
    log_error = Mock()
    init_database = Mock()
    monkeypatch.setattr(app, "get_settings", Mock(side_effect=error))
    monkeypatch.setattr(app, "configure_logging", configure_logging)
    monkeypatch.setattr(app.logger, "error", log_error)
    monkeypatch.setattr(app, "init_database", init_database)

    with pytest.raises(SystemExit) as exc_info:
        app.main()

    assert exc_info.value.code == 2
    configure_logging.assert_called_once_with("ERROR")
    log_error.assert_called_once_with("Configuration error: %s", error)
    init_database.assert_not_called()


@pytest.mark.parametrize("api_key", ["", "sk-your-key-here"])
def test_main_configures_logging_before_missing_or_placeholder_key_exit(
    monkeypatch, api_key
):
    settings = SimpleNamespace(openai_api_key=api_key, log_level="WARNING")
    configure_logging = Mock()
    init_database = Mock()
    monkeypatch.setattr(app, "get_settings", Mock(return_value=settings))
    monkeypatch.setattr(app, "configure_logging", configure_logging)
    monkeypatch.setattr(app, "init_database", init_database)

    with pytest.raises(SystemExit) as exc_info:
        app.main()

    assert exc_info.value.code == 1
    configure_logging.assert_called_once_with("WARNING")
    init_database.assert_not_called()


def test_main_reports_database_initialization_failure(monkeypatch):
    settings = SimpleNamespace(openai_api_key="test-key", log_level="INFO")
    failure = RuntimeError("download unavailable")
    build_graph = Mock()
    monkeypatch.setattr(app, "get_settings", Mock(return_value=settings))
    monkeypatch.setattr(app, "configure_logging", Mock())
    monkeypatch.setattr(app, "init_database", Mock(side_effect=failure))
    monkeypatch.setattr(app.logger, "exception", Mock())
    monkeypatch.setattr(app, "build_graph", build_graph)

    with pytest.raises(SystemExit) as exc_info:
        app.main()

    assert exc_info.value.code == 1
    app.logger.exception.assert_called_once()
    build_graph.assert_not_called()


def test_main_reports_agent_graph_initialization_failure(monkeypatch):
    settings = SimpleNamespace(openai_api_key="test-key", log_level="INFO")
    failure = RuntimeError("invalid model configuration")
    build_ui = Mock()
    monkeypatch.setattr(app, "get_settings", Mock(return_value=settings))
    monkeypatch.setattr(app, "configure_logging", Mock())
    monkeypatch.setattr(app, "init_database", Mock())
    monkeypatch.setattr(
        app, "verify_database", Mock(return_value={"status": "healthy"})
    )
    monkeypatch.setattr(app, "build_graph", Mock(side_effect=failure))
    monkeypatch.setattr(app.logger, "exception", Mock())
    monkeypatch.setattr(app, "build_ui", build_ui)

    with pytest.raises(SystemExit) as exc_info:
        app.main()

    assert exc_info.value.code == 1
    app.logger.exception.assert_called_once()
    build_ui.assert_not_called()
