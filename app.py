"""Application entry point.

Run with: ``python app.py`` (requires OPENAI_API_KEY in the environment
or a .env file). Loads the Chinook database into in-memory SQLite,
compiles the multi-agent graph, and launches the Gradio chat UI.
"""

import logging
import sys

import uvicorn

from src.config import ConfigurationError, configure_logging, get_settings
from src.database import init_database, verify_database
from src.graph import build_graph
from src.pwa import build_pwa_app
from src.ui import build_ui

logger = logging.getLogger("app")


def main() -> None:
    try:
        settings = get_settings()
    except ConfigurationError as exc:
        configure_logging("ERROR")
        logger.error("Configuration error: %s", exc)
        sys.exit(2)

    configure_logging(settings.log_level)
    if not settings.openai_api_key or settings.openai_api_key == "sk-your-key-here":
        logger.error(
            "OPENAI_API_KEY is missing or still uses the example placeholder. "
            "Copy .env.example to .env and add your key, or export it in "
            "your shell."
        )
        sys.exit(1)

    logger.info("Initializing Chinook database…")
    try:
        init_database()
        health = verify_database()
    except Exception:
        logger.exception(
            "Database initialization failed. Check the Chinook download, "
            "cache path, and integrity settings."
        )
        sys.exit(1)
    logger.info("Database health: %s", health)
    if health.get("status") != "healthy":
        logger.error("Database failed its health check; aborting startup.")
        sys.exit(1)

    logger.info("Compiling multi-agent graph…")
    try:
        graph = build_graph()
    except Exception:
        logger.exception(
            "Agent graph initialization failed. Check the model and API settings."
        )
        sys.exit(1)

    demo = build_ui(graph)
    logger.info("Launching UI on %s:%s", settings.host, settings.port)
    web_app = build_pwa_app(
        demo,
        host=settings.host,
        port=settings.port,
        auth=settings.app_auth,
    )
    uvicorn.run(
        web_app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        log_config=None,
    )


if __name__ == "__main__":
    main()
