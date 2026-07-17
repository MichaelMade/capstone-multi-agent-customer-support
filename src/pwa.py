"""Progressive Web App shell for the Gradio customer-support interface.

The PWA endpoints live on a small FastAPI parent application so the service
worker can control the root scope. Gradio remains mounted at ``/`` and keeps
all of its existing API and queue behavior.

Privacy rule: only immutable icons and the static offline page are eligible
for CacheStorage. The live page, conversations, account data, and every
Gradio/API response remain network-only and receive explicit no-store headers.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response

PWA_ASSET_DIR = Path(__file__).resolve().parent.parent / "pwa"

APP_NAME = "Digital Music Store — Customer Support"
APP_SHORT_NAME = "Music Support"
APP_DESCRIPTION = (
    "Multi-agent support for browsing the music catalog and reviewing "
    "verified customer invoices and purchases."
)
THEME_COLOR = "#1d4ed8"
BACKGROUND_COLOR = "#f8fafc"

PWA_HEAD = """
<link rel="icon" type="image/png" sizes="192x192" href="/pwa/icon-192.png" />
<link rel="apple-touch-icon" sizes="180x180" href="/pwa/icon-180.png" />
<meta name="theme-color" content="#1d4ed8" />
<meta name="apple-mobile-web-app-capable" content="yes" />
<meta name="apple-mobile-web-app-status-bar-style" content="default" />
<meta name="apple-mobile-web-app-title" content="Music Support" />
<script>
(() => {
  let installPrompt = null;

  const syncInstallButton = () => {
    const container = document.querySelector("#install-app");
    if (!container) return;
    const standalone = window.matchMedia("(display-mode: standalone)").matches;
    container.style.display = installPrompt && !standalone ? "block" : "none";
  };

  window.addEventListener("beforeinstallprompt", (event) => {
    event.preventDefault();
    installPrompt = event;
    syncInstallButton();
  });

  window.addEventListener("appinstalled", () => {
    installPrompt = null;
    syncInstallButton();
  });

  document.addEventListener("click", async (event) => {
    const target = event.target instanceof Element
      ? event.target.closest("#install-app")
      : null;
    if (!target || !installPrompt) return;
    event.preventDefault();
    await installPrompt.prompt();
    await installPrompt.userChoice;
    installPrompt = null;
    syncInstallButton();
  });

  new MutationObserver(syncInstallButton).observe(document.documentElement, {
    childList: true,
    subtree: true,
  });

  if ("serviceWorker" in navigator) {
    const registerServiceWorker = () => {
      navigator.serviceWorker.register("/service-worker.js", {
        scope: "/",
        updateViaCache: "none",
      }).catch((error) => console.error("Service worker registration failed", error));
    };

    if (document.readyState === "loading") {
      window.addEventListener("load", registerServiceWorker, { once: true });
    } else {
      registerServiceWorker();
    }
  }
})();
</script>
"""

PWA_CSS = """
#install-app {
  display: none;
}
"""

_NO_STORE_PATHS = frozenset({"/", "/config", "/healthz", "/login", "/logout"})
_NO_STORE_PREFIXES = (
    "/gradio_api",
    "/api",
    "/queue",
    "/monitoring",
    "/file",
    "/stream",
)
_BLOCKED_RUNTIME_PATHS = frozenset(
    {
        "/gradio_api/deep_link",
        "/gradio_api/deep_link/",
        "/openapi.json",
        "/openapi.json/",
    }
)


def _requires_no_store(path: str) -> bool:
    return path in _NO_STORE_PATHS or path.startswith(_NO_STORE_PREFIXES)


def _asset_response(filename: str, media_type: str) -> FileResponse:
    return FileResponse(
        PWA_ASSET_DIR / filename,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


def build_pwa_app(
    demo: gr.Blocks,
    *,
    host: str,
    port: int,
    auth: tuple[str, str] | None = None,
) -> FastAPI:
    """Mount ``demo`` in an installable, offline-aware PWA host."""
    app = FastAPI(
        title=APP_NAME,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def prevent_sensitive_response_caching(
        request: Request, call_next
    ) -> Response:
        if request.url.path in _BLOCKED_RUNTIME_PATHS:
            return JSONResponse(
                {"detail": "Not Found"},
                status_code=404,
                headers={
                    "Cache-Control": "no-store, private, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        response = await call_next(request)
        if _requires_no_store(request.url.path):
            response.headers["Cache-Control"] = "no-store, private, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @app.get("/manifest.json", include_in_schema=False)
    def manifest() -> JSONResponse:
        return JSONResponse(
            {
                "id": "/",
                "name": APP_NAME,
                "short_name": APP_SHORT_NAME,
                "description": APP_DESCRIPTION,
                "start_url": "/",
                "scope": "/",
                "display": "standalone",
                "background_color": BACKGROUND_COLOR,
                "theme_color": THEME_COLOR,
                "prefer_related_applications": False,
                "categories": ["music", "business"],
                "icons": [
                    {
                        "src": "/pwa/icon-192.png",
                        "sizes": "192x192",
                        "type": "image/png",
                        "purpose": "any",
                    },
                    {
                        "src": "/pwa/icon-512.png",
                        "sizes": "512x512",
                        "type": "image/png",
                        "purpose": "any",
                    },
                    {
                        "src": "/pwa/icon-maskable-512.png",
                        "sizes": "512x512",
                        "type": "image/png",
                        "purpose": "maskable",
                    },
                ],
            },
            media_type="application/manifest+json",
            headers={"Cache-Control": "no-cache, max-age=0"},
        )

    @app.get("/healthz", include_in_schema=False)
    def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/service-worker.js", include_in_schema=False)
    def service_worker() -> FileResponse:
        return FileResponse(
            PWA_ASSET_DIR / "service-worker.js",
            media_type="application/javascript",
            headers={
                "Cache-Control": "no-cache, max-age=0",
                "Service-Worker-Allowed": "/",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/offline.html", include_in_schema=False)
    def offline_page() -> FileResponse:
        return FileResponse(
            PWA_ASSET_DIR / "offline.html",
            media_type="text/html",
            headers={"Cache-Control": "no-cache, max-age=0"},
        )

    @app.get("/pwa/icon-180.png", include_in_schema=False)
    def icon_180() -> FileResponse:
        return _asset_response("icon-180.png", "image/png")

    @app.get("/pwa/icon-192.png", include_in_schema=False)
    def icon_192() -> FileResponse:
        return _asset_response("icon-192.png", "image/png")

    @app.get("/pwa/icon-512.png", include_in_schema=False)
    def icon_512() -> FileResponse:
        return _asset_response("icon-512.png", "image/png")

    @app.get("/pwa/icon-maskable-512.png", include_in_schema=False)
    def icon_maskable_512() -> FileResponse:
        return _asset_response("icon-maskable-512.png", "image/png")

    return gr.mount_gradio_app(
        app,
        demo,
        path="/",
        server_name=host,
        server_port=port,
        favicon_path=str(PWA_ASSET_DIR / "icon-192.png"),
        pwa=True,
        auth=auth,
        auth_message=(
            "Private capstone demo. Sign in with the deployment credentials."
            if auth
            else None
        ),
        enable_monitoring=False,
        footer_links=[],
        head=PWA_HEAD,
        css=PWA_CSS,
    )
