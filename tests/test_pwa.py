"""Deterministic checks for the installable, privacy-safe PWA shell."""

from __future__ import annotations

import asyncio
import json
import re
import struct
from html.parser import HTMLParser

import gradio as gr
import httpx
import pytest

from src.pwa import (
    APP_DESCRIPTION,
    APP_NAME,
    APP_SHORT_NAME,
    BACKGROUND_COLOR,
    THEME_COLOR,
    build_pwa_app,
)


class _HeadParser(HTMLParser):
    """Collect links and metadata from a small HTML head fragment."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str | None]] = []
        self.meta: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "link":
            self.links.append(attributes)
        elif tag == "meta":
            self.meta.append(attributes)


class _ASGIClient:
    """Small synchronous facade over httpx's async in-process transport."""

    def __init__(self, app) -> None:
        self.app = app

    def get(self, path: str) -> httpx.Response:
        async def request() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
                follow_redirects=True,
            ) as client:
                return await client.get(path)

        return asyncio.run(request())


@pytest.fixture(scope="module")
def pwa_client():
    """Serve a minimal Gradio UI through the production PWA wrapper."""
    with gr.Blocks(title="PWA test") as demo:
        gr.Markdown("PWA test fixture")

    app = build_pwa_app(demo, host="127.0.0.1", port=7860)
    return _ASGIClient(app)


def _embedded_gradio_config(html: str) -> dict:
    match = re.search(
        r"window\.gradio_config = (\{.*?\});</script>",
        html,
        flags=re.DOTALL,
    )
    assert match is not None, "Gradio configuration was not embedded in the root page"
    return json.loads(match.group(1))


def test_manifest_has_installability_fields_and_safe_cache_headers(pwa_client):
    response = pwa_client.get("/manifest.json")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/manifest+json"
    assert response.headers["cache-control"] == "no-cache, max-age=0"
    assert response.json() == {
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
    }


def test_health_endpoint_is_minimal_and_never_cached(pwa_client):
    response = pwa_client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["cache-control"] == "no-store, private, max-age=0"
    assert response.headers["pragma"] == "no-cache"
    assert response.headers["expires"] == "0"


def test_service_worker_is_root_scoped_and_only_precaches_offline_shell(
    pwa_client,
):
    response = pwa_client.get("/service-worker.js")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/javascript"
    assert response.headers["cache-control"] == "no-cache, max-age=0"
    assert response.headers["service-worker-allowed"] == "/"
    assert response.headers["x-content-type-options"] == "nosniff"

    worker = response.text
    shell_match = re.search(
        r"const SHELL_ASSETS = Object\.freeze\(\[(.*?)\]\);",
        worker,
        flags=re.DOTALL,
    )
    assert shell_match is not None
    assert 'const OFFLINE_URL = "/offline.html";' in worker
    assert "OFFLINE_URL" in shell_match.group(1)
    assert set(re.findall(r'"([^"\n]+)"', shell_match.group(1))) == {
        "/pwa/icon-180.png",
        "/pwa/icon-192.png",
        "/pwa/icon-512.png",
        "/pwa/icon-maskable-512.png",
    }
    assert "cache.addAll(SHELL_ASSETS)" in worker
    assert 'request.mode === "navigate"' in worker
    assert 'fetch(request, { cache: "no-store" })' in worker
    assert "caches.match(OFFLINE_URL)" in worker
    assert "SHELL_PATHS.has(url.pathname)" in worker
    assert "cache.put" not in worker


def test_offline_page_explains_network_need_and_privacy(pwa_client):
    response = pwa_client.get("/offline.html")

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    offline_page = response.text.casefold()
    assert "you’re offline" in offline_page
    assert "reconnect" in offline_page
    assert "conversations and account information" in offline_page
    assert "never saved for offline use" in offline_page
    assert "select try again to reconnect" in offline_page
    assert 'src="/pwa/icon-192.png"' in response.text
    assert "window.location.reload()" in response.text


@pytest.mark.parametrize(
    ("path", "expected_dimensions"),
    [
        ("/pwa/icon-180.png", (180, 180)),
        ("/pwa/icon-192.png", (192, 192)),
        ("/pwa/icon-512.png", (512, 512)),
        ("/pwa/icon-maskable-512.png", (512, 512)),
    ],
)
def test_pwa_icons_are_valid_pngs(pwa_client, path, expected_dimensions):
    response = pwa_client.get(path)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == ("public, max-age=31536000, immutable")
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert response.content[12:16] == b"IHDR"
    assert struct.unpack(">II", response.content[16:24]) == expected_dimensions


@pytest.mark.parametrize("path", ["/", "/config"])
def test_live_app_responses_are_never_cacheable(pwa_client, path):
    response = pwa_client.get(path)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store, private, max-age=0"
    assert response.headers["pragma"] == "no-cache"
    assert response.headers["expires"] == "0"


def test_root_has_one_manifest_link_and_supplies_pwa_head(pwa_client):
    response = pwa_client.get("/")

    assert response.status_code == 200
    document_parser = _HeadParser()
    document_parser.feed(response.text)
    manifest_links = [
        link
        for link in document_parser.links
        if "manifest" in (link.get("rel") or "").split()
    ]
    assert manifest_links == [{"rel": "manifest", "href": "/manifest.json"}]

    config = _embedded_gradio_config(response.text)
    assert config["pwa"] is True
    pwa_head = config["head"]
    head_parser = _HeadParser()
    head_parser.feed(pwa_head)

    assert {
        "rel": "icon",
        "type": "image/png",
        "sizes": "192x192",
        "href": "/pwa/icon-192.png",
    } in head_parser.links
    assert {
        "rel": "apple-touch-icon",
        "sizes": "180x180",
        "href": "/pwa/icon-180.png",
    } in head_parser.links

    metadata = {
        item["name"]: item.get("content")
        for item in head_parser.meta
        if item.get("name")
    }
    assert metadata == {
        "theme-color": THEME_COLOR,
        "apple-mobile-web-app-capable": "yes",
        "apple-mobile-web-app-status-bar-style": "default",
        "apple-mobile-web-app-title": APP_SHORT_NAME,
    }
    assert 'navigator.serviceWorker.register("/service-worker.js"' in pwa_head
    assert 'scope: "/"' in pwa_head
    assert 'updateViaCache: "none"' in pwa_head
    assert 'document.readyState === "loading"' in pwa_head
    assert "registerServiceWorker();" in pwa_head


def test_optional_auth_protects_gradio_routes_without_leaking_password():
    password = "strong-demo-password"
    with gr.Blocks(title="Private PWA test") as demo:
        gr.Markdown("Private fixture")

    client = _ASGIClient(
        build_pwa_app(
            demo,
            host="127.0.0.1",
            port=7860,
            auth=("reviewer", password),
        )
    )

    root = client.get("/")
    assert root.status_code == 200
    assert password not in root.text
    assert _embedded_gradio_config(root.text)["auth_required"] is True
    assert client.get("/config").status_code == 401
    assert client.get("/gradio_api/info").status_code == 401
    assert client.get("/healthz").status_code == 200

    for blocked_path in (
        "/gradio_api/deep_link?session_hash=unknown",
        "/gradio_api/deep_link/",
        "/openapi.json",
        "/openapi.json/",
    ):
        blocked = client.get(blocked_path)
        assert blocked.status_code == 404
        assert blocked.headers["cache-control"] == ("no-store, private, max-age=0")
