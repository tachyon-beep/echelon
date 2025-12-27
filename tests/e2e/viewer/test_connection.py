"""Tests for SSE connection status."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import requests
from playwright.sync_api import expect

if TYPE_CHECKING:
    from playwright.sync_api import Page


pytestmark = pytest.mark.e2e


def test_connection_status_visible(viewer_page: Page) -> None:
    """Connection status indicator is visible."""
    status = viewer_page.locator("#connectionStatus")
    expect(status).to_be_visible()


def test_connection_establishes(viewer_page: Page) -> None:
    """SSE connection shows connected status."""
    status = viewer_page.locator("#connectionStatus")
    expect(status).to_have_class("connected", timeout=5000)


def test_connection_shows_connected_text(viewer_page: Page) -> None:
    """Connected status shows correct text."""
    status = viewer_page.locator("#connectionStatus")
    expect(status).to_have_class("connected", timeout=5000)
    expect(status).to_contain_text("Connected")


def test_ui_panel_visible(viewer_page: Page) -> None:
    """Main UI panel is visible."""
    ui = viewer_page.locator("#ui")
    expect(ui).to_be_visible()
    expect(viewer_page.locator("#title")).to_contain_text("ECHELON")


def test_health_endpoint_accessible(live_server: str) -> None:
    """Health endpoint returns OK."""
    resp = requests.get(f"{live_server}/health", timeout=5.0)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "uptime_s" in data


def test_static_files_served(viewer_page: Page, live_server: str) -> None:
    """Static HTML files are served correctly."""
    # Viewer already loaded, check dashboard
    viewer_page.goto(f"{live_server}/dashboard.html")
    expect(viewer_page.locator("h1")).to_contain_text("Echelon")

    # Back to viewer
    viewer_page.goto(live_server)
    expect(viewer_page.locator("#title")).to_contain_text("ECHELON")


def test_header_elements_present(viewer_page: Page) -> None:
    """Header contains required elements."""
    expect(viewer_page.locator("#header")).to_be_visible()
    expect(viewer_page.locator("#title")).to_be_visible()
    expect(viewer_page.locator("#connectionStatus")).to_be_visible()
