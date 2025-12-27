"""Tests for matchup overlay."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from playwright.sync_api import expect

if TYPE_CHECKING:
    from playwright.sync_api import Page

    from tests.e2e.conftest import APIClient


pytestmark = pytest.mark.e2e


def test_matchup_overlay_hidden_initially(viewer_page: Page) -> None:
    """Overlay hidden when no active matchup."""
    overlay = viewer_page.locator("#matchup-overlay")
    expect(overlay).to_have_class(re.compile(r"hidden"))


def test_matchup_overlay_shows_on_api(connected_viewer: Page, api_client: APIClient) -> None:
    """Overlay shows when matchup set via API."""
    api_client.set_matchup("alpha", 1500.0, "beta", 1450.0)

    overlay = connected_viewer.locator("#matchup-overlay")
    expect(overlay).not_to_have_class(re.compile(r"hidden"), timeout=10000)

    expect(connected_viewer.locator("#blue-name")).to_contain_text("alpha")
    expect(connected_viewer.locator("#red-name")).to_contain_text("beta")


def test_matchup_overlay_m_key_toggle(connected_viewer: Page, api_client: APIClient) -> None:
    """'M' key toggles overlay visibility."""
    api_client.set_matchup("alpha", 1500.0, "beta", 1450.0)

    overlay = connected_viewer.locator("#matchup-overlay")
    expect(overlay).not_to_have_class(re.compile(r"hidden"), timeout=10000)

    connected_viewer.keyboard.press("m")
    expect(overlay).to_have_class(re.compile(r"hidden"))

    connected_viewer.keyboard.press("M")
    expect(overlay).not_to_have_class(re.compile(r"hidden"))
