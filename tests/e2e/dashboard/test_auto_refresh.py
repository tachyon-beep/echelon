"""Tests for dashboard auto-refresh."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from playwright.sync_api import expect

if TYPE_CHECKING:
    from playwright.sync_api import Page


pytestmark = pytest.mark.e2e


def test_refresh_indicator_visible(dashboard_page: Page) -> None:
    """Auto-refresh indicator is visible."""
    indicator = dashboard_page.locator(".status-dot")
    expect(indicator).to_be_visible()


def test_last_updated_shows(dashboard_page: Page) -> None:
    """Last updated timestamp is displayed."""
    timestamp = dashboard_page.locator("#last-updated")
    expect(timestamp).to_be_visible()
    expect(timestamp).to_contain_text("Last updated")


def test_manual_refresh(dashboard_page: Page) -> None:
    """Manual refresh triggers API calls (timestamp may not change if same second)."""
    # Wait a second to ensure timestamp will change
    dashboard_page.wait_for_timeout(1100)

    timestamp = dashboard_page.locator("#last-updated")
    initial = timestamp.text_content()

    dashboard_page.evaluate("refresh()")
    dashboard_page.wait_for_timeout(500)

    # The timestamp should have changed since we waited over a second
    expect(timestamp).not_to_have_text(initial)
