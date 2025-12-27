"""Tests for match history display."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from playwright.sync_api import expect

if TYPE_CHECKING:
    from playwright.sync_api import Page


pytestmark = pytest.mark.e2e


def test_matches_section_exists(dashboard_page: Page) -> None:
    """Matches section exists in DOM."""
    # Wait for dashboard data to load first
    dashboard_page.wait_for_selector("#standings tbody tr", timeout=5000)
    # Empty <ul> has zero height, so check it exists rather than is visible
    expect(dashboard_page.locator("#matches")).to_be_attached()


def test_empty_matches_state(dashboard_page: Page) -> None:
    """Shows empty state when no matches."""
    # With fresh league, no matches exist
    empty = dashboard_page.locator("#matches-empty")
    expect(empty).to_be_visible(timeout=5000)
