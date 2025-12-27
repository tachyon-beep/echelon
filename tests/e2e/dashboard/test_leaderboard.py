"""Tests for arena dashboard leaderboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from playwright.sync_api import expect

if TYPE_CHECKING:
    from playwright.sync_api import Page


pytestmark = pytest.mark.e2e


def test_dashboard_loads(dashboard_page: Page) -> None:
    """Dashboard page loads successfully."""
    expect(dashboard_page.locator("h1")).to_contain_text("Echelon Arena")


def test_leaderboard_table_visible(dashboard_page: Page) -> None:
    """Leaderboard table is visible."""
    expect(dashboard_page.locator("#standings")).to_be_visible()


def test_leaderboard_has_data(dashboard_page: Page) -> None:
    """Leaderboard populates with commander data."""
    dashboard_page.wait_for_selector("#standings tbody tr", timeout=5000)

    rows = dashboard_page.locator("#standings tbody tr")
    expect(rows).to_have_count(1)  # Just heuristic


def test_leaderboard_columns(dashboard_page: Page) -> None:
    """Leaderboard shows expected columns."""
    dashboard_page.wait_for_selector("#standings tbody tr", timeout=5000)

    row = dashboard_page.locator("#standings tbody tr").first
    expect(row.locator(".rank")).to_be_visible()
    expect(row.locator(".commander-name")).to_be_visible()
    expect(row.locator(".rating")).to_be_visible()


def test_sparkline_canvas_exists(dashboard_page: Page) -> None:
    """Sparkline canvases are created for each commander."""
    dashboard_page.wait_for_selector("#standings tbody tr", timeout=5000)

    canvases = dashboard_page.locator("canvas.sparkline")
    expect(canvases).to_have_count(1)  # One for heuristic


def test_commander_name_displayed(dashboard_page: Page) -> None:
    """Commander names are displayed correctly."""
    dashboard_page.wait_for_selector("#standings tbody tr", timeout=5000)

    name = dashboard_page.locator(".commander-name").first
    expect(name).to_contain_text("Lieutenant")  # Heuristic name
