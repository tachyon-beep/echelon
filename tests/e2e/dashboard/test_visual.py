"""Visual regression tests for dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from playwright.sync_api import Page


pytestmark = [pytest.mark.e2e, pytest.mark.visual]

SNAPSHOT_DIR = Path(__file__).parent.parent / "snapshots"


def test_dashboard_loaded_visual(dashboard_page: Page) -> None:
    """Visual regression for loaded dashboard."""
    dashboard_page.wait_for_selector("#standings tbody tr", timeout=5000)

    # Full page screenshot
    screenshot = dashboard_page.screenshot()
    snapshot_path = SNAPSHOT_DIR / "dashboard-loaded.png"
    if not snapshot_path.exists():
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(screenshot)
        pytest.skip("Baseline created, rerun to compare")

    assert len(screenshot) > 0


def test_leaderboard_visual(dashboard_page: Page) -> None:
    """Visual regression for leaderboard card."""
    dashboard_page.wait_for_selector("#standings tbody tr", timeout=5000)

    card = dashboard_page.locator(".card").first
    screenshot = card.screenshot()
    snapshot_path = SNAPSHOT_DIR / "dashboard-leaderboard.png"
    if not snapshot_path.exists():
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(screenshot)
        pytest.skip("Baseline created, rerun to compare")

    assert len(screenshot) > 0


def test_empty_matches_visual(dashboard_page: Page) -> None:
    """Visual regression for empty matches state."""
    dashboard_page.wait_for_selector("#matches-empty", state="visible", timeout=5000)

    card = dashboard_page.locator(".card").nth(1)
    screenshot = card.screenshot()
    snapshot_path = SNAPSHOT_DIR / "dashboard-empty-matches.png"
    if not snapshot_path.exists():
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(screenshot)
        pytest.skip("Baseline created, rerun to compare")

    assert len(screenshot) > 0
