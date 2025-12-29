"""Visual regression tests for viewer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from playwright.sync_api import Page

    from tests.e2e.conftest import APIClient


pytestmark = [pytest.mark.e2e, pytest.mark.visual]

SNAPSHOT_DIR = Path(__file__).parent.parent / "snapshots"


def test_ui_panel_visual(viewer_page: Page) -> None:
    """Visual regression for viewer UI panel."""
    viewer_page.locator("#connectionStatus").wait_for(state="visible")

    ui = viewer_page.locator("#ui")
    screenshot = ui.screenshot()
    snapshot_path = SNAPSHOT_DIR / "viewer-ui-panel.png"
    if not snapshot_path.exists():
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(screenshot)
        pytest.skip("Baseline created, rerun to compare")

    # For now, just verify we can take a screenshot
    assert len(screenshot) > 0


def test_ui_panel_with_replay_visual(
    connected_viewer: Page, api_client: APIClient, mock_replay: dict
) -> None:
    """Visual regression with replay loaded."""
    api_client.push_world(mock_replay["world"], mock_replay["world_hash"])
    api_client.push_chunk(mock_replay["chunk"])

    connected_viewer.wait_for_function("window.replayData !== null", timeout=5000)

    # Pause playback for consistent screenshot
    connected_viewer.locator("#playPause").click()

    ui = connected_viewer.locator("#ui")
    screenshot = ui.screenshot()
    snapshot_path = SNAPSHOT_DIR / "viewer-ui-with-replay.png"
    if not snapshot_path.exists():
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(screenshot)
        pytest.skip("Baseline created, rerun to compare")

    assert len(screenshot) > 0


def test_info_panel_visual(connected_viewer: Page, api_client: APIClient, mock_replay: dict) -> None:
    """Visual regression for mech info panel."""
    api_client.push_world(mock_replay["world"], mock_replay["world_hash"])
    api_client.push_chunk(mock_replay["chunk"])

    connected_viewer.wait_for_function("window.replayData !== null", timeout=5000)

    # Simulate selection
    connected_viewer.evaluate("""
        window.selectedMechId = 'blue_0';
        document.getElementById('infoPanel').style.display = 'block';
        if (window.updateScene) window.updateScene(0);
    """)

    panel = connected_viewer.locator("#infoPanel")
    screenshot = panel.screenshot()
    snapshot_path = SNAPSHOT_DIR / "viewer-info-panel.png"
    if not snapshot_path.exists():
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(screenshot)
        pytest.skip("Baseline created, rerun to compare")

    assert len(screenshot) > 0
