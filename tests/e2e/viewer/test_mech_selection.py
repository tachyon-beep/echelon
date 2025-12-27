"""Tests for mech selection and info panel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from playwright.sync_api import expect

if TYPE_CHECKING:
    from playwright.sync_api import Page

    from tests.e2e.conftest import APIClient

from echelon.server.world_cache import WorldCache

pytestmark = pytest.mark.e2e


def test_info_panel_hidden_initially(viewer_page: Page) -> None:
    """Info panel hidden when no mech selected."""
    panel = viewer_page.locator("#infoPanel")
    expect(panel).to_be_hidden()


@pytest.mark.skip(reason="Info panel update requires internal viewer state not accessible via JS")
def test_info_panel_shows_on_selection(
    connected_viewer: Page, api_client: APIClient, mock_replay: dict
) -> None:
    """Info panel appears when mech selected via JS."""
    api_client.push_world(mock_replay["world"], mock_replay["world_hash"])
    api_client.push_chunk(mock_replay["chunk"])

    connected_viewer.wait_for_function("window.replayData !== null", timeout=5000)

    # Simulate selection (3D raycasting not testable)
    connected_viewer.evaluate("""
        window.selectedMechId = 'blue_0';
        document.getElementById('infoPanel').style.display = 'block';
        if (window.updateScene) window.updateScene(0);
    """)

    panel = connected_viewer.locator("#infoPanel")
    expect(panel).to_be_visible()
    expect(connected_viewer.locator("#selId")).to_contain_text("blue_0")
    expect(connected_viewer.locator("#selClass")).to_contain_text("medium")


@pytest.mark.skip(reason="Info panel update requires internal viewer state not accessible via JS")
def test_all_info_panel_fields(connected_viewer: Page, api_client: APIClient, mock_replay: dict) -> None:
    """All info panel fields display correctly."""
    api_client.push_world(mock_replay["world"], mock_replay["world_hash"])
    api_client.push_chunk(mock_replay["chunk"])

    connected_viewer.wait_for_function("window.replayData !== null", timeout=5000)

    connected_viewer.evaluate("""
        window.selectedMechId = 'blue_0';
        document.getElementById('infoPanel').style.display = 'block';
        if (window.updateScene) window.updateScene(0);
    """)

    expect(connected_viewer.locator("#selHp")).to_contain_text("100")
    expect(connected_viewer.locator("#selMaxHp")).to_contain_text("100")
    expect(connected_viewer.locator("#selHeat")).to_contain_text("0")
    expect(connected_viewer.locator("#selStab")).to_contain_text("100")
    expect(connected_viewer.locator("#selLegged")).to_contain_text("no")
    expect(connected_viewer.locator("#selState")).to_contain_text("Active")


@pytest.mark.skip(reason="Info panel update requires internal viewer state not accessible via JS")
def test_selection_updates_on_time_change(
    connected_viewer: Page, api_client: APIClient, mock_world: dict
) -> None:
    """Info panel updates when time changes."""
    # Create frames with changing HP
    frames = []
    for i in range(5):
        frames.append(
            {
                "t": i * 0.1,
                "mechs": {
                    "blue_0": {
                        "pos": [25.0, 25.0, 1.0],
                        "yaw": 0.0,
                        "hp": 100.0 - i * 10,  # HP decreases
                        "hp_max": 100.0,
                        "heat": 0.0,
                        "heat_cap": 100.0,
                        "stability": 100.0,
                        "stability_max": 100.0,
                        "class": "medium",
                        "team": "blue",
                        "alive": True,
                        "fallen": False,
                        "legged": False,
                        "ecm_on": False,
                        "eccm_on": False,
                    },
                },
                "objective": {"zone_center": [50.0, 50.0], "zone_radius": 15.0},
                "events": [],
                "smoke_clouds": [],
            }
        )

    world_hash = WorldCache.compute_hash(mock_world)
    api_client.push_world(mock_world, world_hash)
    api_client.push_chunk(
        {
            "replay_id": "hp-test",
            "world_ref": world_hash,
            "chunk_index": 0,
            "chunk_count": 1,
            "frames": frames,
            "meta": {},
        }
    )

    connected_viewer.wait_for_function("window.replayData !== null", timeout=5000)
    connected_viewer.locator("#playPause").click()  # Pause

    # Select mech and check initial HP
    connected_viewer.evaluate("""
        window.selectedMechId = 'blue_0';
        document.getElementById('infoPanel').style.display = 'block';
        window.currentTime = 0;
        window.updateScene(0);
    """)
    expect(connected_viewer.locator("#selHp")).to_contain_text("100")

    # Change time and verify HP updates
    connected_viewer.evaluate("window.currentTime = 0.4; window.updateScene(0.4);")
    expect(connected_viewer.locator("#selHp")).to_contain_text("60")
