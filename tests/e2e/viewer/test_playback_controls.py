"""Tests for playback controls."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from playwright.sync_api import expect

if TYPE_CHECKING:
    from playwright.sync_api import Page

    from tests.e2e.conftest import APIClient


pytestmark = pytest.mark.e2e


def test_playback_controls_visible(viewer_page: Page) -> None:
    """All playback controls are visible."""
    expect(viewer_page.locator("#playPause")).to_be_visible()
    expect(viewer_page.locator("#stepBack")).to_be_visible()
    expect(viewer_page.locator("#stepForward")).to_be_visible()
    expect(viewer_page.locator("#timeSlider")).to_be_visible()
    expect(viewer_page.locator("#speedSelect")).to_be_visible()


def test_speed_selector_options(viewer_page: Page) -> None:
    """Speed selector has correct options."""
    speed = viewer_page.locator("#speedSelect")
    expect(speed).to_have_value("1")

    speed.select_option("2")
    expect(speed).to_have_value("2")

    speed.select_option("0.5")
    expect(speed).to_have_value("0.5")


def test_play_pause_with_replay(connected_viewer: Page, api_client: APIClient, mock_replay: dict) -> None:
    """Play/pause toggles when replay loaded."""
    api_client.push_world(mock_replay["world"], mock_replay["world_hash"])
    api_client.push_chunk(mock_replay["chunk"])

    connected_viewer.wait_for_function("window.replayData !== null", timeout=5000)

    btn = connected_viewer.locator("#playPause")
    expect(btn).to_contain_text("Pause")  # Auto-plays

    btn.click()
    expect(btn).to_contain_text("Play")


def test_time_display_updates(connected_viewer: Page, api_client: APIClient, mock_replay: dict) -> None:
    """Time display shows current time."""
    api_client.push_world(mock_replay["world"], mock_replay["world_hash"])
    api_client.push_chunk(mock_replay["chunk"])

    connected_viewer.wait_for_function("window.replayData !== null", timeout=5000)

    display = connected_viewer.locator("#timeDisplay")
    expect(display).to_contain_text("/")  # Format: "X.Xs / Y.Ys"


def test_step_back_at_start(connected_viewer: Page, api_client: APIClient, mock_replay: dict) -> None:
    """Step back at start stays at beginning."""
    api_client.push_world(mock_replay["world"], mock_replay["world_hash"])
    api_client.push_chunk(mock_replay["chunk"])

    connected_viewer.wait_for_function("window.replayData !== null", timeout=5000)
    connected_viewer.locator("#playPause").click()  # Pause

    # Go to start
    connected_viewer.evaluate("window.currentTime = 0")
    connected_viewer.locator("#stepBack").click()

    time_val = connected_viewer.evaluate("window.currentTime")
    assert time_val == 0


@pytest.mark.skip(reason="Slider max update timing is inconsistent in test environment")
def test_slider_max_updates_with_replay(
    connected_viewer: Page, api_client: APIClient, mock_replay: dict
) -> None:
    """Slider max value updates when replay loads."""
    slider = connected_viewer.locator("#timeSlider")
    initial_max = slider.evaluate("el => el.max")

    api_client.push_world(mock_replay["world"], mock_replay["world_hash"])
    api_client.push_chunk(mock_replay["chunk"])

    connected_viewer.wait_for_function("window.replayData !== null", timeout=5000)

    new_max = slider.evaluate("el => el.max")
    assert float(new_max) > float(initial_max)


@pytest.mark.skip(reason="Playback loop timing depends on real animation frame timing")
def test_time_loops_at_end(connected_viewer: Page, api_client: APIClient, mock_replay: dict) -> None:
    """Playback loops back to start at end."""
    api_client.push_world(mock_replay["world"], mock_replay["world_hash"])
    api_client.push_chunk(mock_replay["chunk"])

    connected_viewer.wait_for_function("window.replayData !== null", timeout=5000)

    # Jump to near end
    connected_viewer.evaluate("window.currentTime = window.maxTime - 0.01")

    # Wait for loop (playback should reset to 0)
    connected_viewer.wait_for_function("window.currentTime < 0.1", timeout=5000)
