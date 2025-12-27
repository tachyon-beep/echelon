"""Tests for SSE streaming behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from playwright.sync_api import expect

if TYPE_CHECKING:
    from playwright.sync_api import Page

    from tests.e2e.conftest import APIClient

from echelon.server.world_cache import WorldCache

pytestmark = pytest.mark.e2e


def test_replay_auto_plays_on_first_chunk(
    connected_viewer: Page, api_client: APIClient, mock_replay: dict
) -> None:
    """Replay auto-plays when first chunk arrives."""
    api_client.push_world(mock_replay["world"], mock_replay["world_hash"])
    api_client.push_chunk(mock_replay["chunk"])

    connected_viewer.wait_for_function("window.replayData !== null", timeout=5000)

    btn = connected_viewer.locator("#playPause")
    expect(btn).to_contain_text("Pause")


@pytest.mark.skip(reason="Multi-chunk streaming timing is inconsistent in test environment")
def test_replay_extends_during_streaming(
    connected_viewer: Page,
    api_client: APIClient,
    mock_world: dict,
    mock_frames: list[dict],
) -> None:
    """Playback extends as new chunks arrive."""
    world_hash = WorldCache.compute_hash(mock_world)
    api_client.push_world(mock_world, world_hash)

    # Push first chunk
    chunk1 = {
        "replay_id": "extend-test",
        "world_ref": world_hash,
        "chunk_index": 0,
        "chunk_count": 2,
        "frames": mock_frames[:2],
        "meta": {},
    }
    api_client.push_chunk(chunk1)

    connected_viewer.wait_for_function("window.replayData !== null", timeout=5000)

    # Get initial max time
    initial_max = connected_viewer.evaluate("window.maxTime")

    # Push second chunk with more frames
    extended_frames = [
        {**mock_frames[-1], "t": 1.0},
        {**mock_frames[-1], "t": 1.5},
    ]
    chunk2 = {
        "replay_id": "extend-test",
        "world_ref": world_hash,
        "chunk_index": 1,
        "chunk_count": 2,
        "frames": extended_frames,
        "meta": {},
    }
    api_client.push_chunk(chunk2)

    # Max time should increase
    connected_viewer.wait_for_function(f"window.maxTime > {initial_max}", timeout=5000)


@pytest.mark.skip(reason="Replay replacement timing is inconsistent in test environment")
def test_new_replay_replaces_old(connected_viewer: Page, api_client: APIClient, mock_replay: dict) -> None:
    """New replay_start replaces previous replay."""
    api_client.push_world(mock_replay["world"], mock_replay["world_hash"])
    api_client.push_chunk(mock_replay["chunk"])

    connected_viewer.wait_for_function("window.replayData !== null", timeout=5000)

    first_id = connected_viewer.evaluate("window.currentReplay.id")

    # Push new replay
    new_chunk = {**mock_replay["chunk"], "replay_id": "new-replay"}
    api_client.push_chunk(new_chunk)

    connected_viewer.wait_for_function(
        f"window.currentReplay && window.currentReplay.id !== '{first_id}'",
        timeout=5000,
    )


@pytest.mark.skip(reason="Network interception not reliable in test environment")
def test_connection_reconnects_after_error(connected_viewer: Page) -> None:
    """SSE connection reconnects after interruption."""
    status = connected_viewer.locator("#connectionStatus")
    # Already connected via connected_viewer fixture

    # Simulate network error
    connected_viewer.route("**/events", lambda route: route.abort())
    expect(status).not_to_have_class("connected", timeout=5000)

    # Remove route to allow reconnection
    connected_viewer.unroute("**/events")
    expect(status).to_have_class("connected", timeout=15000)
