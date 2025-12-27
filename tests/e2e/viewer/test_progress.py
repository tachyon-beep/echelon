"""Tests for replay loading progress bar."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from playwright.sync_api import expect

if TYPE_CHECKING:
    from playwright.sync_api import Page

    from tests.e2e.conftest import APIClient

from echelon.server.world_cache import WorldCache

pytestmark = pytest.mark.e2e


def test_progress_hidden_initially(viewer_page: Page) -> None:
    """Progress bar hidden when no replay loading."""
    progress = viewer_page.locator("#progress")
    expect(progress).to_be_hidden()


def test_progress_shows_during_multi_chunk(
    connected_viewer: Page,
    api_client: APIClient,
    mock_world: dict,
    mock_frames: list[dict],
) -> None:
    """Progress bar visible during multi-chunk loading."""
    world_hash = WorldCache.compute_hash(mock_world)
    api_client.push_world(mock_world, world_hash)

    # Push first chunk of multi-chunk replay
    chunk = {
        "replay_id": "multi-test",
        "world_ref": world_hash,
        "chunk_index": 0,
        "chunk_count": 3,
        "frames": mock_frames[:2],
        "meta": {},
    }
    api_client.push_chunk(chunk)

    progress = connected_viewer.locator("#progress")
    expect(progress).to_be_visible(timeout=5000)
    expect(connected_viewer.locator("#progressText")).to_contain_text("1/3")


def test_progress_updates_per_chunk(
    connected_viewer: Page,
    api_client: APIClient,
    mock_world: dict,
    mock_frames: list[dict],
) -> None:
    """Progress text updates as chunks arrive."""
    world_hash = WorldCache.compute_hash(mock_world)
    api_client.push_world(mock_world, world_hash)

    for i in range(3):
        chunk = {
            "replay_id": "progress-test",
            "world_ref": world_hash,
            "chunk_index": i,
            "chunk_count": 3,
            "frames": mock_frames[i : i + 1] if i < len(mock_frames) else mock_frames[-1:],
            "meta": {} if i > 0 else {"test": True},
        }
        api_client.push_chunk(chunk)

    # After final chunk, progress should hide
    progress = connected_viewer.locator("#progress")
    expect(progress).to_be_hidden(timeout=5000)
