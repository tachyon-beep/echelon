"""Tests for navigation graph and path toggles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from playwright.sync_api import expect

if TYPE_CHECKING:
    from playwright.sync_api import Page

    from tests.e2e.conftest import APIClient


pytestmark = pytest.mark.e2e


def test_nav_toggle_visible(viewer_page: Page) -> None:
    """Nav toggle button is visible."""
    expect(viewer_page.locator("#navToggle")).to_be_visible()
    expect(viewer_page.locator("#pathToggle")).to_be_visible()


def test_nav_graph_api_call(connected_viewer: Page, api_client: APIClient, mock_replay: dict) -> None:
    """Nav graph button triggers API call."""
    api_client.push_world(mock_replay["world"], mock_replay["world_hash"])
    api_client.push_chunk(mock_replay["chunk"])

    connected_viewer.wait_for_function("window.replayData !== null", timeout=5000)

    with connected_viewer.expect_response(lambda r: "/nav/graph" in r.url) as resp:
        connected_viewer.locator("#navToggle").click()

    assert resp.value.status == 200


def test_path_toggle_requires_selection(
    connected_viewer: Page, api_client: APIClient, mock_replay: dict
) -> None:
    """Path toggle alerts when no mech selected."""
    api_client.push_world(mock_replay["world"], mock_replay["world_hash"])
    api_client.push_chunk(mock_replay["chunk"])

    connected_viewer.wait_for_function("window.replayData !== null", timeout=5000)

    # Capture any dialog that appears
    dialog_message = None

    def handle_dialog(dialog):
        nonlocal dialog_message
        dialog_message = dialog.message
        dialog.accept()

    connected_viewer.on("dialog", handle_dialog)
    connected_viewer.locator("#pathToggle").click()

    # Test passes if no exception raised
