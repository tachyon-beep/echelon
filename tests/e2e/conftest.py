"""E2E test fixtures for Playwright browser testing."""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import requests

if TYPE_CHECKING:
    from collections.abc import Generator

    from playwright.sync_api import Page

from echelon.arena.league import League
from echelon.server.world_cache import WorldCache


def _find_free_port() -> int:
    """Find an available port for the test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: float = 10.0) -> None:
    """Wait for server to be ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{url}/health", timeout=1.0)
            if resp.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.1)
    raise TimeoutError(f"Server at {url} not ready after {timeout}s")


@pytest.fixture(scope="session")
def test_league(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Create test league and matches directory."""
    base = tmp_path_factory.mktemp("arena")
    league_path = base / "league.json"
    matches_path = base / "matches"
    matches_path.mkdir()

    league = League()
    league.add_heuristic()
    league.save(league_path)

    return league_path, matches_path


@pytest.fixture(scope="function")
def live_server(test_league: tuple[Path, Path]) -> Generator[str]:
    """Launch uvicorn server for E2E tests."""
    league_path, matches_path = test_league
    port = _find_free_port()

    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "python",
            "-m",
            "echelon.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--league",
            str(league_path),
            "--matches",
            str(matches_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(Path(__file__).parent.parent.parent),
    )

    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_server(base_url)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@pytest.fixture
def viewer_page(page: Page, live_server: str) -> Page:
    """Navigate to viewer."""
    page.goto(live_server)
    return page


@pytest.fixture
def connected_viewer(page: Page, live_server: str) -> Page:
    """Navigate to viewer and wait for SSE connection."""
    from playwright.sync_api import expect

    page.goto(live_server)
    expect(page.locator("#connectionStatus")).to_have_class("connected", timeout=5000)
    return page


@pytest.fixture
def dashboard_page(page: Page, live_server: str) -> Page:
    """Navigate to dashboard."""
    page.goto(f"{live_server}/dashboard.html")
    return page


class APIClient:
    """HTTP client for pushing test data to the server."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def push_world(self, world: dict, world_hash: str) -> requests.Response:
        """Push world data to server."""
        return requests.put(f"{self.base_url}/worlds/{world_hash}", json=world, timeout=5.0)

    def push_chunk(self, chunk: dict) -> requests.Response:
        """Push replay chunk to server."""
        return requests.post(f"{self.base_url}/push", json=chunk, timeout=5.0)

    def set_matchup(
        self, blue_id: str, blue_rating: float, red_id: str, red_rating: float
    ) -> requests.Response:
        """Set current matchup for viewer overlay."""
        return requests.post(
            f"{self.base_url}/api/live/matchup",
            json={
                "blue_entry_id": blue_id,
                "blue_rating": blue_rating,
                "red_entry_id": red_id,
                "red_rating": red_rating,
            },
            timeout=5.0,
        )

    def clear_matchup(self) -> requests.Response:
        """Clear current matchup."""
        return requests.delete(f"{self.base_url}/api/live/matchup", timeout=5.0)


@pytest.fixture
def api_client(live_server: str) -> APIClient:
    """HTTP client for pushing test data."""
    return APIClient(live_server)


@pytest.fixture
def mock_world() -> dict:
    """Minimal world for testing."""
    return {
        "size": [100, 100, 20],
        "walls": [[50, 50, 0, 1]],
        "meta": {"capture_zone": {"center": [50.0, 50.0], "radius": 15.0}},
    }


@pytest.fixture
def mock_frames() -> list[dict]:
    """Minimal replay frames."""
    frames = []
    for i in range(5):
        frames.append(
            {
                "t": i * 0.1,
                "mechs": {
                    "blue_0": {
                        "pos": [25.0 + i, 25.0, 1.0],
                        "yaw": 0.0,
                        "hp": 100.0,
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
    return frames


@pytest.fixture
def mock_replay(mock_world: dict, mock_frames: list[dict]) -> dict:
    """Complete replay payload."""
    world_hash = WorldCache.compute_hash(mock_world)
    return {
        "world": mock_world,
        "world_hash": world_hash,
        "chunk": {
            "replay_id": "test-replay",
            "world_ref": world_hash,
            "chunk_index": 0,
            "chunk_count": 1,
            "frames": mock_frames,
            "meta": {},
        },
    }


def push_replay(api_client: APIClient, replay: dict) -> None:
    """Helper to push a complete replay to the server."""
    api_client.push_world(replay["world"], replay["world_hash"])
    api_client.push_chunk(replay["chunk"])
