# tests/unit/server/test_live_routes.py
"""Tests for live training status endpoints."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from echelon.arena.league import League
from echelon.server import create_app


@pytest.fixture
def app_client(tmp_path: Path):
    """Create app with test league."""
    league_path = tmp_path / "league.json"
    league = League()
    league.add_heuristic()
    league.save(league_path)

    app = create_app(league_path=league_path, matches_path=tmp_path / "matches")
    return TestClient(app)


def test_get_live_matchup_no_training(app_client):
    """GET /api/live/matchup returns null when not training."""
    response = app_client.get("/api/live/matchup")
    assert response.status_code == 200
    data = response.json()
    assert data["active"] is False


def test_set_and_get_live_matchup(app_client):
    """Live matchup can be set and retrieved."""
    # POST to set current matchup (internal endpoint)
    response = app_client.post(
        "/api/live/matchup",
        json={
            "blue_entry_id": "contender",
            "blue_rating": 1450.0,
            "red_entry_id": "heuristic",
            "red_rating": 1500.0,
        },
    )
    assert response.status_code == 200

    # GET to retrieve
    response = app_client.get("/api/live/matchup")
    assert response.status_code == 200
    data = response.json()
    assert data["active"] is True
    assert data["blue_entry_id"] == "contender"
    assert data["red_entry_id"] == "heuristic"


def test_clear_live_matchup(app_client):
    """DELETE /api/live/matchup clears the current matchup."""
    # Set a matchup first
    app_client.post(
        "/api/live/matchup",
        json={
            "blue_entry_id": "contender",
            "blue_rating": 1450.0,
            "red_entry_id": "heuristic",
            "red_rating": 1500.0,
        },
    )

    # Clear it
    response = app_client.delete("/api/live/matchup")
    assert response.status_code == 200

    # Verify it's cleared
    response = app_client.get("/api/live/matchup")
    assert response.status_code == 200
    data = response.json()
    assert data["active"] is False
