# tests/unit/server/test_league_routes.py
"""Tests for league API endpoints."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from echelon.arena.league import League
from echelon.server import create_app


@pytest.fixture
def app_with_league(tmp_path: Path):
    """Create app with a test league."""
    league_path = tmp_path / "league.json"
    league = League()
    league.add_heuristic()
    league.save(league_path)

    app = create_app(league_path=league_path, matches_path=tmp_path / "matches")
    return TestClient(app)


def test_get_league(app_with_league):
    """GET /api/league returns league data."""
    response = app_with_league.get("/api/league")
    assert response.status_code == 200
    data = response.json()
    assert "entries" in data
    assert "heuristic" in data["entries"]


def test_get_standings(app_with_league):
    """GET /api/league/standings returns sorted leaderboard."""
    response = app_with_league.get("/api/league/standings")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1  # At least heuristic
    assert "entry_id" in data[0]
    assert "rating" in data[0]


def test_get_commander(app_with_league):
    """GET /api/commanders/{id} returns commander profile."""
    response = app_with_league.get("/api/commanders/heuristic")
    assert response.status_code == 200
    data = response.json()
    assert data["entry_id"] == "heuristic"
    assert data["commander_name"] == "Lieutenant Heuristic"
    assert "rating" in data
    assert "rating_history" in data
    assert "aggregate_stats" in data


def test_get_commander_not_found(app_with_league):
    """GET /api/commanders/{id} returns 404 for unknown commander."""
    response = app_with_league.get("/api/commanders/nonexistent")
    assert response.status_code == 404
