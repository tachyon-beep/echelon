# tests/unit/server/test_match_routes.py
"""Tests for match API endpoints."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from echelon.arena.history import MatchHistory
from echelon.arena.league import League
from echelon.arena.stats import MatchRecord, TeamStats
from echelon.server import create_app


@pytest.fixture
def app_with_matches(tmp_path: Path):
    """Create app with test league and matches."""
    league_path = tmp_path / "league.json"
    league = League()
    league.add_heuristic()
    league.save(league_path)

    matches_path = tmp_path / "matches"
    history = MatchHistory(matches_path)

    # Add some test matches
    for i in range(3):
        record = MatchRecord(
            match_id=f"match-{i:03d}",
            timestamp=1703750400.0 + i * 100,
            blue_entry_id="contender",
            red_entry_id="heuristic",
            winner="blue" if i % 2 == 0 else "red",
            duration_steps=500 + i * 10,
            termination="zone",
            blue_stats=TeamStats(kills=2 + i, deaths=1),
            red_stats=TeamStats(kills=1, deaths=2 + i),
        )
        history.save(record)

    app = create_app(league_path=league_path, matches_path=matches_path)
    return TestClient(app)


def test_get_matches(app_with_matches):
    """GET /api/matches returns recent matches."""
    response = app_with_matches.get("/api/matches")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    # Most recent first
    assert data[0]["match_id"] == "match-002"


def test_get_matches_filtered(app_with_matches):
    """GET /api/matches?entry_id=X filters by commander."""
    response = app_with_matches.get("/api/matches?entry_id=heuristic")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3  # All matches involve heuristic


def test_get_match_detail(app_with_matches):
    """GET /api/matches/{id} returns full match detail."""
    response = app_with_matches.get("/api/matches/match-001")
    assert response.status_code == 200
    data = response.json()
    assert data["match_id"] == "match-001"
    assert "blue_stats" in data
    assert "red_stats" in data


def test_get_match_not_found(app_with_matches):
    """GET /api/matches/{id} returns 404 for unknown match."""
    response = app_with_matches.get("/api/matches/nonexistent")
    assert response.status_code == 404
