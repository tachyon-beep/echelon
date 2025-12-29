"""Tests for match statistics data structures."""

from echelon.arena.stats import MatchRecord, TeamStats


def test_team_stats_defaults():
    """TeamStats initializes with zero values."""
    stats = TeamStats()
    assert stats.kills == 0
    assert stats.damage_dealt == 0.0
    assert stats.zone_ticks == 0
    assert stats.orders_issued == {}


def test_team_stats_weapon_counts():
    """TeamStats tracks weapon usage counts."""
    stats = TeamStats(primary_uses=10, secondary_uses=5, tertiary_uses=3)
    assert stats.primary_uses == 10
    assert stats.secondary_uses == 5
    assert stats.tertiary_uses == 3


def test_match_record_creation():
    """MatchRecord captures full match data."""
    blue = TeamStats(kills=3, deaths=2)
    red = TeamStats(kills=2, deaths=3)

    record = MatchRecord(
        match_id="test-123",
        timestamp=1703750400.0,
        blue_entry_id="contender",
        red_entry_id="heuristic",
        winner="blue",
        duration_steps=500,
        termination="zone",
        blue_stats=blue,
        red_stats=red,
    )

    assert record.winner == "blue"
    assert record.blue_stats.kills == 3
    assert record.termination == "zone"


def test_match_record_to_dict():
    """MatchRecord serializes to dict for JSON storage."""
    blue = TeamStats(kills=1)
    red = TeamStats(kills=2)

    record = MatchRecord(
        match_id="test-456",
        timestamp=1703750400.0,
        blue_entry_id="a",
        red_entry_id="b",
        winner="red",
        duration_steps=100,
        termination="elimination",
        blue_stats=blue,
        red_stats=red,
    )

    d = record.to_dict()
    assert d["match_id"] == "test-456"
    assert d["blue_stats"]["kills"] == 1
    assert d["red_stats"]["kills"] == 2


def test_match_record_from_dict():
    """MatchRecord deserializes from dict."""
    d = {
        "match_id": "test-789",
        "timestamp": 1703750400.0,
        "blue_entry_id": "x",
        "red_entry_id": "y",
        "winner": "draw",
        "duration_steps": 200,
        "termination": "timeout",
        "blue_stats": {"kills": 0, "deaths": 0},
        "red_stats": {"kills": 0, "deaths": 0},
    }

    record = MatchRecord.from_dict(d)
    assert record.match_id == "test-789"
    assert record.winner == "draw"
