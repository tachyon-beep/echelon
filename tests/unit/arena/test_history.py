"""Tests for match history storage."""

from pathlib import Path

import pytest

from echelon.arena.history import MatchHistory
from echelon.arena.stats import MatchRecord, TeamStats


@pytest.fixture
def tmp_history(tmp_path: Path) -> MatchHistory:
    """Create a MatchHistory in a temp directory."""
    return MatchHistory(tmp_path / "matches")


def test_match_history_save_and_load(tmp_history: MatchHistory):
    """MatchHistory saves and loads records."""
    record = MatchRecord(
        match_id="test-001",
        timestamp=1703750400.0,
        blue_entry_id="a",
        red_entry_id="b",
        winner="blue",
        duration_steps=100,
        termination="zone",
        blue_stats=TeamStats(kills=2),
        red_stats=TeamStats(kills=1),
    )

    tmp_history.save(record)
    loaded = tmp_history.load("test-001")

    assert loaded is not None
    assert loaded.match_id == "test-001"
    assert loaded.blue_stats.kills == 2


def test_match_history_list_recent(tmp_history: MatchHistory):
    """MatchHistory lists recent matches sorted by timestamp."""
    for i in range(5):
        record = MatchRecord(
            match_id=f"test-{i:03d}",
            timestamp=1703750400.0 + i * 100,
            blue_entry_id="a",
            red_entry_id="b",
            winner="blue",
            duration_steps=100,
            termination="zone",
            blue_stats=TeamStats(),
            red_stats=TeamStats(),
        )
        tmp_history.save(record)

    recent = tmp_history.list_recent(limit=3)

    assert len(recent) == 3
    # Most recent first
    assert recent[0].match_id == "test-004"
    assert recent[1].match_id == "test-003"
    assert recent[2].match_id == "test-002"


def test_match_history_filter_by_entry(tmp_history: MatchHistory):
    """MatchHistory filters by entry_id."""
    records = [
        MatchRecord("m1", 100.0, "alice", "bob", "blue", 50, "zone", TeamStats(), TeamStats()),
        MatchRecord("m2", 200.0, "alice", "carol", "red", 50, "zone", TeamStats(), TeamStats()),
        MatchRecord("m3", 300.0, "bob", "carol", "blue", 50, "zone", TeamStats(), TeamStats()),
    ]
    for r in records:
        tmp_history.save(r)

    alice_matches = tmp_history.list_recent(entry_id="alice")
    assert len(alice_matches) == 2

    bob_matches = tmp_history.list_recent(entry_id="bob")
    assert len(bob_matches) == 2  # m1 and m3
