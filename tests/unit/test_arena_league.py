"""Arena League tests.

Tests for league management:
- Entry creation and serialization
- League serialization round-trip
- Checkpoint upsert and deduplication
- Commander ranking and promotion
- Rating period application
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from echelon.arena.glicko2 import GameResult, Glicko2Config, Glicko2Rating
from echelon.arena.league import (
    League,
    LeagueEntry,
    _commander_name,
    _now_iso,
    _stable_id,
)


class TestHelperFunctions:
    """Test helper functions."""

    def test_now_iso_format(self):
        """_now_iso returns valid ISO timestamp."""
        ts = _now_iso()
        assert isinstance(ts, str)
        # Format: YYYY-MM-DDTHH:MM:SSZ
        assert len(ts) == 20
        assert ts[4] == "-"
        assert ts[7] == "-"
        assert ts[10] == "T"
        assert ts[13] == ":"
        assert ts[16] == ":"
        assert ts.endswith("Z")

    def test_stable_id_uses_file_metadata(self):
        """_stable_id generates ID from file stem, size, mtime."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"test checkpoint data")
            f.flush()
            path = Path(f.name)

        try:
            sid = _stable_id(path)
            assert isinstance(sid, str)
            assert path.stem in sid
            # Format: stem:size:mtime
            parts = sid.split(":")
            assert len(parts) == 3
            assert parts[0] == path.stem
            assert int(parts[1]) == path.stat().st_size
        finally:
            path.unlink()

    def test_stable_id_deterministic(self):
        """Same file gives same ID."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"test data")
            f.flush()
            path = Path(f.name)

        try:
            id1 = _stable_id(path)
            id2 = _stable_id(path)
            assert id1 == id2
        finally:
            path.unlink()

    def test_commander_name_deterministic(self):
        """Same seed gives same name."""
        name1 = _commander_name(42)
        name2 = _commander_name(42)
        assert name1 == name2

    def test_commander_name_format(self):
        """Commander names have correct format."""
        name = _commander_name(123)
        # Format: "Adjective Noun #XXXX"
        parts = name.split()
        assert len(parts) == 3
        assert parts[2].startswith("#")
        assert len(parts[2]) == 5  # #XXXX

    def test_commander_name_varies_with_seed(self):
        """Different seeds give different names."""
        names = {_commander_name(i) for i in range(100)}
        # Should have variety (not all the same)
        assert len(names) > 10


class TestLeagueEntry:
    """Test LeagueEntry dataclass."""

    def test_entry_creation(self):
        """LeagueEntry can be created with defaults."""
        entry = LeagueEntry(
            entry_id="test-id",
            ckpt_path="/path/to/ckpt.pt",
            kind="candidate",
        )
        assert entry.entry_id == "test-id"
        assert entry.ckpt_path == "/path/to/ckpt.pt"
        assert entry.kind == "candidate"
        assert entry.commander_name is None
        assert entry.games == 0
        assert isinstance(entry.rating, Glicko2Rating)

    def test_entry_as_dict(self):
        """as_dict produces serializable dict."""
        entry = LeagueEntry(
            entry_id="test-id",
            ckpt_path="/path/to/ckpt.pt",
            kind="commander",
            commander_name="Iron Warden #1234",
            rating=Glicko2Rating(rating=1600, rd=75, vol=0.05),
            games=10,
            meta={"epoch": 100},
        )
        d = entry.as_dict()

        assert d["id"] == "test-id"
        assert d["ckpt_path"] == "/path/to/ckpt.pt"
        assert d["kind"] == "commander"
        assert d["commander_name"] == "Iron Warden #1234"
        assert d["games"] == 10
        assert d["meta"]["epoch"] == 100
        assert d["rating"]["rating"] == 1600

        # Should be JSON-serializable
        json.dumps(d)

    def test_entry_roundtrip(self):
        """from_dict(as_dict()) preserves data."""
        cfg = Glicko2Config()
        original = LeagueEntry(
            entry_id="test-id",
            ckpt_path="/path/to/ckpt.pt",
            kind="commander",
            commander_name="Silent Viper #abcd",
            rating=Glicko2Rating(rating=1650, rd=80, vol=0.055),
            games=25,
            meta={"foo": "bar"},
        )

        d = original.as_dict()
        restored = LeagueEntry.from_dict(d, cfg=cfg)

        assert restored.entry_id == original.entry_id
        assert restored.ckpt_path == original.ckpt_path
        assert restored.kind == original.kind
        assert restored.commander_name == original.commander_name
        assert restored.games == original.games
        assert restored.meta == original.meta
        assert abs(restored.rating.rating - original.rating.rating) < 0.01
        assert abs(restored.rating.rd - original.rating.rd) < 0.01

    def test_entry_rating_history(self):
        """LeagueEntry tracks rating history over time."""
        entry = LeagueEntry(
            entry_id="test",
            ckpt_path="",
            kind="commander",
            commander_name="Test",
            rating=Glicko2Rating(),
            games=0,
        )

        # Initially empty
        assert entry.rating_history == []

        # Record rating
        entry.record_rating(1703750400.0)
        assert len(entry.rating_history) == 1
        assert entry.rating_history[0] == (1703750400.0, 1500.0)

        # Update rating and record again
        entry.rating = Glicko2Rating(rating=1550.0)
        entry.record_rating(1703750500.0)
        assert len(entry.rating_history) == 2
        assert entry.rating_history[1] == (1703750500.0, 1550.0)

    def test_entry_serializes_rating_history(self):
        """Rating history survives serialization round-trip."""
        cfg = Glicko2Config()
        entry = LeagueEntry(
            entry_id="test",
            ckpt_path="",
            kind="commander",
            commander_name="Test",
            rating=Glicko2Rating(),
            games=0,
            rating_history=[(1703750400.0, 1500.0), (1703750500.0, 1520.0)],
        )

        d = entry.as_dict()
        restored = LeagueEntry.from_dict(d, cfg=cfg)

        assert restored.rating_history == entry.rating_history


class TestLeague:
    """Test League class."""

    def test_league_creation(self):
        """League can be created with defaults."""
        league = League()
        assert isinstance(league.cfg, Glicko2Config)
        assert len(league.entries) == 0

    def test_league_as_dict(self):
        """as_dict produces serializable dict."""
        league = League()
        league.entries["e1"] = LeagueEntry(entry_id="e1", ckpt_path="/a.pt", kind="candidate")
        d = league.as_dict()

        assert d["schema_version"] == 1
        assert "glicko2" in d
        assert len(d["entries"]) == 1

        # Should be JSON-serializable
        json.dumps(d)

    def test_league_roundtrip(self):
        """from_dict(as_dict()) preserves data."""
        original = League(
            cfg=Glicko2Config(tau=0.4, rating0=1400),
            env_signature={"obs_mode": "full"},
        )
        original.entries["e1"] = LeagueEntry(
            entry_id="e1",
            ckpt_path="/a.pt",
            kind="commander",
            commander_name="Test Commander #0000",
            rating=Glicko2Rating(rating=1700, rd=60, vol=0.05),
            games=50,
        )

        d = original.as_dict()
        restored = League.from_dict(d)

        assert abs(restored.cfg.tau - original.cfg.tau) < 0.01
        assert abs(restored.cfg.rating0 - original.cfg.rating0) < 0.01
        assert restored.env_signature == original.env_signature
        assert len(restored.entries) == 1
        assert "e1" in restored.entries

    def test_league_save_load(self):
        """save() and load() work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "league.json"

            original = League()
            original.entries["e1"] = LeagueEntry(entry_id="e1", ckpt_path="/a.pt", kind="commander")
            original.save(path)

            assert path.exists()

            restored = League.load(path)
            assert len(restored.entries) == 1
            assert "e1" in restored.entries


class TestUpsertCheckpoint:
    """Test checkpoint upsert functionality."""

    def test_upsert_creates_new_entry(self):
        """upsert_checkpoint creates new entry for unknown checkpoint."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"checkpoint data")
            path = Path(f.name)

        try:
            league = League()
            entry = league.upsert_checkpoint(path, kind="candidate")

            assert entry.kind == "candidate"
            assert str(path.resolve()) in entry.ckpt_path
            assert len(league.entries) == 1
            assert entry.rating.rating == league.cfg.rating0
        finally:
            path.unlink()

    def test_upsert_returns_existing(self):
        """upsert_checkpoint returns existing entry for same checkpoint."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"checkpoint data")
            path = Path(f.name)

        try:
            league = League()
            e1 = league.upsert_checkpoint(path, kind="candidate")
            e1.games = 10  # Modify

            e2 = league.upsert_checkpoint(path, kind="candidate")

            assert e1 is e2
            assert e2.games == 10
            assert len(league.entries) == 1
        finally:
            path.unlink()

    def test_upsert_never_demotes_commander(self):
        """upsert_checkpoint never demotes commander to candidate."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"checkpoint data")
            path = Path(f.name)

        try:
            league = League()
            e1 = league.upsert_checkpoint(path, kind="commander")
            assert e1.kind == "commander"

            # Try to "demote" back to candidate
            e2 = league.upsert_checkpoint(path, kind="candidate")

            assert e2.kind == "commander"  # Should stay commander
        finally:
            path.unlink()

    def test_upsert_can_promote_to_commander(self):
        """upsert_checkpoint can promote candidate to commander."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"checkpoint data")
            path = Path(f.name)

        try:
            league = League()
            e1 = league.upsert_checkpoint(path, kind="candidate")
            assert e1.kind == "candidate"

            e2 = league.upsert_checkpoint(path, kind="commander")
            assert e2.kind == "commander"
        finally:
            path.unlink()


class TestRankingAndPromotion:
    """Test ranking and promotion functionality."""

    def test_top_commanders_empty(self):
        """top_commanders returns empty for empty league."""
        league = League()
        assert league.top_commanders(5) == []

    def test_top_commanders_sorts_by_rating(self):
        """top_commanders returns sorted by rating descending."""
        league = League()
        for i, rating in enumerate([1400, 1600, 1500]):
            league.entries[f"e{i}"] = LeagueEntry(
                entry_id=f"e{i}",
                ckpt_path=f"/{i}.pt",
                kind="commander",
                rating=Glicko2Rating(rating=rating, rd=50, vol=0.06),
            )

        top = league.top_commanders(3)
        ratings = [e.rating.rating for e in top]
        assert ratings == [1600, 1500, 1400]

    def test_top_commanders_excludes_candidates(self):
        """top_commanders excludes candidates."""
        league = League()
        league.entries["cmd"] = LeagueEntry(
            entry_id="cmd",
            ckpt_path="/cmd.pt",
            kind="commander",
            rating=Glicko2Rating(rating=1500, rd=50, vol=0.06),
        )
        league.entries["cand"] = LeagueEntry(
            entry_id="cand",
            ckpt_path="/cand.pt",
            kind="candidate",
            rating=Glicko2Rating(rating=1800, rd=50, vol=0.06),  # Higher rating
        )

        top = league.top_commanders(5)
        assert len(top) == 1
        assert top[0].entry_id == "cmd"

    def test_recent_candidates_sorts_by_created_at(self):
        """recent_candidates returns sorted by created_at descending."""
        league = League()
        for i, ts in enumerate(["2024-01-01T00:00:00Z", "2024-01-03T00:00:00Z", "2024-01-02T00:00:00Z"]):
            league.entries[f"e{i}"] = LeagueEntry(
                entry_id=f"e{i}",
                ckpt_path=f"/{i}.pt",
                kind="candidate",
                created_at=ts,
            )

        recent = league.recent_candidates(3)
        timestamps = [e.created_at for e in recent]
        assert timestamps == [
            "2024-01-03T00:00:00Z",
            "2024-01-02T00:00:00Z",
            "2024-01-01T00:00:00Z",
        ]

    def test_recent_candidates_excludes_id(self):
        """recent_candidates can exclude specific ID."""
        league = League()
        for i in range(3):
            league.entries[f"e{i}"] = LeagueEntry(
                entry_id=f"e{i}",
                ckpt_path=f"/{i}.pt",
                kind="candidate",
            )

        recent = league.recent_candidates(10, exclude_id="e1")
        ids = [e.entry_id for e in recent]
        assert "e1" not in ids
        assert len(ids) == 2

    def test_promote_if_topk_promotes_when_qualified(self):
        """promote_if_topk promotes entry when in top k."""
        league = League()

        # Add some commanders
        for i in range(3):
            league.entries[f"cmd{i}"] = LeagueEntry(
                entry_id=f"cmd{i}",
                ckpt_path=f"/cmd{i}.pt",
                kind="commander",
                rating=Glicko2Rating(rating=1500 + i * 50, rd=30, vol=0.06),
            )

        # Add strong candidate
        league.entries["cand"] = LeagueEntry(
            entry_id="cand",
            ckpt_path="/cand.pt",
            kind="candidate",
            rating=Glicko2Rating(rating=1700, rd=30, vol=0.06),  # Strongest
        )

        promoted = league.promote_if_topk("cand", top_k=2)

        assert promoted is True
        assert league.entries["cand"].kind == "commander"
        assert league.entries["cand"].commander_name is not None

    def test_promote_if_topk_uses_conservative_rating(self):
        """promote_if_topk uses conservative rating (rating - 2*rd)."""
        league = League()

        # Commander with low RD (confident)
        league.entries["cmd"] = LeagueEntry(
            entry_id="cmd",
            ckpt_path="/cmd.pt",
            kind="commander",
            rating=Glicko2Rating(rating=1500, rd=30, vol=0.06),  # Conservative: 1440
        )

        # Candidate with high RD (uncertain) but higher rating
        league.entries["cand"] = LeagueEntry(
            entry_id="cand",
            ckpt_path="/cand.pt",
            kind="candidate",
            rating=Glicko2Rating(rating=1520, rd=100, vol=0.06),  # Conservative: 1320
        )

        # top_k=1: only room for 1, candidate's conservative rating is lower
        promoted = league.promote_if_topk("cand", top_k=1)

        assert promoted is False
        assert league.entries["cand"].kind == "candidate"

    def test_promote_if_topk_unknown_entry_raises(self):
        """promote_if_topk raises KeyError for unknown entry."""
        league = League()

        with pytest.raises(KeyError):
            league.promote_if_topk("nonexistent", top_k=5)

    def test_promote_assigns_commander_name(self):
        """Promotion assigns a commander name."""
        league = League()
        league.entries["cand"] = LeagueEntry(
            entry_id="cand",
            ckpt_path="/cand.pt",
            kind="candidate",
        )

        league.promote_if_topk("cand", top_k=1)

        assert league.entries["cand"].commander_name is not None
        assert "#" in league.entries["cand"].commander_name

    def test_promote_does_not_change_existing_name(self):
        """Re-promotion does not change existing commander name."""
        league = League()
        league.entries["cmd"] = LeagueEntry(
            entry_id="cmd",
            ckpt_path="/cmd.pt",
            kind="commander",
            commander_name="Custom Name #1234",
        )

        league.promote_if_topk("cmd", top_k=1)

        assert league.entries["cmd"].commander_name == "Custom Name #1234"


class TestRatingPeriod:
    """Test rating period application."""

    def test_apply_rating_period_updates_ratings(self):
        """apply_rating_period updates ratings correctly."""
        league = League()
        league.entries["p1"] = LeagueEntry(
            entry_id="p1",
            ckpt_path="/p1.pt",
            kind="commander",
            rating=Glicko2Rating(rating=1500, rd=50, vol=0.06),
            games=0,
        )
        league.entries["p2"] = LeagueEntry(
            entry_id="p2",
            ckpt_path="/p2.pt",
            kind="commander",
            rating=Glicko2Rating(rating=1500, rd=50, vol=0.06),
            games=0,
        )

        # Snapshot opponent ratings before applying
        opp_rating_p2 = Glicko2Rating(
            rating=league.entries["p2"].rating.rating,
            rd=league.entries["p2"].rating.rd,
            vol=league.entries["p2"].rating.vol,
        )
        opp_rating_p1 = Glicko2Rating(
            rating=league.entries["p1"].rating.rating,
            rd=league.entries["p1"].rating.rd,
            vol=league.entries["p1"].rating.vol,
        )

        # p1 beats p2
        results = {
            "p1": [GameResult(opponent=opp_rating_p2, score=1.0)],
            "p2": [GameResult(opponent=opp_rating_p1, score=0.0)],
        }

        league.apply_rating_period(results)

        # p1 should have higher rating now
        assert league.entries["p1"].rating.rating > 1500
        assert league.entries["p2"].rating.rating < 1500
        assert league.entries["p1"].games == 1
        assert league.entries["p2"].games == 1

    def test_apply_rating_period_ignores_unknown_entries(self):
        """apply_rating_period ignores results for unknown entries."""
        league = League()

        results = {
            "nonexistent": [GameResult(opponent=Glicko2Rating(), score=1.0)],
        }

        # Should not raise
        league.apply_rating_period(results)

    def test_apply_rating_period_two_phase_commit(self):
        """apply_rating_period uses two-phase commit (frozen opponent ratings)."""
        league = League()
        league.entries["p1"] = LeagueEntry(
            entry_id="p1",
            ckpt_path="/p1.pt",
            kind="commander",
            rating=Glicko2Rating(rating=1500, rd=50, vol=0.06),
        )
        league.entries["p2"] = LeagueEntry(
            entry_id="p2",
            ckpt_path="/p2.pt",
            kind="commander",
            rating=Glicko2Rating(rating=1500, rd=50, vol=0.06),
        )

        # Freeze ratings at start
        opp_p2_frozen = Glicko2Rating(rating=1500, rd=50, vol=0.06)
        opp_p1_frozen = Glicko2Rating(rating=1500, rd=50, vol=0.06)

        results = {
            "p1": [GameResult(opponent=opp_p2_frozen, score=1.0)],
            "p2": [GameResult(opponent=opp_p1_frozen, score=0.0)],
        }

        league.apply_rating_period(results)

        # Both updates should use frozen ratings, not updated ones
        # This means p1's gain and p2's loss should be symmetric
        delta_p1 = league.entries["p1"].rating.rating - 1500
        delta_p2 = 1500 - league.entries["p2"].rating.rating

        # Should be approximately equal (symmetric update)
        assert abs(delta_p1 - delta_p2) < 1.0

    def test_apply_rating_period_accumulates_games(self):
        """apply_rating_period accumulates game count."""
        league = League()
        league.entries["p1"] = LeagueEntry(
            entry_id="p1",
            ckpt_path="/p1.pt",
            kind="commander",
            games=5,
        )

        opp = Glicko2Rating()
        results = {
            "p1": [
                GameResult(opponent=opp, score=1.0),
                GameResult(opponent=opp, score=0.5),
                GameResult(opponent=opp, score=0.0),
            ],
        }

        league.apply_rating_period(results)

        assert league.entries["p1"].games == 8  # 5 + 3


class TestCommanderRetirement:
    """Test commander retirement functionality."""

    def test_retire_stale_commanders(self):
        """Test that old, low-rated commanders can be retired."""
        league = League()

        # Add some commanders with varying ratings and game counts
        for i in range(10):
            entry = LeagueEntry(
                entry_id=f"cmd_{i}",
                ckpt_path=f"/fake/path_{i}.pt",
                kind="commander",
                commander_name=f"Commander {i}",
                rating=Glicko2Rating(
                    rating=1500 - i * 50,  # Decreasing ratings
                    rd=50.0,
                    vol=0.06,
                ),
                games=100,
            )
            league.entries[entry.entry_id] = entry

        # Retire bottom 3 commanders
        retired = league.retire_commanders(keep_top=7)

        assert len(retired) == 3
        assert len([e for e in league.entries.values() if e.kind == "commander"]) == 7

        # Verify the lowest rated were retired
        retired_ids = {e.entry_id for e in retired}
        assert "cmd_7" in retired_ids
        assert "cmd_8" in retired_ids
        assert "cmd_9" in retired_ids

    def test_retire_commanders_min_games(self):
        """Don't retire commanders that haven't played enough games."""
        league = League()

        # Add commanders: some new (low games), some established
        for i in range(6):
            games = 5 if i < 3 else 100  # First 3 are new
            entry = LeagueEntry(
                entry_id=f"cmd_{i}",
                ckpt_path=f"/fake/path_{i}.pt",
                kind="commander",
                commander_name=f"Commander {i}",
                rating=Glicko2Rating(rating=1500 - i * 100, rd=50.0, vol=0.06),
                games=games,
            )
            league.entries[entry.entry_id] = entry

        # Try to retire to keep 4, but require 20 games minimum
        retired = league.retire_commanders(keep_top=4, min_games=20)

        # Only cmd_4 and cmd_5 (established + low rating) should be retired
        assert len(retired) == 2
        for e in retired:
            assert e.games >= 20

    def test_retire_commanders_empty_league(self):
        """retire_commanders handles empty league gracefully."""
        league = League()
        retired = league.retire_commanders(keep_top=5)
        assert retired == []

    def test_retire_commanders_fewer_than_keep_top(self):
        """retire_commanders does nothing when pool is smaller than keep_top."""
        league = League()

        # Add only 3 commanders
        for i in range(3):
            entry = LeagueEntry(
                entry_id=f"cmd_{i}",
                ckpt_path=f"/fake/path_{i}.pt",
                kind="commander",
                rating=Glicko2Rating(rating=1500, rd=50.0, vol=0.06),
                games=100,
            )
            league.entries[entry.entry_id] = entry

        # Try to keep top 5 (more than exist)
        retired = league.retire_commanders(keep_top=5)
        assert retired == []
        assert len([e for e in league.entries.values() if e.kind == "commander"]) == 3

    def test_retire_commanders_uses_conservative_rating(self):
        """retire_commanders ranks by conservative rating (rating - 2*RD)."""
        league = League()

        # Commander with high rating but high RD (uncertain)
        league.entries["uncertain"] = LeagueEntry(
            entry_id="uncertain",
            ckpt_path="/uncertain.pt",
            kind="commander",
            rating=Glicko2Rating(rating=1600, rd=150, vol=0.06),  # Conservative: 1300
            games=100,
        )

        # Commander with lower rating but low RD (confident)
        league.entries["confident"] = LeagueEntry(
            entry_id="confident",
            ckpt_path="/confident.pt",
            kind="commander",
            rating=Glicko2Rating(rating=1500, rd=30, vol=0.06),  # Conservative: 1440
            games=100,
        )

        # Keep only 1 - uncertain should be retired despite higher rating
        retired = league.retire_commanders(keep_top=1)

        assert len(retired) == 1
        assert retired[0].entry_id == "uncertain"

    def test_retire_commanders_marks_as_retired(self):
        """retire_commanders sets kind='retired' to preserve history."""
        league = League()

        for i in range(3):
            entry = LeagueEntry(
                entry_id=f"cmd_{i}",
                ckpt_path=f"/fake/path_{i}.pt",
                kind="commander",
                rating=Glicko2Rating(rating=1500 - i * 100, rd=50.0, vol=0.06),
                games=100,
            )
            league.entries[entry.entry_id] = entry

        retired = league.retire_commanders(keep_top=2)

        assert len(retired) == 1
        assert retired[0].entry_id == "cmd_2"
        # Entry should still exist but be marked as retired
        assert league.entries["cmd_2"].kind == "retired"

    def test_retire_commanders_excludes_candidates(self):
        """retire_commanders only considers commanders, not candidates."""
        league = League()

        # Add 2 commanders
        for i in range(2):
            league.entries[f"cmd_{i}"] = LeagueEntry(
                entry_id=f"cmd_{i}",
                ckpt_path=f"/cmd_{i}.pt",
                kind="commander",
                rating=Glicko2Rating(rating=1500, rd=50.0, vol=0.06),
                games=100,
            )

        # Add 3 candidates (should not be touched)
        for i in range(3):
            league.entries[f"cand_{i}"] = LeagueEntry(
                entry_id=f"cand_{i}",
                ckpt_path=f"/cand_{i}.pt",
                kind="candidate",
                rating=Glicko2Rating(rating=1200, rd=50.0, vol=0.06),  # Low rating
                games=100,
            )

        retired = league.retire_commanders(keep_top=1)

        assert len(retired) == 1
        assert retired[0].kind == "retired"
        # Candidates should be untouched
        for i in range(3):
            assert league.entries[f"cand_{i}"].kind == "candidate"


class TestHeuristicEntry:
    """Test heuristic baseline support."""

    def test_add_heuristic_creates_entry(self):
        """add_heuristic creates a heuristic entry with fixed ID."""
        league = League()
        entry = league.add_heuristic()

        assert entry.entry_id == "heuristic"
        assert entry.kind == "heuristic"
        assert entry.commander_name == "Lieutenant Heuristic"
        assert entry.ckpt_path == ""
        assert entry.rating.rating == league.cfg.rating0

    def test_add_heuristic_idempotent(self):
        """Calling add_heuristic twice returns same entry."""
        league = League()
        entry1 = league.add_heuristic()
        entry1.rating = Glicko2Rating(rating=1600.0, rd=100.0, vol=0.05)
        entry2 = league.add_heuristic()

        assert entry1 is entry2
        assert entry2.rating.rating == 1600.0  # Preserved

    def test_heuristic_included_in_top_commanders(self):
        """top_commanders includes heuristic entries."""
        league = League()
        league.add_heuristic()

        top = league.top_commanders(10)
        assert len(top) == 1
        assert top[0].kind == "heuristic"

    def test_heuristic_never_retired(self):
        """retire_commanders skips heuristic entries."""
        league = League()
        heur = league.add_heuristic()
        heur.games = 100
        heur.rating = Glicko2Rating(rating=800.0, rd=50.0, vol=0.05)

        for i in range(25):
            entry = LeagueEntry(
                entry_id=f"cmd_{i}",
                ckpt_path=f"/fake/path_{i}.pt",
                kind="commander",
                commander_name=f"Commander {i}",
                rating=Glicko2Rating(rating=1500.0 + i * 10, rd=50.0, vol=0.05),
                games=50,
            )
            league.entries[entry.entry_id] = entry

        retired = league.retire_commanders(keep_top=10, min_games=20)

        assert heur.kind == "heuristic"
        assert "heuristic" not in [e.entry_id for e in retired]

    def test_heuristic_serialization_roundtrip(self):
        """Heuristic entry survives save/load cycle."""
        import tempfile
        from pathlib import Path

        league = League()
        league.add_heuristic()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "league.json"
            league.save(path)
            loaded = League.load(path)

        assert "heuristic" in loaded.entries
        entry = loaded.entries["heuristic"]
        assert entry.kind == "heuristic"
        assert entry.commander_name == "Lieutenant Heuristic"
