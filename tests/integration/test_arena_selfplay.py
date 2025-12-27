"""Integration test for arena self-play with Lieutenant Heuristic."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from echelon.arena.league import League


class TestArenaSelfPlay:
    """Test arena self-play integration."""

    @pytest.fixture
    def league_with_heuristic(self) -> tuple[Path, League]:
        """Create a temporary league with Lieutenant Heuristic."""
        tmpdir = tempfile.mkdtemp()
        league_path = Path(tmpdir) / "league.json"

        league = League()
        league.add_heuristic()
        league.save(league_path)

        return league_path, league

    def test_heuristic_in_arena_pool(self, league_with_heuristic):
        """Lieutenant Heuristic appears in arena pool."""
        league_path, _league = league_with_heuristic

        # Reload to verify persistence
        loaded = League.load(league_path)

        pool = loaded.top_commanders(10) + loaded.recent_candidates(5)
        assert len(pool) == 1
        assert pool[0].entry_id == "heuristic"
        assert pool[0].kind == "heuristic"
        assert pool[0].commander_name == "Lieutenant Heuristic"

    def test_heuristic_rating_updates(self, league_with_heuristic):
        """Heuristic rating can be updated from match results."""
        from echelon.arena.glicko2 import GameResult, Glicko2Rating

        _league_path, league = league_with_heuristic
        heuristic = league.entries["heuristic"]
        initial_rating = heuristic.rating.rating

        # Simulate heuristic losing a match
        fake_opponent_rating = Glicko2Rating(rating=1500.0, rd=100.0, vol=0.06)
        results = {
            "heuristic": [GameResult(opponent=fake_opponent_rating, score=0.0)]  # Loss
        }
        league.apply_rating_period(results)

        # Rating should decrease after a loss
        assert heuristic.rating.rating < initial_rating
        assert heuristic.games == 1

    def test_heuristic_excluded_from_eval_pool(self, league_with_heuristic):
        """Heuristic is excluded from evaluation pool (can't load checkpoint)."""
        _league_path, league = league_with_heuristic

        # Simulate what eval-candidate does: get pool and filter out heuristic
        pool = league.top_commanders(10) + league.recent_candidates(5)
        # Filter as arena.py does
        eval_pool = [e for e in pool if e.kind != "heuristic"]

        # Heuristic should be excluded since it has no checkpoint
        assert len(eval_pool) == 0
        assert all(e.kind != "heuristic" for e in eval_pool)
