"""Glicko-2 rating system tests.

Verify the rating system correctness:
- Expected score calculations
- Rating update mechanics
- RD (rating deviation) dynamics
- Numerical stability
"""

import math

from echelon.arena.glicko2 import (
    GameResult,
    Glicko2Rating,
    expected_score,
    rate,
)


class TestExpectedScore:
    """Verify expected score calculations."""

    def test_equal_ratings_give_half(self):
        """Equal ratings give 0.5 expected score."""
        r1 = Glicko2Rating(rating=1500, rd=50, vol=0.06)
        r2 = Glicko2Rating(rating=1500, rd=50, vol=0.06)

        e = expected_score(r1, r2)
        assert abs(e - 0.5) < 0.01, f"Equal ratings should give ~0.5, got {e}"

    def test_higher_rating_favored(self):
        """Higher rating gives > 0.5 expected score."""
        strong = Glicko2Rating(rating=1700, rd=50, vol=0.06)
        weak = Glicko2Rating(rating=1300, rd=50, vol=0.06)

        e = expected_score(strong, weak)
        assert e > 0.5, f"Higher rated player should have E > 0.5, got {e}"
        assert e < 1.0, f"Expected score should be < 1.0, got {e}"

    def test_expected_score_symmetry(self):
        """E(A,B) + E(B,A) = 1.0 (approximately for same RD)."""
        # Use same RD for both players to get true symmetry
        r1 = Glicko2Rating(rating=1600, rd=60, vol=0.06)
        r2 = Glicko2Rating(rating=1400, rd=60, vol=0.06)

        e1 = expected_score(r1, r2)
        e2 = expected_score(r2, r1)

        assert abs(e1 + e2 - 1.0) < 0.001, f"E(A,B) + E(B,A) should = 1.0, got {e1 + e2}"


class TestRatingUpdates:
    """Verify rating update mechanics."""

    def test_win_increases_rating(self):
        """Winning increases rating."""
        r = Glicko2Rating(rating=1500, rd=50, vol=0.06)
        opp = Glicko2Rating(rating=1500, rd=50, vol=0.06)

        results = [GameResult(opponent=opp, score=1.0)]  # Win
        new_r = rate(r, results)

        assert new_r.rating > r.rating, f"Win should increase rating: {r.rating} -> {new_r.rating}"

    def test_loss_decreases_rating(self):
        """Losing decreases rating."""
        r = Glicko2Rating(rating=1500, rd=50, vol=0.06)
        opp = Glicko2Rating(rating=1500, rd=50, vol=0.06)

        results = [GameResult(opponent=opp, score=0.0)]  # Loss
        new_r = rate(r, results)

        assert new_r.rating < r.rating, f"Loss should decrease rating: {r.rating} -> {new_r.rating}"

    def test_draw_moves_toward_opponent(self):
        """Draw moves rating toward opponent's."""
        strong = Glicko2Rating(rating=1700, rd=50, vol=0.06)
        weak = Glicko2Rating(rating=1300, rd=50, vol=0.06)

        # Strong player draws with weak player
        results = [GameResult(opponent=weak, score=0.5)]
        new_strong = rate(strong, results)

        # Strong player's rating should decrease (underperformed expectation)
        assert (
            new_strong.rating < strong.rating
        ), f"Strong player drawing weak should lose rating: {strong.rating} -> {new_strong.rating}"

    def test_upset_win_gives_larger_gain(self):
        """Beating higher-rated opponent gives larger rating gain."""
        underdog = Glicko2Rating(rating=1400, rd=50, vol=0.06)
        favorite = Glicko2Rating(rating=1600, rd=50, vol=0.06)
        equal = Glicko2Rating(rating=1400, rd=50, vol=0.06)

        # Beat favorite
        gain_upset = rate(underdog, [GameResult(opponent=favorite, score=1.0)]).rating - underdog.rating
        # Beat equal
        gain_expected = rate(underdog, [GameResult(opponent=equal, score=1.0)]).rating - underdog.rating

        assert (
            gain_upset > gain_expected
        ), f"Upset win should give larger gain: upset={gain_upset}, expected={gain_expected}"


class TestRDDynamics:
    """Verify RD (rating deviation) dynamics."""

    def test_rd_decreases_with_games(self):
        """RD decreases as games are played."""
        r = Glicko2Rating(rating=1500, rd=100, vol=0.06)
        opp = Glicko2Rating(rating=1500, rd=50, vol=0.06)

        results = [GameResult(opponent=opp, score=0.5)]
        new_r = rate(r, results)

        assert new_r.rd < r.rd, f"RD should decrease after game: {r.rd} -> {new_r.rd}"

    def test_rd_increases_without_games(self):
        """RD increases when no games played (rating period)."""
        r = Glicko2Rating(rating=1500, rd=50, vol=0.06)

        # No games
        new_r = rate(r, [])

        assert new_r.rd > r.rd, f"RD should increase without games: {r.rd} -> {new_r.rd}"


class TestNumericalStability:
    """Verify numerical stability of Glicko-2."""

    def test_extreme_rating_difference(self):
        """Handles extreme rating differences gracefully."""
        strong = Glicko2Rating(rating=3000, rd=50, vol=0.06)
        weak = Glicko2Rating(rating=500, rd=50, vol=0.06)

        e = expected_score(strong, weak)
        assert 0.99 < e < 1.0, f"Extreme difference should give E near 1: {e}"
        assert math.isfinite(e), "Expected score should be finite"

    def test_very_low_rd(self):
        """Handles very low RD gracefully."""
        r = Glicko2Rating(rating=1500, rd=10, vol=0.06)  # Very confident
        opp = Glicko2Rating(rating=1500, rd=50, vol=0.06)

        results = [GameResult(opponent=opp, score=1.0)]
        new_r = rate(r, results)

        assert math.isfinite(new_r.rating), "Rating should be finite"
        assert math.isfinite(new_r.rd), "RD should be finite"
        assert new_r.rd > 0, "RD should remain positive"

    def test_serialization_roundtrip(self):
        """Rating survives dict serialization."""
        original = Glicko2Rating(rating=1650.5, rd=75.25, vol=0.055)

        # Roundtrip through dict
        d = original.as_dict()
        restored = Glicko2Rating.from_dict(d)

        assert abs(original.rating - restored.rating) < 0.01
        assert abs(original.rd - restored.rd) < 0.01
        assert abs(original.vol - restored.vol) < 0.001
