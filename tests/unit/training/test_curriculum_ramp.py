"""Unit tests for curriculum ramp calculations."""

import pytest


def compute_curriculum_progress(
    episodes: int,
    update: int,
    opfor_ramp_games: int,
    opfor_ramp_updates: int,
) -> float:
    """Compute curriculum progress (extracted logic for testing).

    This mirrors the logic in train_ppo.py for curriculum ramping.
    """
    if opfor_ramp_games > 0:
        return min(1.0, episodes / opfor_ramp_games)
    elif opfor_ramp_updates > 0:
        return min(1.0, (update - 1) / opfor_ramp_updates)
    else:
        return 1.0  # No ramp = instant full difficulty


def compute_weapon_prob(
    progress: float,
    weapon_start: float,
    weapon_end: float,
) -> float:
    """Compute weapon probability from progress."""
    return weapon_start + progress * (weapon_end - weapon_start)


class TestGameBasedRamp:
    """Tests for game-based curriculum ramp."""

    def test_game_ramp_at_zero_games(self):
        """At 0 games, progress should be 0."""
        progress = compute_curriculum_progress(
            episodes=0, update=1, opfor_ramp_games=1000, opfor_ramp_updates=0
        )
        assert progress == 0.0

    def test_game_ramp_at_half(self):
        """At 500/1000 games, progress should be 0.5."""
        progress = compute_curriculum_progress(
            episodes=500, update=100, opfor_ramp_games=1000, opfor_ramp_updates=0
        )
        assert progress == 0.5

    def test_game_ramp_at_completion(self):
        """At 1000/1000 games, progress should be 1.0."""
        progress = compute_curriculum_progress(
            episodes=1000, update=500, opfor_ramp_games=1000, opfor_ramp_updates=0
        )
        assert progress == 1.0

    def test_game_ramp_caps_at_one(self):
        """Progress should cap at 1.0 even if games exceed target."""
        progress = compute_curriculum_progress(
            episodes=2000, update=1000, opfor_ramp_games=1000, opfor_ramp_updates=0
        )
        assert progress == 1.0

    def test_game_ramp_overrides_update_ramp(self):
        """Game ramp takes priority over update ramp when both set."""
        # With game ramp set, update value should be ignored
        progress = compute_curriculum_progress(
            episodes=500, update=1, opfor_ramp_games=1000, opfor_ramp_updates=100
        )
        # If update-based were used, progress would be 0.0 (update=1)
        # But game-based should give 0.5
        assert progress == 0.5


class TestUpdateBasedRamp:
    """Tests for update-based curriculum ramp (legacy)."""

    def test_update_ramp_at_start(self):
        """At update 1, progress should be 0."""
        progress = compute_curriculum_progress(
            episodes=0, update=1, opfor_ramp_games=0, opfor_ramp_updates=100
        )
        assert progress == 0.0

    def test_update_ramp_at_half(self):
        """At update 51/100, progress should be 0.5."""
        progress = compute_curriculum_progress(
            episodes=1000, update=51, opfor_ramp_games=0, opfor_ramp_updates=100
        )
        assert progress == 0.5

    def test_update_ramp_caps_at_one(self):
        """Progress should cap at 1.0."""
        progress = compute_curriculum_progress(
            episodes=5000, update=200, opfor_ramp_games=0, opfor_ramp_updates=100
        )
        assert progress == 1.0


class TestNoRamp:
    """Tests for no curriculum ramp (instant full difficulty)."""

    def test_no_ramp_returns_one(self):
        """With no ramp configured, progress should be 1.0."""
        progress = compute_curriculum_progress(episodes=0, update=1, opfor_ramp_games=0, opfor_ramp_updates=0)
        assert progress == 1.0


class TestWeaponProbCalculation:
    """Tests for weapon probability calculation from progress."""

    def test_weapon_prob_at_start(self):
        """At progress 0, should return start value."""
        prob = compute_weapon_prob(0.0, weapon_start=0.15, weapon_end=1.0)
        assert prob == 0.15

    def test_weapon_prob_at_end(self):
        """At progress 1, should return end value."""
        prob = compute_weapon_prob(1.0, weapon_start=0.15, weapon_end=1.0)
        assert prob == 1.0

    def test_weapon_prob_at_half(self):
        """At progress 0.5, should interpolate linearly."""
        prob = compute_weapon_prob(0.5, weapon_start=0.2, weapon_end=0.8)
        assert prob == pytest.approx(0.5)

    def test_weapon_prob_custom_range(self):
        """Test with custom start/end values."""
        prob = compute_weapon_prob(0.25, weapon_start=0.1, weapon_end=0.5)
        # 0.1 + 0.25 * (0.5 - 0.1) = 0.1 + 0.25 * 0.4 = 0.1 + 0.1 = 0.2
        assert prob == pytest.approx(0.2)
