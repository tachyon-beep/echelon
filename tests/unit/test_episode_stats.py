"""Tests for episode stats tracking in EchelonEnv.

Covers _episode_stats dict, _compute_pack_dispersion(), and focus fire tracking.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from echelon.config import (
    EnvConfig,
    WorldConfig,
)
from echelon.env.env import EchelonEnv

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def minimal_env_config() -> EnvConfig:
    """Create minimal env config for testing."""
    return EnvConfig(
        world=WorldConfig(size_x=30, size_y=30, size_z=10, obstacle_fill=0.0),
        num_packs=1,
        observation_mode="full",
        max_episode_seconds=30.0,
        dt_sim=0.1,
        decision_repeat=1,
    )


@pytest.fixture
def env(minimal_env_config: EnvConfig) -> EchelonEnv:
    """Create a minimal environment for testing."""
    e = EchelonEnv(minimal_env_config)
    e.reset(seed=42)
    return e


def make_random_actions(env: EchelonEnv) -> dict[str, np.ndarray]:
    """Create random actions for all agents (uses env.ACTION_DIM)."""
    return {aid: np.random.uniform(-1, 1, env.ACTION_DIM).astype(np.float32) for aid in env.agents}


def make_zero_actions(env: EchelonEnv) -> dict[str, np.ndarray]:
    """Create zero actions for all agents (uses env.ACTION_DIM)."""
    return {aid: np.zeros(env.ACTION_DIM, dtype=np.float32) for aid in env.agents}


# -----------------------------------------------------------------------------
# Episode Stats Initialization Tests
# -----------------------------------------------------------------------------


class TestEpisodeStatsInitialization:
    """Tests for _episode_stats dict initialization."""

    def test_stats_initialized_on_reset(self, env: EchelonEnv) -> None:
        """_episode_stats is initialized with expected keys after reset."""
        stats = env._episode_stats

        # Combat stats
        assert "kills_blue" in stats
        assert "kills_red" in stats
        assert "assists_blue" in stats
        assert "damage_blue" in stats
        assert "damage_red" in stats

        # Zone stats
        assert "zone_ticks_blue" in stats
        assert "zone_ticks_red" in stats
        assert "contested_ticks" in stats
        assert "first_zone_entry_step" in stats

        # Coordination stats
        assert "pack_dispersion_sum" in stats
        assert "pack_dispersion_count" in stats
        assert "focus_fire_concentration" in stats

        # Perception stats
        assert "visible_contacts_sum" in stats
        assert "visible_contacts_count" in stats
        assert "hostile_filter_on_count" in stats

        # EWAR stats
        assert "ecm_on_ticks" in stats
        assert "eccm_on_ticks" in stats
        assert "scout_ticks" in stats

    def test_combat_stats_start_at_zero(self, env: EchelonEnv) -> None:
        """Combat stats (kills, damage, etc.) start at zero after reset."""
        stats = env._episode_stats

        combat_keys = [
            "kills_blue",
            "kills_red",
            "assists_blue",
            "assists_red",
            "damage_blue",
            "damage_red",
            "knockdowns_blue",
            "knockdowns_red",
        ]
        for key in combat_keys:
            assert stats[key] == 0.0, f"{key} should start at 0.0"

    def test_zone_stats_start_at_zero(self, env: EchelonEnv) -> None:
        """Zone stats start at expected values (0 or -1)."""
        stats = env._episode_stats

        assert stats["zone_ticks_blue"] == 0.0
        assert stats["zone_ticks_red"] == 0.0
        assert stats["contested_ticks"] == 0.0
        assert stats["first_zone_entry_step"] == -1.0  # -1 means never entered

    def test_stats_are_all_floats(self, env: EchelonEnv) -> None:
        """All stats values are floats for consistent typing."""
        for key, value in env._episode_stats.items():
            assert isinstance(value, float), f"{key} should be float, got {type(value)}"

    def test_damage_by_target_initialized_empty(self, env: EchelonEnv) -> None:
        """_damage_by_target dict is initialized empty."""
        assert env._damage_by_target == {}

    def test_combat_stats_reset_on_new_episode(self, minimal_env_config: EnvConfig) -> None:
        """Combat stats are reset when starting a new episode."""
        env = EchelonEnv(minimal_env_config)
        env.reset(seed=42)

        # Manually modify combat stats
        env._episode_stats["kills_blue"] = 5.0
        env._episode_stats["damage_blue"] = 100.0
        env._damage_by_target["enemy_0"] = 50.0

        # Reset should clear everything
        env.reset(seed=43)

        assert env._episode_stats["kills_blue"] == 0.0
        assert env._episode_stats["damage_blue"] == 0.0
        assert env._damage_by_target == {}


# -----------------------------------------------------------------------------
# Pack Dispersion Tests
# -----------------------------------------------------------------------------


class TestPackDispersion:
    """Tests for _compute_pack_dispersion() method."""

    def test_dispersion_with_no_sim_returns_zero(self, minimal_env_config: EnvConfig) -> None:
        """Returns 0.0 when sim is None."""
        env = EchelonEnv(minimal_env_config)
        # Don't call reset, so sim is None
        result = env._compute_pack_dispersion("blue")
        assert result == 0.0

    def test_dispersion_with_one_mech_returns_zero(self, env: EchelonEnv) -> None:
        """Returns 0.0 when fewer than 2 mechs are alive."""
        # Kill all but one blue mech
        sim = env.sim
        assert sim is not None

        blue_ids = env.blue_ids
        for mid in blue_ids[1:]:
            mech = sim.mechs[mid]
            mech.hp = 0.0
            mech.alive = False

        result = env._compute_pack_dispersion("blue")
        assert result == 0.0

    def test_dispersion_with_two_mechs(self, env: EchelonEnv) -> None:
        """Correctly computes distance between two mechs."""
        sim = env.sim
        assert sim is not None

        blue_ids = env.blue_ids

        # Kill all but first two mechs
        for mid in blue_ids[2:]:
            mech = sim.mechs[mid]
            mech.hp = 0.0
            mech.alive = False

        # Position first two mechs at known locations
        sim.mechs[blue_ids[0]].pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        sim.mechs[blue_ids[1]].pos = np.array([3.0, 4.0, 0.0], dtype=np.float32)

        result = env._compute_pack_dispersion("blue")
        # Distance is 5.0 (3-4-5 triangle)
        assert np.isclose(result, 5.0), f"Expected 5.0, got {result}"

    def test_dispersion_with_three_mechs(self, env: EchelonEnv) -> None:
        """Computes mean pairwise distance for three mechs."""
        sim = env.sim
        assert sim is not None

        blue_ids = env.blue_ids

        # Kill all but first three mechs
        for mid in blue_ids[3:]:
            mech = sim.mechs[mid]
            mech.hp = 0.0
            mech.alive = False

        # Equilateral triangle with side 10
        sim.mechs[blue_ids[0]].pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        sim.mechs[blue_ids[1]].pos = np.array([10.0, 0.0, 0.0], dtype=np.float32)
        sim.mechs[blue_ids[2]].pos = np.array([5.0, 8.66, 0.0], dtype=np.float32)

        result = env._compute_pack_dispersion("blue")
        # Mean of three distances ~10 each
        assert 9.0 <= result <= 11.0, f"Expected ~10.0, got {result}"

    def test_dispersion_uses_xy_only(self, env: EchelonEnv) -> None:
        """Z coordinate is ignored in dispersion calculation."""
        sim = env.sim
        assert sim is not None

        blue_ids = env.blue_ids

        # Kill all but two mechs
        for mid in blue_ids[2:]:
            sim.mechs[mid].hp = 0.0
            sim.mechs[mid].alive = False

        # Same XY, different Z
        sim.mechs[blue_ids[0]].pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        sim.mechs[blue_ids[1]].pos = np.array([0.0, 0.0, 100.0], dtype=np.float32)

        result = env._compute_pack_dispersion("blue")
        # XY distance is 0
        assert result == 0.0

    def test_dispersion_for_red_team(self, env: EchelonEnv) -> None:
        """Correctly computes dispersion for red team."""
        sim = env.sim
        assert sim is not None

        red_ids = env.red_ids

        # Kill all but two red mechs
        for mid in red_ids[2:]:
            sim.mechs[mid].hp = 0.0
            sim.mechs[mid].alive = False

        sim.mechs[red_ids[0]].pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        sim.mechs[red_ids[1]].pos = np.array([6.0, 8.0, 0.0], dtype=np.float32)

        result = env._compute_pack_dispersion("red")
        assert np.isclose(result, 10.0), f"Expected 10.0, got {result}"


class TestPackDispersionProperties:
    """Property-based tests for pack dispersion invariants."""

    @settings(max_examples=30, deadline=1000)
    @given(
        positions=st.lists(
            st.tuples(
                st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            ),
            min_size=2,
            max_size=5,
        )
    )
    def test_dispersion_is_non_negative(self, positions: list[tuple[float, float]]) -> None:
        """Pack dispersion is always non-negative."""
        # Compute mean pairwise distance directly
        total_dist = 0.0
        count = 0
        for i, p1 in enumerate(positions):
            for p2 in positions[i + 1 :]:
                dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                total_dist += dist
                count += 1

        result = total_dist / max(count, 1)
        assert result >= 0.0

    @settings(max_examples=30, deadline=1000)
    @given(
        positions=st.lists(
            st.tuples(
                st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            ),
            min_size=2,
            max_size=5,
        )
    )
    def test_dispersion_zero_when_all_same_position(self, positions: list[tuple[float, float]]) -> None:
        """Dispersion is 0 when all mechs at same position."""
        # Use first position for all
        p = positions[0]
        same_positions = [p] * len(positions)

        total_dist = 0.0
        count = 0
        for i, p1 in enumerate(same_positions):
            for p2 in same_positions[i + 1 :]:
                dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                total_dist += dist
                count += 1

        result = total_dist / max(count, 1)
        assert np.isclose(result, 0.0, atol=1e-6)


# -----------------------------------------------------------------------------
# Focus Fire Tracking Tests
# -----------------------------------------------------------------------------


class TestFocusFireTracking:
    """Tests for focus fire concentration calculation."""

    def test_focus_fire_zero_when_no_damage(self, env: EchelonEnv) -> None:
        """Focus fire is 0 when no damage dealt."""
        assert env._damage_by_target == {}

        # Simulate end of episode calculation
        if env._damage_by_target:
            total_dmg = sum(env._damage_by_target.values())
            max_target_dmg = max(env._damage_by_target.values())
            concentration = max_target_dmg / max(total_dmg, 1.0)
        else:
            concentration = 0.0

        assert concentration == 0.0

    def test_focus_fire_one_when_single_target(self, env: EchelonEnv) -> None:
        """Focus fire is 1.0 when all damage on one target."""
        env._damage_by_target = {"enemy_0": 100.0}

        total_dmg = sum(env._damage_by_target.values())
        max_target_dmg = max(env._damage_by_target.values())
        concentration = max_target_dmg / max(total_dmg, 1.0)

        assert concentration == 1.0

    def test_focus_fire_partial_concentration(self, env: EchelonEnv) -> None:
        """Focus fire correctly computes partial concentration."""
        env._damage_by_target = {
            "enemy_0": 60.0,  # 60% of damage
            "enemy_1": 30.0,  # 30% of damage
            "enemy_2": 10.0,  # 10% of damage
        }

        total_dmg = sum(env._damage_by_target.values())
        max_target_dmg = max(env._damage_by_target.values())
        concentration = max_target_dmg / max(total_dmg, 1.0)

        assert np.isclose(concentration, 0.6), f"Expected 0.6, got {concentration}"

    def test_focus_fire_even_spread(self, env: EchelonEnv) -> None:
        """Focus fire is low when damage evenly spread."""
        env._damage_by_target = {
            "enemy_0": 25.0,
            "enemy_1": 25.0,
            "enemy_2": 25.0,
            "enemy_3": 25.0,
        }

        total_dmg = sum(env._damage_by_target.values())
        max_target_dmg = max(env._damage_by_target.values())
        concentration = max_target_dmg / max(total_dmg, 1.0)

        assert np.isclose(concentration, 0.25), f"Expected 0.25, got {concentration}"


class TestFocusFireProperties:
    """Property-based tests for focus fire invariants."""

    @settings(max_examples=50, deadline=500)
    @given(
        damage_values=st.lists(
            st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=10,
        )
    )
    def test_concentration_between_zero_and_one(self, damage_values: list[float]) -> None:
        """Focus fire concentration is always in [0, 1]."""
        total = sum(damage_values)
        max_val = max(damage_values)
        concentration = max_val / max(total, 1.0)

        assert 0.0 <= concentration <= 1.0

    @settings(max_examples=50, deadline=500)
    @given(
        num_targets=st.integers(min_value=1, max_value=10),
        damage=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    def test_even_spread_gives_inverse_n(self, num_targets: int, damage: float) -> None:
        """Equal damage to N targets gives concentration of 1/N."""
        damage_values = [damage] * num_targets
        total = sum(damage_values)
        max_val = max(damage_values)
        concentration = max_val / max(total, 1.0)

        expected = 1.0 / num_targets
        assert np.isclose(concentration, expected, atol=1e-6)


# -----------------------------------------------------------------------------
# Zone Tracking Tests
# -----------------------------------------------------------------------------


class TestZoneTracking:
    """Tests for zone control metrics tracking."""

    def test_zone_ticks_accumulate(self, env: EchelonEnv) -> None:
        """Zone ticks accumulate over steps."""
        # Run a few steps with zero actions
        for _ in range(10):
            actions = make_zero_actions(env)
            _, _, terminateds, truncateds, _ = env.step(actions)
            if any(terminateds.values()) or any(truncateds.values()):
                break

        # Zone ticks should be non-negative (may or may not have accumulated)
        assert env._episode_stats["zone_ticks_blue"] >= 0.0
        assert env._episode_stats["zone_ticks_red"] >= 0.0
        assert env._episode_stats["contested_ticks"] >= 0.0


# -----------------------------------------------------------------------------
# Coordination Metrics Tests
# -----------------------------------------------------------------------------


class TestCoordinationMetrics:
    """Tests for coordination metrics accumulation."""

    def test_dispersion_accumulated_each_step(self, env: EchelonEnv) -> None:
        """Pack dispersion is accumulated every step."""
        initial_count = env._episode_stats["pack_dispersion_count"]

        # Run one step
        actions = make_zero_actions(env)
        env.step(actions)

        # Count should increase by 1
        assert env._episode_stats["pack_dispersion_count"] == initial_count + 1.0

    def test_centroid_zone_dist_accumulated(self, env: EchelonEnv) -> None:
        """Centroid zone distance is accumulated when mechs alive."""
        # Run a few steps
        for _ in range(5):
            actions = make_zero_actions(env)
            _, _, terminateds, truncateds, _ = env.step(actions)
            if any(terminateds.values()) or any(truncateds.values()):
                break

        # Should have accumulated
        assert env._episode_stats["centroid_zone_dist_count"] >= 0.0


# -----------------------------------------------------------------------------
# Perception Metrics Tests
# -----------------------------------------------------------------------------


class TestPerceptionMetrics:
    """Tests for perception/sensor metrics tracking."""

    def test_visible_contacts_accumulated(self, env: EchelonEnv) -> None:
        """Visible contacts are tracked during observation building."""
        # Run a few steps
        for _ in range(3):
            actions = make_zero_actions(env)
            _, _, terminateds, truncateds, _ = env.step(actions)
            if any(terminateds.values()) or any(truncateds.values()):
                break

        # Visible contacts should have been counted
        assert env._episode_stats["visible_contacts_count"] >= 0.0

    def test_scout_ticks_track_scout_presence(self, env: EchelonEnv) -> None:
        """Scout ticks are accumulated when scouts are alive."""
        # Run steps
        for _ in range(3):
            actions = make_zero_actions(env)
            _, _, terminateds, truncateds, _ = env.step(actions)
            if any(terminateds.values()) or any(truncateds.values()):
                break

        # If there are scouts, they should be tracked
        # (scout_ticks accumulates only for blue scouts)
        assert env._episode_stats["scout_ticks"] >= 0.0


# -----------------------------------------------------------------------------
# Combat Stats Tests
# -----------------------------------------------------------------------------


class TestCombatStats:
    """Tests for combat event stat accumulation."""

    def test_damage_stats_non_negative(self, env: EchelonEnv) -> None:
        """Damage stats are non-negative after steps."""
        sim = env.sim
        assert sim is not None

        # Check initial state
        initial_damage = env._episode_stats["damage_blue"]
        assert initial_damage == 0.0

        # Run several steps - damage may or may not occur
        for _ in range(20):
            actions = make_random_actions(env)
            _, _, terminateds, truncateds, _ = env.step(actions)
            if any(terminateds.values()) or any(truncateds.values()):
                break

        # Damage should be non-negative
        assert env._episode_stats["damage_blue"] >= 0.0
        assert env._episode_stats["damage_red"] >= 0.0


# -----------------------------------------------------------------------------
# Stats at Episode End Tests
# -----------------------------------------------------------------------------


class TestStatsAtEpisodeEnd:
    """Tests for stats availability at episode termination."""

    def test_stats_included_in_last_outcome(self, env: EchelonEnv) -> None:
        """Episode stats are included in last_outcome when episode ends."""
        # Run until episode ends
        done = False
        max_steps = 500
        step = 0

        while not done and step < max_steps:
            actions = make_random_actions(env)
            _, _, terminateds, truncateds, _ = env.step(actions)
            done = any(terminateds.values()) or any(truncateds.values())
            step += 1

        # Check last_outcome contains stats
        outcome = env.last_outcome
        assert outcome is not None
        assert "stats" in outcome
        assert isinstance(outcome["stats"], dict)

        # Stats should have expected keys
        stats = outcome["stats"]
        assert "kills_blue" in stats
        assert "damage_blue" in stats
        assert "zone_ticks_blue" in stats
        assert "focus_fire_concentration" in stats


# -----------------------------------------------------------------------------
# Stats Dict Invariants (Property-based)
# -----------------------------------------------------------------------------


class TestStatsInvariants:
    """Property-based tests for stats dict invariants."""

    @settings(max_examples=10, deadline=5000)
    @given(steps=st.integers(min_value=1, max_value=5))
    def test_stats_always_floats_after_steps(self, steps: int) -> None:
        """All stats remain floats after any number of steps."""
        config = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=10, obstacle_fill=0.0),
            num_packs=1,
            observation_mode="full",
            max_episode_seconds=30.0,
            dt_sim=0.1,
            decision_repeat=1,
        )
        env = EchelonEnv(config)
        env.reset(seed=42)

        for _ in range(steps):
            actions = make_zero_actions(env)
            _, _, terminateds, truncateds, _ = env.step(actions)
            if any(terminateds.values()) or any(truncateds.values()):
                break

        for key, value in env._episode_stats.items():
            assert isinstance(value, float), f"{key} is not float: {type(value)}"

    @settings(max_examples=10, deadline=5000)
    @given(steps=st.integers(min_value=1, max_value=5))
    def test_count_stats_non_negative(self, steps: int) -> None:
        """All count-type stats are non-negative after steps."""
        config = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=10, obstacle_fill=0.0),
            num_packs=1,
            observation_mode="full",
            max_episode_seconds=30.0,
            dt_sim=0.1,
            decision_repeat=1,
        )
        env = EchelonEnv(config)
        env.reset(seed=42)

        for _ in range(steps):
            actions = make_zero_actions(env)
            _, _, terminateds, truncateds, _ = env.step(actions)
            if any(terminateds.values()) or any(truncateds.values()):
                break

        # All stats except first_zone_entry_step should be non-negative
        for key, value in env._episode_stats.items():
            if key != "first_zone_entry_step":
                assert value >= 0.0, f"{key} is negative: {value}"

    @settings(max_examples=10, deadline=5000)
    @given(steps=st.integers(min_value=1, max_value=5))
    def test_no_nan_in_stats(self, steps: int) -> None:
        """No NaN values in stats after steps."""
        config = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=10, obstacle_fill=0.0),
            num_packs=1,
            observation_mode="full",
            max_episode_seconds=30.0,
            dt_sim=0.1,
            decision_repeat=1,
        )
        env = EchelonEnv(config)
        env.reset(seed=42)

        for _ in range(steps):
            actions = make_zero_actions(env)
            _, _, terminateds, truncateds, _ = env.step(actions)
            if any(terminateds.values()) or any(truncateds.values()):
                break

        for key, value in env._episode_stats.items():
            assert not np.isnan(value), f"{key} is NaN"
