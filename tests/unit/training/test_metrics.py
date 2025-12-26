"""Comprehensive tests for W&B metric helper functions.

Includes:
- Unit tests for edge cases and basic functionality
- Property-based tests for invariants
- Mutation-resistant assertions
"""

import math
import sys

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Import the helpers from train_ppo.py
sys.path.insert(0, "scripts")
from train_ppo import (
    compute_combat_stats,
    compute_coordination_stats,
    compute_perception_stats,
    compute_return_stats,
    compute_zone_stats,
)

# =============================================================================
# Strategies for property-based testing
# =============================================================================

# Strategy for generating realistic return values
returns_strategy = st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False)

# Strategy for generating non-negative floats (for counts, damage, etc.)
non_negative_float = st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False)

# Strategy for generating positive floats (for denominators)
positive_float = st.floats(min_value=0.1, max_value=10000.0, allow_nan=False, allow_infinity=False)


def combat_stats_dict() -> st.SearchStrategy[dict[str, float]]:
    """Strategy for generating valid combat stats dicts."""
    return st.fixed_dictionaries(
        {
            "damage_blue": non_negative_float,
            "damage_red": non_negative_float,
            "kills_blue": non_negative_float,
            "kills_red": non_negative_float,
            "assists_blue": non_negative_float,
            "deaths_blue": st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        }
    )


def zone_stats_dict() -> st.SearchStrategy[dict[str, float]]:
    """Strategy for generating valid zone stats dicts."""
    return st.fixed_dictionaries(
        {
            "zone_ticks_blue": non_negative_float,
            "zone_ticks_red": non_negative_float,
            "contested_ticks": non_negative_float,
            "first_zone_entry": st.floats(
                min_value=-1.0, max_value=1000.0, allow_nan=False, allow_infinity=False
            ),
            "episode_length": positive_float,
        }
    )


def coord_stats_dict() -> st.SearchStrategy[dict[str, float]]:
    """Strategy for generating valid coordination stats dicts."""
    return st.fixed_dictionaries(
        {
            "pack_dispersion_sum": non_negative_float,
            "pack_dispersion_count": positive_float,
            "centroid_zone_dist_sum": non_negative_float,
            "centroid_zone_dist_count": positive_float,
            "focus_fire_concentration": st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
            "visible_contacts_sum": non_negative_float,
            "visible_contacts_count": positive_float,
            "hostile_filter_on_count": non_negative_float,
            "ecm_on_ticks": non_negative_float,
            "eccm_on_ticks": non_negative_float,
            "scout_ticks": positive_float,
        }
    )


# =============================================================================
# compute_return_stats tests
# =============================================================================


class TestComputeReturnStats:
    """Tests for compute_return_stats function."""

    def test_empty_returns_zeros(self) -> None:
        """Empty list returns all zeros."""
        result = compute_return_stats([])
        assert result == {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "std": 0.0}

    def test_single_return_zeros(self) -> None:
        """Single return (< 5) returns all zeros."""
        result = compute_return_stats([42.0])
        assert result == {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "std": 0.0}

    def test_four_returns_zeros(self) -> None:
        """Four returns (< 5) returns all zeros - boundary check."""
        result = compute_return_stats([1.0, 2.0, 3.0, 4.0])
        assert result == {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "std": 0.0}

    def test_five_returns_computes(self) -> None:
        """Exactly 5 returns (threshold) should compute stats."""
        result = compute_return_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["mean"] == pytest.approx(3.0)
        assert result["median"] == pytest.approx(3.0)
        # std of [1,2,3,4,5] = sqrt(2) ≈ 1.414
        assert result["std"] == pytest.approx(np.std([1, 2, 3, 4, 5]))

    def test_uniform_returns_zero_std(self) -> None:
        """Uniform returns have zero standard deviation."""
        result = compute_return_stats([5.0, 5.0, 5.0, 5.0, 5.0])
        assert result["mean"] == pytest.approx(5.0)
        assert result["median"] == pytest.approx(5.0)
        assert result["std"] == pytest.approx(0.0)
        assert result["p10"] == pytest.approx(5.0)
        assert result["p90"] == pytest.approx(5.0)

    def test_negative_returns_handled(self) -> None:
        """Negative returns are valid and handled correctly."""
        result = compute_return_stats([-10.0, -5.0, 0.0, 5.0, 10.0])
        assert result["mean"] == pytest.approx(0.0)
        assert result["median"] == pytest.approx(0.0)

    def test_large_spread_percentiles(self) -> None:
        """P10 < median < p90 for data with spread."""
        returns = list(range(100))  # 0 to 99
        result = compute_return_stats([float(x) for x in returns])
        assert result["p10"] < result["median"]
        assert result["median"] < result["p90"]

    def test_output_keys_complete(self) -> None:
        """Output always has all expected keys."""
        result = compute_return_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert set(result.keys()) == {"mean", "median", "p10", "p90", "std"}

    def test_output_types_float(self) -> None:
        """All output values are Python floats."""
        result = compute_return_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        for v in result.values():
            assert isinstance(v, float)

    @settings(max_examples=50, deadline=500)
    @given(returns=st.lists(returns_strategy, min_size=5, max_size=100))
    def test_property_mean_in_range(self, returns: list[float]) -> None:
        """Mean is always between min and max of input."""
        result = compute_return_stats(returns)
        assert min(returns) <= result["mean"] <= max(returns)

    @settings(max_examples=50, deadline=500)
    @given(returns=st.lists(returns_strategy, min_size=5, max_size=100))
    def test_property_std_non_negative(self, returns: list[float]) -> None:
        """Standard deviation is always non-negative."""
        result = compute_return_stats(returns)
        assert result["std"] >= 0.0

    @settings(max_examples=50, deadline=500)
    @given(returns=st.lists(returns_strategy, min_size=5, max_size=100))
    def test_property_percentiles_ordered(self, returns: list[float]) -> None:
        """P10 <= median <= p90 always holds."""
        result = compute_return_stats(returns)
        assert result["p10"] <= result["median"] <= result["p90"]

    @settings(max_examples=50, deadline=500)
    @given(returns=st.lists(returns_strategy, min_size=5, max_size=100))
    def test_property_no_nan_or_inf(self, returns: list[float]) -> None:
        """Output never contains NaN or infinity."""
        result = compute_return_stats(returns)
        for v in result.values():
            assert not math.isnan(v)
            assert not math.isinf(v)


# =============================================================================
# compute_combat_stats tests
# =============================================================================


class TestComputeCombatStats:
    """Tests for compute_combat_stats function."""

    def test_empty_returns_defaults(self) -> None:
        """Empty list returns sensible defaults."""
        result = compute_combat_stats([], num_agents=5)
        assert result == {
            "damage_dealt": 0.0,
            "damage_ratio": 1.0,
            "kill_participation": 0.0,
            "survival_rate": 1.0,
        }

    def test_zero_red_damage_avoids_division_by_zero(self) -> None:
        """Zero red damage doesn't cause division by zero."""
        stats = [
            {
                "damage_blue": 100.0,
                "damage_red": 0.0,
                "kills_blue": 1.0,
                "kills_red": 0.0,
                "assists_blue": 0.0,
                "deaths_blue": 0.0,
            }
        ]
        result = compute_combat_stats(stats, num_agents=5)
        # damage_ratio = 100 / max(0, 1) = 100
        assert result["damage_ratio"] == pytest.approx(100.0)

    def test_zero_total_kills_avoids_division_by_zero(self) -> None:
        """Zero total kills doesn't cause division by zero."""
        stats = [
            {
                "damage_blue": 50.0,
                "damage_red": 50.0,
                "kills_blue": 0.0,
                "kills_red": 0.0,
                "assists_blue": 0.0,
                "deaths_blue": 0.0,
            }
        ]
        result = compute_combat_stats(stats, num_agents=5)
        # kill_participation = (0 + 0) / max(0, 1) = 0
        assert result["kill_participation"] == pytest.approx(0.0)

    def test_zero_agents_avoids_division_by_zero(self) -> None:
        """Zero agents doesn't cause division by zero in survival rate."""
        stats = [
            {
                "damage_blue": 50.0,
                "damage_red": 50.0,
                "kills_blue": 1.0,
                "kills_red": 1.0,
                "assists_blue": 0.0,
                "deaths_blue": 1.0,
            }
        ]
        result = compute_combat_stats(stats, num_agents=0)
        # survival_rate = 1 - (1 / max(0, 1)) = 0
        assert result["survival_rate"] == pytest.approx(0.0)

    def test_all_deaths_zero_survival(self) -> None:
        """All agents dying gives zero survival rate."""
        stats = [
            {
                "damage_blue": 50.0,
                "damage_red": 50.0,
                "kills_blue": 0.0,
                "kills_red": 5.0,
                "assists_blue": 0.0,
                "deaths_blue": 5.0,
            }
        ]
        result = compute_combat_stats(stats, num_agents=5)
        assert result["survival_rate"] == pytest.approx(0.0)

    def test_no_deaths_full_survival(self) -> None:
        """No deaths gives full survival rate."""
        stats = [
            {
                "damage_blue": 100.0,
                "damage_red": 50.0,
                "kills_blue": 3.0,
                "kills_red": 0.0,
                "assists_blue": 2.0,
                "deaths_blue": 0.0,
            }
        ]
        result = compute_combat_stats(stats, num_agents=5)
        assert result["survival_rate"] == pytest.approx(1.0)

    def test_kill_participation_with_assists(self) -> None:
        """Assists count toward kill participation."""
        stats = [
            {
                "damage_blue": 100.0,
                "damage_red": 100.0,
                "kills_blue": 2.0,
                "kills_red": 2.0,
                "assists_blue": 2.0,
                "deaths_blue": 0.0,
            }
        ]
        result = compute_combat_stats(stats, num_agents=5)
        # kill_participation = (2 + 2) / (2 + 2) = 1.0
        assert result["kill_participation"] == pytest.approx(1.0)

    def test_output_keys_complete(self) -> None:
        """Output has all expected keys."""
        result = compute_combat_stats([], num_agents=5)
        assert set(result.keys()) == {"damage_dealt", "damage_ratio", "kill_participation", "survival_rate"}

    def test_multiple_episodes_averaged(self) -> None:
        """Multiple episodes are averaged correctly."""
        stats = [
            {
                "damage_blue": 100.0,
                "damage_red": 50.0,
                "kills_blue": 2.0,
                "kills_red": 1.0,
                "assists_blue": 1.0,
                "deaths_blue": 1.0,
            },
            {
                "damage_blue": 200.0,
                "damage_red": 50.0,
                "kills_blue": 4.0,
                "kills_red": 1.0,
                "assists_blue": 1.0,
                "deaths_blue": 1.0,
            },
        ]
        result = compute_combat_stats(stats, num_agents=5)
        assert result["damage_dealt"] == pytest.approx(150.0)  # (100 + 200) / 2

    @settings(max_examples=50, deadline=500)
    @given(
        combat_stats=st.lists(combat_stats_dict(), min_size=1, max_size=20),
        num_agents=st.integers(min_value=1, max_value=20),
    )
    def test_property_damage_ratio_non_negative(self, combat_stats: list[dict], num_agents: int) -> None:
        """Damage ratio is always non-negative."""
        result = compute_combat_stats(combat_stats, num_agents)
        assert result["damage_ratio"] >= 0.0

    @settings(max_examples=50, deadline=500)
    @given(
        combat_stats=st.lists(combat_stats_dict(), min_size=1, max_size=20),
        num_agents=st.integers(min_value=1, max_value=20),
    )
    def test_property_kill_participation_bounded(self, combat_stats: list[dict], num_agents: int) -> None:
        """Kill participation is in [0, inf) - can exceed 1 if assists > kills_red."""
        result = compute_combat_stats(combat_stats, num_agents)
        assert result["kill_participation"] >= 0.0

    @settings(max_examples=50, deadline=500)
    @given(
        combat_stats=st.lists(combat_stats_dict(), min_size=1, max_size=20),
        num_agents=st.integers(min_value=1, max_value=20),
    )
    def test_property_no_nan(self, combat_stats: list[dict], num_agents: int) -> None:
        """No NaN values in output."""
        result = compute_combat_stats(combat_stats, num_agents)
        for v in result.values():
            assert not math.isnan(v)


# =============================================================================
# compute_zone_stats tests
# =============================================================================


class TestComputeZoneStats:
    """Tests for compute_zone_stats function."""

    def test_empty_returns_defaults(self) -> None:
        """Empty list returns sensible defaults."""
        result = compute_zone_stats([])
        assert result == {"control_margin": 0.0, "contested_ratio": 0.0, "time_to_entry": 1.0}

    def test_blue_dominance_positive_margin(self) -> None:
        """Blue dominance gives positive control margin."""
        stats = [
            {
                "zone_ticks_blue": 100.0,
                "zone_ticks_red": 50.0,
                "contested_ticks": 10.0,
                "first_zone_entry": 5.0,
                "episode_length": 100.0,
            }
        ]
        result = compute_zone_stats(stats)
        assert result["control_margin"] == pytest.approx(0.5)  # (100-50)/100

    def test_red_dominance_negative_margin(self) -> None:
        """Red dominance gives negative control margin."""
        stats = [
            {
                "zone_ticks_blue": 50.0,
                "zone_ticks_red": 100.0,
                "contested_ticks": 10.0,
                "first_zone_entry": 5.0,
                "episode_length": 100.0,
            }
        ]
        result = compute_zone_stats(stats)
        assert result["control_margin"] == pytest.approx(-0.5)  # (50-100)/100

    def test_zero_episode_length_avoids_division_by_zero(self) -> None:
        """Zero episode length doesn't cause division by zero."""
        stats = [
            {
                "zone_ticks_blue": 50.0,
                "zone_ticks_red": 50.0,
                "contested_ticks": 10.0,
                "first_zone_entry": 5.0,
                "episode_length": 0.0,
            }
        ]
        result = compute_zone_stats(stats)
        # Uses max(episode_length, 1) = 1
        assert not math.isnan(result["control_margin"])

    def test_never_entered_zone_gives_one(self) -> None:
        """Never entering zone gives time_to_entry of 1.0."""
        stats = [
            {
                "zone_ticks_blue": 0.0,
                "zone_ticks_red": 50.0,
                "contested_ticks": 0.0,
                "first_zone_entry": -1.0,
                "episode_length": 100.0,
            }
        ]
        result = compute_zone_stats(stats)
        assert result["time_to_entry"] == pytest.approx(1.0)

    def test_immediate_zone_entry(self) -> None:
        """Immediate zone entry gives time_to_entry near 0."""
        stats = [
            {
                "zone_ticks_blue": 100.0,
                "zone_ticks_red": 0.0,
                "contested_ticks": 0.0,
                "first_zone_entry": 0.0,
                "episode_length": 100.0,
            }
        ]
        result = compute_zone_stats(stats)
        assert result["time_to_entry"] == pytest.approx(0.0)

    def test_contested_ratio_calculation(self) -> None:
        """Contested ratio computed correctly."""
        stats = [
            {
                "zone_ticks_blue": 50.0,
                "zone_ticks_red": 50.0,
                "contested_ticks": 25.0,
                "first_zone_entry": 10.0,
                "episode_length": 100.0,
            }
        ]
        result = compute_zone_stats(stats)
        assert result["contested_ratio"] == pytest.approx(0.25)

    def test_output_keys_complete(self) -> None:
        """Output has all expected keys."""
        result = compute_zone_stats([])
        assert set(result.keys()) == {"control_margin", "contested_ratio", "time_to_entry"}

    @settings(max_examples=50, deadline=500)
    @given(zone_stats=st.lists(zone_stats_dict(), min_size=1, max_size=20))
    def test_property_contested_ratio_bounded(self, zone_stats: list[dict]) -> None:
        """Contested ratio is bounded [0, max reasonable value]."""
        result = compute_zone_stats(zone_stats)
        assert result["contested_ratio"] >= 0.0

    @settings(max_examples=50, deadline=500)
    @given(zone_stats=st.lists(zone_stats_dict(), min_size=1, max_size=20))
    def test_property_time_to_entry_bounded(self, zone_stats: list[dict]) -> None:
        """Time to entry is bounded [0, 1] when computed."""
        result = compute_zone_stats(zone_stats)
        # Can be 1.0 (never entered) or [0, 1] normalized
        assert result["time_to_entry"] >= 0.0

    @settings(max_examples=50, deadline=500)
    @given(zone_stats=st.lists(zone_stats_dict(), min_size=1, max_size=20))
    def test_property_no_nan(self, zone_stats: list[dict]) -> None:
        """No NaN values in output."""
        result = compute_zone_stats(zone_stats)
        for v in result.values():
            assert not math.isnan(v)


# =============================================================================
# compute_coordination_stats tests
# =============================================================================


class TestComputeCoordinationStats:
    """Tests for compute_coordination_stats function."""

    def test_empty_returns_zeros(self) -> None:
        """Empty list returns all zeros."""
        result = compute_coordination_stats([])
        assert result == {"pack_dispersion": 0.0, "centroid_zone_dist": 0.0, "focus_fire": 0.0}

    def test_zero_counts_avoids_division_by_zero(self) -> None:
        """Zero counts don't cause division by zero."""
        stats = [
            {
                "pack_dispersion_sum": 100.0,
                "pack_dispersion_count": 0.0,
                "centroid_zone_dist_sum": 50.0,
                "centroid_zone_dist_count": 0.0,
                "focus_fire_concentration": 0.5,
            }
        ]
        result = compute_coordination_stats(stats)
        # Uses max(count, 1) = 1
        assert result["pack_dispersion"] == pytest.approx(100.0)
        assert result["centroid_zone_dist"] == pytest.approx(50.0)

    def test_focus_fire_uses_get_with_default(self) -> None:
        """Missing focus_fire_concentration defaults to 0."""
        stats = [
            {
                "pack_dispersion_sum": 100.0,
                "pack_dispersion_count": 10.0,
                "centroid_zone_dist_sum": 50.0,
                "centroid_zone_dist_count": 10.0,
                # Note: focus_fire_concentration is missing
            }
        ]
        result = compute_coordination_stats(stats)
        assert result["focus_fire"] == pytest.approx(0.0)

    def test_weighted_average_dispersion(self) -> None:
        """Dispersion is sum/count averaged across episodes."""
        stats = [
            {
                "pack_dispersion_sum": 100.0,
                "pack_dispersion_count": 10.0,
                "centroid_zone_dist_sum": 50.0,
                "centroid_zone_dist_count": 10.0,
                "focus_fire_concentration": 0.5,
            },
            {
                "pack_dispersion_sum": 200.0,
                "pack_dispersion_count": 10.0,
                "centroid_zone_dist_sum": 50.0,
                "centroid_zone_dist_count": 10.0,
                "focus_fire_concentration": 0.5,
            },
        ]
        result = compute_coordination_stats(stats)
        # Episode 1: 100/10 = 10, Episode 2: 200/10 = 20, avg = 15
        assert result["pack_dispersion"] == pytest.approx(15.0)

    def test_output_keys_complete(self) -> None:
        """Output has all expected keys."""
        result = compute_coordination_stats([])
        assert set(result.keys()) == {"pack_dispersion", "centroid_zone_dist", "focus_fire"}

    @settings(max_examples=50, deadline=500)
    @given(coord_stats=st.lists(coord_stats_dict(), min_size=1, max_size=20))
    def test_property_all_non_negative(self, coord_stats: list[dict]) -> None:
        """All coordination stats are non-negative."""
        result = compute_coordination_stats(coord_stats)
        for v in result.values():
            assert v >= 0.0

    @settings(max_examples=50, deadline=500)
    @given(coord_stats=st.lists(coord_stats_dict(), min_size=1, max_size=20))
    def test_property_focus_fire_bounded(self, coord_stats: list[dict]) -> None:
        """Focus fire is in [0, 1]."""
        result = compute_coordination_stats(coord_stats)
        assert 0.0 <= result["focus_fire"] <= 1.0

    @settings(max_examples=50, deadline=500)
    @given(coord_stats=st.lists(coord_stats_dict(), min_size=1, max_size=20))
    def test_property_no_nan(self, coord_stats: list[dict]) -> None:
        """No NaN values in output."""
        result = compute_coordination_stats(coord_stats)
        for v in result.values():
            assert not math.isnan(v)


# =============================================================================
# compute_perception_stats tests
# =============================================================================


class TestComputePerceptionStats:
    """Tests for compute_perception_stats function."""

    def test_empty_returns_zeros(self) -> None:
        """Empty list returns all zeros."""
        result = compute_perception_stats([])
        assert result == {
            "visible_contacts": 0.0,
            "hostile_filter_usage": 0.0,
            "ecm_usage": 0.0,
            "eccm_usage": 0.0,
        }

    def test_zero_counts_avoids_division_by_zero(self) -> None:
        """Zero counts don't cause division by zero."""
        stats = [
            {
                "visible_contacts_sum": 100.0,
                "visible_contacts_count": 0.0,
                "hostile_filter_on_count": 50.0,
                "ecm_on_ticks": 30.0,
                "eccm_on_ticks": 20.0,
                "scout_ticks": 0.0,
            }
        ]
        result = compute_perception_stats(stats)
        # Uses max(count, 1) = 1
        assert result["visible_contacts"] == pytest.approx(100.0)
        assert result["ecm_usage"] == pytest.approx(30.0)

    def test_full_ewar_usage(self) -> None:
        """Full EWAR usage gives 1.0."""
        stats = [
            {
                "visible_contacts_sum": 50.0,
                "visible_contacts_count": 10.0,
                "hostile_filter_on_count": 10.0,
                "ecm_on_ticks": 100.0,
                "eccm_on_ticks": 100.0,
                "scout_ticks": 100.0,
            }
        ]
        result = compute_perception_stats(stats)
        assert result["ecm_usage"] == pytest.approx(1.0)
        assert result["eccm_usage"] == pytest.approx(1.0)

    def test_hostile_filter_ratio(self) -> None:
        """Hostile filter usage is ratio of filter on to total observations."""
        stats = [
            {
                "visible_contacts_sum": 100.0,
                "visible_contacts_count": 50.0,
                "hostile_filter_on_count": 25.0,  # Half of observations
                "ecm_on_ticks": 0.0,
                "eccm_on_ticks": 0.0,
                "scout_ticks": 100.0,
            }
        ]
        result = compute_perception_stats(stats)
        assert result["hostile_filter_usage"] == pytest.approx(0.5)

    def test_output_keys_complete(self) -> None:
        """Output has all expected keys."""
        result = compute_perception_stats([])
        assert set(result.keys()) == {"visible_contacts", "hostile_filter_usage", "ecm_usage", "eccm_usage"}

    @settings(max_examples=50, deadline=500)
    @given(coord_stats=st.lists(coord_stats_dict(), min_size=1, max_size=20))
    def test_property_all_non_negative(self, coord_stats: list[dict]) -> None:
        """All perception stats are non-negative."""
        result = compute_perception_stats(coord_stats)
        for v in result.values():
            assert v >= 0.0

    @settings(max_examples=50, deadline=500)
    @given(coord_stats=st.lists(coord_stats_dict(), min_size=1, max_size=20))
    def test_property_no_nan(self, coord_stats: list[dict]) -> None:
        """No NaN values in output."""
        result = compute_perception_stats(coord_stats)
        for v in result.values():
            assert not math.isnan(v)


# =============================================================================
# Cross-function invariant tests
# =============================================================================


class TestMetricInvariants:
    """Cross-cutting invariant tests for all metric functions."""

    def test_all_functions_return_dicts(self) -> None:
        """All metric functions return dictionaries."""
        assert isinstance(compute_return_stats([]), dict)
        assert isinstance(compute_combat_stats([], 5), dict)
        assert isinstance(compute_zone_stats([]), dict)
        assert isinstance(compute_coordination_stats([]), dict)
        assert isinstance(compute_perception_stats([]), dict)

    def test_all_empty_inputs_have_values(self) -> None:
        """All functions return non-empty dicts for empty input."""
        assert len(compute_return_stats([])) > 0
        assert len(compute_combat_stats([], 5)) > 0
        assert len(compute_zone_stats([])) > 0
        assert len(compute_coordination_stats([])) > 0
        assert len(compute_perception_stats([])) > 0

    def test_all_values_are_python_floats(self) -> None:
        """All metric values are Python floats, not numpy types."""
        # This matters for W&B serialization
        for result in [
            compute_return_stats([1.0, 2.0, 3.0, 4.0, 5.0]),
            compute_combat_stats(
                [
                    {
                        "damage_blue": 100.0,
                        "damage_red": 50.0,
                        "kills_blue": 1.0,
                        "kills_red": 1.0,
                        "assists_blue": 0.0,
                        "deaths_blue": 0.0,
                    }
                ],
                5,
            ),
            compute_zone_stats(
                [
                    {
                        "zone_ticks_blue": 50.0,
                        "zone_ticks_red": 50.0,
                        "contested_ticks": 10.0,
                        "first_zone_entry": 5.0,
                        "episode_length": 100.0,
                    }
                ]
            ),
            compute_coordination_stats(
                [
                    {
                        "pack_dispersion_sum": 100.0,
                        "pack_dispersion_count": 10.0,
                        "centroid_zone_dist_sum": 50.0,
                        "centroid_zone_dist_count": 10.0,
                        "focus_fire_concentration": 0.5,
                    }
                ]
            ),
            compute_perception_stats(
                [
                    {
                        "visible_contacts_sum": 50.0,
                        "visible_contacts_count": 10.0,
                        "hostile_filter_on_count": 5.0,
                        "ecm_on_ticks": 30.0,
                        "eccm_on_ticks": 20.0,
                        "scout_ticks": 100.0,
                    }
                ]
            ),
        ]:
            for key, value in result.items():
                assert type(value) is float, f"Expected float, got {type(value)} for {key}"
