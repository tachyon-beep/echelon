# Test Suite Implementation Plan

## Executive Summary

This plan operationalizes the test suite design into concrete implementation tasks. It maps the 10-layer architecture to specific files, tests, and fixtures, with clear priorities and dependencies.

**Current state**: 90+ tests across unit/integration/performance
**Target state**: ~250 tests with comprehensive invariant coverage and RL-specific verification

---

## Part I: Gap Analysis

### What We Have

| Category | Count | Coverage |
|----------|-------|----------|
| Mechanics/Combat | 12 | Missiles, splash, knockdown, paint |
| LOS/Raycast | 15 | Single ray, batch, materials |
| Navigation | 9 | Graph building, pathfinding |
| Determinism | 2 | Basic trajectory, debris |
| API Fuzzing | 2 | NaN/Inf, OOB actions |
| Generation | 5 | Layout, biomes, connectivity |
| Training/PPO | 10 | Losses, clipping, optimizer |
| Rollout/GAE | 15 | Buffer, advantage estimation |
| Normalization | 12 | Running stats, checkpoints |
| VecEnv | 7 | Protocol, context manager |
| Evaluation | 5 | Stats, replay, seeds |
| Performance | 2 | SPS, memory |

### Critical Gaps

| Gap | Risk | Priority |
|-----|------|----------|
| **Reward calculation tests** | Training failure | P0 |
| **Heat/stability invariants** | Silent physics bugs | P0 |
| **Glicko-2 rating tests** | Self-play corruption | P1 |
| **EWAR mechanics** | ECM/ECCM untested | P1 |
| **Multi-seed statistical tests** | Seed-dependent bugs | P1 |
| **Observation contract tests** | Policy can't learn | P1 |
| **Gradient flow tests** | Training instability | P2 |
| **Chaos/recovery tests** | Production failures | P2 |

---

## Part II: Implementation Phases

### Phase 1: Foundation (Week 1)

**Goal**: Catch show-stopping bugs before training

#### 1.1 Core Invariants (`tests/unit/test_invariants.py`)

```python
# New file: tests/unit/test_invariants.py

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

from echelon.sim.sim import Sim
from echelon.sim.world import VoxelWorld
from echelon.config import WorldConfig
from echelon.actions import ACTION_DIM


class TestHeatInvariants:
    """Heat system invariants that must never break."""

    @given(heat=st.floats(min_value=0, max_value=500))
    def test_heat_never_negative_after_dissipation(self, make_mech, heat):
        """Heat remains >= 0 after any dissipation."""
        mech = make_mech("m", "blue", [5, 5, 1], "heavy")
        mech.heat = heat
        # Simulate dissipation
        new_heat = max(0.0, heat - mech.spec.heat_dissipation * 0.05)
        assert new_heat >= 0.0

    def test_shutdown_threshold_enforced(self, make_mech):
        """Mech shuts down exactly when heat > heat_cap."""
        mech = make_mech("m", "blue", [5, 5, 1], "heavy")
        # Just under threshold
        mech.heat = mech.spec.heat_cap - 0.1
        assert not mech.shutdown
        # Just over threshold
        mech.heat = mech.spec.heat_cap + 0.1
        # (After sim step would set shutdown flag)


class TestStabilityInvariants:
    """Stability system invariants."""

    @given(stability=st.floats(min_value=-100, max_value=200))
    def test_stability_clamped_to_bounds(self, make_mech, stability):
        """Stability always in [0, max_stability]."""
        mech = make_mech("m", "blue", [5, 5, 1], "heavy")
        mech.stability = np.clip(stability, 0.0, mech.max_stability)
        assert 0.0 <= mech.stability <= mech.max_stability


class TestPositionInvariants:
    """Position and movement invariants."""

    def test_dead_mechs_immobile(self, make_mech):
        """Dead mechs have zero velocity."""
        mech = make_mech("m", "blue", [5, 5, 1], "heavy")
        mech.alive = False
        mech.vel[:] = 0.0
        assert np.allclose(mech.vel, 0.0)

    def test_positions_in_world_bounds(self):
        """All mech positions within world boundaries."""
        # Create env, run episode, verify all positions in bounds
        pass


class TestDamageConservation:
    """Damage dealt must equal damage taken."""

    def test_laser_damage_conservation(self, make_mech):
        """Laser damage dealt equals damage taken by target."""
        pass

    def test_splash_damage_sums_correctly(self, make_mech):
        """Total splash damage distributed equals explosion damage."""
        pass
```

**Tests to implement**: 12 tests
**Estimated time**: 4 hours

#### 1.2 Reward Correctness (`tests/unit/test_rewards.py`)

```python
# New file: tests/unit/test_rewards.py

import pytest
import numpy as np
from echelon import EchelonEnv, EnvConfig
from echelon.config import WorldConfig


class TestRewardPolarity:
    """Verify reward signs are correct."""

    def test_zone_control_positive_for_holder(self):
        """Team in zone gets positive per-tick reward."""
        cfg = EnvConfig(
            world=WorldConfig(size_x=20, size_y=20, size_z=10),
            num_packs=1,
            seed=0,
        )
        env = EchelonEnv(cfg)
        env.reset(seed=0)

        # Move blue into zone, red out
        # Step and verify blue reward > 0
        pass

    def test_kill_reward_goes_to_shooter(self):
        """Kill reward attributed to shooter, not victim."""
        pass

    def test_damage_reward_positive_for_dealer(self):
        """Dealing damage gives positive reward."""
        pass

    def test_death_penalty_to_victim(self):
        """Dying gives negative reward to victim."""
        pass


class TestRewardGradients:
    """Verify reward increases for desired behaviors."""

    def test_approach_reward_gradient(self):
        """Moving toward zone gives higher reward than away."""
        pass

    def test_damage_scales_with_amount(self):
        """More damage = proportionally more reward."""
        pass


class TestTerminalRewards:
    """Verify terminal reward distribution."""

    def test_winner_gets_positive_terminal(self):
        """Winning team gets positive terminal reward."""
        pass

    def test_loser_gets_negative_terminal(self):
        """Losing team gets negative terminal reward."""
        pass

    def test_draw_terminal_reward(self):
        """Draw gives zero or small terminal reward."""
        pass


class TestRewardBounds:
    """Verify rewards are bounded appropriately."""

    def test_per_step_reward_bounded(self):
        """Per-step reward magnitude is reasonable."""
        pass

    def test_episode_total_bounded(self):
        """Total episode reward is in expected range."""
        pass
```

**Tests to implement**: 11 tests
**Estimated time**: 6 hours

#### 1.3 Enhanced Determinism (`tests/unit/test_determinism.py`)

Add to existing file:

```python
class TestDeterminismExtended:
    """Extended determinism verification."""

    @pytest.mark.parametrize("seed", [0, 42, 12345, 999999])
    def test_determinism_multiple_seeds(self, seed):
        """Determinism holds for various seeds."""
        pass

    def test_reset_restores_initial_state(self):
        """reset(seed=X) after steps equals fresh env(seed=X)."""
        pass

    def test_recipe_hash_reproducibility(self):
        """Same recipe hash regenerates identical world."""
        pass

    def test_determinism_across_long_episode(self):
        """Determinism holds over 1000+ steps."""
        pass

    def test_combat_outcome_determinism(self):
        """Combat outcomes are deterministic given same inputs."""
        pass
```

**Tests to add**: 5 tests
**Estimated time**: 3 hours

#### 1.4 Observation Sanitization (`tests/unit/test_observations.py`)

```python
# New file: tests/unit/test_observations.py

import pytest
import numpy as np
from hypothesis import given, strategies as st

from echelon import EchelonEnv, EnvConfig
from echelon.config import WorldConfig


class TestObservationSanitization:
    """Verify observations are always valid."""

    def test_no_nan_in_observations(self):
        """Observations never contain NaN."""
        cfg = EnvConfig(
            world=WorldConfig(size_x=20, size_y=20, size_z=10),
            num_packs=1,
            seed=0,
        )
        env = EchelonEnv(cfg)
        obs, _ = env.reset(seed=0)

        for agent_id, agent_obs in obs.items():
            assert np.all(np.isfinite(agent_obs)), f"NaN/Inf in {agent_id}"

    def test_no_inf_in_observations(self):
        """Observations never contain Inf."""
        pass

    def test_observation_dimension_consistent(self):
        """Observation dimension matches declared space."""
        pass

    def test_observation_bounds_respected(self):
        """Observations within declared bounds."""
        pass


class TestContactSlots:
    """Verify contact slot structure."""

    def test_contact_slot_structure(self):
        """Contact slots have valid structure."""
        pass

    def test_visible_contacts_have_valid_data(self):
        """Visible contacts have non-zero features."""
        pass

    def test_invisible_contacts_zeroed(self):
        """Invisible contact slots are zeroed."""
        pass


class TestObservationInformation:
    """Verify observations contain necessary information."""

    def test_self_state_in_observation(self):
        """Agent's own HP, heat, stability in obs."""
        pass

    def test_position_info_available(self):
        """Position information (relative or absolute) available."""
        pass

    def test_observations_differentiate_states(self):
        """Different game states produce different observations."""
        pass
```

**Tests to implement**: 10 tests
**Estimated time**: 4 hours

---

### Phase 2: Combat & Physics (Week 2)

**Goal**: Verify simulation correctness

#### 2.1 Heat Mechanics (`tests/unit/test_heat.py`)

```python
# New file: tests/unit/test_heat.py

class TestHeatGeneration:
    """Verify weapon heat generation."""

    def test_laser_generates_heat(self, make_mech):
        """Firing laser increases heat by weapon.heat."""
        pass

    def test_missile_generates_heat(self, make_mech):
        """Firing missile increases heat by weapon.heat."""
        pass

    def test_kinetic_generates_heat(self, make_mech):
        """Firing kinetic weapon increases heat."""
        pass


class TestHeatDissipation:
    """Verify heat dissipation mechanics."""

    def test_passive_dissipation_rate(self, make_mech):
        """Heat decreases by spec.heat_dissipation * dt."""
        pass

    def test_vent_multiplier_applied(self, make_mech):
        """Venting multiplies dissipation rate."""
        pass

    def test_dissipation_doesnt_go_negative(self, make_mech):
        """Dissipation cannot reduce heat below 0."""
        pass


class TestShutdown:
    """Verify shutdown behavior."""

    def test_shutdown_at_capacity(self, make_mech):
        """Mech shuts down when heat > heat_cap."""
        pass

    def test_shutdown_blocks_weapons(self, make_mech):
        """Shutdown mechs cannot fire weapons."""
        pass

    def test_shutdown_allows_physics(self, make_mech):
        """Shutdown mechs still affected by gravity/momentum."""
        pass

    def test_shutdown_recovery(self, make_mech):
        """Mech recovers when heat drops below threshold."""
        pass
```

**Tests to implement**: 10 tests
**Estimated time**: 4 hours

#### 2.2 Stability & Knockdown (`tests/unit/test_stability.py`)

```python
# New file: tests/unit/test_stability.py

class TestStabilityDamage:
    """Verify stability reduction mechanics."""

    def test_missile_reduces_stability(self, make_mech):
        """Missile hits reduce stability."""
        pass

    def test_kinetic_reduces_stability(self, make_mech):
        """Kinetic hits reduce stability."""
        pass

    def test_leg_damage_affects_stability(self, make_mech):
        """Leg damage impacts stability recovery."""
        pass


class TestKnockdown:
    """Verify knockdown mechanics."""

    def test_knockdown_at_zero_stability(self, make_mech):
        """Mech falls when stability reaches 0."""
        pass

    def test_knockdown_duration(self, make_mech):
        """Mech stays down for knockdown_duration."""
        pass

    def test_knockdown_blocks_movement(self, make_mech):
        """Fallen mech cannot move laterally."""
        pass

    def test_knockdown_blocks_rotation(self, make_mech):
        """Fallen mech cannot rotate."""
        pass

    def test_recovery_restores_stability(self, make_mech):
        """Standing up restores partial stability."""
        pass
```

**Tests to implement**: 8 tests
**Estimated time**: 3 hours

#### 2.3 EWAR Mechanics (`tests/unit/test_ewar.py`)

```python
# New file: tests/unit/test_ewar.py

class TestECM:
    """Verify ECM mechanics."""

    def test_ecm_toggle_on(self, make_mech):
        """ECM can be activated."""
        pass

    def test_ecm_degrades_sensor_quality(self, make_mech):
        """ECM reduces enemy sensor quality."""
        pass

    def test_ecm_affects_missile_lock(self, make_mech):
        """ECM prevents/degrades missile locks."""
        pass


class TestECCM:
    """Verify ECCM mechanics."""

    def test_eccm_toggle_on(self, make_mech):
        """ECCM can be activated."""
        pass

    def test_eccm_counters_ecm(self, make_mech):
        """ECCM reduces ECM effectiveness."""
        pass


class TestSensorQuality:
    """Verify sensor quality calculations."""

    def test_base_sensor_quality(self, make_mech):
        """Base sensor quality by mech class."""
        pass

    def test_sensor_quality_with_ecm(self, make_mech):
        """Sensor quality reduced under ECM."""
        pass

    def test_sensor_quality_affects_contacts(self, make_mech):
        """Low sensor quality degrades contact info."""
        pass
```

**Tests to implement**: 8 tests
**Estimated time**: 4 hours

#### 2.4 Property-Based Physics (`tests/unit/test_physics_properties.py`)

```python
# New file: tests/unit/test_physics_properties.py

from hypothesis import given, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant


class SimulationStateMachine(RuleBasedStateMachine):
    """Stateful property testing for simulation."""

    def __init__(self):
        super().__init__()
        # Initialize env

    @invariant()
    def heat_non_negative(self):
        for m in self.sim.mechs.values():
            assert m.heat >= 0.0

    @invariant()
    def stability_bounded(self):
        for m in self.sim.mechs.values():
            assert 0.0 <= m.stability <= m.max_stability + 1e-6

    @invariant()
    def dead_mechs_stay_dead(self):
        for mid, alive_before in self._alive_snapshot.items():
            if not alive_before:
                assert not self.sim.mechs[mid].alive

    @invariant()
    def positions_in_bounds(self):
        for m in self.sim.mechs.values():
            if m.alive:
                assert 0 <= m.pos[0] <= self.world.size_x
                assert 0 <= m.pos[1] <= self.world.size_y
                assert 0 <= m.pos[2] <= self.world.size_z

    @rule()
    def step_with_random_actions(self):
        """Take a step with random actions."""
        pass

    @rule()
    def fire_all_weapons(self):
        """Attempt to fire all weapons."""
        pass


TestSimulationProperties = SimulationStateMachine.TestCase
```

**Tests to implement**: 1 stateful test (covers many invariants)
**Estimated time**: 4 hours

---

### Phase 3: Self-Play Infrastructure (Week 3)

**Goal**: Verify arena and rating systems

#### 3.1 Glicko-2 Tests (`tests/unit/test_glicko2.py`)

```python
# New file: tests/unit/test_glicko2.py

import pytest
import math
from echelon.arena.glicko2 import Glicko2Rating, update_rating, expected_score


class TestExpectedScore:
    """Verify expected score calculations."""

    def test_equal_ratings_give_half(self):
        """Equal ratings give 0.5 expected score."""
        r1 = Glicko2Rating(rating=1500, rd=50, vol=0.06)
        r2 = Glicko2Rating(rating=1500, rd=50, vol=0.06)
        e = expected_score(r1, r2)
        assert abs(e - 0.5) < 0.001

    def test_higher_rating_favored(self):
        """Higher rating gives > 0.5 expected score."""
        r1 = Glicko2Rating(rating=1700, rd=50, vol=0.06)
        r2 = Glicko2Rating(rating=1500, rd=50, vol=0.06)
        e = expected_score(r1, r2)
        assert e > 0.5

    def test_expected_score_symmetry(self):
        """E(A,B) + E(B,A) = 1.0"""
        r1 = Glicko2Rating(rating=1600, rd=60, vol=0.06)
        r2 = Glicko2Rating(rating=1400, rd=80, vol=0.06)
        e1 = expected_score(r1, r2)
        e2 = expected_score(r2, r1)
        assert abs(e1 + e2 - 1.0) < 0.001


class TestRatingUpdates:
    """Verify rating update mechanics."""

    def test_win_increases_rating(self):
        """Winning increases rating."""
        r = Glicko2Rating(rating=1500, rd=50, vol=0.06)
        opp = Glicko2Rating(rating=1500, rd=50, vol=0.06)
        new_r = update_rating(r, [(opp, 1.0)])  # Win
        assert new_r.rating > r.rating

    def test_loss_decreases_rating(self):
        """Losing decreases rating."""
        r = Glicko2Rating(rating=1500, rd=50, vol=0.06)
        opp = Glicko2Rating(rating=1500, rd=50, vol=0.06)
        new_r = update_rating(r, [(opp, 0.0)])  # Loss
        assert new_r.rating < r.rating

    def test_rd_decreases_with_games(self):
        """RD decreases as games are played."""
        r = Glicko2Rating(rating=1500, rd=100, vol=0.06)
        opp = Glicko2Rating(rating=1500, rd=50, vol=0.06)
        new_r = update_rating(r, [(opp, 0.5)])
        assert new_r.rd < r.rd

    def test_upset_win_gives_larger_gain(self):
        """Beating higher-rated opponent gives larger rating gain."""
        underdog = Glicko2Rating(rating=1400, rd=50, vol=0.06)
        favorite = Glicko2Rating(rating=1600, rd=50, vol=0.06)
        weak_opp = Glicko2Rating(rating=1400, rd=50, vol=0.06)

        gain_upset = update_rating(underdog, [(favorite, 1.0)]).rating - underdog.rating
        gain_expected = update_rating(underdog, [(weak_opp, 1.0)]).rating - underdog.rating

        assert gain_upset > gain_expected


class TestRDDecay:
    """Verify RD increases without games."""

    def test_rd_increases_over_time(self):
        """RD increases when player is inactive."""
        # Simulate time passing without games
        pass


class TestNumericalStability:
    """Verify numerical stability of Glicko-2."""

    def test_extreme_rating_difference(self):
        """Handles extreme rating differences gracefully."""
        r1 = Glicko2Rating(rating=3000, rd=50, vol=0.06)
        r2 = Glicko2Rating(rating=500, rd=50, vol=0.06)
        e = expected_score(r1, r2)
        assert 0.99 < e < 1.0
        assert math.isfinite(e)

    def test_very_low_rd(self):
        """Handles very low RD gracefully."""
        r = Glicko2Rating(rating=1500, rd=5, vol=0.06)
        opp = Glicko2Rating(rating=1500, rd=50, vol=0.06)
        new_r = update_rating(r, [(opp, 1.0)])
        assert math.isfinite(new_r.rating)
        assert math.isfinite(new_r.rd)
```

**Tests to implement**: 12 tests
**Estimated time**: 5 hours

#### 3.2 Model Cache Tests (`tests/unit/test_model_cache.py`)

```python
# New file: tests/unit/test_model_cache.py

class TestLRUModelCache:
    """Verify model cache behavior."""

    def test_cache_stores_models(self):
        """Cache stores and retrieves models."""
        pass

    def test_lru_eviction(self):
        """Least recently used models evicted."""
        pass

    def test_cache_size_limit(self):
        """Cache respects size limit."""
        pass

    def test_cache_hit_updates_recency(self):
        """Accessing model updates recency."""
        pass
```

**Tests to implement**: 4 tests
**Estimated time**: 2 hours

---

### Phase 4: Statistical & Chaos (Week 4)

**Goal**: Catch statistical and edge-case bugs

#### 4.1 Multi-Seed Tests (`tests/integration/test_multi_seed.py`)

```python
# New file: tests/integration/test_multi_seed.py

import pytest
import numpy as np
from echelon import EchelonEnv, EnvConfig
from echelon.config import WorldConfig


class TestMultiSeedRobustness:
    """Verify behavior across many seeds."""

    @pytest.mark.parametrize("seed", range(100))
    def test_episode_completes_all_seeds(self, seed):
        """Episode completes without crash for 100 seeds."""
        cfg = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=15),
            num_packs=1,
            seed=seed,
            max_episode_seconds=10.0,
        )
        env = EchelonEnv(cfg)
        obs, _ = env.reset(seed=seed)

        for _ in range(100):
            actions = {aid: np.zeros(env.ACTION_DIM) for aid in obs}
            obs, _, terms, truncs, _ = env.step(actions)
            if all(terms.values()) or all(truncs.values()):
                break

        # Just verify no crash

    def test_outcome_distribution(self):
        """Win rate over many games is roughly 50% for symmetric setup."""
        blue_wins = 0
        red_wins = 0
        draws = 0

        for seed in range(100):
            # Run episode with random actions
            # Count outcome
            pass

        # Allow variance but expect roughly equal
        total = blue_wins + red_wins + draws
        if total > 0:
            blue_rate = blue_wins / total
            assert 0.3 < blue_rate < 0.7, f"Blue win rate {blue_rate} seems biased"


class TestRewardStatistics:
    """Verify reward statistics across episodes."""

    def test_mean_reward_bounded(self):
        """Mean reward per episode in expected range."""
        pass

    def test_reward_variance_reasonable(self):
        """Reward variance indicates learning signal."""
        pass
```

**Tests to implement**: 4 tests
**Estimated time**: 3 hours

#### 4.2 Chaos Tests (`tests/chaos/test_recovery.py`)

```python
# New file: tests/chaos/test_recovery.py

import pytest
import numpy as np
from echelon import EchelonEnv, EnvConfig
from echelon.config import WorldConfig


class TestNaNRecovery:
    """Verify graceful handling of NaN inputs."""

    def test_nan_action_handled(self):
        """NaN actions don't corrupt state."""
        cfg = EnvConfig(
            world=WorldConfig(size_x=20, size_y=20, size_z=10),
            num_packs=1,
            seed=0,
        )
        env = EchelonEnv(cfg)
        obs, _ = env.reset(seed=0)

        # Send NaN action
        nan_action = np.full(env.ACTION_DIM, np.nan)
        actions = {"blue_0": nan_action}

        # Should not crash, should produce valid obs
        obs, _, _, _, _ = env.step(actions)
        assert np.all(np.isfinite(obs["blue_0"]))

    def test_inf_action_clipped(self):
        """Inf actions are clipped safely."""
        pass


class TestMalformedInputs:
    """Verify handling of malformed inputs."""

    def test_missing_agent_action(self):
        """Missing agent actions use zero action."""
        pass

    def test_extra_agent_action_ignored(self):
        """Actions for non-existent agents ignored."""
        pass

    def test_wrong_action_dimension(self):
        """Wrong action dimension handled gracefully."""
        pass


class TestRapidReset:
    """Verify rapid reset handling."""

    def test_multiple_rapid_resets(self):
        """Rapid reset() calls don't corrupt state."""
        cfg = EnvConfig(
            world=WorldConfig(size_x=20, size_y=20, size_z=10),
            num_packs=1,
            seed=0,
        )
        env = EchelonEnv(cfg)

        for seed in range(10):
            obs, _ = env.reset(seed=seed)
            # Immediately reset again
            obs, _ = env.reset(seed=seed + 100)

            # Verify state is valid
            assert obs is not None

    def test_reset_during_episode(self):
        """Mid-episode reset works correctly."""
        pass
```

**Tests to implement**: 8 tests
**Estimated time**: 3 hours

#### 4.3 Training Stability (`tests/integration/test_training_stability.py`)

```python
# New file: tests/integration/test_training_stability.py

import pytest
import torch
import numpy as np


class TestGradientHealth:
    """Verify gradient health during training."""

    def test_no_nan_in_gradients(self):
        """Gradients never contain NaN."""
        pass

    def test_no_exploding_gradients(self):
        """Gradient norms stay bounded."""
        pass

    def test_gradient_flow_to_all_layers(self):
        """Gradients flow to all trainable parameters."""
        pass


class TestValueFunction:
    """Verify value function behavior."""

    def test_value_predictions_bounded(self):
        """Value function outputs reasonable values."""
        pass

    def test_value_correlates_with_returns(self):
        """Value predictions correlate with actual returns."""
        pass


class TestPolicyHealth:
    """Verify policy health metrics."""

    def test_entropy_doesnt_collapse(self):
        """Policy entropy stays above minimum."""
        pass

    def test_action_probabilities_valid(self):
        """Action probabilities sum to 1, all positive."""
        pass
```

**Tests to implement**: 7 tests
**Estimated time**: 4 hours

---

## Part III: Infrastructure Updates

### 3.1 Enhanced Fixtures (`tests/conftest.py`)

Add to existing conftest:

```python
@pytest.fixture
def empty_world():
    """10x10x10 world with no obstacles."""
    cfg = WorldConfig(size_x=10, size_y=10, size_z=10, obstacle_fill=0.0)
    world = VoxelWorld.generate(cfg, np.random.default_rng(0))
    world.voxels.fill(VoxelWorld.AIR)
    return world


@pytest.fixture
def wall_world():
    """World with solid wall for LOS testing."""
    cfg = WorldConfig(size_x=20, size_y=20, size_z=10, obstacle_fill=0.0)
    world = VoxelWorld.generate(cfg, np.random.default_rng(0))
    world.voxels.fill(VoxelWorld.AIR)
    world.set_box_solid(10, 0, 0, 11, 20, 10, True)
    return world


@pytest.fixture
def small_env():
    """Minimal environment for fast tests."""
    cfg = EnvConfig(
        world=WorldConfig(size_x=20, size_y=20, size_z=10, obstacle_fill=0.0),
        num_packs=1,
        seed=0,
        max_episode_seconds=5.0,
    )
    return EchelonEnv(cfg)


@pytest.fixture
def training_env():
    """Environment configured for training tests."""
    cfg = EnvConfig(
        world=WorldConfig(size_x=40, size_y=40, size_z=20),
        num_packs=1,
        seed=0,
        max_episode_seconds=30.0,
    )
    return EchelonEnv(cfg)
```

### 3.2 Test Markers (`pyproject.toml` or `pytest.ini`)

```ini
[tool:pytest]
markers =
    slow: marks tests as slow (> 10 seconds)
    fast: marks tests as fast (< 1 second)
    unit: unit tests
    integration: integration tests
    property: property-based tests
    performance: performance benchmarks
    chaos: chaos/recovery tests
    determinism: determinism tests
    reward: reward calculation tests
    physics: physics simulation tests
    combat: combat mechanics tests
    nav: navigation tests
    training: training infrastructure tests
    critical: must pass before merge
    mutation: targets mutation testing
```

### 3.3 Test Organization

```
tests/
├── conftest.py                    # Shared fixtures
├── unit/
│   ├── test_invariants.py         # NEW: Core invariants
│   ├── test_rewards.py            # NEW: Reward verification
│   ├── test_observations.py       # NEW: Observation contract
│   ├── test_heat.py               # NEW: Heat mechanics
│   ├── test_stability.py          # NEW: Stability/knockdown
│   ├── test_ewar.py               # NEW: ECM/ECCM
│   ├── test_physics_properties.py # NEW: Property-based physics
│   ├── test_glicko2.py            # NEW: Rating system
│   ├── test_model_cache.py        # NEW: Arena model cache
│   ├── test_mechanics.py          # Existing
│   ├── test_determinism.py        # Existing (enhanced)
│   ├── test_los.py                # Existing
│   ├── test_nav*.py               # Existing
│   └── training/                  # Existing
├── integration/
│   ├── test_multi_seed.py         # NEW: Statistical tests
│   ├── test_training_stability.py # NEW: Gradient health
│   ├── test_convergence.py        # Existing
│   └── test_learning.py           # Existing
├── chaos/
│   └── test_recovery.py           # NEW: Chaos tests
└── performance/
    └── test_benchmark.py          # Existing
```

---

## Part IV: Implementation Schedule

| Week | Focus | New Tests | Total Tests |
|------|-------|-----------|-------------|
| 1 | Foundation | 38 | ~130 |
| 2 | Combat & Physics | 27 | ~160 |
| 3 | Self-Play | 16 | ~175 |
| 4 | Statistical & Chaos | 19 | ~195 |

**Post-Phase Work**:
- Mutation testing configuration
- CI integration
- Coverage reporting
- Documentation

---

## Part V: Success Criteria

### Phase 1 Complete When:
- [ ] All core invariants tested
- [ ] Reward polarity verified
- [ ] Determinism verified for 10+ seeds
- [ ] No NaN/Inf in observations

### Phase 2 Complete When:
- [ ] Heat mechanics 100% covered
- [ ] Stability/knockdown verified
- [ ] EWAR basics tested
- [ ] Property-based sim tests pass

### Phase 3 Complete When:
- [ ] Glicko-2 math verified
- [ ] Expected score symmetry proven
- [ ] Model cache LRU works

### Phase 4 Complete When:
- [ ] 100 seeds complete without crash
- [ ] NaN/Inf recovery verified
- [ ] Gradient health verified
- [ ] Chaos tests pass

---

## Part VI: Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Tests too slow | Mark with `@pytest.mark.slow`, run separately |
| Flaky tests | Root cause within 1 week, add `@pytest.mark.flaky` temporarily |
| Missing dependencies | Add hypothesis, pytest-timeout to dev deps |
| Coverage gaps missed | Run mutation testing after Phase 2 |

---

## Appendix: Quick Reference

### Running Specific Test Categories

```bash
# Phase 1 tests
PYTHONPATH=. uv run pytest tests/unit/test_invariants.py tests/unit/test_rewards.py tests/unit/test_observations.py -v

# Property-based tests
PYTHONPATH=. uv run pytest -m property --timeout=120

# Critical tests only (must pass before merge)
PYTHONPATH=. uv run pytest -m critical --timeout=60

# Chaos tests
PYTHONPATH=. uv run pytest tests/chaos -v

# Multi-seed statistical tests
PYTHONPATH=. uv run pytest tests/integration/test_multi_seed.py -v
```

### Adding New Invariant Tests

1. Identify the invariant (e.g., "heat >= 0")
2. Add to `test_invariants.py` with `@given` decorator if property-based
3. Mark with `@pytest.mark.critical`
4. Add to mutation testing targets
