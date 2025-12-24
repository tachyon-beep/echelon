# Echelon Test Suite Design: A World-Class Testing Framework for DRL Environments

## Executive Summary

This document presents a comprehensive test suite design for Echelon, a Deep Reinforcement Learning environment for mech tactics. The design addresses the unique challenges of testing RL environments: silent failures, non-determinism, emergent behaviors, and the coupling between simulation correctness, policy learning, and training infrastructure.

The guiding principle: **Tests must catch bugs that slip through casual observation**. In RL, a subtle reward miscalculation or physics glitch can cause training to fail after 10 million steps without any obvious error. Our tests prevent this.

---

## Part I: Testing Philosophy

### 1.1 What Makes RL Environment Testing Unique

Unlike traditional software where correctness is binary (works/doesn't work), RL environments have failure modes that are:

1. **Silent**: A reward off by 0.01 can derail training without raising exceptions
2. **Statistical**: Bugs may only manifest in 1% of episodes or after 1000 steps
3. **Emergent**: Correct components can combine into incorrect behavior
4. **Seed-Dependent**: Bugs appear only for specific random seeds
5. **Gradient-Coupled**: Environment bugs can appear as "training instability"

### 1.2 The Testing Pyramid for RL

```
                    /\
                   /  \  Statistical Tests
                  /    \   (Multi-seed validation, convergence bounds)
                 /------\
                /        \  Integration Tests
               /          \   (Agent-Env interface, training loops)
              /------------\
             /              \  Property-Based Tests
            /                \   (Invariants, metamorphic relations)
           /------------------\
          /                    \  Unit Tests
         /                      \   (Physics, rewards, observations)
        --------------------------
```

### 1.3 Core Testing Principles

1. **Determinism is Sacred**: Same seed must produce identical trajectories. Test this relentlessly.

2. **Conservation Laws are Invariants**: Energy, mass, damage totals - these must balance.

3. **Test the Gradient Path**: If observations flow to actions via a network, verify the chain.

4. **Mutation Testing Reveals Coverage Gaps**: If you can break the sim without failing a test, add one.

5. **Property-Based > Example-Based**: "For all valid actions, stability >= 0" beats "stability = 50 after this action".

6. **Statistical Rigor**: Report confidence intervals, not point estimates.

---

## Part II: Testing Layers Architecture

### Layer 1: Unit Tests (`tests/unit/`)

Fast, isolated tests for individual components. Target: <5 seconds total.

#### 1.1 Simulation Physics (`test_physics.py`)

**What to test:**
- Movement mechanics (velocity, acceleration, gravity)
- Collision detection and resolution
- Heat generation and dissipation
- Stability mechanics and knockdown
- Damage calculations (base, rear armor, leg hits)

**Key invariants:**
```python
# Heat is always non-negative
assert mech.heat >= 0.0

# Stability is bounded
assert 0.0 <= mech.stability <= mech.max_stability

# HP cannot exceed spec max
assert mech.hp <= mech.spec.hp

# Dead mechs don't move
if not mech.alive:
    assert np.allclose(mech.vel, 0.0)

# Shutdown mechs don't rotate
if mech.shutdown:
    assert mech.yaw == yaw_before_step
```

**Example test structure:**
```python
class TestHeatMechanics:
    """Verify heat generation, dissipation, and shutdown behavior."""

    def test_laser_generates_heat(self, make_mech):
        """Firing laser increases heat by weapon.heat."""

    def test_heat_dissipation_rate(self, make_mech):
        """Heat decreases by spec.heat_dissipation * dt per step."""

    def test_vent_multiplier(self, make_mech):
        """Venting increases dissipation by VENT_HEAT_MULT."""

    def test_shutdown_at_capacity(self, make_mech):
        """Mech shuts down when heat > heat_cap."""

    def test_shutdown_blocks_weapons(self, make_mech):
        """Shutdown mechs cannot fire weapons."""

    @given(heat=st.floats(min_value=0, max_value=200))
    def test_heat_never_negative(self, make_mech, heat):
        """Property: heat is never negative after any operation."""
```

#### 1.2 Combat Systems (`test_combat.py`)

**What to test:**
- Weapon firing conditions (cooldowns, arcs, range)
- Damage multipliers (rear armor, paint bonus)
- Splash damage and occlusion
- Projectile physics (ballistic drop, guidance)
- Point defense (AMS interception)

**Critical invariants:**
```python
# Damage dealt equals damage taken (conservation)
total_dealt = sum(m.dealt_damage for m in mechs.values())
total_taken = sum(m.took_damage for m in mechs.values())
assert abs(total_dealt - total_taken) < 1e-6

# Rear armor multiplier is applied correctly
if is_rear_hit:
    assert damage == base_damage * REAR_ARMOR_DAMAGE_MULT

# Splash damage is occluded by terrain
if raycast_blocked(explosion_pos, target_pos):
    assert target.took_damage == 0.0
```

#### 1.3 Observation System (`test_observations.py`)

**What to test:**
- Observation dimension consistency
- NaN/Inf sanitization
- Contact slot filling and prioritization
- Local map generation
- Sensor quality and EWAR effects

**Key invariants:**
```python
# Observation dimension matches spec
assert obs.shape == (env._obs_dim(),)

# No NaN/Inf in observations
assert np.all(np.isfinite(obs))

# Contact slots have valid structure
contacts = obs[:CONTACT_SLOTS * CONTACT_DIM].reshape(CONTACT_SLOTS, CONTACT_DIM)
for slot in contacts:
    if slot[21] > 0:  # visible flag
        assert slot[8] >= 0.0  # hp_norm
        assert 0.0 <= slot[10] <= 1.0  # stab_norm
```

#### 1.4 Action System (`test_actions.py`)

**What to test:**
- Action clipping to [-1, 1]
- NaN/Inf action handling
- Action-to-mech-behavior mapping
- Target selection mechanics
- EWAR toggle behavior

#### 1.5 Reward System (`test_rewards.py`)

**What to test:**
- Reward component calculations
- Terminal reward distribution
- Zone control scoring
- Damage/kill attribution

**Critical: Reward Mutation Tests**

Reward bugs are catastrophic for training. Use mutation testing specifically on reward code:

```python
@pytest.mark.mutation
class TestRewardMutations:
    """Tests that detect reward calculation mutations."""

    def test_zone_tick_reward_sign(self):
        """Zone control gives POSITIVE reward to controlling team."""

    def test_approach_reward_gradient(self):
        """Moving toward zone gives higher reward than moving away."""

    def test_kill_reward_attribution(self):
        """Kill reward goes to shooter, not victim."""

    def test_terminal_win_reward_polarity(self):
        """Winners get positive, losers get negative terminal reward."""
```

### Layer 2: Property-Based Tests (`tests/unit/test_properties.py`)

Use Hypothesis for generative testing of invariants.

#### 2.1 Simulation Invariants

```python
class SimStateMachine(RuleBasedStateMachine):
    """Stateful property testing for simulation."""

    @invariant()
    def heat_non_negative(self):
        for m in self.sim.mechs.values():
            assert m.heat >= 0.0, f"Mech {m.mech_id} has negative heat"

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
            assert 0 <= m.pos[0] <= self.world.size_x
            assert 0 <= m.pos[1] <= self.world.size_y
            assert 0 <= m.pos[2] <= self.world.size_z
```

#### 2.2 Metamorphic Relations

```python
class TestMetamorphicRelations:
    """Tests based on mathematical properties of the simulation."""

    def test_action_scaling_symmetry(self):
        """Doubling forward throttle doubles (approx) displacement."""

    def test_damage_commutativity(self):
        """Order of attacks doesn't affect total damage dealt."""

    def test_yaw_wraparound(self):
        """Yaw +2pi and yaw +0 produce identical behavior."""

    def test_team_symmetry(self):
        """Swapping team labels with mirrored positions gives mirrored outcomes."""
```

### Layer 3: Determinism Tests (`tests/unit/test_determinism.py`)

Determinism is the foundation of reproducible RL research.

```python
class TestDeterminism:
    """Verify bit-exact reproducibility."""

    def test_same_seed_same_trajectory(self):
        """Two envs with same seed produce identical trajectories."""

    def test_reset_restores_initial_state(self):
        """reset(seed=X) after many steps equals fresh env with seed=X."""

    def test_determinism_across_runs(self):
        """Save trajectory to file, reload, verify match."""

    def test_recipe_hash_reproducibility(self):
        """Same recipe hash regenerates identical world."""

    @given(seed=st.integers(min_value=0, max_value=2**31-1))
    def test_determinism_any_seed(self, seed):
        """Property: determinism holds for any seed."""
```

**Important**: Test determinism across:
- Different Python versions (if applicable)
- NumPy versions
- Platform (Linux/Mac/Windows)

### Layer 4: Procedural Generation Tests (`tests/unit/test_gen/`)

#### 4.1 Layout and Biome Tests

```python
class TestLayoutGeneration:
    """Verify layout generation invariants."""

    @given(seed=st.integers(0, 2**31-1), size=st.integers(30, 100))
    def test_layout_produces_four_quadrants(self, seed, size):
        """Layout always produces exactly 4 biome quadrants."""

    def test_spawn_corners_are_clear(self):
        """Spawn regions are guaranteed obstacle-free."""

    def test_capture_zone_is_clear(self):
        """Capture zone region is obstacle-free."""
```

#### 4.2 Connectivity Validation

```python
class TestConnectivity:
    """Verify path existence guarantees."""

    @given(seed=st.integers(0, 2**31-1))
    def test_blue_spawn_to_objective_path_exists(self, seed):
        """Blue team can always reach objective."""

    @given(seed=st.integers(0, 2**31-1))
    def test_red_spawn_to_objective_path_exists(self, seed):
        """Red team can always reach objective."""

    def test_validator_fixups_are_minimal(self):
        """Validator carves minimal corridors."""

    def test_validator_terminates(self):
        """Validator doesn't infinite loop on adversarial input."""
```

### Layer 5: Integration Tests (`tests/integration/`)

Test component interactions and contracts.

#### 5.1 Agent-Environment Interface (`test_env_contract.py`)

```python
class TestEnvContract:
    """Verify Gym-like API contract."""

    def test_reset_returns_valid_obs(self):
        """reset() returns (obs_dict, info_dict) with correct shapes."""

    def test_step_returns_valid_tuple(self):
        """step() returns (obs, rewards, term, trunc, info) tuple."""

    def test_observation_space_matches_obs(self):
        """Returned obs matches declared observation_space."""

    def test_action_space_bounds(self):
        """Actions outside bounds are clipped/rejected gracefully."""

    def test_episode_terminates(self):
        """Episodes eventually terminate (timeout or game end)."""

    def test_dead_agents_get_zero_reward(self):
        """Dead agents receive 0 reward (except terminal)."""
```

#### 5.2 Training Loop Integration (`test_training_integration.py`)

```python
class TestTrainingIntegration:
    """Verify training infrastructure works end-to-end."""

    def test_model_forward_pass(self):
        """Model produces valid actions from observations."""

    def test_gradient_flow(self):
        """Gradients flow from loss to all parameters."""

    def test_lstm_state_handling(self):
        """LSTM state is properly reset on episode boundaries."""

    def test_rollout_buffer_shapes(self):
        """Rollout buffer stores correct shapes."""

    def test_ppo_update_decreases_loss(self):
        """PPO update on fixed batch decreases surrogate loss."""
```

#### 5.3 Convergence Tests (`test_convergence.py`)

```python
class TestConvergence:
    """Verify learning signal exists."""

    def test_trivial_task_convergence(self):
        """Agent learns to move toward static target."""

    def test_policy_gradient_on_batch(self):
        """Surrogate loss decreases when optimizing fixed batch."""

    def test_value_prediction_improves(self):
        """Value predictions correlate with returns after training."""
```

### Layer 6: Navigation System Tests (`tests/unit/test_nav/`)

#### 6.1 NavGraph Tests

```python
class TestNavGraph:
    """Verify navigation graph correctness."""

    def test_flat_world_has_ground_nodes(self):
        """Flat world has nodes at z=0."""

    def test_platform_world_has_multiple_levels(self):
        """Elevated platforms create new node levels."""

    def test_clearance_respects_ceiling(self):
        """Low ceilings block node creation."""

    def test_edges_are_bidirectional(self):
        """If A->B exists, B->A exists (for walkable edges)."""

    def test_mech_radius_shrinks_walkable_area(self):
        """Larger mech radius excludes more nodes."""
```

#### 6.2 Pathfinding Tests

```python
class TestPathfinding:
    """Verify A* pathfinding correctness."""

    def test_path_exists_in_open_world(self):
        """Path found between any two walkable points."""

    def test_path_blocked_by_walls(self):
        """Path avoids solid obstacles."""

    def test_path_optimality(self):
        """Found path is optimal (or near-optimal with heuristic)."""

    def test_no_path_returns_gracefully(self):
        """Disconnected points return empty path, not crash."""
```

### Layer 7: Line-of-Sight Tests (`tests/unit/test_los.py`)

#### 7.1 Raycast Correctness

```python
class TestRaycast:
    """Verify raycasting accuracy."""

    def test_empty_world_los_clear(self):
        """LOS is clear in empty world."""

    def test_solid_blocks_los(self):
        """Solid voxels block LOS."""

    def test_glass_allows_los(self):
        """Glass blocks movement but not LOS."""

    def test_foliage_partial_occlusion(self):
        """Foliage has probabilistic LOS blocking (if implemented)."""

    def test_numba_matches_pure_python(self):
        """JIT-compiled raycast matches reference implementation."""
```

#### 7.2 Batch Raycast

```python
class TestBatchRaycast:
    """Verify batch LOS operations."""

    def test_batch_matches_individual(self):
        """Batch results equal individual raycast results."""

    def test_batch_empty_input(self):
        """Empty batch returns empty array."""

    def test_batch_performance(self):
        """Batch is faster than N individual calls."""
```

### Layer 8: Self-Play Infrastructure Tests (`tests/unit/test_arena/`)

#### 8.1 Glicko-2 Rating Tests

```python
class TestGlicko2:
    """Verify rating system correctness."""

    def test_win_increases_rating(self):
        """Winning increases rating."""

    def test_loss_decreases_rating(self):
        """Losing decreases rating."""

    def test_rd_decreases_with_games(self):
        """RD (uncertainty) decreases as games are played."""

    def test_rd_increases_without_games(self):
        """RD increases when player is inactive."""

    def test_expected_score_symmetry(self):
        """E(A, B) + E(B, A) = 1.0"""

    def test_identical_players_draw_expected(self):
        """Equal ratings give 0.5 expected score."""
```

#### 8.2 Match and League Tests

```python
class TestLeague:
    """Verify league management."""

    def test_match_records_outcome(self):
        """Match results are recorded correctly."""

    def test_ratings_update_after_match(self):
        """Ratings update after match completion."""

    def test_model_versioning(self):
        """Models are correctly versioned and tracked."""
```

### Layer 9: Performance Tests (`tests/performance/`)

```python
class TestPerformance:
    """Verify performance requirements."""

    def test_sps_minimum(self):
        """Simulation achieves minimum SPS threshold."""

    def test_memory_stability(self):
        """Memory doesn't grow unboundedly over episodes."""

    def test_raycast_batch_speedup(self):
        """Batch raycast is faster than individual calls."""

    def test_observation_generation_time(self):
        """Observation generation is fast enough for real-time."""
```

### Layer 10: Chaos Engineering Tests (`tests/chaos/`)

```python
class TestChaos:
    """Verify graceful degradation under stress."""

    def test_action_nan_recovery(self):
        """NaN actions are handled without corruption."""

    def test_action_inf_recovery(self):
        """Inf actions are clipped safely."""

    def test_partial_action_dict(self):
        """Missing agent actions use zero action."""

    def test_reset_during_episode(self):
        """Mid-episode reset works correctly."""

    def test_multiple_rapid_resets(self):
        """Rapid reset() calls don't corrupt state."""
```

---

## Part III: RL-Specific Testing Strategies

### 3.1 Reward Function Verification

Reward bugs are the most insidious in RL. Strategies:

1. **Sign Tests**: Verify positive/negative for expected outcomes
2. **Gradient Tests**: Verify reward increases for desired behaviors
3. **Bounds Tests**: Verify rewards are bounded as expected
4. **Attribution Tests**: Verify reward goes to correct agent

```python
class TestRewardCorrectness:
    def test_zone_control_reward_polarity(self):
        """Blue in zone alone -> blue gets positive reward."""

    def test_damage_reward_attribution(self):
        """Shooter gets damage reward, not victim."""

    def test_win_loss_reward_distribution(self):
        """Winner gets W_WIN, loser gets W_LOSE."""

    def test_reward_scale_reasonable(self):
        """Rewards are in expected magnitude range."""
```

### 3.2 Observation-Action Contract Testing

Verify the policy can learn from observations:

```python
class TestObsActionContract:
    def test_observations_differentiate_states(self):
        """Different game states produce different observations."""

    def test_observation_information_content(self):
        """Key state (HP, position, zone) is represented in obs."""

    def test_action_effect_is_observable(self):
        """Action effects appear in subsequent observations."""
```

### 3.3 Multi-Seed Statistical Testing

Single-seed tests can miss bugs. Use statistical testing:

```python
class TestMultiSeed:
    @pytest.mark.parametrize("seed", range(100))
    def test_episode_completion_all_seeds(self, seed):
        """Episode completes without crash for 100 seeds."""

    def test_outcome_distribution(self):
        """Win rate over 1000 games is roughly 50% for symmetric setup."""

    def test_reward_statistics(self):
        """Mean reward per episode is in expected range."""
```

### 3.4 Training Stability Tests

```python
class TestTrainingStability:
    def test_no_nan_in_gradients(self):
        """Gradients never contain NaN."""

    def test_no_exploding_gradients(self):
        """Gradient norms stay bounded."""

    def test_value_predictions_bounded(self):
        """Value function outputs reasonable values."""

    def test_policy_entropy_reasonable(self):
        """Policy entropy doesn't collapse to zero."""
```

---

## Part IV: Mutation Testing Strategy

### 4.1 Target Modules

Focus mutation testing on high-risk modules:

| Module | Risk Level | Rationale |
|--------|------------|-----------|
| `sim/sim.py` | Critical | Core physics, combat |
| `env/env.py` | Critical | Reward calculation, obs generation |
| `gen/layout.py` | High | Map generation affects playability |
| `gen/biomes.py` | High | Biome placement affects gameplay |
| `gen/validator.py` | High | Connectivity guarantees |
| `rl/model.py` | High | Policy architecture |
| `training/ppo.py` | High | Training algorithm |

### 4.2 Custom Mutation Operators

Standard mutations plus RL-specific:

```python
# Standard
- Arithmetic operator substitution (+, -, *, /)
- Comparison operator substitution (<, >, <=, >=, ==, !=)
- Boolean operator substitution (and, or, not)
- Constant substitution (0, 1, -1, etc.)

# RL-Specific
- Reward sign flip (r -> -r)
- Reward scaling (r -> r * 0.1, r -> r * 10)
- Reward delay (r_t -> r_{t+1})
- Observation shuffle (swap obs components)
- Action mapping swap (swap action meanings)
- Discount factor flip (gamma -> 1 - gamma)
```

### 4.3 Kill Ratio Targets

| Category | Target Kill Ratio |
|----------|-------------------|
| Reward mutations | > 95% |
| Physics mutations | > 90% |
| Observation mutations | > 85% |
| Combat mutations | > 90% |
| Generation mutations | > 80% |

---

## Part V: Coverage and Quality Metrics

### 5.1 Code Coverage Targets

| Module | Line Coverage | Branch Coverage |
|--------|---------------|-----------------|
| `sim/` | > 95% | > 90% |
| `env/` | > 90% | > 85% |
| `gen/` | > 85% | > 80% |
| `nav/` | > 90% | > 85% |
| `rl/` | > 85% | > 80% |
| `arena/` | > 90% | > 85% |
| `training/` | > 85% | > 80% |

### 5.2 Property Coverage

Track which invariants are tested:

- [ ] Heat non-negative
- [ ] Stability bounded
- [ ] Dead mechs immobile
- [ ] Shutdown blocks weapons
- [ ] Damage conservation
- [ ] Position bounds
- [ ] Observation finite
- [ ] Reward bounded
- [ ] Determinism (any seed)
- [ ] Connectivity (any seed)

### 5.3 Test Execution Time Targets

| Suite | Target Time |
|-------|-------------|
| Unit tests | < 30 seconds |
| Property tests | < 2 minutes |
| Integration tests | < 5 minutes |
| Performance tests | < 2 minutes |
| Full suite | < 10 minutes |

---

## Part VI: Implementation Priorities

### Phase 1: Foundation (Weeks 1-2)

**Priority: Catch show-stopping bugs before they reach training**

1. **Determinism Test Suite** (`test_determinism.py`)
   - Same-seed trajectory matching
   - Reset state consistency
   - Recipe hash reproducibility

2. **Core Invariant Tests** (`test_invariants.py`)
   - Heat/stability bounds
   - Position bounds
   - Dead mech state
   - Damage conservation

3. **Reward Correctness Tests** (`test_rewards.py`)
   - Sign tests for all reward components
   - Attribution tests
   - Terminal reward distribution

4. **Observation Sanitization** (`test_observations.py`)
   - NaN/Inf detection
   - Dimension consistency
   - Bounds checking

### Phase 2: Combat & Physics (Weeks 3-4)

**Priority: Verify simulation correctness**

5. **Combat Tests** (`test_combat.py`)
   - Weapon mechanics
   - Damage multipliers
   - Splash occlusion
   - Projectile physics

6. **Physics Tests** (`test_physics.py`)
   - Movement mechanics
   - Collision resolution
   - Gravity and jets

7. **Property-Based Sim Tests** (`test_properties.py`)
   - Stateful machine testing
   - Metamorphic relations

### Phase 3: Generation & Navigation (Weeks 5-6)

**Priority: Verify procedural content**

8. **Layout Tests** (`test_layout.py`)
   - Quadrant generation
   - Spawn clearing
   - Zone placement

9. **Connectivity Tests** (`test_connectivity.py`)
   - Path existence guarantees
   - Validator correctness

10. **Navigation Tests** (`test_nav.py`)
    - NavGraph building
    - Pathfinding correctness

### Phase 4: Training Infrastructure (Weeks 7-8)

**Priority: Verify training pipeline**

11. **Training Loop Tests** (`test_training.py`)
    - Forward pass
    - Gradient flow
    - LSTM state handling

12. **PPO Update Tests** (`test_ppo.py`)
    - Loss computation
    - Clipping behavior
    - Advantage normalization

13. **Convergence Tests** (`test_convergence.py`)
    - Trivial task learning
    - Value prediction accuracy

### Phase 5: Self-Play & Performance (Weeks 9-10)

**Priority: Verify infrastructure**

14. **Glicko-2 Tests** (`test_glicko2.py`)
    - Rating updates
    - RD dynamics
    - Expected score

15. **Performance Tests** (`test_benchmark.py`)
    - SPS benchmarks
    - Memory stability
    - Batch operation speedup

16. **Chaos Tests** (`test_chaos.py`)
    - NaN/Inf recovery
    - Malformed input handling

---

## Part VII: Test Infrastructure

### 7.1 Fixtures (`conftest.py`)

```python
@pytest.fixture
def make_mech():
    """Factory for creating test mechs."""

@pytest.fixture
def empty_world():
    """10x10x10 world with no obstacles."""

@pytest.fixture
def wall_world():
    """World with solid wall for LOS testing."""

@pytest.fixture
def small_env():
    """Minimal environment for fast tests."""

@pytest.fixture
def training_env():
    """Environment configured for training tests."""
```

### 7.2 Test Markers

```python
# Speed-based
pytest.mark.slow       # > 10 seconds
pytest.mark.fast       # < 1 second

# Type-based
pytest.mark.unit
pytest.mark.integration
pytest.mark.property
pytest.mark.performance
pytest.mark.chaos

# Feature-based
pytest.mark.determinism
pytest.mark.reward
pytest.mark.physics
pytest.mark.combat
pytest.mark.nav
pytest.mark.training

# Risk-based
pytest.mark.critical   # Must pass before merge
pytest.mark.mutation   # Targets mutation testing
```

### 7.3 CI Configuration

```yaml
# .github/workflows/test.yml
test-suite:
  stages:
    - name: lint
      run: uv run ruff check . && uv run mypy echelon/

    - name: unit-fast
      run: pytest tests/unit -m "fast" --timeout=5

    - name: unit-all
      run: pytest tests/unit --timeout=30

    - name: property
      run: pytest tests/unit -m "property" --timeout=120

    - name: integration
      run: pytest tests/integration --timeout=300

    - name: performance
      run: pytest tests/performance --timeout=120

    - name: mutation
      run: mutmut run --paths-to-mutate=echelon/sim/sim.py
```

---

## Part VIII: Reporting and Monitoring

### 8.1 Test Result Format

```python
{
    "suite": "unit",
    "duration_s": 12.3,
    "passed": 145,
    "failed": 0,
    "skipped": 2,
    "coverage": {
        "line": 0.92,
        "branch": 0.87
    },
    "mutation_score": 0.89,
    "flaky_tests": [],
    "slow_tests": ["test_large_world_generation"]
}
```

### 8.2 Flaky Test Protocol

1. Mark as `@pytest.mark.flaky`
2. Root cause analysis within 1 week
3. Fix or document acceptable variance
4. Remove flaky mark when fixed

### 8.3 Performance Regression Detection

Track metrics over time:
- SPS (steps per second)
- Memory per episode
- Observation generation time
- Training update time

Alert on > 10% regression.

---

## Part IX: Specialist Collaboration

### 9.1 When to Involve Specialists

| Situation | Specialist | Skill Pack |
|-----------|------------|------------|
| Reward shaping questions | drl-expert | yzmir-deep-rl |
| Gradient issues | pytorch-expert | yzmir-pytorch-engineering |
| Terrain/nav bugs | voxel-systems-specialist | - |
| Training instability | drl-expert | yzmir-training-optimization |
| Type system issues | - | axiom-python-engineering |

### 9.2 Escalation Protocol

1. **Test failure in reward code**: Escalate to drl-expert immediately
2. **NaN in gradients**: Escalate to pytorch-expert
3. **Connectivity failures**: Escalate to voxel-systems-specialist
4. **Persistent flaky tests**: Root cause analysis before escalation

---

## Part X: Appendix

### A. Existing Test Inventory

Current tests in the codebase:

| File | Count | Coverage |
|------|-------|----------|
| `test_mechanics.py` | 11 | Combat, pack comm, visibility |
| `test_determinism.py` | 1 | Basic determinism |
| `test_properties.py` | 1 | Stateful sim properties |
| `test_gen_properties.py` | 2 | Layout invariants, connectivity |
| `test_nav_graph.py` | 5 | NavGraph building |
| `test_los.py` | 15 | Raycast, batch LOS |
| `test_api_fuzzing.py` | 2 | NaN/Inf handling |
| `test_convergence.py` | 1 | Batch optimization |
| `test_learning.py` | 1 | Training loop |
| `test_benchmark.py` | 2 | SPS, memory |

**Gap Analysis**: Missing tests for rewards, Glicko-2, EWAR, multi-seed statistics.

### B. Recommended Tools

| Tool | Purpose |
|------|---------|
| pytest | Test runner |
| hypothesis | Property-based testing |
| mutmut | Mutation testing |
| coverage.py | Code coverage |
| pytest-timeout | Test timeouts |
| pytest-xdist | Parallel execution |
| tracemalloc | Memory profiling |

### C. References

- "Testing Deep Reinforcement Learning" (Henderson et al., 2018)
- Glicko-2 rating system (Glickman, 2012)
- Hypothesis documentation (hypothesis.works)
- mutmut documentation (mutmut.readthedocs.io)

---

## Summary

This test suite design provides a comprehensive framework for ensuring Echelon's correctness, robustness, and reliability. The key innovations are:

1. **RL-specific testing layers** that address silent failures in reward, observation, and training code
2. **Property-based testing** for simulation invariants that example-based tests miss
3. **Mutation testing strategy** focused on high-risk modules
4. **Statistical testing** for multi-seed validation
5. **Clear prioritization** for phased implementation

The ultimate goal: **Every bug that could derail training should be caught by a test before it reaches the training loop.**
