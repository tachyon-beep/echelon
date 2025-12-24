# Test Implementation Plan Review

**Reviewer**: DRL Test Engineer
**Date**: 2025-12-24
**Document Reviewed**: `/home/john/echelon/docs/test-implementation-plan.md`

---

## Executive Assessment

The implementation plan is **solid and well-structured**, demonstrating good understanding of RL testing requirements. However, I have identified several gaps, priority adjustments, and implementation concerns that should be addressed before execution.

**Overall Grade**: B+

**Strengths**:
- Correct prioritization of reward and determinism testing
- Good use of property-based testing with Hypothesis
- Comprehensive Glicko-2 coverage
- Practical fixture design

**Weaknesses**:
- Missing critical RL-specific tests (value function, temporal consistency)
- Reward tests lack gradient verification methodology
- LSTM state handling undertested
- Self-play infrastructure gaps (League, match scheduling)

---

## 1. Coverage Gaps

### 1.1 Critical Missing Test Categories

#### GAE (Generalized Advantage Estimation) Verification

The plan has no tests for GAE correctness despite existing `RolloutBuffer.compute_gae()`. This is critical because:
- Incorrect GAE produces training that "works" but learns poorly
- Sign errors in advantage are catastrophic
- The recursive nature makes bugs hard to spot visually

**Missing Tests**:
```python
class TestGAECorrectness:
    def test_gae_simple_episode(self):
        """Manual verification: 3-step episode, known rewards/values."""
        # Hand-compute expected advantages, verify match

    def test_gae_terminal_bootstrap(self):
        """Terminal states don't bootstrap (done=1)."""

    def test_gae_lambda_extremes(self):
        """lambda=0 gives TD(0), lambda=1 gives Monte Carlo."""

    def test_gae_discount_effect(self):
        """gamma=0 ignores future, gamma=1 weights equally."""
```

**Priority**: P0 - Add to Phase 1

---

#### LSTM State Continuity Tests

The plan mentions LSTM state handling but lacks tests for the most common failure mode: incorrect state reset at episode boundaries.

**Missing Tests**:
```python
class TestLSTMStateContinuity:
    def test_lstm_state_reset_on_done(self):
        """LSTM state zeros when done flag is 1."""

    def test_lstm_state_preserved_mid_episode(self):
        """LSTM state carries through when done=0."""

    def test_lstm_state_per_agent_isolation(self):
        """Agent A's done doesn't reset Agent B's state."""

    def test_lstm_state_matches_sequential_inference(self):
        """Batched inference matches single-agent sequential."""
```

**Priority**: P0 - Add to Phase 1 (gradient path depends on this)

---

#### Temporal Consistency Tests

RL environments must maintain temporal consistency: reward at time t must reflect state/action at time t, not t-1 or t+1.

**Missing Tests**:
```python
class TestTemporalConsistency:
    def test_reward_reflects_current_step_action(self):
        """Damage dealt at step t appears in reward[t], not reward[t+1]."""

    def test_observation_reflects_post_step_state(self):
        """obs[t] returned by step() reflects state AFTER action."""

    def test_done_timing(self):
        """done[t] indicates episode ended AFTER step t."""

    def test_info_dict_timing(self):
        """info['kills'] at step t counts kills from that step."""
```

**Priority**: P0 - Off-by-one errors here corrupt learning

---

#### Action Effect Observability

The plan tests observations are valid but not that actions have observable effects:

**Missing Tests**:
```python
class TestActionEffectObservability:
    def test_forward_action_changes_position_observation(self):
        """Forward throttle > 0 changes position in next obs."""

    def test_fire_action_changes_cooldown_observation(self):
        """Firing weapon updates cooldown in next obs."""

    def test_heat_action_observable(self):
        """Vent action reduces heat observable in next step."""
```

**Priority**: P1 - Important for verifying the learning signal exists

---

### 1.2 RL-Specific Failure Modes Not Addressed

#### Reward Scaling Issues

Rewards with wrong magnitude cause:
- Too small: vanishing gradients, no learning
- Too large: exploding gradients, unstable training

**Add to `test_rewards.py`**:
```python
class TestRewardScale:
    def test_reward_magnitude_order(self):
        """Per-step rewards are O(0.01-1.0), not O(100) or O(0.0001)."""

    def test_reward_variance_reasonable(self):
        """Reward variance allows meaningful advantage estimation."""

    def test_terminal_reward_dominance(self):
        """Terminal reward is large enough to dominate trajectory sum."""
```

---

#### Credit Assignment Distance

Long credit assignment distances make learning hard. Test that rewards are timely:

```python
class TestCreditAssignment:
    def test_damage_reward_immediate(self):
        """Damage reward at same step as damage dealt."""

    def test_approach_reward_dense(self):
        """Approach reward given at every step, not just goal."""
```

---

#### Observation Normalization

Observations should be normalized for stable learning:

```python
class TestObservationNormalization:
    def test_observations_centered(self):
        """Mean observation is approximately zero."""

    def test_observations_scaled(self):
        """Observation std is approximately 1.0."""

    def test_no_observation_components_dominate(self):
        """No single obs component has variance >> others."""
```

---

### 1.3 Invariants Missing from Phase 1

Add these to `test_invariants.py`:

```python
class TestCooldownInvariants:
    def test_cooldowns_non_negative(self):
        """All weapon cooldowns >= 0."""

    def test_cooldowns_decrease_over_time(self):
        """Cooldowns decrease by dt each step (if not firing)."""


class TestVelocityInvariants:
    def test_velocity_bounded_by_max_speed(self):
        """Mech velocity magnitude <= spec.max_speed."""

    def test_grounded_mechs_no_vertical_velocity(self):
        """Mechs on ground have vel_z = 0 (unless jumping)."""


class TestWeaponInvariants:
    def test_cant_fire_on_cooldown(self):
        """Weapon doesn't fire if cooldown > 0."""

    def test_painted_target_bonus_applied(self):
        """Painted targets take bonus damage."""
```

---

## 2. Priority Assessment

### 2.1 Tests That Should Move Earlier

| Test | Current Phase | Recommended Phase | Rationale |
|------|---------------|-------------------|-----------|
| GAE correctness | Not in plan | Phase 1 | Breaks training silently |
| LSTM state reset | Phase 4 (implicit) | Phase 1 | Core gradient path |
| Temporal consistency | Not in plan | Phase 1 | Off-by-one breaks learning |
| Observation normalization | Not in plan | Phase 2 | Training stability |

### 2.2 Tests That Should Move Later

| Test | Current Phase | Recommended Phase | Rationale |
|------|---------------|-------------------|-----------|
| EWAR mechanics | Phase 2 | Phase 3 | Lower risk than core physics |
| Model cache LRU | Phase 3 | Phase 4 | Infrastructure, not correctness |
| Multi-seed 100 seeds | Phase 4 | Phase 3 | Should run earlier to catch bugs |

### 2.3 P0/P1/P2 Adjustments

**Current P0 (Critical)**:
- Reward calculation tests (correct)
- Heat/stability invariants (correct)

**Should Also Be P0**:
- GAE verification
- LSTM state handling
- Temporal consistency
- Determinism (already there, but emphasize)

**Downgrade from Implicit P0 to P1**:
- EWAR mechanics (nice-to-have, not blocking training)

---

### 2.4 Tests That Seem Over-Engineered

1. **`test_upset_win_gives_larger_gain`** (Glicko-2): This is mathematically guaranteed by the algorithm. A simpler test of "win increases rating more vs. weaker opponent" would suffice.

2. **Multi-seed 100 tests**: 100 parametrized tests is expensive. Consider:
   - 10 seeds for regular CI
   - 100 seeds for nightly/weekly
   - Mark with `@pytest.mark.slow`

3. **Chaos tests with missing agents**: The current codebase likely handles this gracefully already (agents dict). Verify before writing elaborate tests.

---

## 3. Implementation Concerns

### 3.1 Unrealistic Test Examples

#### `test_zone_control_positive_for_holder`

```python
def test_zone_control_positive_for_holder(self):
    """Team in zone gets positive per-tick reward."""
    # Move blue into zone, red out
    # Step and verify blue reward > 0
    pass
```

**Problem**: Moving mechs into specific positions requires understanding spawn locations, zone location, and pathfinding. This test stub is too simple.

**Better Implementation**:
```python
def test_zone_control_positive_for_holder(self, small_env):
    """Team in zone gets positive per-tick reward."""
    env = small_env
    obs, _ = env.reset(seed=0)

    # Find zone center from env/sim metadata
    zone = env.sim.capture_zone
    zone_center = np.array([zone['center'][0], zone['center'][1], 0])

    # Teleport a blue mech into zone, red mech out
    blue_mech = env.sim.mechs['blue_0']
    red_mech = env.sim.mechs['red_0']
    blue_mech.pos = zone_center.copy()
    red_mech.pos = np.array([0, 0, 0], dtype=np.float32)

    # Take null action, observe reward
    actions = {aid: np.zeros(env.ACTION_DIM) for aid in env.agents}
    _, rewards, _, _, _ = env.step(actions)

    # Blue should get positive zone reward
    assert rewards['blue_0'] > 0, "Blue in zone should get positive reward"
```

---

#### Heat/Stability Property Tests

The current stubs don't actually test sim behavior:

```python
@given(heat=st.floats(min_value=0, max_value=500))
def test_heat_never_negative_after_dissipation(self, make_mech, heat):
    mech = make_mech("m", "blue", [5, 5, 1], "heavy")
    mech.heat = heat
    new_heat = max(0.0, heat - mech.spec.heat_dissipation * 0.05)
    assert new_heat >= 0.0  # This just tests max(0, x) >= 0, not the sim!
```

**Better**: Run actual sim step and check mech.heat afterward.

---

### 3.2 Codebase Structure Challenges

#### Reward Calculation Location

Rewards are computed in `env.py::step()`. To test individual reward components, you'll need:

1. **Option A**: Refactor reward calculation into a separate function (preferred)
2. **Option B**: Test rewards via integration (observe behavior)
3. **Option C**: Inspect `info` dict if rewards are broken down there

**Recommendation**: Check if `info` dict contains reward breakdown. If not, consider adding it for testability.

---

#### EWAR Testing Difficulty

ECM/ECCM effects are subtle:
- ECM reduces sensor quality
- Sensor quality affects observation fidelity
- Testing requires controlling ECM state, sensor calculations, and observation generation

**Recommendation**: Start with a simpler test:
```python
def test_ecm_toggle_changes_mech_state(self, make_mech):
    """ECM toggle action sets mech.ecm_on flag."""
    # This tests the action mapping, not the full effect
```

---

### 3.3 Fixture Design Gaps

Current fixtures lack:

#### Sim-Only Fixture

Many tests need just the Sim, not a full Env:

```python
@pytest.fixture
def sim_with_two_mechs(make_mech):
    """Minimal sim with one mech per team for combat tests."""
    world = VoxelWorld(voxels=np.zeros((10, 20, 20), dtype=np.uint8))
    world.voxels[0, :, :] = VoxelWorld.SOLID  # Floor

    mechs = {
        'blue_0': make_mech('blue_0', 'blue', [5, 10, 1], 'medium'),
        'red_0': make_mech('red_0', 'red', [15, 10, 1], 'medium'),
    }

    sim = Sim(world, dt_sim=0.1, rng=np.random.default_rng(42))
    sim.reset(mechs)
    return sim
```

---

#### Reward Inspection Fixture

```python
@pytest.fixture
def reward_env():
    """Env configured for reward testing with known positions."""
    cfg = EnvConfig(
        world=WorldConfig(size_x=30, size_y=30, size_z=10, obstacle_fill=0.0),
        num_packs=1,
        seed=0,
    )
    env = EchelonEnv(cfg)
    env.reset(seed=0)

    # Expose reward calculation internals if needed
    return env
```

---

## 4. RL-Specific Critique

### 4.1 Reward Tests: Sufficient to Catch Training Bugs?

**Current Coverage**: Good for sign and attribution, weak for:

- **Reward shaping bugs**: Need tests that verify the shaped reward gradient points toward goal
- **Reward delay bugs**: Need temporal consistency tests
- **Reward magnitude bugs**: Need scale tests

**Add This Critical Test**:
```python
def test_optimal_policy_maximizes_reward(self):
    """
    A 'cheating' policy with perfect info gets higher reward than random.
    This proves the reward function is learnable.
    """
    # Run 100 steps with random policy, record total reward
    # Run 100 steps with heuristic policy (move to zone, fire at enemies), record total reward
    # Heuristic reward >> random reward
```

---

### 4.2 Property-Based Approach: Appropriate?

**Verdict**: Mostly appropriate, with caveats.

**Good Uses**:
- Invariant testing (heat >= 0, stability bounded)
- Fuzz testing (random actions don't crash)
- Metamorphic testing (symmetry properties)

**Caution Needed**:
- **Stateful machine testing**: The `SimStateMachine` test may be slow and flaky. Limit `max_examples`.
- **Continuous action spaces**: Hypothesis generates many edge cases (0.0, -0.0, tiny values) that may not be meaningful.

**Recommendation**: Add settings decorators:
```python
@settings(max_examples=50, deadline=500)
class SimStateMachine(RuleBasedStateMachine):
    ...
```

---

### 4.3 Multi-Seed Testing Strategy: Adequate?

**Current Plan**: 100 seeds parametrized

**Issues**:
1. 100 tests is slow (~10+ minutes)
2. Parametrized tests don't aggregate statistics
3. Doesn't test outcome distribution

**Better Approach**:
```python
class TestMultiSeedStatistics:
    def test_episode_completion_rate(self):
        """>=99% of episodes complete without crash."""
        crashes = 0
        for seed in range(100):
            try:
                run_episode(seed)
            except Exception:
                crashes += 1
        assert crashes <= 1  # Allow 1% failure rate

    def test_outcome_balance(self):
        """Win rate is 50% +/- 10% for symmetric setup."""
        blue_wins = sum(run_episode(seed) == 'blue' for seed in range(200))
        rate = blue_wins / 200
        assert 0.4 <= rate <= 0.6

    def test_reward_distribution(self):
        """Mean episode reward is within expected range."""
        rewards = [get_episode_reward(seed) for seed in range(100)]
        mean_reward = np.mean(rewards)
        assert -100 < mean_reward < 100  # Expected range
```

---

### 4.4 Subtle RL Failure Modes Not Covered

#### 1. Observation Aliasing

Different states producing identical observations:
```python
def test_observations_distinguish_positions(self):
    """Mechs at different positions have different observations."""
```

#### 2. Reward Hacking

Agents exploiting reward bugs:
```python
def test_no_reward_from_self_damage(self):
    """Agent doesn't get damage reward from hurting self."""

def test_no_reward_farming(self):
    """Agent can't farm infinite reward by repeated action."""
```

#### 3. Catastrophic Forgetting in LSTM

```python
def test_lstm_remembers_across_steps(self):
    """Information from step 0 affects action at step 10."""
```

#### 4. Advantage Sign Consistency

```python
def test_advantage_signs_consistent_with_returns(self):
    """Positive advantage implies above-average return."""
```

---

## 5. Self-Play Infrastructure

### 5.1 Glicko-2 Tests: Comprehensive Enough?

**Current Coverage**: Good for core math, missing:

```python
class TestGlicko2EdgeCases:
    def test_many_games_converges_rd(self):
        """RD converges to minimum after many games."""

    def test_rating_period_batch_update(self):
        """Multiple games in period update correctly."""

    def test_serialization_roundtrip(self):
        """Rating survives JSON save/load."""
```

**Critical Missing**: The plan tests `update_rating` but the codebase uses `rate()` function. Verify the function name matches.

---

### 5.2 Arena/League Behaviors to Test

The plan completely misses `League` class testing:

```python
# New file: tests/unit/test_league.py

class TestLeague:
    def test_upsert_checkpoint_creates_entry(self):
        """New checkpoint creates league entry."""

    def test_upsert_checkpoint_updates_existing(self):
        """Existing checkpoint updates, doesn't duplicate."""

    def test_top_commanders_ordering(self):
        """top_commanders returns by rating descending."""

    def test_promote_if_topk(self):
        """Candidate promoted to commander if in top K."""

    def test_apply_rating_period_two_phase(self):
        """Rating updates are atomic (no order dependence)."""

    def test_league_save_load_roundtrip(self):
        """League survives JSON serialization."""

    def test_commander_naming_deterministic(self):
        """Same entry_id always gets same commander name."""
```

**Priority**: P1 - Self-play correctness depends on this

---

### 5.3 Model Versioning Concerns

The plan mentions model cache but not:

1. **Checkpoint compatibility**: Old checkpoints load with new model code?
2. **State dict matching**: Keys in saved state match model architecture?
3. **Device handling**: CPU checkpoints load on GPU and vice versa?

```python
class TestModelVersioning:
    def test_checkpoint_loads_on_different_device(self):
        """CPU checkpoint loads on GPU (if available)."""

    def test_checkpoint_state_dict_keys_match(self):
        """Saved keys match model.state_dict().keys()."""

    def test_old_checkpoint_migration(self):
        """Older checkpoint format is handled gracefully."""
```

---

## 6. Suggestions

### 6.1 What to Add

| Addition | Rationale | Priority |
|----------|-----------|----------|
| GAE unit tests | Silent training failure | P0 |
| LSTM state tests | Gradient path correctness | P0 |
| Temporal consistency | Off-by-one bugs | P0 |
| League tests | Self-play correctness | P1 |
| Observation normalization | Training stability | P1 |
| Reward hacking tests | Emergent exploits | P2 |

---

### 6.2 Tests Lost in Translation

From original design document to implementation plan:

1. **Damage Conservation Test**: Original had explicit `total_dealt == total_taken` test. Implementation plan has stub only.

2. **Metamorphic Relations**: Original proposed "action scaling symmetry", "damage commutativity", "team symmetry". None in implementation plan.

3. **Training Loop Tests**: Original had "gradient flow to all layers", "LSTM state handling". Implementation plan defers to Phase 4 without specifics.

4. **Coverage Targets**: Original specified 95% for sim/, 90% for env/. Implementation plan doesn't mention coverage goals.

---

### 6.3 Mutation Testing Strategy

**Current**: Mentions mutmut targeting sim.py, layout.py, biomes.py.

**Missing**:
- Target `env.py` (reward calculations)
- Target `training/ppo.py` (loss calculations)
- Target `arena/glicko2.py` (rating math)

**Custom Operators Needed**:
```python
# Add to mutmut config or custom plugin
REWARD_MUTATIONS = [
    # Flip reward signs
    ('r = W_ZONE_TICK', 'r = -W_ZONE_TICK'),
    # Scale rewards
    ('r = damage * W_DAMAGE', 'r = damage * W_DAMAGE * 0.1'),
    # Remove reward components
    ('reward += zone_reward', 'reward += 0'),
]
```

---

## 7. Recommended Phase 1 Revision

Based on this review, here is the revised Phase 1:

### Phase 1 (Week 1) - Revised

**Goal**: Catch show-stopping bugs before training

#### 1.1 Core Invariants (`test_invariants.py`) - 15 tests
- Heat >= 0
- Stability in [0, max]
- Dead mechs immobile
- Position in bounds
- Cooldowns >= 0
- Velocity <= max_speed
- Damage conservation

#### 1.2 Reward Correctness (`test_rewards.py`) - 12 tests
- Sign tests (zone, damage, kill, death)
- Attribution tests
- Gradient tests (approach)
- Scale tests (magnitude, variance)
- Terminal distribution

#### 1.3 Determinism (`test_determinism.py`) - 6 tests
- Multi-seed (5 seeds)
- Reset consistency
- Recipe hash
- Long episode (1000 steps)
- Combat outcome

#### 1.4 GAE Verification (`test_gae.py`) - 5 tests [NEW]
- Simple episode hand-calc
- Terminal bootstrap
- Lambda extremes
- Gamma effect
- Advantage signs

#### 1.5 LSTM State (`test_lstm_state.py`) - 4 tests [NEW]
- Reset on done
- Preserve mid-episode
- Per-agent isolation
- Batch vs sequential

#### 1.6 Temporal Consistency (`test_temporal.py`) - 4 tests [NEW]
- Reward timing
- Observation timing
- Done timing
- Info dict timing

#### 1.7 Observation Sanitization (`test_observations.py`) - 8 tests
- No NaN/Inf
- Dimension consistent
- Bounds respected
- Contact slots valid
- Self-state present
- Positions differentiate

**Phase 1 Total**: ~54 tests (increased from 38)

---

## 8. Summary of Action Items

### Immediate (Before Implementation)

1. [ ] Add GAE tests to Phase 1
2. [ ] Add LSTM state tests to Phase 1
3. [ ] Add temporal consistency tests to Phase 1
4. [ ] Add League tests to Phase 3
5. [ ] Fix reward test stubs to use actual sim manipulation

### During Implementation

6. [ ] Verify `rate()` function name matches tests
7. [ ] Check if `info` dict has reward breakdown
8. [ ] Add `@settings` decorators to Hypothesis tests
9. [ ] Create `sim_with_two_mechs` fixture

### Post-Phase 2

10. [ ] Run mutation testing on env.py and ppo.py
11. [ ] Verify coverage targets are being met
12. [ ] Add metamorphic relation tests

---

## Appendix: Test Count Summary

| Category | Original Plan | After Review |
|----------|---------------|--------------|
| Phase 1 | 38 | 54 |
| Phase 2 | 27 | 27 |
| Phase 3 | 16 | 28 (+League) |
| Phase 4 | 19 | 15 (-moved) |
| **Total** | **100** | **124** |

The additional tests are critical for catching RL-specific failure modes that would otherwise result in training that "runs but doesn't learn."
