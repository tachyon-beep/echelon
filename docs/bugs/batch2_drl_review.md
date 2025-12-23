# DRL Environment Review - Batch 2

**Reviewer**: Claude Opus 4.5
**Date**: 2025-12-23
**Scope**: `/home/john/echelon/echelon/env/env.py`, `/home/john/echelon/echelon/actions.py`, `/home/john/echelon/echelon/config.py`

---

## Summary

This review identifies **1 critical bug**, **3 significant issues**, and **8 improvement opportunities** in the Echelon DRL environment. The environment implements a PettingZoo-parallel-like API for multi-agent mech combat with complex observation/action spaces. While the overall architecture is sophisticated and well-designed, several bugs and design choices could significantly impact training stability and sample efficiency.

**Critical Finding**: Missing class attribute `EWAR_DIM` causes runtime crash in error messages (line 1005).

**Key Issues**:
- Action space validation references undefined attribute
- Observation space lacks normalization for several critical features
- Reward shaping has potential scale mismatch and sparse combat signals
- Missing Gym-standard space definitions

---

## Bugs Found

### BUG-1: Missing EWAR_DIM Class Attribute (CRITICAL)
**File**: `/home/john/echelon/echelon/env/env.py:1005`
**Severity**: Critical (Runtime Crash)

The error message in `step()` references `self.EWAR_DIM`, which is never defined as a class attribute.

```python
# Line 1005
raise ValueError(
    f"action[{aid!r}] has size {a.size}, expected {self.ACTION_DIM} "
    f"(base={self.BASE_ACTION_DIM}, target={self.TARGET_DIM}, ewar={self.EWAR_DIM}, "
    #                                                                    ^^^^^^^^^^
    f"obs_ctrl={self.OBS_CTRL_DIM}, comm_dim={self.comm_dim})"
)
```

**Impact**: When action dimensionality is incorrect, the environment crashes with `AttributeError: 'EchelonEnv' object has no attribute 'EWAR_DIM'` instead of the intended helpful error message.

**Root Cause**: Documentation mentions "EWAR (2): [ECM, ECCM]" (line 127), but EWAR controls are encoded in the SECONDARY action slot (line 1055-1065), not as separate dimensions. The docstring is outdated or the implementation changed.

**Fix**: Either:
1. Remove EWAR_DIM from error message (if EWAR is embedded in base actions)
2. Add `EWAR_DIM = 0` as a class constant to clarify EWAR is not a separate dimension

**Test Coverage Gap**: `/home/john/echelon/tests/unit/test_api_fuzzing.py` tests incorrect action sizes but doesn't catch this because it uses valid `env.ACTION_DIM`.

---

## Issues

### ISSUE-1: No Gym Space Definitions
**File**: `/home/john/echelon/echelon/env/env.py:112-210`
**Severity**: High (API Compliance)

The `EchelonEnv` class lacks `observation_space` and `action_space` attributes, which are Gym API standards.

**Impact**:
- RL libraries (stable-baselines3, RLlib, CleanRL) expect these attributes for:
  - Automatic network sizing
  - Action/observation validation
  - Wrapper compatibility
- Users must manually track dimensions
- Prevents use of standard Gym wrappers (NormalizeObservation, ClipAction, etc.)

**Current State**:
- Action dim computed dynamically: `self.ACTION_DIM = self.COMM_START + self.comm_dim` (line 163)
- Observation dim computed in `_obs_dim()` (lines 953-963)
- No Box/Dict space objects

**Recommendation**: Add in `__init__`:
```python
from gymnasium.spaces import Box, Dict

self.observation_space = Dict({
    aid: Box(low=-np.inf, high=np.inf, shape=(self._obs_dim(),), dtype=np.float32)
    for aid in self.possible_agents
})

self.action_space = Dict({
    aid: Box(low=-1.0, high=1.0, shape=(self.ACTION_DIM,), dtype=np.float32)
    for aid in self.possible_agents
})
```

**PettingZoo Note**: PettingZoo Parallel API expects `observation_spaces` (plural) and `action_spaces` (plural) as dicts. The class docstring (line 114) claims "PettingZoo-parallel-like API" but doesn't implement these required attributes.

---

### ISSUE-2: Observation Normalization Issues
**File**: `/home/john/echelon/echelon/env/env.py:603-951`
**Severity**: High (Training Stability)

Several observation features have inconsistent or missing normalization, which can harm learning:

#### A) Relative Velocity (lines 470, 870)
```python
# Contact features
rel_vel = (other.vel - viewer.vel).astype(np.float32, copy=False) / 10.0  # line 470

# Self velocity
self_vel = (viewer.vel / 10.0).astype(np.float32, copy=False)  # line 870
```

**Problem**: Hardcoded division by 10.0 assumes max velocity ~10 m/s, but config shows:
- Scout max_speed: 7.0 m/s (`/home/john/echelon/echelon/env/env.py:44`)
- Light: 6.0 m/s (line 58)
- Heavy: 3.3 m/s (line 80)

Actual max velocity appears to be ~7 m/s. With vertical jet acceleration (scouts/lights have `max_jet_accel=16-18`), velocities could exceed 10 m/s, causing unbounded observations.

#### B) Local Map (lines 558-601)
```python
out = np.ones((size, size), dtype=np.float32)  # line 597
out[mask] = occupancy_2d[wy[mask], wx[mask]].astype(np.float32)  # line 599
```

**Problem**: Out-of-bounds cells default to 1.0 (solid), in-bounds cells are boolean {0.0, 1.0}. This is technically normalized but semantically different from other features (most use [-1, 1] or [0, 1] with specific meanings).

#### C) Satellite Telemetry (lines 647-671)
Binary {0.0, 1.0} - **OK**, consistent with local map.

#### D) Acoustic Intensities (lines 692-722)
```python
acoustic_intensities = np.log1p(acoustic_intensities) / 5.0  # line 722
```

**Problem**: Comment says "5.0 is approx ln(150)", but `ln(150) ≈ 5.01`. This works but is a magic constant. If noise levels change, normalization breaks. Should reference actual max noise level from config or compute dynamically.

#### E) Heat Normalization (line 475)
```python
heat_norm = float(np.clip(other.heat / max(1.0, other.spec.heat_cap), 0.0, 2.0))
```

**Problem**: Why clip to 2.0 instead of 1.0? Looking at line 71 (`shutdown = self.heat > self.spec.heat_cap`), mechs can overheat beyond capacity. However, this creates inconsistent feature ranges (most are [0, 1], this is [0, 2]).

**Recommendation**:
- Document that heat can reach 2x capacity
- Or clip to 1.0 and add separate binary "overheated" flag

---

### ISSUE-3: Reward Scale and Shaping Concerns
**File**: `/home/john/echelon/echelon/env/env.py:1239-1315`
**Severity**: Medium (Sample Efficiency)

#### A) Reward Components Have Mismatched Scales (lines 1283-1312)

```python
W_ZONE_TICK = 0.10
W_APPROACH = 0.25
```

**Zone Tick Reward**:
- Range: [0, W_ZONE_TICK] per step
- Max: 0.10 per step
- Over 240 steps (60s episode @ 0.25s decision interval): 24.0 total

**Approach Shaping**:
- Potential: `-distance / max_xy` (normalized to [-1, 0])
- Potential-based shaping: `W_APPROACH * (phi_new - phi_old)`
- Max single-step gain: ~0.25 * (speed * dt / max_xy)
  - Scout at 7 m/s, dt=0.25s, map=100m: 0.25 * (1.75/100) ≈ 0.004 per step
  - Over full episode: ~1.0 total (if moving toward zone entire time)

**Imbalance**: Zone tick rewards (24.0 total) dominate approach rewards (1.0 total) by 24x. This may be intentional (objective-focused), but could cause:
- Agents ignore movement optimization once in zone
- Camping behavior at zone edge
- Difficulty learning approach during early training (sparse signal until zone reached)

#### B) "Breadcrumb" Termination Logic (lines 1286-1308)

```python
team_reached_zone = {
    "blue": in_zone_tonnage["blue"] > 0,
    "red": in_zone_tonnage["red"] > 0
}

# Later:
if not team_reached_zone[m.team]:
    # Apply approach shaping
```

**Problem**: Once ANY team member enters zone, ALL teammates stop receiving approach shaping. This creates:
- **Credit assignment issues**: Scout enters zone first, rest of team still approaching but gets zero shaping
- **Discontinuous reward**: Sudden drop to zero when teammate enters zone
- **Anti-pattern**: Incentivizes NOT having scouts advance (keep breadcrumbs flowing)

**Potential Fix**: Use per-agent zone status, not team-wide:
```python
if not in_zone_by_agent[aid]:
    # Apply approach shaping
```

#### C) No Combat Rewards (lines 1244-1247)

```python
# Reward encodes desired behaviors:
# 1) Move toward the objective (dense shaping).
# 2) Control the objective (tonnage-based ratio reward).
# nothing else.
```

Comment explicitly states "nothing else", but episode stats track:
- kills, assists, damage, laser_hits, missile_launches (lines 226-247)

**Problem**:
- No reward for combat actions → agents may learn to ignore weapons
- Damage/kills only matter if they prevent opponent from contesting zone
- Could lead to "rush and stand" behavior with no defensive shooting

This may be **intentional design** (pure objective-based rewards), but creates risk of:
- Ignoring tactical opportunities (e.g., killing isolated enemy)
- Not learning weapon mechanics until late in training
- Difficulty in curriculum learning (can't reward "dealing damage" early, then "winning" later)

#### D) Dead Agent Reward Handling (lines 1294-1297)

```python
# Dead agents get 0 after the death step.
if not (m.alive or m.died):
    rewards[aid] = 0.0
    continue
```

**Logic Issue**: Dead agents get 0.0 reward, but terminations[aid] is set to True (line 1255). In most RL algorithms:
- Terminated agents stop receiving observations/rewards
- Value bootstrapping uses `V(s') = 0` for terminal states
- GAE/TD computations expect final reward at death step

**Current Behavior**:
- Step T-1: Agent alive, gets reward
- Step T: Agent dies (`m.died=True`), gets final reward
- Step T+1: Agent dead (`m.alive=False, m.died=False`), gets 0.0 but already terminated

The `m.died` flag (from `/home/john/echelon/echelon/sim/mech.py:50`) handles the death step correctly. The `if not (m.alive or m.died)` check ensures only "long-dead" agents get 0.0. This is **correct** but unusual - most envs terminate immediately.

**Minor concern**: If RL code doesn't handle multi-agent terminations correctly, could have off-by-one errors in advantage computation.

---

## Improvement Opportunities

### IMPROVE-1: Action Space Documentation Mismatch
**File**: `/home/john/echelon/echelon/env/env.py:119-133`, `/home/john/echelon/echelon/actions.py:6`

Docstring says action space is 9 base dimensions, but `/home/john/echelon/echelon/actions.py:6` defines `ACTION_DIM = 9`. The actual action space is:

```
Total = BASE (9) + TARGET (5) + OBS_CTRL (4) + COMM (comm_dim)
      = 18 + comm_dim (default: 18 + 8 = 26 dims)
```

**Issue**: New users reading `actions.py` will expect 9-dim actions and get shape errors.

**Fix**: Add comment in `actions.py`:
```python
ACTION_DIM = 9  # Base control actions only. See EchelonEnv for full action space.
```

---

### IMPROVE-2: Observation Dimension Calculation
**File**: `/home/john/echelon/echelon/env/env.py:953-963`

The `_obs_dim()` calculation is complex and error-prone:

```python
def _obs_dim(self) -> int:
    comm_dim = PACK_SIZE * int(max(0, int(getattr(self.config, "comm_dim", 0))))
    # Comment lists 40 self features...
    self_dim = 40
    telemetry_dim = 16 * 16
    return self.CONTACT_SLOTS * self.CONTACT_DIM + comm_dim + int(self.LOCAL_MAP_DIM) + telemetry_dim + self_dim
```

**Issues**:
- `comm_dim` recalculates `PACK_SIZE * self.comm_dim` instead of using `self.comm_dim` directly (inconsistent with line 162)
- `self.LOCAL_MAP_DIM` is already an int, `int()` is redundant
- Magic number `40` for self features - if feature list changes, dimension breaks silently
- Comment says "40" but doesn't validate

**Recommendation**:
1. Add assertions to validate dimension matches concatenation in `_obs()`
2. Compute `self_dim` from actual feature list length
3. Add unit test that checks `len(obs[aid]) == env._obs_dim()` for all agents

---

### IMPROVE-3: Contact Slot Repurposing Logic
**File**: `/home/john/echelon/echelon/env/env.py:673-809`

The contact slot allocation (lines 788-808) is complex with reserved slots, repurposing, and priority lists:

```python
reserved = {"friendly": 3, "hostile": 1, "neutral": 1}
repurpose_priority = ["hostile", "friendly", "neutral"]

# ... later ...
while len(selected) < self.CONTACT_SLOTS:
    filled = False
    for rel in repurpose_local:
        i = used_counts[rel]
        if i < len(contacts[rel]):
            # repurpose
```

**Issues**:
- Asymmetric allocation (3 friendly, 1 hostile, 1 neutral) is not justified in comments
- With `hostile_only` filter, all 5 slots go to hostiles (lines 782-783), but if <5 hostiles visible, slots are empty (no friendlies shown)
- **Observability problem**: Agent can't tell if slot is empty vs. no contacts exist vs. filtered out

**Recommendations**:
1. Add a "valid" bit to contact features (current "visible" bit is always 1.0 for visible contacts)
2. Document rationale for 3:1:1 split (is it because typical engagement is 10v10 with more friendlies nearby?)
3. Consider symmetric allocation (2:2:1 or 2:3:0) for more balanced learning

---

### IMPROVE-4: Target Selection Validation
**File**: `/home/john/echelon/echelon/env/env.py:1034-1050`

Target selection uses argmax over contact slots from the **previous** observation:

```python
# Line 1036
prefs = a[self.TARGET_START : self.TARGET_START + self.TARGET_DIM]
# ...
slots = self._last_contact_slots.get(aid) or []
if i < len(slots):
    cand = slots[i]
    if cand is not None:
        tgt = sim.mechs.get(cand)
        if tgt is not None and tgt.alive and tgt.team != m.team:
            focus_id = str(cand)
```

**Issues**:
- Validation checks `tgt.alive` and `tgt.team != m.team`, but not `sim.has_los(m.pos, tgt.pos)`
- Agent can target enemies behind walls, which may be intentional (fire support, indirect fire), but could cause confusion
- If contact list changes drastically between steps, argmax index may select wrong target
- No penalty for invalid target selection (just silently fails to set `focus_id`)

**Observability Gap**: Agent doesn't know if target selection succeeded. Should add to observation:
```python
target_locked = 1.0 if m.focus_target_id is not None else 0.0
```

---

### IMPROVE-5: Episode Termination Conditions
**File**: `/home/john/echelon/echelon/env/env.py:1316-1360`

Termination logic has three conditions (zone control win, elimination, time up), but "zone control win" sets both `terminations[aid] = True` AND `truncations[aid] = True` (line 1332):

```python
if self.team_zone_score["blue"] >= self.zone_score_to_win or ...:
    winner = ...
    reason = "zone_control"
    for aid in terminations:
        terminations[aid] = True
        truncations[aid] = True  # Why both?
```

**Problem**:
- `terminations` should indicate natural episode end (agent died or objective met)
- `truncations` should indicate artificial cutoff (time limit, external stop)
- Zone control win is a **natural** termination, not a truncation
- Setting both conflicts with Gym semantics: `terminated=True` means "reached terminal state", `truncated=True` means "stopped early"

**Gym Docs**: "terminated and truncated should not both be true"

**Fix**: Only set `terminations[aid] = True` for zone win and elimination. Only set `truncations[aid] = True` for time limit.

**Impact on Learning**:
- Value bootstrapping treats truncations differently: `V(s_terminal) = 0` for terminations, `V(s_truncated) = critic(s_last)` for truncations
- Current code makes zone wins look like truncations, which may undervalue states near victory

---

### IMPROVE-6: Reset Seed Handling
**File**: `/home/john/echelon/echelon/env/env.py:212-218`

Reset handles seed inconsistently:

```python
def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    if seed is not None:
        self.rng = np.random.default_rng(seed)
        episode_seed = int(seed)
    else:
        episode_seed = int(self.rng.integers(0, 2**31 - 1))
    self._last_reset_seed = episode_seed
```

**Issues**:
1. If `seed=None`, uses `self.rng` to generate `episode_seed`, but `self.rng` was initialized with `config.seed` (line 166)
2. Calling `reset(seed=None)` multiple times produces different episodes (good!), but the sequence depends on how many times `reset()` was called (bad for reproducibility)
3. Gym API expects `reset(seed=X)` to reseed the internal RNG, making future `reset()` calls deterministic

**Recommendation**: Follow Gymnasium best practices:
```python
def reset(self, seed: int | None = None, options: dict | None = None):
    if seed is not None:
        self.rng = np.random.default_rng(seed)
    episode_seed = int(self.rng.integers(0, 2**31 - 1))
    # ... rest of reset
```

This way:
- `reset(seed=42)` reseeds the RNG, next `reset()` is deterministic
- `reset()` without seed uses current RNG state

---

### IMPROVE-7: Info Dict Contents
**File**: `/home/john/echelon/echelon/env/env.py:1362-1384`

Info dicts contain useful data, but some inconsistencies:

```python
infos[aid] = {"events": [], "alive": alive}  # line 1257
# Later:
infos[aid]["zone_score"] = dict(self.team_zone_score)
infos[aid]["zone_control"] = float(control)
infos[aid]["in_zone"] = bool(in_zone_by_agent.get(aid, False))
# Later:
if events:
    for aid in infos:
        infos[aid]["events"] = events  # Overwrites empty list from line 1257
```

**Issues**:
1. `infos[aid]["events"]` is initialized to `[]` then overwritten (redundant initialization)
2. All agents get the same `events` list - global events like "kill" are broadcast to all agents, even enemies
3. **Information leak**: Dead agents get outcome info (line 1383-1384)

```python
if self.last_outcome is not None:
    for aid in infos:  # All agents, including dead ones
        infos[aid]["outcome"] = self.last_outcome
```

This is probably fine (dead agents are terminated, RL doesn't use their info), but worth noting.

**Recommendation**:
- Remove redundant `"events": []` initialization
- Consider per-agent event filtering (agents only see events they'd know about)
- Document that info dicts are for logging, not observation (no info leakage to policy)

---

### IMPROVE-8: Navigation Assist Mode
**File**: `/home/john/echelon/echelon/env/env.py:1098-1148`

The nav assist mode (lines 1098-1148) post-processes RL actions to blend with pathfinding:

```python
if self.config.nav_mode == "assist" and self.nav_graph:
    # ...
    alpha = 0.3
    a[ActionIndex.FORWARD] = (1.0 - alpha) * a[ActionIndex.FORWARD] + alpha * fwd_desired
    a[ActionIndex.STRAFE] = (1.0 - alpha) * a[ActionIndex.STRAFE] + alpha * side_desired
```

**Issues**:
1. **Reduces action space expressiveness**: Agent can't fully commit to non-navigation actions (e.g., backing away from enemy)
2. **Curriculum learning anti-pattern**: Agent never learns navigation, depends on assist
3. **Non-Markovian**: Assist behavior depends on cached path state (`self._cached_paths`), which isn't in observation
4. **Evaluation mismatch**: If assist is disabled at test time, policy fails

**Recommendations**:
- Document this is for **curriculum learning** or **action space simplification**, not production
- Add observation features exposing nav assist state (waypoint direction, distance to waypoint)
- Consider hierarchical RL: separate navigation policy vs. combat policy
- Add config flag `nav_mode_eval` to disable assist during evaluation

---

### IMPROVE-9: Performance - Observation Generation
**File**: `/home/john/echelon/echelon/env/env.py:603-951`

The `_obs()` method is called every decision step and does significant computation:

**Potential optimizations**:
1. **Cache satellite telemetry** (lines 655-671): Terrain is static, can compute once per episode
2. **Cache occupancy_2d** (line 655): Recomputed for every agent's local map
3. **LOS/smoke caching** (lines 631-645): Good! But could be vectorized
4. **Acoustic intensity loop** (lines 693-719): Could vectorize with numpy

**Example vectorization for acoustic (pseudo-code)**:
```python
# Compute all pairwise distances once
positions = np.array([m.pos for m in sim.mechs.values()])
noise_levels = np.array([m.noise_level for m in sim.mechs.values()])
dists = cdist(viewer.pos, positions)
intensities = noise_levels / (1.0 + dists**2)
# Then aggregate by quadrant
```

**Impact**: `_obs()` is likely the bottleneck in step latency (20 agents * complex features). Profiling recommended.

---

### IMPROVE-10: Missing Observation: Leg Damage
**File**: `/home/john/echelon/echelon/env/env.py:454-524`

Contact features include `is_legged` (line 478), but not **leg HP**. Agents can see if enemy is legged (mobility penalty), but can't see how close to legging they are.

**Impact**:
- Can't learn to prioritize leg damage
- Can't estimate if one more hit will leg the target
- Asymmetric with hull HP (which is included, line 474)

**Recommendation**: Add `leg_hp_norm` to contact features:
```python
leg_hp_norm = float(np.clip(other.leg_hp / max(1.0, other.spec.leg_hp), 0.0, 1.0))
feat[??] = leg_hp_norm
```

**Note**: This increases `CONTACT_DIM` from 22 to 23, breaking checkpoint compatibility.

---

## Recommendations Summary

**Critical**:
1. Fix BUG-1 (missing EWAR_DIM) immediately - blocks error reporting

**High Priority**:
2. Add Gym space definitions (ISSUE-1) - improves compatibility
3. Fix observation normalization (ISSUE-2) - affects training stability
4. Review reward shaping (ISSUE-3) - major impact on learned behavior

**Medium Priority**:
5. Fix termination vs truncation semantics (IMPROVE-5)
6. Add dimension validation tests (IMPROVE-2)
7. Document/fix target selection observability (IMPROVE-4)

**Low Priority (Polish)**:
8. Contact slot allocation rationale (IMPROVE-3)
9. Info dict cleanup (IMPROVE-7)
10. Navigation assist documentation (IMPROVE-8)
11. Performance profiling (IMPROVE-9)
12. Consider adding leg HP observation (IMPROVE-10)

---

## Testing Gaps

The existing test suite (`/home/john/echelon/tests/unit/test_api_fuzzing.py`) covers NaN/Inf handling but misses:

1. **Action dimension validation**: Test incorrect action sizes to trigger EWAR_DIM bug
2. **Observation dimension validation**: Check `len(obs[aid]) == env._obs_dim()` for all agents
3. **Reward scale tests**: Verify reward magnitudes are in expected ranges
4. **Termination semantics**: Check that zone wins set only `terminated`, not `truncated`
5. **Observation bounds**: Check all features are in documented ranges (especially after normalization changes)
6. **Target selection**: Verify focus_target_id is set correctly and observable

**Recommended new test**:
```python
def test_observation_dimensions():
    env = EchelonEnv(EnvConfig(num_packs=1, seed=42))
    obs, _ = env.reset()
    expected_dim = env._obs_dim()
    for aid, o in obs.items():
        assert len(o) == expected_dim, f"Agent {aid} obs dim mismatch"
        assert np.all(np.isfinite(o)), f"Agent {aid} has non-finite obs"
```

---

## Conclusion

The Echelon environment is a sophisticated multi-agent RL environment with well-designed observation and action spaces. The primary concerns are:

1. **Stability**: Missing normalizations and the EWAR_DIM bug need immediate fixes
2. **Compatibility**: Adding Gym space definitions enables ecosystem tools
3. **Sample Efficiency**: Reward shaping design choices (breadcrumbs, no combat rewards) may hinder learning

The environment shows evidence of careful engineering (LOS caching, bad action handling, deterministic replay), but would benefit from more extensive unit testing of the Gym API surface and reward semantics.
