# DRL Review: Simulation Code Analysis

**Reviewer**: Claude Opus 4.5
**Date**: 2025-12-23
**Files Reviewed**:
- `/home/john/echelon/echelon/sim/sim.py`
- `/home/john/echelon/echelon/sim/mech.py`
- `/home/john/echelon/echelon/agents/heuristic.py`
- `/home/john/echelon/echelon/env/env.py`

---

## Summary

This review examines the Echelon simulation from a deep RL training perspective, focusing on reward design, state transitions, determinism, heuristic baseline quality, and observable vs hidden state separation. The codebase demonstrates strong engineering practices overall, but several issues affect learning efficiency and curriculum design.

**Key Findings**:
- **Critical**: Reward signal is extremely sparse with weak shaping that stops prematurely
- **Critical**: Heuristic baseline has exploitable weaknesses that may prevent effective curriculum learning
- **Major**: Death/respawn handling creates immediate zero-reward absorption states
- **Major**: Observable state leaks hidden physics state in critical areas
- **Moderate**: Non-deterministic debris placement affects reproducibility
- **Moderate**: Anti-stuck logic in heuristic uses frame-counting which is non-portable

**Overall Assessment**: The simulation is well-structured for RL, but reward sparsity and curriculum quality are likely bottlenecks for training convergence.

---

## Bugs Found

### 1. Non-Deterministic Debris Placement (sim.py:356)
**File**: `/home/john/echelon/echelon/sim/sim.py:356`

```python
is_solid_wreck = self.rng.random() < p_solid
```

**Issue**: Per-voxel debris type selection is randomized, affecting collision geometry after death. This breaks reproducibility for the same seed.

**Impact**:
- Deterministic replay from seed is compromised
- Affects navigation around destroyed mechs
- May cause divergence in distributed training if seeds aren't synchronized

**Recommendation**: Use mech_id hash + position hash to deterministically select debris type, or always use the same debris type for a given class.

---

### 2. Heuristic Stuck Detection Frame Dependency (heuristic.py:118-133)
**File**: `/home/john/echelon/echelon/agents/heuristic.py:118-133`

```python
if dist_moved < 0.2: # Stuck threshold
    state["stuck_counter"] += 1
else:
    state["stuck_counter"] = max(0, state["stuck_counter"] - 1)

if state["stuck_counter"] > 10: # Stuck for ~10 decision steps (approx 2-3s)
```

**Issue**: Stuck detection uses decision step counting, which is timing-dependent and won't work correctly if `decision_repeat` or `dt_sim` change.

**Impact**:
- Heuristic behavior changes with timing parameters
- Makes it difficult to compare heuristic baseline across different training configurations
- Comment says "~10 decision steps (approx 2-3s)" but calculation assumes specific dt

**Recommendation**: Track time explicitly using `dt = env.config.dt_sim * env.config.decision_repeat` and threshold on accumulated stuck time in seconds.

---

### 3. Observable State Pollution: Velocity in Shutdown (sim.py:218-223)
**File**: `/home/john/echelon/echelon/sim/sim.py:218-223`

```python
def _apply_movement(self, mech: MechState, action: np.ndarray) -> None:
    if mech.shutdown:
        # No control while shutdown, but physics should continue (gravity + damping).
        g_vox = 9.81 / float(self.world.voxel_size_m)
        mech.vel[2] = float(mech.vel[2] - g_vox * self.dt)
        mech.vel *= 0.98
        return
```

**Issue**: Shutdown mechs still have physics applied to velocity, but agents can observe this in the observation (env.py:870 `self_vel`). This leaks "hidden" state since the agent cannot influence velocity during shutdown.

**Impact**:
- Agent may learn spurious correlations with uncontrollable velocity
- Adds noise to the observation space during critical shutdown moments
- Not a bug per se, but violates MDP assumption that observable state should be actionable

**Recommendation**: Either zero out velocity in observations during shutdown, or make shutdown velocity controllable (e.g., emergency stabilization action).

---

## Issues

### 1. Extremely Sparse Reward Signal (env.py:1244-1314)
**File**: `/home/john/echelon/echelon/env/env.py:1244-1314`

**Severity**: Critical

**Description**: The current reward function has only two components:
1. **Approach shaping** (W_APPROACH=0.25): Dense potential-based reward for moving toward zone
2. **Zone control** (W_ZONE_TICK=0.10): Tonnage ratio reward when in zone

**Problems**:

a) **Breadcrumb Termination** (env.py:1302-1308):
```python
if not team_reached_zone[m.team]:
    d0 = dist_to_zone_before.get(aid)
    d1 = dist_to_zone_after.get(aid)
    if d0 is not None and d1 is not None and max_xy > 0.0:
        phi0 = -float(d0 / max_xy)
        phi1 = -float(d1 / max_xy)
        r += W_APPROACH * (phi1 - phi0)
```

The approach reward **stops as soon as any teammate reaches the zone**. This creates several issues:
- Scouts/lights reach first, then other mechs lose navigation signal
- No reward for tactical positioning outside the zone
- Encourages death-charging rather than coordinated assault

b) **No Combat Incentive**: There's zero direct reward for:
- Dealing damage
- Getting kills
- Painting targets
- Suppressing enemies
- Using AMS effectively
- Providing ECCM support

c) **Zone Control Undervalued**: W_ZONE_TICK=0.10 is very small. Over a 60s episode:
- Maximum zone reward per agent ≈ 0.10 * 60 = 6.0 (if uncontested for full episode)
- Maximum approach reward ≈ 0.25 * (sqrt(2) * size) / size ≈ 0.35 (diagonal map traverse)
- Actual combat events yield 0 reward

**Impact on Learning**:
- Agents will learn to move to zone, then have no guidance on what to do
- No incentive to learn weapon usage, target selection, or tactics
- Episode outcomes (win/loss) are not reinforced except through terminal state
- Combat skills must emerge purely from sparse terminal feedback

**Recommendations**:
1. Add combat shaping:
   - Small reward for damage dealt (e.g., +0.01 per point of damage)
   - Larger reward for kills (e.g., +1.0 per kill, scaled by enemy class)
   - Assist rewards for painters (+0.5 when paint leads to kill)
2. Fix breadcrumb logic:
   - Use exponential decay instead of hard cutoff: `alpha = exp(-num_teammates_in_zone)`
   - Or switch to global team centroid approach
3. Increase zone control weight to W_ZONE_TICK=0.5 to make objective more salient
4. Consider survival reward (small negative for dying: -2.0) to discourage suicide charges

---

### 2. Death Creates Immediate Absorbing State (env.py:1295-1297)
**File**: `/home/john/echelon/echelon/env/env.py:1295-1297`

**Severity**: Major

```python
# Dead agents get 0 after the death step.
if not (m.alive or m.died):
    rewards[aid] = 0.0
    continue
```

**Issue**: Dead agents receive exactly 0.0 reward for all remaining steps. Combined with sparse rewards, this means:
- An agent that dies early gets no terminal feedback about whether its team won
- No credit assignment for pre-death contributions
- The `m.died` flag only lasts one step, so death is only "felt" for one decision cycle

**Impact**:
- Value function struggles to propagate terminal outcomes through death
- Agents don't learn to avoid death unless it immediately precedes a loss
- Surviving agents monopolize credit for wins, even if a scout's early sacrifice was critical

**Recommendations**:
1. Backpropagate terminal reward to dead agents (common in team games):
   ```python
   if not m.alive:
       if episode_done:
           rewards[aid] = team_win_bonus  # e.g., +10.0 for win, -10.0 for loss
       else:
           rewards[aid] = 0.0
   ```
2. Add immediate death penalty (e.g., -2.0) on the step where `m.died == True`
3. Track "contribution score" (damage dealt, paints, assists) and distribute terminal reward proportionally

---

### 3. Heuristic Baseline is Exploitable (heuristic.py:29-313)
**File**: `/home/john/echelon/echelon/agents/heuristic.py`

**Severity**: Major

**Description**: The heuristic baseline has several weaknesses that make it a poor curriculum opponent:

a) **Predictable Target Selection** (heuristic.py:136-148):
```python
best = None
for other in sim.mechs.values():
    if not other.alive or other.team == mech.team:
        continue
    d = other.pos - mech.pos
    dist = float(np.linalg.norm(d))
    if best is None or dist < best[0]:
        best = (dist, other, d)
```
Always targets nearest enemy, no target prioritization by threat or class.

b) **Fixed Engagement Range** (heuristic.py:46-47, 199-203):
```python
desired_range: float = 5.5  # Constructor parameter
# ...
if dist < self.desired_range and not (heavy_objective and in_zone):
    forward_throttle = -0.3  # Back off
```
Hardcoded desired range of 5.5 voxels is suboptimal for all classes:
- Heavies have 60-voxel Gauss range but close to 5.5
- Scouts/lights should kite at longer ranges to avoid damage
- No adaptive range based on relative HP or heat state

c) **No Weapon Prioritization** (heuristic.py:218-263):
All weapons fire opportunistically with simple range checks. No logic for:
- Conserving missiles for priority targets
- Using paint before firing missiles from other mechs
- Coordinating focus fire
- Heat management (only vents at 75% heat cap, no predictive cooling)

d) **Simplistic Movement** (heuristic.py:191-207):
```python
forward_throttle = float(np.clip(np.dot(dir_xy, forward), -1.0, 1.0))
strafe_throttle = float(np.clip(np.dot(dir_xy, right), -1.0, 1.0))
# ...
forward_throttle *= self.approach_speed_scale  # 0.5 default
strafe_throttle *= self.approach_speed_scale
```
- Always faces target (no torso twist simulation)
- Strafe scaled down to 25% or 12.5% after speed_scale
- No circle-strafing or advanced movement patterns

e) **No Smoke/AMS Usage** (heuristic.py:274):
```python
a[ActionIndex.SPECIAL] = 0.0 # Smoke not used by heuristic yet
```
Heuristic never uses smoke, so RL agents won't learn to counter it.

f) **Handicap is Naive** (heuristic.py:294-306):
```python
if self.weapon_fire_prob < 1.0:
    wants_fire = (...)
    if wants_fire and float(getattr(env, "rng", np.random).random()) > self.weapon_fire_prob:
        a[ActionIndex.PRIMARY] = 0.0
        # ... zero all weapons
```
Handicap randomly suppresses all weapons uniformly. Better approach: scale damage or introduce random action noise.

**Impact on Curriculum**:
- RL agents can easily exploit heuristic by staying at range and using long-range weapons
- No pressure to learn advanced tactics (smoke, focus fire, target prioritization)
- Handicapped heuristic (fire_prob < 1.0) becomes uncoordinated rather than "weaker but competent"
- Self-play may be necessary earlier than expected

**Recommendations**:
1. **Target Prioritization**: Implement threat scoring (damage potential * (1 - HP ratio))
2. **Dynamic Engagement Range**: Set per-class optimal ranges, adjust based on relative HP/heat
3. **Heat Prediction**: Vent when `heat + next_shot_heat > 0.9 * heat_cap`
4. **Weapon Coordination**: Painters should paint high-value targets; missile mechs should wait for paint
5. **Add Smoke Usage**: Deploy smoke when retreating or covering advances
6. **Better Handicap**: Instead of fire_prob, use aim error (add noise to yaw), slower reaction time (delay actions), or reduced weapon damage

---

### 4. Observable vs Hidden State Blurring (env.py:603-951)
**File**: `/home/john/echelon/echelon/env/env.py:603-951`

**Severity**: Moderate

**Issues**:

a) **Velocity Observable During Shutdown** (env.py:870):
```python
self_vel = (viewer.vel / 10.0).astype(np.float32, copy=False)
```
As mentioned in Bugs #3, shutdown mechs cannot control velocity but it's still in obs.

b) **Smoke Cloud Visibility** (sim.py:136-148):
```python
for cloud in self.smoke_clouds:
    if not cloud.alive:
        continue
    # ...
    if dist2 <= cloud.radius * cloud.radius:
        return False
```
Smoke clouds affect LOS globally, but agents don't observe smoke cloud locations directly. They only see the effects (contacts disappearing). This is actually good design (partial observability), but should be documented.

c) **Suppressed Time** (env.py:866):
```python
suppressed_norm = float(np.clip(viewer.suppressed_time / max(1e-6, float(SUPPRESS_DURATION_S)), 0.0, 1.0))
```
Suppression is observable, which is correct, but the exact duration (SUPPRESS_DURATION_S=1.2) is exposed. Agents could learn to exploit this with frame-perfect timing.

d) **Painted Remaining** (env.py:846):
```python
painted = 1.0 if viewer.painted_remaining > 0.0 else 0.0
```
Binary painted state is observable (good), but exact timer is not (also good). However, agents observe enemies' painted state in contact features (env.py:506), which is consistent with "paint reveals targets to pack."

**Analysis**: Most observable/hidden separation is correct. The shutdown velocity issue is the only clear violation.

**Recommendations**:
1. Zero out `self_vel` in observations when shutdown, or remove shutdown as a state (make it gradual heat penalties instead)
2. Add smoke cloud positions to observation (at least for clouds in sensor range) to make them learnable
3. Document which state is observable vs hidden in `MechState` dataclass comments

---

### 5. Step-Up Teleportation Risk (sim.py:281-300)
**File**: `/home/john/echelon/echelon/sim/sim.py:281-300`

**Severity**: Minor

```python
step_up_max = 1.1 # slightly more than 1 voxel to be safe
# ...
if abs(vel[2]) < 1.0:
    up_trial = target_pos.copy()
    up_trial[2] += step_up_max
    if not self._collides_any(mech, up_trial):
        # We can step up!
        # But we need to check if there's a floor underneath the NEW position
        # to avoid 'teleporting' through a thin wall into air.
        # For v0 sim, we just allow it if the destination is clear.
        target_pos[2] = up_trial[2]
        return True
```

**Issue**: The comment acknowledges the risk: step-up allows teleporting through 1-voxel-thick horizontal barriers if there's no floor check. An agent could exploit this to clip through thin ceilings.

**Impact**:
- Rare in practice (requires specific map geometry)
- Could be exploited if agents discover it
- Deterministic given positions/velocities (not a bug, just exploitable)

**Recommendation**: Add floor check: after stepping up, raycast downward to ensure there's ground within `step_up_max` distance. If not, reject the step-up.

---

## Improvement Opportunities

### 1. Reward Shaping for Combat Skills

**Priority**: High

**Description**: Add dense shaping rewards for combat-relevant actions to bootstrap learning:

```python
# Pseudo-code for improved reward function
W_DAMAGE = 0.005       # Per point of damage dealt
W_KILL = 1.0           # Per kill (scaled by enemy class)
W_ASSIST = 0.5         # Per assist (painter bonus)
W_DEATH = -2.0         # Penalty for dying
W_SURVIVAL = 0.001     # Small survival bonus per step
W_ZONE_APPROACH = 0.25 # Existing
W_ZONE_TICK = 0.5      # Increased from 0.10

# Per-step reward:
r = 0.0
r += W_DAMAGE * mech.dealt_damage  # From MechState scratch stats
r += W_KILL * mech.kills * class_value[target_class]
r += W_SURVIVAL if mech.alive else W_DEATH * int(mech.died)
r += W_ZONE_TICK * zone_ratio
r += W_ZONE_APPROACH * (phi_after - phi_before) if not team_reached_zone else 0.0
```

**Justification**: Current reward is 95% sparse (terminal outcome). Adding combat shaping:
- Accelerates initial learning (agents discover weapons faster)
- Provides continuous feedback during episodes
- Allows curriculum: start with combat-only, then add objective focus

**Risk**: Over-shaping can lead to reward hacking. Mitigate by:
- Using small weights (0.005 for damage is tiny compared to episode length)
- Gradually annealing shaping weights as training progresses
- Monitoring for degenerate behaviors (e.g., farming damage on low-HP enemies)

---

### 2. Heuristic Curriculum Levels

**Priority**: High

**Description**: Implement skill-based heuristic variants for staged curriculum:

```python
class HeuristicPolicy:
    def __init__(self, skill_level: str = "medium"):
        # skill_level in ["trivial", "easy", "medium", "hard", "expert"]
        self.config = {
            "trivial": {
                "aim_error_deg": 45.0,
                "reaction_delay_s": 1.0,
                "fire_prob": 0.3,
                "use_smoke": False,
                "target_priority": False,
            },
            "easy": {
                "aim_error_deg": 20.0,
                "reaction_delay_s": 0.5,
                "fire_prob": 0.6,
                "use_smoke": False,
                "target_priority": False,
            },
            "medium": {  # Current baseline
                "aim_error_deg": 0.0,
                "reaction_delay_s": 0.0,
                "fire_prob": 1.0,
                "use_smoke": False,
                "target_priority": False,
            },
            "hard": {
                "aim_error_deg": 0.0,
                "reaction_delay_s": 0.0,
                "fire_prob": 1.0,
                "use_smoke": True,
                "target_priority": True,  # Threat-based targeting
            },
            "expert": {
                # Add predictive aiming, heat management, coordinated tactics
            },
        }[skill_level]
```

**Benefits**:
- Smoother learning curve (start vs trivial, progress to hard)
- Reduces need for self-play in early training
- Provides consistent evaluation benchmarks

---

### 3. Observation Enhancements

**Priority**: Medium

**Description**:

a) **Smoke Cloud Awareness**: Add visible smoke clouds to observation
```python
# In _obs(), add:
SMOKE_SLOTS = 3
smoke_features = np.zeros((SMOKE_SLOTS, 4), dtype=np.float32)  # [rel_xyz, radius]
visible_smoke = [
    c for c in sim.smoke_clouds
    if c.alive and np.linalg.norm(c.pos - viewer.pos) <= radar_range
]
for i, cloud in enumerate(sorted(visible_smoke, key=lambda c: np.linalg.norm(c.pos - viewer.pos))[:SMOKE_SLOTS]):
    rel = (cloud.pos - viewer.pos) / max_dim
    smoke_features[i] = [rel[0], rel[1], rel[2], cloud.radius / max_dim]
```

b) **Cooldown Ratios Instead of Normalized**: Current cooldowns are normalized by 5.0 constant, but weapon cooldowns vary (0.15s for flamer, 4.0s for Gauss). Better to use per-weapon ratios:
```python
laser_cd = viewer.laser_cooldown / max(1e-6, LASER.cooldown_s)
```

c) **Relative Team HP**: Add team-wide HP ratio to observation to help agents understand when to be aggressive vs defensive:
```python
my_team_hp = sum(m.hp for m in sim.mechs.values() if m.alive and m.team == viewer.team)
enemy_team_hp = sum(m.hp for m in sim.mechs.values() if m.alive and m.team != viewer.team)
hp_advantage = (my_team_hp - enemy_team_hp) / max(1.0, my_team_hp + enemy_team_hp)
```

---

### 4. Determinism Hardening

**Priority**: Medium

**Description**: Fix all sources of non-determinism for reproducible training:

a) **Debris Type** (sim.py:356):
```python
# Replace:
is_solid_wreck = self.rng.random() < p_solid

# With:
debris_seed = hash((mech.mech_id, int(mech.pos[0]), int(mech.pos[1]))) & 0x7FFFFFFF
is_solid_wreck = (debris_seed % 100) < int(p_solid * 100)
```

b) **Document RNG Usage**: Add comments to all `self.rng` calls explaining what they affect and whether they're deterministic given the seed.

c) **Seed Logging**: Log the seed used for each episode in training metrics to enable exact replay.

---

### 5. Death Penalty and Terminal Reward Distribution

**Priority**: High

**Description**: Modify reward structure to better handle death and terminal outcomes:

```python
# In env.step(), after computing winner:
if reason is not None:  # Episode ended
    team_win_bonus = {winner: 10.0, "draw": 0.0}
    for aid in self.agents:
        m = sim.mechs[aid]
        if m.team == winner:
            terminal_reward = team_win_bonus.get(winner, 0.0)
        elif winner == "draw":
            terminal_reward = 0.0
        else:
            terminal_reward = -10.0

        # Distribute terminal reward even to dead agents
        rewards[aid] += terminal_reward

# Add death penalty on the death step:
for aid in self.agents:
    m = sim.mechs[aid]
    if m.died:  # Only true for one step after death
        rewards[aid] += -2.0  # Death penalty
```

**Benefits**:
- Dead agents receive terminal outcome feedback
- Immediate death penalty discourages reckless behavior
- Value function can propagate win/loss through entire episode

---

### 6. Add Replay Buffer Prioritization Hooks

**Priority**: Low

**Description**: The current reward structure (sparse + terminal) would benefit from prioritized experience replay (PER). Add metadata to support it:

```python
# In env.step(), add to infos:
for aid in infos:
    infos[aid]["td_error_hint"] = abs(rewards[aid])  # Rough proxy for TD error
    infos[aid]["is_terminal"] = terminations[aid] or truncations[aid]
    infos[aid]["episode_return"] = sum_of_rewards_so_far  # Track cumulative
```

This enables training code to prioritize:
- High-reward steps (kills, zone captures)
- Terminal states
- High-variance steps

---

### 7. Multi-Phase Curriculum

**Priority**: Medium

**Description**: Structure training into phases with different reward emphasis:

**Phase 1: Combat Basics** (Updates 0-1M)
- W_DAMAGE = 0.01 (high)
- W_ZONE_TICK = 0.0 (disabled)
- W_ZONE_APPROACH = 0.0 (disabled)
- Opponent: Heuristic (trivial)
- Goal: Learn to aim, fire, avoid damage

**Phase 2: Objective Introduction** (Updates 1M-3M)
- W_DAMAGE = 0.005
- W_ZONE_TICK = 0.3
- W_ZONE_APPROACH = 0.25
- Opponent: Heuristic (easy → medium)
- Goal: Learn to move to zone while fighting

**Phase 3: Objective Focus** (Updates 3M-10M)
- W_DAMAGE = 0.002
- W_ZONE_TICK = 0.5
- W_ZONE_APPROACH = 0.1
- Opponent: Heuristic (hard) → Self-play
- Goal: Optimize for winning (zone control)

**Phase 4: Self-Play** (Updates 10M+)
- W_DAMAGE = 0.0
- W_ZONE_TICK = 0.5
- W_ZONE_APPROACH = 0.05
- Opponent: League (Glicko-based matching)
- Goal: Emergent tactics, meta-game

---

## Conclusion

The Echelon simulation is well-designed for DRL with clean state transitions, proper POMDP structure, and good separation of concerns. However, the current reward function is critically sparse, and the heuristic baseline has exploitable weaknesses that will limit curriculum learning.

**Immediate Priorities**:
1. Add dense combat shaping rewards (damage, kills, survival)
2. Fix breadcrumb logic for approach shaping
3. Implement heuristic skill levels (trivial → expert)
4. Add terminal reward distribution to dead agents
5. Fix debris non-determinism

**Long-Term Improvements**:
- Multi-phase curriculum with annealing reward weights
- Enhanced observations (smoke clouds, team HP advantage)
- Prioritized experience replay integration
- Advanced heuristic tactics (smoke, coordinated fire, predictive aiming)

With these changes, training should converge more reliably and produce agents that learn combat skills before being asked to optimize for zone control.
