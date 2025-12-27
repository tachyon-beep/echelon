# Reward Curriculum with Automatic Phase Transitions

**Date:** 2025-12-24
**Status:** GREEN LIGHT - Ready for Implementation
**Author:** Brainstorming session with Claude
**Reviewers:** Gemini, ChatGPT, DRL Expert Agent, Simulation Systems Specialist

---

## Review Summary (2025-12-24)

All blocking issues from specialist reviews have been addressed:

| Issue | Source | Resolution | Section |
|-------|--------|------------|---------|
| Reward scale normalization | DRL C1 | Added constant-sum normalization | §3.5 |
| Phase 4 sparsity cliff | DRL C2 | Added 5% shaping floor | §3.6 |
| Free-rider credit assignment | DRL C3 | Counterfactual difference rewards | §8.5 |
| Delayed Brier calibration | DRL H2 | Rolling K-step window | §7.3 |
| Phase transition noise | DRL H3 | 500-episode windows + significance testing | §8.4 |
| Suppression semantics mismatch | Sim C1 | Redefined using stability proxy | §4.5.1 |
| Detection state undefined | Sim C2 | Operational LOS + sensor range check | §4.5.2 |
| Missing trajectory tracking | Sim H2 | Added visited_nav_nodes to MechState | §4.5.3 |
| Missing detection timers | Sim H3 | Added detection_timers to MechState | §4.5.3 |
| NavGraph rebuild cost | Sim H1 | LRU cache by terrain seed | Directive B |

**Remaining Phase 2 Extensions** (not blocking):
- Zone definitions for Assault/Hold objectives (§4.5.4)
- Spawn geometry control
- Force composition flexibility

---

## Overview

This design describes a comprehensive reward curriculum system for Echelon DRL training. The goal is to train disciplined "soldier" agents that follow mission orders, using a 4-phase curriculum that fades from dense shaping rewards to binary terminal rewards based on mission success.

Key insight: **Win = Mission Success, not Team-vs-Team outcome.** Both teams can "win" if they executed their orders correctly.

## 1. Mission Embedding Structure

### 1.1 Squad-Level Mission Verbs (7 types)

| Verb | Description | Success Metric |
|------|-------------|----------------|
| **Assault** | Take and hold objective | Objective captured + force preserved |
| **Hold** | Defend position against attack | Position retained + attacker attrition |
| **Overwatch** | Provide fire support for friendlies | Targets suppressed + friendlies supported |
| **Flank** | Maneuver to enemy side/rear | Achieve angle + remain undetected until engagement |
| **Suppress** | Pin enemies, prevent movement | Enemy movement denied + ammo efficiency |
| **Scout** | Locate and report enemy positions | % enemies found / % terrain covered |
| **Stage** | Move to position, maintain readiness | Transition speed to follow-on mission + resources preserved |

**Stage** is the Warning Order (WARNO) verb - "go here, be ready for something toward there." The agent doesn't know what mission is coming or when.

**Retroactive Evaluation:** Stage success is measured when the follow-on order arrives:
- Time from "GO" to first contact (Assault follow-on)
- Time from "GO" to coverage start (Scout follow-on)
- Time from "GO" to new position (any movement follow-on)
- Resources available at transition (ammo %, heat headroom, cooldowns ready)
- Formation cohesion at transition (did the squad stay together?)

This creates the right incentive: staging isn't about looking busy, it's about being *actually ready* when the real order comes.

### 1.2 Mission Parameters (4 continuous values)

| Parameter | Range | Meaning |
|-----------|-------|---------|
| **risk** | 0.0-1.0 | Expected enemy strength (0=negligible, 1=overwhelming) |
| **loss_appetite** | 0.0-1.0 | Acceptable casualties (0=preserve all, 1=expendable) |
| **time_pressure** | 0.0-1.0 | Urgency (0=take your time, 1=immediate) |
| **grouping** | 0.0-1.0 | Formation tightness (0=dispersed, 1=concentrated) |

### 1.3 Spatial Context (4 values)

```python
spatial_context = [
    objective_distance,      # Normalized distance to objective
    objective_bearing_sin,   # sin(theta) - no discontinuity at wrap
    objective_bearing_cos,   # cos(theta) - no discontinuity at wrap
    terrain_complexity       # LOS complexity of AO (see note below)
]
```

**Note on bearing:** Using `(sin(θ), cos(θ))` instead of `atan2/π` avoids the ±π discontinuity that creates unnecessary learning tax.

**Note on terrain_complexity:** Use voxel density in target zone as a cheap proxy (0.0 = open field, 1.0 = dense urban maze). Don't over-engineer a full LOS analysis for this single float.

### 1.4 Full Embedding (15 dimensions)

```python
@dataclass
class MissionSpec:
    verb: MissionVerb                    # Enum of 6 types
    risk: float                          # 0.0 - 1.0
    loss_appetite: float                 # 0.0 - 1.0
    time_pressure: float                 # 0.0 - 1.0
    grouping: float                      # 0.0 - 1.0
    objective_pos: tuple[float, float, float]
    terrain_complexity: float            # 0.0 - 1.0 (voxel density proxy)

    def to_embedding(self, agent_pos: np.ndarray) -> np.ndarray:
        """15-dimensional mission embedding for observation."""
        verb_onehot = np.zeros(7)  # 7 verbs including Stage
        verb_onehot[self.verb.value] = 1.0

        params = np.array([
            self.risk,
            self.loss_appetite,
            self.time_pressure,
            self.grouping
        ])

        # Relative spatial context (sin/cos avoids bearing discontinuity)
        delta = np.array(self.objective_pos) - agent_pos
        distance = np.linalg.norm(delta) / 200.0  # Normalize by max range
        theta = np.arctan2(delta[1], delta[0])
        bearing_sin = np.sin(theta)
        bearing_cos = np.cos(theta)

        spatial = np.array([distance, bearing_sin, bearing_cos, self.terrain_complexity])

        return np.concatenate([verb_onehot, params, spatial])
```

## 2. Command Hierarchy

### 2.1 Force Structure

| Level | Size | Command Ratio |
|-------|------|---------------|
| **Unit** | 1 mech | - |
| **Squad** | 10 (9+1 leader) | 3:1 subordinate |
| **Platoon** | 32 (30+1+1) | 3 squads + command |
| **Company** | 100 | 3 platoons + command |

### 2.2 Mission Verbs by Level

**Company Level:** Seize, Defend, Delay, Withdraw, Screen
**Platoon Level:** Assault, Hold, Support, Flank, Recon
**Squad Level:** Assault, Hold, Overwatch, Flank, Suppress, Scout

The same 4 parameters (risk, loss_appetite, time_pressure, grouping) apply at all levels, but with different scales and interpretations.

## 3. Curriculum Phases

### 3.1 Four-Phase Structure

| Phase | Progress | Focus | Shaping Weight | Terminal Weight |
|-------|----------|-------|----------------|-----------------|
| **1** | 0-25% | Learn mechanics | 1.0 | 0.0 |
| **2** | 25-50% | Objective focus | 0.6 | 0.2 |
| **3** | 50-75% | Mission execution | 0.2 | 0.6 |
| **4** | 75-100% | Binary terminal | 0.0 | 1.0 |

### 3.2 Reward Components

**Note:** This shows the **target** curriculum `RewardWeights` structure. The existing `echelon/env/rewards.py` module has a zone-control focused `RewardWeights` (zone_tick, approach, damage, kill, etc.). During implementation, extend the existing structure with these mission-focused fields.

```python
@dataclass
class CurriculumRewardWeights:
    # Dense shaping (fades out)
    survival: float          # Staying alive
    damage_dealt: float      # Combat effectiveness
    heat_management: float   # Resource management
    pack_cohesion: float     # Tactical spacing

    # Objective-based (fades in, then out)
    zone_control: float      # Territory objectives
    mission_progress: float  # Milestone completion

    # Terminal only (fades in)
    mission_success: float   # Binary success/failure
```

### 3.3 Continuous Fade Formula

```python
def get_reward_weights(self, progress: float) -> RewardWeights:
    """Smooth interpolation across training."""

    # Shaping fades from 1.0 to 0.0 at 75% progress (Phase 4 start)
    shaping_weight = max(0.0, 1.0 - progress * 1.33)

    # Objective: triangle peaking at 50%, hitting 0 at 75% (Phase 4 = terminal only)
    if progress < 0.25:
        objective_weight = progress * 4.0  # Ramp up
    elif progress < 0.5:
        objective_weight = 1.0  # Peak
    elif progress < 0.75:
        objective_weight = (0.75 - progress) * 4.0  # Fade to 0
    else:
        objective_weight = 0.0  # Phase 4: terminal only

    # Terminal kicks in at 50%, full strength at 100%
    terminal_weight = max(0.0, (progress - 0.5) * 2.0)

    return RewardWeights(
        survival=0.3 * shaping_weight,
        damage_dealt=0.2 * shaping_weight,
        heat_management=0.1 * shaping_weight,
        pack_cohesion=0.1 * shaping_weight,
        zone_control=0.5 * objective_weight,
        mission_progress=0.3 * objective_weight,
        mission_success=1.0 * terminal_weight
    )
```

### 3.4 Verb-Conditioned Shaping (Critical)

**Problem:** Generic "damage_dealt" rewards poison Scout/Flank behavior early by teaching "combat is always good."

**Solution:** Gate shaping components by mission verb:

| Component | Assault | Hold | Overwatch | Flank | Suppress | Scout | Stage |
|-----------|---------|------|-----------|-------|----------|-------|-------|
| damage_dealt | 1.0 | 0.5 | 0.7 | 0.3 | 0.2 | 0.0 | 0.0 |
| survival | 0.5 | 1.0 | 0.8 | 0.7 | 0.5 | 1.0 | 1.0 |
| stealth | 0.0 | 0.0 | 0.3 | 1.0 | 0.0 | 0.8 | 0.5 |
| detection | 0.0 | 0.3 | 0.5 | 0.2 | 0.3 | 1.0 | 0.3 |
| suppression_effect | 0.2 | 0.3 | 0.8 | 0.0 | 1.0 | 0.0 | 0.0 |
| zone_progress | 1.0 | 0.0 | 0.0 | 0.5 | 0.0 | 0.3 | 0.8 |
| resource_preservation | 0.3 | 0.5 | 0.4 | 0.6 | 0.3 | 0.7 | 1.0 |

```python
VERB_SHAPING_GATES = {
    MissionVerb.SCOUT: {"damage_dealt": 0.0, "stealth": 0.8, "detection": 1.0},
    MissionVerb.FLANK: {"damage_dealt": 0.3, "stealth": 1.0, "zone_progress": 0.5},
    MissionVerb.SUPPRESS: {"damage_dealt": 0.2, "suppression_effect": 1.0},
    MissionVerb.STAGE: {"damage_dealt": 0.0, "zone_progress": 0.8, "resource_preservation": 1.0},
    # ... etc
}

def get_verb_gated_shaping(self, verb: MissionVerb, base_weights: dict) -> dict:
    gates = VERB_SHAPING_GATES.get(verb, {})
    return {k: v * gates.get(k, 1.0) for k, v in base_weights.items()}
```

**Additionally, modulate by mission parameters:**
- `loss_appetite` scales casualty penalties (low = harsh penalty for deaths)
- `time_pressure` amplifies progress-per-time and penalizes dithering

### 3.5 Reward Normalization Strategy (Critical)

**Problem:** Phase transitions cause reward scale discontinuities that destabilize value function learning. If Phase 1 rewards sum to ~5.0 per step and Phase 2 sums to ~2.0, the value function must re-learn magnitude.

**Solution:** Normalize all reward weights to a constant sum across phases:

```python
REWARD_BUDGET = 1.0  # Constant total reward weight across all phases

def get_normalized_weights(self, progress: float) -> RewardWeights:
    """Ensure rewards sum to constant value regardless of phase."""
    raw_weights = self.get_reward_weights(progress)

    # Compute raw sum
    raw_sum = (
        raw_weights.survival + raw_weights.damage_dealt +
        raw_weights.heat_management + raw_weights.pack_cohesion +
        raw_weights.zone_control + raw_weights.mission_progress +
        raw_weights.mission_success
    )

    # Scale to constant budget
    scale = REWARD_BUDGET / max(raw_sum, 1e-8)

    return RewardWeights(
        survival=raw_weights.survival * scale,
        damage_dealt=raw_weights.damage_dealt * scale,
        # ... etc
    )
```

**Per-Verb Normalization:** Verb-conditioned gates (Section 3.4) also need normalization to prevent distribution shift:

```python
def get_verb_gated_shaping(self, verb: MissionVerb, base_weights: dict) -> dict:
    gates = VERB_SHAPING_GATES.get(verb, {})
    gated = {k: v * gates.get(k, 1.0) for k, v in base_weights.items()}

    # Re-normalize after gating
    gate_sum = sum(gated.values())
    base_sum = sum(base_weights.values())
    if gate_sum > 1e-8 and base_sum > 1e-8:
        scale = base_sum / gate_sum
        gated = {k: v * scale for k, v in gated.items()}

    return gated
```

### 3.6 Phase 4 Shaping Floor (Critical)

**Problem:** Pure terminal reward (Phase 4) creates sparsity cliff. With ~240 decision steps per episode and reward only at terminal, credit assignment degrades catastrophically.

**Decision:** Keep 5% shaping floor in Phase 4:

```python
def get_reward_weights(self, progress: float) -> RewardWeights:
    # Shaping floor: never go below 5% to maintain learning signal
    SHAPING_FLOOR = 0.05

    # Shaping fades from 1.0 to 0.05 (not 0.0)
    shaping_weight = max(SHAPING_FLOOR, 1.0 - progress * (1.0 - SHAPING_FLOOR) / 0.75)

    # ... rest unchanged
```

**Rationale:**
- 5% shaping provides weak but consistent learning signal
- Terminal reward still dominates at 95% weight in Phase 4
- Prevents catastrophic policy collapse when entering Phase 4
- Alternative: Use Phase 4 as evaluation-only (no training), but this wastes samples

## 4. Mission Success Criteria

### 4.1 Core Principle

**Win = Mission Success**, not team-vs-team outcome. Both teams can "succeed" if they executed orders correctly.

```python
def evaluate_mission_success(mission: MissionSpec, outcome: MissionOutcome) -> float:
    """Returns 0.0 (failure) to 1.0 (success)."""

    evaluator = MISSION_EVALUATORS[mission.verb]

    # Base success from mission-specific criteria
    base_success = evaluator(mission, outcome)

    # Modify by constraint adherence
    casualty_ratio = outcome.casualties / outcome.initial_strength
    if casualty_ratio > mission.loss_appetite:
        base_success *= 0.5  # Exceeded loss budget

    if outcome.time_taken > mission.time_budget:
        base_success *= 0.8  # Exceeded time budget

    return base_success
```

### 4.2 Per-Verb Evaluation

```python
def _scout_evaluator(m: MissionSpec, o: MissionOutcome) -> float:
    """Scout mission: find enemies or confirm area clear."""
    # Edge case: empty room - confirming 0 enemies is success
    if o.total_enemies == 0:
        enemy_score = 1.0  # Success! Confirmed no enemies present.
    else:
        enemy_score = o.enemies_detected / o.total_enemies

    terrain_score = o.terrain_covered / o.total_terrain
    return enemy_score * 0.6 + terrain_score * 0.4

MISSION_EVALUATORS = {
    MissionVerb.SCOUT: _scout_evaluator,
    MissionVerb.ASSAULT: lambda m, o: (
        float(o.objective_captured) * 0.7 +
        (1.0 - o.casualties / o.initial_strength) * 0.3
    ),
    MissionVerb.HOLD: lambda m, o: (
        float(o.position_retained) * 0.6 +
        o.attacker_attrition * 0.4
    ),
    # ... etc
}
```

### 4.3 Terminal Reward

```python
# Final reward is symmetric around 0
terminal_reward = 2.0 * mission_success - 1.0  # Maps [0,1] to [-1,1]
```

### 4.4 Anti-Cheese Metric Definitions (Critical)

**Problem:** Loosely defined metrics get exploited:
- "Coverage" by spinning in place while sensor raycasts tick cells
- "Suppression" by shooting walls near enemies to trigger flags
- "Undetected" by kiting detection thresholds without doing task

**Solution:** Define metrics in terms of world-state-changes that are hard to fake:

| Metric | Exploitable Definition | Hardened Definition |
|--------|------------------------|---------------------|
| **terrain_covered** | LOS raycast hit count | NavGraph nodes physically visited (agent position must enter cell) |
| **enemies_detected** | Any sensor ping | Confirmed track ID with persistence (>2s continuous tracking) |
| **suppression_effect** | Bullets near enemy | Enemy action constraint: reduced move speed OR forced cover state |
| **position_retained** | Still in zone | Zone control percentage weighted by time; contested time penalized |
| **undetected** | Below detection threshold | No enemy acquired tracking (not just "haven't shot yet") |

```python
# Example: Hardened coverage metric
def compute_terrain_covered(agent_trajectory: list[Vec3], nav_graph: NavGraph) -> float:
    """Count unique navmesh nodes agent physically entered."""
    visited_nodes = set()
    for pos in agent_trajectory:
        node = nav_graph.get_nearest_node(pos)
        if node and distance(pos, node.pos) < VISIT_RADIUS:
            visited_nodes.add(node.id)
    return len(visited_nodes) / len(nav_graph.nodes)

# Example: Hardened suppression metric
def compute_suppression_effect(target: Mech, suppression_events: list) -> float:
    """Measure actual movement denial, not just bullets-near."""
    time_movement_denied = sum(
        e.duration for e in suppression_events
        if e.target == target and e.caused_cover_state
    )
    return min(1.0, time_movement_denied / TARGET_SUPPRESS_TIME)
```

### 4.5 Simulation Integration (Critical)

This section maps design metrics to actual Echelon simulation state and identifies required state extensions.

#### 4.5.1 Suppression Semantics Decision

**Current Sim State:** `mech.suppressed_time` affects stability regeneration only (sim.py:1413-1414). There is no "movement denied" or "cover state" in the simulation.

**Decision:** Accept current semantics for Phase 1. Redefine suppression success as "target stability kept low":

```python
def compute_suppression_effect_v1(target: MechState, suppress_duration: float) -> float:
    """
    V1: Use stability as proxy for suppression effectiveness.

    Suppression in Echelon debuffs stability regen. A suppressed target
    that takes additional hits will be knocked down sooner.
    """
    # Target was under suppression for this duration
    # Success = how much stability was degraded during suppression
    if suppress_duration <= 0:
        return 0.0

    # Measure: low stability during suppression = good suppression
    stability_ratio = target.stability / target.spec.max_stability
    return suppress_duration * (1.0 - stability_ratio)
```

**Future Extension (Phase 2+):** Add movement penalty to suppressed mechs:
```python
# In sim._apply_movement():
if mech.suppressed_time > 0:
    move_speed *= SUPPRESS_SPEED_PENALTY  # e.g., 0.5
```

#### 4.5.2 Detection State Definition

**Current Sim State:** No per-enemy "tracking" persistence. LOS checks are instantaneous.

**Operational Definition:** "Detected" = any enemy has LOS + sensor range to this mech:

```python
def is_detected(self, mech: MechState) -> bool:
    """Returns True if any enemy can currently see this mech."""
    for enemy in self.enemies_of(mech):
        if not enemy.alive or enemy.shutdown:
            continue

        # Check LOS
        if not self.has_los(enemy.pos, mech.pos):
            continue

        # Check sensor range (degraded by ECM/ECCM)
        quality = self._sensor_quality(enemy)
        dist = np.linalg.norm(enemy.pos - mech.pos)
        effective_range = BASE_SENSOR_RANGE * quality

        if dist <= effective_range:
            return True

    return False

BASE_SENSOR_RANGE = 150.0  # meters, modified by mech class
```

#### 4.5.3 Required MechState Extensions

Add these fields to track metrics across episode:

```python
@dataclass
class MechState:
    # ... existing fields ...

    # NEW: Trajectory tracking for coverage metric
    visited_nav_nodes: set[int] = field(default_factory=set)

    # NEW: Detection timers for confirmed tracking (>2s persistence)
    detection_timers: dict[str, float] = field(default_factory=dict)
    # Key = enemy mech_id, Value = continuous seconds of LOS

    # NEW: Individual contribution metrics for counterfactual credit
    damage_dealt_this_episode: float = 0.0
    enemies_detected_this_episode: int = 0
    suppression_time_inflicted: float = 0.0
```

**Update in Sim per-step:**

```python
def _update_tracking_state(self, mech: MechState, dt: float) -> None:
    """Update per-step tracking metrics."""

    # Update visited nodes (O(1) with lookup table)
    if self._nav_lookup is not None:
        ix = int(mech.pos[0] / self.voxel_size)
        iy = int(mech.pos[1] / self.voxel_size)
        iz = int(mech.pos[2] / self.voxel_size)
        if 0 <= iz < self._nav_lookup.shape[0]:
            node_id = self._nav_lookup[iz, iy, ix]
            if node_id >= 0:
                mech.visited_nav_nodes.add(node_id)

    # Update detection timers
    visible_enemies = self._get_visible_enemies(mech)
    visible_ids = {e.mech_id for e in visible_enemies}

    # Increment timers for visible enemies
    for enemy_id in visible_ids:
        if enemy_id in mech.detection_timers:
            mech.detection_timers[enemy_id] += dt
        else:
            mech.detection_timers[enemy_id] = dt

    # Reset timers for enemies that left LOS
    for enemy_id in list(mech.detection_timers.keys()):
        if enemy_id not in visible_ids:
            del mech.detection_timers[enemy_id]

def get_confirmed_detections(self, mech: MechState, min_duration: float = 2.0) -> int:
    """Count enemies with persistent tracking (>2s continuous LOS)."""
    return sum(1 for t in mech.detection_timers.values() if t >= min_duration)
```

#### 4.5.4 Simulation-Reward Coupling Summary

| Metric | Sim State | Cost | Status |
|--------|-----------|------|--------|
| survival | `mech.alive` | O(1) | ✅ Direct |
| damage_dealt | `mech.damage_dealt_this_episode` | O(1) | ✅ Add field |
| heat_management | `mech.heat / spec.heat_cap` | O(1) | ✅ Direct |
| pack_cohesion | Distance between pack members | O(pack²) | ✅ Compute |
| terrain_covered | `mech.visited_nav_nodes` | O(1) per step | ✅ Add field |
| enemies_detected | `get_confirmed_detections(mech)` | O(enemies) | ✅ Add field |
| suppression_effect | Stability during `suppressed_time` | O(1) | ✅ Redefine |
| is_detected | `is_detected(mech)` | O(enemies) | ✅ Add method |
| zone_control | Distance to objective zone | O(1) | ⚠️ Add zone definition |

## 5. Scenario Generator

### 5.1 The "Dungeon Master" Pattern

The scenario generator creates tactical problems (puzzles) for the agent to solve, not just random battles.

```python
@dataclass
class ScenarioSpec:
    """A tactical problem to solve."""
    # Mission orders
    mission: MissionSpec

    # The Challenge (unknown to agent)
    enemy_count: int                    # Agent discovers through play
    enemy_disposition: str              # "patrol", "ambush", "defensive"
    enemy_stealth_tier: float           # Affects detection difficulty

    # The Assets (known to agent)
    friendly_count: int
    friendly_health_dist: str           # "full", "mixed", "critical"
    friendly_composition: str           # "balanced", "scout_heavy", "assault_heavy"

    # Terrain context
    terrain_type: str
    los_complexity: float

class ScenarioGenerator:
    """Procedural scenario factory."""

    def generate(self, mission_verb: str, difficulty: float) -> ScenarioSpec:
        template = self.scenario_bank[mission_verb]
        return template.instantiate(difficulty=difficulty)
```

### 5.2 Scenario Templates

| Verb | Enemy Range | Friendly Range | Key Tension |
|------|-------------|----------------|-------------|
| Scout | 0-2.0 packs | 0.2-1.0 packs | Find vs Survive |
| Assault | 0.5-1.5 packs | 0.6-1.0 packs | Overwhelm vs Preserve |
| Hold | 1.0-2.5 packs | 0.4-1.0 packs | Attrition vs Position |
| Overwatch | 0-1.0 packs | 0.3-0.8 packs | Patience vs Response |
| Flank | 0.8-1.5 packs | 0.4-0.8 packs | Speed vs Detection |
| Suppress | 0.5-1.2 packs | 0.5-1.0 packs | Ammo vs Effect |
| Stage | 0-1.0 packs | 0.3-1.0 packs | Readiness vs Exposure |

## 6. Scenario Levers

### 6.1 Terrain Levers

```python
@dataclass
class TerrainConfig:
    map_size: int                    # 60-200 voxels
    terrain_type: str                # "urban", "forest", "industrial", "open"
    los_complexity: float            # 0.0 (open) to 1.0 (dense)
    cover_density: float             # Available hard cover
    elevation_variance: float        # Flat vs multi-level
    choke_points: int                # Forces engagement funnels
    objective_placement: str         # "center", "edge", "distributed"
```

### 6.2 Force Levers

```python
@dataclass
class ForceConfig:
    strength: float                  # Pack-equivalent (1.0 = 10 mechs)
    composition: dict[str, int]      # {"heavy": 1, "medium": 5, "light": 3, "scout": 1}
    health_distribution: str         # "full", "mixed", "critical"
    avg_health: float                # 0.3-1.0
    cohesion: str                    # "tight", "spread", "scattered"
    awareness: str                   # "cold", "warm", "hot"

    @property
    def pack_equivalent(self) -> float:
        weights = {"heavy": 2.0, "medium": 1.0, "light": 0.7, "scout": 0.5}
        return sum(weights[k] * v for k, v in self.composition.items()) / 10.0
```

### 6.3 Starting Geometry

```python
@dataclass
class SpawnConfig:
    engagement_distance: float       # 50m - 300m
    friendly_formation: str          # "line", "wedge", "column", "scattered"
    angle_of_approach: float         # 0 (head-on) to 180 (flank)
    initial_los: bool                # Start with visual contact?
```

## 7. Compliance & Status Signaling

### 7.1 No Refuse Button

Agents always *attempt* the mission. The `loss_appetite` parameter produces refusal-like behavior naturally:

- **Low loss_appetite + high risk** = minimal exposure, "malicious compliance"
- **High loss_appetite + high risk** = commit fully, accept casualties

This emerges from reward structure, not explicit coding.

### 7.2 Status Output Signal

```python
class StatusLevel(Enum):
    GREEN = 1.0   # "Proceeding as ordered"
    YELLOW = 0.5  # "Progress stalled / Taking fire"
    RED = 0.0     # "Combat ineffective / Pinned"
```

Added to action space as continuous output [0, 1].

### 7.3 Truth Anchor: Proper Scoring Rules (Critical)

**Problem:** Without external truth anchor, co-trained agents will drift into "preference falsification" - a shared fiction where "GREEN" means whatever equilibrium they settle into. This mirrors real-world "zero-defects mentality" where bad news stops flowing because reporting problems is punished.

**Solution:** Treat status as a probabilistic forecast and score with a proper scoring rule:

```python
def compute_status_calibration_loss(
    reported_status: float,      # Agent's output [0, 1]
    actual_outcome: float,       # Ground truth from episode end
) -> float:
    """Brier score: rewards calibrated forecasts, penalizes overconfidence."""
    # If status = 0.8 (confident), and outcome = 0.2 (bad), heavy penalty
    # If status = 0.3 (pessimistic), and outcome = 0.2, small penalty
    return (reported_status - actual_outcome) ** 2

# Ground truth signals to forecast:
# - survival_odds: Did this agent survive the episode?
# - mission_progress: How much objective progress was made?
# - mobility: Was agent pinned/suppressed for extended time?
```

**Rolling Calibration Window (Critical - addresses delayed credit):**

Terminal-only calibration creates long credit assignment delay (~240 steps). Use rolling K-step window:

```python
CALIBRATION_WINDOW = 20  # ~5 seconds at 4 decisions/second

class RollingCalibrationTracker:
    def __init__(self, window_size: int = CALIBRATION_WINDOW):
        self.status_history: deque[float] = deque(maxlen=window_size)
        self.window_size = window_size

    def record_status(self, status: float) -> None:
        self.status_history.append(status)

    def compute_calibration_bonus(self, current_outcome: float) -> float:
        """
        Compare status from K steps ago to current outcome.

        current_outcome examples:
        - Agent still alive? 1.0 else 0.0
        - Agent under fire? 0.3
        - Agent advancing toward objective? 0.7
        """
        if len(self.status_history) < self.window_size:
            return 0.0  # Not enough history

        past_status = self.status_history[0]  # K steps ago
        brier_error = (past_status - current_outcome) ** 2
        return W_CALIBRATION * (1.0 - brier_error)

W_CALIBRATION = 0.05  # Reduced from 0.1 since computed every step
```

**Why rolling works:**
- Agent reports "GREEN" (0.8) at step t
- At step t+20, agent is under heavy fire (outcome=0.3)
- Calibration penalty applies at t+20, linked to t's prediction
- 20-step delay is tractable for credit assignment (vs 240-step)

**Implementation:**
1. **Phase 1-2:** Rolling calibration reward active
2. **Phase 3-4:** Fade calibration reward weight, but habit persists

**Why this works:** The environment pays out for accuracy, not for being reassuring. Even if commander policy changes, honest forecasting remains the winning strategy. This prevents the "everyone lies, then pretends not to know" equilibrium that humans often reach.

## 8. Phase Transition Logic

### 8.1 Metrics for Transition

```python
@dataclass
class PhaseMetrics:
    window_size: int = 100  # episodes

    # Phase 1 → 2
    survival_rate: float = 0.0
    mission_attempt_rate: float = 0.0

    # Phase 2 → 3
    mission_completion_rate: float = 0.0
    efficiency_score: float = 0.0

    # Phase 3 → 4
    win_rate_vs_curriculum: float = 0.0
    variance: float = 0.0
```

### 8.2 Transition Thresholds

```python
THRESHOLDS = {
    1: {"survival_rate": 0.6, "mission_attempt_rate": 0.7},
    2: {"mission_completion_rate": 0.5, "efficiency_score": 0.4},
    3: {"win_rate_vs_curriculum": 0.6, "variance": 0.2},
}

REGRESSION_THRESHOLD = 0.3  # Drop 30% below threshold → regress
```

### 8.3 Automatic Progression with Regression

```python
def evaluate(self, metrics: PhaseMetrics, current_phase: int) -> int:
    # Check advancement
    if self._meets_threshold(metrics, current_phase):
        return min(4, current_phase + 1)

    # Check regression
    if self._below_regression(metrics, current_phase):
        return max(1, current_phase - 1)

    return current_phase
```

### 8.4 Phase Stability: Hysteresis, Dwell Time, and Statistical Significance

**Problem:** Noisy metrics cause phase oscillation (thrashing between phases).

**Solution:** Add stabilizers with statistical rigor:

```python
@dataclass
class PhaseTransitionConfig:
    # Larger window for statistical power
    window_size: int = 500             # Increased from 100 for significance
    min_dwell_episodes: int = 500      # Minimum episodes before allowing transition

    # Hysteresis margins
    advance_threshold_margin: float = 0.1   # Must exceed threshold by this margin
    regress_threshold_margin: float = 0.3   # Must fall below by this margin

    # Statistical significance
    min_effect_size: float = 0.1       # Cohen's d minimum
    confidence_level: float = 0.95      # 95% confidence required
```

**Statistical Significance Testing:**

```python
from scipy import stats

def is_transition_significant(
    metrics: PhaseMetrics,
    threshold: float,
    direction: str,  # "advance" or "regress"
    config: PhaseTransitionConfig
) -> bool:
    """Require statistical significance for phase transitions."""
    if len(metrics.recent_values) < config.window_size:
        return False

    # Compute 95% confidence interval
    mean = np.mean(metrics.recent_values)
    sem = stats.sem(metrics.recent_values)
    ci_low, ci_high = stats.t.interval(
        config.confidence_level,
        len(metrics.recent_values) - 1,
        loc=mean,
        scale=sem
    )

    if direction == "advance":
        # Lower bound of CI must exceed threshold + margin
        target = threshold + config.advance_threshold_margin
        return ci_low > target
    else:  # regress
        # Upper bound of CI must be below threshold - margin
        target = threshold - config.regress_threshold_margin
        return ci_high < target
```

**Why 500 episodes?**
- 100 episodes → high variance, frequent false transitions
- 500 episodes → stable mean estimates, ~2-3% SEM on binary outcomes
- 1000 episodes → very stable but slow curriculum progression

**Example with significance:**
- Phase 2 threshold: `mission_completion_rate = 0.5`
- Advance target: `0.6` (threshold + 0.1 margin)
- Current mean: `0.62`, 95% CI: `[0.58, 0.66]`
- Decision: No advance (CI lower bound 0.58 < 0.6 target)
- After more training: mean `0.65`, CI: `[0.62, 0.68]`
- Decision: Advance (CI lower bound 0.62 > 0.6 target)

### 8.5 Credit Assignment: Counterfactual Difference Rewards (Critical)

**Problem:** Shared squad rewards create free-rider problem. With 10 agents sharing reward:
- Individual gradient signal is 1/10th strength
- Agents learn to hide behind teammates
- No incentive to take personal risk for team benefit

**Solution:** Counterfactual difference rewards from start (not "if needed later"):

```python
def compute_counterfactual_reward(
    agent: Agent,
    squad_agents: list[Agent],
    mission_outcome: MissionOutcome,
    counterfactual_baseline: str = "leave_one_out"
) -> float:
    """
    Difference reward: How much did THIS agent contribute?

    R_i = R(team) - R(team without agent_i)
    """
    team_reward = evaluate_mission_success(mission_outcome)

    if counterfactual_baseline == "leave_one_out":
        # Compute what would have happened without this agent
        # Option 1: Agent contributes nothing (dies at start)
        # Option 2: Agent takes no actions (stands still)
        # Option 3: Agent replaced by mean policy

        # For Echelon, use agent's individual contribution metrics:
        counterfactual_reward = compute_team_reward_without_agent(
            agent, squad_agents, mission_outcome
        )
    elif counterfactual_baseline == "mean_action":
        # What if agent took average actions?
        counterfactual_reward = compute_mean_action_baseline(
            agent, squad_agents, mission_outcome
        )

    return team_reward - counterfactual_reward

def compute_team_reward_without_agent(
    agent: Agent,
    squad_agents: list[Agent],
    mission_outcome: MissionOutcome
) -> float:
    """
    Estimate team reward if agent had not contributed.

    Uses agent's individual metrics to estimate counterfactual:
    - damage_dealt: If agent did 20% of team damage, counterfactual
      assumes that 20% less damage was dealt
    - enemies_detected: Agent's detections subtracted from total
    - suppression_effect: Agent's suppression subtracted
    """
    # Compute agent's fractional contribution
    team_damage = sum(a.damage_dealt for a in squad_agents)
    agent_damage_frac = safe_div(agent.damage_dealt, team_damage, 0.0)

    team_detections = sum(a.enemies_detected for a in squad_agents)
    agent_detect_frac = safe_div(agent.enemies_detected, team_detections, 0.0)

    # Estimate mission outcome without agent's contribution
    modified_outcome = MissionOutcome(
        objective_captured=mission_outcome.objective_captured,
        enemies_detected=int(mission_outcome.enemies_detected * (1 - agent_detect_frac)),
        damage_dealt=mission_outcome.damage_dealt * (1 - agent_damage_frac),
        # ... other fields
    )

    return evaluate_mission_success(modified_outcome)
```

**Blended Reward Formula:**

```python
def compute_agent_reward(
    agent: Agent,
    squad_agents: list[Agent],
    mission_outcome: MissionOutcome,
    progress: float
) -> float:
    """Final reward with fading individual shaping."""

    # Base: counterfactual difference reward (always active)
    counterfactual = compute_counterfactual_reward(agent, squad_agents, mission_outcome)

    # Individual shaping (fades with curriculum)
    shaping_weight = max(0.05, 1.0 - progress * 1.27)  # 5% floor
    individual_shaping = shaping_weight * (
        agent.survival_bonus * 0.3 +
        agent.positioning_bonus * 0.2 +
        agent.heat_management_bonus * 0.1
    )

    return counterfactual + individual_shaping
```

**Why counterfactual from start:**
- Prevents free-rider learning before it becomes entrenched
- 10 agents with difference rewards ≈ 10 independent learners (gradient-wise)
- Team coordination still emerges because counterfactual measures *contribution*
- Agent hiding behind teammates gets low counterfactual (team does fine without them)

## 9. Integration with EchelonEnv

### 9.1 Modified Environment

**Note:** The existing `RewardComputer.compute(ctx)` pattern in `echelon/env/rewards.py` should be preserved. Use a `CurriculumRewardComputer` subclass that wraps the curriculum logic.

```python
class EchelonEnv:
    def __init__(
        self,
        config: EnvConfig,
        curriculum: CurriculumManager | None = None,
        scenario_gen: ScenarioGenerator | None = None,
    ):
        self.curriculum = curriculum or CurriculumManager.default()
        self.scenario_gen = scenario_gen or ScenarioGenerator(self.curriculum)
        # Use curriculum-aware reward computer (subclass of RewardComputer)
        self.reward_computer = CurriculumRewardComputer(
            curriculum=self.curriculum
        )

    def reset(self, seed=None, options=None):
        # Generate scenario
        self.current_scenario = self.scenario_gen.generate(...)

        # Configure world and spawn forces
        self._spawn_forces(self.current_scenario)

        # Add mission embedding to observation (per-agent, uses agent position)
        for agent_id, agent in self.agents.items():
            obs[agent_id]["mission_embedding"] = self.current_scenario.mission.to_embedding(agent.pos)
        return obs, {}

    def step(self, actions):
        # Build step context (existing pattern from rewards.py)
        ctx = StepContext(...)

        # Compute rewards using curriculum-aware computer
        rewards, components = self.reward_computer.compute(ctx)

        if done:
            mission_outcome = self._get_mission_outcome()
            # Terminal reward added by CurriculumRewardComputer
            terminal = self.reward_computer.compute_terminal(
                self.current_scenario.mission, mission_outcome
            )
            for aid in rewards:
                rewards[aid] += terminal[aid]

        return obs, rewards, done, truncated, info
```

### 9.2 Observation Space Additions

```python
"mission_embedding": Box(low=-1, high=1, shape=(15,))  # 7 verbs + 4 params + 4 spatial (sin/cos can be negative)
```

### 9.3 Action Space Additions

```python
# Status report is an OUTPUT (agent tells commander), not an input
"status_report": Box(low=0, high=1, shape=(1,))  # 0.0=RED, 0.5=YELLOW, 1.0=GREEN
```

**Why action space?** The agent must learn to *actively communicate* its status. If status were an observation, the agent would passively watch a meter it can't control. By making it an action, the agent learns honest reporting through experience:
- Lie "GREEN" when pinned → Commander sends no reinforcement → death
- Report "RED" honestly → Commander adapts → survival

## 10. File Structure

```
echelon/
├── curriculum/                # NEW: Curriculum system
│   ├── __init__.py
│   ├── mission.py             # MissionSpec, MissionVerb, embedding
│   ├── scenario.py            # ScenarioSpec, ScenarioGenerator
│   ├── phase.py               # PhaseMetrics, PhaseTransition logic
│   ├── evaluator.py           # Per-verb success evaluation
│   └── manager.py             # CurriculumManager orchestration
├── env/
│   ├── env.py                 # Orchestration (~1450 lines, slim)
│   ├── rewards.py             # EXISTING: RewardWeights, RewardComputer, StepContext
│   └── observations.py        # EXISTING: ObservationBuilder, ObservationContext
└── traits/                    # Future: personality modifiers
    └── __init__.py
```

**Note (2025-12-26):** The `env/rewards.py` module already contains:
- `RewardWeights` dataclass - ready for curriculum-based interpolation
- `RewardComputer` class - designed to be subclassed for curriculum logic
- `StepContext` dataclass - bundles all per-step state for reward computation

The curriculum system should extend `RewardComputer` rather than modifying `env.py` directly.

## 11. Implementation Order

1. **mission.py** - Core data structures and embedding
2. **phase.py** - Phase transition logic (reward weights now in `rewards.py`)
3. **scenario.py** - Scenario generation with levers
4. **evaluator.py** - Success criteria per verb
5. **manager.py** - Orchestration
6. **rewards.py integration** - Extend `RewardComputer` with curriculum-aware subclass
7. **env.py integration** - Wire `CurriculumManager` into environment reset/step
8. **Training script updates** - Checkpoint curriculum state

## 12. Implementation Directives

**Critical notes for implementation to prevent common DRL coding errors:**

### Directive A: Status Report as Auxiliary Reward

The Brier score calibration (Section 7.3) must be implemented as an **auxiliary reward**, not a loss function modification (which breaks Stable Baselines 3 compatibility):

```python
# CORRECT: Add to reward
calibration_bonus = w_calibration * (1.0 - (reported_status - actual_outcome) ** 2)
reward += calibration_bonus  # w_calibration ≈ 0.1 (small but persistent)

# WRONG: Do NOT modify PPO loss function directly
```

### Directive B: NavGraph Coverage Lookup and Caching

Computing `terrain_covered` via NavGraph spatial search is expensive. Pre-bake node IDs at map generation:

```python
# At map generation time:
self.node_lookup: np.ndarray  # shape (z, y, x), dtype=int32, -1 for no node

def get_node_at_pos(self, pos: Vec3) -> int:
    """O(1) lookup instead of spatial search."""
    ix, iy, iz = int(pos.x / voxel_size), int(pos.y / voxel_size), int(pos.z / voxel_size)
    if 0 <= iz < self.node_lookup.shape[0] and ...:
        return self.node_lookup[iz, iy, ix]
    return -1
```

**NavGraph Caching Strategy (Critical for Scenario Generator):**

Scenario generator may modify terrain per-episode, requiring NavGraph rebuild. This is O(voxels) ~10-50ms per map. At scale (1000s of parallel envs), this becomes a bottleneck.

```python
class NavGraphCache:
    """LRU cache for NavGraphs keyed by terrain seed."""

    def __init__(self, max_size: int = 100):
        self._cache: OrderedDict[int, NavGraph] = OrderedDict()
        self._max_size = max_size

    def get_or_build(self, world: VoxelWorld, seed: int) -> NavGraph:
        """Return cached NavGraph or build and cache new one."""
        if seed in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(seed)
            return self._cache[seed]

        # Build new NavGraph
        graph = NavGraph.build(world)

        # Add to cache
        self._cache[seed] = graph
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)  # Remove oldest

        return graph

# Usage in EchelonEnv:
class EchelonEnv:
    _nav_cache = NavGraphCache(max_size=100)  # Class-level, shared across instances

    def reset(self, seed=None, options=None):
        # ... world generation ...
        self._nav_graph = self._nav_cache.get_or_build(self.world, seed)
```

**Alternative: Lazy NavGraph construction** - Only build NavGraph on first access during episode. Many reward metrics don't need it (survival, damage, heat).

### Directive C: SafeDiv Helper

DRL runs millions of steps. If a denominator *can* be zero, it *will* be zero. Use universally:

```python
def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that handles zero denominators."""
    return numerator / denominator if denominator > 1e-8 else default

# Usage:
enemy_score = safe_div(enemies_detected, total_enemies, default=1.0)  # Empty room = success
```

## 13. Future Extensions

- **Trait System:** Personality modifiers (Aggressive, Cautious, Coward) via reward function adjustments
- **Hierarchical Command:** Platoon/Company level orders decomposed to squad missions
- **Adaptive Opponents:** Enemy skill policy progression alongside friendly curriculum
- **Multi-Objective Missions:** Compound missions with primary/secondary objectives

---

## Appendix A: Example Scenarios

### Scout Mission - "Nervous Empty Room"

```python
ScenarioSpec(
    mission=MissionSpec(
        verb=MissionVerb.SCOUT,
        risk=0.3,
        loss_appetite=0.1,
        time_pressure=0.4,
        grouping=0.3,
    ),
    enemy_count=0,           # Empty - but agent doesn't know
    friendly_count=5,
    terrain_type="urban",
    los_complexity=0.8,
)
# Success: Cover 80%+ terrain, don't lose anyone
```

### Scout Mission - "Hornet's Nest"

```python
ScenarioSpec(
    mission=MissionSpec(
        verb=MissionVerb.SCOUT,
        risk=0.3,              # Command THINKS low risk
        loss_appetite=0.1,
        time_pressure=0.4,
        grouping=0.3,
    ),
    enemy_count=18,           # Surprise! Full strength + reserves
    friendly_count=5,
    terrain_type="urban",
)
# Success: Detect enemies + survive to report
```

### Assault Mission - "Depleted Force"

```python
ScenarioSpec(
    mission=MissionSpec(
        verb=MissionVerb.ASSAULT,
        risk=0.7,
        loss_appetite=0.8,     # "Take that objective at all costs"
        time_pressure=0.9,
        grouping=0.8,
    ),
    enemy_count=8,
    friendly_count=3,
    friendly_health_dist="critical",
)
# Teaches: When to commit vs when constraints force caution
```

---

## Appendix B: Prior Art and Theoretical Foundations

This design builds on established research in curriculum learning, multi-agent credit assignment, and reward specification. Key references and their relevance:

### Curriculum Learning Foundations

| Reference | Key Contribution | Relevance to Our Design |
|-----------|------------------|------------------------|
| **Narvekar et al., 2020** | Curriculum RL survey formalizing task sequencing | 4-phase curriculum with automatic transitions |
| **Florensa et al., 2017** | Reverse curriculum with hysteresis-like progression | Performance windows + significance tests for phase changes |
| **Portelas et al., 2019** (ALP-GMM) | Adaptive task sampling via learning progress | Automatic phase transitions based on where learning is happening |
| **Dennis et al., 2020** (PAIRED) | Adversarial curriculum with solvability constraints | Phase stability rules ensuring difficulty matches capability |

### Goal-Conditioned and Multi-Task RL

| Reference | Key Contribution | Relevance to Our Design |
|-----------|------------------|------------------------|
| **Schaul et al., 2015** (UVFA) | Goal-conditioned policies via embedding | Mission verb + params embedded in observation |
| **Liu et al., 2024** | Logical reward shaping for multi-task MARL | Verb gating table = simplified logical reward shaping |
| **Ma et al., 2025** (CRA) | Shared shaping across tasks | Common shaping components with verb-specific weights |

### Credit Assignment

| Reference | Key Contribution | Relevance to Our Design |
|-----------|------------------|------------------------|
| **Foerster et al., 2018** (COMA) | Counterfactual multi-agent policy gradients | Difference rewards (§8.5) |
| **Tumer & Wolpert, 2004** | Difference rewards theory (D_i = G_team - G_without_i) | Direct implementation of counterfactual credit |
| **Wang et al., 2022** (IRAT) | Blending individual + team rewards | 5% shaping floor retains individual incentive (§3.6) |

### Truthful Communication

| Reference | Key Contribution | Relevance to Our Design |
|-----------|------------------|------------------------|
| **Stangel et al., 2025** | Calibrated confidence via proper scoring rules | Status report Brier score (§7.3) |
| **Brier, 1950** | Quadratic strictly proper scoring rule | Truth anchor ensuring honest reporting |

### Reward Specification Risks

| Reference | Key Insight | Our Mitigation |
|-----------|-------------|----------------|
| **Popov et al., 2017** | Specification gaming (Lego stack → flip block) | Hardened metrics (§4.4) |
| **Amodei et al., 2016** | CoastRunners: shaping creates reward loops | Potential-based shaping, anti-cheese definitions |
| **Pan et al., 2022** | More capable agents exploit flaws more severely | Continuous metric refinement, red-team testing |

### Known Risks and Mitigations

**From prior art analysis, these risks require ongoing attention:**

1. **Manual Tuning Complexity**: Many hand-tuned parameters (gating weights, thresholds). Mitigation: Start with literature-informed defaults, tune empirically.

2. **Credit Assignment Approximation**: Linear subtraction may miss non-linear team dynamics where agents have synergistic effects. Mitigation: Monitor for agents with unexpectedly low counterfactual despite visible contribution.

3. **Emergent Collusion**: Both teams optimizing "mission success" might learn mutual non-engagement. Mitigation: Scenario generator ensures opposing missions (Assault vs Hold) create inherent conflict.

4. **Scalability to New Verbs**: Adding mission types requires updating gating table. Future work: Learned reward weights via meta-gradient RL.

5. **Performance Overhead**: Tracking nav nodes, detection timers adds per-step cost. Mitigation: O(1) lookups, lazy evaluation where possible.

### Future Research Directions (from prior art)

- **Learned Curriculum Schedules**: Meta-gradient RL to tune phase timing automatically
- **Value Decomposition**: VDN/QMIX for neural credit assignment
- **Adversarial Reward Testing**: PAIRED-style loophole discovery
- **Hierarchical Policies**: Options framework with verb-specific sub-policies
- **Human-in-the-Loop**: Expert validation of tactical behaviors

---

*Prior art analysis generated by ChatGPT (2025-12-24). Full document: `docs/reference/Prior Art and Novelty Analysis of the MARL_DRL System Design.pdf`*
