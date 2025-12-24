# Reward Curriculum with Automatic Phase Transitions

**Date:** 2025-12-24
**Status:** Approved with patches (Gemini + ChatGPT reviews 2025-12-24)
**Author:** Brainstorming session with Claude

## Overview

This design describes a comprehensive reward curriculum system for Echelon DRL training. The goal is to train disciplined "soldier" agents that follow mission orders, using a 4-phase curriculum that fades from dense shaping rewards to binary terminal rewards based on mission success.

Key insight: **Win = Mission Success, not Team-vs-Team outcome.** Both teams can "win" if they executed their orders correctly.

## 1. Mission Embedding Structure

### 1.1 Squad-Level Mission Verbs (6 types)

| Verb | Description | Success Metric |
|------|-------------|----------------|
| **Assault** | Take and hold objective | Objective captured + force preserved |
| **Hold** | Defend position against attack | Position retained + attacker attrition |
| **Overwatch** | Provide fire support for friendlies | Targets suppressed + friendlies supported |
| **Flank** | Maneuver to enemy side/rear | Achieve angle + remain undetected until engagement |
| **Suppress** | Pin enemies, prevent movement | Enemy movement denied + ammo efficiency |
| **Scout** | Locate and report enemy positions | % enemies found / % terrain covered |

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

### 1.4 Full Embedding (14 dimensions)

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
        """14-dimensional mission embedding for observation."""
        verb_onehot = np.zeros(6)
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

```python
@dataclass
class RewardWeights:
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

| Component | Assault | Hold | Overwatch | Flank | Suppress | Scout |
|-----------|---------|------|-----------|-------|----------|-------|
| damage_dealt | 1.0 | 0.5 | 0.7 | 0.3 | 0.2 | 0.0 |
| survival | 0.5 | 1.0 | 0.8 | 0.7 | 0.5 | 1.0 |
| stealth | 0.0 | 0.0 | 0.3 | 1.0 | 0.0 | 0.8 |
| detection | 0.0 | 0.3 | 0.5 | 0.2 | 0.3 | 1.0 |
| suppression_effect | 0.2 | 0.3 | 0.8 | 0.0 | 1.0 | 0.0 |
| zone_progress | 1.0 | 0.0 | 0.0 | 0.5 | 0.0 | 0.3 |

```python
VERB_SHAPING_GATES = {
    MissionVerb.SCOUT: {"damage_dealt": 0.0, "stealth": 0.8, "detection": 1.0},
    MissionVerb.FLANK: {"damage_dealt": 0.3, "stealth": 1.0, "zone_progress": 0.5},
    MissionVerb.SUPPRESS: {"damage_dealt": 0.2, "suppression_effect": 1.0},
    # ... etc
}

def get_verb_gated_shaping(self, verb: MissionVerb, base_weights: dict) -> dict:
    gates = VERB_SHAPING_GATES.get(verb, {})
    return {k: v * gates.get(k, 1.0) for k, v in base_weights.items()}
```

**Additionally, modulate by mission parameters:**
- `loss_appetite` scales casualty penalties (low = harsh penalty for deaths)
- `time_pressure` amplifies progress-per-time and penalizes dithering

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

**Implementation:**
1. **Phase 1-2:** Auxiliary reward for calibrated status (truth anchor active)
2. **Phase 3-4:** Fade auxiliary reward, but calibration habit should persist

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

### 8.4 Phase Stability: Hysteresis and Dwell Time

**Problem:** Noisy metrics cause phase oscillation (thrashing between phases).

**Solution:** Add stabilizers:

```python
@dataclass
class PhaseTransitionConfig:
    min_dwell_episodes: int = 200      # Minimum episodes before allowing transition
    advance_threshold_margin: float = 0.1   # Must exceed threshold by this margin to advance
    regress_threshold_margin: float = 0.3   # Must fall below by this margin to regress

    # Hysteresis: advance threshold is HIGHER than regress threshold
    # This creates a "sticky" zone where neither transition occurs
```

**Example:**
- Advance from Phase 2 requires `mission_completion_rate > 0.6` (0.5 + 0.1 margin)
- Regress to Phase 1 requires `mission_completion_rate < 0.35` (0.5 - 0.15 margin)
- Between 0.35 and 0.6: stay in current phase

### 8.5 Credit Assignment: Squad vs Individual

**Decision:** Use shared squad-level rewards with tiny individual shaping in early phases.

**Rationale:** Per-agent rewards for damage/survival create selfish play that harms mission outcomes. "Soldier agents" need collective success.

```python
def compute_squad_reward(squad_agents: list[Agent], mission_outcome: MissionOutcome) -> float:
    """Shared reward for all squad members."""
    return evaluate_mission_success(mission_outcome)

def compute_individual_shaping(agent: Agent, shaping_weight: float) -> float:
    """Tiny individual shaping, fades with curriculum progress."""
    # Only active in Phase 1-2, scaled by shaping_weight
    return shaping_weight * (
        agent.survival_bonus +
        agent.positioning_bonus
    )

# Final reward per agent:
# reward = squad_reward + individual_shaping * (1.0 - progress)
```

**Alternative (if needed):** Difference rewards or counterfactual credit if agents learn to free-ride on squad success.

## 9. Integration with EchelonEnv

### 9.1 Modified Environment

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
        weights = self.curriculum.get_reward_weights()
        reward = self._compute_reward(weights)

        if done:
            mission_outcome = self._get_mission_outcome()
            success = self.curriculum.compute_terminal_reward(
                self.current_scenario.mission, mission_outcome
            )
            reward += weights.mission_success * success

        return obs, reward, done, truncated, info
```

### 9.2 Observation Space Additions

```python
"mission_embedding": Box(low=-1, high=1, shape=(14,))  # sin/cos can be negative
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
├── curriculum/
│   ├── __init__.py
│   ├── mission.py        # MissionSpec, MissionVerb, embedding
│   ├── scenario.py       # ScenarioSpec, ScenarioGenerator
│   ├── phase.py          # PhaseMetrics, PhaseTransition, RewardWeights
│   ├── evaluator.py      # Per-verb success evaluation
│   └── manager.py        # CurriculumManager orchestration
├── env/
│   └── env.py            # Modified with curriculum integration
└── traits/               # Future: personality modifiers
    └── __init__.py
```

## 11. Implementation Order

1. **mission.py** - Core data structures and embedding
2. **phase.py** - Reward weight interpolation
3. **scenario.py** - Scenario generation with levers
4. **evaluator.py** - Success criteria per verb
5. **manager.py** - Orchestration
6. **env.py integration** - Wire curriculum into environment
7. **Training script updates** - Checkpoint curriculum state

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

### Directive B: NavGraph Coverage Lookup

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
