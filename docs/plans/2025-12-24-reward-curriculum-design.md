# Reward Curriculum with Automatic Phase Transitions

**Date:** 2025-12-24
**Status:** Design Complete - Ready for Review
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

### 1.3 Spatial Context (3 values)

```python
spatial_context = [
    objective_distance,      # Normalized distance to objective
    objective_bearing,       # Angle to objective (-1 to 1)
    terrain_complexity       # LOS complexity of AO
]
```

### 1.4 Full Embedding (13 dimensions)

```python
@dataclass
class MissionSpec:
    verb: MissionVerb                    # Enum of 6 types
    risk: float                          # 0.0 - 1.0
    loss_appetite: float                 # 0.0 - 1.0
    time_pressure: float                 # 0.0 - 1.0
    grouping: float                      # 0.0 - 1.0
    objective_pos: tuple[float, float, float]

    def to_embedding(self, agent_pos: np.ndarray) -> np.ndarray:
        """13-dimensional mission embedding for observation."""
        verb_onehot = np.zeros(6)
        verb_onehot[self.verb.value] = 1.0

        params = np.array([
            self.risk,
            self.loss_appetite,
            self.time_pressure,
            self.grouping
        ])

        # Relative spatial context
        delta = np.array(self.objective_pos) - agent_pos
        distance = np.linalg.norm(delta) / 200.0  # Normalize by max range
        bearing = np.arctan2(delta[1], delta[0]) / np.pi  # -1 to 1

        spatial = np.array([distance, bearing, self.terrain_complexity])

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

    # Shaping fades from 1.0 to 0.0 over first 66% of training
    shaping_weight = max(0.0, 1.0 - progress * 1.5)

    # Objective ramps up then slightly down
    objective_weight = min(1.0, progress * 2.0) * max(0.3, 1.0 - progress)

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
MISSION_EVALUATORS = {
    MissionVerb.SCOUT: lambda m, o: (
        o.enemies_detected / max(1, o.total_enemies) * 0.6 +
        o.terrain_covered / o.total_terrain * 0.4
    ),
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

Added to action space as continuous output [0, 1]. Agent learns honest reporting because:
- Lying "GREEN" when pinned → Commander sends no help → death
- Honest "RED" → Commander adapts → survival

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

        # Add mission embedding to observation
        obs["mission_embedding"] = self.current_scenario.mission.to_embedding()
        return obs, {}

    def step(self, actions):
        weights = self.curriculum.get_reward_weights()
        reward = self._compute_reward(weights)

        if done:
            mission_success = self.curriculum.compute_terminal_reward(...)
            reward += weights.terminal * mission_success

        return obs, reward, done, truncated, info
```

### 9.2 Observation Space Additions

```python
"mission_embedding": Box(low=0, high=1, shape=(13,))
"status_report": Box(low=0, high=1, shape=(1,))
```

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

## 12. Future Extensions

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
