# Combat Suite System Design

## Variable-Length Observation Architecture for Class-Differentiated Tactical Awareness

**Date:** 2025-12-25
**Status:** DRAFT - Design Exploration
**Context:** Response to strategic audit of mission embeddings; evolved into class-differentiated observation system

---

## Executive Summary

### The Problem

100v100 mech combat creates a 2,500-dimensional observation space per agent (100 entities × 25 features). This is unlearnable, and worse, it grants every agent perfect battlefield awareness—destroying any possibility of tactical gameplay.

### The Solution

**Combat Suite as Loadout**: Each mech class has a different sensor/display package that determines what information is available. Scouts see 20 contacts, heavies see 5. The cockpit is a game mechanic, not a rendering artifact.

**Key Components**:
- **Display Manager Boundary**: Policy sees only what the cockpit shows, never the full track store
- **Set Encoders (DeepSets)**: Compress variable entity counts to fixed-size representations
- **Hierarchy as Compression**: Each command level sees summaries, not raw data
- **Order Broadcast**: Shared awareness through explicit communication, not omniscience

### Architecture

```
World State → Sensors → Track Store → Display Manager → Policy Input
                                            ↑
                                    (Suite determines K slots,
                                     alert injection, panel layout)
```

### Key Ablations

| Ablation | Tests |
|----------|-------|
| Oracle vs Suite-Bounded | Does cockpit constraint enable tactical behavior? |
| Remove Reports | Does hierarchy require communication? |
| Remove Stickiness | Does stable display improve training? |
| Suite Swap (Same Stats) | Is behavior driven by perception, not weapons? |

---

## 1. Design Philosophy

### 1.1 Core Insight

Traditional RL observation spaces use fixed-size vectors, forcing all agents to perceive the battlefield identically. This is both computationally wasteful (padding) and tactically unrealistic (a scout and a heavy have different jobs, different cockpits, different information needs).

**Key Realization:** Set encoders (DeepSets, Transformers) compress variable-length entity lists into fixed-size representations. This means:
- The **environment** can provide class-appropriate observation streams
- The **model** compresses them to constant size regardless of input count
- Different mechs can have **fundamentally different perceptions** while sharing policy weights

### 1.2 The Cockpit Metaphor

Each mech class has a different cockpit built for its role:

| Class | Cockpit Optimized For | Display Priority |
|-------|----------------------|------------------|
| **Scout** | Detection, reporting | Contacts, contacts, contacts |
| **Light** | Mobility, opportunism | Threats + escape routes |
| **Medium** | Flexible engagement | Balanced tactical picture |
| **Heavy** | Sustained firepower | Current target, threat warnings |
| **Command** | Coordination | Squad status, order management |

The pilot doesn't choose what their cockpit shows - the **combat suite** determines available information streams.

### 1.3 Common Operating Picture (COP)

All units in a squad share awareness through broadcast channels:
- **Order Channel**: Commander's assignments, visible to all
- **Contact Channel**: Shared sensor fusion (painted targets, confirmed kills)
- **Status Channel**: Squad health/position (varies by suite)

Each unit sees the same broadcast data, but their suite determines:
- How much bandwidth they have for it
- Where it appears in their priority stack
- How much detail they receive

**COP Access Rules:**

| Condition | Effect on COP |
|-----------|---------------|
| **Comms healthy** | Full access per suite spec |
| **Comms degraded** | Increased staleness, reduced detail |
| **Comms severed** | Own sensors only, no datalink updates |
| **Jamming active** | Track confidence drops, classification uncertain |
| **Chain of command** | Orders from unauthorized sources ignored |

The COP is **not** omniscience—it's the result of explicit communication. If scouts don't report, command doesn't know. If comms fail, the squad fragments. This is a core gameplay mechanic.

---

## 2. Combat Suite Specifications

### 2.1 Suite Definitions

```python
from dataclasses import dataclass
from enum import Enum, auto

class SuiteType(Enum):
    SCOUT_RECON = auto()
    LIGHT_SKIRMISH = auto()
    MEDIUM_ASSAULT = auto()
    HEAVY_FIRE_SUPPORT = auto()
    TACTICAL_COMMAND = auto()

@dataclass(frozen=True)
class CombatSuiteSpec:
    """Defines a mech's sensor/display package."""

    suite_type: SuiteType

    # === CONTACT TRACKING ===
    visual_contact_slots: int       # Own sensor contacts
    sensor_range_mult: float        # Detection range modifier
    sensor_fidelity: float          # Detail level on contacts (0-1)

    # === SQUAD AWARENESS ===
    squad_position_slots: int       # How many squadmates shown
    squad_detail_level: int         # 0=icon, 1=status, 2=full
    sees_squad_engaged: bool        # See what squadmates are fighting

    # === COMMAND INTEGRATION ===
    receives_orders: bool           # Gets order broadcast
    issues_orders: bool             # Can send orders (command only)
    order_detail_level: int         # 0=own only, 1=squad orders, 2=full COP

    # === SHARED INTEL ===
    receives_painted_targets: bool  # See pack's paint locks
    receives_contact_reports: bool  # See scout's reported contacts
    intel_slots: int                # How many shared contacts shown

    # === DERIVED CONTINUOUS KNOBS ===
    # These are computed from the discrete settings above.
    # Useful for: damage degradation, EW effects, experimentation.

    @property
    def bandwidth(self) -> float:
        """Total information throughput capacity. 0.0-1.0."""
        # Higher = more slots, more detail, more channels
        slot_contribution = (self.visual_contact_slots + self.intel_slots) / 30.0
        detail_contribution = self.squad_detail_level / 2.0
        return min(1.0, (slot_contribution + detail_contribution) / 2.0)

    @property
    def outward_bias(self) -> float:
        """Focus on external threats vs internal coordination. 0.0-1.0."""
        # High = scout (many contacts, few squad details)
        # Low = command (few contacts, full squad awareness)
        contact_weight = self.visual_contact_slots / 20.0
        squad_weight = 1.0 - (self.squad_detail_level / 2.0)
        return (contact_weight + squad_weight) / 2.0

    @property
    def coordination_level(self) -> int:
        """C2 integration depth. 0=loner, 1=team player, 2=coordinator."""
        if self.issues_orders:
            return 2
        elif self.order_detail_level >= 1 and self.sees_squad_engaged:
            return 1
        return 0


# === SUITE LIBRARY ===

COMBAT_SUITES: dict[SuiteType, CombatSuiteSpec] = {

    SuiteType.SCOUT_RECON: CombatSuiteSpec(
        suite_type=SuiteType.SCOUT_RECON,
        # Scouts see EVERYTHING in range - their job is finding
        visual_contact_slots=20,
        sensor_range_mult=1.5,
        sensor_fidelity=0.8,        # Good detail for reporting
        # Minimal squad awareness - focused outward
        squad_position_slots=4,      # Know where friendlies are (don't shoot them)
        squad_detail_level=0,        # Icons only
        sees_squad_engaged=False,    # Not their concern
        # Orders: receive only, high priority
        receives_orders=True,
        issues_orders=False,
        order_detail_level=1,        # See squad orders for deconfliction
        # Intel: they CREATE intel, less need to receive
        receives_painted_targets=True,
        receives_contact_reports=False,  # They ARE the contact reports
        intel_slots=5,
    ),

    SuiteType.LIGHT_SKIRMISH: CombatSuiteSpec(
        suite_type=SuiteType.LIGHT_SKIRMISH,
        # Good awareness for mobility/flanking
        visual_contact_slots=10,
        sensor_range_mult=1.2,
        sensor_fidelity=0.6,
        # Need to know where squad is for coordination
        squad_position_slots=9,      # Full squad minus self
        squad_detail_level=1,        # Status level
        sees_squad_engaged=False,
        # Orders: receive, see squad picture
        receives_orders=True,
        issues_orders=False,
        order_detail_level=1,
        # Intel: receives shared targeting
        receives_painted_targets=True,
        receives_contact_reports=True,
        intel_slots=8,
    ),

    SuiteType.MEDIUM_ASSAULT: CombatSuiteSpec(
        suite_type=SuiteType.MEDIUM_ASSAULT,
        # Moderate awareness - balanced
        visual_contact_slots=6,
        sensor_range_mult=1.0,
        sensor_fidelity=0.7,
        # Squad awareness for formation
        squad_position_slots=9,
        squad_detail_level=1,
        sees_squad_engaged=False,
        # Orders
        receives_orders=True,
        issues_orders=False,
        order_detail_level=1,
        # Intel
        receives_painted_targets=True,
        receives_contact_reports=True,
        intel_slots=6,
    ),

    SuiteType.HEAVY_FIRE_SUPPORT: CombatSuiteSpec(
        suite_type=SuiteType.HEAVY_FIRE_SUPPORT,
        # Narrow focus - kill what's in front of you
        visual_contact_slots=5,
        sensor_range_mult=0.8,       # Shorter range, relies on squad
        sensor_fidelity=1.0,         # But VERY detailed on what it sees
        # Minimal squad awareness
        squad_position_slots=4,
        squad_detail_level=0,        # Icons - don't shoot friendlies
        sees_squad_engaged=False,
        # Orders: just tell me who to kill
        receives_orders=True,
        issues_orders=False,
        order_detail_level=0,        # Own orders only
        # Intel: receives targeting data
        receives_painted_targets=True,
        receives_contact_reports=True,
        intel_slots=3,               # Few slots, but high priority targets
    ),

    SuiteType.TACTICAL_COMMAND: CombatSuiteSpec(
        suite_type=SuiteType.TACTICAL_COMMAND,
        # Moderate personal awareness
        visual_contact_slots=5,
        sensor_range_mult=1.0,
        sensor_fidelity=0.6,
        # FULL squad awareness - this is the job
        squad_position_slots=9,
        squad_detail_level=2,        # Full detail
        sees_squad_engaged=True,     # See what everyone is fighting
        # Orders: issues AND sees everything
        receives_orders=True,        # Higher echelon orders
        issues_orders=True,          # Can assign targets
        order_detail_level=2,        # Full COP
        # Intel: full picture
        receives_painted_targets=True,
        receives_contact_reports=True,
        intel_slots=10,
    ),
}
```

### 2.2 Suite Assignment

```python
# Default suite by mech class
DEFAULT_SUITES: dict[str, SuiteType] = {
    "scout": SuiteType.SCOUT_RECON,
    "light": SuiteType.LIGHT_SKIRMISH,
    "medium": SuiteType.MEDIUM_ASSAULT,
    "heavy": SuiteType.HEAVY_FIRE_SUPPORT,
}

# Squad leader override (one per squad)
def assign_suites(squad: list[MechState], leader_id: str) -> dict[str, SuiteType]:
    assignments = {}
    for mech in squad:
        if mech.mech_id == leader_id:
            assignments[mech.mech_id] = SuiteType.TACTICAL_COMMAND
        else:
            assignments[mech.mech_id] = DEFAULT_SUITES[mech.spec.name]
    return assignments
```

### 2.3 Visual Representation

```
SCOUT (20 slots)
+---------------------------------------------+
| RECON DISPLAY                               |
| =========================================== |
| CONTACTS: 17/20 tracked                     |
| [#### ### ## # ### ## # ### ...]            |
|                                             |
| SQUAD: <><><><><> (icons only, no detail)   |
+---------------------------------------------+

HEAVY (5 slots)
+---------------------------------------------+
| FIRE CONTROL                                |
| =========================================== |
| CONTACTS: 3/5                               |
| 1. HEAVY   45m  ========-- TARGET LOCK      |
| 2. MEDIUM  62m  ==========                  |
| 3. LIGHT   78m  ====------ PAINTED          |
|                                             |
| SQUAD: <><><><><>                           |
+---------------------------------------------+

COMMAND (special)
+---------------------------------------------+
| TACTICAL COMMAND                            |
| =========================================== |
| SQUAD STATUS:          ENGAGED:             |
| <> Scout  100% SE 45m   -> Enemy Heavy      |
| <> Light  80%  E  30m   -> Enemy Medium     |
| <> Med 1  60%  NE 25m   -> (same)           |
| <> Med 2  100% N  40m   (overwatching)      |
| <> YOU ARE HERE                             |
|                                             |
| OWN CONTACTS: 2/5                           |
| 1. MEDIUM  50m (engaged by Med 1)           |
| 2. SCOUT   120m (new contact)               |
+---------------------------------------------+
```

---

## 3. Observation Streams

### 3.1 Stream Architecture

Each mech's observation is composed of multiple **streams**, each processed by a dedicated encoder:

```
+------------------------------------------------------------------+
|                        OBSERVATION                                |
+------------------------------------------------------------------+
|  STREAM A: Self (fixed)                                          |
|  +- Proprioception: velocity, orientation, health, heat...       |
|  +- 47 dims -> SelfEncoder -> 64 dims                            |
+------------------------------------------------------------------+
|  STREAM B: Visual Contacts (variable by suite)                   |
|  +- Scout: up to 20 entities                                     |
|  +- Light: up to 10 entities                                     |
|  +- Medium: up to 6 entities                                     |
|  +- Heavy: up to 5 entities                                      |
|  +- N x 25 dims -> ContactEncoder -> 64 dims                     |
+------------------------------------------------------------------+
|  STREAM C: Squad Status (variable by suite)                      |
|  +- Position + status for 0-9 squadmates                         |
|  +- Detail level varies: icon(4d) / status(12d) / full(20d)      |
|  +- N x D dims -> SquadEncoder -> 48 dims                        |
+------------------------------------------------------------------+
|  STREAM D: Orders (variable, priority-sorted)                    |
|  +- Own order FIRST (if assigned)                                |
|  +- Squad orders for COP (if suite allows)                       |
|  +- N x 16 dims -> OrderEncoder -> 32 dims                       |
+------------------------------------------------------------------+
|  STREAM E: Shared Intel (variable by suite)                      |
|  +- Painted targets, reported contacts                           |
|  +- N x 14 dims -> IntelEncoder -> 32 dims                       |
+------------------------------------------------------------------+
|  STREAM F: Panel Stats (fixed)                                   |
|  +- 8 dims -> Linear -> 16 dims                                  |
|  +- Counts + threat mass (compensates for mean pooling)          |
+------------------------------------------------------------------+
|  STREAM G: Mission (fixed)                                       |
|  +- 15 dims -> Linear -> 16 dims                                 |
+------------------------------------------------------------------+
|  STREAM H: Local Map (fixed)                                     |
|  +- 128 dims -> MapEncoder -> 48 dims                            |
+------------------------------------------------------------------+
|                                                                   |
|  TOTAL TO LSTM: 64+64+48+32+32+16+16+48 = 320 dims (constant)    |
|                                                                   |
+------------------------------------------------------------------+
```

### 3.2 Stream Details

#### Stream A: Self (Fixed, 55 dims)

Proprioceptive data plus suite descriptor:

```python
self_features = [
    # Acoustic threat warning (4)
    acoustic_quadrants[0:4],

    # Hull type (4) - one-hot
    hull_type_onehot[0:4],

    # Status flags (6)
    targeted,           # Someone has LOS to me
    under_fire,         # Recently took damage
    painted,            # Enemy has lock
    shutdown,           # Systems offline
    crit_heat,          # Overheating

    # Resources (6)
    hp_norm,            # Current HP / Max HP
    heat_norm,          # Current heat / Heat capacity
    heat_headroom,      # Room before overheat
    stability_risk,     # Knockdown risk
    sensor_quality,     # Current sensor effectiveness
    jam_level,          # ECM interference

    # State flags (4)
    ecm_on,
    eccm_on,
    suppressed_norm,
    ams_cooldown,

    # Kinematics (3)
    velocity_local[0:3],

    # Weapon cooldowns (4)
    laser_cd, missile_cd, kinetic_cd, painter_cd,

    # Objective (8)
    in_zone,
    zone_relative[0:3],
    zone_radius_norm,
    my_zone_control,
    my_score_norm,
    enemy_score_norm,

    # Time (1)
    time_fraction,

    # Current obs settings (4) - what filter/sort is active
    sort_mode_onehot[0:3],
    hostile_filter_on,

    # === SUITE DESCRIPTOR (8) ===
    # Tells policy what kind of mech it is - critical for shared weights
    suite_type_onehot[0:5],  # (5) one-hot: scout/light/medium/heavy/command
    squad_detail_level_norm, # (1) 0=icon, 0.5=status, 1=full
    issues_orders,           # (1) 1.0 if command mech
    contact_capacity_norm,   # (1) visual_contact_slots / 20.0
]
```

**Note on Suite Descriptor**: This 8-dim vector is mandatory for shared policy weights to work correctly. Without it, the policy must infer role from indirect cues (observation richness, which slots are filled). With it, the network can learn role-specific behavior conditioning.

#### Stream B: Visual Contacts (Variable, 5-20 entities x 25 dims)

Entities the mech can directly sense. **Count varies by suite.**

```python
@dataclass
class ContactFeatures:
    """Per-entity feature vector for visual contacts. 25 dims."""

    # Relative geometry (6)
    relative_pos: np.ndarray      # (3,) normalized by world size
    relative_vel: np.ndarray      # (3,) normalized

    # Orientation (2)
    yaw_sin: float                # sin(their_yaw)
    yaw_cos: float                # cos(their_yaw)

    # Status (3)
    hp_norm: float                # Their HP %
    heat_norm: float              # Their heat %
    stability_norm: float         # Their stability %

    # State (2)
    is_fallen: float              # Knocked down?
    is_legged: float              # Leg destroyed?

    # Identity (7)
    relation: np.ndarray          # (3,) one-hot: friendly/hostile/neutral
    mech_class: np.ndarray        # (4,) one-hot: scout/light/medium/heavy

    # Targeting (3)
    is_painted: float             # Locked by my pack
    is_my_target: float           # Assigned to me by command
    is_targeting_me: float        # This entity is aiming at me

    # Ballistics (2)
    lead_pitch: float             # Suggested pitch for ballistic hit
    closing_rate: float           # Approaching or separating
```

#### Stream C: Squad Status (Variable, 0-9 entities x 4-20 dims)

Information about squadmates. **Detail level varies by suite.**

```python
# Detail Level 0: Icons (4 dims per squadmate)
squad_icon = [
    relative_bearing_sin,    # Direction to them
    relative_bearing_cos,
    distance_norm,           # How far
    alive,                   # Still active
]

# Detail Level 1: Status (12 dims per squadmate)
squad_status = [
    *squad_icon,             # (4)
    hp_norm,                 # Their health
    heat_norm,               # Their heat
    in_combat,               # Currently engaged
    suppressed,              # Pinned down
    velocity_norm,           # Moving or static
    relative_heading_sin,    # Which way facing
    relative_heading_cos,
    mech_class_compact,      # 0-3 normalized
]

# Detail Level 2: Full (20 dims per squadmate) - Command only
squad_full = [
    *squad_status,           # (12)
    weapon_ready,            # Primary off cooldown
    target_bearing_sin,      # Where they're aiming
    target_bearing_cos,
    target_range_norm,       # How far their target
    ammo_norm,               # Resource status
    stability_norm,          # Knockdown risk
    current_order_verb,      # What they're tasked (0-6 normalized)
    order_progress,          # Completion estimate
]
```

#### Stream D: Orders (Variable, 0-10 orders x 16 dims)

**The Command Broadcast Channel.** All active orders visible to those with clearance.

```python
@dataclass
class OrderFeatures:
    """Per-order feature vector. 16 dims."""

    # Priority (1)
    is_my_order: float            # 1.0 if assigned to me, 0.0 otherwise

    # Order type (7) - one-hot verb
    verb: np.ndarray              # Assault/Hold/Overwatch/Flank/Suppress/Scout/Stage

    # Target/Objective (4)
    target_bearing_sin: float     # Direction to objective
    target_bearing_cos: float
    target_distance_norm: float   # How far
    target_type: float            # 0=position, 0.5=area, 1=entity

    # Assigned unit (2)
    assignee_bearing_sin: float   # Where is the assigned unit
    assignee_bearing_cos: float

    # Timing (2)
    time_issued_norm: float       # How old is this order
    urgency: float                # From mission params
```

**Order Sorting:** Own order always comes first, then by relevance:

```python
def sort_orders(orders: list[Order], viewer_id: str) -> list[Order]:
    def priority(o: Order) -> tuple:
        return (
            0 if o.assignee == viewer_id else 1,   # Own order first
            -o.urgency,                              # Then by urgency
            o.time_issued,                           # Then by recency
        )
    return sorted(orders, key=priority)
```

**Note on Permutation Invariance:**

The order stream encoder (DeepSet) is permutation-invariant, so sorting only matters for:
1. **Truncation**: When there are more orders than `MAX_ORDERS`, sorting ensures important orders aren't dropped
2. **Display**: The COP shows "mine first" for the human-readable tactical picture

The encoder does **not** use positional semantics. The `is_my_order` feature (1.0 vs 0.0) tells the network which order is assigned to this unit. The policy learns to pay attention to orders with `is_my_order=1.0`, not "slot 0".

If positional meaning is ever needed, add a `slot_index_norm` feature or use a separate fixed slot for "my current order".

#### Stream E: Shared Intel (Variable, 0-10 contacts x 14 dims)

Contacts reported by squadmates or command. Lower fidelity than visual.

```python
@dataclass
class IntelFeatures:
    """Per-intel-contact feature vector. 14 dims."""

    # Source (3)
    source_type: np.ndarray       # (3,) one-hot: scout_report/paint_lock/command_designate

    # Location (4)
    bearing_sin: float
    bearing_cos: float
    distance_norm: float
    altitude_relative: float

    # Identity (5)
    mech_class: np.ndarray        # (4,) one-hot, may be uncertain
    confidence: float             # How reliable is this intel

    # Age (1)
    staleness: float              # How old (0=fresh, 1=stale)

    # Priority (1)
    is_priority_target: float     # Command designated
```

#### Stream F: Mission (Fixed, 15 dims)

Unchanged from curriculum design. Current mission orders:

```python
mission_embedding = [
    verb_onehot[0:7],           # 7 mission types including Stage
    risk,                        # Expected enemy strength
    loss_appetite,               # Acceptable casualties
    time_pressure,               # Urgency
    grouping,                    # Formation tightness
    objective_distance_norm,     # How far to goal
    objective_bearing_sin,       # Direction to goal
    objective_bearing_cos,
    terrain_complexity,          # AO difficulty
]
```

#### Stream G: Local Map (Fixed, 128 dims)

Ego-centric occupancy/terrain. Unchanged from current.

---

## 4. Entity Feature Encoding Standards

### 4.1 Normalization Conventions

All features normalized to ranges compatible with neural networks:

| Type | Range | Notes |
|------|-------|-------|
| Ratios (HP, heat, etc.) | [0, 1] | Clipped if exceeds max |
| Angles (bearing, yaw) | sin/cos pairs | No discontinuity |
| Distances | [0, 1] | Divided by max relevant range |
| Velocities | [-1, 1] | Divided by max speed |
| Booleans | {0, 1} | Float for differentiability |
| Categories | One-hot | Fixed dimension per category |

### 4.2 Relative vs Absolute

**All spatial data is relative to the viewer:**

```python
def encode_relative_position(viewer: MechState, target: MechState, world: World) -> np.ndarray:
    """Position relative to viewer, normalized."""
    rel = target.pos - viewer.pos
    max_dim = max(world.size_x, world.size_y, world.size_z)
    return rel / max(1e-6, max_dim)

def encode_relative_bearing(viewer: MechState, target: MechState) -> tuple[float, float]:
    """Bearing from viewer to target as sin/cos."""
    delta = target.pos - viewer.pos
    theta = np.arctan2(delta[1], delta[0]) - viewer.yaw  # Relative to viewer's heading
    return np.sin(theta), np.cos(theta)
```

---

## 5. Model Architecture

### 5.1 Set Encoders

Each variable-length stream uses a DeepSet encoder:

```python
class DeepSetEncoder(nn.Module):
    """Permutation-invariant set encoder with masking."""

    def __init__(
        self,
        entity_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 64,
        aggregation: str = "sum",  # "sum", "max", or "mean"
    ):
        super().__init__()
        self.aggregation = aggregation

        # Per-entity transform (applied independently)
        self.phi = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Post-aggregation transform
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, entities: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            entities: (batch, max_entities, entity_dim)
            mask: (batch, max_entities) - 1.0 for padding, 0.0 for real

        Returns:
            (batch, output_dim)
        """
        # Encode each entity
        h = self.phi(entities)  # (batch, max_entities, hidden_dim)

        # Zero out padding
        valid_mask = (1.0 - mask).unsqueeze(-1)  # (batch, max_entities, 1)
        h = h * valid_mask

        # Aggregate
        if self.aggregation == "sum":
            pooled = h.sum(dim=1)
        elif self.aggregation == "max":
            h = h.masked_fill(mask.unsqueeze(-1).bool(), float('-inf'))
            pooled = h.max(dim=1).values
            pooled = pooled.masked_fill(pooled == float('-inf'), 0.0)
        elif self.aggregation == "mean":
            count = valid_mask.sum(dim=1).clamp(min=1.0)
            pooled = h.sum(dim=1) / count

        return self.rho(pooled)
```

### 5.2 Full Model

```python
class CombatSuiteActorCritic(nn.Module):
    """Actor-Critic with variable observation streams."""

    # Maximum entities per stream (for padding/allocation)
    # NOTE: These are GLOBAL caps. Each suite fills a subset based on its spec.
    # e.g., Scout fills 20 of 32 contact slots; Heavy fills 5 of 32.
    MAX_CONTACTS = 32
    MAX_SQUAD = 9
    MAX_ORDERS = 10
    MAX_INTEL = 10

    def __init__(
        self,
        self_dim: int = 55,  # 47 base + 8 suite descriptor
        contact_dim: int = 25,
        squad_dim: int = 20,      # Max (full detail)
        order_dim: int = 16,
        intel_dim: int = 14,
        mission_dim: int = 15,
        map_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # === ENCODERS ===

        # Self: fixed size, simple MLP
        self.self_encoder = nn.Sequential(
            nn.Linear(self_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # Contacts: variable count, set encoder
        # Use MEAN pooling to avoid count-dependent activation magnitudes
        # (Scout with 20 contacts vs Heavy with 5 would have 4x different norms with sum)
        self.contact_encoder = DeepSetEncoder(
            entity_dim=contact_dim,
            hidden_dim=64,
            output_dim=64,
            aggregation="mean",
        )

        # Squad: variable count + variable detail level
        self.squad_encoder = DeepSetEncoder(
            entity_dim=squad_dim,
            hidden_dim=48,
            output_dim=48,
            aggregation="mean",  # Mean for consistent scaling
        )

        # Orders: variable count
        self.order_encoder = DeepSetEncoder(
            entity_dim=order_dim,
            hidden_dim=32,
            output_dim=32,
            aggregation="mean",
        )

        # Intel: variable count
        self.intel_encoder = DeepSetEncoder(
            entity_dim=intel_dim,
            hidden_dim=32,
            output_dim=32,
            aggregation="mean",
        )

        # Panel stats: explicit counts and aggregate threat metrics
        # This compensates for mean pooling losing count information
        # 8 dims: contact_count, intel_count, order_count, squad_count,
        #         threat_mass, friendly_mass, has_alert, detail_level
        self.panel_stats_encoder = nn.Linear(8, 16)

        # Mission: fixed, simple projection
        self.mission_encoder = nn.Linear(mission_dim, 16)

        # Map: fixed, MLP
        self.map_encoder = nn.Sequential(
            nn.Linear(map_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 48),
        )

        # === LSTM CORE ===
        # Combined: 64 + 64 + 48 + 32 + 32 + 16 + 16 + 48 = 320
        # (self + contacts + squad + orders + intel + panel_stats + mission + map)
        combined_dim = 64 + 64 + 48 + 32 + 32 + 16 + 16 + 48

        self.lstm = nn.LSTM(
            input_size=combined_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # === OUTPUT HEADS ===
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_DIM),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        obs: dict[str, Tensor],
        hidden: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        """
        Forward pass with variable-size observations.

        obs keys:
            'self': (batch, 55)  # Includes 8-dim suite descriptor
            'contacts': (batch, max_contacts, 25)
            'contact_mask': (batch, max_contacts)
            'squad': (batch, max_squad, squad_dim)  # dim varies by detail level
            'squad_mask': (batch, max_squad)
            'orders': (batch, max_orders, 16)
            'order_mask': (batch, max_orders)
            'intel': (batch, max_intel, 14)
            'intel_mask': (batch, max_intel)
            'panel_stats': (batch, 8)  # Explicit counts + aggregate metrics
            'mission': (batch, 15)
            'local_map': (batch, 128)
        """
        batch_size = obs['self'].shape[0]

        # Encode each stream
        self_feats = self.self_encoder(obs['self'])
        contact_feats = self.contact_encoder(obs['contacts'], obs['contact_mask'])
        squad_feats = self.squad_encoder(obs['squad'], obs['squad_mask'])
        order_feats = self.order_encoder(obs['orders'], obs['order_mask'])
        intel_feats = self.intel_encoder(obs['intel'], obs['intel_mask'])
        panel_stats_feats = self.panel_stats_encoder(obs['panel_stats'])
        mission_feats = self.mission_encoder(obs['mission'])
        map_feats = self.map_encoder(obs['local_map'])

        # Concatenate to fixed size
        combined = torch.cat([
            self_feats,         # 64
            contact_feats,      # 64
            squad_feats,        # 48
            order_feats,        # 32
            intel_feats,        # 32
            panel_stats_feats,  # 16
            mission_feats,      # 16
            map_feats,          # 48
        ], dim=-1)  # = 320

        # LSTM expects (batch, seq, features)
        combined = combined.unsqueeze(1)

        if hidden is None:
            lstm_out, hidden = self.lstm(combined)
        else:
            lstm_out, hidden = self.lstm(combined, hidden)

        lstm_out = lstm_out.squeeze(1)  # (batch, hidden_dim)

        # Action and value
        action_logits = self.actor(lstm_out)
        value = self.critic(lstm_out)

        return action_logits, value, hidden
```

### 5.3 Handling Variable Detail Levels

Squad observations have variable dimensions (4/12/20) based on suite. Handle with padding:

```python
def _encode_squad(
    self,
    viewer: MechState,
    squad: list[MechState],
    detail_level: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode squad with suite-appropriate detail level."""

    MAX_DIM = 20  # Full detail
    features = np.zeros((self.MAX_SQUAD, MAX_DIM), dtype=np.float32)
    mask = np.ones(self.MAX_SQUAD, dtype=np.float32)

    for i, mate in enumerate(squad[:self.MAX_SQUAD]):
        if mate.mech_id == viewer.mech_id:
            continue  # Don't include self

        if detail_level == 0:
            feat = self._squad_icon_features(viewer, mate)  # 4 dims
        elif detail_level == 1:
            feat = self._squad_status_features(viewer, mate)  # 12 dims
        else:
            feat = self._squad_full_features(viewer, mate)  # 20 dims

        # Pad to max dim
        features[i, :len(feat)] = feat
        mask[i] = 0.0

    return features, mask
```

---

## 6. Order System Detail

### 6.1 Order Structure

```python
from enum import IntEnum

class OrderVerb(IntEnum):
    # Observable verbs (these appear in order observations, 7 dims)
    ASSAULT = 0
    HOLD = 1
    OVERWATCH = 2
    FLANK = 3
    SUPPRESS = 4
    SCOUT = 5
    STAGE = 6
    # Meta-action (not in observation one-hot, just command action)
    CANCEL = 7  # Revoke previous order - removes order from broadcast, not shown

@dataclass
class TacticalOrder:
    """An order from command to a unit."""

    order_id: str                # Unique identifier
    issuer_id: str               # Who gave the order (command mech)
    assignee_id: str             # Who must execute
    verb: OrderVerb              # What to do

    # Target
    target_pos: np.ndarray | None     # Position objective
    target_entity_id: str | None      # Entity objective
    target_zone_radius: float = 0.0   # Area objective

    # Parameters
    urgency: float = 0.5              # 0=when convenient, 1=now

    # Metadata
    time_issued: float = 0.0          # Sim time
    acknowledged: bool = False         # Unit confirmed receipt
    completed: bool = False            # Unit reports done
```

### 6.2 Command Mech Actions

The command mech has additional action dimensions for issuing orders:

```python
# Extended action space for command suite
COMMAND_ACTION_DIMS = {
    'movement': 4,           # Same as everyone
    'weapons': 5,            # Same as everyone
    'systems': 2,            # ECM, ECCM
    'targeting': 5,          # Target selection
    'obs_ctrl': 4,           # Filter/sort

    # COMMAND ONLY:
    'order_issue': 1,        # >0.5 = issue order this frame
    'order_verb': 8,         # Argmax selects verb (7 real + CANCEL)
    'order_target_type': 2,  # Position vs entity
    'order_assignee': 9,     # Which squad member (argmax)
    'order_urgency': 1,      # Continuous urgency
}
# NOTE: CANCEL (verb=7) removes existing order from broadcast.
# Cancelled orders don't appear in observations - verb one-hot stays 7 dims.
```

### 6.3 Order Flow

```
1. Command observes tactical situation
   - Squad status (Stream C with full detail)
   - Engaged contacts (what squad is fighting)
   - Own contacts + intel

2. Command decides to issue order
   - action['order_issue'] > 0.5
   - action['order_verb'] -> FLANK
   - action['order_assignee'] -> Light mech
   - action['order_target_type'] -> entity
   - Target from contact slots

3. Order created and broadcast
   - Added to active order list
   - All squad members receive in Stream D

4. Assignee sees order FIRST in their Stream D
   - 'is_my_order' = 1.0 for first entry
   - Can adjust behavior accordingly

5. Other squad members see order in COP
   - Know what Light is tasked with
   - Can support/deconflict

6. Assignee completes or fails
   - Status report (action space) signals progress
   - Command sees via Stream C (sees_squad_engaged)
```

---

## 7. Implementation in Environment

### 7.1 Observation Building

```python
class EchelonEnv:

    def _build_obs(self, viewer: MechState) -> dict[str, np.ndarray]:
        """Build observation dict based on viewer's combat suite."""

        suite = self._get_suite(viewer)

        # === STREAM A: Self (fixed) ===
        self_feats = self._build_self_features(viewer)

        # === STREAM B: Visual Contacts (variable by suite) ===
        visible = self._get_visible_entities(viewer, suite.sensor_range_mult)
        visible = self._apply_contact_filters(visible, viewer)
        visible = visible[:suite.visual_contact_slots]

        contacts = np.zeros((self.MAX_CONTACTS, 25), dtype=np.float32)
        contact_mask = np.ones(self.MAX_CONTACTS, dtype=np.float32)
        for i, entity in enumerate(visible):
            contacts[i] = self._encode_contact(viewer, entity, suite.sensor_fidelity)
            contact_mask[i] = 0.0

        # === STREAM C: Squad Status (variable by suite) ===
        squad, squad_mask = self._encode_squad(
            viewer,
            self._get_squad(viewer),
            suite.squad_detail_level,
        )

        # Add engaged contacts if command suite
        if suite.sees_squad_engaged:
            engaged = self._get_squad_engaged_contacts(viewer)
            # Append to squad stream or separate stream

        # === STREAM D: Orders (variable) ===
        orders, order_mask = self._encode_orders(
            viewer,
            self._get_active_orders(),
            suite.order_detail_level,
        )

        # === STREAM E: Shared Intel (variable by suite) ===
        intel_list = []
        if suite.receives_painted_targets:
            intel_list.extend(self._get_painted_targets(viewer))
        if suite.receives_contact_reports:
            intel_list.extend(self._get_contact_reports(viewer))
        intel_list = intel_list[:suite.intel_slots]

        intel = np.zeros((self.MAX_INTEL, 14), dtype=np.float32)
        intel_mask = np.ones(self.MAX_INTEL, dtype=np.float32)
        for i, contact in enumerate(intel_list):
            intel[i] = self._encode_intel(viewer, contact)
            intel_mask[i] = 0.0

        # === STREAM F: Mission (fixed) ===
        mission = self._get_mission_embedding(viewer)

        # === STREAM G: Local Map (fixed) ===
        local_map = self._build_local_map(viewer)

        return {
            'self': self_feats,
            'contacts': contacts,
            'contact_mask': contact_mask,
            'squad': squad,
            'squad_mask': squad_mask,
            'orders': orders,
            'order_mask': order_mask,
            'intel': intel,
            'intel_mask': intel_mask,
            'mission': mission,
            'local_map': local_map,
        }
```

### 7.2 Order Processing

```python
def _process_command_actions(self, commander: MechState, action: np.ndarray) -> None:
    """Process order-issuing actions from command mech."""

    suite = self._get_suite(commander)
    if not suite.issues_orders:
        return

    # Check if issuing order this frame
    if action[ORDER_ISSUE_IDX] <= 0.5:
        return

    # Decode order
    verb = OrderVerb(int(np.argmax(action[ORDER_VERB_START:ORDER_VERB_END])))
    assignee_idx = int(np.argmax(action[ORDER_ASSIGNEE_START:ORDER_ASSIGNEE_END]))
    urgency = float(np.clip(action[ORDER_URGENCY_IDX], 0.0, 1.0))

    # Get assignee
    squad = self._get_squad(commander)
    if assignee_idx >= len(squad):
        return  # Invalid assignment
    assignee = squad[assignee_idx]

    # Get target from commander's current target selection
    target_type = "position" if action[ORDER_TARGET_TYPE_IDX] < 0.5 else "entity"
    if target_type == "entity":
        target_entity = self._get_selected_target(commander)
        target_pos = target_entity.pos if target_entity else None
    else:
        # Use objective position from mission or contact location
        target_pos = self._get_target_position(commander, action)

    # Create and broadcast order
    order = TacticalOrder(
        order_id=f"ord_{self.step_count}_{commander.mech_id}",
        issuer_id=commander.mech_id,
        assignee_id=assignee.mech_id,
        verb=verb,
        target_pos=target_pos,
        target_entity_id=target_entity.mech_id if target_type == "entity" else None,
        urgency=urgency,
        time_issued=self._sim_time,
    )

    self._active_orders.append(order)

    # Cancel conflicting orders to same assignee
    self._cancel_orders_for(assignee.mech_id, except_id=order.order_id)
```

---

## 8. What Else Can We Use This Technique For?

The variable-length set encoder pattern unlocks many possibilities beyond contact tracking. Any time you have:
- **Variable count** of similar items
- **Need fixed-size output** for downstream processing
- **Permutation shouldn't matter** (or order can be encoded as a feature)

### 8.1 Weapon Loadout Encoding

Mechs could have variable weapon loadouts instead of fixed slots:

```python
@dataclass
class WeaponFeatures:
    """Per-weapon feature vector. 12 dims."""
    weapon_class: np.ndarray      # (4,) one-hot: energy/ballistic/missile/support
    damage_norm: float            # Relative damage output
    range_norm: float             # Effective range
    cooldown_current: float       # Time until ready
    ammo_remaining: float         # 1.0 for energy weapons
    heat_per_shot: float          # Thermal cost
    spread: float                 # Accuracy

# Different mechs have different weapon counts
# Heavy: 6 weapons -> encode -> 32 dims
# Scout: 2 weapons -> encode -> 32 dims (same output!)
```

### 8.2 Damage State / Component Status

Instead of single HP value, detailed component tracking:

```python
@dataclass
class ComponentFeatures:
    """Per-component status. 8 dims."""
    location: np.ndarray          # (6,) one-hot: head/torso/arms/legs
    hp_fraction: float            # Current / max
    is_critical: bool             # Contains critical systems
    is_destroyed: bool            # No longer functional

# Mechs have 6-12 components
# Set encoder compresses to fixed "structural integrity" vector
```

### 8.3 Terrain Features Around Agent

Variable number of terrain features in sensor range:

```python
@dataclass
class TerrainFeatures:
    """Per-terrain-feature. 10 dims."""
    feature_type: np.ndarray      # (5,) cover/elevation/hazard/objective/building
    relative_pos: np.ndarray      # (3,)
    size: float                   # How big
    traversable: float            # Can walk through

# Dense urban: 20 features nearby
# Open field: 2 features nearby
# Same policy handles both
```

### 8.4 Historical Events / Memory

Recent events as a variable-length sequence:

```python
@dataclass
class EventMemory:
    """Recent tactical events. 8 dims."""
    event_type: np.ndarray        # (4,) took_damage/dealt_damage/spotted/lost_contact
    time_ago: float               # Recency
    location_relative: np.ndarray # (3,) where it happened

# Last 20 events -> encode -> "recent history" vector
# LSTM still handles temporal, but events are pre-summarized
```

### 8.5 Communication Messages

Variable-length message queue from squadmates:

```python
@dataclass
class CommMessage:
    """Incoming communication. 10 dims."""
    sender_relation: np.ndarray   # (3,) squad/command/allied
    message_type: np.ndarray      # (4,) contact_report/request_support/status/order
    urgency: float
    location_relative: np.ndarray # (3,) referenced location

# 0-5 pending messages -> encode -> "comms summary" vector
```

### 8.6 Multi-Objective Tracking

Multiple simultaneous objectives with priorities:

```python
@dataclass
class ObjectiveFeatures:
    """Per-objective. 12 dims."""
    objective_type: np.ndarray    # (4,) capture/destroy/defend/reach
    priority: float               # 0-1
    progress: float               # Completion
    distance: float
    bearing: np.ndarray           # (2,) sin/cos
    time_remaining: float         # Deadline pressure
    assigned_to_me: float         # Am I responsible

# Some missions have 1 objective, some have 5
# Policy handles both with same architecture
```

### 8.7 Inventory / Resources

Consumable items, deployables, special equipment:

```python
@dataclass
class InventoryItem:
    """Per-item. 8 dims."""
    item_type: np.ndarray         # (6,) ammo/repair/sensor/deployable/buff/special
    quantity: float               # How many
    ready: float                  # Cooldown status
    effect_magnitude: float       # How powerful

# Light mech: 2 items (ammo + sensor beacon)
# Support mech: 8 items (full utility loadout)
```

### 8.8 Squad Composition Embedding

For commander: encode entire force structure as set:

```python
@dataclass
class UnitSummary:
    """Per-subordinate-unit. 15 dims."""
    unit_type: np.ndarray         # (4,) scout/light/medium/heavy
    health_category: float        # 0=critical, 0.5=damaged, 1=fresh
    ammo_category: float          # Resource level
    current_task: np.ndarray      # (7,) verb one-hot
    task_progress: float
    threat_level: float           # Currently engaged intensity
    distance_to_objective: float

# Squad of 10 -> encode -> "force readiness" vector (64 dims)
# Company of 100 -> same architecture, bigger input
```

### 8.9 The General Pattern

```python
class VariableStreamObservation:
    """Generic pattern for variable-length observation streams."""

    def __init__(self, entity_dim: int, max_entities: int, output_dim: int):
        self.encoder = DeepSetEncoder(entity_dim, output_dim)
        self.max_entities = max_entities
        self.entity_dim = entity_dim

    def encode(self, entities: list[Any], encode_fn: Callable) -> tuple[Tensor, Tensor]:
        """Convert variable list to padded tensor + mask."""
        features = torch.zeros(self.max_entities, self.entity_dim)
        mask = torch.ones(self.max_entities)

        for i, entity in enumerate(entities[:self.max_entities]):
            features[i] = encode_fn(entity)
            mask[i] = 0.0

        return self.encoder(features.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)
```

**Anywhere you currently have:**
- Fixed slots with padding
- Concatenated features that scale with count
- Truncation losing information

**You can replace with:**
- Variable input -> Set Encoder -> Fixed output
- Suite/role determines how much input is available
- Same model handles all variations

---

## 9. Summary

### 9.1 Key Insights

1. **Combat Suite as Loadout**: Observation structure is a mech characteristic, not a global constant

2. **Set Encoders Enable Variable Perception**: DeepSets compress any number of entities to fixed size

3. **Hierarchical Information Flow**: Scouts gather -> Command synthesizes -> Heavies execute

4. **Order Broadcast Creates COP**: All units share awareness, own orders prioritized

5. **Constant LSTM Input**: Regardless of suite, policy sees fixed 304 dims

### 9.2 Benefits

| Benefit | Mechanism |
|---------|-----------|
| Realistic class roles | Different suites = different perceptions |
| Efficient representation | 20 contacts -> 64 dims, not 500 |
| Scalable coordination | Command channel handles N units |
| Shared policy weights | Same model, different obs generation |
| Emergent specialization | Roles arise from information access |

### 9.3 Implementation Priority

1. **Phase 1**: DeepSet encoder for contacts (replace fixed slots)
2. **Phase 2**: Suite-variable contact counts
3. **Phase 3**: Order broadcast channel
4. **Phase 4**: Full squad awareness streams
5. **Phase 5**: Intel sharing channel

---

## 10. Refined Perception Pipeline

*Additions from external review consolidating the mental model.*

### 10.1 Three-Stage Model

The clean separation that makes this design work:

```
Stage 1: SENSORS         Stage 2: DISPLAY MANAGER      Stage 3: PILOT AGENT
(what can be known)      (what gets presented)         (what to look at)
─────────────────────────────────────────────────────────────────────────────
• Fog of war            • Slots, pages, panels        • Filter actions
• LOS checks            • Alert injection             • Sort actions
• Sensor range          • Priority sorting            • Page actions
• Noise/uncertainty     • Squad picture               • Target lock
• Track persistence     • Datalink fusion             • Request intel
```

**Key insight**: "The mech can know more than it shows" - Stage 1 produces tracks, Stage 2 selects what fills the display, Stage 3 is policy control over Stage 2.

### 10.2 Track Model with Uncertainty

Sensor fusion (Stage 1) produces **tracks**, not perfect entity states:

```python
@dataclass
class SensorTrack:
    """A detected entity with uncertainty."""

    track_id: str                    # Persistent ID for this track

    # Position estimate (relative to sensor platform)
    position_estimate: np.ndarray    # (3,) mean position
    position_uncertainty: float      # Confidence radius
    velocity_estimate: np.ndarray    # (3,) estimated velocity

    # Classification (probabilistic)
    class_probabilities: np.ndarray  # (4,) scout/light/medium/heavy
    class_confidence: float          # How sure are we

    # Track quality
    confidence: float                # Overall track quality 0-1
    time_since_update: float         # Seconds since last sensor hit
    is_stale: bool                   # Exceeded persistence threshold

    # Source attribution
    source_visual: bool              # Direct LOS observation
    source_radar: bool               # Active sensor ping
    source_datalink: bool            # Received from squadmate
    source_inferred: bool            # Predicted from motion model

    # Threat assessment
    is_targeting_me: bool            # Pointing at observer
    threat_priority: float           # Computed threat score
```

This enables:
- Jamming reducing `confidence` and increasing `time_since_update`
- ECCM restoring `class_confidence`
- Decoys injecting false tracks with low confidence
- Track ageing creating urgency to re-acquire

### 10.3 Display Manager Default Policies

Each suite has a **default slot filling policy** that the pilot can override:

```python
DISPLAY_POLICIES = {
    SuiteType.SCOUT_RECON: {
        # Scouts want maximum coverage
        "slots_0_3": "highest_threat_to_self",
        "slots_4_12": "newest_contacts",
        "slots_13_19": "closest_contacts",
    },

    SuiteType.LIGHT_SKIRMISH: {
        # Lights need escape routes and squad awareness
        "slots_0_2": "threats_targeting_me",
        "slots_3_6": "squad_leader_and_nearest_friendlies",
        "slots_7_9": "mission_relevant_targets",
    },

    SuiteType.HEAVY_FIRE_SUPPORT: {
        # Heavies need "the correct five, not twenty"
        "slot_0": "FORCED_ALERT",  # Missile, lock, collision - cannot be overridden
        "slots_1_3": "threats_with_clear_lof",
        "slot_4": "designated_target_from_command",
    },

    SuiteType.TACTICAL_COMMAND: {
        # Command needs squad picture, minimal personal contacts
        "panel_a": "squad_table",
        "panel_b": "engagements_list",
        "panel_c": "personal_contacts_5",
        "panel_d": "orders_mine_first",
    },
}
```

### 10.4 Alert Injection (Forced Interrupts)

Some events **force** themselves into the display regardless of current filters:

```python
class AlertType(Enum):
    MISSILE_LOCK = auto()      # Someone has missile lock on me
    MISSILE_INCOMING = auto()  # Missile in flight toward me
    COLLISION_WARNING = auto() # About to hit terrain/friendly
    REACTOR_CRITICAL = auto()  # Heat emergency
    STABILITY_CRITICAL = auto() # About to fall
    AMBUSH_DETECTED = auto()   # Multiple new contacts in threat arc

@dataclass
class ForcedAlert:
    """Alert that overrides normal display priority."""
    alert_type: AlertType
    source_entity: str | None     # Who caused it
    source_direction: np.ndarray  # Where to look
    severity: float               # 0-1, affects visual urgency
    duration: float               # How long to force-display
```

Heavy mechs get slot 0 as a **forced alert slot** - when an alert fires, it takes over that slot regardless of what the pilot was looking at.

### 10.5 Suite Descriptor Vector

Explicitly tell the policy what mech it is (prevents having to infer from indirect cues):

```python
def build_suite_descriptor(suite: CombatSuiteSpec) -> np.ndarray:
    """Small vector conditioning the policy on suite capabilities."""
    return np.array([
        # Suite type one-hot (5 dims)
        *one_hot(suite.suite_type, 5),

        # Capability scalars (normalized)
        suite.visual_contact_slots / 20.0,  # Contact capacity
        suite.sensor_range_mult,             # Sensor quality
        suite.sensor_fidelity,               # Classification quality
        suite.squad_detail_level / 2.0,      # Squad awareness

        # Panel presence flags (4 dims)
        float(suite.sees_squad_engaged),
        float(suite.issues_orders),
        float(suite.receives_contact_reports),
        float(suite.intel_slots > 5),        # "Has good intel access"
    ], dtype=np.float32)  # 14 dims total
```

**Training implication**: With shared policy weights, this vector lets the network learn role-specific behavior. Consider FiLM-style modulation (Feature-wise Linear Modulation) where suite embedding gates encoder layer outputs.

---

## 11. Role-Specific Actions and Reports

### 11.1 Scout-Unique Actions

```python
SCOUT_ACTIONS = {
    "mark_target": {
        # Creates high-confidence datalink track for command
        "description": "Paint target for squad awareness",
        "cost": "reveals_position",
        "effect": "target becomes priority intel for command",
    },
    "burst_scan": {
        # Temporary sensor boost
        "description": "Active sensor pulse",
        "cost": "heat + signature spike",
        "effect": "refresh all tracks, detect new contacts",
    },
}
```

### 11.2 Command-Unique Actions

```python
COMMAND_ACTIONS = {
    "assign_order": {
        "description": "Issue order to specific mech",
        "params": ["target_mech", "verb", "objective", "urgency"],
    },
    "request_report": {
        "description": "Ping mech for updated status",
        "effect": "forces status update, reveals their current state",
    },
    "set_roe": {
        "description": "Set rules of engagement for squad",
        "options": ["aggressive", "cautious", "hold", "withdraw"],
        "effect": "modifies subordinate behavior constraints",
    },
    "designate_priority": {
        "description": "Mark target as squad priority",
        "effect": "target appears in all squad intel streams",
    },
}
```

### 11.3 Uplink Reports (Squad to Command)

Reports are what make command powerful without being psychic:

```python
@dataclass
class ContactReport:
    """Scout/unit reports a contact to command."""
    reporter_id: str
    track_summary: SensorTrack  # What they saw
    confidence: float
    timestamp: float

@dataclass
class StatusReport:
    """Unit reports own state to command."""
    reporter_id: str
    health_fraction: float
    ammo_fraction: float
    heat_fraction: float
    is_engaged: bool
    needs_help: bool
    current_order_progress: float

@dataclass
class IntentReport:
    """Unit broadcasts intended action for coordination."""
    reporter_id: str
    intent_verb: str            # "moving", "attacking", "withdrawing"
    intent_target: np.ndarray   # Position or direction
    eta: float                  # Expected time to complete
```

**Key insight**: Command doesn't see squad state directly - they see what squad **reports**. This creates:
- Incentive for honest reporting (Brier-scored status)
- Communication bandwidth as resource
- Failure modes when comms are jammed

---

## 12. Damage and Degradation Effects

### 12.1 Sensor Damage

Damage isn't just "HP goes down" - it changes what you can perceive:

```python
def apply_sensor_damage(suite: CombatSuiteSpec, damage_state: DamageState) -> CombatSuiteSpec:
    """Degrade suite capabilities based on damage."""

    degraded = copy(suite)

    # Sensor mast destroyed
    if damage_state.sensor_mast_destroyed:
        degraded.visual_contact_slots = max(2, suite.visual_contact_slots // 3)
        degraded.sensor_range_mult *= 0.5
        degraded.sensor_fidelity *= 0.3

    # Heat affecting sensors
    if damage_state.heat_fraction > 0.8:
        degraded.sensor_fidelity *= (1.0 - damage_state.heat_fraction)
        # Track confidence drops when overheating

    # Comms damage
    if damage_state.comms_damaged:
        degraded.receives_contact_reports = False
        degraded.order_detail_level = 0  # Own orders only

    return degraded
```

### 12.2 EW Effects on Perception

Electronic warfare becomes gameplay, not flavor:

```python
def apply_ew_effects(track: SensorTrack, ew_state: EWState) -> SensorTrack:
    """Modify track based on electronic warfare environment."""

    modified = copy(track)

    # Jamming degrades track quality
    if ew_state.jamming_level > 0:
        modified.confidence *= (1.0 - ew_state.jamming_level * 0.7)
        modified.time_since_update += ew_state.jamming_level * 2.0
        modified.class_confidence *= (1.0 - ew_state.jamming_level * 0.5)

    # ECCM partially restores
    if ew_state.eccm_active:
        restore = min(ew_state.jamming_level, 0.5)
        modified.confidence += restore * 0.5
        modified.class_confidence += restore * 0.3

    # Decoys inject false tracks
    # (handled at track creation, not modification)

    return modified
```

---

## 13. Training Implications

### 13.1 Role Conditioning is Mandatory

With shared weights across mech classes:
- Suite descriptor vector is the **minimum**
- Better: FiLM-style modulation of encoder layers based on suite embedding
- Policy must know "what kind of mech am I" to produce appropriate behavior

### 13.2 Credit Assignment Needs Structure

| Role | Rewarded For | Not Rewarded For |
|------|--------------|------------------|
| Scout | Contacts reported, tracks painted, area covered | Damage dealt (unless self-defense) |
| Light | Flanking success, objective progress | Raw kills |
| Heavy | Damage on designated targets, suppression | Navigation, scouting |
| Command | Squad survival, mission success, order effectiveness | Personal kills |

**Without this structure**: Scouts learn to brawl, command learns to solo, roles collapse.

### 13.3 Paging and Filters Create POMDP

If the pilot can page through contacts or filter views:
- Policy needs memory (LSTM/GRU)
- Current display state must be observable (what filter/sort is active)
- Track age and staleness become critical features

### 13.4 Noise and Uncertainty Should Be Consistent

If track confidence exists in observation:
- Policy can learn to discount low-confidence information
- Creates natural "fog of war" behavior
- Enables EW gameplay to matter for learning

### 13.5 Skill Ceiling via UI Mastery

Display control actions create learnable skills:
- When to switch filters
- When to page through contacts
- When to request reports
- When to trust vs distrust tracks

This is "pilot workload management" - a rich vein for emergent behavior.

---

## 14. The Cockpit Bottleneck (Critical Design Constraint)

*The key insight that prevents the design from collapsing into omniscience.*

### 14.1 The Hard Layer Separation

Even with set encoders and variable-length processing, we maintain a strict information boundary:

```
+------------------+     +------------------+     +------------------+
|   WORLD STATE    | --> |  SENSOR FUSION   | --> |   TRACK STORE    |
|  (ground truth)  |     |  (detection +    |     |  (all detected   |
|                  |     |   noise + decay) |     |   entities)      |
+------------------+     +------------------+     +------------------+
                                                          |
                                                          | NOT directly
                                                          | observed
                                                          v
                         +------------------+     +------------------+
                         |  SUITE DISPLAY   | <-- | DISPLAY MANAGER  |
                         |  (K slots shown) |     | (curation logic) |
                         +------------------+     +------------------+
                                  |
                                  | THIS is what
                                  | the policy sees
                                  v
                         +------------------+
                         |  POLICY INPUT    |
                         |  (obs dict)      |
                         +------------------+
```

**Critical rule**: The policy receives the **display output only**, never the full track store.

### 14.2 Avoiding the Omniscience Trap

A common failure mode when adding set encoders:

> "We'll keep the cockpit fiction but also feed the model the full track set through a transformer"

This quietly makes every mech omniscient again, destroying the design.

**The simple rule that keeps the message honest:**
- **Only encode what the suite shows**
- The full sensor track store exists internally, but is **not directly observed**
- The only way to influence what's shown is through suite settings and display-control actions

```python
# WRONG - feeds everything to encoder (omniscient)
def build_obs(viewer):
    all_tracks = self.track_store.get_all()  # Could be 50+ entities
    return self.set_encoder(all_tracks)       # Policy sees everything

# RIGHT - feeds only what cockpit shows (constrained)
def build_obs(viewer):
    all_tracks = self.track_store.get_all()   # Internal only
    displayed = self.display_manager.render(   # Suite curates
        all_tracks,
        viewer.suite,
        viewer.display_settings
    )  # Returns K slots based on suite
    return self.set_encoder(displayed)         # Policy sees cockpit
```

### 14.3 The Polar Histogram "Scope" Panel

Real cockpits give you both:
- A short ranked list of track files (high detail, low count)
- A coarse scope/tactical picture (low detail, high coverage)

Model this as a **fixed-size summary** that provides awareness without identity:

```python
@dataclass
class ScopePanel:
    """Low-resolution situational awareness without individual tracks."""

    # Polar histogram: bearing bins × range bands
    # e.g., 8 bearings × 3 ranges = 24 bins
    track_count_estimate: np.ndarray    # (24,) how many in each bin
    threat_mass: np.ndarray             # (24,) weighted by mech class
    jamming_level: np.ndarray           # (24,) EW interference per sector
    recent_fire: np.ndarray             # (24,) muzzle flash / projectile activity

    # No identity, no exact vectors, no classification
    # Just "something ugly is building up on the left flank"

def build_scope_panel(tracks: list[SensorTrack], config: ScopeConfig) -> ScopePanel:
    """Aggregate tracks into coarse situational awareness."""
    panel = ScopePanel(...)

    for track in tracks:
        bearing_bin = int((track.bearing + np.pi) / (2 * np.pi) * config.n_bearings)
        range_bin = int(np.clip(track.range / config.max_range, 0, 1) * config.n_ranges)
        bin_idx = bearing_bin * config.n_ranges + range_bin

        panel.track_count_estimate[bin_idx] += track.confidence
        panel.threat_mass[bin_idx] += THREAT_WEIGHTS[track.class_estimate] * track.confidence
        # etc

    return panel
```

**Result**: The pilot can "sense" that something is building up on the left flank, but still has to use the contact list, filters, and locks to get actionable details.

### 14.4 Slot Stickiness (Hysteresis)

Contacts do **not** reshuffle every frame. Tracks stay in their slots unless:
- They time out (track age exceeded)
- The pilot pages
- A new track outranks by a significant margin
- An alert forces an insertion

```python
def update_display_slots(
    current_slots: list[Track | None],
    candidate_tracks: list[Track],
    config: DisplayConfig,
) -> list[Track | None]:
    """Update slots with hysteresis to prevent jitter."""

    new_slots = list(current_slots)  # Start with current

    for i, current in enumerate(current_slots):
        if current is None:
            # Empty slot - fill with best available
            best = pop_best_candidate(candidate_tracks)
            if best:
                new_slots[i] = best
        elif current.is_stale:
            # Timed out - replace
            new_slots[i] = pop_best_candidate(candidate_tracks)
        else:
            # Check if something significantly better exists
            best = peek_best_candidate(candidate_tracks)
            if best and best.priority > current.priority + config.hysteresis_margin:
                new_slots[i] = pop_best_candidate(candidate_tracks)
            # else: keep current (stability)

    return new_slots
```

**Training benefit**: Stable, non-stationary-resistant input stream.

### 14.5 Pilot Workload Costs

Changing filters, paging, requesting datalink refresh, burst scans, etc. should cost something:

| Action | Cost |
|--------|------|
| Change filter/sort | Small time delay (0.5s) |
| Page to next screen | Attention cost (reduced aim stability) |
| Burst scan | Heat + signature spike |
| Request datalink refresh | Comms bandwidth, reveals interest |
| Lock target | Alerts target's RWR |

```python
def apply_workload_cost(action: DisplayAction, mech_state: MechState) -> None:
    """Apply costs for UI manipulation."""

    if action.type == ActionType.CHANGE_FILTER:
        mech_state.aim_stability *= 0.9  # Brief distraction
        mech_state.action_delay += 0.3   # Processing time

    elif action.type == ActionType.BURST_SCAN:
        mech_state.heat += 5.0
        mech_state.signature_spike = 2.0  # Seconds of elevated visibility

    elif action.type == ActionType.REQUEST_DATALINK:
        mech_state.comms_bandwidth -= 1
        # Enemies with SIGINT might detect the request
```

**Result**: "UI mastery" becomes a learnable skill. Frantic spam is punished.

---

## 15. Hierarchy as Data Compression

### 15.1 The Thesis

> **"Hierarchy is not just who gives orders. It is how we compress reality into actionable decisions."**

This is the core insight that makes 100v100 tractable without omniscient policies.

### 15.2 Compression at Each Level

| Level | Sees | Produces |
|-------|------|----------|
| **Squad mech** | Small contact display + immediate orders | Status reports, contact reports |
| **Squad leader** | Richer squad panel, micro-orders | Squad summary, intent reports |
| **Platoon leader** | Squad summaries, allocation decisions | Platoon status, maneuver intent |
| **Company commander** | Platoon summaries, operational picture | Objectives, priorities, ROE |

Each level sees **less raw detail** but more **structured meaning**.

### 15.3 Why This Works

```
100 mechs × 25 features = 2,500 dims (impossible to learn)

vs.

Squad member sees:
  - 5 contact slots × 25 features = 125 dims
  - 4 squad icons × 4 features = 16 dims
  - 1 order × 16 features = 16 dims
  - Total: ~160 dims (learnable)

Squad leader sees:
  - Same personal view
  - Plus 9 squad status × 12 features = 108 dims
  - Plus 5 engagement summaries × 8 features = 40 dims
  - Total: ~310 dims (learnable)

Company commander sees:
  - Minimal personal view
  - 10 squad summaries × 15 features = 150 dims
  - Total: ~200 dims (learnable)
```

The hierarchy **is** the compression algorithm.

---

## 16. Implementation: Keeping VecEnv Happy

### 16.1 Fixed Shapes with Suite Variation

Keep a single global cap, but each suite has `K_visible`:

```python
MAX_CONTACTS = 32  # Global cap for array allocation

class EchelonEnv:
    def _build_obs(self, viewer: MechState) -> dict:
        suite = self._get_suite(viewer)

        # Always allocate MAX_CONTACTS
        contacts = np.zeros((MAX_CONTACTS, CONTACT_DIM), dtype=np.float32)
        contact_mask = np.ones(MAX_CONTACTS, dtype=np.float32)

        # But only fill suite.visual_contact_slots
        displayed = self.display_manager.render(viewer, suite)
        for i, track in enumerate(displayed[:suite.visual_contact_slots]):
            contacts[i] = self._encode_track(track)
            contact_mask[i] = 0.0

        # Rest remains padding (mask = 1.0)
        return {
            'contacts': contacts,
            'contact_mask': contact_mask,
            # ...
        }
```

**Result**: Vectorized RL gets fixed shapes; cognitive constraint is preserved; suite variation works.

---

## 17. Demonstration and Ablation Ideas

### 17.1 Key Ablations for Validation

| Ablation | Change | Expected Result |
|----------|--------|-----------------|
| **A: Oracle vs Suite-Bounded** | Give one team full track store access | Oracle learns weird omniscient twitch tactics; suite-bounded learns scouting, reporting, coordination |
| **B: Same Stats, Different Suites** | Identical chassis, swap suite packages | Behavior changes dramatically because perception changed, not weapons |
| **C: Remove Reports** | Disable uplink from squad to command | Command goes blind; squads become local brawlers; coordination collapses |
| **D: Remove Alert Injection** | Disable forced interrupts | Heavies die to missiles and flankers more; add alerts back, survivability rises |
| **E: Remove Slot Stickiness** | Reshuffle every frame | Training becomes unstable; policies learn jittery target-switching |
| **F: Remove Scope Panel** | No coarse situational awareness | Flanks get ignored; ambushes more deadly; add scope back, survivability rises |

### 17.2 Demo-Friendly Contrasts

For showing the design in 30 seconds:

**Contrast 1: Scout vs Heavy perception**
```
Scout display: 20 contacts, tracks enemy formation, reports to command
Heavy display: 5 contacts, sees only immediate threats, waits for targeting data
Same battle, completely different cognitive experience
```

**Contrast 2: Command with/without reports**
```
Without: Command sees only personal contacts, issues blind orders
With: Command sees full squad picture, coordinates effectively
Toggle one boolean, behavior transforms
```

**Contrast 3: Information vs Weapons**
```
Team A: Better guns, worse suites
Team B: Worse guns, better suites
Hypothesis: Suite advantage wins because coordination beats firepower
```

### 17.3 The Project Page Pitch

> "We don't give agents everything they can see. We give them a cockpit with designed limitations. The cockpit is a game mechanic, not a rendering artifact."

> "Hierarchy is not just who gives orders. It is how we compress reality into actionable decisions."

> "Role differentiation emerges from information contracts, not hard-coded behavior. Scouts scout because scouting is what their perception system makes easy."

---

## 18. Relationship to External Audit

This design emerged from reviewing an external "Strategic Audit of 14-Dimensional Mission Embeddings" report. Key findings:

**Report's Valid Concerns (already addressed in Echelon):**
- Use relative coordinates, not absolute ✓
- Use sin/cos for angles ✓
- Use one-hot for categoricals ✓
- Normalize all features ✓

**Report's Recommendations We Adopted:**
- Set encoders for variable entity counts (this document)
- Explicit tactical features in observation

**Report's Recommendations We Rejected:**
- VLA/language-conditioned tasks (overkill for 7 verbs)
- Full Transformer attention (DeepSets sufficient for our scale)
- Omniscient entity perception (contradicts cockpit metaphor)

**Novel Extensions Beyond Report:**
- Combat Suite as class feature
- Order broadcast channel for C2
- Hierarchical information architecture matching military doctrine

---

## 19. Attention Budget Model

*A refinement from external review: model cognitive load explicitly.*

### 19.1 The Concept

Rather than "scout gets 20 contact slots", think in terms of **attention points**:

```python
@dataclass
class AttentionBudget:
    """Cognitive load model for display allocation."""

    total_points: int = 100

    # Costs per item
    contact_cost: int = 5      # Each tracked contact
    squad_icon_cost: int = 2   # Each squad icon
    squad_status_cost: int = 8 # Each squad status entry
    order_cost: int = 3        # Each visible order
    intel_cost: int = 4        # Each intel report
    alert_cost: int = 0        # Alerts are mandatory, no cost

    def allocate(self, suite: CombatSuiteSpec) -> dict[str, int]:
        """Compute slots available for each panel given budget."""
        remaining = self.total_points

        # Alerts are free and mandatory
        alert_slots = 2

        # Prioritized allocation
        contact_slots = min(
            suite.visual_contact_slots,
            remaining // self.contact_cost
        )
        remaining -= contact_slots * self.contact_cost

        squad_slots = min(
            suite.squad_position_slots,
            remaining // (self.squad_icon_cost if suite.squad_detail_level == 0
                         else self.squad_status_cost)
        )
        # ... etc

        return {
            'contacts': contact_slots,
            'squad': squad_slots,
            'alerts': alert_slots,
            # ...
        }
```

### 19.2 Why This Matters

| Advantage | Explanation |
|-----------|-------------|
| **Cognitive realism** | Pilots have finite attention, not infinite slots |
| **Smooth degradation** | Damage/heat reduces budget, not slot count |
| **EW integration** | Jamming increases contact_cost (harder to track) |
| **Flexible tradeoffs** | Scout can trade squad awareness for more contacts |

### 19.3 Implementation Note

The attention budget operates at the **display manager** level, not the encoder level:

1. Track store may contain 50+ entities
2. Attention budget determines how many **get rendered**
3. Rendered items go to set encoder
4. VecEnv still sees fixed MAX_CONTACTS (just fewer filled slots)

Stickiness and hysteresis still apply—budget changes don't cause instant reshuffling.

---

## 20. Role Behavior Metrics

*How do we know the design is working?*

### 20.1 Role Differentiation Scoreboard

Simple, interpretable metrics that prove roles emerge from perception, not hard-coding:

**Scout-ish signals:**
```python
scout_metrics = {
    'contacts_reported_per_minute': ...,    # Should be high
    'unique_enemies_first_spotted': ...,    # Scouts find things
    'avg_range_to_nearest_enemy': ...,      # Tends to be larger (standoff)
    'time_with_outward_display_mode': ...,  # Focused externally
    'survival_rate': ...,                   # Should be high (they avoid fights)
}
```

**Heavy-ish signals:**
```python
heavy_metrics = {
    'damage_to_designated_targets': ...,    # Following orders
    'time_with_lof_to_target': ...,         # Firing lane discipline
    'time_stationary': ...,                 # Holding position
    'shots_at_command_targets': ...,        # C2 integration
    'damage_received_ratio': ...,           # Trading damage
}
```

**Command-ish signals:**
```python
command_metrics = {
    'orders_issued_per_minute': ...,        # Active management
    'order_effectiveness': ...,             # Downstream damage within N seconds
    'squad_survival_rate': ...,             # Kept team alive
    'cohesion_score': ...,                  # Squad stayed together
    'retask_rate': ...,                     # Adapted to situation
}
```

### 20.2 Role Separation Score

The ultimate test: **can a simple classifier predict role from behavior?**

```python
def compute_role_separation(episode_logs: list[EpisodeLog]) -> float:
    """
    Higher = more distinct role behavior.
    Train a tiny linear model to predict role from metrics.
    Accuracy above chance = design is working.
    """
    X = []  # Feature matrix from metrics
    y = []  # Role labels

    for episode in episode_logs:
        for mech in episode.mechs:
            X.append(extract_metrics(mech))
            y.append(mech.suite_type)

    clf = LogisticRegression()
    scores = cross_val_score(clf, X, y, cv=5)
    return scores.mean()  # > 0.5 = roles are distinct
```

### 20.3 Ablation Ladder (Curriculum as Validation)

| Phase | Configuration | What We Learn |
|-------|---------------|---------------|
| 1 | All identical suites | Baseline behavior |
| 2 | Different suites, no descriptor | Does obs contract alone differentiate? |
| 3 | Add suite descriptor | Does explicit role help? |
| 4 | Add order system | Does C2 emerge? |
| 5 | Add comm constraints | Does realism break anything? |

Each phase should show measurable increase in role_separation_score.

---

## 21. Order Contracts (Not Absolute Commands)

*Orders are contracts with escape clauses, not robot directives.*

### 21.1 The Problem

If orders are absolute:
- Units do suicidal things
- "The AI is stupid" when units blindly follow

If orders are ignored:
- C2 is meaningless
- Command agents learn orders don't matter

### 21.2 The Solution: Contract Model

```python
@dataclass
class OrderContract:
    """Orders as contracts with conditions and escape clauses."""

    order: TacticalOrder

    # Satisfaction tracking
    progress: float = 0.0              # 0.0 to 1.0
    acknowledged: bool = False
    blocked: bool = False
    override_reason: OrderOverride | None = None

    # Valid override reasons
    class OrderOverride(Enum):
        BLOCKED_PATH = auto()          # Can't reach objective
        CRITICAL_THREAT = auto()       # Must handle immediate danger
        TARGET_DESTROYED = auto()      # Mission complete (by others)
        RESOURCE_DEPLETED = auto()     # Out of ammo/heat capacity
        COMMS_LOST = auto()            # Can't confirm order still valid

    def is_valid_override(self) -> bool:
        """Only these reasons justify not following orders."""
        return self.override_reason is not None

# Reward shaping:
# - Penalize ignoring orders WITHOUT valid override
# - Reward following orders
# - Reward commanders for issuing orders that get followed
# - Reward quick cancellation of bad orders
```

### 21.3 Status Reporting

Units must explain themselves:

```python
@dataclass
class StatusReport:
    """Unit explains their state to command."""

    reporter_id: str
    current_order_id: str | None
    order_progress: float

    # If not following order, why?
    override_active: bool
    override_reason: OrderOverride | None

    # Current state
    is_engaged: bool
    is_blocked: bool
    needs_support: bool
    ammo_fraction: float
    health_fraction: float
```

**Key insight**: Command doesn't see squad state directly—they see what units **report**. This creates incentive for honest communication.

---

## 22. Sensor Failure Fallbacks

*When information collapses, what do agents do?*

### 22.1 Failure Modes

| Failure | Cause | Observable Effect |
|---------|-------|-------------------|
| **Sensor damage** | Combat damage | `sensor_quality` drops |
| **Jamming** | Enemy ECM | `jam_level` increases, confidence drops |
| **Overheat** | Thermal overload | Classification degrades |
| **Comms loss** | Distance, terrain, EW | Datalink tracks go stale |

### 22.2 Graceful Degradation

Sensor failure isn't "information disappears"—it's "information becomes unreliable":

```python
def apply_sensor_degradation(
    tracks: list[SensorTrack],
    sensor_quality: float,  # 0.0 = blind, 1.0 = perfect
) -> list[SensorTrack]:
    """Degrade track quality with sensor damage."""

    degraded = []
    for track in tracks:
        new_track = copy(track)

        # Confidence drops
        new_track.confidence *= sensor_quality

        # Classification becomes uncertain
        if sensor_quality < 0.5:
            new_track.class_probabilities = uniform_distribution()

        # Position uncertainty grows
        new_track.position_uncertainty /= max(0.1, sensor_quality)

        # Tracks age faster (can't refresh them)
        new_track.time_since_update += (1.0 - sensor_quality) * 2.0

        degraded.append(new_track)

    return degraded
```

### 22.3 Trained Fallback Behaviors

With proper training, agents should learn these fallback patterns:

| Condition | Expected Behavior |
|-----------|------------------|
| **Low confidence tracks** | Conservative engagement, avoid friendly fire |
| **Stale tracks** | Don't rely on old position estimates |
| **Comms lost** | Regroup on last known leader position |
| **All sensors failed** | Withdraw to cover, await recovery |
| **Jammed** | Activate ECCM, close to visual range |

### 22.4 Testing Fallbacks

Explicit tests for failure handling:

```python
def test_sensor_failure_behavior():
    """Verify agents don't glitch when sensors fail."""

    # Setup: agent with high damage and failed sensors
    env.reset()
    agent.sensor_quality = 0.1

    for _ in range(100):
        action = policy(agent.obs)

        # Should NOT:
        # - Fire at empty space
        # - Charge into unknown
        # - Ignore immediate threats (alerts still work)

        # SHOULD:
        # - Move toward last known squad position
        # - Seek cover
        # - Request status reports
```

---

## 23. Implementation Priority (Revised)

Based on all feedback, the recommended build order:

### Phase 1: Core (MVP)
1. DeepSet encoder for contacts (mean pooling)
2. Panel stats (explicit counts)
3. Suite-variable contact slots (5/10/20)

### Phase 2: Role Differentiation
4. Suite descriptor in self features
5. Role behavior metrics
6. Basic ablation tests

### Phase 3: C2 Integration
7. Order broadcast channel
8. Order contracts with overrides
9. Status reporting

### Phase 4: Resilience
10. Sensor degradation
11. Comms failure handling
12. Attention budget model

### Phase 5: Polish
13. Alert injection
14. Slot stickiness
15. Scope panel

---

## 24. External Review Notes (Gemini)

*Key insights from strategic audit.*

### 24.1 Tunnel Vision is Doctrine, Not Deficiency

> "If a Heavy has 5 slots filled with low-threat enemies in front, and a high-threat enemy enters from the flank, the Heavy is effectively blind."

**This is correct behavior.** The Heavy's job is to kill what's in front of them. Situational awareness is the Scout's job. Coordination is Command's job.

The "blindness" forces:
- **Scan discipline**: Agents learn to periodically `CHANGE_FILTER` or `PAGE_NEXT` to "check six"
- **Trust**: Heavy trusts the Order Stream to override local priority when Command sees the flanker
- **Role emergence**: Scouts exist *because* heavies can't see everything

### 24.2 Low-Fi Scope Panel (The "Tap on Shoulder")

The Scope Panel should be **deliberately vague**—peripheral vision, not radar:

```python
@dataclass
class ScopePanelLowFi:
    """Minimal peripheral awareness. 4-8 floats."""

    # 4 quadrants: Front, Right, Rear, Left
    # Each is just "intensity" - sum of threat mass, no identity
    quadrant_intensity: np.ndarray  # (4,) normalized 0-1

    # Optional: "something changed" flag per quadrant
    quadrant_delta: np.ndarray      # (4,) positive = new threats appeared

    # Total: 8 dims

    # This tells you WHERE to look, not WHAT to shoot
```

**The hook**: It doesn't give coordinates. It gives a reason to turn your head or cycle your display.

### 24.3 EW as Mana Denial

> "If I jam your Scout, I am not just lowering their accuracy; I am 'tapping out' your Command unit because the IntelStream dries up."

This is the correct framing. Jamming in this architecture is **information denial**, not just accuracy debuff:

| Traditional RL | This Architecture |
|----------------|-------------------|
| Jamming = `prob_hit *= 0.7` | Jamming = IntelStream goes stale |
| Scout jammed = scout worse | Scout jammed = **Command blind** |
| Local effect | **Systemic effect** |

The value of a Scout isn't their personal combat effectiveness—it's their contribution to the COP.

### 24.4 Credit Assignment for Command (Counterfactual Baseline)

The hardest training challenge: Commander issues `ORDER_FLANK`, Light executes perfectly but dies to RNG. Commander gets negative reward?

**Solution**: Reward Command for *tactical advantage created*, not outcomes:

```python
def command_reward(order: TacticalOrder, before: GameState, after: GameState) -> float:
    """Counterfactual baseline: did the order improve the situation?"""

    # What was the assignee's expected value BEFORE the order?
    ev_before = estimate_value(before, order.assignee_id)

    # What was it AFTER receiving the order (but before execution)?
    ev_after = estimate_value(after, order.assignee_id)

    # Reward the delta, not the outcome
    return ev_after - ev_before
```

Command optimizes for *putting units in advantageous positions*, not for RNG outcomes.

### 24.5 Zombie Slot Decay

Stickiness is good, but stale tracks must eventually yield:

```python
def update_slot_priority(track: SensorTrack, dt: float) -> float:
    """Decay priority of un-refreshed tracks."""

    # Base priority from threat assessment
    priority = track.threat_priority

    # Decay if not refreshed by sensors or intel
    staleness_penalty = 1.0 - min(1.0, track.time_since_update / MAX_TRACK_AGE)
    priority *= staleness_penalty

    # Dead/destroyed tracks decay faster
    if track.confidence < 0.1:
        priority *= 0.1

    return priority
```

Fresh contacts naturally bubble up; ghosts fade out.

### 24.6 The OODA Loop Implementation

> "Most RL collapses Observe/Orient into one step. By separating them, you allow the agent to learn *how to Orient* as a distinct skill from *how to Decide*."

This is the architectural thesis:

| OODA Phase | Implementation | Learnable? |
|------------|----------------|------------|
| **Observe** | Sensors → Track Store | No (physics) |
| **Orient** | Display Manager → Encoder | **Yes** (filter/sort actions) |
| **Decide** | LSTM → Action Head | Yes (policy) |
| **Act** | Motor Output | No (physics) |

Traditional RL learns Decide. This architecture also learns Orient.

### 24.7 The JTAC Analogy

> "The Heavy doesn't need to *see* the flanker to know it's there; it just needs to trust the Commander's order."

The Order Stream (Stream D) is the "voice in the headset":
- Heavy's visual stream is full of immediate threats
- Command sees flanker via Scout's intel
- Command issues: `TARGET_ENTITY_ID: flanker`
- Order appears in Heavy's Stream D with `is_my_order=1.0`
- Heavy learns to trust orders over local perception

This is **doctrinal trust**, not omniscience.

---

*This design enables the shift from "all mechs are identical observers" to "each role sees the battlefield differently" - creating genuine tactical specialization through information architecture.*
