# Subsystem Catalog

**Project:** Echelon
**Analysis Date:** 2025-12-24
**Total Subsystems:** 8

---

## Table of Contents

1. [Simulation Core](#1-simulation-core)
2. [Environment](#2-environment)
3. [Procedural Generation](#3-procedural-generation)
4. [Navigation](#4-navigation)
5. [RL Model](#5-rl-model)
6. [Arena/Self-Play](#6-arenaself-play)
7. [Heuristic Agents](#7-heuristic-agents)
8. [Configuration & Actions](#8-configuration--actions)

---

## 1. Simulation Core

**Location:** `echelon/sim/`
**Files:** sim.py (1,407 lines), world.py (255 lines), mech.py (73 lines), projectile.py (27 lines), los.py (345 lines)
**Total Lines:** ~2,107
**Confidence:** HIGH

### Responsibility
Orchestrates physics simulation, combat mechanics, projectile dynamics, line-of-sight computation, and world state management for real-time mech combat.

### Key Components

| Component | Purpose |
|-----------|---------|
| `Sim` | Main orchestrator managing mechs, projectiles, smoke clouds, physics integration |
| `VoxelWorld` | 3D voxel grid with 10 material types, per-voxel HP, collision queries |
| `MechState` | 20-field dataclass tracking position, HP, heat, stability, weapons |
| `Projectile` | State container for homing/ballistic/linear projectiles |
| `SpatialGrid` | O(1) spatial hash for efficient collision/target queries |
| `has_los()` | Numba-accelerated 3D DDA raycasting for visibility |

### Key Patterns
- **Event-Driven Combat:** All actions emit dict events for observation/replay
- **Spatial Optimization:** SpatialGrid reduces O(n) queries to O(r^2/cell_size^2)
- **Fixed Timestep:** dt_sim (0.01s) with decision_repeat substeps
- **Deterministic Randomness:** Single seeded RNG for reproducible replays

### Dependencies
- **Inbound:** env/env.py (EchelonEnv.step calls Sim.step)
- **Outbound:** config.py (weapon specs), actions.py (indices), NumPy

---

## 2. Environment

**Location:** `echelon/env/`
**Files:** env.py (~1,630 lines)
**Confidence:** HIGH

### Responsibility
Provides Gymnasium-compatible RL interface for 20-agent multi-agent combat. Manages world generation, observation construction, reward calculation, and episode termination.

### Key Components

| Component | Purpose |
|-----------|---------|
| `EchelonEnv` | Main Gym-like environment class |
| `reset()` | World gen, mech spawning, NavGraph building |
| `step()` | Action processing, sim loop, rewards, termination |
| `_obs()` | Per-agent observation (contacts, comm, local_map, telemetry) |

### Observation Structure (~607 dims)
- **Contacts:** 5 slots x 22 dims = 110 dims (position, velocity, HP, class, visibility)
- **Pack Comm:** 10 mechs x 8 dims = 80 dims (1-step delayed)
- **Local Map:** 11x11 = 121 dims (ego-centric occupancy)
- **Telemetry:** 16x16 = 256 dims (world overview)
- **Self Features:** 40 dims (status, weapons, objective, scoring)

### Reward Components
| Component | Weight | Description |
|-----------|--------|-------------|
| Zone tick | 0.10 | Proportional tonnage in capture zone |
| Approach | 0.25 | Potential-shaped distance to zone |
| Damage | 0.005 | Per damage point dealt |
| Kill | 1.0 | Per kill credited |
| Assist | 0.5 | Per paint assist |
| Death | -0.5 | Penalty for dying |
| Terminal | +/-5.0 | Win/loss at episode end |

### Dependencies
- **Inbound:** train_ppo.py, eval_policy.py, smoke.py, arena.py, tests
- **Outbound:** sim/, gen/, nav/, config.py, actions.py

---

## 3. Procedural Generation

**Location:** `echelon/gen/`
**Files:** layout.py, biomes.py, corridors.py, validator.py, recipe.py, objective.py, transforms.py
**Confidence:** HIGH

### Responsibility
Generates deterministic 3D voxel maps by composing procedural biome fill, connectivity validation, and macro corridor carving.

### Generation Pipeline
```
seed → Layout (quadrant split) → Biomes (16 types) → Corridors → Validator → NavGraph
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `generate_layout()` | Quadrant split, biome assignment, brush application |
| `BiomeBrush` | 16 terrain types (urban, industrial, forest, volcanic, etc.) |
| `carve_macro_corridors()` | Objective ring + spawn lanes |
| `ConnectivityValidator` | NavGraph test + 2D A* fallback with staircase carving |
| `build_recipe()` | Deterministic map hash for replay reproduction |

### Key Patterns
- **Brush-Based Fill:** Stateless BiomeBrush functions for procedural features
- **Retry Validation:** 3-attempt loop: NavGraph → 2D A* + carve → fail
- **Deterministic Randomness:** Single RNG threads through entire pipeline

### Dependencies
- **Inbound:** env/env.py (VoxelWorld.generate in reset)
- **Outbound:** sim/world.py, nav/graph.py, nav/planner.py, config.py

---

## 4. Navigation

**Location:** `echelon/nav/`
**Files:** graph.py (212 lines), planner.py (92 lines)
**Total Lines:** 304
**Confidence:** HIGH

### Responsibility
Constructs 2.5D walkable surface graph from voxel terrain and provides A* pathfinding.

### Key Components

| Component | Purpose |
|-----------|---------|
| `NavGraph.build()` | Vectorized node detection + iterative edge building |
| `NavNode` | Dataclass: id, world position, edges |
| `Planner.find_path()` | A* with Euclidean heuristic, max_visited limit |
| `PathStats` | Result container (found, path, cost, visited) |

### Architecture
1. **Walkable Detection:** Solid below + clearance above (vectorized NumPy)
2. **Footprint Erosion:** scipy.ndimage.minimum_filter for mech radius
3. **Edge Building:** 8-neighbor connectivity with step-up rules (max 2 voxels)
4. **Z-Height Bias:** 1000-unit penalty for mismatched heights in nearest-node

### Dependencies
- **Inbound:** gen/validator.py, server.py, env/env.py (nav_assist mode)
- **Outbound:** sim/world.py (VoxelWorld), scipy, NumPy

---

## 5. RL Model

**Location:** `echelon/rl/`
**Files:** model.py (126 lines)
**Confidence:** HIGH

### Responsibility
Actor-Critic LSTM policy network for PPO training with Gaussian action distributions and tanh squashing.

### Architecture
```
obs [batch, obs_dim]
  → encoder (2-layer Tanh MLP, hidden_dim=128)
  → LSTM core (lstm_hidden_dim=128)
  → actor_mean + actor_logstd → Normal distribution
  → tanh squashing → action [-1, 1]
  → critic head → value
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `ActorCriticLSTM` | Main nn.Module with encoder, LSTM, actor/critic heads |
| `LSTMState` | Frozen dataclass holding (h, c) tensors |
| `get_action_and_value()` | Returns action, logprob, entropy, value, next_state |
| `_atanh()` | Numerically stable inverse tanh for log-prob |

### Key Patterns
- **Tanh Squashing:** Bounded actions with Jacobian correction for log-prob
- **LSTM State Reset:** done mask clears hidden state at episode boundaries
- **Orthogonal Init:** Small gain (0.01) on actor, standard (1.0) on critic

### Dependencies
- **Inbound:** train_ppo.py, eval_policy.py, arena.py
- **Outbound:** PyTorch (nn, distributions)

---

## 6. Arena/Self-Play

**Location:** `echelon/arena/`
**Files:** league.py, match.py, glicko2.py
**Confidence:** HIGH

### Responsibility
Self-play infrastructure for policy evaluation and skill-based ranking via Glicko-2 rating system.

### Key Components

| Component | Purpose |
|-----------|---------|
| `League` | Checkpoint pool, entry management, promotion mechanics |
| `LeagueEntry` | Policy metadata (path, kind, rating, game count) |
| `play_match()` | Execute 20-agent match between two policies |
| `Glicko2Rating` | Rating (mu), deviation (phi), volatility (sigma) |
| `rate()` | Core Glicko-2 update algorithm |

### Self-Play Flow
1. Maintain top-K commanders + recent candidates
2. Sample opponent from pool for training
3. Execute matches with deterministic seeding
4. Update ratings via two-phase commit (snapshot then commit)
5. Promote if candidate in conservative top-K

### Key Patterns
- **Snapshotted Ratings:** GameResult captures opponent rating at match time
- **Two-Phase Atomicity:** Compute all ratings first, then commit
- **Stable Versioning:** Uses file metadata instead of content hashing

### Dependencies
- **Inbound:** train_ppo.py (arena mode), scripts/arena.py (CLI)
- **Outbound:** env/env.py, rl/model.py, config.py, torch

---

## 7. Heuristic Agents

**Location:** `echelon/agents/`
**Files:** heuristic.py (320 lines)
**Confidence:** HIGH

### Responsibility
Rule-based baseline agent policies for testing, evaluation, and self-play benchmarking.

### Key Components

| Component | Purpose |
|-----------|---------|
| `HeuristicPolicy` | Main class with configurable parameters |
| `act()` | Single-step decision: movement + targeting + firing |

### Decision Logic
1. **Anti-Stuck:** Detect movement lockup, trigger escape maneuver
2. **Target Selection:** Nearest alive enemy (O(n) scan)
3. **Movement:** Approach/back-off with squad cohesion blending
4. **Firing:** Class-specific weapon logic (range, LOS, heat checks)
5. **Handicap:** Optional weapon fire probability for balance

### Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| `desired_range` | 5.5 | Brawling distance target |
| `approach_speed_scale` | 0.5 | Movement throttle multiplier |
| `weapon_fire_prob` | 0.5 | Fraction of ticks weapons fire |

### Dependencies
- **Inbound:** smoke.py, eval_policy.py, train_ppo.py
- **Outbound:** env/env.py, sim/sim.py, sim/los.py, gen/objective.py

---

## 8. Configuration & Actions

**Location:** `echelon/config.py`, `echelon/actions.py`, `echelon/constants.py`
**Confidence:** HIGH

### Responsibility
Define immutable configuration hierarchies, action space enumerations, and global constants.

### Key Components

| Component | Purpose |
|-----------|---------|
| `WorldConfig` | Voxel world parameters (size, connectivity) |
| `MechClassConfig` | Per-class mech stats (HP, speed, heat) |
| `WeaponSpec` | Weapon definitions (range, damage, cooldown) |
| `EnvConfig` | Top-level config (world + timing + features) |
| `ActionIndex` | IntEnum for 9D action space |
| `PACK_SIZE` | Constant = 10 (mech pack composition) |

### Configuration Hierarchy
```
EnvConfig (top-level)
  ├── WorldConfig (voxel world, connectivity)
  ├── num_packs, dt_sim, decision_repeat
  ├── observation_mode, comm_dim
  └── Feature toggles (enable_ewar, enable_comm, etc.)
```

### Action Space (9D Continuous)
| Index | Action | Range |
|-------|--------|-------|
| 0-3 | Movement (forward, strafe, vertical, yaw) | [-1, 1] |
| 4 | PRIMARY weapon | [-1, 1] |
| 5 | VENT (heat dump) | [-1, 1] |
| 6 | SECONDARY (missile/ECM) | [-1, 1] |
| 7 | TERTIARY (paint/gauss/autocannon) | [-1, 1] |
| 8 | SPECIAL (smoke) | [-1, 1] |

### Key Patterns
- **Frozen Dataclasses:** Immutable configuration for thread safety
- **Global Weapon Singletons:** LASER, MISSILE, GAUSS, etc.
- **Feature Toggles:** Enable ablations without code changes

### Dependencies
- **Inbound:** All subsystems import config
- **Outbound:** stdlib only (dataclasses, enum)

---

## Dependency Summary

```
                    ┌─────────────┐
                    │   config    │
                    │   actions   │
                    │  constants  │
                    └──────┬──────┘
                           │ (used by all)
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
    ▼                      ▼                      ▼
┌────────┐           ┌──────────┐           ┌─────────┐
│  sim/  │◄──────────│   env/   │──────────►│  gen/   │
│        │           │          │           │         │
└────┬───┘           └────┬─────┘           └────┬────┘
     │                    │                      │
     │                    │                      ▼
     │                    │                 ┌─────────┐
     │                    │                 │  nav/   │
     │                    │                 └─────────┘
     │                    │
     │              ┌─────┴─────┐
     │              │           │
     ▼              ▼           ▼
┌─────────┐   ┌──────────┐   ┌─────────┐
│ agents/ │   │   rl/    │   │ arena/  │
└─────────┘   └──────────┘   └─────────┘
```

---

## Confidence Summary

| Subsystem | Confidence | Notes |
|-----------|------------|-------|
| Simulation Core | HIGH | Comprehensive code review, clear patterns |
| Environment | HIGH | Complete API traced, reward components documented |
| Procedural Generation | HIGH | Full pipeline analyzed, determinism verified |
| Navigation | HIGH | Compact, well-tested, clear algorithm |
| RL Model | HIGH | Standard PPO architecture, well-documented |
| Arena/Self-Play | HIGH | Clear self-play loop, Glicko-2 standard |
| Heuristic Agents | HIGH | Straightforward rule-based logic |
| Configuration | HIGH | Frozen dataclasses, comprehensive coverage |

---

*Generated by 8 parallel subagent analyses on 2025-12-24*
