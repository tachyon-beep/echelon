# Discovery Findings

## Project Overview

**Name:** Echelon
**Version:** 0.1.0
**License:** MIT
**Python:** 3.13 (required)
**Purpose:** Educational Deep Reinforcement Learning environment for mech tactics simulation

## Technology Stack

### Core Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥2.6 | Neural network training, CUDA support |
| NumPy | ≥2.1 | Tensor operations, voxel grid storage |
| Numba | ≥0.58 | JIT compilation for performance-critical loops |
| SciPy | ≥1.10 | Scientific computing utilities |
| FastAPI | ≥0.125 | Replay server REST/SSE API |
| Uvicorn | ≥0.38 | ASGI server |
| W&B | ≥0.19 | Experiment tracking |
| Gymnasium/Gym | (optional) | RL environment interface compatibility |
| Pydantic | ≥2.12 | Settings management |
| WebSockets | ≥13.0 | Real-time communication |

### Development Dependencies

- pytest, pytest-cov - Testing
- hypothesis - Property-based testing
- mypy - Type checking
- ruff - Linting/formatting
- vulture - Dead code detection
- mutmut - Mutation testing
- pre-commit - Git hooks

## Codebase Metrics

| Metric | Value |
|--------|-------|
| Total Python files (excl. venv) | 285 |
| Lines of code | ~273K |
| Main package | `echelon/` (38 files) |
| Test files | 26 |
| Script files | 17 |

## Directory Structure

```
echelon/
├── echelon/           # Main package
│   ├── sim/           # Simulation core (physics, combat, LOS)
│   ├── env/           # Gym-like environment wrapper
│   ├── gen/           # Procedural world generation
│   ├── nav/           # Navigation graph and pathfinding
│   ├── rl/            # Actor-Critic LSTM model
│   ├── arena/         # Self-play infrastructure
│   ├── agents/        # Heuristic agent behaviors
│   ├── config.py      # Configuration dataclasses
│   ├── actions.py     # Action space definitions
│   └── constants.py   # Global constants
├── scripts/           # Training, evaluation, utilities
├── tests/             # Unit, integration, performance tests
└── server.py          # FastAPI visualization server
```

## Entry Points

### Primary

| Entry Point | Location | Purpose |
|-------------|----------|---------|
| `EchelonEnv` | `echelon/env/env.py` | Main RL environment (Gym-like interface) |
| `train_ppo.py` | `scripts/train_ppo.py` | PPO training script (~58K lines) |
| `server.py` | `server.py` | SSE-based replay visualization server |

### Secondary

| Entry Point | Location | Purpose |
|-------------|----------|---------|
| `smoke.py` | `scripts/smoke.py` | Quick environment validation |
| `arena.py` | `scripts/arena.py` | Self-play league execution |
| `eval_policy.py` | `scripts/eval_policy.py` | Policy evaluation |

## Subsystem Identification

### 1. Simulation Core (`echelon/sim/`)
**Files:** sim.py, world.py, mech.py, projectile.py, los.py
**Responsibility:** Physics simulation, combat mechanics, voxel world state
**Complexity:** HIGH
**Key Classes:** `Sim`, `VoxelWorld`, `MechState`, `Projectile`, `SpatialGrid`

### 2. Environment (`echelon/env/`)
**Files:** env.py
**Responsibility:** Gym-like interface, observation construction, reward calculation
**Complexity:** HIGH
**Key Classes:** `EchelonEnv`

### 3. Procedural Generation (`echelon/gen/`)
**Files:** layout.py, biomes.py, corridors.py, validator.py, recipe.py, objective.py, transforms.py
**Responsibility:** Deterministic map generation, connectivity validation
**Complexity:** MEDIUM
**Key Classes:** `Blueprint`, `ConnectivityValidator`, `NavGraph`

### 4. Navigation (`echelon/nav/`)
**Files:** graph.py, planner.py
**Responsibility:** 2.5D navigation graph, A* pathfinding
**Complexity:** MEDIUM
**Key Classes:** `NavGraph`, `Planner`, `NodeID`

### 5. RL Model (`echelon/rl/`)
**Files:** model.py
**Responsibility:** Actor-Critic LSTM policy network
**Complexity:** MEDIUM
**Key Classes:** `ActorCriticLSTM`, `LSTMState`

### 6. Arena/Self-Play (`echelon/arena/`)
**Files:** league.py, match.py, glicko2.py
**Responsibility:** Self-play training infrastructure, skill rating
**Complexity:** MEDIUM
**Key Classes:** `League`, `Match`, `Glicko2Rating`

### 7. Heuristic Agents (`echelon/agents/`)
**Files:** heuristic.py
**Responsibility:** Rule-based baseline agents
**Complexity:** LOW
**Key Classes:** `HeuristicAgent`

### 8. Configuration (`echelon/config.py`)
**Responsibility:** Frozen dataclass configuration (worlds, mechs, weapons)
**Complexity:** LOW
**Key Classes:** `EnvConfig`, `WorldConfig`, `MechClassConfig`, `WeaponSpec`

### 9. Actions (`echelon/actions.py`)
**Responsibility:** Action space enumeration
**Complexity:** LOW
**Key Classes:** `ActionIndex`

## Key Architectural Patterns

### 1. Deterministic Simulation
- All randomness controlled via seed
- Maps reproducible via `recipe.py` hash
- Critical for RL benchmarking and replay

### 2. Pack-Based Team Composition
- Each team has 1 "pack" of 10 mechs
- Composition: 1 Heavy, 5 Medium, 3 Light, 1 Scout
- Pack-scoped communication and paint locks

### 3. Voxel World Representation
- 3D grid with `[z, y, x]` indexing
- Material types: SOLID, LAVA, WATER, GLASS, FOLIAGE, DEBRIS
- Per-voxel HP for destructible terrain

### 4. Multi-Agent RL
- 20 agents total (10 blue, 10 red)
- Shared policy across all agents
- LSTM for temporal memory
- 9-dimensional continuous action space

### 5. Simulation Hierarchy
```
EchelonEnv
    └── Sim
        ├── VoxelWorld
        ├── MechState[] (20 mechs)
        ├── Projectile[]
        └── SpatialGrid (collision optimization)
```

### 6. Generation Pipeline
```
seed → Layout → Biomes → Corridors → Validator → NavGraph
```

## Observation Modes

| Mode | Description |
|------|-------------|
| `full` | Complete voxel maps visible |
| `partial` | Sensor-limited (fog of war) |

## Action Space (9D Continuous)

| Index | Action | Range |
|-------|--------|-------|
| 0-3 | Movement (forward, strafe, vertical, yaw) | [-1, 1] |
| 4 | PRIMARY weapon (laser/flamer/paint) | [-1, 1] |
| 5 | VENT (heat dump) | [-1, 1] |
| 6 | SECONDARY (missile/ECM) | [-1, 1] |
| 7 | TERTIARY (paint/gauss/autocannon) | [-1, 1] |
| 8 | SPECIAL (smoke) | [-1, 1] |

## External Interfaces

### Visualization Server (FastAPI)
- **Protocol:** Server-Sent Events (SSE)
- **Port:** 8090 (default)
- **Features:** Replay streaming, channel subscriptions, backpressure

### W&B Integration
- Experiment tracking
- Training metrics logging
- Checkpoint management

## Test Structure

```
tests/
├── unit/           # Fast isolated tests
├── integration/    # System-level tests (convergence, golden)
├── performance/    # Benchmarks
└── benchmark/      # LOS, Nav performance
```

**Testing Approach:**
- Hypothesis for property-based testing
- Mutation testing via mutmut on critical paths (sim.py, layout.py, biomes.py)

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Directory structure | HIGH | Clear, conventional layout |
| Technology stack | HIGH | Explicit in pyproject.toml |
| Subsystem boundaries | HIGH | Well-separated packages |
| Data flow | MEDIUM | Need deeper trace through sim loop |
| Configuration patterns | HIGH | Frozen dataclasses throughout |
| Test coverage | MEDIUM | Need to run coverage report |

## Items Requiring Deeper Analysis

1. **Simulation Loop** - Exact tick/step mechanics in Sim class
2. **Observation Construction** - How env.py builds agent observations
3. **Reward Shaping** - Reward components and their weights
4. **Generation Pipeline** - Layout → Biome → Validation flow
5. **NavGraph Construction** - How walkable surfaces are extracted
6. **Self-Play Loop** - League/Match coordination
