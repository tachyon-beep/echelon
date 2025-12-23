# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Echelon is a Deep Reinforcement Learning environment for mech tactics. It simulates 10v10 asymmetric team combat in a voxel world with physics (heat management, stability/knockdown, line-of-sight targeting). Teams are composed of "packs" of 10 mechs (1 Heavy, 5 Medium, 3 Light, 1 Scout).

## Commands

### Setup
```bash
uv python install 3.13
uv sync -p 3.13
```

### Testing
```bash
# All tests
PYTHONPATH=. uv run pytest tests

# Unit tests (fast)
PYTHONPATH=. uv run pytest tests/unit

# Single test file
PYTHONPATH=. uv run pytest tests/unit/test_mechanics.py

# Single test
PYTHONPATH=. uv run pytest tests/unit/test_mechanics.py::test_name -v

# Integration tests (slower, check convergence)
PYTHONPATH=. uv run pytest tests/integration

# Performance benchmarks
PYTHONPATH=. uv run pytest tests/performance
```

### Lint
```bash
uv run ruff check .
```

### Smoke Test
```bash
uv run python scripts/smoke.py --episodes 1 --packs-per-team 1 --size 40 --mode full
```

### Training
```bash
# Basic PPO training
uv run python scripts/train_ppo.py --packs-per-team 1 --size 100 --mode full

# With W&B tracking
uv run python scripts/train_ppo.py --wandb --wandb-run-name "experiment-01"

# Resume training
uv run python scripts/train_ppo.py --run-dir runs/train --resume latest --updates 200
```

### Visualization
```bash
uv run server.py  # Replay server on port 8090
# Open viewer.html in browser
```

### Mutation Testing
```bash
uv run mutmut run  # Targets sim.py, layout.py, biomes.py
```

## Architecture

### Core Modules

- **`echelon/env/env.py`**: Gym-like environment (`EchelonEnv`) with `reset()`/`step()`. Manages 20 agents (10 blue, 10 red per pack).

- **`echelon/sim/sim.py`**: Core simulation loop. Fixed timestep with `dt_sim` and `decision_repeat`. Handles physics, combat, heat, stability.

- **`echelon/sim/world.py`**: VoxelWorld - 3D grid (`[z, y, x]` order) with material types (SOLID, LAVA, WATER, GLASS, FOLIAGE, DEBRIS). Per-voxel HP for destruction.

- **`echelon/gen/`**: Procedural content generation pipeline:
  - `layout.py`: Quadrant-based blueprint with jittered split
  - `biomes.py`: Fill functions (urban, industrial, forest, etc.)
  - `validator.py`: ConnectivityValidator ensures playability via A* + dig fixups
  - `recipe.py`: Deterministic map reproduction from seed

- **`echelon/nav/`**: Navigation system (v0, 2.5D):
  - `graph.py`: NavGraph scans walkable surfaces, connects via step/traverse rules
  - `planner.py`: A* pathfinding over NavGraph

- **`echelon/rl/model.py`**: ActorCriticLSTM - encoder + LSTM core + actor/critic heads

- **`echelon/arena/`**: Self-play infrastructure with Glicko-2 ratings

### Action Space

9-dimensional continuous (`echelon/actions.py`):
- `[0-3]` Movement: forward, strafe, vertical, yaw_rate
- `[4]` PRIMARY: Laser/Flamer/Paint (class-dependent)
- `[5]` VENT: Heat dump
- `[6]` SECONDARY: Missile/ECM toggle
- `[7]` TERTIARY: Paint/Gauss/Autocannon
- `[8]` SPECIAL: Smoke

### Configuration

`echelon/config.py` defines:
- `WorldConfig`: Voxel grid size, obstacle fill, connectivity params
- `MechClassConfig`: Per-class stats (speed, HP, heat capacity)
- `WeaponSpec`: Weapon definitions (range, damage, cooldown, guidance)
- `EnvConfig`: Sim timing, observation mode, feature toggles

### Key Patterns

**Determinism**: Maps reproducible via seed + recipe hash. Critical for RL benchmarking.

**Generation Pipeline**: Layout → Biomes → Corridor Carving → Connectivity Validation → Fixups

**Observation Modes**: `full` (complete voxel maps) or `partial` (sensor-limited)

**Pack Mechanics**: Paint locks and comms are pack-scoped, not team-wide

## Development Notes

- Line length: 110 (ruff)
- Type hints used extensively
- Simulation logic (`sim.py`) is mutation-tested for robustness
- If sandboxed CI blocks `~/.cache/uv`, set `UV_CACHE_DIR=.uv_cache`