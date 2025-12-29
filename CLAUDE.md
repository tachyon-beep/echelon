# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Echelon is a Deep Reinforcement Learning environment for mech tactics. It simulates 5v5 asymmetric team combat in a voxel world with physics (heat management, stability/knockdown, line-of-sight targeting). Teams are composed of "packs" of 5 mechs (1 Heavy, 2 Medium, 1 Light, 1 Scout). Two packs can form a squad (10v10) with a squad leader who can dynamically allocate units to fire/assault elements.

**Mission:** Educational sandbox that teaches DRL through the "bones of a simulated world" — mechanics that make reward shaping tractable and emergent behavior legible.

## Project Philosophy

Echelon is built with "vibe coding + rigor":

1. **Design liberally, throw away freely** — Always have a design, but redesign without hesitation when learning demands it
2. **TDD as AI guardrails** — Tests catch when agents refactor without you noticing
3. **Tiny testable changes** — Small increments; phased delivery for large features
4. **Risk/complexity caps** — No high-risk work without explicit risk reduction first
5. **No legacy code** — Pre-release means freedom to break everything. Delete old code completely; no backwards compatibility shims, no deprecation warnings

## Specialist Subagents and Skills

Use these liberally when working on relevant areas:

| Agent | Use For |
|-------|---------|
| **drl-expert** | PPO implementation, reward engineering, training stability, policy architecture, algorithm selection, RL debugging |
| **pytorch-expert** | torch.compile, FSDP/distributed training, memory profiling, tensor ops, custom kernels, performance optimization |
| **voxel-systems-specialist** | Terrain representation, procedural generation, chunk systems, navigation mesh extraction, fluid simulation, LOS/cover systems |
| **ux-specialist** | Visualization design, dashboard layout, keyboard navigation, status indicators, replay viewer UX |

| Skill Pack | Use For |
|------------|---------|
| **yzmir-deep-rl** | Algorithm selection (PPO/SAC), reward shaping, exploration-exploitation, multi-agent RL |
| **yzmir-pytorch-engineering** | PyTorch patterns, CUDA debugging, memory optimization |
| **yzmir-training-optimization** | NaN losses, gradient issues, learning rate scheduling, convergence |
| **axiom-python-engineering** | Python patterns, type systems, code review |

**Default stance:** When in doubt, spawn the specialist. The overhead is trivial compared to subtle bugs in tensor ops, RL training loops, or voxel connectivity.

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

### Lint & Type Check
```bash
uv run ruff check .          # Linting
uv run ruff format .         # Formatting
uv run mypy echelon/         # Type checking
uv run vulture echelon/      # Dead code detection
uv run pre-commit run --all-files  # Run all checks
```

**MANDATORY: You must ensure all code is lint-free and type-clean before committing. Do this now, not later.**

Run `uv run ruff check . && uv run mypy echelon/` before every commit. If errors exist, fix them immediately.

When fixing lint/type errors, use the appropriate skill:

| Error Type | Skill to Use |
|------------|--------------|
| Ruff errors | `axiom-python-engineering:systematic-delinting` |
| Mypy errors | `axiom-python-engineering:resolving-mypy-errors` |

These skills provide systematic methodologies that prevent common mistakes like over-using `# noqa` or `# type: ignore`.

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

### Arena Self-Play Training

Train against past versions of the policy plus a permanent heuristic baseline:

```bash
# Bootstrap league with Lieutenant Heuristic
uv run python scripts/arena.py bootstrap

# Train in arena mode
uv run python scripts/train_ppo.py --train-mode arena --arena-league runs/arena/league.json

# Add trained checkpoints to league
uv run python scripts/arena.py add runs/train/best.pt --kind commander
```

The arena uses PFSP (Prioritized Fictitious Self-Play) to sample opponents by rating similarity.
Lieutenant Heuristic provides a stable baseline that never retires.

### Visualization
```bash
uv run python -m echelon.server  # Replay server on port 8090
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