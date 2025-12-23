# Echelon — AI Agent Guidelines

This file provides guidance for AI coding assistants working on this repository.

## First Steps

**On session start, read these files for project context:**

1. **`README.md`** — Project overview, three value axes, quick start
2. **`ROADMAP.md`** — Capability tiers (works / being tuned / future vision)
3. **`docs/GLOSSARY.md`** — Domain terms, acronyms, mech classes

## Project Overview

Echelon is an educational deep reinforcement learning environment for mech tactics. It simulates 10v10 asymmetric team combat in a voxel world with physics (heat management, stability/knockdown, line-of-sight targeting).

**Mission:** Educational sandbox that teaches DRL through the "bones of a simulated world" — mechanics that make reward shaping tractable and emergent behavior legible.

## Project Philosophy

1. **Design liberally, throw away freely** — Always have a design, but redesign without hesitation
2. **TDD as guardrails** — Tests catch when you refactor without realizing
3. **Tiny testable changes** — Small increments; phased delivery for large features
4. **Risk/complexity caps** — No high-risk work without explicit risk reduction first
5. **No legacy code** — Pre-release means freedom to break everything; no backwards compatibility

## Commands

Use `uv` for all Python operations.

### Setup
```bash
uv python install 3.13
uv sync -p 3.13
```

### Testing
```bash
PYTHONPATH=. uv run pytest tests           # All tests
PYTHONPATH=. uv run pytest tests/unit      # Fast unit tests
PYTHONPATH=. uv run pytest tests/integration  # Slower integration tests
PYTHONPATH=. uv run pytest tests/performance  # Benchmarks
```

### Lint & Type Check
```bash
uv run ruff check .           # Linting
uv run ruff format .          # Formatting
uv run mypy echelon/          # Type checking
uv run pre-commit run --all-files  # Run all checks
```

**Run lint and type checks before every commit.**

### Smoke Test
```bash
uv run python scripts/smoke.py --episodes 1 --packs-per-team 1 --size 40 --mode full
```

### Training
```bash
uv run python scripts/train_ppo.py --packs-per-team 1 --size 100 --mode full
```

## Architecture

```
echelon/
├── env/          # Gym-compatible environment
├── sim/          # Core simulation (physics, combat, heat, stability)
├── gen/          # Procedural map generation (layout, biomes, corridors)
├── nav/          # Navigation (2.5D NavGraph, A* pathfinding)
├── rl/           # Neural network (ActorCriticLSTM)
└── arena/        # Self-play infrastructure (Glicko-2 leagues)
```

## Key Documentation

| Document | Purpose |
|----------|---------|
| `docs/adr/` | Architecture Decision Records — why things are the way they are |
| `docs/GLOSSARY.md` | Domain terms and acronyms |
| `docs/RISK_REGISTER.md` | High-risk items requiring mitigation |
| `docs/bugs/` | Bug reports, enhancements, jank tracking |

## Development Notes

- **Line length:** 110 (ruff)
- **Type hints:** Used extensively
- **Python version:** 3.13 only
- **No legacy code:** Delete old code completely; no backwards compatibility shims
- If sandboxed CI blocks `~/.cache/uv`, set `UV_CACHE_DIR=.uv_cache`
