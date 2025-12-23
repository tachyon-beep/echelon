# Echelon

**Learn to build DRL environments by watching neural networks learn the art of war.**

## What is this?

Echelon is an educational deep reinforcement learning environment disguised as a mech tactics war game. It started as a question: *"What does an MVP for a hierarchical war game look like?"* The answer: less than a day of vibe coding with a bit of rigor.

The project serves three purposes:

- **Learn DRL by building** — See how complex concepts (heat management, target locks, pack coordination) become reward functions and observation spaces
- **Watch AI evolve tactics** — Self-play leagues with Glicko-2 ratings produce emergent strategies you can spectate
- **Understand simulation** — The "bones of the world" (physics, LOS, stability) are the real curriculum, not the voxel colors

This isn't novel research — it's a showcase of what you can build quickly with modern DRL tools and disciplined iteration.

## Quick Start

Requires [uv](https://github.com/astral-sh/uv) and Python 3.13.

```bash
# Setup
uv python install 3.13
uv sync -p 3.13

# Smoke test
uv run python scripts/smoke.py --episodes 1 --packs-per-team 1 --size 40 --mode full

# Train
uv run python scripts/train_ppo.py --packs-per-team 1 --size 100 --mode full
```

## What You'll Learn

Echelon teaches DRL through the **bones of a simulated world** — the mechanics that make reward shaping tractable and emergent behavior legible:

| Mechanic | What It Teaches |
|----------|-----------------|
| **Heat management** | Resource constraints beyond HP — overheat and you shut down |
| **Stability/knockdown** | Physics-based control disruption — get hit hard enough and you fall |
| **Line-of-sight & cover** | Spatial reasoning — voxel raycasting determines what you can see and shoot |
| **Pack coordination** | Multi-agent cooperation — paint-locks and comms are scoped to your pack |
| **Asymmetric classes** | Combined-arms dependencies — Heavies, Mediums, Lights, and Scouts play different roles |

These aren't arbitrary game rules — they're the kind of constraints that make RL problems interesting and solutions transferable.

## Watch It Fight

```bash
# Start the replay server
uv run python server.py

# Open viewer.html in your browser
```

The web viewer connects via WebSocket and renders matches in real-time.

<!-- TODO: Add Twitch link when streaming -->

## Architecture Overview

<!-- TODO: Pending architectural audit -->

```
echelon/
├── env/          # Gym-compatible environment
├── sim/          # Core simulation (physics, combat, heat, stability)
├── gen/          # Procedural map generation (layout, biomes, corridors)
├── nav/          # Navigation (2.5D NavGraph, A* pathfinding)
├── rl/           # Neural network (ActorCriticLSTM)
└── arena/        # Self-play infrastructure (Glicko-2 leagues)
```

## Development Philosophy

Echelon is built with "vibe coding + rigor":

1. **Design liberally, throw away freely** — Always have a design, but redesign without hesitation
2. **TDD as AI guardrails** — Tests catch when agents refactor without you noticing
3. **Tiny testable changes** — Small increments; phased delivery for large features
4. **No legacy code** — Pre-release means freedom to break everything; no backwards compatibility

## Roadmap

See [ROADMAP.md](ROADMAP.md) for capability tiers: what currently works, what's being tuned, and where we're headed.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version:
- Run `uv run ruff check .` and `PYTHONPATH=. uv run pytest tests` before submitting
- Keep changes small and focused
- No backwards compatibility required

## License

[MIT](LICENSE) — Copyright 2024 John Morrissey (tachyon-beep)
