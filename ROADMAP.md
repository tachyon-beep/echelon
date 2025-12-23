# Roadmap

Echelon evolves through live playtesting — like a TCG R&D team, mechanics are tuned based on observed play, not predetermined specs.

**All game mechanics are permanently subject to change.** If something dominates too hard or creates degenerate strategies, it gets rebalanced. The simulation "bones" (physics, LOS, heat) are stable foundations; the "flesh" (damage numbers, cooldowns, pack composition) is always negotiable.

This roadmap reflects capability tiers, not promises.

## Currently Works

Core capabilities that are functional and demonstrable:

- **Voxel world** — 3D grid with materials (solid, lava, water, glass, foliage, debris), per-voxel HP for destruction
- **Mech simulation** — Heat management, stability/knockdown, continuous movement with fixed timestep
- **Combat** — Hitscan lasers, projectile missiles, LOS raycasting (Numba JIT optimized)
- **Procedural maps** — Layout → biomes → corridors → connectivity validation with A* fixups
- **PPO + LSTM training** — Per-mech recurrent policies with partial observability support
- **Self-play arena** — Glicko-2 rated league, opponent sampling, snapshot promotion to hall-of-fame
- **Web visualization** — Real-time replay viewer via WebSocket

## Being Tuned

Point-in-time implementations — expect these to change frequently:

- **Weapon balance** — Damage/cooldown/range ratios across lasers, missiles, gauss, autocannon
- **Heat curves** — Dissipation rates, overheat thresholds, shutdown duration
- **Stability mechanics** — Knockdown triggers, recovery rates, fallen-state duration
- **Pack composition** — Class ratios (currently 1 Heavy, 5 Medium, 3 Light, 1 Scout)
- **Reward shaping** — Kill/damage/objective weightings, survival bonuses
- **Map generation** — Biome distributions, obstacle density, corridor widths

## Future Vision

**Where the AI becomes the designer:**

- **Commander AI** — Agents allocate points to build formations, not hardcoded packs
- **Architecture search** — AI discovers interesting subsystem configurations
- **Expanded mech tree** — More classes, varied loadouts, specialization paths
- **Hierarchical control** — Squad-level and company-level decision layers
- **Narrative layer** — Named commanders with emergent "personalities" from play style metrics

**Helping you understand and engage with the system:**

- **Download and run at home** — One-command setup, train your own commanders, watch them fight
- **Training telemetry dashboard** — Live metrics, reward breakdowns, policy divergence tracking
- **Replay annotation** — "Why did the AI do that?" explanations for key decision moments
- **Curriculum walkthroughs** — Guided tours of each simulation mechanic
- **Experiment templates** — Pre-configured ablations to explore "what if" questions
- **Interactive tuning** — Tweak mechanics mid-match and watch adaptation in real-time
