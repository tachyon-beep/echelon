# Project Professionalization Design

**Date:** 2024-12-24
**Status:** Approved

## Overview

Professionalize Echelon for GitHub publication, establishing it as an educational DRL sandbox that produces entertaining emergent AI behavior.

## Project Identity

**Tagline:** "Learn to build DRL environments by watching neural networks learn the art of war."

**Origin hook:** "What does an MVP hierarchical war game look like? Less than a day of vibe coding with a bit of rigor."

**Three Value Axes:**
1. **Pedagogical** — How to translate complex concepts into reward functions; the "bones of the world"
2. **Entertainment** — Self-play leagues, emergent tactics, named commanders (Twitch audience)
3. **Simulation** — Heat, stability, LOS, pack coordination as teachable mechanics

**Platform Strategy:**
- GitHub: A (DRL practitioners) → C (simulation learners) → B (game enthusiasts)
- Twitch: B → C → A

## Files to Create/Update

### README.md (rewrite)
Structure:
1. Tagline + What is this (origin story, three axes)
2. Quick Start
3. What You'll Learn (simulation bones)
4. Watch It Fight (Twitch/visualization hook)
5. Architecture Overview (TODO pending audit)
6. Development Philosophy (B-level, brief)
7. Roadmap link
8. Contributing / License

### ROADMAP.md (new)
Capability tiers approach:
- **Currently Works** — Functional, demonstrable capabilities
- **Being Tuned** — Point-in-time implementations, expect frequent changes
- **Future Vision** — AI-as-designer, hierarchy, engagement tooling

Key framing: "All game mechanics are permanently subject to change."

### CLAUDE.md (update)
Add:
- Project Philosophy section (5 rigor principles)
- Specialist Subagents table (drl-expert, pytorch-expert, voxel-systems-specialist, elspeth-ux-specialist)
- Skill Packs table
- No legacy code stance

### LICENSE (new)
MIT License, Copyright John Morrissey (tachyon-beep)

### CONTRIBUTING.md (new, lightweight)
- Setup instructions
- Test commands
- Code style (ruff, line-length 110)
- PR expectations
- Bugs go in docs/bugs/
- Contact: john[at]foundryside.dev

### SECURITY.md (new, boilerplate)
Standard "educational project, no secrets, open issues" policy

### CODE_OF_CONDUCT.md (new)
Contributor Covenant 2.1

### pyproject.toml (update)
- version: 0.1.0
- description: updated
- license: MIT
- authors: John Morrissey
- keywords: RL, DRL, pytorch, game-ai, simulation, voxel, multi-agent, self-play, educational
- classifiers: Alpha, Education, Science/Research, AI, Simulation
- project.urls: Homepage, Repository, Issues

### .gitignore (update)
Add: IDE files, OS files, type checking cache, expanded Python patterns

## Design Decisions

1. **Methodology as secondary theme (B)** — Present in README but not leading; code quality speaks for itself
2. **No legacy code** — Hard rule for pre-release project
3. **Capability tiers for roadmap** — Honest about state, no false promises
4. **Bugs in docs/** — No GitHub issue templates for now
5. **Contact via email** — john[at]foundryside.dev
