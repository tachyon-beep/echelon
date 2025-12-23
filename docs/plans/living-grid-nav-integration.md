---
name: living-grid-nav-integration
description: Integrated plan for Living Grid terrain and nav/planner
---

# Plan

Deliver a versioned “Living Grid” 3D terrain pipeline and a navigation-graph/local-planner stack together, so every generated map is deterministically reproducible, provably traversable with the actual movement model, and playable without agents getting stuck or pathing into terrain.

## Requirements

- Determinism: given the same seed + recipe, world voxels + validation fixups + nav graph are identical (hashable).
- Playability: blue/red spawns can reach objective using the same traversal rules the sim uses.
- Backward compatibility: legacy generator + current movement remain available via version/flags.
- Performance: generation + nav build stay within a defined budget for default map sizes.
- Debuggability: every replay can report generator version, recipe, fixups, and connectivity stats.

## Complexity & Risk Assessment

- **Living Grid pipeline (terrain + recipe + validation):** Complexity High; Risk High (core subsystem, determinism/connectivity/perf are brittle).
- **Nav graph + local planner (stairs/ramps, multi-level):** Complexity High; Risk High (nav/physics mismatch, perf scaling, training/action-space implications).
- **Combined package:** Very High complexity; High risk, but mitigable via strict versioning/flags, golden seeds + hash provenance, and a staged rollout starting with heuristic-only nav before touching RL behavior.

## Scope

- In: versioned terrain pipeline, recipe schema + reconstruction, 2.5D→3D connectivity validation, nav graph builder, local planner, optional nav-assist for movement.
- Out: drones, full destructibility/falling/fluids, major RL action-space redesign (unless explicitly chosen later).

## Files and Entry Points

- Terrain: `echelon/sim/world.py`, `echelon/gen/*`, `echelon/env/env.py`, `echelon/gen/validator.py`, `echelon/gen/recipe.py`, `echelon/gen/transforms.py`
- Movement: `echelon/sim/sim.py`, `echelon/sim/mech.py`
- New: `echelon/nav/*` (graph builder, planner, costs, caching)
- Debug/tools: `scripts/terrain_audit.py`, `scripts/reproduce_from_replay.py`, optionally `viewer.html`

## Data Model / API Changes

- Add version/feature flags:
  - `WorldConfig.generator_version` (e.g. `"v1_legacy"`, `"v2_living_grid"`)
  - `EnvConfig.nav_mode` (e.g. `"off"`, `"assist"`, `"planner"` for heuristic)
- Formalize traversal semantics used by both validator + nav:
  - “walkable node” definition, step/ledge rules, per-class vertical traversal capability (jets vs stairs).
- Extend `world.meta["recipe"]` and/or `build_recipe(...)` to include:
  - generator id/version, transform, fixups, connectivity stats, and a nav-build signature/hash.

## Action Items

- [ ] Audit + freeze current behavior: document current `VoxelWorld.generate`, current fixups (`ConnectivityValidator`), and current movement/collision constraints as the baseline contract.
- [ ] Version the generator: wrap existing terrain as `v1_legacy` and introduce a `v2` pipeline interface that always emits a recipe + deterministic meta + voxel hash.
- [ ] Recipe reconstruction: implement “build world from recipe” path and add a determinism harness (seed → recipe → voxels hash) with golden seeds.
- [ ] Nav graph v0 (2.5D): add `echelon/nav/graph.py` that builds a traversability grid from current voxels (treat SOLID/SOLID_DEBRIS blocked; hazards as cost), and exposes spawn→objective path queries.
- [ ] Local planner v0: implement A* over the nav graph + waypoint-following controller; wire it into `HeuristicPolicy` first (no RL changes yet).
- [ ] Nav-assist (optional, minimal): add an env flag that post-processes movement intents to reduce “walk into wall” failures (e.g., waypoint projection / wall-slide), keeping the RL action space unchanged.
- [ ] Unify validation with nav: use the nav graph as the source of truth for connectivity validation and fixups; make fixups write to `world.meta["fixups"]` deterministically.
- [ ] Introduce multi-level primitives carefully: extend `v2` generator with a small set of 3D-safe features (ramps/stairs first), and extend nav graph to 3D nodes/edges that exactly match traversal semantics.
- [ ] Align sim movement with traversal: either (a) extend movement/collision to support the same “step/ramp” transitions the nav allows, or (b) constrain generator primitives to what movement already supports; gate by `generator_version`.
- [ ] Debug + observability: add `scripts/terrain_audit.py` outputs (voxel hash, recipe hash, fixups, path lengths/costs, nav node/edge counts) and optional viewer overlays for paths/waypoints.
- [ ] Rollout plan: ship `v2` + nav behind flags; then flip defaults once determinism + connectivity + “stuck rate” targets are met.

## Testing and Validation

- Determinism: golden seeds where `recipe.hashes.solid` and nav signature match across runs.
- Connectivity: property-based “spawn↔objective reachable” for random seeds per generator version.
- Movement regression: scenario tests for “planner can reach objective without collisions” in simplified maps.
- Performance: `tests/performance` benchmarks for generation + nav build time at standard sizes.

## Risks and Edge Cases

- Mismatch between nav “walkable” and sim AABB collision/vertical physics (most common source of “reachable in theory, stuck in practice”).
- Determinism drift from RNG coupling or non-stable fixup ordering.
- Nav build/pathfinding cost scaling with map size and 3D complexity; need caching and frequency limits.
- Replay/schema churn and large metadata payloads if nav data is serialized naïvely.

## Open Questions

- Should RL remain continuous + nav-assist, or move to goal/waypoint actions (bigger training change)?
- What vertical traversal should exist for each class (stairs for all vs jets for some vs no vertical for heavies)?
- What are acceptable budgets (e.g., max generation+validation time at `100x100x20`, max per-step planner updates)?
