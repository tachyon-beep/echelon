# ADR-001: Why Voxels

**Date:** 2024-12-24
**Status:** Accepted

## Context

Echelon needs a terrain representation that supports:
1. **Line-of-sight (LOS) queries** — Can mech A see mech B?
2. **Cover mechanics** — Partial occlusion, peeking, hull-down positions
3. **Destructible terrain** — Explosions create craters, buildings collapse
4. **Verticality** — Multi-level combat, bridges, tunnels, knockback into voids
5. **Procedural generation** — Deterministic, seed-reproducible map creation
6. **Navigation** — Pathfinding over complex 3D geometry

Traditional approaches (heightmaps, mesh-based terrain) handle some of these well but struggle with true 3D features like tunnels, overhangs, and multi-story structures.

## Decision

Use a **3D voxel grid** as the primary terrain representation.

- Each voxel is a discrete cell with a material type (AIR, SOLID, GLASS, FOLIAGE, WATER, LAVA, DEBRIS, REINFORCED)
- Grid uses `[z, y, x]` indexing for cache-friendly vertical slices
- Per-voxel HP enables granular destruction
- Navigation built on top via "walkable air" detection (AIR with SOLID floor)

## Consequences

### Pros

- **True 3D simulation** — Tunnels, bridges, overhangs, multi-story buildings are native, not special-cased
- **Uniform LOS** — Raycasting through a regular grid is simple and fast (DDA algorithm, Numba JIT)
- **Destructibility is trivial** — Change voxel type, decrement HP, done
- **Procedural generation is composable** — Biome brushes paint regions, validators ensure connectivity
- **RL-friendly observations** — Grid structure maps naturally to CNN encoders or flattened vectors
- **Deterministic** — Integer grid + seed = perfectly reproducible maps

### Cons

- **Memory scaling** — Dense 3D arrays grow as O(n³); mitigated by keeping arena sizes modest (50-100 voxels per axis)
- **Visual fidelity** — "Blocky" aesthetic; acceptable for training, requires smoothing for spectator client
- **Navigation complexity** — 2.5D NavGraph extraction required; can't use off-the-shelf navmesh tools
- **No continuous surfaces** — Slopes/ramps are staircases; physics feel "chunky"

## Alternatives Considered

### Heightmap + Obstacle Meshes

Standard game dev approach. Rejected because:
- No native support for tunnels, overhangs, bridges
- Destruction requires complex mesh deformation
- LOS through complex meshes is expensive

### Sparse Octree

Better memory scaling for large sparse worlds. Rejected because:
- Added complexity for modest arena sizes (100³ is ~4MB dense, acceptable)
- Octree traversal complicates raycasting hot path
- Premature optimization for current scale

### Continuous Physics (No Grid)

Full PhysX/bullet simulation. Rejected because:
- Non-deterministic across platforms
- Harder to generate procedurally
- LOS requires actual raycasts against mesh colliders (slower)
- Overkill for tactics-scale simulation

## References

- Minecraft (voxel terrain, destruction, procedural generation)
- Teardown (voxel destruction physics)
- Space Engineers (structural integrity on voxel grids)
- DDA raycasting algorithm for voxel traversal
