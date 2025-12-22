# Navigation Graph v0 Architecture

**Status:** Implemented (Initial)
**Date:** 2025-12-22
**Implementation:** `echelon/nav/`

## Overview

The NavGraph v0 is a 2.5D graph representation of the voxel world, designed to replace ad-hoc A* grids and support future true-3D navigation. It scans the `VoxelWorld` to identify "Walkable Surfaces" and connects them based on movement rules (step height, drop height).

## Core Concepts

### Nodes: Walkable Surfaces
A node represents a safe place to stand. It is identified by a `NodeID` tuple `(z, y, x)`.
- `z`: The index of the **solid voxel underneath the feet**. 
  - If `z=-1`, the surface is the bedrock at world `z=0.0`.
  - If `z=5`, the surface is on top of the voxel at index 5, meaning world `z=30.0` (assuming 5m voxels).
- `y, x`: The grid coordinates.

### Edges: Reachability
An edge connects two nodes if a mech can move between them directly.
- **Horizontal**: Adjacent XY, same Z.
- **Step Up**: Adjacent XY, Z difference <= `step_height_max` (currently 1 voxel).
- **Step Down**: Adjacent XY, Z difference >= `-step_height_max`.
- **Cost**: Based on Euclidean distance, with a penalty for climbing.

## Components

### `echelon.nav.graph.NavGraph`
- **Builder**: `NavGraph.build(world)` scans columns to find surfaces and links neighbors.
- **Storage**: `self.nodes` maps `NodeID` -> `NavNode` (position, edges).
- **Spatial Index**: `self.nodes_by_col` maps `(y, x)` -> `[z, ...]` for fast nearest-neighbor lookups.

### `echelon.nav.planner.Planner`
- Implements A* search over the generic `NavGraph`.
- Returns `path` (list of NodeIDs) and `stats` (cost, visited count).

## Usage

```python
from echelon.nav.graph import NavGraph
from echelon.nav.planner import Planner

# 1. Build Graph
graph = NavGraph.build(world)

# 2. Find Nodes
start_node = graph.get_nearest_node(start_pos)
goal_node = graph.get_nearest_node(goal_pos)

# 3. Plan
planner = Planner(graph)
path, stats = planner.find_path(start_node, goal_node)

if stats.found:
    print(f"Path found: {len(path)} steps")
```

## Future Roadmap (v1+)
- **True 3D**: Support overhangs and tunnels fully (already partially supported by the node ID schema).
- **Jump Links**: Special edges for Jump Jet capable mechs.
- **Hierarchical Pathfinding**: HPA* for large maps.
- **Dynamic Updates**: Re-patch graph when terrain is destroyed.
