from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

import numpy as np

from ..sim.world import VoxelWorld

# Type alias for a node ID: (z, y, x)
NodeID = Tuple[int, int, int]

@dataclass
class NavNode:
    id: NodeID
    pos: Tuple[float, float, float]  # World center of the walkable surface
    edges: List[Tuple[NodeID, float]] = field(default_factory=list) # List of (neighbor_id, cost)

class NavGraph:
    """
    A graph representation of the walkable surfaces in the VoxelWorld.
    
    v0 Implementation (2.5D):
    - Scans the voxel grid to identify "Walkable Surfaces".
    - A surface is a SOLID voxel with AIR above it.
    - Edges connect adjacent surfaces if they are reachable (step up/down).
    """
    def __init__(self, clearance_z: int = 2):
        self.nodes: Dict[NodeID, NavNode] = {}
        self.nodes_by_col: Dict[Tuple[int, int], List[int]] = {} # (y, x) -> [z, z...]
        self.clearance_z = clearance_z
        self.voxel_size = 5.0 # Should match world config, passed in build?

    @classmethod
    def build(cls, world: VoxelWorld, clearance_z: int = 4, mech_radius: int = 1) -> NavGraph:
        graph = cls(clearance_z=clearance_z)
        graph.voxel_size = world.voxel_size_m
        
        # 1. Identify Nodes (Walkable Surfaces)
        sx, sy, sz = world.size_x, world.size_y, world.size_z
        solid = (world.voxels == VoxelWorld.SOLID) | (world.voxels == VoxelWorld.KILLED_HULL)
        
        # Obstacle Inflation (Footprint check)
        # A cell is blocked if it is solid OR if it's too close to a solid (mech width).
        def _is_blocked_xy(x: int, y: int, z_floor: int) -> bool:
            # Stand on z_floor, check clearance from z_floor+1 to z_floor+clearance
            # Check footprint: square of radius mech_radius
            for dy in range(-mech_radius, mech_radius + 1):
                for dx in range(-mech_radius, mech_radius + 1):
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < sx and 0 <= ny < sy):
                        return True # OOB is blocked
                    
                    # Check vertical headroom
                    for k in range(1, clearance_z + 1):
                        iz = z_floor + k
                        if iz >= sz: break
                        if solid[iz, ny, nx]:
                            return True
            return False

        # Iterate all columns
        for y in range(sy):
            for x in range(sx):
                # Check bedrock layer
                is_clear = not _is_blocked_xy(x, y, -1)
                if is_clear:
                    graph._add_node((-1, y, x), world.voxel_size_m)
                    
                # Check voxel layers
                for z in range(sz - 1):
                    if not solid[z, y, x]:
                        continue
                    if not _is_blocked_xy(x, y, z):
                        graph._add_node((z, y, x), world.voxel_size_m)

        # 2. Build Edges
        # Neighbor offsets (dy, dx, cost_mult)
        neighbors = [
            (0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
        ]
        
        step_height_max = 2
            
        for (z, y, x), node in graph.nodes.items():
            for dy, dx, dist_mult in neighbors:
                ny, nx = y + dy, x + dx
                
                potential_zs = graph.nodes_by_col.get((ny, nx))
                if not potential_zs:
                    continue
                
                # Diagonal Corner-Cutting Check:
                # If moving diagonally (dy!=0 and dx!=0), check the two adjacent orthogonals.
                # If either is blocked at the CURRENT Z, we can't cut the corner.
                if dy != 0 and dx != 0:
                    ortho1 = (z, y + dy, x)
                    ortho2 = (z, y, x + dx)
                    if ortho1 not in graph.nodes or ortho2 not in graph.nodes:
                        # Optimization: This is strict (requires same Z). 
                        # A better check would be 'is traversable'.
                        # But for 2.5D this is a good safety.
                        continue

                for nz in potential_zs:
                    dz = nz - z
                    if abs(dz) <= step_height_max:
                        base_cost = world.voxel_size_m * dist_mult
                        if dz > 0: base_cost *= 1.2
                        node.edges.append(((nz, ny, nx), base_cost))

        return graph

    def _add_node(self, nid: NodeID, scale: float):
        z, y, x = nid
        # World Z of the surface is (z + 1) * scale
        # If z=-1, surface is at 0.0
        wz = (z + 1) * scale
        wx = (x + 0.5) * scale
        wy = (y + 0.5) * scale
        self.nodes[nid] = NavNode(nid, (wx, wy, wz))
        
        if (y, x) not in self.nodes_by_col:
            self.nodes_by_col[(y, x)] = []
        self.nodes_by_col[(y, x)].append(z)

    def get_nearest_node(self, pos: Tuple[float, float, float]) -> Optional[NodeID]:
        """Find the nearest graph node to a world position."""
        scale = self.voxel_size
        ix = int(pos[0] // scale)
        iy = int(pos[1] // scale)
        wz = pos[2] # World Z
        
        best_node: Optional[NodeID] = None
        best_dist_sq = float('inf')
        
        search_r = 1
        for dy in range(-search_r, search_r + 1):
            for dx in range(-search_r, search_r + 1):
                ny, nx = iy + dy, ix + dx
                
                zs = self.nodes_by_col.get((ny, nx))
                if not zs: continue
                
                for z in zs:
                    node = self.nodes[(z, ny, nx)]
                    # Simple euclidean dist sq
                    d_sq = (node.pos[0] - pos[0])**2 + (node.pos[1] - pos[1])**2 + (node.pos[2] - wz)**2
                    
                    # Bias towards nodes at similar Z height (don't pick the bridge overhead)
                    dz = abs(node.pos[2] - wz)
                    if dz > scale * 1.5:
                        d_sq += 1000.0 # Heavy penalty for Z mismatch
                    
                    if d_sq < best_dist_sq:
                        best_dist_sq = d_sq
                        best_node = node.id
        
        return best_node

    def to_dict(self) -> dict:
        """Serialize graph to a JSON-safe dictionary."""
        out_nodes = []
        for nid, node in self.nodes.items():
            out_nodes.append({
                "id": list(nid),
                "pos": list(node.pos),
                "edges": [list(e[0]) for e in node.edges]
            })
        return {
            "voxel_size": self.voxel_size,
            "clearance_z": self.clearance_z,
            "nodes": out_nodes
        }

