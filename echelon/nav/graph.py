from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

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

    v1 Implementation (vectorized):
    - Uses NumPy broadcasting to identify walkable surfaces
    - A surface is walkable if: solid below AND clear above for clearance_z
    - Applies mech_radius footprint via erosion
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

        collides = world.collides_mask
        sz, sy, sx = collides.shape

        # --- Vectorized Node Detection ---

        # 1. Find solid voxels (potential surfaces to stand ON)
        #    We need solid at z, and clear from z+1 to z+clearance_z

        # Walkable mask: start with all False, then mark valid positions
        # A position (z, y, x) is walkable if:
        #   - collides[z, y, x] is True (solid to stand on)
        #   - collides[z+1:z+1+clearance_z, y, x] are all False (headroom)

        walkable = np.zeros((sz, sy, sx), dtype=np.bool_)

        # For each z level that could be a floor (0 to sz-2 at minimum)
        for z in range(sz - 1):
            # Check if this z is solid
            solid_here = collides[z]

            # Check clearance above
            z_top = min(z + 1 + clearance_z, sz)
            if z + 1 >= sz:
                # Top of world, can stand here if solid
                clear_above = np.ones((sy, sx), dtype=np.bool_)
            else:
                # All voxels from z+1 to z_top must be non-colliding
                clear_above = ~np.any(collides[z+1:z_top], axis=0)

            walkable[z] = solid_here & clear_above

        # 2. Also check bedrock (z=-1, standing on virtual floor below z=0)
        #    Need clearance from z=0 upward
        z_top_bedrock = min(clearance_z, sz)
        if z_top_bedrock > 0:
            clear_from_bedrock = ~np.any(collides[0:z_top_bedrock], axis=0)
        else:
            clear_from_bedrock = np.ones((sy, sx), dtype=np.bool_)

        # 3. Apply mech_radius footprint erosion
        if mech_radius > 0:
            # Erode walkable mask - a cell is only walkable if all cells
            # within mech_radius are also walkable
            from scipy.ndimage import minimum_filter
            footprint = np.ones((1, 2*mech_radius+1, 2*mech_radius+1), dtype=np.bool_)
            walkable = minimum_filter(walkable.astype(np.uint8), footprint=footprint, mode='constant', cval=0).astype(np.bool_)

            # Also erode bedrock walkability
            footprint_2d = np.ones((2*mech_radius+1, 2*mech_radius+1), dtype=np.bool_)
            clear_from_bedrock = minimum_filter(clear_from_bedrock.astype(np.uint8), footprint=footprint_2d, mode='constant', cval=0).astype(np.bool_)

        # 4. Extract node coordinates and build node dict
        # Bedrock nodes (z=-1)
        bedrock_coords = np.argwhere(clear_from_bedrock)
        for coord in bedrock_coords:
            y, x = int(coord[0]), int(coord[1])
            graph._add_node((-1, y, x), world.voxel_size_m)

        # Regular nodes
        node_coords = np.argwhere(walkable)
        for coord in node_coords:
            z, y, x = int(coord[0]), int(coord[1]), int(coord[2])
            graph._add_node((z, y, x), world.voxel_size_m)

        # --- Edge Building (still iterative over nodes) ---
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

