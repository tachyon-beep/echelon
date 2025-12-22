from __future__ import annotations

import numpy as np

from ..sim.world import VoxelWorld
from ..config import WorldConfig
from .biomes import CATALOG as BIOME_CATALOG, get_biome_brush
from .corridors import carve_macro_corridors

def generate_layout(config: WorldConfig, rng: np.random.Generator) -> VoxelWorld:
    """
    Structured generation pipeline:
    1. Init empty world.
    2. Split into jittered quadrants.
    3. Assign biomes to quadrants.
    4. Fill biomes (brush pass).
    5. Carve macro corridors (skeleton pass) - done AFTER fill to guarantee connectivity.
    """
    
    # 1. Init
    voxels = np.zeros((config.size_z, config.size_y, config.size_x), dtype=np.uint8)
    world = VoxelWorld(voxels=voxels, voxel_size_m=config.voxel_size_m)
    
    # 2. Quadrant Split (Jittered)
    sx, sy = config.size_x, config.size_y
    mid_x = sx // 2
    mid_y = sy // 2
    
    # Jitter the split point slightly (e.g. +/- 10%)
    jitter_x = int(sx * 0.1)
    jitter_y = int(sy * 0.1)
    split_x = mid_x + rng.integers(-jitter_x, jitter_x + 1)
    split_y = mid_y + rng.integers(-jitter_y, jitter_y + 1)
    
    # 3. Biome Assignment
    # Pick 4 biomes from the catalog
    keys = list(BIOME_CATALOG.keys())
    # Weighted choice? For now uniform random 4 unique if possible
    chosen_biomes = rng.choice(keys, size=4, replace=(len(keys) < 4))
    
    # Assign to quadrants: TL, TR, BL, BR
    # Quadrant coords: (min_x, min_y, max_x, max_y)
    quadrants = {
        "BL": (0, 0, split_x, split_y),
        "BR": (split_x, 0, sx, split_y),
        "TL": (0, split_y, split_x, sy),
        "TR": (split_x, split_y, sx, sy),
    }
    
    # Random permutation of assignment
    q_names = list(quadrants.keys())
    rng.shuffle(q_names) # type: ignore
    
    assignment = {q_name: biome for q_name, biome in zip(q_names, chosen_biomes)}
    world.meta["biome_layout"] = assignment
    world.meta["generator"] = "layout_v2"
    
    # 4. Fill Biomes
    for q_name, biome_name in assignment.items():
        min_x, min_y, max_x, max_y = quadrants[q_name]
        brush = get_biome_brush(biome_name)
        # Apply brush
        brush(world, min_x, min_y, max_x, max_y, rng)
        
    # 5. Carve Skeleton (Handled by the caller usually, but we can do a preliminary one here?)
    # Actually, the caller (env.py) calls `carve_macro_corridors` *after* generation in the original code.
    # But in the V2 plan, `generate` should produce the "canonical" world, which includes the skeleton.
    # However, `carve_macro_corridors` depends on spawn points which are decided in `env.py`.
    # 
    # COMPROMISE:
    # We will let `env.py` handle the spawn-dependent carving (lanes).
    # But we should enforce a "Central Clearing" or generic structure here if we wanted.
    # For now, we trust the `env.py` pipeline which calls `carve_macro_corridors` immediately after `generate`.
    #
    # Wait, `env.py` calls `VoxelWorld.generate`, then `apply_transform`, then `carve_macro_corridors`.
    # This is correct. The biomes provide the "noise", the carver provides the "signal".
    
    world.ensure_ground_layer()
    return world
