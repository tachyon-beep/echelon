from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import WorldConfig


@dataclass
class VoxelWorld:
    solid: np.ndarray  # bool[sz, sy, sx] (z-major)
    voxel_size_m: float = 5.0

    @property
    def size_z(self) -> int:
        return int(self.solid.shape[0])

    @property
    def size_y(self) -> int:
        return int(self.solid.shape[1])

    @property
    def size_x(self) -> int:
        return int(self.solid.shape[2])

    def in_bounds(self, ix: int, iy: int, iz: int) -> bool:
        return 0 <= ix < self.size_x and 0 <= iy < self.size_y and 0 <= iz < self.size_z

    def is_solid_index(self, ix: int, iy: int, iz: int) -> bool:
        if not self.in_bounds(ix, iy, iz):
            return True
        return bool(self.solid[iz, iy, ix])

    def set_box_solid(
        self,
        min_ix: int,
        min_iy: int,
        min_iz: int,
        max_ix_excl: int,
        max_iy_excl: int,
        max_iz_excl: int,
        value: bool,
    ) -> None:
        min_ix = max(min_ix, 0)
        min_iy = max(min_iy, 0)
        min_iz = max(min_iz, 0)
        max_ix_excl = min(max_ix_excl, self.size_x)
        max_iy_excl = min(max_iy_excl, self.size_y)
        max_iz_excl = min(max_iz_excl, self.size_z)
        if min_ix >= max_ix_excl or min_iy >= max_iy_excl or min_iz >= max_iz_excl:
            return
        self.solid[min_iz:max_iz_excl, min_iy:max_iy_excl, min_ix:max_ix_excl] = value

    def aabb_collides(self, aabb_min: np.ndarray, aabb_max: np.ndarray) -> bool:
        # Treat out-of-bounds as solid.
        if np.any(aabb_min < 0) or aabb_max[0] > self.size_x or aabb_max[1] > self.size_y or aabb_max[2] > self.size_z:
            return True

        min_ix = int(np.floor(aabb_min[0]))
        min_iy = int(np.floor(aabb_min[1]))
        min_iz = int(np.floor(aabb_min[2]))
        max_ix = int(np.ceil(aabb_max[0]))
        max_iy = int(np.ceil(aabb_max[1]))
        max_iz = int(np.ceil(aabb_max[2]))
        min_ix = max(min_ix, 0)
        min_iy = max(min_iy, 0)
        min_iz = max(min_iz, 0)
        max_ix = min(max_ix, self.size_x)
        max_iy = min(max_iy, self.size_y)
        max_iz = min(max_iz, self.size_z)

        region = self.solid[min_iz:max_iz, min_iy:max_iy, min_ix:max_ix]
        return bool(np.any(region))

    @classmethod
    def generate(cls, config: WorldConfig, rng: np.random.Generator) -> VoxelWorld:
        solid = np.zeros((config.size_z, config.size_y, config.size_x), dtype=bool)
        world = cls(solid=solid, voxel_size_m=config.voxel_size_m)

        # Helper to clear a path
        def carve_line(x0, y0, x1, y1, width_vox):
            dist = np.hypot(x1 - x0, y1 - y0)
            steps = int(dist * 1.5)
            half_w = width_vox / 2.0
            for i in range(steps + 1):
                t = i / max(1, steps)
                px = x0 + (x1 - x0) * t
                py = y0 + (y1 - y0) * t
                min_x, max_x = int(px - half_w), int(px + half_w + 1)
                min_y, max_y = int(py - half_w), int(py + half_w + 1)
                world.set_box_solid(min_x, min_y, 0, max_x, max_y, 4, False)

        # Select Archetype
        archetype = rng.integers(0, 3)
        # archetype = 0 # Force Citadel (Debug)
        
        # Common Scatter Config
        cx, cy = config.size_x // 2, config.size_y // 2
        
        if archetype == 0: # CITADEL (Central Keep + X Roads)
            # Dense Central Keep
            keep_radius = max(5, int(min(config.size_x, config.size_y) * 0.15))
            for _ in range(12): 
                sx, sy, sz = rng.integers(3, 10), rng.integers(3, 10), rng.integers(3, max(4, config.size_z // 2))
                ix = int(rng.integers(cx - keep_radius, cx + keep_radius - sx + 1))
                iy = int(rng.integers(cy - keep_radius, cy + keep_radius - sy + 1))
                world.set_box_solid(ix, iy, 0, ix + sx, iy + sy, sz, True)
            
            # Roads: X-Shape
            road_width = 8.0
            carve_line(0, 0, cx, cy, road_width)
            carve_line(config.size_x, 0, cx, cy, road_width)
            carve_line(0, config.size_y, cx, cy, road_width)
            carve_line(config.size_x, config.size_y, cx, cy, road_width)
            
        elif archetype == 1: # URBAN GRID (Regular Blocks)
            # Create a grid of city blocks
            block_size = 12
            street_width = 6
            step = block_size + street_width
            
            for y in range(0, config.size_y, step):
                for x in range(0, config.size_x, step):
                    # For each "city block", fill it with buildings
                    # Chance to leave empty for open plaza
                    if rng.random() > 0.8: continue
                    
                    # Fill block with 1-3 buildings
                    for _ in range(rng.integers(1, 4)):
                        bx = x + rng.integers(0, 4)
                        by = y + rng.integers(0, 4)
                        bw = rng.integers(4, block_size - 2)
                        bh = rng.integers(4, block_size - 2)
                        bz = rng.integers(3, max(5, config.size_z // 2))
                        world.set_box_solid(bx, by, 0, bx + bw, by + bh, bz, True)

        elif archetype == 2: # HIGHWAY (Diagonal Split)
            # Main heavy diagonal road
            road_width = 12.0
            carve_line(0, 0, config.size_x, config.size_y, road_width)
            
            # Heavy industrial noise everywhere else
            num_blocks = int(config.size_x * config.size_y * 0.02) # Dense
            for _ in range(num_blocks):
                sx, sy = rng.integers(2, 6), rng.integers(2, 6)
                sz = rng.integers(2, 8)
                ix = rng.integers(0, config.size_x - sx)
                iy = rng.integers(0, config.size_y - sy)
                # Don't block the highway (approx check)
                # Distance from point to line x=y
                dist = abs(ix - iy) / 1.414
                if dist > road_width / 2 + 2:
                    world.set_box_solid(ix, iy, 0, ix + sx, iy + sy, sz, True)

        # General Scatter (Applied to all maps to add cover to open areas)
        sparse_fill = config.obstacle_fill * 0.3
        target_solids = int(sparse_fill * config.size_x * config.size_y * max(1, config.size_z // 3))
        placed = 0
        attempts = 0
        while placed < target_solids and attempts < target_solids * 50:
            attempts += 1
            sx, sy, sz = rng.integers(2, 5), rng.integers(2, 5), rng.integers(2, 5)
            ix = int(rng.integers(0, max(1, config.size_x - sx)))
            iy = int(rng.integers(0, max(1, config.size_y - sy)))
            
            # Re-carve roads after scatter to ensure they stay clear? 
            # Better to just check collision with existing solid?
            # For simplicity, we just add scatter. 
            # If it blocks a road, that's "debris". 
            # But for Citadel/Highway, we really want clear roads.
            
            before = np.count_nonzero(world.solid[0:sz, iy:iy+sy, ix:ix+sx])
            world.set_box_solid(ix, iy, 0, ix + sx, iy + sy, sz, True)
            after = np.count_nonzero(world.solid[0:sz, iy:iy+sy, ix:ix+sx])
            placed += int(after - before)

        # Final cleanup: Re-Carve main roads to ensure they aren't blocked by scatter
        if archetype == 0:
            road_width = 8.0
            carve_line(0, 0, cx, cy, road_width)
            carve_line(config.size_x, 0, cx, cy, road_width)
            carve_line(0, config.size_y, cx, cy, road_width)
            carve_line(config.size_x, config.size_y, cx, cy, road_width)
        elif archetype == 2:
            carve_line(0, 0, config.size_x, config.size_y, 12.0)

        return world

