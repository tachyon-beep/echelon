from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import WorldConfig


class _SolidMask:
    def __init__(self, world: VoxelWorld) -> None:
        self._world = world

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._world.voxels.shape

    @property
    def dtype(self) -> type[np.bool_]:
        return np.bool_

    @property
    def ndim(self) -> int:
        return int(self._world.voxels.ndim)

    @property
    def size(self) -> int:
        return int(self._world.voxels.size)

    def __array__(self, dtype: Any | None = None) -> np.ndarray:
        out = self._world.voxels == self._world.SOLID
        if dtype is None:
            return out
        return out.astype(dtype, copy=False)

    def __getitem__(self, key: Any) -> Any:
        return self._world.voxels[key] == self._world.SOLID

    def __setitem__(self, key: Any, value: Any) -> None:
        if np.isscalar(value):
            self._world.voxels[key] = self._world.SOLID if bool(value) else self._world.AIR
            return

        arr = np.asarray(value)
        if arr.dtype == np.bool_ or arr.dtype == bool:
            self._world.voxels[key] = np.where(arr, self._world.SOLID, self._world.AIR).astype(
                np.uint8, copy=False
            )
            return

        self._world.voxels[key] = arr.astype(np.uint8, copy=False)

    def copy(self) -> np.ndarray:
        return np.asarray(self).copy()


@dataclass
class VoxelWorld:
    # Voxel types
    AIR = 0
    SOLID = 1
    LAVA = 2
    WATER = 3

    voxels: np.ndarray  # uint8[sz, sy, sx] (z-major)
    voxel_size_m: float = 5.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def solid(self) -> Any:
        # Backward-compatible "solid mask" view.
        # Historically, callers treated `world.solid` as a mutable boolean ndarray.
        # Now that terrain is typed (uint8 voxels), we provide a lightweight proxy
        # that supports numpy reads and in-place boolean writes.
        return _SolidMask(self)

    @solid.setter
    def solid(self, value: Any) -> None:
        arr = np.asarray(value)
        if arr.shape != self.voxels.shape:
            raise ValueError(f"solid has shape {arr.shape}, expected {self.voxels.shape}")
        if arr.dtype == np.bool_ or arr.dtype == bool:
            self.voxels[:] = np.where(arr, self.SOLID, self.AIR).astype(np.uint8, copy=False)
            return
        self.voxels[:] = arr.astype(np.uint8, copy=False)

    @property
    def size_z(self) -> int:
        return int(self.voxels.shape[0])

    @property
    def size_y(self) -> int:
        return int(self.voxels.shape[1])

    @property
    def size_x(self) -> int:
        return int(self.voxels.shape[2])

    def in_bounds(self, ix: int, iy: int, iz: int) -> bool:
        return 0 <= ix < self.size_x and 0 <= iy < self.size_y and 0 <= iz < self.size_z

    def get_voxel(self, ix: int, iy: int, iz: int) -> int:
        if not self.in_bounds(ix, iy, iz):
            return self.SOLID
        return int(self.voxels[iz, iy, ix])

    def is_solid_index(self, ix: int, iy: int, iz: int) -> bool:
        return self.get_voxel(ix, iy, iz) == self.SOLID

    def set_box_solid(
        self,
        min_ix: int,
        min_iy: int,
        min_iz: int,
        max_ix_excl: int,
        max_iy_excl: int,
        max_iz_excl: int,
        value: bool | int,
    ) -> None:
        min_ix = max(min_ix, 0)
        min_iy = max(min_iy, 0)
        min_iz = max(min_iz, 0)
        max_ix_excl = min(max_ix_excl, self.size_x)
        max_iy_excl = min(max_iy_excl, self.size_y)
        max_iz_excl = min(max_iz_excl, self.size_z)
        if min_ix >= max_ix_excl or min_iy >= max_iy_excl or min_iz >= max_iz_excl:
            return
        
        val = int(value) if isinstance(value, (int, np.integer)) else (self.SOLID if value else self.AIR)
        self.voxels[min_iz:max_iz_excl, min_iy:max_iy_excl, min_ix:max_ix_excl] = val

    def aabb_collides(self, aabb_min: np.ndarray, aabb_max: np.ndarray) -> bool:
        # Only SOLID voxels (walls) cause physics collisions.
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

        region = self.voxels[min_iz:max_iz, min_iy:max_iy, min_ix:max_ix]
        return bool(np.any(region == self.SOLID))

    @classmethod
    def generate(cls, config: WorldConfig, rng: np.random.Generator) -> VoxelWorld:
        voxels = np.zeros((config.size_z, config.size_y, config.size_x), dtype=np.uint8)
        world = cls(voxels=voxels, voxel_size_m=config.voxel_size_m)

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
                world.set_box_solid(min_x, min_y, 0, max_x, max_y, world.size_z, False)

        # Select Archetype
        archetype = rng.integers(0, 3)
        # archetype = 0 # Force Citadel (Debug)
        archetype_i = int(archetype)
        world.meta["archetype"] = archetype_i
        world.meta["archetype_name"] = {0: "citadel", 1: "urban_grid", 2: "highway"}.get(archetype_i, "unknown")
        
        # Common Scatter Config
        cx, cy = config.size_x // 2, config.size_y // 2
        area = config.size_x * config.size_y
        
        if archetype == 0: # CITADEL (Central Keep + X Roads)
            # Roads first? No, carve last to clear scatter.
            # But Keep blocks must NOT be on the road lines or they float.
            
            road_width = 8.0
            
            # Helper to check if a box overlaps the X-Roads
            def overlaps_road(bx, by, bw, bh):
                # Check 4 corners of box against lines
                # Line 1: y = (cy/cx)*x  (approx)
                # Line 2: y = size_y - (cy/cx)*x
                # Simplification: Just avoid the exact center axes +- road width?
                # The X roads go corner-to-center. 
                # Let's just avoid the diagonal bands.
                
                # Check center of box
                bcx, bcy = bx + bw/2, by + bh/2
                
                # Dist to Diag 1 (y=x for square map)
                # |Ax + By + C| / sqrt(A^2+B^2). 
                # Line: -height*x + width*y = 0.
                # Just simplified: abs(bcy/size_y - bcx/size_x) < threshold
                
                norm_x = bcx / config.size_x
                norm_y = bcy / config.size_y
                
                # Main Diag (BL to TR)
                if abs(norm_y - norm_x) < 0.15: return True
                # Anti Diag (TL to BR)
                if abs(norm_y - (1.0 - norm_x)) < 0.15: return True
                
                return False

            keep_radius = max(5, int(min(config.size_x, config.size_y) * 0.25)) # Increased radius
            
            # Scale count by area. 100x100=10000 -> 12 blocks. 50x50=2500 -> 3 blocks (Too few).
            # Let's aim for density.
            num_keep_blocks = int(area * 0.002) # 20 blocks for 100x100
            
            placed_k = 0
            attempts_k = 0
            while placed_k < num_keep_blocks and attempts_k < 1000:
                attempts_k += 1
                sx, sy, sz = rng.integers(4, 12), rng.integers(4, 12), rng.integers(4, max(5, config.size_z - 2))
                
                ix = int(rng.integers(cx - keep_radius, cx + keep_radius - sx + 1))
                iy = int(rng.integers(cy - keep_radius, cy + keep_radius - sy + 1))
                
                # Clamp to bounds
                if ix < 0 or iy < 0 or ix+sx > config.size_x or iy+sy > config.size_y: continue

                if not overlaps_road(ix, iy, sx, sy):
                    world.set_box_solid(ix, iy, 0, ix + sx, iy + sy, sz, True)
                    placed_k += 1
            
            # Roads: X-Shape
            carve_line(0, 0, cx, cy, road_width)
            carve_line(config.size_x, 0, cx, cy, road_width)
            carve_line(0, config.size_y, cx, cy, road_width)
            carve_line(config.size_x, config.size_y, cx, cy, road_width)
            
        elif archetype == 1: # URBAN GRID (Regular Blocks)
            # Create a grid of city blocks
            # Scale block size to map size? 
            # 12 is nice for gameplay.
            # If map is small (<60), use smaller blocks.
            if config.size_x < 60:
                block_size = 8
                street_width = 4
            else:
                block_size = 12
                street_width = 6
                
            step = block_size + street_width
            
            for y in range(0, config.size_y, step):
                for x in range(0, config.size_x, step):
                    if rng.random() > 0.9: continue 
                    
                    # Fill block with 2-5 buildings
                    for _ in range(rng.integers(2, 6)):
                        bx = x + rng.integers(0, max(1, block_size//3))
                        by = y + rng.integers(0, max(1, block_size//3))
                        bw = rng.integers(3, block_size - 1)
                        bh = rng.integers(3, block_size - 1)
                        # Ensure it fits in "block"
                        if bx + bw > x + block_size: bw = x + block_size - bx
                        if by + bh > y + block_size: bh = y + block_size - by
                        
                        bz = rng.integers(5, max(8, config.size_z - 2)) 
                        world.set_box_solid(bx, by, 0, bx + bw, by + bh, bz, True)

        elif archetype == 2: # HIGHWAY (Diagonal Split)
            # Main heavy diagonal road
            road_width = 12.0
            if config.size_x < 60: road_width = 8.0
            
            # Randomize diagonal direction to avoid consistent quadrant bias.
            # 0: BL->TR, 1: TL->BR
            diag = int(rng.integers(0, 2))
            world.meta["highway_diagonal"] = diag
            world.meta["highway_road_width"] = float(road_width)

            if diag == 0:
                carve_line(0, 0, config.size_x, config.size_y, road_width)
                # Line through (0,0) and (size_x, size_y): size_y*x - size_x*y = 0
                A = float(config.size_y)
                B = -float(config.size_x)
                C = 0.0
            else:
                carve_line(0, config.size_y, config.size_x, 0, road_width)
                # Line through (0,size_y) and (size_x,0): size_y*x + size_x*y - size_x*size_y = 0
                A = float(config.size_y)
                B = float(config.size_x)
                C = -float(config.size_x * config.size_y)
            denom = float(np.hypot(A, B))
            
            # Heavy industrial noise everywhere else
            num_blocks = int(area * 0.04)
            for _ in range(num_blocks):
                sx, sy = rng.integers(3, 8), rng.integers(3, 8)
                sz = rng.integers(3, 12)
                max_ix0 = config.size_x - int(sx)
                max_iy0 = config.size_y - int(sy)
                if max_ix0 < 0 or max_iy0 < 0:
                    continue
                ix = int(rng.integers(0, max_ix0 + 1))
                iy = int(rng.integers(0, max_iy0 + 1))

                cx0 = float(ix) + float(sx) * 0.5
                cy0 = float(iy) + float(sy) * 0.5
                dist = abs(A * cx0 + B * cy0 + C) / denom if denom > 0.0 else 0.0
                if dist > road_width / 2 + 2:
                    world.set_box_solid(ix, iy, 0, ix + sx, iy + sy, sz, True)

        # General Scatter (Applied to all maps to add cover to open areas)
        # Reduced density for Urban Grid to preserve streets
        scatter_factor = 0.8
        if archetype == 1: scatter_factor = 0.2 # Keep streets clear
        
        sparse_fill = config.obstacle_fill * scatter_factor
        target_solids = int(sparse_fill * area * max(1, config.size_z // 3))

        placed = 0
        attempts = 0
        while placed < target_solids and attempts < target_solids * 50:
            attempts += 1
            sx, sy, sz = rng.integers(2, 5), rng.integers(2, 5), rng.integers(2, 5)
            max_ix0 = config.size_x - int(sx)
            max_iy0 = config.size_y - int(sy)
            if max_ix0 < 0 or max_iy0 < 0:
                continue
            ix = int(rng.integers(0, max_ix0 + 1))
            iy = int(rng.integers(0, max_iy0 + 1))

            before = np.count_nonzero(world.solid[0:sz, iy:iy+sy, ix:ix+sx])
            world.set_box_solid(ix, iy, 0, ix + sx, iy + sy, sz, True)
            after = np.count_nonzero(world.solid[0:sz, iy:iy+sy, ix:ix+sx])
            placed += int(after - before)

        # Final cleanup: Re-Carve main roads to ensure they aren't blocked by scatter
        if archetype == 0:
            road_width = 8.0
            carve_line(0, 0, cx, cy, road_width)
            road_width = 8.0
            carve_line(config.size_x, 0, cx, cy, road_width)
            carve_line(0, config.size_y, cx, cy, road_width)
            carve_line(config.size_x, config.size_y, cx, cy, road_width)
            
            # Add a lava pit in the very center objective
            world.set_box_solid(cx-4, cy-4, 0, cx+4, cy+4, 1, VoxelWorld.LAVA)
        elif archetype == 1:
            # Re-carve grid streets
            if config.size_x < 60:
                block_size, street_width = 8, 4
            else:
                block_size, street_width = 12, 6
            step = block_size + street_width
            # Vertical streets
            for x in range(block_size, config.size_x, step):
                carve_line(x, 0, x, config.size_y, float(street_width))
            # Horizontal streets
            for y in range(block_size, config.size_y, step):
                carve_line(0, y, config.size_x, y, float(street_width))
            
            # Add some water 'fountains' or canals
            for x in range(0, config.size_x, step * 3):
                world.set_box_solid(x + block_size + 1, 0, 0, x + block_size + 1 + street_width - 2, config.size_y, 1, VoxelWorld.WATER)

        elif archetype == 2:
            road_width = float(world.meta.get("highway_road_width", 12.0))
            diag = int(world.meta.get("highway_diagonal", 0))
            if diag == 0:
                carve_line(0, 0, config.size_x, config.size_y, road_width)
            else:
                carve_line(0, config.size_y, config.size_x, 0, road_width)
            
            # Add lava patches along the highway
            for _ in range(5):
                lx = rng.integers(0, config.size_x - 10)
                ly = rng.integers(0, config.size_y - 10)
                world.set_box_solid(lx, ly, 0, lx+6, ly+6, 1, VoxelWorld.LAVA)

        return world
